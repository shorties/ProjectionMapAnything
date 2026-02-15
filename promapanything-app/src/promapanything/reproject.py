"""Depth reprojection: camera-perspective depth → projector-perspective depth.

Uses the full ProCam calibration (K_cam, K_proj, R, T) to unproject camera
depth pixels to 3D, transform to projector space, and re-project into the
projector's image plane with Z-buffering.
"""

from __future__ import annotations

import cv2
import numpy as np


class DepthReprojector:
    """Reprojects camera-view depth to projector-view depth.

    Pre-computes pixel grids and inverse intrinsics so that per-frame work
    is purely vectorized numpy operations.
    """

    def __init__(
        self,
        K_cam: np.ndarray,
        dist_cam: np.ndarray,
        K_proj: np.ndarray,
        dist_proj: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        cam_w: int,
        cam_h: int,
        proj_w: int,
        proj_h: int,
    ):
        self.proj_w = proj_w
        self.proj_h = proj_h
        self.cam_w = cam_w
        self.cam_h = cam_h

        # Camera → projector transform
        self.R = R.astype(np.float64)
        self.T = T.reshape(3, 1).astype(np.float64)

        # Projector intrinsics
        self.fx_p = K_proj[0, 0]
        self.fy_p = K_proj[1, 1]
        self.cx_p = K_proj[0, 2]
        self.cy_p = K_proj[1, 2]

        # Derive depth scale from calibration baseline
        # ML depth [0,1] is relative; we map it to a range centered on the
        # baseline so the units are consistent with the calibrated R, T.
        baseline = float(np.linalg.norm(T))
        self._depth_near = baseline * 2.0   # min scene depth ~ 2x baseline
        self._depth_far = baseline * 20.0   # max scene depth ~ 20x baseline

        # Pre-compute camera pixel grid in normalized coordinates
        # Undistort the camera pixel grid to get ideal normalized coords
        u = np.arange(cam_w, dtype=np.float64)
        v = np.arange(cam_h, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)  # (cam_h, cam_w)

        # Undistort pixel coordinates
        pts = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float64)
        pts_undist = cv2.undistortPoints(
            pts.reshape(-1, 1, 2), K_cam, dist_cam
        )  # returns normalized coords (x/z, y/z)
        pts_undist = pts_undist.reshape(-1, 2)

        # Store as (3, N) homogeneous rays: [x_norm, y_norm, 1]
        N = cam_w * cam_h
        self._rays = np.ones((3, N), dtype=np.float64)
        self._rays[0, :] = pts_undist[:, 0]
        self._rays[1, :] = pts_undist[:, 1]

    def reproject(self, depth_cam: np.ndarray) -> np.ndarray:
        """Convert camera-perspective depth to projector-perspective depth.

        Args:
            depth_cam: (cam_h, cam_w) float32 [0,1] from Depth Anything V2

        Returns:
            (proj_h, proj_w) float32 [0,1] depth from projector's viewpoint
        """
        import logging
        logger = logging.getLogger(__name__)

        h, w = depth_cam.shape
        N = h * w

        logger.debug(f"Reprojecting depth: camera {w}x{h} -> projector {self.proj_w}x{self.proj_h}")

        # Verify input dimensions match what we were initialized with
        if h != self.cam_h or w != self.cam_w:
            logger.error(f"Depth resolution mismatch! Expected {self.cam_w}x{self.cam_h}, got {w}x{h}")
            # Could resize here, but for now just return zeros
            return np.zeros((self.proj_h, self.proj_w), dtype=np.float32)

        # Flatten depth and scale rays to 3D points in camera space
        # P_cam = ray * depth (since ray = [x_norm, y_norm, 1], depth = Z)
        # This gives P_cam = [X/Z * Z, Y/Z * Z, Z] = [X, Y, Z]
        #
        # ML depth [0,1] is relative. Scale to the same unit system as the
        # calibrated R, T (derived from the stereo baseline).
        depth_flat = depth_cam.ravel().astype(np.float64)
        depth_flat = self._depth_near + depth_flat * (self._depth_far - self._depth_near)

        # Mask out zero/invalid depth
        valid = depth_flat > 0.001
        valid_idx = np.where(valid)[0]
        logger.debug(f"Valid depth pixels: {len(valid_idx)}/{N} ({100*len(valid_idx)/N:.1f}%)")
        if len(valid_idx) == 0:
            logger.warning("No valid depth pixels to reproject")
            return np.zeros((self.proj_h, self.proj_w), dtype=np.float32)

        d = depth_flat[valid_idx]
        rays = self._rays[:, valid_idx]  # (3, M)

        # 3D points in camera space: P_cam = ray * depth
        P_cam = rays * d[np.newaxis, :]  # (3, M)

        # Transform to projector space: P_proj = R @ P_cam + T
        P_proj = self.R @ P_cam + self.T  # (3, M)

        # Filter points behind the projector
        z_proj = P_proj[2, :]
        in_front = z_proj > 0.001
        P_proj = P_proj[:, in_front]
        z_proj = z_proj[in_front]

        logger.debug(f"Points in front of projector: {len(z_proj)}/{len(valid_idx)}")
        if len(z_proj) == 0:
            logger.warning("No points in front of projector")
            return np.zeros((self.proj_h, self.proj_w), dtype=np.float32)

        # Project to projector pixels
        u_proj = (self.fx_p * P_proj[0, :] / z_proj + self.cx_p)
        v_proj = (self.fy_p * P_proj[1, :] / z_proj + self.cy_p)

        # Round to integer pixel coords
        u_i = np.round(u_proj).astype(np.int32)
        v_i = np.round(v_proj).astype(np.int32)

        # Filter to valid projector pixel range
        in_bounds = (
            (u_i >= 0) & (u_i < self.proj_w)
            & (v_i >= 0) & (v_i < self.proj_h)
        )
        u_i = u_i[in_bounds]
        v_i = v_i[in_bounds]
        z_proj = z_proj[in_bounds]

        logger.debug(f"Points in projector bounds: {len(z_proj)}  z range: [{z_proj.min():.3f}, {z_proj.max():.3f}]")
        if len(z_proj) == 0:
            logger.warning("No points within projector bounds")
            return np.zeros((self.proj_h, self.proj_w), dtype=np.float32)

        # Z-buffer scatter: keep minimum depth (closest surface)
        proj_depth = np.full((self.proj_h, self.proj_w), np.inf, dtype=np.float64)
        np.minimum.at(proj_depth, (v_i, u_i), z_proj)

        # Replace inf with 0
        proj_depth[proj_depth == np.inf] = 0.0

        # Hole fill: dilate to cover small gaps, then bilateral filter
        valid_mask = (proj_depth > 0).astype(np.uint8)
        proj_f32 = proj_depth.astype(np.float32)

        if np.count_nonzero(valid_mask) > 0:
            # Dilate valid regions to fill 1-2 pixel holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(proj_f32, kernel, iterations=2)
            hole_mask = (valid_mask == 0)
            proj_f32[hole_mask] = dilated[hole_mask]

        # Normalize to [0, 1]
        valid_vals = proj_f32[proj_f32 > 0]
        if len(valid_vals) > 0:
            d_min = np.percentile(valid_vals, 1)
            d_max = np.percentile(valid_vals, 99)
            logger.debug(f"Reprojected depth before norm: range [{valid_vals.min():.3f}, {valid_vals.max():.3f}]  1%/99%: [{d_min:.3f}, {d_max:.3f}]")
            if d_max > d_min:
                proj_f32 = (proj_f32 - d_min) / (d_max - d_min)
                proj_f32 = np.clip(proj_f32, 0.0, 1.0)
            else:
                proj_f32 = np.zeros_like(proj_f32)

        logger.debug(f"Reprojected depth output: shape {proj_f32.shape}  range [{proj_f32.min():.3f}, {proj_f32.max():.3f}]")
        return proj_f32
