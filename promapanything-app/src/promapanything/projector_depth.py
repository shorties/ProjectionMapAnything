"""Build metric depth map from projector perspective using calibrated stereo + Gray codes.

This implements the proper RoomAlive approach:
1. Calibrate projector + camera using checkerboard (metric calibration)
2. Project Gray codes to get dense correspondence
3. Triangulate 3D points using the calibrated stereo pair
4. No depth sensor needed - pure geometric reconstruction
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ProjectorDepthBuilder:
    """Builds metric depth map using calibrated stereo triangulation.
    
    Uses the calibrated projector+camera system as a stereo pair.
    Gray code correspondences provide dense matching.
    Triangulation gives 3D points in metric units (from calibration).
    """
    
    def __init__(
        self,
        K_cam: np.ndarray,
        dist_cam: np.ndarray,
        K_proj: np.ndarray,
        dist_proj: np.ndarray,
        R: np.ndarray,  # Camera to projector rotation
        T: np.ndarray,  # Camera to projector translation (in meters/mm from calibration)
        proj_w: int,
        proj_h: int,
    ):
        self.K_cam = K_cam.astype(np.float64)
        self.dist_cam = dist_cam.astype(np.float64) if dist_cam is not None else np.zeros(5)
        self.K_proj = K_proj.astype(np.float64)
        self.dist_proj = dist_proj.astype(np.float64) if dist_proj is not None else np.zeros(5)
        self.R = R.astype(np.float64)
        self.T = T.reshape(3, 1).astype(np.float64)
        self.proj_w = proj_w
        self.proj_h = proj_h
        
        # Build projection matrices for triangulation
        # P0 (camera): K_cam @ [I | 0]
        self.P_cam = self.K_cam @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # P1 (projector): K_proj @ [R | T]
        self.P_proj = self.K_proj @ np.hstack([self.R, self.T])
        
        # Compute baseline (distance between camera and projector)
        self.baseline = np.linalg.norm(self.T)
        
        logger.info(f"ProjectorDepthBuilder: {proj_w}x{proj_h}")
        logger.info(f"Baseline (cam-to-proj distance): {self.baseline:.1f} mm")
        
    def triangulate_point(self, pt_cam: np.ndarray, pt_proj: np.ndarray) -> np.ndarray | None:
        """Triangulate a single 3D point from camera and projector rays.
        
        Args:
            pt_cam: (2,) camera pixel coordinates (u, v)
            pt_proj: (2,) projector pixel coordinates (u, v)
            
        Returns:
            (3,) 3D point in camera coordinates, or None if invalid
        """
        # Undistort points
        pt_cam_u = cv2.undistortPoints(
            pt_cam.reshape(1, 1, 2).astype(np.float64),
            self.K_cam, self.dist_cam, None, self.K_cam
        ).reshape(2)
        
        pt_proj_u = cv2.undistortPoints(
            pt_proj.reshape(1, 1, 2).astype(np.float64),
            self.K_proj, self.dist_proj, None, self.K_proj
        ).reshape(2)
        
        # Triangulate
        points_4d = cv2.triangulatePoints(
            self.P_cam, self.P_proj,
            pt_cam_u.reshape(2, 1),
            pt_proj_u.reshape(2, 1)
        )
        
        # Convert from homogeneous
        X = points_4d[:3] / points_4d[3]
        return X.flatten()
    
    def build_depth_map(
        self,
        map_x: np.ndarray,  # (proj_h, proj_w) camera x for each projector pixel
        map_y: np.ndarray,  # (proj_h, proj_w) camera y for each projector pixel
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build metric depth map using stereo triangulation.
        
        For each projector pixel with valid correspondence, triangulate
        the 3D point and get depth (Z in projector coordinate system).
        
        Returns:
            depth_map: (proj_h, proj_w) float32 depth in calibration units (mm)
            valid_mask: (proj_h, proj_w) bool
        """
        h, w = map_x.shape
        assert h == self.proj_h and w == self.proj_w
        
        if mask is None:
            mask = (map_x >= 0) & (map_y >= 0)
        
        depth_map = np.zeros((h, w), dtype=np.float32)
        valid_mask = np.zeros((h, w), dtype=bool)
        
        # Get valid pixel coordinates
        v_p, u_p = np.where(mask)
        n_points = len(u_p)
        
        if n_points == 0:
            logger.warning("No valid correspondences")
            return depth_map, valid_mask
        
        logger.info(f"Triangulating {n_points} points...")
        
        # Get corresponding camera pixels
        u_c = map_x[v_p, u_p]
        v_c = map_y[v_p, u_p]
        
        # Prepare points for batch triangulation
        pts_cam = np.column_stack([u_c, v_c]).astype(np.float64).reshape(-1, 1, 2)
        pts_proj = np.column_stack([u_p, v_p]).astype(np.float64).reshape(-1, 1, 2)
        
        # Undistort
        pts_cam_u = cv2.undistortPoints(pts_cam, self.K_cam, self.dist_cam, None, self.K_cam)
        pts_proj_u = cv2.undistortPoints(pts_proj, self.K_proj, self.dist_proj, None, self.K_proj)
        
        # Reshape for triangulatePoints (2, N)
        pts_cam_T = pts_cam_u.reshape(-1, 2).T
        pts_proj_T = pts_proj_u.reshape(-1, 2).T
        
        # Triangulate all points at once
        points_4d = cv2.triangulatePoints(self.P_cam, self.P_proj, pts_cam_T, pts_proj_T)
        
        # Convert from homogeneous (3, N)
        points_3d = points_4d[:3] / points_4d[3]
        
        # Transform to projector coordinate system
        # P_proj = R @ P_cam + T
        points_in_proj = (self.R @ points_3d + self.T).T  # (N, 3)
        
        # Get depth (Z in projector space)
        depths = points_in_proj[:, 2]
        
        # Filter valid depths (positive, reasonable range)
        # The calibration baseline gives us approximate scale
        min_depth = self.baseline * 0.5  # ~half baseline
        max_depth = self.baseline * 50   # ~50x baseline
        
        valid_depths = (depths > min_depth) & (depths < max_depth) & np.isfinite(depths)
        
        # Store results
        valid_indices = np.where(valid_depths)[0]
        depth_map[v_p[valid_indices], u_p[valid_indices]] = depths[valid_depths]
        valid_mask[v_p[valid_indices], u_p[valid_indices]] = True
        
        n_valid = np.count_nonzero(valid_mask)
        if n_valid > 0:
            d_min = depth_map[valid_mask].min()
            d_max = depth_map[valid_mask].max()
            logger.info(f"Depth map: {n_valid}/{n_points} valid points")
            logger.info(f"Depth range: [{d_min:.1f}, {d_max:.1f}] mm")
        
        return depth_map, valid_mask
    
    def build_rgb_view(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        color_frame: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Build RGB image from projector's perspective by sampling camera.
        
        Returns:
            rgb_projector: (proj_h, proj_w, 3) uint8
        """
        h, w = map_x.shape
        rgb_proj = np.zeros((h, w, 3), dtype=np.uint8)
        
        if color_frame is None:
            return rgb_proj
        
        v_p, u_p = np.where(valid_mask)
        u_c = map_x[v_p, u_p].astype(np.int32)
        v_c = map_y[v_p, u_p].astype(np.int32)
        
        # Clip to camera bounds
        cam_h, cam_w = color_frame.shape[:2]
        in_bounds = (u_c >= 0) & (u_c < cam_w) & (v_c >= 0) & (v_c < cam_h)
        
        rgb_proj[v_p[in_bounds], u_p[in_bounds]] = color_frame[v_c[in_bounds], u_c[in_bounds]]
        
        return rgb_proj
    
    def build_depth_with_color(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        color_frame: np.ndarray | None,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build both depth map and RGB view from projector perspective.

        Convenience method combining build_depth_map + build_rgb_view.

        Returns:
            depth_map: (proj_h, proj_w) float32 depth in calibration units
            rgb_proj: (proj_h, proj_w, 3) uint8 RGB from projector's view
            valid_mask: (proj_h, proj_w) bool
        """
        depth_map, valid_mask = self.build_depth_map(map_x, map_y, mask)
        rgb_proj = self.build_rgb_view(map_x, map_y, color_frame, valid_mask)
        return depth_map, rgb_proj, valid_mask

    def fill_holes(
        self,
        depth_map: np.ndarray,
        valid_mask: np.ndarray,
        max_hole_size: int = 10,
    ) -> np.ndarray:
        """Fill small holes using morphological operations."""
        if np.all(valid_mask):
            return depth_map
        
        # Create hole mask
        hole_mask = (~valid_mask).astype(np.uint8) * 255
        
        # Fill small holes with dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_hole_size, max_hole_size))
        
        # Dilate valid regions into holes
        valid_float = depth_map.copy()
        valid_float[~valid_mask] = 0
        
        dilated = cv2.dilate(valid_float, kernel, iterations=1)
        
        # Only fill small holes (not large unknown regions)
        # Erode the hole mask to find small holes
        small_holes = cv2.erode(hole_mask, kernel, iterations=1)
        small_holes = hole_mask - small_holes  # Only the thin borders
        
        result = depth_map.copy()
        fill_mask = (small_holes > 0) & (dilated > 0) & (~valid_mask)
        result[fill_mask] = dilated[fill_mask]
        
        return result
    
    def save_depth_map(
        self,
        depth_map: np.ndarray,
        valid_mask: np.ndarray,
        output_dir: str | Path,
        prefix: str = "projector_depth",
    ) -> dict[str, Path]:
        """Save depth map in multiple formats.
        
        Returns paths to saved files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved = {}
        
        # 1. Raw depth (metric units - mm or whatever calibration used)
        npy_path = output_dir / f"{prefix}.npy"
        np.save(str(npy_path), depth_map)
        saved['npy'] = npy_path
        
        # 2. Normalized PNG for ControlNet (near=white, far=black)
        valid_depths = depth_map[valid_mask]
        if len(valid_depths) > 0:
            d_min, d_max = valid_depths.min(), valid_depths.max()
            # Invert: near = 255 (white), far = 0 (black) - ControlNet convention
            depth_norm = (255 - (depth_map - d_min) / (d_max - d_min) * 255).clip(0, 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_map, dtype=np.uint8)
        
        # Fill holes with interpolation for cleaner output
        depth_norm_filled = cv2.inpaint(depth_norm, (~valid_mask).astype(np.uint8) * 255, 3, cv2.INPAINT_NS)
        
        png_path = output_dir / f"{prefix}_controlnet.png"
        cv2.imwrite(str(png_path), depth_norm_filled)
        saved['controlnet_png'] = png_path
        
        # 3. Turbo colormap for visualization
        if len(valid_depths) > 0:
            depth_vis = ((depth_map - d_min) / (d_max - d_min) * 255).clip(0, 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth_map, dtype=np.uint8)
        
        turbo = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        turbo[~valid_mask] = 0  # Black out invalid
        turbo_path = output_dir / f"{prefix}_turbo.png"
        cv2.imwrite(str(turbo_path), turbo)
        saved['turbo_png'] = turbo_path
        
        # 4. Metadata
        meta = {
            'width': self.proj_w,
            'height': self.proj_h,
            'depth_units': 'calibration_units',
            'depth_min': float(depth_map[valid_mask].min()) if np.any(valid_mask) else 0,
            'depth_max': float(depth_map[valid_mask].max()) if np.any(valid_mask) else 0,
            'valid_pixels': int(np.count_nonzero(valid_mask)),
            'total_pixels': self.proj_w * self.proj_h,
            'baseline_mm': float(self.baseline),
            'method': 'stereo_triangulation',
            'note': 'Pure geometric reconstruction - no depth sensor used',
        }
        import json
        meta_path = output_dir / f"{prefix}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        saved['meta'] = meta_path
        
        logger.info(f"Saved depth map to {output_dir}:")
        for k, v in saved.items():
            logger.info(f"  {k}: {v.name}")
        
        return saved
