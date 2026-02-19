"""Depth map derivation and re-projection for projection mapping.

Derives real depth from structured light calibration correspondence maps
(map_x, map_y) using the Jacobian of the projector→camera mapping.
Also supports warping camera-space depth maps to projector-space.

Output convention: near = bright (1), far = dark (0), grayscale (R=G=B).
This is inverse depth (disparity), matching VACE's expected input from
Depth Anything V2 / MiDaS.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def derive_depth_from_calibration(
    map_x: np.ndarray,
    map_y: np.ndarray,
    proj_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Derive a relative depth map from structured light correspondence maps.

    The projector→camera mapping encodes depth: the Jacobian determinant
    measures the local area scaling of the mapping, which is proportional
    to 1/Z² (closer surfaces stretch across more camera pixels).

    Parameters
    ----------
    map_x, map_y : np.ndarray
        (proj_h, proj_w) float32 — camera pixel coordinates for each
        projector pixel, from Gray code calibration.
    proj_valid_mask : np.ndarray | None
        Boolean mask of valid (decoded) projector pixels.

    Returns
    -------
    np.ndarray
        (proj_h, proj_w) float32 in [0, 1].
        Convention: near = bright (1), far = dark (0).
    """
    # Partial derivatives of the correspondence map
    # d(cam_x)/d(proj_x), d(cam_x)/d(proj_y), etc.
    dx_dpx = np.gradient(map_x, axis=1)
    dx_dpy = np.gradient(map_x, axis=0)
    dy_dpx = np.gradient(map_y, axis=1)
    dy_dpy = np.gradient(map_y, axis=0)

    # Jacobian determinant = local area scaling factor
    # |det(J)| ∝ 1/Z² → sqrt(|det(J)|) ∝ 1/Z (disparity)
    det_j = np.abs(dx_dpx * dy_dpy - dx_dpy * dy_dpx)
    disparity = np.sqrt(np.clip(det_j, 0.0, None))

    # Mask invalid regions
    if proj_valid_mask is not None:
        disparity[~proj_valid_mask] = 0.0

    # Smooth to reduce gradient noise
    disparity = cv2.GaussianBlur(disparity.astype(np.float32), (11, 11), 0)

    # Percentile normalization to [0, 1]
    valid = disparity > 1e-6
    if not np.any(valid):
        logger.warning("No valid depth data from calibration")
        return np.full_like(map_x, 0.5, dtype=np.float32)

    p2 = float(np.percentile(disparity[valid], 2))
    p98 = float(np.percentile(disparity[valid], 98))
    if p98 - p2 < 1e-6:
        return np.full_like(map_x, 0.5, dtype=np.float32)

    depth = (disparity - p2) / (p98 - p2)
    depth = np.clip(depth, 0.0, 1.0).astype(np.float32)

    # Inpaint holes for a clean result
    holes = (depth < 0.01).astype(np.uint8) * 255
    if proj_valid_mask is not None:
        holes[proj_valid_mask & (depth >= 0.01)] = 0
    depth_u8 = (depth * 255).clip(0, 255).astype(np.uint8)
    if np.any(holes):
        depth_u8 = cv2.inpaint(depth_u8, holes, 15, cv2.INPAINT_NS)
        depth = depth_u8.astype(np.float32) / 255.0

    logger.info(
        "Derived depth from structured light: range [%.3f, %.3f], "
        "valid %.1f%%",
        depth[valid].min(), depth[valid].max(),
        100.0 * np.count_nonzero(valid) / valid.size,
    )
    return depth


def apply_depth_output_settings(
    depth_raw: np.ndarray,
    *,
    clip_lo: float = 0.0,
    clip_hi: float = 100.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    equalize: bool = False,
) -> np.ndarray:
    """Apply output settings to a raw depth map.

    Parameters
    ----------
    depth_raw : np.ndarray
        (H, W) float32 raw depth values (any range).

    Returns
    -------
    np.ndarray
        (H, W) float32 in [0, 1], processed for consumption.
    """
    # Percentile-based normalization
    valid_vals = depth_raw[depth_raw > 0] if np.any(depth_raw > 0) else depth_raw.ravel()
    if len(valid_vals) == 0:
        return np.zeros_like(depth_raw)

    d_lo = np.percentile(valid_vals, clip_lo)
    d_hi = np.percentile(valid_vals, clip_hi)
    if d_hi - d_lo < 1e-6:
        d_hi = d_lo + 1e-6

    depth_norm = (depth_raw - d_lo) / (d_hi - d_lo)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    # CLAHE histogram equalization
    if equalize:
        depth_u8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        depth_u8 = clahe.apply(depth_u8)
        depth_norm = depth_u8.astype(np.float32) / 255.0

    # Gamma correction
    if abs(gamma - 1.0) > 0.01:
        depth_norm = np.power(np.clip(depth_norm, 0, 1), gamma)

    # Brightness + contrast
    if abs(brightness) > 0.001 or abs(contrast - 1.0) > 0.001:
        depth_norm = contrast * depth_norm + brightness
        depth_norm = np.clip(depth_norm, 0.0, 1.0)

    return depth_norm


def build_warped_depth_image(
    depth_raw: torch.Tensor,
    map_x: np.ndarray,
    map_y: np.ndarray,
    proj_valid_mask: np.ndarray | None,
    proj_h: int,
    proj_w: int,
    *,
    clip_lo: float = 0.0,
    clip_hi: float = 100.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    equalize: bool = False,
    invert: bool = False,
    blur: float = 0.0,
    colormap: str = "grayscale",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a depth image in camera space, then warp it like RGB.

    This is the proven approach from the standalone app:
    1. Run depth output settings (percentile, CLAHE, gamma, etc.) in camera space
    2. Convert to BGR image
    3. Warp to projector using cv2.remap (same as RGB)
    4. Inpaint holes (same as RGB)
    5. Convert back to torch tensor

    Parameters
    ----------
    depth_raw : torch.Tensor
        (H_cam, W_cam) float32 raw depth in [0, 1].
    map_x, map_y : np.ndarray
        (proj_h, proj_w) float32 camera pixel coordinates.
    proj_valid_mask : np.ndarray | None
        Bool mask of valid projector pixels (before inpainting).
    proj_h, proj_w : int
        Projector resolution.

    Returns
    -------
    torch.Tensor
        (H, W, 3) float32 RGB in [0, 1].
    """
    if device is None:
        device = depth_raw.device

    depth_np = depth_raw.cpu().numpy()

    # Step 1: Apply output settings in camera space
    depth_img = apply_depth_output_settings(
        depth_np,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        brightness=brightness,
        contrast=contrast,
        gamma=gamma,
        equalize=equalize,
    )

    # Invert
    if invert:
        depth_img = 1.0 - depth_img

    # Blur in camera space
    if blur > 0.5:
        ksize = int(blur) * 2 + 1
        depth_img = cv2.GaussianBlur(depth_img, (ksize, ksize), 0)

    # Step 2: Convert to BGR image
    depth_uint8 = (depth_img * 255).clip(0, 255).astype(np.uint8)

    if colormap == "grayscale":
        depth_bgr = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
    else:
        cv_maps = {
            "turbo": cv2.COLORMAP_TURBO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "magma": cv2.COLORMAP_MAGMA,
        }
        cv_id = cv_maps.get(colormap, cv2.COLORMAP_TURBO)
        depth_bgr = cv2.applyColorMap(depth_uint8, cv_id)

    # Step 3: Warp to projector (SAME method as RGB)
    depth_proj_bgr = cv2.remap(
        depth_bgr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Step 4: Inpaint holes
    depth_holes = np.all(depth_proj_bgr == 0, axis=2).astype(np.uint8) * 255
    if np.any(depth_holes):
        depth_proj_bgr = cv2.inpaint(depth_proj_bgr, depth_holes, 15, cv2.INPAINT_NS)

    # Step 5: Convert BGR -> RGB -> torch
    depth_proj_rgb = cv2.cvtColor(depth_proj_bgr, cv2.COLOR_BGR2RGB)
    result = torch.from_numpy(depth_proj_rgb.astype(np.float32) / 255.0).to(device)

    return result  # (H, W, 3)
