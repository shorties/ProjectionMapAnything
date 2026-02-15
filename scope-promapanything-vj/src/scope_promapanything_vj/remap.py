"""Depth map re-projection from camera perspective to projector perspective.

Uses the calibration mapping (map_x, map_y) produced by Gray code calibration
to warp a camera-space depth map into projector-space coordinates.

Two approaches:
1. warp_depth_to_projector() — warp raw depth float values (legacy)
2. build_warped_depth_image() — create a depth IMAGE in camera space first
   (with CLAHE, percentile clip, gamma, etc.) then warp it identically to
   how RGB is warped.  This is the proven approach from the standalone app
   that produces ControlNet-optimized depth output.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

# Pre-built colormaps (256 entries, RGB float32)
_COLORMAP_CACHE: dict[str, torch.Tensor] = {}


def _build_colormap(name: str) -> torch.Tensor:
    """Build a (256, 3) float32 RGB lookup table for a named colormap."""
    if name in _COLORMAP_CACHE:
        return _COLORMAP_CACHE[name]

    if name == "grayscale":
        lut = torch.linspace(0, 1, 256).unsqueeze(1).expand(256, 3)
    else:
        cv_maps = {
            "turbo": cv2.COLORMAP_TURBO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "magma": cv2.COLORMAP_MAGMA,
        }
        cv_id = cv_maps.get(name, cv2.COLORMAP_TURBO)
        idx = np.arange(256, dtype=np.uint8).reshape(1, 256)
        bgr = cv2.applyColorMap(idx, cv_id).squeeze(0)  # (256, 3) BGR uint8
        rgb = bgr[:, ::-1].copy().astype(np.float32) / 255.0
        lut = torch.from_numpy(rgb)

    _COLORMAP_CACHE[name] = lut
    return lut


def apply_depth_output_settings(
    depth_raw: np.ndarray,
    *,
    clip_lo: float = 2.0,
    clip_hi: float = 98.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    equalize: bool = True,
) -> np.ndarray:
    """Apply ControlNet-optimized output settings to a raw depth map.

    Parameters
    ----------
    depth_raw : np.ndarray
        (H, W) float32 raw depth values (any range).

    Returns
    -------
    np.ndarray
        (H, W) float32 in [0, 1], processed for ControlNet consumption.
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

    # CLAHE histogram equalization for maximum contrast
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
    clip_lo: float = 2.0,
    clip_hi: float = 98.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    equalize: bool = True,
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

    # Scale/invert
    if invert:
        depth_img = 1.0 - depth_img

    # Blur in camera space
    if blur > 0.5:
        ksize = int(blur) * 2 + 1
        depth_img = cv2.GaussianBlur(depth_img, (ksize, ksize), 0)

    # Step 2: Convert to BGR image (same as RGB pipeline)
    depth_uint8 = (depth_img * 255).clip(0, 255).astype(np.uint8)

    # Apply colormap
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

    # Step 4: Inpaint holes (same as RGB)
    depth_holes = np.all(depth_proj_bgr == 0, axis=2).astype(np.uint8) * 255
    if np.any(depth_holes):
        depth_proj_bgr = cv2.inpaint(depth_proj_bgr, depth_holes, 15, cv2.INPAINT_NS)

    # Step 5: Convert BGR -> RGB -> torch
    depth_proj_rgb = cv2.cvtColor(depth_proj_bgr, cv2.COLOR_BGR2RGB)
    result = torch.from_numpy(depth_proj_rgb.astype(np.float32) / 255.0).to(device)

    return result  # (H, W, 3)


def warp_frame_to_projector(
    frame: torch.Tensor,
    map_x: np.ndarray,
    map_y: np.ndarray,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Warp a color frame from camera space to projector space.

    Used in the "Depth -> AI -> Warp" pipeline mode to remap an AI-generated
    RGB frame into projector perspective using the calibration mapping.

    Parameters
    ----------
    frame : torch.Tensor
        (H, W, 3) or (1, H, W, 3) float32 RGB in [0, 1].
    map_x, map_y : np.ndarray
        (proj_h, proj_w) float32 camera pixel coordinates from calibration.

    Returns
    -------
    torch.Tensor
        (H_proj, W_proj, 3) float32 RGB in [0, 1].
    """
    if device is None:
        device = frame.device

    f = frame.squeeze(0) if frame.ndim == 4 else frame  # (H, W, 3)

    # Convert to numpy BGR uint8 for cv2.remap
    rgb_np = (f.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    # Warp using calibration mapping
    warped_bgr = cv2.remap(
        bgr_np, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Inpaint holes (zero pixels from unmapped regions)
    holes = np.all(warped_bgr == 0, axis=2).astype(np.uint8) * 255
    if np.any(holes):
        warped_bgr = cv2.inpaint(warped_bgr, holes, 15, cv2.INPAINT_NS)

    # Convert back to torch RGB float32
    warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(warped_rgb.astype(np.float32) / 255.0).to(device)


def warp_depth_to_projector(
    depth: torch.Tensor,
    map_x: np.ndarray,
    map_y: np.ndarray,
    proj_h: int,
    proj_w: int,
) -> torch.Tensor:
    """Warp a camera-space depth map to projector space (legacy).

    Parameters
    ----------
    depth : torch.Tensor
        (H_cam, W_cam) float32 depth in [0, 1].
    map_x, map_y : np.ndarray
        (proj_h, proj_w) float32 camera pixel coordinates.

    Returns
    -------
    torch.Tensor
        (proj_h, proj_w) float32 depth in [0, 1].
    """
    depth_np = depth.cpu().numpy()
    warped = cv2.remap(
        depth_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return torch.from_numpy(warped).to(depth.device)


def apply_depth_adjustments(
    depth: torch.Tensor,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    blur: float = 0.0,
    invert: bool = False,
    colormap: str = "grayscale",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Apply user-controlled adjustments and colormap to a depth map (legacy).

    Parameters
    ----------
    depth : torch.Tensor
        (H, W) float32 depth in [0, 1].

    Returns
    -------
    torch.Tensor
        (H, W, 3) float32 RGB in [0, 1].
    """
    if device is None:
        device = depth.device

    d = depth.clone()

    if scale != 1.0 or offset != 0.0:
        d = d * scale + offset

    if invert:
        d = 1.0 - d

    d = d.clamp(0, 1)

    if blur > 0.5:
        ksize = int(blur) * 2 + 1
        d_np = d.cpu().numpy()
        d_np = cv2.GaussianBlur(d_np, (ksize, ksize), 0)
        d = torch.from_numpy(d_np).to(device)

    lut = _build_colormap(colormap).to(device)  # (256, 3)
    indices = (d.clamp(0, 1) * 255).long().clamp(0, 255)
    rgb = lut[indices]  # (H, W, 3)

    return rgb
