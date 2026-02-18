"""Subject isolation — extract foreground subjects from depth maps.

Three isolation strategies:
1. **depth_band** — Histogram peak detection in nearest depths + band mask
2. **mask** — User-uploaded mask image
3. **rembg** — AI background removal via rembg (lazy-loaded)

All functions return a float32 mask in [0, 1] at the input resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def isolate_by_depth_band(
    depth: np.ndarray,
    band_width: float = 0.3,
    feather: float = 5.0,
) -> np.ndarray:
    """Isolate the nearest surfaces using histogram peak detection.

    Finds the dominant depth peak in the nearest 50% of depth values,
    then builds a band mask around it.

    Parameters
    ----------
    depth : np.ndarray
        (H, W) float32 depth in [0, 1]. 0 = near, 1 = far.
    band_width : float
        Width of the depth band to keep, as a fraction of [0, 1].
    feather : float
        Gaussian blur radius for soft mask edges.

    Returns
    -------
    np.ndarray
        (H, W) float32 mask in [0, 1].
    """
    # Only look at non-black pixels (valid depth)
    valid = depth > 0.02
    if not np.any(valid):
        return np.zeros_like(depth)

    valid_depths = depth[valid]

    # Histogram of the nearest 50% of depth values
    hist, bin_edges = np.histogram(valid_depths, bins=50, range=(0.0, 0.5))

    # Find the dominant peak
    peak_bin = int(np.argmax(hist))
    peak_center = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0

    # Build band mask
    half_band = band_width / 2.0
    lo = max(0.0, peak_center - half_band)
    hi = min(1.0, peak_center + half_band)

    mask = ((depth >= lo) & (depth <= hi) & valid).astype(np.float32)

    # Feather the edges
    if feather > 0.5:
        ksize = int(feather) * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    return mask


def isolate_by_mask(
    mask_path: str | Path,
    h: int,
    w: int,
    feather: float = 5.0,
) -> np.ndarray | None:
    """Load a custom mask image and resize to target dimensions.

    Parameters
    ----------
    mask_path : str | Path
        Path to the mask image (any format OpenCV can read).
    h, w : int
        Target height and width.
    feather : float
        Gaussian blur radius for soft edges.

    Returns
    -------
    np.ndarray | None
        (H, W) float32 mask in [0, 1], or None if the file doesn't exist.
    """
    path = Path(mask_path)
    if not path.is_file():
        return None

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize to target
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = img.astype(np.float32) / 255.0

    # Feather
    if feather > 0.5:
        ksize = int(feather) * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    return mask


# Cached rembg session (lazy-loaded)
_rembg_session = None


def isolate_by_rembg(
    camera_frame: np.ndarray,
    feather: float = 5.0,
) -> np.ndarray:
    """Remove background using rembg AI model.

    Lazy-loads the rembg u2net session on first call.

    Parameters
    ----------
    camera_frame : np.ndarray
        (H, W, 3) uint8 RGB image.
    feather : float
        Gaussian blur radius for soft mask edges.

    Returns
    -------
    np.ndarray
        (H, W) float32 mask in [0, 1].
    """
    global _rembg_session

    try:
        from rembg import new_session, remove
    except ImportError:
        logger.warning("rembg not installed — subject isolation disabled")
        return np.ones(camera_frame.shape[:2], dtype=np.float32)

    if _rembg_session is None:
        logger.info("Loading rembg u2net session (first use)...")
        _rembg_session = new_session("u2net")
        logger.info("rembg session loaded")

    # rembg expects BGR or RGB uint8
    mask = remove(
        camera_frame,
        session=_rembg_session,
        only_mask=True,
    )

    # mask is uint8 (H, W)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask_f = mask.astype(np.float32) / 255.0

    # Feather
    if feather > 0.5:
        ksize = int(feather) * 2 + 1
        mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)

    return mask_f
