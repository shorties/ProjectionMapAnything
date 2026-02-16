"""Gray code calibration with proper timing for standalone app.

Unlike the Scope plugin version (which relies on frame counting), this uses
real millisecond timing and direct projector window control for reliable
calibration.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


# ── Gray code utilities ──────────────────────────────────────────────────────


def _gray_encode(n: int) -> int:
    return n ^ (n >> 1)


def _gray_decode(g: int) -> int:
    n = g
    mask = n >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


def _num_bits(resolution: int) -> int:
    return max(1, math.ceil(math.log2(max(resolution, 2))))


def generate_pattern(
    bit: int, axis: int, inverted: bool, proj_w: int, proj_h: int
) -> np.ndarray:
    """Generate a Gray code pattern. Returns (proj_h, proj_w) uint8."""
    if axis == 0:
        indices = np.arange(proj_w, dtype=np.int32)
        gray = np.vectorize(_gray_encode)(indices)
        stripe = ((gray >> bit) & 1).astype(np.uint8) * 255
        pattern = np.tile(stripe[np.newaxis, :], (proj_h, 1))
    else:
        indices = np.arange(proj_h, dtype=np.int32)
        gray = np.vectorize(_gray_encode)(indices)
        stripe = ((gray >> bit) & 1).astype(np.uint8) * 255
        pattern = np.tile(stripe[:, np.newaxis], (1, proj_w))

    if inverted:
        pattern = 255 - pattern
    return pattern


# ── Calibration runner ───────────────────────────────────────────────────────


class CalibrationRunner:
    """Runs the full calibration sequence with real timing.

    Usage:
        runner = CalibrationRunner(proj_w, proj_h, settle_ms=200)
        patterns = runner.get_pattern_sequence()
        for i, pattern in enumerate(patterns):
            display_on_projector(pattern)
            time.sleep(settle_ms / 1000)
            capture = capture_from_camera()
            runner.feed_capture(i, capture)
        map_x, map_y = runner.decode()
    """

    def __init__(
        self,
        proj_w: int,
        proj_h: int,
        settle_ms: float = 200,
        capture_frames: int = 3,
        decode_threshold: float = 20.0,
        morph_cleanup: bool = True,
        morph_kernel_size: int = 5,
        spatial_consistency: bool = True,
        consistency_max_diff: float = 3.0,
        bit_threshold: float = 3.0,
        fill_kernel_size: int = 11,
    ):
        self.proj_w = proj_w
        self.proj_h = proj_h
        self.settle_ms = settle_ms
        self.capture_frames = max(1, capture_frames)
        self.decode_threshold = decode_threshold
        self.morph_cleanup = morph_cleanup
        self.morph_kernel_size = morph_kernel_size | 1  # ensure odd
        self.spatial_consistency = spatial_consistency
        self.consistency_max_diff = consistency_max_diff
        self.bit_threshold = bit_threshold
        self.fill_kernel_size = fill_kernel_size | 1  # ensure odd

        self.bits_x = _num_bits(proj_w)
        self.bits_y = _num_bits(proj_h)

        # Each slot holds a list of frames (for multi-frame averaging)
        self._captures: list[list[np.ndarray]] = []
        self._patterns: list[dict] = []
        self._build_pattern_sequence()

        # Forward correspondence (populated after decode())
        # decoded_col[cam_y, cam_x] = projector column for that camera pixel
        # decoded_row[cam_y, cam_x] = projector row for that camera pixel
        self.decoded_col: np.ndarray | None = None
        self.decoded_row: np.ndarray | None = None
        self.decode_mask: np.ndarray | None = None  # bool array of valid pixels (camera space)
        self.proj_valid_mask: np.ndarray | None = None  # bool array of valid pixels (projector space)

        # Decode statistics (populated after decode())
        self.total_camera_pixels: int = 0
        self.valid_camera_pixels: int = 0       # camera pixels with valid Gray code decode
        self.valid_projector_pixels: int = 0    # projector pixels that got correspondence
        self.total_projector_pixels: int = proj_w * proj_h
        self.camera_coverage_pct: float = 0.0   # % of camera pixels recognized
        self.projector_coverage_pct: float = 0.0 # % of projector pixels mapped

    def _build_pattern_sequence(self) -> None:
        """Pre-build the ordered list of patterns to project."""
        self._patterns = []

        # White reference
        self._patterns.append({"type": "white"})
        # Black reference
        self._patterns.append({"type": "black"})

        # X-axis Gray codes (pos + neg pairs)
        for bit in range(self.bits_x):
            self._patterns.append({"type": "gray", "axis": 0, "bit": bit, "inv": False})
            self._patterns.append({"type": "gray", "axis": 0, "bit": bit, "inv": True})

        # Y-axis Gray codes (pos + neg pairs)
        for bit in range(self.bits_y):
            self._patterns.append({"type": "gray", "axis": 1, "bit": bit, "inv": False})
            self._patterns.append({"type": "gray", "axis": 1, "bit": bit, "inv": True})

        self._captures = [[] for _ in self._patterns]

    @property
    def total_patterns(self) -> int:
        return len(self._patterns)

    def get_pattern_image(self, index: int) -> np.ndarray:
        """Get the pattern image for a given index. Returns (H, W) uint8."""
        p = self._patterns[index]
        if p["type"] == "white":
            return np.full((self.proj_h, self.proj_w), 255, dtype=np.uint8)
        elif p["type"] == "black":
            return np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
        else:
            return generate_pattern(
                p["bit"], p["axis"], p["inv"], self.proj_w, self.proj_h
            )

    def feed_capture(self, index: int, capture: np.ndarray) -> None:
        """Feed a camera capture for a given pattern index (accumulates for averaging)."""
        if capture.ndim == 3:
            capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
        self._captures[index].append(capture.astype(np.float32))

    def captures_for_index(self, index: int) -> int:
        """Return how many frames have been captured for the given pattern index."""
        return len(self._captures[index])

    def _get_averaged(self, index: int) -> np.ndarray:
        """Return the mean of accumulated frames for a pattern (float32)."""
        frames = self._captures[index]
        if not frames:
            raise ValueError(f"No captures for pattern index {index}")
        if len(frames) == 1:
            return frames[0]
        return np.mean(frames, axis=0).astype(np.float32)

    def decode(self) -> tuple[np.ndarray, np.ndarray]:
        """Decode all captures into a projector-to-camera pixel mapping.

        Returns (map_x, map_y) each of shape (proj_h, proj_w) float32.
        """
        if not self._captures[0] or not self._captures[1]:
            raise ValueError("Missing white/black reference captures")

        # Use averaged frames (float32) for better noise rejection
        white_f = self._get_averaged(0)
        black_f = self._get_averaged(1)

        h, w = white_f.shape[:2]
        diff = white_f - black_f
        valid_mask = diff > self.decode_threshold

        # Decode X Gray codes using averaged frames
        # Track per-bit reliability: reject pixels at bit boundaries
        decoded_x = np.zeros((h, w), dtype=np.int32)
        x_reliable = np.ones((h, w), dtype=bool)
        base = 2  # Skip white and black
        for bit in range(self.bits_x):
            pos_idx = base + bit * 2
            neg_idx = base + bit * 2 + 1
            if self._captures[pos_idx] and self._captures[neg_idx]:
                pos_f = self._get_averaged(pos_idx)
                neg_f = self._get_averaged(neg_idx)
                bit_val = (pos_f > neg_f).astype(np.int32)
                decoded_x |= bit_val << bit
                if self.bit_threshold > 0:
                    bit_diff = np.abs(pos_f - neg_f)
                    x_reliable &= (bit_diff >= self.bit_threshold)

        # Decode Y Gray codes using averaged frames
        base_y = base + self.bits_x * 2
        decoded_y = np.zeros((h, w), dtype=np.int32)
        y_reliable = np.ones((h, w), dtype=bool)
        for bit in range(self.bits_y):
            pos_idx = base_y + bit * 2
            neg_idx = base_y + bit * 2 + 1
            if self._captures[pos_idx] and self._captures[neg_idx]:
                pos_f = self._get_averaged(pos_idx)
                neg_f = self._get_averaged(neg_idx)
                bit_val = (pos_f > neg_f).astype(np.int32)
                decoded_y |= bit_val << bit
                if self.bit_threshold > 0:
                    bit_diff = np.abs(pos_f - neg_f)
                    y_reliable &= (bit_diff >= self.bit_threshold)

        # Apply bit reliability mask
        if self.bit_threshold > 0:
            valid_mask = valid_mask & x_reliable & y_reliable

        # Vectorized Gray-to-binary decode
        binary_x = decoded_x.copy()
        shift = binary_x >> 1
        while np.any(shift):
            binary_x ^= shift
            shift >>= 1
        decoded_x = binary_x

        binary_y = decoded_y.copy()
        shift = binary_y >> 1
        while np.any(shift):
            binary_y ^= shift
            shift >>= 1
        decoded_y = binary_y

        decoded_x = np.clip(decoded_x, 0, self.proj_w - 1)
        decoded_y = np.clip(decoded_y, 0, self.proj_h - 1)

        # ── Post-processing on camera-space decode ─────────────────────
        if self.morph_cleanup:
            k = self.morph_kernel_size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            # Close then open the validity mask to fill small holes and remove specks
            mask_u8 = valid_mask.astype(np.uint8) * 255
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            valid_mask = mask_u8 > 127
            # Note: coordinate smoothing is deferred to the Gaussian splat
            # in the inverse mapping, avoiding boundary artifacts from
            # median filtering across valid/invalid regions.

        if self.spatial_consistency:
            # Reject pixels where decoded coord differs from local mean.
            # Compute weighted local average (only valid pixels contribute)
            # using sum/count approach to avoid zero-bleed from invalid regions.
            ksize = max(self.morph_kernel_size, 5)
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            mask_f = valid_mask.astype(np.float32)
            dx_f = decoded_x.astype(np.float32) * mask_f
            dy_f = decoded_y.astype(np.float32) * mask_f
            local_sum_x = cv2.blur(dx_f, (ksize, ksize))
            local_sum_y = cv2.blur(dy_f, (ksize, ksize))
            local_count = cv2.blur(mask_f, (ksize, ksize))
            # Avoid division by zero — only check where we have neighbors
            has_neighbors = local_count > 0.01
            safe_count = np.where(has_neighbors, local_count, 1.0)
            local_mean_x = np.where(has_neighbors, local_sum_x / safe_count, decoded_x.astype(np.float32))
            local_mean_y = np.where(has_neighbors, local_sum_y / safe_count, decoded_y.astype(np.float32))
            diff_x = np.abs(decoded_x.astype(np.float32) - local_mean_x)
            diff_y = np.abs(decoded_y.astype(np.float32) - local_mean_y)
            consistent = (diff_x <= self.consistency_max_diff) & (diff_y <= self.consistency_max_diff)
            valid_mask = valid_mask & consistent

        # Store forward correspondences for RoomAlive-style calibration
        self.decoded_col = decoded_x  # cam pixel → proj col
        self.decoded_row = decoded_y  # cam pixel → proj row
        self.decode_mask = valid_mask

        # Compute statistics
        self.total_camera_pixels = h * w
        self.valid_camera_pixels = int(np.count_nonzero(valid_mask))
        self.camera_coverage_pct = (
            self.valid_camera_pixels / max(self.total_camera_pixels, 1) * 100.0
        )

        # ── Build inverse mapping: projector pixel → camera pixel ─────
        # Accumulate weighted sums (initialized to 0 for Gaussian splat)
        sum_cx = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        sum_cy = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        count = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)

        cam_ys, cam_xs = np.where(valid_mask)
        proj_xs = decoded_x[cam_ys, cam_xs]
        proj_ys = decoded_y[cam_ys, cam_xs]

        np.add.at(sum_cx, (proj_ys, proj_xs), cam_xs.astype(np.float32))
        np.add.at(sum_cy, (proj_ys, proj_xs), cam_ys.astype(np.float32))
        np.add.at(count, (proj_ys, proj_xs), 1.0)

        # Sparse stats (before fill)
        sparse_valid = int(np.count_nonzero(count > 0))

        # Gaussian splat: spread correspondences to fill gaps between
        # sparse projector pixels. Each valid point's contribution is
        # smeared over a (k x k) neighborhood, then weighted-averaged.
        # This is the same method that produced the perfect RGB output.
        fk = self.fill_kernel_size
        if fk > 1:
            fk = fk if fk % 2 == 1 else fk + 1
            sum_cx = cv2.GaussianBlur(sum_cx, (fk, fk), 0)
            sum_cy = cv2.GaussianBlur(sum_cy, (fk, fk), 0)
            count = cv2.GaussianBlur(count, (fk, fk), 0)

        valid_proj = count > 0.01
        map_x = np.full((self.proj_h, self.proj_w), -1.0, dtype=np.float32)
        map_y = np.full((self.proj_h, self.proj_w), -1.0, dtype=np.float32)
        map_x[valid_proj] = sum_cx[valid_proj] / count[valid_proj]
        map_y[valid_proj] = sum_cy[valid_proj] / count[valid_proj]

        # Projector coverage stats
        self.valid_projector_pixels = int(np.count_nonzero(valid_proj))
        self.projector_coverage_pct = (
            self.valid_projector_pixels / max(self.total_projector_pixels, 1) * 100.0
        )
        self.sparse_projector_pixels = sparse_valid

        # Store validity mask BEFORE inpainting
        self.proj_valid_mask = valid_proj.copy()

        # Only inpaint small internal holes — NOT large uncovered boundary
        # regions.  Inpainting large holes extrapolates coordinates, creating
        # fold-overs that distort the entire output.  Small internal holes
        # (< 0.5% of image) are safe to fill via inpainting.
        hole_mask = (~valid_proj).astype(np.uint8) * 255
        if np.any(~valid_proj):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                hole_mask, connectivity=8,
            )
            max_small_area = max(int(self.proj_w * self.proj_h * 0.005), 100)
            small_hole_mask = np.zeros_like(hole_mask)
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < max_small_area:
                    small_hole_mask[labels == label_id] = 255
            if np.any(small_hole_mask):
                inpaint_r = max(fk, 5)
                map_x = cv2.inpaint(map_x, small_hole_mask, inpaint_r, cv2.INPAINT_NS)
                map_y = cv2.inpaint(map_y, small_hole_mask, inpaint_r, cv2.INPAINT_NS)

        return map_x, map_y


# ── Map cleanup ──────────────────────────────────────────────────────────────


def sanitize_correspondence_maps(
    map_x: np.ndarray, map_y: np.ndarray,
    max_deviation: float = 20.0,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove fold-overs from correspondence maps via iterative outlier removal.

    Uses weighted local smoothing to estimate what the map *should* look
    like, then removes pixels that deviate significantly.  Each iteration
    refines the estimate as outliers are excluded from the smooth, so the
    prediction gets more accurate and subtler fold-overs get caught.

    Preserves the original per-pixel accuracy in valid regions.
    """
    h, w = map_x.shape
    valid = (map_x >= 0) & (map_y >= 0)
    had_valid = valid.copy()

    if not np.any(valid):
        return map_x, map_y

    ksize = 101
    ksize = ksize if ksize % 2 == 1 else ksize + 1

    map_x = map_x.copy().astype(np.float64)
    map_y = map_y.copy().astype(np.float64)

    for it in range(iterations):
        # Weighted Gaussian smooth: only valid pixels contribute
        mask_f = valid.astype(np.float64)
        wx = np.where(valid, map_x, 0.0) * mask_f
        wy = np.where(valid, map_y, 0.0) * mask_f

        smooth_wx = cv2.GaussianBlur(wx, (ksize, ksize), 0)
        smooth_wy = cv2.GaussianBlur(wy, (ksize, ksize), 0)
        smooth_count = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)

        # Avoid div by zero — only check where we have local data
        has_data = smooth_count > 0.01
        safe_count = np.where(has_data, smooth_count, 1.0)
        pred_x = smooth_wx / safe_count
        pred_y = smooth_wy / safe_count

        # Compare original to local prediction
        diff = np.sqrt(
            (map_x - pred_x) ** 2 + (map_y - pred_y) ** 2
        )
        newly_bad = (diff > max_deviation) & valid

        n_removed = int(newly_bad.sum())
        if n_removed == 0:
            break

        valid = valid & ~newly_bad

    # ── Enforce per-row / per-column monotonicity ───────────────────
    # map_x must be non-decreasing along each row (left → right).
    # map_y must be non-decreasing along each column (top → bottom).
    # The running-maximum approach catches fold-overs that the smooth
    # missed (gradual reversals within a large neighborhood).
    noise_tol = 3.0
    for y in range(h):
        idx = np.where(valid[y])[0]
        if len(idx) < 2:
            continue
        vals = map_x[y, idx]
        cummax = np.maximum.accumulate(vals)
        back = (cummax - vals) > noise_tol
        for i in np.where(back)[0]:
            valid[y, idx[i]] = False

    for x in range(w):
        idx = np.where(valid[:, x])[0]
        if len(idx) < 2:
            continue
        vals = map_y[idx, x]
        cummax = np.maximum.accumulate(vals)
        back = (cummax - vals) > noise_tol
        for i in np.where(back)[0]:
            valid[idx[i], x] = False

    total_removed = int((had_valid & ~valid).sum())
    if total_removed == 0:
        return map_x.astype(np.float32), map_y.astype(np.float32)

    # Small dilation to clean ragged edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inv_u8 = (~valid & had_valid).astype(np.uint8) * 255
    inv_u8 = cv2.dilate(inv_u8, kernel, iterations=1)
    valid = valid & ~(inv_u8 > 0)

    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    kept = int(np.count_nonzero(map_x >= 0))
    total = h * w
    removed = int((had_valid & ~valid).sum())
    print(f"[calibration] sanitize_maps: removed {removed} bad pixels, "
          f"kept {kept}/{total} ({100*kept/total:.1f}%)")

    return map_x.astype(np.float32), map_y.astype(np.float32)


# ── File I/O ─────────────────────────────────────────────────────────────────


def save_calibration(
    map_x: np.ndarray, map_y: np.ndarray,
    path: str | Path, proj_w: int, proj_h: int,
) -> None:
    data = {
        "version": 1,
        "projector_width": proj_w,
        "projector_height": proj_h,
        "map_x": map_x.tolist(),
        "map_y": map_y.tolist(),
    }
    Path(path).write_text(json.dumps(data))


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    data = json.loads(Path(path).read_text())
    map_x = np.array(data["map_x"], dtype=np.float32)
    map_y = np.array(data["map_y"], dtype=np.float32)
    map_x, map_y = sanitize_correspondence_maps(map_x, map_y)
    return map_x, map_y, data["projector_width"], data["projector_height"]
