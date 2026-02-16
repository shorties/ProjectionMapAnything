"""Gray code structured light calibration for projector-camera systems.

Projects binary stripe patterns (Gray codes) and decodes camera captures to
build a dense pixel mapping from camera space to projector space.  The mapping
is stored as two float32 arrays (map_x, map_y) that can be used with
``cv2.remap`` to warp images from camera perspective to projector perspective.

Ported from the standalone app with improvements:
- Multi-frame averaging per pattern for noise rejection
- Per-bit reliability threshold (rejects bit-boundary pixels)
- Configurable decode threshold
- Vectorized Gray-to-binary decode
- Gaussian splat fill (blur sum/count) for dense correspondence
- Morphological cleanup + spatial consistency filtering
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
import torch


# -- Gray code utilities ------------------------------------------------------


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
    bit: int,
    axis: int,
    inverted: bool,
    proj_w: int,
    proj_h: int,
) -> np.ndarray:
    """Generate a single Gray code pattern image.

    Returns (proj_h, proj_w) uint8.
    """
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


# -- Calibration state machine ------------------------------------------------


class CalibrationPhase(Enum):
    IDLE = auto()
    WHITE = auto()
    BLACK = auto()
    PATTERNS = auto()
    DECODING = auto()
    DONE = auto()


class CalibrationState:
    """Manages the multi-frame Gray code calibration sequence.

    Usage
    -----
    1. Call ``start()`` to begin calibration.
    2. Each frame, call ``step(camera_frame)`` which returns the pattern tensor
       to output (or ``None`` when decoding/done).
    3. When ``phase == DONE``, retrieve the mapping with ``get_mapping()``.
    """

    def __init__(
        self,
        proj_w: int,
        proj_h: int,
        settle_frames: int = 6,
        capture_frames: int = 3,
        decode_threshold: float = 20.0,
        bit_threshold: float = 3.0,
        morph_cleanup: bool = True,
        morph_kernel_size: int = 5,
        spatial_consistency: bool = True,
        consistency_max_diff: float = 3.0,
        fill_kernel_size: int = 11,
    ):
        self.proj_w = proj_w
        self.proj_h = proj_h
        self.settle_frames = settle_frames
        self.capture_frames = max(1, capture_frames)
        self.decode_threshold = decode_threshold
        self.bit_threshold = bit_threshold
        self.morph_cleanup = morph_cleanup
        self.morph_kernel_size = morph_kernel_size | 1  # ensure odd
        self.spatial_consistency = spatial_consistency
        self.consistency_max_diff = consistency_max_diff
        self.fill_kernel_size = fill_kernel_size | 1  # ensure odd

        self.bits_x = _num_bits(proj_w)
        self.bits_y = _num_bits(proj_h)

        # Total patterns: white + black + 2*(bits_x + bits_y) pos/neg pairs
        self.total_patterns = 2 + 2 * (self.bits_x + self.bits_y)

        self.phase = CalibrationPhase.IDLE
        self._pattern_index = 0
        self._settle_counter = 0
        self._waiting_for_settle = False
        self._frame_count = 0  # frames captured for current pattern

        # Each slot holds a list of float32 frames (for multi-frame averaging)
        self._captures: list[list[np.ndarray]] = [[] for _ in range(self.total_patterns)]

        # Result mapping arrays
        self.map_x: np.ndarray | None = None
        self.map_y: np.ndarray | None = None
        self.proj_valid_mask: np.ndarray | None = None  # bool, valid before inpainting

    def start(self) -> None:
        self.phase = CalibrationPhase.WHITE
        self._pattern_index = 0
        self._settle_counter = 0
        self._waiting_for_settle = True
        self._frame_count = 0
        self._captures = [[] for _ in range(self.total_patterns)]
        self.map_x = None
        self.map_y = None
        self.proj_valid_mask = None

    def _camera_to_gray(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """Convert a (1, H, W, C) tensor to a grayscale uint8 numpy array.

        Handles both float [0,1] and uint8 [0,255] input formats.
        """
        img = frame_tensor.squeeze(0).cpu().numpy()
        # Convert float [0,1] to uint8 [0,255]
        if img.dtype != np.uint8:
            if img.max() <= 1.5:
                img = (img * 255.0).clip(0, 255)
            img = img.astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def _pattern_to_tensor(self, pattern: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert a (H, W) uint8 pattern to (1, H, W, 3) float32 [0,1] tensor."""
        rgb = np.stack([pattern] * 3, axis=-1).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).unsqueeze(0).to(device)

    def _current_capture_index(self) -> int:
        """Map current phase/pattern_index to the flat capture list index."""
        if self.phase == CalibrationPhase.WHITE:
            return 0
        elif self.phase == CalibrationPhase.BLACK:
            return 1
        else:
            return 2 + self._pattern_index

    def step(
        self,
        camera_frame: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Advance calibration by one frame.

        Returns the pattern to project as (1, H, W, 3) [0,1], or None if
        calibration just finished.
        """
        # Handle settle delay
        if self._waiting_for_settle:
            self._settle_counter += 1
            if self._settle_counter < self.settle_frames:
                return self._current_pattern(device)
            self._waiting_for_settle = False
            self._settle_counter = 0
            self._frame_count = 0

        # Capture frame (accumulate for multi-frame averaging)
        gray = self._camera_to_gray(camera_frame)
        cap_idx = self._current_capture_index()
        self._captures[cap_idx].append(gray.astype(np.float32))
        self._frame_count += 1

        # Check if we have enough frames for this pattern
        if self._frame_count < self.capture_frames:
            # Need more frames — keep projecting the same pattern (no settle)
            return self._current_pattern(device)

        # Done with this pattern, advance to next
        if self.phase == CalibrationPhase.WHITE:
            self.phase = CalibrationPhase.BLACK
            self._waiting_for_settle = True
            return self._current_pattern(device)

        if self.phase == CalibrationPhase.BLACK:
            self.phase = CalibrationPhase.PATTERNS
            self._pattern_index = 0
            self._waiting_for_settle = True
            return self._current_pattern(device)

        if self.phase == CalibrationPhase.PATTERNS:
            self._pattern_index += 1
            if self._pattern_index >= 2 * (self.bits_x + self.bits_y):
                self.phase = CalibrationPhase.DECODING
                self._decode()
                self.phase = CalibrationPhase.DONE
                return None
            self._waiting_for_settle = True
            return self._current_pattern(device)

        return None

    def _current_pattern(self, device: torch.device) -> torch.Tensor:
        if self.phase == CalibrationPhase.WHITE:
            pattern = np.full((self.proj_h, self.proj_w), 255, dtype=np.uint8)
        elif self.phase == CalibrationPhase.BLACK:
            pattern = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
        elif self.phase == CalibrationPhase.PATTERNS:
            idx = self._pattern_index
            total_x = 2 * self.bits_x
            if idx < total_x:
                bit = idx // 2
                inverted = idx % 2 == 1
                pattern = generate_pattern(bit, axis=0, inverted=inverted,
                                           proj_w=self.proj_w, proj_h=self.proj_h)
            else:
                y_idx = idx - total_x
                bit = y_idx // 2
                inverted = y_idx % 2 == 1
                pattern = generate_pattern(bit, axis=1, inverted=inverted,
                                           proj_w=self.proj_w, proj_h=self.proj_h)
        else:
            pattern = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)

        return self._pattern_to_tensor(pattern, device)

    def _get_averaged(self, index: int) -> np.ndarray:
        """Return the mean of accumulated frames for a capture index (float32)."""
        frames = self._captures[index]
        if not frames:
            raise ValueError(f"No captures for index {index}")
        if len(frames) == 1:
            return frames[0]
        return np.mean(frames, axis=0).astype(np.float32)

    def _decode(self) -> None:
        """Decode captured Gray code patterns into a camera->projector mapping.

        Uses the same proven methodology as the standalone app:
        1. Multi-frame averaged captures for noise rejection
        2. Per-bit reliability thresholding
        3. Vectorized Gray-to-binary decode
        4. Morphological cleanup + spatial consistency
        5. Gaussian splat fill for dense correspondence
        6. Inpainting remaining holes
        """
        white_f = self._get_averaged(0)
        black_f = self._get_averaged(1)

        h, w = white_f.shape[:2]
        diff = white_f - black_f
        valid_mask = diff > self.decode_threshold

        # Decode X (column) Gray codes with per-bit reliability
        base = 2  # skip white + black
        decoded_x = np.zeros((h, w), dtype=np.int32)
        x_reliable = np.ones((h, w), dtype=bool)
        for bit_idx in range(self.bits_x):
            pos_idx = base + bit_idx * 2
            neg_idx = base + bit_idx * 2 + 1
            if self._captures[pos_idx] and self._captures[neg_idx]:
                pos_f = self._get_averaged(pos_idx)
                neg_f = self._get_averaged(neg_idx)
                bit_val = (pos_f > neg_f).astype(np.int32)
                decoded_x |= bit_val << bit_idx
                if self.bit_threshold > 0:
                    bit_diff = np.abs(pos_f - neg_f)
                    x_reliable &= (bit_diff >= self.bit_threshold)

        # Decode Y (row) Gray codes with per-bit reliability
        base_y = base + self.bits_x * 2
        decoded_y = np.zeros((h, w), dtype=np.int32)
        y_reliable = np.ones((h, w), dtype=bool)
        for bit_idx in range(self.bits_y):
            pos_idx = base_y + bit_idx * 2
            neg_idx = base_y + bit_idx * 2 + 1
            if self._captures[pos_idx] and self._captures[neg_idx]:
                pos_f = self._get_averaged(pos_idx)
                neg_f = self._get_averaged(neg_idx)
                bit_val = (pos_f > neg_f).astype(np.int32)
                decoded_y |= bit_val << bit_idx
                if self.bit_threshold > 0:
                    bit_diff = np.abs(pos_f - neg_f)
                    y_reliable &= (bit_diff >= self.bit_threshold)

        # Apply bit reliability mask
        if self.bit_threshold > 0:
            valid_mask = valid_mask & x_reliable & y_reliable

        # Vectorized Gray-to-binary decode (much faster than per-pixel loop)
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

        # Clamp to projector bounds
        decoded_x = np.clip(decoded_x, 0, self.proj_w - 1)
        decoded_y = np.clip(decoded_y, 0, self.proj_h - 1)

        # -- Post-processing on camera-space decode --
        if self.morph_cleanup:
            k = self.morph_kernel_size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_u8 = valid_mask.astype(np.uint8) * 255
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            valid_mask = mask_u8 > 127

        if self.spatial_consistency:
            ksize = max(self.morph_kernel_size, 5)
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            mask_f = valid_mask.astype(np.float32)
            dx_f = decoded_x.astype(np.float32) * mask_f
            dy_f = decoded_y.astype(np.float32) * mask_f
            local_sum_x = cv2.blur(dx_f, (ksize, ksize))
            local_sum_y = cv2.blur(dy_f, (ksize, ksize))
            local_count = cv2.blur(mask_f, (ksize, ksize))
            has_neighbors = local_count > 0.01
            safe_count = np.where(has_neighbors, local_count, 1.0)
            local_mean_x = np.where(has_neighbors, local_sum_x / safe_count,
                                    decoded_x.astype(np.float32))
            local_mean_y = np.where(has_neighbors, local_sum_y / safe_count,
                                    decoded_y.astype(np.float32))
            diff_x = np.abs(decoded_x.astype(np.float32) - local_mean_x)
            diff_y = np.abs(decoded_y.astype(np.float32) - local_mean_y)
            consistent = ((diff_x <= self.consistency_max_diff) &
                          (diff_y <= self.consistency_max_diff))
            valid_mask = valid_mask & consistent

        # -- Build inverse mapping: projector pixel -> camera pixel --
        # Accumulate weighted sums for Gaussian splat
        sum_cx = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        sum_cy = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        count = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)

        cam_ys, cam_xs = np.where(valid_mask)
        proj_xs = decoded_x[cam_ys, cam_xs]
        proj_ys = decoded_y[cam_ys, cam_xs]

        np.add.at(sum_cx, (proj_ys, proj_xs), cam_xs.astype(np.float32))
        np.add.at(sum_cy, (proj_ys, proj_xs), cam_ys.astype(np.float32))
        np.add.at(count, (proj_ys, proj_xs), 1.0)

        # Gaussian splat: spread correspondences to fill gaps
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

        # Store validity mask BEFORE inpainting
        self.proj_valid_mask = valid_proj.copy()

        # Inpaint only small internal holes — NOT large uncovered boundary regions
        # (blanket inpainting of large regions causes fold-overs / distortion)
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

        self.map_x = map_x
        self.map_y = map_y

    def get_mapping(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.map_x is not None and self.map_y is not None:
            return self.map_x, self.map_y
        return None

    @property
    def progress(self) -> float:
        """Calibration progress as a float in [0, 1]."""
        if self.phase == CalibrationPhase.IDLE:
            return 0.0
        if self.phase == CalibrationPhase.DONE:
            return 1.0

        # Count completed captures (each pattern can have N frames)
        total_captures = self.total_patterns * self.capture_frames
        completed = 0

        for slot in self._captures:
            completed += len(slot)

        # Add current sub-frame progress
        return min(completed / max(total_captures, 1), 0.99)


# -- Calibration file I/O ----------------------------------------------------


def save_calibration(
    map_x: np.ndarray,
    map_y: np.ndarray,
    path: str | Path,
    proj_w: int,
    proj_h: int,
) -> None:
    """Save calibration mapping to a JSON file."""
    data = {
        "version": 1,
        "projector_width": proj_w,
        "projector_height": proj_h,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "map_x": map_x.tolist(),
        "map_y": map_y.tolist(),
    }
    Path(path).write_text(json.dumps(data))


def load_calibration(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    """Load calibration mapping from a JSON file.

    Returns (map_x, map_y, proj_w, proj_h, timestamp_iso).
    """
    data = json.loads(Path(path).read_text())
    map_x = np.array(data["map_x"], dtype=np.float32)
    map_y = np.array(data["map_y"], dtype=np.float32)
    timestamp = data.get("timestamp", "unknown")
    return map_x, map_y, data["projector_width"], data["projector_height"], timestamp
