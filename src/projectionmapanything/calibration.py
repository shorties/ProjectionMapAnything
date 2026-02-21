"""Gray code structured light calibration for projector-camera systems.

Projects binary Gray code stripe patterns, captures them with a camera,
and decodes the per-pixel correspondence to build a dense mapping from
projector pixels to camera pixels (map_x, map_y) usable with ``cv2.remap``.

The decode pipeline:
1. Multi-frame averaged captures for noise rejection
2. Adaptive thresholding (Otsu on white-black diff) with per-bit fallback
3. Per-bit reliability filtering
4. Vectorized Gray-to-binary decode
5. Morphological cleanup + spatial consistency
6. Gaussian splat fill for dense correspondence
7. Outlier rejection (median-based)
8. Final smoothing (median + bilateral filter)
9. Inpainting remaining small holes
"""

from __future__ import annotations

import json
import logging
import math
import time as _time
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[PMA Cal] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)


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

    Parameters
    ----------
    bit : int
        Which bit of the Gray code to display.
    axis : int
        0 = column encoding (vertical stripes), 1 = row encoding (horizontal).
    inverted : bool
        If True, return the inverse pattern (255 - pattern).
    proj_w, proj_h : int
        Projector resolution.

    Returns
    -------
    np.ndarray
        (proj_h, proj_w) uint8 pattern image.
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
    2. Each frame, call ``step(camera_frame)`` which returns the pattern
       tensor to output (or ``None`` when decoding/done).
    3. When ``phase == DONE``, retrieve the mapping with ``get_mapping()``.
    """

    def __init__(
        self,
        proj_w: int,
        proj_h: int,
        *,
        settle_frames: int = 60,
        capture_frames: int = 3,
        decode_threshold: float = 10.0,
        bit_threshold: float = 3.0,
        morph_cleanup: bool = True,
        morph_kernel_size: int = 5,
        spatial_consistency: bool = True,
        consistency_max_diff: float = 3.0,
        fill_kernel_size: int = 11,
        max_brightness: int = 255,
        change_threshold: float = 5.0,
        stability_threshold: float = 3.0,
    ):
        self.proj_w = proj_w
        self.proj_h = proj_h
        self.settle_frames = settle_frames  # timeout fallback only
        self.capture_frames = max(1, capture_frames)
        self.max_brightness = max(10, min(255, max_brightness))
        self.decode_threshold = decode_threshold
        self.bit_threshold = bit_threshold
        self.morph_cleanup = morph_cleanup
        self.morph_kernel_size = morph_kernel_size | 1  # ensure odd
        self.spatial_consistency = spatial_consistency
        self.consistency_max_diff = consistency_max_diff
        self.fill_kernel_size = fill_kernel_size | 1  # ensure odd
        self.change_threshold = change_threshold
        self.stability_threshold = stability_threshold

        # Gray code bit counts
        self.bits_x = _num_bits(proj_w)
        self.bits_y = _num_bits(proj_h)

        # Pattern count: pos/neg pairs for each bit, 2 axes
        self._num_pattern_steps = 2 * (self.bits_x + self.bits_y)

        # Total: white + black + all pattern steps
        self.total_patterns = 2 + self._num_pattern_steps

        self.phase = CalibrationPhase.IDLE
        self._pattern_index = 0
        self._frame_count = 0

        # Change-detection settle state
        self._waiting_for_settle = False
        self._settle_counter = 0
        self._settle_baseline: np.ndarray | None = None
        self._prev_settle_frame: np.ndarray | None = None
        self._stable_count: int = 0
        self._change_detected: bool = False
        self._settle_start_time: float = 0.0

        # Cached pattern tensor (avoids regenerating every frame)
        self._cached_pattern: torch.Tensor | None = None
        self._cached_pattern_key: tuple | None = None

        # Each slot holds a list of float32 frames (for multi-frame averaging)
        self._captures: list[list[np.ndarray]] = [
            [] for _ in range(self.total_patterns)
        ]

        # Result mapping arrays
        self.map_x: np.ndarray | None = None
        self.map_y: np.ndarray | None = None
        self.proj_valid_mask: np.ndarray | None = None

        logger.info(
            "CalibrationState: %dx%d, %d patterns (%d bits_x, %d bits_y), "
            "settle=%d (timeout), capture=%d",
            proj_w, proj_h, self.total_patterns,
            self.bits_x, self.bits_y,
            settle_frames, capture_frames,
        )

    # -- Public API -----------------------------------------------------------

    def start(self) -> None:
        """Begin the calibration sequence."""
        self.phase = CalibrationPhase.WHITE
        self._pattern_index = 0
        self._frame_count = 0
        self._captures = [[] for _ in range(self.total_patterns)]
        self.map_x = None
        self.map_y = None
        self.proj_valid_mask = None
        self._begin_settle()

    def get_mapping(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (map_x, map_y) if calibration is done, else None."""
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
        total_captures = self.total_patterns * self.capture_frames
        completed = sum(len(slot) for slot in self._captures)
        return min(completed / max(total_captures, 1), 0.99)

    def get_pattern_info(self) -> str:
        """Human-readable description of the current calibration pattern."""
        if self.phase == CalibrationPhase.WHITE:
            return "Capturing white reference"
        elif self.phase == CalibrationPhase.BLACK:
            return "Capturing black reference"
        elif self.phase == CalibrationPhase.DECODING:
            return "Decoding patterns..."
        elif self.phase != CalibrationPhase.PATTERNS:
            return ""

        captured = sum(len(s) for s in self._captures)
        total_cap = self.total_patterns * self.capture_frames
        idx = self._pattern_index
        total_x = 2 * self.bits_x

        if idx < total_x:
            bit = idx // 2 + 1
            inv = "inv" if idx % 2 == 1 else "pos"
            return (
                f"X bit {bit}/{self.bits_x} ({inv})"
                f" — {captured}/{total_cap} captures"
            )
        else:
            y_idx = idx - total_x
            bit = y_idx // 2 + 1
            inv = "inv" if y_idx % 2 == 1 else "pos"
            return (
                f"Y bit {bit}/{self.bits_y} ({inv})"
                f" — {captured}/{total_cap} captures"
            )

    # -- Frame stepping -------------------------------------------------------

    def step(
        self,
        camera_frame: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Advance calibration by one frame.

        Returns the pattern to project as (1, H, W, 3) [0,1], or None if
        calibration just finished.
        """
        # Run decode on a separate frame so the pipeline can push progress
        if self.phase == CalibrationPhase.DECODING:
            logger.info("Running Gray code decode ...")
            self._decode()
            self.phase = CalibrationPhase.DONE
            return None

        if self.phase in (CalibrationPhase.IDLE, CalibrationPhase.DONE):
            return None

        # -- Change-detection settle ------------------------------------------
        if self._waiting_for_settle:
            self._settle_counter += 1
            gray = self._camera_to_gray(camera_frame)
            gray_f = gray.astype(np.float32)
            elapsed = _time.monotonic() - self._settle_start_time

            # Capture baseline on first frame
            if self._settle_baseline is None:
                self._settle_baseline = gray_f.copy()
                self._prev_settle_frame = gray_f.copy()
                return self._current_pattern(device)

            if not self._change_detected:
                # Phase 1: watch for the new pattern to arrive
                structural_diff = float(
                    np.mean(np.abs(gray_f - self._settle_baseline))
                )
                if structural_diff > self.change_threshold:
                    self._change_detected = True
                    self._prev_settle_frame = gray_f.copy()
                    self._stable_count = 0
                    logger.debug(
                        "Pattern change detected: %d frames / %.2fs (diff=%.1f)",
                        self._settle_counter, elapsed, structural_diff,
                    )
                    return self._current_pattern(device)

                # Timeout: no change detected after max settle frames
                if self._settle_counter >= self.settle_frames:
                    logger.warning(
                        "Settle timeout: %d frames / %.1fs — no change "
                        "detected (diff=%.1f)",
                        self._settle_counter, elapsed, structural_diff,
                    )
                    self._waiting_for_settle = False
                    self._frame_count = 0
                else:
                    self._prev_settle_frame = gray_f.copy()
                    return self._current_pattern(device)
            else:
                # Phase 2: change detected, wait for stability
                frame_diff = float(
                    np.mean(np.abs(gray_f - self._prev_settle_frame))
                )
                self._prev_settle_frame = gray_f.copy()

                if frame_diff < self.stability_threshold:
                    self._stable_count += 1
                else:
                    self._stable_count = 0

                if self._stable_count >= 2:
                    logger.info(
                        "Settle done: %d frames / %.2fs (change + %d stable)",
                        self._settle_counter, elapsed, self._stable_count,
                    )
                    self._waiting_for_settle = False
                    self._frame_count = 0
                else:
                    return self._current_pattern(device)

        # -- Capture frame ----------------------------------------------------
        gray = self._camera_to_gray(camera_frame)
        cap_idx = self._current_capture_index()
        if cap_idx >= len(self._captures):
            logger.error(
                "Capture index %d out of range (%d slots) — skipping",
                cap_idx, len(self._captures),
            )
            return None
        self._captures[cap_idx].append(gray.astype(np.float32))
        self._frame_count += 1

        # Debug logging for white/black references
        if cap_idx <= 1 and self._frame_count == 1:
            label = "WHITE" if cap_idx == 0 else "BLACK"
            logger.info(
                "Captured %s reference: shape=%s min=%.1f max=%.1f mean=%.1f",
                label, gray.shape, gray.min(), gray.max(), gray.mean(),
            )

        # Check if we have enough frames for this pattern
        if self._frame_count < self.capture_frames:
            return self._current_pattern(device)

        # -- Advance to next pattern ------------------------------------------
        if self.phase == CalibrationPhase.WHITE:
            self.phase = CalibrationPhase.BLACK
            self._begin_settle()
            return self._current_pattern(device)

        if self.phase == CalibrationPhase.BLACK:
            self.phase = CalibrationPhase.PATTERNS
            self._pattern_index = 0
            self._begin_settle()
            return self._current_pattern(device)

        if self.phase == CalibrationPhase.PATTERNS:
            self._pattern_index += 1
            if self._pattern_index >= self._num_pattern_steps:
                self.phase = CalibrationPhase.DECODING
                return None
            self._begin_settle()
            return self._current_pattern(device)

        return None

    # -- Internal helpers -----------------------------------------------------

    def _begin_settle(self) -> None:
        """Reset change-detection settle state for the next pattern."""
        self._waiting_for_settle = True
        self._settle_counter = 0
        self._stable_count = 0
        self._change_detected = False
        self._settle_baseline = None
        self._prev_settle_frame = None
        self._settle_start_time = _time.monotonic()

    def _camera_to_gray(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """Convert a (1, H, W, C) tensor to grayscale uint8 numpy array."""
        img = frame_tensor.squeeze(0).cpu().numpy()
        if img.dtype != np.uint8:
            if img.max() <= 1.5:
                img = (img * 255.0).clip(0, 255)
            img = img.astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def _pattern_to_tensor(
        self, pattern: np.ndarray, device: torch.device,
    ) -> torch.Tensor:
        """Convert a (H, W) uint8 pattern to (1, H, W, 3) float32 [0,1]."""
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

    def _current_pattern(self, device: torch.device) -> torch.Tensor:
        """Get the current pattern as a tensor, with caching."""
        key = (self.phase, self._pattern_index)
        if self._cached_pattern is not None and self._cached_pattern_key == key:
            return self._cached_pattern

        if self.phase == CalibrationPhase.WHITE:
            pattern = np.full(
                (self.proj_h, self.proj_w), self.max_brightness, dtype=np.uint8,
            )
        elif self.phase == CalibrationPhase.BLACK:
            pattern = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)
        elif self.phase == CalibrationPhase.PATTERNS:
            pattern = self._graycode_pattern_for_index(self._pattern_index)
        else:
            pattern = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)

        tensor = self._pattern_to_tensor(pattern, device)
        self._cached_pattern = tensor
        self._cached_pattern_key = key
        return tensor

    # -- Gray code pattern generation -----------------------------------------

    def _graycode_pattern_for_index(self, idx: int) -> np.ndarray:
        """Generate a Gray code pattern for the given flat index.

        Patterns are projected MSB-first (coarsest stripes first) for
        robustness — coarse bits survive compression and camera noise,
        while fine bits (LSB) are captured last when they matter least.

        Index layout:
          [0, 2*bits_x)  — X-axis (columns): even=positive, odd=inverted
          [2*bits_x, ..) — Y-axis (rows): even=positive, odd=inverted
        """
        total_x = 2 * self.bits_x
        if idx < total_x:
            capture_bit = idx // 2
            bit = (self.bits_x - 1) - capture_bit  # MSB first
            inverted = idx % 2 == 1
            return generate_pattern(
                bit, axis=0, inverted=inverted,
                proj_w=self.proj_w, proj_h=self.proj_h,
            )
        else:
            y_idx = idx - total_x
            capture_bit = y_idx // 2
            bit = (self.bits_y - 1) - capture_bit  # MSB first
            inverted = y_idx % 2 == 1
            return generate_pattern(
                bit, axis=1, inverted=inverted,
                proj_w=self.proj_w, proj_h=self.proj_h,
            )

    # -- Capture averaging ----------------------------------------------------

    def _get_averaged(
        self, index: int, target_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return the mean of accumulated frames for a capture index (float32).

        If *target_shape* ``(h, w)`` is given, any frames that don't match
        are resized first (guards against resolution changes mid-calibration).
        """
        frames = self._captures[index]
        if not frames:
            raise ValueError(f"No captures for index {index}")

        if target_shape is not None:
            th, tw = target_shape
            frames = [
                cv2.resize(f, (tw, th), interpolation=cv2.INTER_LINEAR)
                if f.shape[:2] != (th, tw) else f
                for f in frames
            ]

        if len(frames) == 1:
            return frames[0]
        return np.mean(frames, axis=0).astype(np.float32)

    # -- Gray code decode -----------------------------------------------------

    def _decode(self) -> None:
        """Decode captured Gray code patterns into a camera->projector mapping.

        Pipeline:
        1. Compute valid mask from white-black difference (Otsu threshold)
        2. Decode each bit: compare positive vs inverted pattern
        3. Per-bit reliability filtering
        4. Vectorized Gray-to-binary conversion
        5. Build remap tables via shared pipeline
        """
        white_f = self._get_averaged(0)
        ref_shape = white_f.shape[:2]
        black_f = self._get_averaged(1, target_shape=ref_shape)

        h, w = ref_shape
        diff = white_f - black_f

        logger.info(
            "Decode: camera %dx%d, projector %dx%d, %d+%d bits",
            w, h, self.proj_w, self.proj_h, self.bits_x, self.bits_y,
        )
        logger.info(
            "  WHITE ref: min=%.1f max=%.1f mean=%.1f",
            white_f.min(), white_f.max(), white_f.mean(),
        )
        logger.info(
            "  BLACK ref: min=%.1f max=%.1f mean=%.1f",
            black_f.min(), black_f.max(), black_f.mean(),
        )

        # -- Adaptive threshold: Otsu on white-black difference ---------------
        diff_clipped = np.clip(diff, 0, 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(
            diff_clipped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        effective_threshold = max(float(otsu_val) * 0.5, self.decode_threshold)
        logger.info(
            "  DIFF: min=%.1f max=%.1f mean=%.1f, Otsu=%.0f, threshold=%.1f",
            diff.min(), diff.max(), diff.mean(),
            float(otsu_val), effective_threshold,
        )

        valid_mask = diff > effective_threshold
        n_valid_diff = int(np.count_nonzero(valid_mask))
        logger.info(
            "  Pixels passing threshold: %d/%d (%.1f%%)",
            n_valid_diff, h * w, 100.0 * n_valid_diff / max(h * w, 1),
        )

        # Track per-bit contrast for fallback valid mask
        any_bit_contrast = np.zeros((h, w), dtype=bool)

        # -- Decode X (column) Gray codes ------------------------------------
        base = 2  # skip white + black
        decoded_x = np.zeros((h, w), dtype=np.int32)
        x_reliable = np.ones((h, w), dtype=bool)

        for capture_idx in range(self.bits_x):
            actual_bit = (self.bits_x - 1) - capture_idx  # MSB first
            pos_idx = base + capture_idx * 2
            neg_idx = base + capture_idx * 2 + 1

            if not self._captures[pos_idx] or not self._captures[neg_idx]:
                logger.warning(
                    "  X bit %d (capture %d): missing captures!", actual_bit, capture_idx,
                )
                continue

            pos_f = self._get_averaged(pos_idx, target_shape=ref_shape)
            neg_f = self._get_averaged(neg_idx, target_shape=ref_shape)
            bit_val = (pos_f > neg_f).astype(np.int32)
            decoded_x |= bit_val << actual_bit

            bit_diff = np.abs(pos_f - neg_f)
            any_bit_contrast |= (bit_diff >= self.decode_threshold)
            if self.bit_threshold > 0:
                x_reliable &= (bit_diff >= self.bit_threshold)

            logger.info(
                "  X bit %d (capture %d): pos=%.1f neg=%.1f diff: "
                "min=%.1f max=%.1f mean=%.1f",
                actual_bit, capture_idx, pos_f.mean(), neg_f.mean(),
                bit_diff.min(), bit_diff.max(), bit_diff.mean(),
            )

        # -- Decode Y (row) Gray codes ---------------------------------------
        base_y = base + self.bits_x * 2
        decoded_y = np.zeros((h, w), dtype=np.int32)
        y_reliable = np.ones((h, w), dtype=bool)

        for capture_idx in range(self.bits_y):
            actual_bit = (self.bits_y - 1) - capture_idx  # MSB first
            pos_idx = base_y + capture_idx * 2
            neg_idx = base_y + capture_idx * 2 + 1

            if not self._captures[pos_idx] or not self._captures[neg_idx]:
                logger.warning(
                    "  Y bit %d (capture %d): missing captures!", actual_bit, capture_idx,
                )
                continue

            pos_f = self._get_averaged(pos_idx, target_shape=ref_shape)
            neg_f = self._get_averaged(neg_idx, target_shape=ref_shape)
            bit_val = (pos_f > neg_f).astype(np.int32)
            decoded_y |= bit_val << actual_bit

            bit_diff = np.abs(pos_f - neg_f)
            any_bit_contrast |= (bit_diff >= self.decode_threshold)
            if self.bit_threshold > 0:
                y_reliable &= (bit_diff >= self.bit_threshold)

            logger.info(
                "  Y bit %d (capture %d): pos=%.1f neg=%.1f diff: "
                "min=%.1f max=%.1f mean=%.1f",
                actual_bit, capture_idx, pos_f.mean(), neg_f.mean(),
                bit_diff.min(), bit_diff.max(), bit_diff.mean(),
            )

        # -- Fallback: if white/black refs failed, use per-bit contrast ------
        if n_valid_diff == 0 and np.count_nonzero(any_bit_contrast) > 0:
            n_bit_valid = int(np.count_nonzero(any_bit_contrast))
            logger.warning(
                "  WHITE/BLACK refs identical — using per-bit contrast "
                "as fallback: %d pixels (%.1f%%)",
                n_bit_valid, 100.0 * n_bit_valid / max(h * w, 1),
            )
            valid_mask = any_bit_contrast

        # -- Apply bit reliability mask --------------------------------------
        if self.bit_threshold > 0:
            n_before = int(np.count_nonzero(valid_mask))
            valid_mask = valid_mask & x_reliable & y_reliable
            n_after = int(np.count_nonzero(valid_mask))
            logger.info(
                "  After bit reliability: %d -> %d pixels (removed %d)",
                n_before, n_after, n_before - n_after,
            )

        # -- Vectorized Gray-to-binary decode --------------------------------
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

        # Build remap tables
        self._build_remap_from_correspondences(
            decoded_x.astype(np.float32),
            decoded_y.astype(np.float32),
            valid_mask, h, w,
        )

    # -- Remap building -------------------------------------------------------

    def _build_remap_from_correspondences(
        self,
        decoded_x: np.ndarray,
        decoded_y: np.ndarray,
        valid_mask: np.ndarray,
        cam_h: int,
        cam_w: int,
    ) -> None:
        """Build projector->camera remap tables from decoded correspondences.

        Takes decoded projector coordinates (float32) for each camera pixel
        and builds the dense inverse mapping (projector pixel -> camera pixel).

        Pipeline:
        1. Morphological cleanup of valid mask
        2. Spatial consistency filtering
        3. Gaussian splat fill for dense correspondence
        4. Outlier rejection (median-based)
        5. Re-fill after outlier removal
        6. Final smoothing (median + bilateral)
        7. Inpainting small holes
        """
        # -- Morphological cleanup -------------------------------------------
        if self.morph_cleanup:
            n_pre = int(np.count_nonzero(valid_mask))
            k = self.morph_kernel_size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_u8 = valid_mask.astype(np.uint8) * 255
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            valid_mask = mask_u8 > 127
            n_post = int(np.count_nonzero(valid_mask))
            logger.info("  After morph cleanup: %d -> %d pixels", n_pre, n_post)

        # -- Spatial consistency filtering -----------------------------------
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
            local_mean_x = np.where(
                has_neighbors, local_sum_x / safe_count,
                decoded_x.astype(np.float32),
            )
            local_mean_y = np.where(
                has_neighbors, local_sum_y / safe_count,
                decoded_y.astype(np.float32),
            )
            diff_x = np.abs(decoded_x.astype(np.float32) - local_mean_x)
            diff_y = np.abs(decoded_y.astype(np.float32) - local_mean_y)
            consistent = (
                (diff_x <= self.consistency_max_diff)
                & (diff_y <= self.consistency_max_diff)
            )
            n_pre_sc = int(np.count_nonzero(valid_mask))
            valid_mask = valid_mask & consistent
            n_post_sc = int(np.count_nonzero(valid_mask))
            logger.info(
                "  After spatial consistency: %d -> %d pixels",
                n_pre_sc, n_post_sc,
            )

        # -- Build inverse mapping: projector pixel -> camera pixel ----------
        sum_cx = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        sum_cy = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        count = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)

        cam_ys, cam_xs = np.where(valid_mask)
        proj_xs = np.round(decoded_x[cam_ys, cam_xs]).astype(np.int32)
        proj_ys = np.round(decoded_y[cam_ys, cam_xs]).astype(np.int32)
        proj_xs = np.clip(proj_xs, 0, self.proj_w - 1)
        proj_ys = np.clip(proj_ys, 0, self.proj_h - 1)

        logger.info(
            "  Valid camera pixels for mapping: %d/%d (%.1f%%)",
            len(cam_ys), cam_h * cam_w,
            100.0 * len(cam_ys) / max(cam_h * cam_w, 1),
        )

        np.add.at(sum_cx, (proj_ys, proj_xs), cam_xs.astype(np.float32))
        np.add.at(sum_cy, (proj_ys, proj_xs), cam_ys.astype(np.float32))
        np.add.at(count, (proj_ys, proj_xs), 1.0)

        # -- Gaussian splat fill ---------------------------------------------
        fk = max(self.fill_kernel_size, 31)
        fk = fk if fk % 2 == 1 else fk + 1
        sum_cx_s = cv2.GaussianBlur(sum_cx, (fk, fk), 0)
        sum_cy_s = cv2.GaussianBlur(sum_cy, (fk, fk), 0)
        count_s = cv2.GaussianBlur(count, (fk, fk), 0)

        valid_proj = count_s > 0.01
        map_x = np.full((self.proj_h, self.proj_w), -1.0, dtype=np.float32)
        map_y = np.full((self.proj_h, self.proj_w), -1.0, dtype=np.float32)
        map_x[valid_proj] = sum_cx_s[valid_proj] / count_s[valid_proj]
        map_y[valid_proj] = sum_cy_s[valid_proj] / count_s[valid_proj]

        n_raw = int(np.count_nonzero(valid_proj))
        logger.info(
            "  Projector pixels after splat fill: %d/%d (%.1f%%)",
            n_raw, self.proj_h * self.proj_w,
            100.0 * n_raw / max(self.proj_h * self.proj_w, 1),
        )

        # -- Outlier rejection (median-based) --------------------------------
        med_k = 5
        for pass_i in range(2):
            med_x = cv2.medianBlur(
                np.where(valid_proj, map_x, 0).astype(np.float32), med_k,
            )
            med_y = cv2.medianBlur(
                np.where(valid_proj, map_y, 0).astype(np.float32), med_k,
            )
            expected_grad = max(cam_w, cam_h) / max(self.proj_w, self.proj_h)
            outlier_thresh = max(expected_grad * med_k, 8.0)
            diff_x = np.abs(map_x - med_x)
            diff_y = np.abs(map_y - med_y)
            outliers = valid_proj & (
                (diff_x > outlier_thresh) | (diff_y > outlier_thresh)
            )
            n_outliers = int(outliers.sum())
            if n_outliers > 0:
                valid_proj[outliers] = False
                map_x[outliers] = -1.0
                map_y[outliers] = -1.0
                logger.info(
                    "  Outlier pass %d: removed %d pixels (thresh=%.1f)",
                    pass_i + 1, n_outliers, outlier_thresh,
                )
            else:
                break

        # -- Re-fill after outlier removal -----------------------------------
        clean_valid = valid_proj & (map_x >= 0) & (map_y >= 0)
        sum_cx2 = np.zeros_like(sum_cx)
        sum_cy2 = np.zeros_like(sum_cy)
        count2 = np.zeros_like(count)
        sum_cx2[clean_valid] = map_x[clean_valid]
        sum_cy2[clean_valid] = map_y[clean_valid]
        count2[clean_valid] = 1.0
        sum_cx2 = cv2.GaussianBlur(sum_cx2, (fk, fk), 0)
        sum_cy2 = cv2.GaussianBlur(sum_cy2, (fk, fk), 0)
        count2 = cv2.GaussianBlur(count2, (fk, fk), 0)
        valid_proj2 = count2 > 0.01
        map_x = np.full_like(map_x, -1.0)
        map_y = np.full_like(map_y, -1.0)
        map_x[valid_proj2] = sum_cx2[valid_proj2] / count2[valid_proj2]
        map_y[valid_proj2] = sum_cy2[valid_proj2] / count2[valid_proj2]
        valid_proj = valid_proj2

        # -- Final smoothing: median + bilateral ----------------------------
        map_x_f = np.where(valid_proj, map_x, 0).astype(np.float32)
        map_y_f = np.where(valid_proj, map_y, 0).astype(np.float32)
        map_x_f = cv2.medianBlur(map_x_f, 5)
        map_y_f = cv2.medianBlur(map_y_f, 5)
        map_x_f = cv2.bilateralFilter(map_x_f, 9, 20.0, 20.0)
        map_y_f = cv2.bilateralFilter(map_y_f, 9, 20.0, 20.0)
        map_x[valid_proj] = map_x_f[valid_proj]
        map_y[valid_proj] = map_y_f[valid_proj]

        # Store validity mask BEFORE inpainting
        self.proj_valid_mask = valid_proj.copy()
        n_valid_proj = int(np.count_nonzero(valid_proj))
        total_proj = self.proj_h * self.proj_w
        logger.info(
            "  Final projector coverage: %d/%d (%.1f%%)",
            n_valid_proj, total_proj, 100.0 * n_valid_proj / max(total_proj, 1),
        )

        # -- Gradient quality check (diagnostic) ----------------------------
        gx = np.abs(np.diff(map_x, axis=1))
        gy = np.abs(np.diff(map_y, axis=0))
        gx_v = gx[valid_proj[:, :-1]]
        gy_v = gy[valid_proj[:-1, :]]
        if gx_v.size > 0:
            logger.info(
                "  Map smoothness: grad_x mean=%.2f std=%.2f p99=%.1f, "
                "grad_y mean=%.2f std=%.2f p99=%.1f",
                gx_v.mean(), gx_v.std(), np.percentile(gx_v, 99),
                gy_v.mean(), gy_v.std(), np.percentile(gy_v, 99),
            )

        # -- Inpaint small internal holes ------------------------------------
        hole_mask = (~valid_proj).astype(np.uint8) * 255
        if np.any(~valid_proj):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                hole_mask, connectivity=8,
            )
            max_small_area = max(int(self.proj_w * self.proj_h * 0.01), 200)
            small_hole_mask = np.zeros_like(hole_mask)
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < max_small_area:
                    small_hole_mask[labels == label_id] = 255
            if np.any(small_hole_mask):
                inpaint_r = max(fk // 2, 7)
                map_x = cv2.inpaint(
                    map_x, small_hole_mask, inpaint_r, cv2.INPAINT_NS,
                )
                map_y = cv2.inpaint(
                    map_y, small_hole_mask, inpaint_r, cv2.INPAINT_NS,
                )

        self.map_x = map_x
        self.map_y = map_y


# -- Calibration file I/O ----------------------------------------------------


def save_calibration(
    map_x: np.ndarray,
    map_y: np.ndarray,
    path: str | Path,
    proj_w: int,
    proj_h: int,
) -> None:
    """Save calibration mapping.

    Uses ``.npz`` for fast I/O.  Also writes a small JSON sidecar for
    metadata and backwards compatibility.
    """
    p = Path(path)

    npz_path = p.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        map_x=map_x,
        map_y=map_y,
        meta=np.array([proj_w, proj_h]),
    )

    meta = {
        "version": 2,
        "projector_width": proj_w,
        "projector_height": proj_h,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "npz_file": npz_path.name,
    }
    p.write_text(json.dumps(meta))
    logger.info("Saved calibration: %s + %s", p.name, npz_path.name)


def load_calibration(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    """Load calibration mapping from JSON+NPZ or legacy JSON.

    Returns (map_x, map_y, proj_w, proj_h, timestamp_iso).
    """
    p = Path(path)
    data = json.loads(p.read_text())
    timestamp = data.get("timestamp", "unknown")

    # New format (v2): metadata JSON + NPZ binary
    if data.get("version", 1) >= 2:
        npz_path = p.parent / data["npz_file"]
        npz = np.load(npz_path)
        return (
            npz["map_x"].astype(np.float32),
            npz["map_y"].astype(np.float32),
            int(data["projector_width"]),
            int(data["projector_height"]),
            timestamp,
        )

    # Legacy format (v1): arrays embedded in JSON
    map_x = np.array(data["map_x"], dtype=np.float32)
    map_y = np.array(data["map_y"], dtype=np.float32)
    return map_x, map_y, data["projector_width"], data["projector_height"], timestamp
