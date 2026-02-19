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
- Change-detection settle (captures as soon as camera shows new pattern)
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
# Ensure our logger output reaches stdout (Scope doesn't forward plugin loggers)
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
        settle_frames: int = 30,
        capture_frames: int = 3,
        decode_threshold: float = 20.0,
        bit_threshold: float = 3.0,
        morph_cleanup: bool = True,
        morph_kernel_size: int = 5,
        spatial_consistency: bool = True,
        consistency_max_diff: float = 3.0,
        fill_kernel_size: int = 11,
        max_brightness: int = 255,
        min_settle: int = 8,
        min_settle_time: float = 2.0,
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
        # Change-detection settle thresholds (mean abs pixel diff, uint8 scale)
        # change_threshold: baseline→current diff to detect new pattern arrival
        #   Gray codes cause 20-100+ diff, noise is typically < 3
        self.change_threshold = change_threshold
        # stability_threshold: frame-to-frame diff to confirm pattern is stable
        #   Sensor noise is typically 0.5-2.0, so 3.0 is robust
        self.stability_threshold = stability_threshold

        self.bits_x = _num_bits(proj_w)
        self.bits_y = _num_bits(proj_h)

        # Total patterns: white + black + 2*(bits_x + bits_y) pos/neg pairs
        self.total_patterns = 2 + 2 * (self.bits_x + self.bits_y)

        self.phase = CalibrationPhase.IDLE
        self._pattern_index = 0
        self._settle_counter = 0
        self._waiting_for_settle = False
        self._frame_count = 0  # frames captured for current pattern

        # Change-detection settle state
        self._settle_baseline: np.ndarray | None = None  # camera at settle start
        self._prev_settle_frame: np.ndarray | None = None  # for stability check
        self._stable_count: int = 0
        self._change_detected: bool = False

        # Cached pattern tensor (avoids regenerating 1920x1080x3 every frame)
        self._cached_pattern: torch.Tensor | None = None
        self._cached_pattern_key: tuple | None = None  # (phase, pattern_index)

        # Each slot holds a list of float32 frames (for multi-frame averaging)
        self._captures: list[list[np.ndarray]] = [[] for _ in range(self.total_patterns)]

        # Result mapping arrays
        self.map_x: np.ndarray | None = None
        self.map_y: np.ndarray | None = None
        self.proj_valid_mask: np.ndarray | None = None  # bool, valid before inpainting

    def start(self) -> None:
        self.phase = CalibrationPhase.WHITE
        self._pattern_index = 0
        self._frame_count = 0
        self._captures = [[] for _ in range(self.total_patterns)]
        self.map_x = None
        self.map_y = None
        self.proj_valid_mask = None
        self._begin_settle()

    def _begin_settle(self) -> None:
        """Reset change-detection settle state for the next pattern.

        The settle mechanism watches for the camera to actually display the
        new pattern (structural diff from baseline), then waits for 1 frame
        of stability before capturing.  Stale repeated frames (pipeline
        faster than camera) naturally have zero diff and are ignored.
        """
        self._waiting_for_settle = True
        self._settle_counter = 0
        self._stable_count = 0
        self._change_detected = False
        self._settle_baseline = None
        self._prev_settle_frame = None
        self._settle_start_time = _time.monotonic()

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
        # Run decode on a separate frame so the pipeline can push progress
        # updates ("Decoding...") to the dashboard before we block.
        if self.phase == CalibrationPhase.DECODING:
            logger.info("Running Gray code decode ...")
            self._decode()
            self.phase = CalibrationPhase.DONE
            return None

        if self.phase in (CalibrationPhase.IDLE, CalibrationPhase.DONE):
            return None

        # Change-detection settle: wait for the camera to actually show the
        # new pattern, then capture as soon as it stabilises.
        #
        # Phase 1 — WAITING FOR CHANGE:
        #   Compare each frame to baseline (captured at settle start, still
        #   showing the old pattern).  Stale repeated frames have zero diff
        #   and are naturally ignored, so this works regardless of how fast
        #   the pipeline runs relative to the camera.
        #
        # Phase 2 — WAITING FOR STABILITY:
        #   Once the structural diff exceeds the change threshold (pattern
        #   arrived), wait for 1 frame where the frame-to-frame diff is low
        #   (projector has fully displayed the new pattern).
        #
        # Timeout: if no change after settle_frames, capture anyway (safety).
        if self._waiting_for_settle:
            self._settle_counter += 1
            gray = self._camera_to_gray(camera_frame)
            gray_f = gray.astype(np.float32)
            elapsed = _time.monotonic() - self._settle_start_time

            # Capture baseline on first frame (camera still shows old pattern)
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
                    # Pattern arrived — move to stability check
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
                    bl_mean = float(self._settle_baseline.mean())
                    logger.warning(
                        "Settle timeout: %d frames / %.1fs — no change "
                        "detected (diff=%.1f, baseline=%.1f, current=%.1f)",
                        self._settle_counter, elapsed, structural_diff,
                        bl_mean, float(gray_f.mean()),
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
                    # 2 consecutive stable frames — pattern fully displayed
                    logger.info(
                        "Settle done: %d frames / %.2fs (change + %d stable)",
                        self._settle_counter, elapsed, self._stable_count,
                    )
                    self._waiting_for_settle = False
                    self._frame_count = 0
                else:
                    return self._current_pattern(device)

        # Capture frame (accumulate for multi-frame averaging)
        gray = self._camera_to_gray(camera_frame)
        cap_idx = self._current_capture_index()
        if cap_idx >= len(self._captures):
            # Safety: pattern_index overshot (should not happen with DECODING guard)
            logger.error(
                "Capture index %d out of range (%d slots), phase=%s idx=%d — skipping",
                cap_idx, len(self._captures), self.phase.name, self._pattern_index,
            )
            return None
        self._captures[cap_idx].append(gray.astype(np.float32))
        self._frame_count += 1

        # Debug: log first capture stats for white/black references
        if cap_idx <= 1 and self._frame_count == 1:
            label = "WHITE" if cap_idx == 0 else "BLACK"
            logger.info(
                "Captured %s reference: shape=%s min=%.1f max=%.1f mean=%.1f",
                label, gray.shape, gray.min(), gray.max(), gray.mean(),
            )

        # Check if we have enough frames for this pattern
        if self._frame_count < self.capture_frames:
            # Need more frames — keep projecting the same pattern (no settle)
            return self._current_pattern(device)

        # Done with this pattern, advance to next
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
            if self._pattern_index >= 2 * (self.bits_x + self.bits_y):
                # Enter DECODING phase — actual decode runs on the NEXT
                # step() call so the pipeline can push a progress update first.
                self.phase = CalibrationPhase.DECODING
                return None
            self._begin_settle()
            return self._current_pattern(device)

        return None

    def _current_pattern(self, device: torch.device) -> torch.Tensor:
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

        tensor = self._pattern_to_tensor(pattern, device)
        self._cached_pattern = tensor
        self._cached_pattern_key = key
        return tensor

    def _get_averaged(
        self, index: int, target_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return the mean of accumulated frames for a capture index (float32).

        If *target_shape* ``(h, w)`` is given, any frames that don't match
        are resized first.  This guards against Scope changing the camera
        resolution mid-calibration.
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
        ref_shape = white_f.shape[:2]  # canonical resolution
        black_f = self._get_averaged(1, target_shape=ref_shape)

        h, w = ref_shape
        diff = white_f - black_f

        logger.info(
            "Decode: camera resolution %dx%d, projector %dx%d",
            w, h, self.proj_w, self.proj_h,
        )
        logger.info(
            "  WHITE ref: min=%.1f max=%.1f mean=%.1f",
            white_f.min(), white_f.max(), white_f.mean(),
        )
        logger.info(
            "  BLACK ref: min=%.1f max=%.1f mean=%.1f",
            black_f.min(), black_f.max(), black_f.mean(),
        )
        logger.info(
            "  DIFF (white-black): min=%.1f max=%.1f mean=%.1f, threshold=%.1f",
            diff.min(), diff.max(), diff.mean(), self.decode_threshold,
        )

        valid_mask = diff > self.decode_threshold
        n_valid_diff = int(np.count_nonzero(valid_mask))
        logger.info(
            "  Pixels passing white-black threshold: %d/%d (%.1f%%)",
            n_valid_diff, h * w, 100.0 * n_valid_diff / max(h * w, 1),
        )

        # Track per-bit contrast for fallback valid mask
        # (handles case where WHITE/BLACK refs are bad but patterns are good)
        any_bit_contrast = np.zeros((h, w), dtype=bool)

        # Decode X (column) Gray codes with per-bit reliability
        base = 2  # skip white + black
        decoded_x = np.zeros((h, w), dtype=np.int32)
        x_reliable = np.ones((h, w), dtype=bool)
        for bit_idx in range(self.bits_x):
            pos_idx = base + bit_idx * 2
            neg_idx = base + bit_idx * 2 + 1
            if self._captures[pos_idx] and self._captures[neg_idx]:
                pos_f = self._get_averaged(pos_idx, target_shape=ref_shape)
                neg_f = self._get_averaged(neg_idx, target_shape=ref_shape)
                bit_val = (pos_f > neg_f).astype(np.int32)
                decoded_x |= bit_val << bit_idx
                bit_diff = np.abs(pos_f - neg_f)
                any_bit_contrast |= (bit_diff >= self.decode_threshold)
                if self.bit_threshold > 0:
                    x_reliable &= (bit_diff >= self.bit_threshold)
                # Log first 3 bits for debugging
                if bit_idx < 3:
                    logger.info(
                        "  X bit %d: pos_mean=%.1f neg_mean=%.1f "
                        "abs_diff: min=%.1f max=%.1f mean=%.1f",
                        bit_idx, pos_f.mean(), neg_f.mean(),
                        bit_diff.min(), bit_diff.max(), bit_diff.mean(),
                    )
            else:
                logger.warning("  X bit %d: missing captures!", bit_idx)

        # Decode Y (row) Gray codes with per-bit reliability
        base_y = base + self.bits_x * 2
        decoded_y = np.zeros((h, w), dtype=np.int32)
        y_reliable = np.ones((h, w), dtype=bool)
        for bit_idx in range(self.bits_y):
            pos_idx = base_y + bit_idx * 2
            neg_idx = base_y + bit_idx * 2 + 1
            if self._captures[pos_idx] and self._captures[neg_idx]:
                pos_f = self._get_averaged(pos_idx, target_shape=ref_shape)
                neg_f = self._get_averaged(neg_idx, target_shape=ref_shape)
                bit_val = (pos_f > neg_f).astype(np.int32)
                decoded_y |= bit_val << bit_idx
                bit_diff = np.abs(pos_f - neg_f)
                any_bit_contrast |= (bit_diff >= self.decode_threshold)
                if self.bit_threshold > 0:
                    y_reliable &= (bit_diff >= self.bit_threshold)
                if bit_idx < 3:
                    logger.info(
                        "  Y bit %d: pos_mean=%.1f neg_mean=%.1f "
                        "abs_diff: min=%.1f max=%.1f mean=%.1f",
                        bit_idx, pos_f.mean(), neg_f.mean(),
                        bit_diff.min(), bit_diff.max(), bit_diff.mean(),
                    )
            else:
                logger.warning("  Y bit %d: missing captures!", bit_idx)

        # Fallback: if WHITE/BLACK refs failed but per-bit contrast is strong,
        # use per-bit contrast as the valid mask.  This handles cases where
        # WHITE and BLACK captured the same stale frame (e.g. pipeline ran
        # faster than camera during reference captures).
        if n_valid_diff == 0 and np.count_nonzero(any_bit_contrast) > 0:
            n_bit_valid = int(np.count_nonzero(any_bit_contrast))
            logger.warning(
                "  WHITE/BLACK refs identical — using per-bit contrast "
                "as fallback valid mask: %d pixels (%.1f%%)",
                n_bit_valid, 100.0 * n_bit_valid / max(h * w, 1),
            )
            valid_mask = any_bit_contrast

        # Apply bit reliability mask
        if self.bit_threshold > 0:
            n_before = int(np.count_nonzero(valid_mask))
            valid_mask = valid_mask & x_reliable & y_reliable
            n_after = int(np.count_nonzero(valid_mask))
            logger.info(
                "  After bit reliability filter: %d → %d pixels (removed %d)",
                n_before, n_after, n_before - n_after,
            )

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
            n_pre = int(np.count_nonzero(valid_mask))
            k = self.morph_kernel_size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_u8 = valid_mask.astype(np.uint8) * 255
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            valid_mask = mask_u8 > 127
            n_post = int(np.count_nonzero(valid_mask))
            logger.info("  After morph cleanup: %d → %d pixels", n_pre, n_post)

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
            n_pre_sc = int(np.count_nonzero(valid_mask))
            valid_mask = valid_mask & consistent
            n_post_sc = int(np.count_nonzero(valid_mask))
            logger.info(
                "  After spatial consistency: %d → %d pixels", n_pre_sc, n_post_sc,
            )

        # -- Build inverse mapping: projector pixel -> camera pixel --
        # Accumulate weighted sums for Gaussian splat
        sum_cx = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        sum_cy = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        count = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)

        cam_ys, cam_xs = np.where(valid_mask)
        proj_xs = decoded_x[cam_ys, cam_xs]
        proj_ys = decoded_y[cam_ys, cam_xs]
        logger.info(
            "  Valid camera pixels for mapping: %d/%d (%.1f%%)",
            len(cam_ys), h * w, 100.0 * len(cam_ys) / max(h * w, 1),
        )

        np.add.at(sum_cx, (proj_ys, proj_xs), cam_xs.astype(np.float32))
        np.add.at(sum_cy, (proj_ys, proj_xs), cam_ys.astype(np.float32))
        np.add.at(count, (proj_ys, proj_xs), 1.0)

        # Gaussian splat: spread correspondences to fill gaps.
        # Use a large kernel — coverage can be sparse (15-40%) and we need
        # to bridge gaps between valid regions.
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
            "  Raw projector pixels after splat fill: %d/%d (%.1f%%)",
            n_raw, self.proj_h * self.proj_w,
            100.0 * n_raw / max(self.proj_h * self.proj_w, 1),
        )

        # -- Outlier rejection: median-based smoothness filter --
        # A valid mapping should be locally smooth (neighboring projector
        # pixels map to nearby camera pixels).  Reject pixels where the
        # map value deviates from the local median by more than a threshold.
        # NOTE: cv2.medianBlur only supports float32 for ksize <= 5.
        med_k = 5
        for pass_i in range(2):
            med_x = cv2.medianBlur(
                np.where(valid_proj, map_x, 0).astype(np.float32), med_k,
            )
            med_y = cv2.medianBlur(
                np.where(valid_proj, map_y, 0).astype(np.float32), med_k,
            )
            # Expected gradient: camera_res / projector_res ≈ pixels per step
            expected_grad = max(w, h) / max(self.proj_w, self.proj_h)
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

        # -- Re-fill after outlier removal with Gaussian splat --
        # Rebuild from cleaned valid pixels only
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

        # -- Final smoothing: median + bilateral filter --
        # Median removes remaining salt-and-pepper noise
        map_x_f = np.where(valid_proj, map_x, 0).astype(np.float32)
        map_y_f = np.where(valid_proj, map_y, 0).astype(np.float32)
        map_x_f = cv2.medianBlur(map_x_f, 5)
        map_y_f = cv2.medianBlur(map_y_f, 5)
        # Bilateral preserves edges while smoothing — sigma tuned for pixel coords
        map_x_f = cv2.bilateralFilter(map_x_f, 9, 20.0, 20.0)
        map_y_f = cv2.bilateralFilter(map_y_f, 9, 20.0, 20.0)
        map_x[valid_proj] = map_x_f[valid_proj]
        map_y[valid_proj] = map_y_f[valid_proj]

        # Store validity mask BEFORE inpainting
        self.proj_valid_mask = valid_proj.copy()
        n_valid_proj = int(np.count_nonzero(valid_proj))
        total_proj = self.proj_h * self.proj_w
        logger.info(
            "  Final projector pixels with correspondence: %d/%d (%.1f%%)",
            n_valid_proj, total_proj, 100.0 * n_valid_proj / max(total_proj, 1),
        )

        # -- Gradient quality check (diagnostic) --
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

        # Inpaint small internal holes (not large uncovered boundary regions)
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
    """Save calibration mapping.

    Uses ``.npz`` for fast I/O.  Also writes a small JSON sidecar for
    metadata and backwards compatibility with tools that read the old format.
    """
    p = Path(path)

    # Fast binary save — instant even for 1920×1080
    npz_path = p.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        map_x=map_x,
        map_y=map_y,
        meta=np.array([proj_w, proj_h]),
    )

    # Small JSON sidecar (metadata only, no huge arrays)
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
