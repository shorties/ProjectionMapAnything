"""Gray code structured light calibration for projector-camera systems.

Projects binary Gray code stripe patterns, captures them with a camera,
and decodes the per-pixel correspondence to build a dense mapping from
projector pixels to camera pixels (map_x, map_y) usable with ``cv2.remap``.

The decode pipeline (simplified, matching the HTML scanner approach):
1. Multi-frame averaged captures for noise rejection
2. Per-bit comparison (pos > neg) with confidence accumulation
3. Vectorized Gray-to-binary decode
4. Confidence-weighted accumulation into projector-space maps
5. Light median filter (3x3) for noise
6. Inpainting remaining holes
7. Disparity map computed as cam_x - proj_x
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
        max_brightness: int = 255,
        change_threshold: float = 5.0,
        stability_threshold: float = 3.0,
    ):
        self.proj_w = proj_w
        self.proj_h = proj_h
        self.settle_frames = settle_frames  # timeout fallback only
        self.capture_frames = max(1, capture_frames)
        self.max_brightness = max(10, min(255, max_brightness))
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
        self.disparity_map: np.ndarray | None = None

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

        Simplified pipeline matching the HTML scanner approach:
        1. Decode each bit: compare positive vs inverted pattern
        2. Accumulate per-axis confidence: sum of |pos - neg|
        3. Vectorized Gray-to-binary conversion
        4. Filter by confidence threshold
        5. Confidence-weighted accumulation into projector-space maps
        6. Light median filter (3x3) for noise
        7. Inpaint remaining holes
        8. Compute disparity map (cam_x - proj_x)
        """
        white_f = self._get_averaged(0)
        ref_shape = white_f.shape[:2]
        black_f = self._get_averaged(1, target_shape=ref_shape)

        h, w = ref_shape

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

        # -- Decode X (column) Gray codes with confidence --------------------
        base = 2  # skip white + black
        decoded_x = np.zeros((h, w), dtype=np.int32)
        conf_x = np.zeros((h, w), dtype=np.float32)

        for capture_idx in range(self.bits_x):
            actual_bit = (self.bits_x - 1) - capture_idx  # MSB first
            pos_idx = base + capture_idx * 2
            neg_idx = base + capture_idx * 2 + 1

            if not self._captures[pos_idx] or not self._captures[neg_idx]:
                logger.warning(
                    "  X bit %d (capture %d): missing captures!",
                    actual_bit, capture_idx,
                )
                continue

            pos_f = self._get_averaged(pos_idx, target_shape=ref_shape)
            neg_f = self._get_averaged(neg_idx, target_shape=ref_shape)
            bit_val = (pos_f > neg_f).astype(np.int32)
            decoded_x |= bit_val << actual_bit
            conf_x += np.abs(pos_f - neg_f)

            logger.info(
                "  X bit %d (capture %d): pos=%.1f neg=%.1f "
                "diff: min=%.1f max=%.1f mean=%.1f",
                actual_bit, capture_idx, pos_f.mean(), neg_f.mean(),
                np.abs(pos_f - neg_f).min(),
                np.abs(pos_f - neg_f).max(),
                np.abs(pos_f - neg_f).mean(),
            )

        # -- Decode Y (row) Gray codes with confidence ----------------------
        base_y = base + self.bits_x * 2
        decoded_y = np.zeros((h, w), dtype=np.int32)
        conf_y = np.zeros((h, w), dtype=np.float32)

        for capture_idx in range(self.bits_y):
            actual_bit = (self.bits_y - 1) - capture_idx  # MSB first
            pos_idx = base_y + capture_idx * 2
            neg_idx = base_y + capture_idx * 2 + 1

            if not self._captures[pos_idx] or not self._captures[neg_idx]:
                logger.warning(
                    "  Y bit %d (capture %d): missing captures!",
                    actual_bit, capture_idx,
                )
                continue

            pos_f = self._get_averaged(pos_idx, target_shape=ref_shape)
            neg_f = self._get_averaged(neg_idx, target_shape=ref_shape)
            bit_val = (pos_f > neg_f).astype(np.int32)
            decoded_y |= bit_val << actual_bit
            conf_y += np.abs(pos_f - neg_f)

            logger.info(
                "  Y bit %d (capture %d): pos=%.1f neg=%.1f "
                "diff: min=%.1f max=%.1f mean=%.1f",
                actual_bit, capture_idx, pos_f.mean(), neg_f.mean(),
                np.abs(pos_f - neg_f).min(),
                np.abs(pos_f - neg_f).max(),
                np.abs(pos_f - neg_f).mean(),
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

        # -- Filter by confidence + bounds -----------------------------------
        conf_thresh_x = 15.0 * self.bits_x
        conf_thresh_y = 15.0 * self.bits_y
        valid_mask = (
            (conf_x > conf_thresh_x)
            & (conf_y > conf_thresh_y)
            & (decoded_x < self.proj_w)
            & (decoded_y < self.proj_h)
        )
        n_valid = int(np.count_nonzero(valid_mask))
        logger.info(
            "  Valid pixels after confidence filter: %d/%d (%.1f%%) "
            "[thresh_x=%.0f, thresh_y=%.0f]",
            n_valid, h * w, 100.0 * n_valid / max(h * w, 1),
            conf_thresh_x, conf_thresh_y,
        )

        # -- Weighted accumulation into projector-space maps -----------------
        cam_ys, cam_xs = np.where(valid_mask)
        proj_xs = decoded_x[cam_ys, cam_xs].astype(np.int32)
        proj_ys = decoded_y[cam_ys, cam_xs].astype(np.int32)
        weights = (conf_x[cam_ys, cam_xs] + conf_y[cam_ys, cam_xs]).astype(np.float32)

        sum_cx = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        sum_cy = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        sum_w = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)

        np.add.at(sum_cx, (proj_ys, proj_xs), cam_xs.astype(np.float32) * weights)
        np.add.at(sum_cy, (proj_ys, proj_xs), cam_ys.astype(np.float32) * weights)
        np.add.at(sum_w, (proj_ys, proj_xs), weights)

        valid_proj = sum_w > 0
        map_x = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        map_y = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        map_x[valid_proj] = sum_cx[valid_proj] / sum_w[valid_proj]
        map_y[valid_proj] = sum_cy[valid_proj] / sum_w[valid_proj]

        n_filled = int(np.count_nonzero(valid_proj))
        total_proj = self.proj_h * self.proj_w
        logger.info(
            "  Projector pixels with data: %d/%d (%.1f%%)",
            n_filled, total_proj, 100.0 * n_filled / max(total_proj, 1),
        )

        # Store validity mask BEFORE smoothing/inpainting
        self.proj_valid_mask = valid_proj.copy()

        # -- Light median filter (3x3) for noise ----------------------------
        map_x_f = np.where(valid_proj, map_x, 0).astype(np.float32)
        map_y_f = np.where(valid_proj, map_y, 0).astype(np.float32)
        map_x_f = cv2.medianBlur(map_x_f, 3)
        map_y_f = cv2.medianBlur(map_y_f, 3)
        map_x[valid_proj] = map_x_f[valid_proj]
        map_y[valid_proj] = map_y_f[valid_proj]

        # -- Inpaint remaining holes ----------------------------------------
        hole_mask = (~valid_proj).astype(np.uint8) * 255
        if np.any(~valid_proj):
            map_x = cv2.inpaint(map_x, hole_mask, 7, cv2.INPAINT_NS)
            map_y = cv2.inpaint(map_y, hole_mask, 7, cv2.INPAINT_NS)

        logger.info(
            "  Final projector coverage: %d/%d (%.1f%%)",
            n_filled, total_proj, 100.0 * n_filled / max(total_proj, 1),
        )

        self.map_x = map_x
        self.map_y = map_y

        # -- Compute disparity map (cam_x - proj_x) -------------------------
        # For each projector pixel, horizontal disparity = camera_x - projector_x
        px = np.arange(self.proj_w, dtype=np.float32)[None, :].repeat(
            self.proj_h, axis=0,
        )
        disparity = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        disparity[valid_proj] = map_x[valid_proj] - px[valid_proj]

        # Normalize to [0, 1] (near=bright, far=dark)
        valid_vals = disparity[valid_proj]
        if valid_vals.size > 0:
            p2 = float(np.percentile(valid_vals, 2))
            p98 = float(np.percentile(valid_vals, 98))
            if p98 - p2 > 1e-6:
                disparity = (disparity - p2) / (p98 - p2)
            else:
                disparity = np.full_like(disparity, 0.5)
        disparity = np.clip(disparity, 0.0, 1.0).astype(np.float32)

        # Inpaint holes in disparity too
        if np.any(~valid_proj):
            disp_hole = (~valid_proj).astype(np.uint8) * 255
            disparity = cv2.inpaint(disparity, disp_hole, 7, cv2.INPAINT_NS)

        self.disparity_map = disparity
        logger.info(
            "  Disparity map computed: range [%.3f, %.3f]",
            disparity.min(), disparity.max(),
        )


# -- Calibration file I/O ----------------------------------------------------


def save_calibration(
    map_x: np.ndarray,
    map_y: np.ndarray,
    path: str | Path,
    proj_w: int,
    proj_h: int,
    disparity_map: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> None:
    """Save calibration mapping.

    Uses ``.npz`` for fast I/O.  Also writes a small JSON sidecar for
    metadata and backwards compatibility.
    """
    p = Path(path)

    npz_path = p.with_suffix(".npz")
    arrays = {
        "map_x": map_x,
        "map_y": map_y,
        "meta": np.array([proj_w, proj_h]),
    }
    if disparity_map is not None:
        arrays["disparity"] = disparity_map
    if valid_mask is not None:
        arrays["valid_mask"] = valid_mask.astype(np.uint8)
    np.savez_compressed(npz_path, **arrays)

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
) -> tuple[np.ndarray, np.ndarray, int, int, str, np.ndarray | None]:
    """Load calibration mapping from JSON+NPZ or legacy JSON.

    Returns (map_x, map_y, proj_w, proj_h, timestamp_iso, disparity_map).
    disparity_map may be None for older calibrations.
    """
    p = Path(path)
    data = json.loads(p.read_text())
    timestamp = data.get("timestamp", "unknown")

    # New format (v2): metadata JSON + NPZ binary
    if data.get("version", 1) >= 2:
        npz_path = p.parent / data["npz_file"]
        npz = np.load(npz_path)
        disparity = (
            npz["disparity"].astype(np.float32)
            if "disparity" in npz.files
            else None
        )
        return (
            npz["map_x"].astype(np.float32),
            npz["map_y"].astype(np.float32),
            int(data["projector_width"]),
            int(data["projector_height"]),
            timestamp,
            disparity,
        )

    # Legacy format (v1): arrays embedded in JSON
    map_x = np.array(data["map_x"], dtype=np.float32)
    map_y = np.array(data["map_y"], dtype=np.float32)
    return (
        map_x, map_y,
        data["projector_width"], data["projector_height"],
        timestamp, None,
    )
