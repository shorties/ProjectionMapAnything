"""Gray code structured light calibration for projector-camera systems.

Projects binary Gray code stripe patterns, captures them with a camera,
and decodes the per-pixel correspondence to build a dense mapping from
projector pixels to camera pixels (map_x, map_y) usable with ``cv2.remap``.

The decode pipeline (RoomAlive-informed):
1. Multi-frame averaged captures for noise rejection
2. Per-bit comparison (pos > neg) with per-bit reliability threshold
3. MSB-only validity gating (LSBs excluded from mask, per RoomAlive)
4. Vectorized Gray-to-binary decode
5. Explicit boundary validation (decoded values in projector range)
6. Confidence-weighted accumulation into projector-space maps
7. Light median filter (3x3) for noise
8. Inpainting remaining holes
9. Disparity map computed as cam_x - proj_x
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
        settle_timeout_sec: float = 15.0,
        capture_frames: int = 3,
        max_brightness: int = 255,
        change_threshold: float = 5.0,
        stability_threshold: float = 3.0,
        bit_threshold: float = 5.0,
        shadow_threshold: float = 40.0,
    ):
        self.proj_w = proj_w
        self.proj_h = proj_h
        self.settle_frames = settle_frames  # frame-count fallback
        self.settle_timeout_sec = settle_timeout_sec  # primary time-based timeout
        self.capture_frames = max(1, capture_frames)
        self.max_brightness = max(10, min(255, max_brightness))
        self.change_threshold = change_threshold
        self.stability_threshold = stability_threshold
        self.bit_threshold = bit_threshold
        self.shadow_threshold = shadow_threshold

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

        # Camera resolution (set during decode)
        self.cam_w: int | None = None
        self.cam_h: int | None = None

        logger.info(
            "CalibrationState: %dx%d, %d patterns (%d bits_x, %d bits_y), "
            "settle_timeout=%.0fs (frame_fallback=%d), capture=%d, "
            "bit_thresh=%.1f, shadow_thresh=%.1f",
            proj_w, proj_h, self.total_patterns,
            self.bits_x, self.bits_y,
            settle_timeout_sec, settle_frames, capture_frames,
            bit_threshold, shadow_threshold,
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

    def get_current_pattern_params(self) -> dict | None:
        """Return lightweight params describing the current pattern for client-side rendering.

        The projector page uses these to render Gray code patterns directly
        on a ``<canvas>`` at native pixel resolution — zero JPEG compression
        or CSS scaling artifacts.

        Includes ``proj_w`` and ``proj_h`` so the canvas renders at exactly
        the resolution the calibration expects (avoids mismatch with
        client-reported screen resolution).

        Returns ``None`` when no pattern should be displayed (IDLE, DECODING, DONE).
        """
        if self.phase == CalibrationPhase.WHITE:
            return {
                "type": "white", "brightness": self.max_brightness,
                "proj_w": self.proj_w, "proj_h": self.proj_h,
            }
        if self.phase == CalibrationPhase.BLACK:
            return {
                "type": "black",
                "proj_w": self.proj_w, "proj_h": self.proj_h,
            }
        if self.phase == CalibrationPhase.PATTERNS:
            total_x = 2 * self.bits_x
            idx = self._pattern_index
            if idx < total_x:
                capture_bit = idx // 2
                bit = (self.bits_x - 1) - capture_bit
                inverted = idx % 2 == 1
                return {
                    "type": "graycode", "axis": 0, "bit": bit,
                    "inverted": inverted, "brightness": self.max_brightness,
                    "proj_w": self.proj_w, "proj_h": self.proj_h,
                }
            else:
                y_idx = idx - total_x
                capture_bit = y_idx // 2
                bit = (self.bits_y - 1) - capture_bit
                inverted = y_idx % 2 == 1
                return {
                    "type": "graycode", "axis": 1, "bit": bit,
                    "inverted": inverted, "brightness": self.max_brightness,
                    "proj_w": self.proj_w, "proj_h": self.proj_h,
                }
        return None

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

                # Timeout: time-based primary, frame-count fallback.
                # CRITICAL: on timeout, SKIP the pattern — do NOT capture
                # garbage.  On high-latency setups (RunPod MJPEG) the
                # camera may still show the PREVIOUS pattern, so capturing
                # would poison the decode.
                timed_out = (
                    elapsed >= self.settle_timeout_sec
                    or self._settle_counter >= self.settle_frames
                )
                if timed_out:
                    logger.warning(
                        "Settle timeout: %d frames / %.1fs — no change "
                        "detected (diff=%.1f). SKIPPING pattern — camera "
                        "may not see projector.",
                        self._settle_counter, elapsed, structural_diff,
                    )
                    self._waiting_for_settle = False
                    # Skip this pattern entirely — advance without capturing
                    return self._advance_to_next_pattern(device)

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
        return self._advance_to_next_pattern(device)

    # -- Internal helpers -----------------------------------------------------

    def _advance_to_next_pattern(
        self, device: torch.device,
    ) -> torch.Tensor | None:
        """Advance the calibration to the next pattern phase.

        Handles WHITE → BLACK → PATTERNS → DECODING transitions.
        Returns the next pattern tensor, or None if entering DECODING.
        """
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

        Pipeline (informed by OpenCV structured_light + RoomAlive):
        1. Shadow mask from white-black reference (OpenCV blackThreshold)
        2. Decode each bit: compare positive vs inverted pattern
        3. Per-bit reliability threshold (OpenCV whiteThreshold)
        4. Only top MSB bits gate validity (RoomAlive: LSBs too noisy)
        5. Vectorized Gray-to-binary conversion
        6. Explicit boundary validation (decoded values in projector range)
        7. Confidence-weighted accumulation into projector-space maps
        8. Light median filter (3x3) for noise
        9. Inpaint remaining holes
        10. Compute disparity map (cam_x - proj_x)
        """
        # Guard: white and black references are required for shadow mask.
        # If either was skipped (settle timeout), calibration cannot proceed.
        if not self._captures[0] or not self._captures[1]:
            logger.error(
                "Missing white/black reference captures (white=%d, black=%d). "
                "Calibration aborted — camera may not see projector patterns.",
                len(self._captures[0]), len(self._captures[1]),
            )
            return

        white_f = self._get_averaged(0)
        ref_shape = white_f.shape[:2]
        black_f = self._get_averaged(1, target_shape=ref_shape)

        h, w = ref_shape
        bt = self.bit_threshold
        st = self.shadow_threshold

        logger.info(
            "Decode: camera %dx%d, projector %dx%d, %d+%d bits, "
            "bit_threshold=%.1f, shadow_threshold=%.1f",
            w, h, self.proj_w, self.proj_h,
            self.bits_x, self.bits_y, bt, st,
        )
        logger.info(
            "  WHITE ref: min=%.1f max=%.1f mean=%.1f",
            white_f.min(), white_f.max(), white_f.mean(),
        )
        logger.info(
            "  BLACK ref: min=%.1f max=%.1f mean=%.1f",
            black_f.min(), black_f.max(), black_f.mean(),
        )

        # -- Shadow mask (OpenCV convention) ---------------------------------
        # Pixels where white-black > shadow_threshold are illuminated by the
        # projector. Everything else is in shadow or outside coverage.
        # This is the critical first filter — without it, noise pixels
        # decode to random projector coordinates and pollute the maps.
        shadow_mask = (white_f - black_f) > st
        n_shadow = int(np.count_nonzero(shadow_mask))
        logger.info(
            "  Shadow mask: %d/%d pixels illuminated (%.1f%%) "
            "[shadow_threshold=%.1f, white-black range: %.1f..%.1f]",
            n_shadow, h * w, 100.0 * n_shadow / max(h * w, 1),
            st, (white_f - black_f).min(), (white_f - black_f).max(),
        )

        if n_shadow == 0:
            logger.error(
                "  ZERO illuminated pixels! The camera may not be seeing "
                "the projector patterns. Check: (1) projector is on and "
                "visible to camera, (2) camera feed is working, "
                "(3) calibration_brightness is high enough. "
                "white mean=%.1f, black mean=%.1f, diff mean=%.1f",
                white_f.mean(), black_f.mean(), (white_f - black_f).mean(),
            )

        # -- Decode X (column) Gray codes -----------------------------------
        base = 2  # skip white + black
        decoded_x = np.zeros((h, w), dtype=np.int32)
        valid_x = shadow_mask.copy()  # start from shadow mask
        conf_x = np.zeros((h, w), dtype=np.float32)

        # Only the top 2 MSB bits gate validity — these are the coarsest
        # stripes with the highest contrast. Finer bits are too noisy to
        # use as a validity gate (especially through webcam/MJPEG compression).
        msb_gate_count_x = min(2, self.bits_x)

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
            diff = pos_f - neg_f
            bit_val = (diff > 0).astype(np.int32)
            decoded_x |= bit_val << actual_bit

            # Per-bit reliability (OpenCV whiteThreshold equivalent)
            abs_diff = np.abs(diff)
            conf_x += abs_diff

            # Only the top N MSB bits gate validity
            if capture_idx < msb_gate_count_x:
                bit_reliable = abs_diff > bt
                valid_x &= bit_reliable

            if capture_idx < 3 or capture_idx == self.bits_x - 1:
                logger.info(
                    "  X bit %d (capture %d): diff mean=%.1f, "
                    "reliable(>%.1f)=%.1f%%, gate=%s",
                    actual_bit, capture_idx, abs_diff.mean(), bt,
                    100.0 * np.count_nonzero(abs_diff > bt) / max(h * w, 1),
                    "yes" if capture_idx < msb_gate_count_x else "no",
                )

        # -- Decode Y (row) Gray codes --------------------------------------
        base_y = base + self.bits_x * 2
        decoded_y = np.zeros((h, w), dtype=np.int32)
        valid_y = shadow_mask.copy()  # start from shadow mask
        conf_y = np.zeros((h, w), dtype=np.float32)

        msb_gate_count_y = min(2, self.bits_y)

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
            diff = pos_f - neg_f
            bit_val = (diff > 0).astype(np.int32)
            decoded_y |= bit_val << actual_bit

            abs_diff = np.abs(diff)
            conf_y += abs_diff

            if capture_idx < msb_gate_count_y:
                bit_reliable = abs_diff > bt
                valid_y &= bit_reliable

            if capture_idx < 3 or capture_idx == self.bits_y - 1:
                logger.info(
                    "  Y bit %d (capture %d): diff mean=%.1f, "
                    "reliable(>%.1f)=%.1f%%, gate=%s",
                    actual_bit, capture_idx, abs_diff.mean(), bt,
                    100.0 * np.count_nonzero(abs_diff > bt) / max(h * w, 1),
                    "yes" if capture_idx < msb_gate_count_y else "no",
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

        # -- Combine shadow mask + per-bit reliability + bounds ---------------
        valid_mask = (
            valid_x
            & valid_y
            & (decoded_x >= 0) & (decoded_x < self.proj_w)
            & (decoded_y >= 0) & (decoded_y < self.proj_h)
        )
        n_valid = int(np.count_nonzero(valid_mask))
        logger.info(
            "  Shadow mask: %d, valid_x: %d, valid_y: %d, "
            "in-bounds: %d, final valid: %d/%d (%.1f%%)",
            n_shadow,
            int(np.count_nonzero(valid_x)),
            int(np.count_nonzero(valid_y)),
            int(np.count_nonzero(
                (decoded_x >= 0) & (decoded_x < self.proj_w)
                & (decoded_y >= 0) & (decoded_y < self.proj_h)
            )),
            n_valid, h * w, 100.0 * n_valid / max(h * w, 1),
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

        # Store camera resolution for stereo triangulation
        self.cam_w = w
        self.cam_h = h

        # -- Compute disparity map (homography residual) ----------------------
        # Raw (cam_x - proj_x) is dominated by the planar projection
        # relationship and produces a smooth gradient even for flat scenes.
        # Instead, fit a homography (perspective transform) to the
        # projector→camera mapping and use the RESIDUAL displacement as
        # depth — closer objects deviate more from the planar model.
        px = np.arange(self.proj_w, dtype=np.float32)[None, :].repeat(
            self.proj_h, axis=0,
        )
        py = np.arange(self.proj_h, dtype=np.float32)[:, None].repeat(
            self.proj_w, axis=1,
        )

        src = np.column_stack([px[valid_proj], py[valid_proj]]).astype(np.float32)
        dst = np.column_stack(
            [map_x[valid_proj], map_y[valid_proj]],
        ).astype(np.float32)

        # Subsample for findHomography performance
        rng = np.random.default_rng(42)
        if len(src) > 10000:
            idx = rng.choice(len(src), 10000, replace=False)
            H, _ = cv2.findHomography(src[idx], dst[idx], cv2.RANSAC, 5.0)
        elif len(src) > 4:
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        else:
            H = None

        disparity = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)

        if H is not None:
            # Apply homography to all projector pixels → predicted camera coords
            pts = np.column_stack(
                [px.ravel(), py.ravel()],
            ).astype(np.float32).reshape(-1, 1, 2)
            pred = cv2.perspectiveTransform(pts, H).reshape(
                self.proj_h, self.proj_w, 2,
            )
            # Residual = actual - predicted correspondence
            res_x = map_x - pred[:, :, 0]
            res_y = map_y - pred[:, :, 1]
            disp = np.sqrt(res_x ** 2 + res_y ** 2)
            disp[~valid_proj] = 0.0
            disparity = disp
            logger.info("  Disparity: homography residual method (H found)")
        else:
            # Fallback: simple horizontal disparity
            disparity[valid_proj] = map_x[valid_proj] - px[valid_proj]
            logger.warning("  Disparity: homography fit failed — raw cam_x - proj_x")

        # Normalize to [0, 1] (near=bright, far=dark)
        valid_vals = disparity[valid_proj]
        if valid_vals.size > 0:
            p2 = float(np.percentile(valid_vals, 2))
            p98 = float(np.percentile(valid_vals, 98))
            if p98 - p2 > 1e-6:
                disparity = (disparity - p2) / (p98 - p2)
            else:
                disparity = np.full_like(disparity, 0.5)
            logger.info(
                "  Disparity normalize: p2=%.4f p98=%.4f spread=%.4f",
                p2, p98, p98 - p2,
            )
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
    cam_w: int | None = None,
    cam_h: int | None = None,
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

    meta: dict = {
        "version": 2,
        "projector_width": proj_w,
        "projector_height": proj_h,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "npz_file": npz_path.name,
    }
    if cam_w is not None:
        meta["camera_width"] = cam_w
    if cam_h is not None:
        meta["camera_height"] = cam_h
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


def load_calibration_meta(path: str | Path) -> dict:
    """Load calibration metadata from the JSON sidecar.

    Returns all JSON fields including optional ``camera_width`` and
    ``camera_height``.  Gracefully handles missing fields for older
    calibration files.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        logger.warning("Could not read calibration metadata from %s", p)
        return {}


# -- Camera intrinsics I/O ---------------------------------------------------

_INTRINSICS_PATH = Path.home() / ".projectionmapanything_intrinsics.json"


def save_camera_intrinsics(
    K_cam: np.ndarray,
    dist_cam: np.ndarray,
    image_size: tuple[int, int],
    reprojection_error: float,
    path: str | Path | None = None,
) -> None:
    """Save camera intrinsics to JSON.

    Parameters
    ----------
    K_cam : np.ndarray
        (3, 3) float64 camera matrix.
    dist_cam : np.ndarray
        (1, 5) or (5,) float64 distortion coefficients.
    image_size : tuple[int, int]
        (width, height) of the calibration images.
    reprojection_error : float
        Mean reprojection error in pixels from ``cv2.calibrateCamera``.
    path : str | Path | None
        Output path.  Defaults to ``_INTRINSICS_PATH``.
    """
    p = Path(path) if path is not None else _INTRINSICS_PATH
    data = {
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "reprojection_error": float(reprojection_error),
        "K_cam": K_cam.tolist(),
        "dist_cam": dist_cam.flatten().tolist(),
    }
    p.write_text(json.dumps(data, indent=2))
    logger.info(
        "Saved camera intrinsics to %s (fx=%.1f, fy=%.1f, error=%.4f px)",
        p.name, K_cam[0, 0], K_cam[1, 1], reprojection_error,
    )


def save_procam_intrinsics(
    K_cam: np.ndarray,
    dist_cam: np.ndarray,
    cam_size: tuple[int, int],
    cam_error: float,
    K_proj: np.ndarray,
    dist_proj: np.ndarray,
    proj_size: tuple[int, int],
    proj_error: float,
    path: str | Path | None = None,
) -> None:
    """Save both camera and projector intrinsics to JSON.

    Parameters
    ----------
    K_cam, K_proj : np.ndarray
        (3, 3) float64 camera/projector matrix.
    dist_cam, dist_proj : np.ndarray
        Distortion coefficients.
    cam_size, proj_size : tuple[int, int]
        (width, height) of calibration images.
    cam_error, proj_error : float
        Mean reprojection error in pixels.
    path : str | Path | None
        Output path. Defaults to ``_INTRINSICS_PATH``.
    """
    p = Path(path) if path is not None else _INTRINSICS_PATH
    data = {
        "version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_width": cam_size[0],
        "image_height": cam_size[1],
        "reprojection_error": float(cam_error),
        "K_cam": K_cam.tolist(),
        "dist_cam": dist_cam.flatten().tolist(),
        "projector_width": proj_size[0],
        "projector_height": proj_size[1],
        "projector_reprojection_error": float(proj_error),
        "K_proj": K_proj.tolist(),
        "dist_proj": dist_proj.flatten().tolist(),
    }
    p.write_text(json.dumps(data, indent=2))
    logger.info(
        "Saved procam intrinsics to %s (cam fx=%.1f err=%.4f, "
        "proj fx=%.1f err=%.4f)",
        p.name, K_cam[0, 0], cam_error, K_proj[0, 0], proj_error,
    )


def load_camera_intrinsics(
    path: str | Path | None = None,
) -> dict | None:
    """Load camera intrinsics from JSON.

    Returns
    -------
    dict | None
        Dict with ``K_cam`` (3x3 ndarray), ``dist_cam`` (1x5 ndarray),
        ``image_size`` (w, h), ``reprojection_error`` (float), and
        ``timestamp``.  Returns ``None`` if the file doesn't exist or
        can't be parsed.
    """
    p = Path(path) if path is not None else _INTRINSICS_PATH
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        K_cam = np.array(data["K_cam"], dtype=np.float64)
        dist_cam = np.array(data["dist_cam"], dtype=np.float64).reshape(1, -1)
        return {
            "K_cam": K_cam,
            "dist_cam": dist_cam,
            "image_size": (data["image_width"], data["image_height"]),
            "reprojection_error": float(data["reprojection_error"]),
            "timestamp": data.get("timestamp", "unknown"),
        }
    except Exception:
        logger.warning("Could not load camera intrinsics from %s", p)
        return None
