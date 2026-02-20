"""Structured light calibration for projector-camera systems.

Supports two calibration methods:

**Multi-Frequency Phase Shifting (MPS)** — default
  Projects smooth sinusoidal patterns at multiple spatial frequencies.
  Survives JPEG compression (MJPEG stream), needs fewer patterns (~32 vs ~46),
  gives sub-pixel precision, and is robust to ambient light and defocus.

**Gray Code** — legacy fallback
  Projects binary stripe patterns.  Requires lossless or high-quality capture
  because JPEG compression destroys the sharp black/white edges.

Both methods produce a dense pixel mapping (map_x, map_y) from camera space
to projector space, usable with ``cv2.remap``.
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


class CalibrationMethod(str, Enum):
    """Structured light calibration method."""
    PHASE_SHIFT = "phase_shift"
    GRAY_CODE = "gray_code"


class CalibrationState:
    """Manages the multi-frame structured light calibration sequence.

    Supports both Multi-Frequency Phase Shifting (MPS) and Gray code methods.

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
        method: CalibrationMethod | str = CalibrationMethod.PHASE_SHIFT,
        num_frequencies: int = 5,
        frequency_ratio: float = 3.0,
        pattern_amplitude: float = 1.0,
        settle_frames: int = 30,
        capture_frames: int = 3,
        decode_threshold: float = 10.0,
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
        self.change_threshold = change_threshold
        self.stability_threshold = stability_threshold

        # Method selection
        if isinstance(method, str):
            self.method = CalibrationMethod(method)
        else:
            self.method = method

        # Gray code params (used by both methods for bits_x/bits_y reference)
        self.bits_x = _num_bits(proj_w)
        self.bits_y = _num_bits(proj_h)

        # MPS params
        self.num_frequencies = num_frequencies
        self.frequency_ratio = frequency_ratio
        self.amplitude = pattern_amplitude

        if self.method == CalibrationMethod.PHASE_SHIFT:
            self._frequencies_x = self._compute_wavelengths(
                proj_w, num_frequencies, frequency_ratio,
            )
            self._frequencies_y = self._compute_wavelengths(
                proj_h, num_frequencies, frequency_ratio,
            )
            # 3 phase shifts per frequency, 2 axes
            self._num_pattern_steps = num_frequencies * 3 * 2
        else:
            self._frequencies_x = []
            self._frequencies_y = []
            # pos/neg pairs for each bit, 2 axes
            self._num_pattern_steps = 2 * (self.bits_x + self.bits_y)

        # Total patterns: white + black + all pattern steps
        self.total_patterns = 2 + self._num_pattern_steps

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

        logger.info(
            "CalibrationState: method=%s, %dx%d, %d patterns "
            "(%d freqs, ratio=%.1f)" if self.method == CalibrationMethod.PHASE_SHIFT
            else "CalibrationState: method=%s, %dx%d, %d patterns "
            "(%d bits_x, %d bits_y)",
            self.method.value, proj_w, proj_h, self.total_patterns,
            *(
                (num_frequencies, frequency_ratio)
                if self.method == CalibrationMethod.PHASE_SHIFT
                else (self.bits_x, self.bits_y)
            ),
        )

    @staticmethod
    def _compute_wavelengths(
        dim: int, num_freq: int, ratio: float,
    ) -> list[float]:
        """Compute hierarchical wavelengths for phase shifting.

        Starting from the full dimension (1 period), each step divides by ratio.
        """
        wavelengths: list[float] = []
        wl = float(dim)
        for _ in range(num_freq):
            wavelengths.append(wl)
            wl = max(2.0, wl / ratio)
        return wavelengths

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

        if self.method == CalibrationMethod.PHASE_SHIFT:
            idx = self._pattern_index
            ppa = self.num_frequencies * 3
            if idx < ppa:
                freq = idx // 3 + 1
                shift = idx % 3 + 1
                return (
                    f"X freq {freq}/{self.num_frequencies} shift {shift}/3"
                    f" ({captured}/{total_cap} captures)"
                )
            else:
                y_idx = idx - ppa
                freq = y_idx // 3 + 1
                shift = y_idx % 3 + 1
                return (
                    f"Y freq {freq}/{self.num_frequencies} shift {shift}/3"
                    f" ({captured}/{total_cap} captures)"
                )
        else:
            idx = self._pattern_index
            total_x = 2 * self.bits_x
            if idx < total_x:
                bit = idx // 2 + 1
                return f"bit {bit}/{self.bits_x} X-axis ({captured}/{total_cap} captures)"
            else:
                y_idx = idx - total_x
                bit = y_idx // 2 + 1
                return f"bit {bit}/{self.bits_y} Y-axis ({captured}/{total_cap} captures)"

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
            method_name = (
                "phase shift" if self.method == CalibrationMethod.PHASE_SHIFT
                else "Gray code"
            )
            logger.info("Running %s decode ...", method_name)
            self._decode()
            self.phase = CalibrationPhase.DONE
            return None

        if self.phase in (CalibrationPhase.IDLE, CalibrationPhase.DONE):
            return None

        # Change-detection settle: wait for the camera to actually show the
        # new pattern, then capture as soon as it stabilises.
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
            if self._pattern_index >= self._num_pattern_steps:
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
            if self.method == CalibrationMethod.PHASE_SHIFT:
                pattern = self._mps_pattern_for_index(self._pattern_index)
            else:
                pattern = self._graycode_pattern_for_index(self._pattern_index)
        else:
            pattern = np.zeros((self.proj_h, self.proj_w), dtype=np.uint8)

        tensor = self._pattern_to_tensor(pattern, device)
        self._cached_pattern = tensor
        self._cached_pattern_key = key
        return tensor

    # -- MPS pattern generation -----------------------------------------------

    def _mps_pattern_for_index(self, idx: int) -> np.ndarray:
        """Generate an MPS sinusoidal pattern for the given flat index.

        Pattern index mapping:
          idx < patterns_per_axis  ->  X-axis
          idx >= patterns_per_axis ->  Y-axis
        Within each axis: freq_idx = idx // 3, shift_idx = idx % 3
        """
        patterns_per_axis = self.num_frequencies * 3

        if idx < patterns_per_axis:
            axis = 0
            freq_idx = idx // 3
            shift_idx = idx % 3
        else:
            axis = 1
            y_idx = idx - patterns_per_axis
            freq_idx = y_idx // 3
            shift_idx = y_idx % 3

        return self._generate_sinusoidal_pattern(freq_idx, shift_idx, axis)

    def _generate_sinusoidal_pattern(
        self, freq_idx: int, shift_idx: int, axis: int,
    ) -> np.ndarray:
        """Generate a single sinusoidal pattern for phase shifting.

        Parameters
        ----------
        freq_idx : int
            Index into the frequency list (0 = coarsest).
        shift_idx : int
            Phase shift index: 0 = 0deg, 1 = 120deg, 2 = 240deg.
        axis : int
            0 = X (horizontal stripes vary across columns),
            1 = Y (vertical stripes vary across rows).

        Returns
        -------
        np.ndarray
            (proj_h, proj_w) uint8 pattern.
        """
        if axis == 0:
            wavelength = self._frequencies_x[freq_idx]
            coords = np.arange(self.proj_w, dtype=np.float64).reshape(1, -1)
        else:
            wavelength = self._frequencies_y[freq_idx]
            coords = np.arange(self.proj_h, dtype=np.float64).reshape(-1, 1)

        phase_shift = shift_idx * 2.0 * np.pi / 3.0
        sinusoid = 0.5 + self.amplitude * 0.5 * np.cos(
            2.0 * np.pi * coords / wavelength + phase_shift
        )
        pattern = (sinusoid * self.max_brightness).clip(0, 255).astype(np.uint8)

        # Broadcast to full projector resolution
        pattern = np.broadcast_to(pattern, (self.proj_h, self.proj_w)).copy()
        return pattern

    # -- Gray code pattern generation -----------------------------------------

    def _graycode_pattern_for_index(self, idx: int) -> np.ndarray:
        """Generate a Gray code pattern for the given flat index."""
        total_x = 2 * self.bits_x
        if idx < total_x:
            bit = idx // 2
            inverted = idx % 2 == 1
            return generate_pattern(
                bit, axis=0, inverted=inverted,
                proj_w=self.proj_w, proj_h=self.proj_h,
            )
        else:
            y_idx = idx - total_x
            bit = y_idx // 2
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

    # -- Decode dispatch ------------------------------------------------------

    def _decode(self) -> None:
        """Decode captured patterns into a camera->projector mapping."""
        if self.method == CalibrationMethod.PHASE_SHIFT:
            self._decode_mps()
        else:
            self._decode_graycode()

    # -- MPS phase utilities --------------------------------------------------

    @staticmethod
    def _phase_to_position(phase: np.ndarray, wavelength: float) -> np.ndarray:
        """Convert wrapped phase (atan2 output, [-pi, pi]) to position [0, wavelength).

        Uses modular arithmetic to map phase to pixel coordinates.  Handles
        the floating-point edge case where ``-epsilon % (2*pi)`` evaluates to
        ``≈ 2*pi`` instead of ``0``, which would otherwise produce a 1-pixel
        error of one full wavelength at position 0.
        """
        TWO_PI = 2.0 * np.pi
        wrapped = phase % TWO_PI  # map [-pi,pi] -> [0, 2pi)
        # Fix edge case: values very close to 2pi should be 0
        wrapped = np.where(wrapped >= TWO_PI - 1e-10, 0.0, wrapped)
        return wrapped / TWO_PI * wavelength

    # -- MPS decode -----------------------------------------------------------

    def _decode_mps(self) -> None:
        """Decode captured phase-shift patterns into a camera->projector mapping.

        Algorithm:
        1. Compute per-pixel wrapped phase at each frequency (3-step formula)
        2. Build valid mask from modulation amplitude
        3. Hierarchical phase unwrapping (coarse -> fine)
        4. Median filter + clamp
        5. Build remap tables (shared with Gray code path)
        """
        white_f = self._get_averaged(0)
        ref_shape = white_f.shape[:2]
        black_f = self._get_averaged(1, target_shape=ref_shape)

        h, w = ref_shape
        diff = white_f - black_f

        logger.info(
            "MPS Decode: camera %dx%d, projector %dx%d, %d frequencies",
            w, h, self.proj_w, self.proj_h, self.num_frequencies,
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
            "  DIFF (white-black): min=%.1f max=%.1f mean=%.1f",
            diff.min(), diff.max(), diff.mean(),
        )
        logger.info(
            "  X wavelengths: %s",
            [f"{wl:.0f}" for wl in self._frequencies_x],
        )
        logger.info(
            "  Y wavelengths: %s",
            [f"{wl:.0f}" for wl in self._frequencies_y],
        )

        base = 2  # skip white + black
        patterns_per_axis = self.num_frequencies * 3
        TWO_PI = 2.0 * np.pi

        # -- Decode X axis: compute phase and amplitude at each frequency -----
        phases_x = np.zeros((self.num_frequencies, h, w), dtype=np.float64)
        amplitudes_x = np.zeros((self.num_frequencies, h, w), dtype=np.float64)

        for freq_idx in range(self.num_frequencies):
            # Captures at phase shifts 0deg, 120deg, 240deg
            i0 = self._get_averaged(
                base + freq_idx * 3 + 0, target_shape=ref_shape,
            ).astype(np.float64)
            i1 = self._get_averaged(
                base + freq_idx * 3 + 1, target_shape=ref_shape,
            ).astype(np.float64)
            i2 = self._get_averaged(
                base + freq_idx * 3 + 2, target_shape=ref_shape,
            ).astype(np.float64)

            # Standard 3-step formula (shifts at 0, 2pi/3, 4pi/3):
            #   phase = atan2(sqrt(3)*(I2-I1), 2*I0-I1-I2)
            # This correctly recovers the spatial phase theta = 2*pi*x/lambda
            num = np.sqrt(3.0) * (i2 - i1)
            den = 2.0 * i0 - i1 - i2
            phases_x[freq_idx] = np.arctan2(num, den)
            amplitudes_x[freq_idx] = np.sqrt(num**2 + den**2) / 3.0

            if freq_idx < 3:
                logger.info(
                    "  X freq %d (wl=%.0f): phase [%.2f, %.2f], "
                    "amp [%.1f, %.1f] mean=%.1f",
                    freq_idx, self._frequencies_x[freq_idx],
                    phases_x[freq_idx].min(), phases_x[freq_idx].max(),
                    amplitudes_x[freq_idx].min(), amplitudes_x[freq_idx].max(),
                    amplitudes_x[freq_idx].mean(),
                )

        # -- Decode Y axis ----------------------------------------------------
        phases_y = np.zeros((self.num_frequencies, h, w), dtype=np.float64)
        amplitudes_y = np.zeros((self.num_frequencies, h, w), dtype=np.float64)

        for freq_idx in range(self.num_frequencies):
            i0 = self._get_averaged(
                base + patterns_per_axis + freq_idx * 3 + 0,
                target_shape=ref_shape,
            ).astype(np.float64)
            i1 = self._get_averaged(
                base + patterns_per_axis + freq_idx * 3 + 1,
                target_shape=ref_shape,
            ).astype(np.float64)
            i2 = self._get_averaged(
                base + patterns_per_axis + freq_idx * 3 + 2,
                target_shape=ref_shape,
            ).astype(np.float64)

            num = np.sqrt(3.0) * (i2 - i1)
            den = 2.0 * i0 - i1 - i2
            phases_y[freq_idx] = np.arctan2(num, den)
            amplitudes_y[freq_idx] = np.sqrt(num**2 + den**2) / 3.0

            if freq_idx < 3:
                logger.info(
                    "  Y freq %d (wl=%.0f): phase [%.2f, %.2f], "
                    "amp [%.1f, %.1f] mean=%.1f",
                    freq_idx, self._frequencies_y[freq_idx],
                    phases_y[freq_idx].min(), phases_y[freq_idx].max(),
                    amplitudes_y[freq_idx].min(), amplitudes_y[freq_idx].max(),
                    amplitudes_y[freq_idx].mean(),
                )

        # -- Valid mask from modulation amplitude -----------------------------
        # Pixels with low amplitude didn't receive projector light.
        # Use Otsu thresholding on the coarsest frequency amplitude.
        amp_combined = (amplitudes_x[0] + amplitudes_y[0]) / 2.0
        amp_max = max(float(amp_combined.max()), 1.0)
        amp_u8 = (amp_combined / amp_max * 255).clip(0, 255).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(
            amp_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        # Use half-Otsu as the cutoff — Otsu can be too aggressive
        effective_amp_thresh = max(int(otsu_thresh * 0.5), 5)
        valid_mask = amp_u8 > effective_amp_thresh

        logger.info(
            "  Amplitude: max=%.1f, Otsu=%d/255, effective thresh=%d/255",
            amp_max, int(otsu_thresh), effective_amp_thresh,
        )

        # Supplement with white-black diff if it shows meaningful contrast
        diff_max = float(diff.max())
        if diff_max > self.decode_threshold:
            # Adaptive: use Otsu on the diff to find the illuminated region
            diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)
            wb_otsu, _ = cv2.threshold(
                diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            wb_mask = diff > max(float(wb_otsu) * 0.3, self.decode_threshold * 0.5)
            valid_mask = valid_mask & wb_mask
            logger.info(
                "  White-black diff Otsu=%d, wb_mask pixels=%d",
                int(wb_otsu), int(np.count_nonzero(wb_mask)),
            )
        else:
            logger.warning(
                "  White-black diff too low (max=%.1f) — using amplitude only",
                diff_max,
            )

        n_valid = int(np.count_nonzero(valid_mask))
        logger.info(
            "  Valid pixels (Otsu on amplitude): %d/%d (%.1f%%), "
            "Otsu threshold=%d/255",
            n_valid, h * w, 100.0 * n_valid / max(h * w, 1),
            int(otsu_thresh),
        )

        # -- Hierarchical phase unwrapping: X axis ----------------------------
        # Coarsest frequency: 1 period across full width, unambiguous.
        # Convert wrapped phase to position using _phase_to_position which
        # handles the floating-point edge case at phase ≈ 0.
        absolute_x = self._phase_to_position(phases_x[0], self._frequencies_x[0])

        for i in range(1, self.num_frequencies):
            # Position within one period of the finer frequency
            fine_px = self._phase_to_position(phases_x[i], self._frequencies_x[i])
            # Number of whole wavelengths offset (from coarser estimate)
            k = np.round((absolute_x - fine_px) / self._frequencies_x[i])
            # Unwrapped fine-resolution estimate
            absolute_x = fine_px + k * self._frequencies_x[i]

        # -- Hierarchical phase unwrapping: Y axis ----------------------------
        absolute_y = self._phase_to_position(phases_y[0], self._frequencies_y[0])

        for i in range(1, self.num_frequencies):
            fine_px = self._phase_to_position(phases_y[i], self._frequencies_y[i])
            k = np.round((absolute_y - fine_px) / self._frequencies_y[i])
            absolute_y = fine_px + k * self._frequencies_y[i]

        # -- Median filter for noise reduction + clamp to valid range ---------
        decoded_x = cv2.medianBlur(absolute_x.astype(np.float32), 5)
        decoded_y = cv2.medianBlur(absolute_y.astype(np.float32), 5)
        decoded_x = np.clip(decoded_x, 0, self.proj_w - 1)
        decoded_y = np.clip(decoded_y, 0, self.proj_h - 1)

        if n_valid > 0:
            logger.info(
                "  Decoded X range: [%.1f, %.1f], Y range: [%.1f, %.1f]",
                float(decoded_x[valid_mask].min()),
                float(decoded_x[valid_mask].max()),
                float(decoded_y[valid_mask].min()),
                float(decoded_y[valid_mask].max()),
            )

        # Build remap tables (shared pipeline with Gray code)
        self._build_remap_from_correspondences(
            decoded_x.astype(np.float32),
            decoded_y.astype(np.float32),
            valid_mask, h, w,
        )

    # -- Gray code decode -----------------------------------------------------

    def _decode_graycode(self) -> None:
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
        ref_shape = white_f.shape[:2]
        black_f = self._get_averaged(1, target_shape=ref_shape)

        h, w = ref_shape
        diff = white_f - black_f

        logger.info(
            "Gray code Decode: camera resolution %dx%d, projector %dx%d",
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
        # Adaptive threshold: use Otsu on the diff if it has enough dynamic
        # range, otherwise fall back to the configured threshold.
        diff_clipped = np.clip(diff, 0, 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(
            diff_clipped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        effective_threshold = max(float(otsu_val) * 0.5, self.decode_threshold)
        logger.info(
            "  DIFF (white-black): min=%.1f max=%.1f mean=%.1f, "
            "Otsu=%.0f, effective threshold=%.1f",
            diff.min(), diff.max(), diff.mean(),
            float(otsu_val), effective_threshold,
        )

        valid_mask = diff > effective_threshold
        n_valid_diff = int(np.count_nonzero(valid_mask))
        logger.info(
            "  Pixels passing white-black threshold: %d/%d (%.1f%%)",
            n_valid_diff, h * w, 100.0 * n_valid_diff / max(h * w, 1),
        )

        # Track per-bit contrast for fallback valid mask
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
        # use per-bit contrast as the valid mask.
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

        # Clamp to projector bounds
        decoded_x = np.clip(decoded_x, 0, self.proj_w - 1)
        decoded_y = np.clip(decoded_y, 0, self.proj_h - 1)

        # Build remap tables (shared pipeline)
        self._build_remap_from_correspondences(
            decoded_x.astype(np.float32),
            decoded_y.astype(np.float32),
            valid_mask, h, w,
        )

    # -- Shared remap building ------------------------------------------------

    def _build_remap_from_correspondences(
        self,
        decoded_x: np.ndarray,
        decoded_y: np.ndarray,
        valid_mask: np.ndarray,
        cam_h: int,
        cam_w: int,
    ) -> None:
        """Build projector->camera remap tables from decoded correspondences.

        Shared by both MPS and Gray code decode paths.  Takes decoded
        projector coordinates (float32) for each camera pixel and builds
        the dense inverse mapping (projector pixel -> camera pixel).

        Pipeline:
        1. Morphological cleanup of valid mask
        2. Spatial consistency filtering
        3. Gaussian splat fill for dense correspondence
        4. Outlier rejection (median-based)
        5. Re-fill after outlier removal
        6. Final smoothing (median + bilateral)
        7. Inpainting small holes
        """
        # -- Morphological cleanup on camera-space valid mask -----------------
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

        # -- Build inverse mapping: projector pixel -> camera pixel -----------
        sum_cx = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        sum_cy = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)
        count = np.zeros((self.proj_h, self.proj_w), dtype=np.float32)

        cam_ys, cam_xs = np.where(valid_mask)
        # Round to integer projector pixel indices for splat accumulation
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

        # Gaussian splat: spread correspondences to fill gaps.
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

        # -- Outlier rejection: median-based smoothness filter ----------------
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

        # -- Re-fill after outlier removal with Gaussian splat ----------------
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

        # -- Final smoothing: median + bilateral filter -----------------------
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
            "  Final projector pixels with correspondence: %d/%d (%.1f%%)",
            n_valid_proj, total_proj, 100.0 * n_valid_proj / max(total_proj, 1),
        )

        # -- Gradient quality check (diagnostic) -----------------------------
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

    # Fast binary save — instant even for 1920x1080
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
