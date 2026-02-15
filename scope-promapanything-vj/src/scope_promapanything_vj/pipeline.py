"""ProMapAnything VJ Tools — pipelines.

Registered pipelines:
1. ProMapAnythingPipeline          — preprocessor (calibration + depth -> ControlNet)
2. ProMapAnythingProjectorPipeline — postprocessor (streams final output to projector)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .calibration import (
    CalibrationPhase,
    CalibrationState,
    load_calibration,
    save_calibration,
)
from .depth_provider import create_depth_provider
from .frame_server import FrameStreamer, get_or_create_streamer
from .remap import build_warped_depth_image
from .schema import ProMapAnythingConfig, ProMapAnythingProjectorConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Calibration + Depth preprocessor
# =============================================================================


class ProMapAnythingPipeline(Pipeline):
    """Projection mapping depth preprocessor.

    In **calibration mode**, projects Gray code patterns via the shared MJPEG
    streamer and captures the camera response to build a projector-camera
    pixel mapping.

    In **normal mode**, estimates depth from the camera feed, warps it to the
    projector's perspective using the calibration mapping, and outputs the
    result as a VACE-optimized grayscale depth map.

    All depth output parameters are hardcoded to VACE-optimal values:
    no CLAHE, full percentile range, no gamma, grayscale, no invert.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ProMapAnythingConfig

    # Resolution scale factors for generation_resolution preset
    _RESOLUTION_SCALES = {"quarter": 0.25, "half": 0.5, "native": 1.0}

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load-time parameters
        self.proj_w: int = kwargs.get("projector_width", 1920)
        self.proj_h: int = kwargs.get("projector_height", 1080)
        self._gen_res: str = kwargs.get("generation_resolution", "half")
        calibration_file: str = kwargs.get("calibration_file", "")

        # Depth estimation (hardcoded auto/small)
        self._depth = create_depth_provider("auto", self.device, model_size="small")

        # Calibration state
        self._calib: CalibrationState | None = None
        self._was_calibrating = False

        # Calibration mapping (populated by calibration or loaded from file)
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self._proj_valid_mask: np.ndarray | None = None

        # Temporal smoothing buffer
        self._prev_depth: torch.Tensor | None = None

        # Shared streamer (lazy-init)
        self._streamer: FrameStreamer | None = None

        # Load external calibration file if provided
        if calibration_file:
            path = Path(calibration_file)
            if path.is_file():
                mx, my, pw, ph = load_calibration(path)
                self._map_x = mx
                self._map_y = my
                self.proj_w = pw
                self.proj_h = ph
                logger.info("Loaded calibration from %s", path)
            else:
                logger.warning("Calibration file not found: %s", path)

        logger.info("Depth source: camera + depth estimation (small)")

    def _get_streamer(self) -> FrameStreamer:
        """Lazy-init the shared singleton streamer."""
        if self._streamer is None or not self._streamer.is_running:
            self._streamer = get_or_create_streamer()
        return self._streamer

    def _get_projector_resolution(self) -> tuple[int, int]:
        """Prefer companion app's resolution, fallback to config fields."""
        streamer = self._get_streamer()
        cfg = streamer.client_config
        if cfg and "width" in cfg and "height" in cfg:
            return int(cfg["width"]), int(cfg["height"])
        return self.proj_w, self.proj_h

    def _get_generation_resolution(self) -> tuple[int, int]:
        """Compute generation resolution from projector res + preset.

        Returns (gen_w, gen_h) rounded to nearest multiple of 8.
        """
        proj_w, proj_h = self._get_projector_resolution()
        scale = self._RESOLUTION_SCALES.get(self._gen_res, 0.5)
        gen_w = max(64, round(proj_w * scale / 8) * 8)
        gen_h = max(64, round(proj_h * scale / 8) * 8)
        return gen_w, gen_h

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("ProMapAnythingPipeline requires video input")

        # Unpack single input frame — (1, H, W, C) [0, 255]
        frame = video[0]

        # Read runtime parameters
        calibrate = kwargs.get("calibrate", False)
        temporal_smoothing = kwargs.get("temporal_smoothing", 0.5)
        depth_blur = kwargs.get("depth_blur", 0.0)

        streamer = self._get_streamer()

        # -- Calibration mode -------------------------------------------------
        if calibrate:
            if not self._was_calibrating:
                # Check for companion app resolution
                proj_w, proj_h = self._get_projector_resolution()
                if proj_w != self.proj_w or proj_h != self.proj_h:
                    self.proj_w, self.proj_h = proj_w, proj_h
                    logger.info(
                        "Using companion app resolution: %dx%d",
                        proj_w, proj_h,
                    )

                self._calib = CalibrationState(self.proj_w, self.proj_h)
                self._calib.start()
                self._was_calibrating = True
                streamer.calibration_active = True
                logger.info(
                    "Calibration started (%d patterns)",
                    self._calib.total_patterns,
                )

            pattern = self._calib.step(frame, self.device)

            if self._calib.phase == CalibrationPhase.DONE:
                mapping = self._calib.get_mapping()
                if mapping is not None:
                    self._map_x, self._map_y = mapping
                    self._proj_valid_mask = self._calib.proj_valid_mask
                    # Auto-save calibration
                    save_path = Path.home() / ".promapanything_calibration.json"
                    save_calibration(
                        self._map_x, self._map_y, save_path,
                        self.proj_w, self.proj_h,
                    )
                    logger.info("Calibration complete — saved to %s", save_path)
                self._was_calibrating = False
                streamer.calibration_active = False
                # Fall through to normal depth processing

            elif pattern is not None:
                # Stream the calibration pattern to the companion app
                t = pattern.squeeze(0) if pattern.ndim == 4 else pattern
                rgb_np = (t.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                streamer.submit_calibration_frame(rgb_np)
                return {"video": pattern}

        else:
            if self._was_calibrating:
                self._was_calibrating = False
                streamer.calibration_active = False

        # -- Normal depth processing ------------------------------------------

        # Normalise input to [0, 1]
        frame_f = frame.squeeze(0).to(device=self.device, dtype=torch.float32) / 255.0

        # Estimate depth
        depth = self._depth.estimate(frame_f)  # (H_cam, W_cam)

        # Temporal smoothing (on raw depth before warping)
        if temporal_smoothing > 0 and self._prev_depth is not None:
            if self._prev_depth.shape == depth.shape:
                depth = (
                    temporal_smoothing * self._prev_depth
                    + (1 - temporal_smoothing) * depth
                )
        self._prev_depth = depth.clone()

        # Warp depth to projector perspective (always depth_warp_ai mode)
        has_calib = self._map_x is not None and self._map_y is not None

        if has_calib:
            # VACE defaults: grayscale, no CLAHE, full range, no gamma
            rgb = build_warped_depth_image(
                depth,
                self._map_x,
                self._map_y,
                self._proj_valid_mask,
                self.proj_h,
                self.proj_w,
                clip_lo=0.0,
                clip_hi=100.0,
                brightness=0.0,
                contrast=1.0,
                gamma=1.0,
                equalize=False,
                invert=False,
                blur=depth_blur,
                colormap="grayscale",
                device=self.device,
            )
        else:
            # No calibration — output camera-space depth as grayscale
            depth_np = depth.cpu().numpy()
            # Simple min-max normalize
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max - d_min > 1e-6:
                depth_norm = (depth_np - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_np)

            if depth_blur > 0.5:
                ksize = int(depth_blur) * 2 + 1
                depth_norm = cv2.GaussianBlur(depth_norm, (ksize, ksize), 0)

            depth_uint8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)
            depth_bgr = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
            depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
            rgb = torch.from_numpy(
                depth_rgb.astype(np.float32) / 255.0
            ).to(self.device)

        # Resize to generation resolution if needed
        gen_w, gen_h = self._get_generation_resolution()
        h, w = rgb.shape[:2]
        if (w, h) != (gen_w, gen_h):
            rgb_np = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            rgb_np = cv2.resize(rgb_np, (gen_w, gen_h), interpolation=cv2.INTER_AREA)
            rgb = torch.from_numpy(rgb_np.astype(np.float32) / 255.0).to(self.device)

        # Output as (1, H, W, 3) in [0, 1]
        output = rgb.unsqueeze(0).clamp(0, 1)
        return {"video": output}

    def __del__(self) -> None:
        # Don't stop the shared streamer — other pipelines may still use it
        pass


# =============================================================================
# Projector output postprocessor
# =============================================================================


class ProMapAnythingProjectorPipeline(Pipeline):
    """Postprocessor that streams the final pipeline output to a projector.

    Receives the finished frame from the main pipeline (e.g. Krea) and
    forwards it to the shared MJPEG streamer for the companion app.
    The video tensor is passed through unmodified.

    The stream starts automatically when this postprocessor is instantiated.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ProMapAnythingProjectorConfig

    def __init__(self, device=None, **kwargs):
        self.device = device or torch.device("cpu")

        # Auto-start the shared streamer
        port = kwargs.get("stream_port", 8765)
        self._streamer = get_or_create_streamer(port)
        logger.info(
            "Projector postprocessor: streaming on port %d", port
        )

    def _get_projector_resolution(self) -> tuple[int, int] | None:
        """Get projector resolution from companion app config, or None."""
        if self._streamer is None:
            return None
        cfg = self._streamer.client_config
        if cfg and "width" in cfg and "height" in cfg:
            return int(cfg["width"]), int(cfg["height"])
        return None

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Projector postprocessor received no video input")

        # Normalize input to (H, W, C) float32 [0, 1]
        frame = video[0].squeeze(0).to(device=self.device, dtype=torch.float32)
        if frame.max() > 1.5:
            frame = frame / 255.0

        output = frame.unsqueeze(0).clamp(0, 1)  # (1, H, W, C)

        # Submit to shared streamer (suppressed during calibration)
        if self._streamer is not None and self._streamer.is_running:
            rgb_np = (frame.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

            # Upscale to projector resolution if requested
            upscale = kwargs.get("upscale_to_projector", True)
            if upscale:
                proj_res = self._get_projector_resolution()
                if proj_res is not None:
                    proj_w, proj_h = proj_res
                    h, w = rgb_np.shape[:2]
                    if (w, h) != (proj_w, proj_h):
                        rgb_np = cv2.resize(
                            rgb_np, (proj_w, proj_h),
                            interpolation=cv2.INTER_LANCZOS4,
                        )

            self._streamer.submit_frame(rgb_np)

        return {"video": output}

    def __del__(self) -> None:
        # Don't stop the shared streamer — other pipelines may still use it
        pass
