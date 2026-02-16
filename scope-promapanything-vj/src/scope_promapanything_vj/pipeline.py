"""ProMapAnything VJ Tools — pipelines.

Registered pipelines:
1. ProMapAnythingCalibratePipeline  — main pipeline (Gray code calibration)
2. ProMapAnythingPipeline           — preprocessor (depth -> ControlNet)
3. ProMapAnythingProjectorPipeline  — postprocessor (streams output to projector)
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
from .frame_server import FrameStreamer, get_or_create_streamer
from .remap import build_warped_depth_image
from .schema import (
    ProMapAnythingCalibrateConfig,
    ProMapAnythingConfig,
    ProMapAnythingProjectorConfig,
)

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)

_DEFAULT_CALIBRATION_PATH = Path.home() / ".promapanything_calibration.json"


# =============================================================================
# Calibration main pipeline
# =============================================================================


class ProMapAnythingCalibratePipeline(Pipeline):
    """Gray code structured light calibration.

    Select as the main pipeline, hit play. The Scope viewer shows the
    camera feed — position it on the projector. Then toggle
    **Start Calibration** to begin projecting patterns. When done, the
    calibration saves to ``~/.promapanything_calibration.json``.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ProMapAnythingCalibrateConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.proj_w: int = kwargs.get("projector_width", 1920)
        self.proj_h: int = kwargs.get("projector_height", 1080)

        # Start the MJPEG streamer for the projector pop-out window
        port = kwargs.get("stream_port", 8765)
        self._streamer = get_or_create_streamer(port)

        # Calibration state — created lazily when start_calibration is toggled
        self._calib: CalibrationState | None = None
        self._calibrating = False
        self._done = False

        # Log the control panel URL
        import os
        pod_id = os.environ.get("RUNPOD_POD_ID", "")
        if pod_id:
            url = f"https://{pod_id}-{port}.proxy.runpod.net/"
        else:
            url = f"http://localhost:{port}/"
        logger.info(
            "Calibration pipeline ready: %dx%d projector — "
            "open %s for projector pop-out",
            self.proj_w, self.proj_h, url,
        )

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Calibration pipeline requires video input")

        frame = video[0]  # (1, H, W, C) [0, 255]
        start = kwargs.get("start_calibration", False)

        # -- Not started yet: pass through camera feed -----------------------
        if not start and not self._calibrating:
            self._done = False
            out = frame.squeeze(0).to(dtype=torch.float32)
            if out.max() > 1.5:
                out = out / 255.0
            # Send camera feed to streamer (projector pop-out shows camera)
            self._submit_to_streamer(out)
            return {"video": out.unsqueeze(0).clamp(0, 1)}

        # -- Start calibration on first toggle ON ----------------------------
        if start and not self._calibrating and not self._done:
            self._calib = CalibrationState(self.proj_w, self.proj_h)
            self._calib.start()
            self._calibrating = True
            logger.info(
                "Calibration started (%d patterns)", self._calib.total_patterns
            )

        # -- Step calibration ------------------------------------------------
        if self._calibrating and self._calib is not None:
            pattern = self._calib.step(frame, self.device)

            if self._calib.phase == CalibrationPhase.DONE:
                mapping = self._calib.get_mapping()
                if mapping is not None:
                    map_x, map_y = mapping
                    save_calibration(
                        map_x, map_y, _DEFAULT_CALIBRATION_PATH,
                        self.proj_w, self.proj_h,
                    )
                    logger.info(
                        "Calibration complete — saved to %s",
                        _DEFAULT_CALIBRATION_PATH,
                    )
                else:
                    logger.warning("Calibration finished but mapping was None")
                self._calibrating = False
                self._done = True
                # Fall through to camera passthrough

            elif pattern is not None:
                # Output the pattern — Scope displays it in the viewer
                t = pattern.squeeze(0) if pattern.ndim == 4 else pattern
                if t.max() > 1.5:
                    t = t / 255.0
                # Also send pattern to the streamer (projector pop-out)
                self._submit_calibration_to_streamer(t)
                return {"video": t.unsqueeze(0).clamp(0, 1)}

        # -- Done or toggled off: camera passthrough -------------------------
        if not start:
            self._calibrating = False
            self._done = False
            self._calib = None

        if self._streamer is not None:
            self._streamer.calibration_active = False

        out = frame.squeeze(0).to(dtype=torch.float32)
        if out.max() > 1.5:
            out = out / 255.0
        self._submit_to_streamer(out)
        return {"video": out.unsqueeze(0).clamp(0, 1)}

    def _submit_to_streamer(self, frame_01: torch.Tensor) -> None:
        """Send a [0,1] float tensor to the MJPEG streamer."""
        if self._streamer is not None and self._streamer.is_running:
            rgb_np = (frame_01.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            self._streamer.submit_frame(rgb_np)

    def _submit_calibration_to_streamer(self, frame_01: torch.Tensor) -> None:
        """Send a calibration pattern to the streamer (bypasses suppression)."""
        if self._streamer is not None and self._streamer.is_running:
            self._streamer.calibration_active = True
            rgb_np = (frame_01.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            self._streamer.submit_calibration_frame(rgb_np)


# =============================================================================
# Depth preprocessor
# =============================================================================


class ProMapAnythingPipeline(Pipeline):
    """Depth estimation + projector warp preprocessor.

    Estimates depth from the camera, warps it to the projector's perspective
    using the saved calibration, and outputs a VACE-optimized grayscale
    depth map.  No calibration code — use the Calibrate pipeline for that.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ProMapAnythingConfig

    _RESOLUTION_SCALES = {"quarter": 0.25, "half": 0.5, "native": 1.0}

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._gen_res: str = kwargs.get("generation_resolution", "half")
        calibration_file: str = kwargs.get("calibration_file", "")

        # Depth estimation (hardcoded auto/small)
        from .depth_provider import create_depth_provider

        self._depth = create_depth_provider("auto", self.device, model_size="small")

        # Calibration mapping
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self._proj_valid_mask: np.ndarray | None = None
        self.proj_w: int = 1920
        self.proj_h: int = 1080

        # Temporal smoothing buffer
        self._prev_depth: torch.Tensor | None = None

        # Load calibration (explicit path or default)
        cal_path = Path(calibration_file) if calibration_file else _DEFAULT_CALIBRATION_PATH
        if cal_path.is_file():
            mx, my, pw, ph = load_calibration(cal_path)
            self._map_x = mx
            self._map_y = my
            self.proj_w = pw
            self.proj_h = ph
            logger.info("Loaded calibration from %s (%dx%d)", cal_path, pw, ph)
        else:
            logger.warning(
                "No calibration found at %s — output will be un-warped camera depth",
                cal_path,
            )

    def _get_generation_resolution(self) -> tuple[int, int]:
        """Compute generation resolution from projector res + preset."""
        scale = self._RESOLUTION_SCALES.get(self._gen_res, 0.5)
        gen_w = max(64, round(self.proj_w * scale / 8) * 8)
        gen_h = max(64, round(self.proj_h * scale / 8) * 8)
        return gen_w, gen_h

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Depth preprocessor requires video input")

        frame = video[0]  # (1, H, W, C) [0, 255]

        temporal_smoothing = kwargs.get("temporal_smoothing", 0.5)
        depth_blur = kwargs.get("depth_blur", 0.0)

        # Normalise input to [0, 1]
        frame_f = frame.squeeze(0).to(device=self.device, dtype=torch.float32) / 255.0

        # Estimate depth
        depth = self._depth.estimate(frame_f)  # (H_cam, W_cam)

        # Temporal smoothing
        if temporal_smoothing > 0 and self._prev_depth is not None:
            if self._prev_depth.shape == depth.shape:
                depth = (
                    temporal_smoothing * self._prev_depth
                    + (1 - temporal_smoothing) * depth
                )
        self._prev_depth = depth.clone()

        # Warp to projector perspective
        has_calib = self._map_x is not None and self._map_y is not None

        if has_calib:
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
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max - d_min > 1e-6:
                depth_norm = (depth_np - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_np)

            if depth_blur > 0.5:
                ksize = int(depth_blur) * 2 + 1
                depth_norm = cv2.GaussianBlur(depth_norm, (ksize, ksize), 0)

            depth_uint8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)
            rgb = torch.from_numpy(
                cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            ).to(self.device)

        # Resize to generation resolution
        gen_w, gen_h = self._get_generation_resolution()
        h, w = rgb.shape[:2]
        if (w, h) != (gen_w, gen_h):
            rgb_np = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            rgb_np = cv2.resize(rgb_np, (gen_w, gen_h), interpolation=cv2.INTER_AREA)
            rgb = torch.from_numpy(rgb_np.astype(np.float32) / 255.0).to(self.device)

        return {"video": rgb.unsqueeze(0).clamp(0, 1)}


# =============================================================================
# Projector output postprocessor
# =============================================================================


class ProMapAnythingProjectorPipeline(Pipeline):
    """Streams the final pipeline output to a projector via MJPEG."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ProMapAnythingProjectorConfig

    def __init__(self, device=None, **kwargs):
        self.device = device or torch.device("cpu")

        port = kwargs.get("stream_port", 8765)
        self._streamer = get_or_create_streamer(port)
        logger.info("Projector postprocessor: streaming on port %d", port)

    def _get_projector_resolution(self) -> tuple[int, int] | None:
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

        frame = video[0].squeeze(0).to(device=self.device, dtype=torch.float32)
        if frame.max() > 1.5:
            frame = frame / 255.0

        output = frame.unsqueeze(0).clamp(0, 1)

        if self._streamer is not None and self._streamer.is_running:
            rgb_np = (frame.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

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
