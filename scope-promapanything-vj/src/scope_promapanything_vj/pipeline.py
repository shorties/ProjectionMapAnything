"""ProMapAnything VJ Tools — pipelines.

Registered pipelines:
1. ProMapAnythingPipeline          — preprocessor (calibration + depth -> ControlNet)
2. ProMapAnythingProjectorPipeline — postprocessor (streams final output to projector)

Internal (not registered):
- ProMapAnythingPreviewPipeline  — preview variant of preprocessor
- ProMapAnythingEffectsPipeline  — standalone VJ effects
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .calibration import (
    CalibrationPhase,
    CalibrationState,
    load_calibration,
    save_calibration,
)
from .depth_provider import DepthProvider, create_depth_provider
from .frame_server import FrameStreamer
from .projector_output import ProjectorOutput
from .remap import (
    apply_depth_adjustments,
    apply_depth_output_settings,
    build_warped_depth_image,
    warp_frame_to_projector,
)
from .schema import (
    ProMapAnythingConfig,
    ProMapAnythingEffectsConfig,
    ProMapAnythingPreviewConfig,
    ProMapAnythingProjectorConfig,
)

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)

_LIVE_DIR = Path.home() / ".promapanything_live"


class ExternalAppReader:
    """Reads live depth/color feeds from the ProMapAnything standalone app.

    The app writes numpy arrays to ~/.promapanything_live/ with a meta.json
    that indicates which feeds are available and the timestamp.
    """

    STALE_THRESHOLD = 5.0  # seconds before we consider the app disconnected

    def __init__(self):
        self._last_meta: dict | None = None
        self._cached_feeds: dict[str, np.ndarray] = {}
        self._last_check = 0.0

    def is_available(self) -> bool:
        """Check if the standalone app is running and exporting data."""
        meta = self._read_meta()
        if meta is None:
            return False
        age = time.time() - meta.get("timestamp", 0)
        return age < self.STALE_THRESHOLD

    def get_available_feeds(self) -> list[str]:
        """Return list of available feed names."""
        meta = self._read_meta()
        if meta is None:
            return []
        feeds = meta.get("feeds", {})
        return [k for k, v in feeds.items() if v]

    def read_feed(self, feed_name: str) -> np.ndarray | None:
        """Read a specific feed from the shared directory."""
        path = _LIVE_DIR / f"{feed_name}.npy"
        try:
            if path.exists():
                return np.load(str(path), allow_pickle=False)
        except Exception as e:
            logger.debug("Failed to read feed %s: %s", feed_name, e)
        return None

    def get_status_text(self) -> str:
        """Human-readable status string."""
        meta = self._read_meta()
        if meta is None:
            return "App not detected"
        age = time.time() - meta.get("timestamp", 0)
        if age > self.STALE_THRESHOLD:
            return f"App stale ({age:.0f}s ago)"
        feeds = self.get_available_feeds()
        return f"Connected — feeds: {', '.join(feeds)}"

    def _read_meta(self) -> dict | None:
        now = time.monotonic()
        if now - self._last_check < 0.5:
            return self._last_meta
        self._last_check = now
        meta_path = _LIVE_DIR / "meta.json"
        try:
            if meta_path.exists():
                self._last_meta = json.loads(meta_path.read_text())
                return self._last_meta
        except Exception:
            pass
        self._last_meta = None
        return None


# =============================================================================
# Calibration + Depth pipeline
# =============================================================================


class ProMapAnythingPipeline(Pipeline):
    """Projection mapping depth preprocessor.

    In **calibration mode**, projects Gray code patterns and captures the
    camera response to build a projector-camera pixel mapping.

    In **normal mode**, estimates depth from the camera feed, warps it to the
    projector's perspective using the calibration mapping, and outputs the
    result as a depth map image for ControlNet conditioning.

    Uses the depth-as-image pipeline: creates a processed depth image in
    camera space first (with CLAHE, percentile clip, gamma), then warps it
    identically to how RGB is warped for consistent results.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ProMapAnythingConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load-time parameters
        self._depth_source: str = kwargs.get("depth_source", "camera")
        depth_mode: str = kwargs.get("depth_provider", "auto")
        model_size: str = kwargs.get("depth_model_size", "small")
        self.proj_w: int = kwargs.get("projector_width", 1920)
        self.proj_h: int = kwargs.get("projector_height", 1080)
        calibration_file: str = kwargs.get("calibration_file", "")

        # External app reader (always created — cheap, checks lazily)
        self._ext_reader = ExternalAppReader()

        # Depth estimation (skip if using external app)
        self._depth: DepthProvider | None = None
        if self._depth_source == "camera":
            self._depth = create_depth_provider(
                depth_mode, self.device, model_size=model_size
            )

        # Calibration state
        self._calib: CalibrationState | None = None
        self._was_calibrating = False

        # Calibration mapping (populated by calibration or loaded from file)
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self._proj_valid_mask: np.ndarray | None = None

        # Temporal smoothing buffer
        self._prev_depth: torch.Tensor | None = None

        # Projector output window
        self._projector: ProjectorOutput | None = None
        self._projector_enabled = False
        self._projector_monitor = 1

        # Projector stream (remote MJPEG)
        self._streamer: FrameStreamer | None = None
        self._stream_enabled = False
        self._stream_port = 8765

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

        if self._depth_source == "external_app":
            logger.info("Depth source: external app (will auto-detect)")
        else:
            logger.info("Depth source: camera + depth estimation (%s)", model_size)

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
        settle_frames = kwargs.get("calibration_settle_frames", 3)
        capture_frames = kwargs.get("capture_frames", 3)
        decode_threshold = kwargs.get("decode_threshold", 20.0)
        bit_threshold = kwargs.get("bit_threshold", 3.0)
        depth_scale = kwargs.get("depth_scale", 1.0)
        depth_offset = kwargs.get("depth_offset", 0.0)
        depth_blur = kwargs.get("depth_blur", 0.0)
        depth_invert = kwargs.get("depth_invert", False)
        colormap = kwargs.get("colormap", "grayscale")
        temporal_smoothing = kwargs.get("temporal_smoothing", 0.5)
        feed_name = kwargs.get("external_app_feed", "depth_bw")

        # ControlNet output settings
        depth_equalize = kwargs.get("depth_equalize", True)
        depth_clip_lo = kwargs.get("depth_clip_lo", 2.0)
        depth_clip_hi = kwargs.get("depth_clip_hi", 98.0)
        depth_gamma = kwargs.get("depth_gamma", 1.0)

        # Pipeline mode
        pipeline_mode = kwargs.get("pipeline_mode", "depth_warp_ai")

        # -- External app mode ------------------------------------------------
        if self._depth_source == "external_app":
            result = self._call_external_app(
                frame, feed_name, depth_scale, depth_offset,
                depth_blur, depth_invert, colormap, temporal_smoothing,
            )
            self._update_projector(kwargs, result["video"])
            return result

        # -- Calibration mode -------------------------------------------------
        if calibrate:
            if not self._was_calibrating:
                self._calib = CalibrationState(
                    self.proj_w, self.proj_h,
                    settle_frames=settle_frames,
                    capture_frames=capture_frames,
                    decode_threshold=decode_threshold,
                    bit_threshold=bit_threshold,
                )
                self._calib.start()
                self._was_calibrating = True
                logger.info(
                    "Calibration started (%d patterns, %d frames each)",
                    self._calib.total_patterns, capture_frames,
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
                # Fall through to normal depth processing

            elif pattern is not None:
                # Output the calibration pattern
                result = {"video": pattern}
                self._update_projector(kwargs, pattern)
                return result

        else:
            if self._was_calibrating:
                self._was_calibrating = False

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

        # Pipeline mode determines whether we warp depth now or later
        has_calib = self._map_x is not None and self._map_y is not None

        if has_calib and pipeline_mode == "depth_warp_ai":
            # Default: Depth -> Warp -> AI
            # Warp depth to projector perspective, AI generates in projector space
            rgb = build_warped_depth_image(
                depth,
                self._map_x,
                self._map_y,
                self._proj_valid_mask,
                self.proj_h,
                self.proj_w,
                clip_lo=depth_clip_lo,
                clip_hi=depth_clip_hi,
                brightness=depth_offset,
                contrast=depth_scale,
                gamma=depth_gamma,
                equalize=depth_equalize,
                invert=depth_invert,
                blur=depth_blur,
                colormap=colormap,
                device=self.device,
            )
        else:
            # depth_ai_warp mode OR no calibration:
            # Output camera-space depth (with output settings but no warp).
            # The effects pipeline will warp the AI output later.
            import cv2

            depth_np = depth.cpu().numpy()
            depth_img = apply_depth_output_settings(
                depth_np,
                clip_lo=depth_clip_lo,
                clip_hi=depth_clip_hi,
                brightness=depth_offset,
                contrast=depth_scale,
                gamma=depth_gamma,
                equalize=depth_equalize,
            )
            if depth_invert:
                depth_img = 1.0 - depth_img
            if depth_blur > 0.5:
                ksize = int(depth_blur) * 2 + 1
                depth_img = cv2.GaussianBlur(depth_img, (ksize, ksize), 0)

            # Apply colormap
            depth_uint8 = (depth_img * 255).clip(0, 255).astype(np.uint8)
            if colormap == "grayscale":
                depth_bgr = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
            else:
                cv_maps = {
                    "turbo": cv2.COLORMAP_TURBO,
                    "viridis": cv2.COLORMAP_VIRIDIS,
                    "magma": cv2.COLORMAP_MAGMA,
                }
                cv_id = cv_maps.get(colormap, cv2.COLORMAP_TURBO)
                depth_bgr = cv2.applyColorMap(depth_uint8, cv_id)
            depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
            rgb = torch.from_numpy(
                depth_rgb.astype(np.float32) / 255.0
            ).to(self.device)

        # Output as (1, H, W, 3) in [0, 1]
        output = rgb.unsqueeze(0).clamp(0, 1)
        self._update_projector(kwargs, output)
        return {"video": output}

    def _call_external_app(
        self, frame, feed_name, depth_scale, depth_offset,
        depth_blur, depth_invert, colormap, temporal_smoothing,
    ) -> dict:
        """Read depth/color from the standalone app's shared files."""
        reader = self._ext_reader

        if not reader.is_available():
            # App not running — pass through the camera frame as-is
            logger.debug("External app not detected, passing through camera")
            frame_f = frame.squeeze(0).to(
                device=self.device, dtype=torch.float32
            ) / 255.0
            return {"video": frame_f.unsqueeze(0).clamp(0, 1)}

        # Read the requested feed
        data = reader.read_feed(feed_name)
        if data is None:
            frame_f = frame.squeeze(0).to(
                device=self.device, dtype=torch.float32
            ) / 255.0
            return {"video": frame_f.unsqueeze(0).clamp(0, 1)}

        # Convert to torch tensor
        if data.ndim == 2:
            # B&W depth (H, W) float32 [0,1]
            depth = torch.from_numpy(data).to(device=self.device, dtype=torch.float32)

            # Temporal smoothing
            if temporal_smoothing > 0 and self._prev_depth is not None:
                if self._prev_depth.shape == depth.shape:
                    depth = (
                        temporal_smoothing * self._prev_depth
                        + (1 - temporal_smoothing) * depth
                    )
            self._prev_depth = depth.clone()

            # Apply adjustments and colormap
            rgb = apply_depth_adjustments(
                depth,
                scale=depth_scale,
                offset=depth_offset,
                blur=depth_blur,
                invert=depth_invert,
                colormap=colormap,
                device=self.device,
            )
        else:
            # RGB feed (H, W, 3) uint8 — depth_color or projector_rgb
            rgb = (
                torch.from_numpy(data).to(device=self.device, dtype=torch.float32)
                / 255.0
            )

        return {"video": rgb.unsqueeze(0).clamp(0, 1)}

    def _update_projector(self, kwargs: dict, output_tensor: torch.Tensor) -> None:
        """Manage the projector output window and stream server, submit frames."""
        enabled = kwargs.get("projector_output", False)
        monitor = kwargs.get("projector_monitor", 1)
        stream_enabled = kwargs.get("projector_stream", False)
        stream_port = kwargs.get("projector_stream_port", 8765)

        # -- Local GLFW window ------------------------------------------------
        if enabled and not self._projector_enabled:
            self._projector = ProjectorOutput()
            self._projector.start(monitor_index=monitor)
            self._projector_enabled = True
            self._projector_monitor = monitor
            logger.info("Projector output started on monitor %d", monitor)
        elif not enabled and self._projector_enabled:
            if self._projector is not None:
                self._projector.stop()
                self._projector = None
            self._projector_enabled = False
            logger.info("Projector output stopped")
        elif enabled and monitor != self._projector_monitor:
            if self._projector is not None:
                self._projector.stop()
            self._projector = ProjectorOutput()
            self._projector.start(monitor_index=monitor)
            self._projector_monitor = monitor
            logger.info("Projector output moved to monitor %d", monitor)

        # -- Remote MJPEG stream ----------------------------------------------
        if stream_enabled and not self._stream_enabled:
            self._streamer = FrameStreamer(port=stream_port)
            self._streamer.start()
            self._stream_enabled = True
            self._stream_port = stream_port
        elif not stream_enabled and self._stream_enabled:
            if self._streamer is not None:
                self._streamer.stop()
                self._streamer = None
            self._stream_enabled = False
        elif stream_enabled and stream_port != self._stream_port:
            if self._streamer is not None:
                self._streamer.stop()
            self._streamer = FrameStreamer(port=stream_port)
            self._streamer.start()
            self._stream_port = stream_port

        # -- Submit frame to active outputs -----------------------------------
        has_output = (
            (self._projector is not None and self._projector.is_running)
            or (self._streamer is not None and self._streamer.is_running)
        )
        if has_output:
            t = output_tensor.squeeze(0) if output_tensor.ndim == 4 else output_tensor
            rgb_np = (t.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            if self._projector is not None and self._projector.is_running:
                self._projector.submit_frame(rgb_np)
            if self._streamer is not None and self._streamer.is_running:
                self._streamer.submit_frame(rgb_np)

    def __del__(self) -> None:
        if self._projector is not None:
            self._projector.stop()
            self._projector = None
        if self._streamer is not None:
            self._streamer.stop()
            self._streamer = None


class ProMapAnythingPreviewPipeline(ProMapAnythingPipeline):
    """Preview variant — outputs the depth map to screen."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ProMapAnythingPreviewConfig


# =============================================================================
# VJ Effects pipeline (standalone)
# =============================================================================


class ProMapAnythingEffectsPipeline(Pipeline):
    """VJ effects pipeline with optional external app input.

    Takes video input OR reads live depth/color from the ProMapAnything
    standalone app, then applies real-time animated effects.
    All effects are GPU-accelerated and controlled via live sliders.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ProMapAnythingEffectsConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._frame_count: int = 0

        # External app support
        self._depth_source: str = kwargs.get("depth_source", "video_input")
        self._ext_reader = ExternalAppReader()

        # Projector output window
        self._projector: ProjectorOutput | None = None
        self._projector_enabled = False
        self._projector_monitor = 1

        # Projector stream (remote MJPEG)
        self._streamer: FrameStreamer | None = None
        self._stream_enabled = False
        self._stream_port = 8765

        # Projector warp (calibration mapping for warping AI frames)
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None

        # Load calibration for projector warp
        calibration_file: str = kwargs.get("calibration_file", "")
        if calibration_file:
            cal_path = Path(calibration_file)
        else:
            cal_path = Path.home() / ".promapanything_calibration.json"
        if cal_path.is_file():
            from .calibration import load_calibration
            mx, my, _pw, _ph = load_calibration(cal_path)
            self._map_x = mx
            self._map_y = my
            logger.info("Effects pipeline: loaded calibration from %s", cal_path)

        if self._depth_source == "external_app":
            logger.info("Effects pipeline: reading from external app")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def _get_input_gray(self, kwargs) -> torch.Tensor:
        """Get the input as a (H, W) grayscale tensor in [0, 1]."""
        feed_name = kwargs.get("external_app_feed", "depth_bw")

        if self._depth_source == "external_app" and self._ext_reader.is_available():
            data = self._ext_reader.read_feed(feed_name)
            if data is not None:
                tensor = torch.from_numpy(data).to(
                    device=self.device, dtype=torch.float32
                )
                if tensor.ndim == 2:
                    return tensor
                else:
                    return (tensor / 255.0).mean(dim=-1)

        # Fallback: use Scope video input
        video = kwargs.get("video")
        if video is None:
            raise ValueError("No input available (no video and external app not detected)")

        frames = torch.stack([f.squeeze(0) for f in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0
        return frames[0].mean(dim=-1)  # (H, W)

    def __call__(self, **kwargs) -> dict:
        from .effects import (
            apply_depth_fog,
            apply_flow_warp,
            apply_geometry_edges,
            apply_kaleido,
            apply_noise_blend,
            apply_pulse,
            apply_radial_zoom,
            apply_shockwave,
            apply_wave_warp,
            apply_wobble,
        )

        self._frame_count += 1
        t = self._frame_count / 30.0

        # -- Projector warp (Depth -> AI -> Warp mode) ------------------------
        # When projector_warp is enabled, warp the incoming video frame to
        # projector perspective before extracting grayscale for effects.
        projector_warp = kwargs.get("projector_warp", False)
        has_calib = self._map_x is not None and self._map_y is not None

        if projector_warp and has_calib:
            # Get the input as RGB tensor first, then warp
            video = kwargs.get("video")
            if video is not None:
                frame = video[0].squeeze(0).to(
                    device=self.device, dtype=torch.float32
                ) / 255.0  # (H, W, 3)
                warped = warp_frame_to_projector(
                    frame, self._map_x, self._map_y, device=self.device,
                )
                gray = warped.mean(dim=-1)  # (H, W)
            else:
                gray = self._get_input_gray(kwargs)
        else:
            gray = self._get_input_gray(kwargs)

        # -- Effects chain ----------------------------------------------------

        if kwargs.get("noise_enabled", False):
            gray = apply_noise_blend(
                gray,
                intensity=kwargs.get("noise_intensity", 0.3),
                scale=kwargs.get("noise_scale", 4.0),
                octaves=kwargs.get("noise_octaves", 4),
                speed=kwargs.get("noise_speed", 0.5),
                time=t,
            )

        if kwargs.get("flow_enabled", False):
            gray = apply_flow_warp(
                gray,
                intensity=kwargs.get("flow_intensity", 0.15),
                scale=kwargs.get("flow_scale", 3.0),
                speed=kwargs.get("flow_speed", 0.3),
                time=t,
            )

        if kwargs.get("pulse_enabled", False):
            gray = apply_pulse(
                gray,
                speed=kwargs.get("pulse_speed", 0.5),
                amount=kwargs.get("pulse_amount", 0.3),
                time=t,
            )

        if kwargs.get("wave_enabled", False):
            gray = apply_wave_warp(
                gray,
                frequency=kwargs.get("wave_frequency", 3.0),
                amplitude=kwargs.get("wave_amplitude", 0.05),
                speed=kwargs.get("wave_speed", 1.0),
                direction=kwargs.get("wave_direction", 0.0),
                time=t,
            )

        if kwargs.get("kaleido_enabled", False):
            gray = apply_kaleido(
                gray,
                segments=kwargs.get("kaleido_segments", 6),
                rotation=kwargs.get("kaleido_rotation", 0.0),
                time=t * kwargs.get("kaleido_spin_speed", 0.0),
            )

        if kwargs.get("shockwave_enabled", False):
            gray = apply_shockwave(
                gray,
                origin_x=kwargs.get("shockwave_origin_x", 0.5),
                origin_y=kwargs.get("shockwave_origin_y", 0.5),
                speed=kwargs.get("shockwave_speed", 0.8),
                thickness=kwargs.get("shockwave_thickness", 0.15),
                strength=kwargs.get("shockwave_strength", 0.4),
                decay=kwargs.get("shockwave_decay", 1.5),
                time=t,
                auto_trigger_interval=kwargs.get("shockwave_interval", 2.0),
            )

        if kwargs.get("wobble_enabled", False):
            gray = apply_wobble(
                gray,
                intensity=kwargs.get("wobble_intensity", 0.08),
                speed=kwargs.get("wobble_speed", 1.0),
                time=t,
            )

        if kwargs.get("edges_enabled", False):
            gray = apply_geometry_edges(
                gray,
                edge_strength=kwargs.get("edges_strength", 0.5),
                glow_width=kwargs.get("edges_glow_width", 3.0),
                pulse_speed=kwargs.get("edges_pulse_speed", 0.0),
                time=t,
            )

        if kwargs.get("fog_enabled", False):
            gray = apply_depth_fog(
                gray,
                density=kwargs.get("fog_density", 0.6),
                near=kwargs.get("fog_near", 0.3),
                far=kwargs.get("fog_far", 0.9),
                animated=kwargs.get("fog_animated", True),
                speed=kwargs.get("fog_speed", 0.5),
                time=t,
            )

        if kwargs.get("zoom_enabled", False):
            gray = apply_radial_zoom(
                gray,
                origin_x=kwargs.get("zoom_origin_x", 0.5),
                origin_y=kwargs.get("zoom_origin_y", 0.5),
                strength=kwargs.get("zoom_strength", 0.15),
                speed=kwargs.get("zoom_speed", 0.5),
                time=t,
            )

        # -- Output -----------------------------------------------------------

        gray = gray.clamp(0, 1)

        # Expand back to RGB
        rgb = gray.unsqueeze(-1).expand(-1, -1, 3)  # (H, W, 3)

        output = rgb.unsqueeze(0).clamp(0, 1)
        self._update_projector(kwargs, output)
        return {"video": output}

    def _update_projector(self, kwargs: dict, output_tensor: torch.Tensor) -> None:
        """Manage the projector output window and stream server, submit frames."""
        enabled = kwargs.get("projector_output", False)
        monitor = kwargs.get("projector_monitor", 1)
        stream_enabled = kwargs.get("projector_stream", False)
        stream_port = kwargs.get("projector_stream_port", 8765)

        # -- Local GLFW window ------------------------------------------------
        if enabled and not self._projector_enabled:
            self._projector = ProjectorOutput()
            self._projector.start(monitor_index=monitor)
            self._projector_enabled = True
            self._projector_monitor = monitor
            logger.info("Effects projector output started on monitor %d", monitor)
        elif not enabled and self._projector_enabled:
            if self._projector is not None:
                self._projector.stop()
                self._projector = None
            self._projector_enabled = False
            logger.info("Effects projector output stopped")
        elif enabled and monitor != self._projector_monitor:
            if self._projector is not None:
                self._projector.stop()
            self._projector = ProjectorOutput()
            self._projector.start(monitor_index=monitor)
            self._projector_monitor = monitor
            logger.info("Effects projector output moved to monitor %d", monitor)

        # -- Remote MJPEG stream ----------------------------------------------
        if stream_enabled and not self._stream_enabled:
            self._streamer = FrameStreamer(port=stream_port)
            self._streamer.start()
            self._stream_enabled = True
            self._stream_port = stream_port
        elif not stream_enabled and self._stream_enabled:
            if self._streamer is not None:
                self._streamer.stop()
                self._streamer = None
            self._stream_enabled = False
        elif stream_enabled and stream_port != self._stream_port:
            if self._streamer is not None:
                self._streamer.stop()
            self._streamer = FrameStreamer(port=stream_port)
            self._streamer.start()
            self._stream_port = stream_port

        # -- Submit frame to active outputs -----------------------------------
        has_output = (
            (self._projector is not None and self._projector.is_running)
            or (self._streamer is not None and self._streamer.is_running)
        )
        if has_output:
            t = output_tensor.squeeze(0) if output_tensor.ndim == 4 else output_tensor
            rgb_np = (t.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            if self._projector is not None and self._projector.is_running:
                self._projector.submit_frame(rgb_np)
            if self._streamer is not None and self._streamer.is_running:
                self._streamer.submit_frame(rgb_np)

    def __del__(self) -> None:  # Effects pipeline cleanup
        if self._projector is not None:
            self._projector.stop()
            self._projector = None
        if self._streamer is not None:
            self._streamer.stop()
            self._streamer = None


# =============================================================================
# Projector output postprocessor
# =============================================================================


class ProMapAnythingProjectorPipeline(Pipeline):
    """Postprocessor that streams the final pipeline output to a projector.

    Receives the finished frame from the main pipeline (e.g. Krea), optionally
    warps it to projector perspective, and forwards it to a local GLFW window
    and/or an MJPEG stream for a remote companion app.  The video tensor is
    passed through unmodified (or warped) so downstream pipelines still work.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ProMapAnythingProjectorConfig

    def __init__(self, device=None, **kwargs):
        self.device = device or torch.device("cpu")

        # Projector window (local GLFW)
        self._projector: ProjectorOutput | None = None
        self._projector_enabled = False
        self._projector_monitor = 1

        # MJPEG stream (remote)
        self._streamer: FrameStreamer | None = None
        self._stream_enabled = False
        self._stream_port = 8765

        # Calibration mapping for projector warp
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None

        calibration_file: str = kwargs.get("calibration_file", "")
        if calibration_file:
            cal_path = Path(calibration_file)
        else:
            cal_path = Path.home() / ".promapanything_calibration.json"
        if cal_path.is_file():
            mx, my, _pw, _ph = load_calibration(cal_path)
            self._map_x = mx
            self._map_y = my
            logger.info("Projector postprocessor: loaded calibration from %s", cal_path)

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

        # Optional projector warp
        projector_warp = kwargs.get("projector_warp", False)
        if projector_warp and self._map_x is not None and self._map_y is not None:
            frame = warp_frame_to_projector(
                frame, self._map_x, self._map_y, device=self.device,
            )

        output = frame.unsqueeze(0).clamp(0, 1)  # (1, H, W, C)
        self._update_projector(kwargs, output)
        return {"video": output}

    def _update_projector(self, kwargs: dict, output_tensor: torch.Tensor) -> None:
        """Manage projector window + MJPEG stream, submit frames."""
        enabled = kwargs.get("projector_output", False)
        monitor = kwargs.get("projector_monitor", 1)
        stream_enabled = kwargs.get("projector_stream", False)
        stream_port = kwargs.get("projector_stream_port", 8765)

        # -- Local GLFW window ------------------------------------------------
        if enabled and not self._projector_enabled:
            self._projector = ProjectorOutput()
            self._projector.start(monitor_index=monitor)
            self._projector_enabled = True
            self._projector_monitor = monitor
            logger.info("Projector postprocessor: output started on monitor %d", monitor)
        elif not enabled and self._projector_enabled:
            if self._projector is not None:
                self._projector.stop()
                self._projector = None
            self._projector_enabled = False
        elif enabled and monitor != self._projector_monitor:
            if self._projector is not None:
                self._projector.stop()
            self._projector = ProjectorOutput()
            self._projector.start(monitor_index=monitor)
            self._projector_monitor = monitor

        # -- Remote MJPEG stream ----------------------------------------------
        if stream_enabled and not self._stream_enabled:
            self._streamer = FrameStreamer(port=stream_port)
            self._streamer.start()
            self._stream_enabled = True
            self._stream_port = stream_port
        elif not stream_enabled and self._stream_enabled:
            if self._streamer is not None:
                self._streamer.stop()
                self._streamer = None
            self._stream_enabled = False
        elif stream_enabled and stream_port != self._stream_port:
            if self._streamer is not None:
                self._streamer.stop()
            self._streamer = FrameStreamer(port=stream_port)
            self._streamer.start()
            self._stream_port = stream_port

        # -- Submit frame to active outputs -----------------------------------
        has_output = (
            (self._projector is not None and self._projector.is_running)
            or (self._streamer is not None and self._streamer.is_running)
        )
        if has_output:
            t = output_tensor.squeeze(0) if output_tensor.ndim == 4 else output_tensor
            rgb_np = (t.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            if self._projector is not None and self._projector.is_running:
                self._projector.submit_frame(rgb_np)
            if self._streamer is not None and self._streamer.is_running:
                self._streamer.submit_frame(rgb_np)

    def __del__(self) -> None:  # Projector postprocessor cleanup
        if self._projector is not None:
            self._projector.stop()
            self._projector = None
        if self._streamer is not None:
            self._streamer.stop()
            self._streamer = None
