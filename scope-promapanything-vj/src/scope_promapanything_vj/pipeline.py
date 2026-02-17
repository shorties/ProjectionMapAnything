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
# Ensure our logger output reaches stdout (Scope doesn't forward plugin loggers)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[ProMap Pipe] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

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
        self._settle_frames: int = kwargs.get("settle_frames", 15)
        self._capture_frames: int = kwargs.get("capture_frames", 3)

        # Start the MJPEG streamer for the projector pop-out window
        port = kwargs.get("stream_port", 8765)
        self._streamer = get_or_create_streamer(port)

        # Calibration state — created lazily when start_calibration is toggled
        self._calib: CalibrationState | None = None
        self._calibrating = False
        self._done = False
        self._reset_armed = False

        # Live depth preview (lazy-loaded)
        self._live_depth_provider = None
        self._live_map_x: np.ndarray | None = None
        self._live_map_y: np.ndarray | None = None
        self._live_proj_w: int = self.proj_w
        self._live_proj_h: int = self.proj_h

        # Build the grey test card (shown on projector before calibration)
        self._test_card = self._build_test_card()

        # Compute the projector URL (RunPod auto-detect)
        import os
        pod_id = os.environ.get("RUNPOD_POD_ID", "")
        if pod_id:
            self._projector_url = f"https://{pod_id}-{port}.proxy.runpod.net/"
        else:
            self._projector_url = f"http://localhost:{port}/"
        logger.info(
            "Calibration pipeline ready: %dx%d projector — "
            "open %s for projector pop-out",
            self.proj_w, self.proj_h, self._projector_url,
        )

        # Auto-open control panel in browser (local only, no-op on RunPod)
        if not pod_id:
            try:
                import webbrowser
                webbrowser.open(self._projector_url)
            except Exception:
                pass

        # Track whether we already opened the browser via the toggle
        self._browser_opened = not pod_id

    def _build_test_card(self) -> torch.Tensor:
        """Build a grey test card image at projector resolution.

        Shown on the projector before calibration starts to avoid
        camera-projector feedback loops.  Returns (H, W, 3) float32 [0,1].
        """
        card = np.full((self.proj_h, self.proj_w, 3), 128, dtype=np.uint8)

        # Thin white border
        cv2.rectangle(card, (2, 2), (self.proj_w - 3, self.proj_h - 3),
                       (200, 200, 200), 1)

        # Centre crosshair
        cx, cy = self.proj_w // 2, self.proj_h // 2
        arm = min(self.proj_w, self.proj_h) // 20
        cv2.line(card, (cx - arm, cy), (cx + arm, cy), (200, 200, 200), 1)
        cv2.line(card, (cx, cy - arm), (cx, cy + arm), (200, 200, 200), 1)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = min(self.proj_w, self.proj_h) / 800.0
        thick = max(1, int(scale * 2))

        msg1 = "READY TO CALIBRATE"
        sz1 = cv2.getTextSize(msg1, font, scale, thick)[0]
        cv2.putText(card, msg1,
                    (cx - sz1[0] // 2, cy - sz1[1] - 10),
                    font, scale, (220, 220, 220), thick, cv2.LINE_AA)

        msg2 = "Toggle Start Calibration to begin"
        scale2 = scale * 0.5
        thick2 = max(1, int(scale2 * 2))
        sz2 = cv2.getTextSize(msg2, font, scale2, thick2)[0]
        cv2.putText(card, msg2,
                    (cx - sz2[0] // 2, cy + sz2[1] + 20),
                    font, scale2, (180, 180, 180), thick2, cv2.LINE_AA)

        return torch.from_numpy(card.astype(np.float32) / 255.0).to(self.device)

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        try:
            return self._call_inner(**kwargs)
        except Exception as exc:
            logger.error("__call__ CRASHED: %s: %s", type(exc).__name__, exc, exc_info=True)
            # Return test card as fallback so pipeline doesn't go silent
            return {"video": self._test_card.unsqueeze(0).clamp(0, 1)}

    def _call_inner(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Calibration pipeline requires video input")

        frame = video[0]  # (1, H, W, C) [0, 255]
        start = kwargs.get("start_calibration", False)
        live_depth = kwargs.get("live_depth_preview", False)
        reset = kwargs.get("reset_calibration", False)

        # Update projector resolution from input-side fields
        new_pw = kwargs.get("projector_width", self.proj_w)
        new_ph = kwargs.get("projector_height", self.proj_h)
        if new_pw != self.proj_w or new_ph != self.proj_h:
            self.proj_w = new_pw
            self.proj_h = new_ph
            self._test_card = self._build_test_card()
            logger.info("Projector resolution updated to %dx%d", self.proj_w, self.proj_h)

        # Reset calibration on rising edge (toggled ON)
        if reset and not self._reset_armed:
            self._reset_armed = True
            self._calib = None
            self._calibrating = False
            self._done = False
            self._live_map_x = None
            self._live_map_y = None
            self._live_depth_provider = None
            if self._streamer is not None:
                self._streamer.clear_calibration_results()
                self._streamer.calibration_active = False
            logger.info("Calibration state reset — ready to recalibrate")
        elif not reset:
            self._reset_armed = False

        # "Open Dashboard" toggle — open browser on rising edge
        open_dash = kwargs.get("open_dashboard", False)
        if open_dash and not self._browser_opened:
            self._browser_opened = True
            try:
                import webbrowser
                webbrowser.open(self._projector_url)
            except Exception:
                pass
        elif not open_dash:
            self._browser_opened = False

        # -- Not started yet: show test card or live depth --------------------
        if not start and not self._calibrating:
            self._done = False
            if self._streamer is not None:
                self._streamer.calibration_active = False
                self._streamer._calibration_live_depth = live_depth

            if live_depth:
                depth_frame = self._get_live_depth_frame(frame)
                if depth_frame is not None:
                    self._submit_to_streamer(depth_frame)
                    return {"video": depth_frame.unsqueeze(0).clamp(0, 1)}

            # Send test card to streamer (projector shows neutral grey)
            self._submit_to_streamer(self._test_card)
            return {"video": self._test_card.unsqueeze(0).clamp(0, 1)}

        # -- Start calibration on first toggle ON ----------------------------
        if start and not self._calibrating and not self._done:
            self._calib = CalibrationState(
                self.proj_w, self.proj_h,
                settle_frames=self._settle_frames,
                capture_frames=self._capture_frames,
            )
            self._calib.start()
            self._calibrating = True
            if self._streamer is not None:
                self._streamer.clear_calibration_results()
                self._streamer._calibration_live_depth = False
            logger.info(
                "Calibration started (%d patterns)", self._calib.total_patterns
            )

        # -- Step calibration ------------------------------------------------
        if self._calibrating and self._calib is not None:
            pattern = self._calib.step(frame, self.device)

            # Update progress on streamer
            self._update_streamer_progress()

            if self._calib.phase == CalibrationPhase.DONE:
                mapping = self._calib.get_mapping()
                if mapping is not None:
                    map_x, map_y = mapping
                    logger.info("Saving calibration to %s ...", _DEFAULT_CALIBRATION_PATH)
                    save_calibration(
                        map_x, map_y, _DEFAULT_CALIBRATION_PATH,
                        self.proj_w, self.proj_h,
                    )
                    logger.info("Calibration saved.")

                    # Compute coverage %
                    coverage_pct = 0.0
                    if self._calib.proj_valid_mask is not None:
                        total = self._calib.proj_valid_mask.size
                        valid = np.count_nonzero(self._calib.proj_valid_mask)
                        coverage_pct = (valid / total) * 100.0 if total > 0 else 0.0

                    self._publish_calibration_results(
                        map_x, map_y, frame, coverage_pct,
                    )

                    # Final progress update
                    if self._streamer is not None:
                        self._streamer.update_calibration_progress(
                            1.0, "DONE", coverage_pct=coverage_pct,
                        )
                else:
                    logger.warning("Calibration finished but mapping was None")
                    if self._streamer is not None:
                        self._streamer.update_calibration_progress(
                            1.0, "DONE", errors=["Mapping was None — decode failed"],
                        )
                self._calibrating = False
                self._done = True
                # Fall through to test card

            elif pattern is not None:
                # Output the pattern — Scope displays it in the viewer
                t = pattern.squeeze(0) if pattern.ndim == 4 else pattern
                if t.max() > 1.5:
                    t = t / 255.0
                # Also send pattern to the streamer (projector pop-out)
                self._submit_calibration_to_streamer(t)
                return {"video": t.unsqueeze(0).clamp(0, 1)}

        # -- Done or toggled off: show test card again -------------------------
        if not start:
            self._calibrating = False
            self._done = False
            self._calib = None

        if self._streamer is not None:
            self._streamer.calibration_active = False

        self._submit_to_streamer(self._test_card)
        return {"video": self._test_card.unsqueeze(0).clamp(0, 1)}

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

    def _update_streamer_progress(self) -> None:
        """Push current calibration progress to the streamer for the dashboard."""
        if self._streamer is None or self._calib is None:
            return

        phase = self._calib.phase.name
        progress = self._calib.progress

        # Build pattern info string
        pattern_info = ""
        if self._calib.phase == CalibrationPhase.PATTERNS:
            idx = self._calib._pattern_index
            total_x = 2 * self._calib.bits_x
            if idx < total_x:
                bit = idx // 2 + 1
                pattern_info = f"bit {bit}/{self._calib.bits_x} X-axis"
            else:
                y_idx = idx - total_x
                bit = y_idx // 2 + 1
                pattern_info = f"bit {bit}/{self._calib.bits_y} Y-axis"

            captured = sum(len(s) for s in self._calib._captures)
            total = self._calib.total_patterns * self._calib.capture_frames
            pattern_info += f" ({captured}/{total} captures)"
        elif self._calib.phase == CalibrationPhase.WHITE:
            pattern_info = "Capturing white reference"
        elif self._calib.phase == CalibrationPhase.BLACK:
            pattern_info = "Capturing black reference"
        elif self._calib.phase == CalibrationPhase.DECODING:
            pattern_info = "Decoding patterns..."

        self._streamer.update_calibration_progress(progress, phase, pattern_info)

    def _get_live_depth_frame(self, camera_frame: torch.Tensor) -> torch.Tensor | None:
        """Estimate depth from camera and warp to projector perspective.

        Returns (H, W, 3) float32 [0,1] or None on failure.
        """
        # Lazy-load depth provider
        if self._live_depth_provider is None:
            try:
                from .depth_provider import create_depth_provider
                self._live_depth_provider = create_depth_provider(
                    "auto", self.device, model_size="small",
                )
                logger.info("Live depth preview: depth provider loaded")
            except Exception:
                logger.warning("Failed to load depth provider for live preview", exc_info=True)
                return None

        # Lazy-load calibration
        if self._live_map_x is None:
            if _DEFAULT_CALIBRATION_PATH.is_file():
                try:
                    mx, my, pw, ph, _ = load_calibration(_DEFAULT_CALIBRATION_PATH)
                    self._live_map_x = mx
                    self._live_map_y = my
                    self._live_proj_w = pw
                    self._live_proj_h = ph
                    logger.info("Live depth preview: calibration loaded (%dx%d)", pw, ph)
                except Exception:
                    logger.warning("Failed to load calibration for live preview", exc_info=True)
            else:
                logger.debug("No calibration file for live depth preview")

        # Estimate depth
        try:
            frame_f = camera_frame.squeeze(0).to(device=self.device, dtype=torch.float32)
            if frame_f.max() > 1.5:
                frame_f = frame_f / 255.0
            depth = self._live_depth_provider.estimate(frame_f)  # (H, W)
        except Exception:
            logger.warning("Depth estimation failed in live preview", exc_info=True)
            return None

        # Warp to projector perspective if calibration available
        if self._live_map_x is not None and self._live_map_y is not None:
            rgb = build_warped_depth_image(
                depth,
                self._live_map_x,
                self._live_map_y,
                None,
                self._live_proj_h,
                self._live_proj_w,
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
            depth_uint8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)
            rgb = torch.from_numpy(
                cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            ).to(self.device)

        return rgb

    def _publish_calibration_results(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        last_frame: torch.Tensor,
        coverage_pct: float = 0.0,
    ) -> None:
        """Generate calibration artifacts, push to streamer, and upload to Scope gallery."""
        from datetime import datetime, timezone

        logger.info("Publishing calibration results (coverage=%.1f%%) ...", coverage_pct)
        timestamp = datetime.now(timezone.utc).isoformat()
        files: dict[str, bytes] = {}

        # 1. Calibration JSON
        try:
            files["calibration.json"] = _DEFAULT_CALIBRATION_PATH.read_bytes()
        except Exception:
            logger.warning("Could not read calibration JSON for download")

        # 2. Coverage map — green where valid, black where inpainted
        if self._calib is not None and self._calib.proj_valid_mask is not None:
            coverage = np.zeros(
                (self.proj_h, self.proj_w, 3), dtype=np.uint8
            )
            coverage[self._calib.proj_valid_mask] = [0, 200, 100]
            ok, buf = cv2.imencode(".png", cv2.cvtColor(coverage, cv2.COLOR_RGB2BGR))
            if ok:
                files["coverage_map.png"] = buf.tobytes()

        # 3. Warped camera image — proves calibration alignment
        try:
            cam_np = last_frame.squeeze(0).cpu().numpy()
            if cam_np.max() > 1.5:
                cam_np = cam_np.astype(np.uint8)
            else:
                cam_np = (cam_np * 255).clip(0, 255).astype(np.uint8)
            warped = cv2.remap(
                cam_np, map_x, map_y,
                cv2.INTER_LINEAR, borderValue=0,
            )
            ok, buf = cv2.imencode(
                ".png", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
            )
            if ok:
                files["warped_camera.png"] = buf.tobytes()
        except Exception:
            logger.warning("Could not generate warped camera image", exc_info=True)

        # 4. Depth maps — two approaches for comparison
        #    (model is pre-cached in background at plugin load time)
        if self._live_depth_provider is None:
            try:
                from .depth_provider import create_depth_provider
                self._live_depth_provider = create_depth_provider(
                    "auto", self.device, model_size="small",
                )
                logger.info("Depth provider loaded for calibration results")
            except Exception:
                logger.warning("Could not load depth provider", exc_info=True)

        if self._live_depth_provider is not None:
            frame_f = last_frame.squeeze(0).to(device=self.device, dtype=torch.float32)
            if frame_f.max() > 1.5:
                frame_f = frame_f / 255.0

            # 4a. depth_then_warp: estimate depth from raw camera, warp result
            #     Depth model sees a natural camera image (best depth quality),
            #     then the depth map is remapped to projector perspective.
            try:
                depth = self._live_depth_provider.estimate(frame_f)  # (H, W)
                depth_rgb = build_warped_depth_image(
                    depth, map_x, map_y,
                    self._calib.proj_valid_mask if self._calib else None,
                    self.proj_h, self.proj_w,
                    device=self.device,
                )  # (H, W, 3) float32 [0,1]
                depth_np = (depth_rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                ok, buf = cv2.imencode(
                    ".png", cv2.cvtColor(depth_np, cv2.COLOR_RGB2BGR)
                )
                if ok:
                    files["depth_then_warp.png"] = buf.tobytes()
                    logger.info("Generated depth_then_warp.png")
            except Exception:
                logger.error("Could not generate depth_then_warp", exc_info=True)

            # 4b. warp_then_depth: warp camera RGB to projector, then estimate depth
            #     Depth model sees the scene from projector's perspective,
            #     which may produce more spatially coherent depth for VACE.
            try:
                warped_f = cv2.remap(
                    frame_f.cpu().numpy(), map_x, map_y,
                    cv2.INTER_LINEAR, borderValue=0,
                )
                warped_t = torch.from_numpy(warped_f).to(self.device)
                depth_w = self._live_depth_provider.estimate(warped_t)  # (H, W)

                # Normalise to [0, 1] grayscale RGB
                d_min, d_max = depth_w.min(), depth_w.max()
                if d_max - d_min > 1e-6:
                    depth_norm = (depth_w - d_min) / (d_max - d_min)
                else:
                    depth_norm = torch.zeros_like(depth_w)
                depth_u8 = (depth_norm.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                depth_rgb2 = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2RGB)
                ok, buf = cv2.imencode(
                    ".png", cv2.cvtColor(depth_rgb2, cv2.COLOR_RGB2BGR)
                )
                if ok:
                    files["warp_then_depth.png"] = buf.tobytes()
                    logger.info("Generated warp_then_depth.png")
            except Exception:
                logger.error("Could not generate warp_then_depth", exc_info=True)

        # Push to streamer for dashboard
        if files and self._streamer is not None:
            self._streamer.set_calibration_results(files, timestamp)
            if coverage_pct > 0:
                self._streamer._calibration_coverage_pct = coverage_pct
            logger.info(
                "Calibration results published for download: %s",
                list(files.keys()),
            )

        # Upload warped image + coverage to Scope's asset gallery for VACE
        self._upload_to_scope_gallery(files)

    def _upload_to_scope_gallery(self, files: dict[str, bytes]) -> None:
        """Upload calibration images to Scope's asset gallery for VACE use."""
        import urllib.request

        scope_url = "http://localhost:8000"
        upload_url = f"{scope_url}/api/v1/assets"

        for name, data in files.items():
            if not name.endswith(".png"):
                continue
            try:
                # Build multipart form data
                boundary = b"----ProMapBoundary"
                body = b"--" + boundary + b"\r\n"
                body += (
                    f'Content-Disposition: form-data; name="file"; '
                    f'filename="{name}"\r\n'
                ).encode()
                body += b"Content-Type: image/png\r\n\r\n"
                body += data + b"\r\n"
                body += b"--" + boundary + b"--\r\n"

                req = urllib.request.Request(
                    upload_url,
                    data=body,
                    headers={
                        "Content-Type": f"multipart/form-data; boundary={boundary.decode()}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    logger.info(
                        "Uploaded %s to Scope gallery (status %d)",
                        name, resp.status,
                    )
            except Exception:
                logger.debug(
                    "Could not upload %s to Scope gallery (API may not be available)",
                    name,
                    exc_info=True,
                )


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

        # MJPEG streamer for input preview on dashboard
        port = kwargs.get("stream_port", 8765)
        self._streamer = get_or_create_streamer(port)

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
            mx, my, pw, ph, ts = load_calibration(cal_path)
            self._map_x = mx
            self._map_y = my
            self.proj_w = pw
            self.proj_h = ph
            logger.info(
                "Calibration loaded from %s (%dx%d, captured %s)",
                cal_path, pw, ph, ts,
            )
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

        depth_mode = kwargs.get("depth_mode", "depth_then_warp")
        temporal_smoothing = kwargs.get("temporal_smoothing", 0.5)
        depth_blur = kwargs.get("depth_blur", 0.0)

        # Normalise input to [0, 1]
        frame_f = frame.squeeze(0).to(device=self.device, dtype=torch.float32)
        if frame_f.max() > 1.5:
            frame_f = frame_f / 255.0

        has_calib = self._map_x is not None and self._map_y is not None

        if depth_mode == "warped_rgb" and has_calib:
            # Warp camera RGB to projector perspective (no depth estimation)
            rgb = self._warped_rgb(frame_f)
        elif depth_mode == "warp_then_depth" and has_calib:
            # Warp camera RGB to projector perspective, then estimate depth
            rgb = self._warp_then_depth(frame_f, depth_blur, temporal_smoothing)
        else:
            # Estimate depth from raw camera, then warp to projector
            rgb = self._depth_then_warp(frame_f, depth_blur, temporal_smoothing)

        # Submit full-resolution preview to dashboard before resizing
        self._submit_input_preview(rgb)

        # Resize to generation resolution
        gen_w, gen_h = self._get_generation_resolution()
        h, w = rgb.shape[:2]
        if (w, h) != (gen_w, gen_h):
            rgb_np = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            rgb_np = cv2.resize(rgb_np, (gen_w, gen_h), interpolation=cv2.INTER_AREA)
            rgb = torch.from_numpy(rgb_np.astype(np.float32) / 255.0).to(self.device)

        return {"video": rgb.unsqueeze(0).clamp(0, 1)}

    def _depth_then_warp(
        self, frame_f: torch.Tensor, depth_blur: float, temporal_smoothing: float,
    ) -> torch.Tensor:
        """Estimate depth from raw camera, then warp to projector perspective."""
        depth = self._depth.estimate(frame_f)

        # Temporal smoothing
        if temporal_smoothing > 0 and self._prev_depth is not None:
            if self._prev_depth.shape == depth.shape:
                depth = temporal_smoothing * self._prev_depth + (1 - temporal_smoothing) * depth
        self._prev_depth = depth.clone()

        if self._map_x is not None and self._map_y is not None:
            return build_warped_depth_image(
                depth,
                self._map_x,
                self._map_y,
                self._proj_valid_mask,
                self.proj_h,
                self.proj_w,
                blur=depth_blur,
                colormap="grayscale",
                device=self.device,
            )

        # No calibration — output camera-space depth as grayscale
        return self._depth_to_grayscale(depth, depth_blur)

    def _warp_then_depth(
        self, frame_f: torch.Tensor, depth_blur: float, temporal_smoothing: float,
    ) -> torch.Tensor:
        """Warp camera RGB to projector perspective, then estimate depth."""
        # Warp camera to projector view
        warped_np = cv2.remap(
            frame_f.cpu().numpy(), self._map_x, self._map_y,
            cv2.INTER_LINEAR, borderValue=0,
        )
        warped_t = torch.from_numpy(warped_np).to(self.device)

        # Estimate depth from the warped (projector-perspective) image
        depth = self._depth.estimate(warped_t)

        # Temporal smoothing
        if temporal_smoothing > 0 and self._prev_depth is not None:
            if self._prev_depth.shape == depth.shape:
                depth = temporal_smoothing * self._prev_depth + (1 - temporal_smoothing) * depth
        self._prev_depth = depth.clone()

        # Already in projector space — just convert to grayscale RGB
        return self._depth_to_grayscale(depth, depth_blur)

    def _warped_rgb(self, frame_f: torch.Tensor) -> torch.Tensor:
        """Warp camera RGB to projector perspective (no depth estimation)."""
        warped_np = cv2.remap(
            frame_f.cpu().numpy(), self._map_x, self._map_y,
            cv2.INTER_LINEAR, borderValue=0,
        )
        return torch.from_numpy(warped_np).to(self.device)

    def _depth_to_grayscale(self, depth: torch.Tensor, blur: float) -> torch.Tensor:
        """Convert depth tensor to grayscale RGB (near=dark, far=bright)."""
        depth_np = depth.cpu().numpy()
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_np)
        if blur > 0.5:
            ksize = int(blur) * 2 + 1
            depth_norm = cv2.GaussianBlur(depth_norm, (ksize, ksize), 0)
        depth_uint8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)
        return torch.from_numpy(
            cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        ).to(self.device)

    def _submit_input_preview(self, rgb: torch.Tensor) -> None:
        """Send the preprocessor output to the streamer for dashboard preview."""
        if self._streamer is not None and self._streamer.is_running:
            rgb_np = (rgb.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            self._streamer.submit_input_preview(rgb_np)


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
