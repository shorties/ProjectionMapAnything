"""ProMapAnything VJ Tools — pipelines.

Registered pipelines:
1. ProMapAnythingCalibratePipeline  — main pipeline (Gray code calibration)
2. ProMapAnythingPipeline           — preprocessor (depth -> ControlNet)
3. ProMapAnythingProjectorPipeline  — postprocessor (streams output to projector)
"""

from __future__ import annotations

import logging
import os
import urllib.request
import webbrowser
from datetime import datetime, timezone
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

        # Build the grey test card (shown on projector before calibration)
        self._test_card = self._build_test_card()

        # Compute the projector URL (RunPod auto-detect)
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
                webbrowser.open(self._projector_url)
            except Exception:
                pass
        elif not open_dash:
            self._browser_opened = False

        # -- Not started yet: show test card -----------------------------------
        if not start and not self._calibrating:
            self._done = False
            if self._streamer is not None:
                self._streamer.calibration_active = False

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

                    # Push "generating results" so dashboard doesn't look stuck
                    if self._streamer is not None:
                        self._streamer.update_calibration_progress(
                            0.99, "Generating results...",
                            pattern_info="Saving calibration images",
                            coverage_pct=coverage_pct,
                        )

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

        # Only clear calibration_active when we're truly done, not during
        # the decode/publish gap between PATTERNS→DECODING→DONE.
        if self._streamer is not None and not self._calibrating:
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

    def _publish_calibration_results(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        last_frame: torch.Tensor,
        coverage_pct: float = 0.0,
    ) -> None:
        """Generate calibration artifacts, push to streamer, and upload to Scope gallery."""
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

        # 3. Warped camera image + grayscale depth-like images.
        # Neural depth is NOT loaded here — it would block calibration for
        # 30+ seconds downloading/loading the model.  Use a live depth mode
        # in the preprocessor if you need true neural depth.
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

            # Grayscale luminance as depth-like conditioning signal
            gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            ok, buf = cv2.imencode(".png", gray_bgr)
            if ok:
                files["depth_then_warp.png"] = buf.tobytes()
                files["warp_then_depth.png"] = buf.tobytes()
                logger.info("Generated grayscale depth images from warped camera")
        except Exception:
            logger.warning("Could not generate warped/depth images", exc_info=True)

        # Save result images to disk for static_calibration mode
        results_dir = _DEFAULT_CALIBRATION_PATH.parent / ".promapanything_results"
        results_dir.mkdir(exist_ok=True)
        for name, data in files.items():
            if name.endswith(".png"):
                (results_dir / name).write_bytes(data)
                logger.info("Saved %s to %s", name, results_dir / name)

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
        self._depth_mode: str = kwargs.get("depth_mode", "depth_then_warp")

        # Depth estimation — lazy-loaded on first use (avoids OOM in static mode)
        self._depth = None

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

        # Log init kwargs for debugging
        logger.info(
            "Depth preprocessor __init__ kwargs: %s",
            {k: v for k, v in kwargs.items() if k != "device"},
        )

        # Load calibration
        cal_path = _DEFAULT_CALIBRATION_PATH
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
        try:
            return self._call_inner(**kwargs)
        except Exception as exc:
            logger.error(
                "Depth preprocessor CRASHED: %s: %s",
                type(exc).__name__, exc, exc_info=True,
            )
            # Return a grey frame at expected resolution as fallback
            out_w = kwargs.get("width", None)
            out_h = kwargs.get("height", None)
            if out_w is None or out_h is None:
                out_w, out_h = self._get_generation_resolution()
            fallback = torch.full(
                (int(out_h), int(out_w), 3), 0.5,
                dtype=torch.float32, device=self.device,
            )
            return {"video": fallback.unsqueeze(0)}

    def _call_inner(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Depth preprocessor requires video input")

        frame = video[0]  # (1, H, W, C) [0, 255]

        depth_mode = kwargs.get("depth_mode", self._depth_mode)
        temporal_smoothing = kwargs.get("temporal_smoothing", 0.5)
        depth_blur = kwargs.get("depth_blur", 0.0)
        edge_erosion = int(kwargs.get("edge_erosion", 0))
        depth_contrast = float(kwargs.get("depth_contrast", 1.0))
        near_clip = float(kwargs.get("depth_near_clip", 0.0))
        far_clip = float(kwargs.get("depth_far_clip", 1.0))

        # Normalise input to [0, 1]
        frame_f = frame.squeeze(0).to(device=self.device, dtype=torch.float32)
        if frame_f.max() > 1.5:
            frame_f = frame_f / 255.0

        has_calib = self._map_x is not None and self._map_y is not None

        if depth_mode.startswith("static_"):
            # Use saved image from calibration (no live depth model)
            rgb = self._get_static_frame(depth_mode)
        elif depth_mode == "warped_rgb" and has_calib:
            # Warp camera RGB to projector perspective (no depth estimation)
            rgb = self._warped_rgb(frame_f)
        elif depth_mode == "warp_then_depth" and has_calib:
            # Warp camera RGB to projector perspective, then estimate depth
            rgb = self._warp_then_depth(frame_f, depth_blur, temporal_smoothing)
        else:
            # Estimate depth from raw camera, then warp to projector
            rgb = self._depth_then_warp(frame_f, depth_blur, temporal_smoothing)

        # Apply edge processing (erosion, contrast, clipping)
        rgb = self._apply_edge_processing(
            rgb, edge_erosion, depth_contrast, near_clip, far_clip,
        )

        # Submit full-resolution preview to dashboard before resizing
        self._submit_input_preview(rgb)

        # Resize to match what Scope / the main pipeline expects.
        # Scope passes target width/height in kwargs; fall back to our own calc.
        out_w = kwargs.get("width", None)
        out_h = kwargs.get("height", None)
        if out_w is None or out_h is None:
            out_w, out_h = self._get_generation_resolution()
        else:
            out_w, out_h = int(out_w), int(out_h)
        h, w = rgb.shape[:2]
        if (w, h) != (out_w, out_h):
            # GPU resize via torch interpolate
            # (H, W, 3) -> (1, 3, H, W) -> interpolate -> (H, W, 3)
            rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
            rgb_nchw = torch.nn.functional.interpolate(
                rgb_nchw, size=(out_h, out_w), mode="area",
            )
            rgb = rgb_nchw.squeeze(0).permute(1, 2, 0)

        return {"video": rgb.unsqueeze(0).clamp(0, 1)}

    def _apply_edge_processing(
        self,
        rgb: torch.Tensor,
        erosion: int,
        contrast: float,
        near_clip: float,
        far_clip: float,
    ) -> torch.Tensor:
        """Apply edge erosion, contrast, and depth clipping.

        Operates on the grayscale depth RGB (H, W, 3) float [0, 1].
        """
        if erosion <= 0 and abs(contrast - 1.0) < 0.01 and near_clip <= 0.0 and far_clip >= 1.0:
            return rgb  # Nothing to do

        img = rgb.cpu().numpy()  # (H, W, 3) float [0, 1]
        gray = img.mean(axis=-1)  # (H, W) for processing

        # 1. Near/far clipping — black out pixels outside the range
        if near_clip > 0.0 or far_clip < 1.0:
            mask = (gray >= near_clip) & (gray <= far_clip)
            # Re-normalise the remaining range to [0, 1]
            if far_clip > near_clip:
                gray = np.where(mask, (gray - near_clip) / (far_clip - near_clip), 0.0)
            else:
                gray = np.zeros_like(gray)
            gray = gray.clip(0, 1)

        # 2. Contrast enhancement — power curve centred at 0.5
        if abs(contrast - 1.0) >= 0.01:
            # Sigmoid-like contrast: push values away from 0.5
            gray = np.where(
                gray > 0,
                np.power(gray.clip(1e-6, 1.0), 1.0 / contrast),
                0.0,
            )

        # 3. Edge erosion — shrink the non-black region inward
        if erosion > 0:
            # Build a mask of "valid" (non-black) pixels
            valid = (gray > 0.02).astype(np.uint8)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erosion * 2 + 1, erosion * 2 + 1),
            )
            eroded = cv2.erode(valid, kernel, iterations=1)
            # Zero out pixels that were removed by erosion
            gray = gray * eroded.astype(np.float32)

        # Convert back to RGB
        out = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        return torch.from_numpy(out).to(rgb.device)

    def _ensure_depth_model(self) -> None:
        """Lazy-load the depth model on first use."""
        if self._depth is None:
            from .depth_provider import create_depth_provider
            self._depth = create_depth_provider("auto", self.device, model_size="small")

    # Map depth_mode values to calibration result filenames
    _STATIC_FILE_MAP = {
        "static_depth_warped": "depth_then_warp.png",
        "static_depth_from_warped": "warp_then_depth.png",
        "static_warped_camera": "warped_camera.png",
    }

    def _get_static_frame(self, mode: str) -> torch.Tensor:
        """Load and cache a static image from calibration results."""
        # Cache per mode so switching modes loads the right image
        cache_key = f"_static_{mode}"
        cached = getattr(self, cache_key, None)
        if cached is not None:
            return cached

        results_dir = _DEFAULT_CALIBRATION_PATH.parent / ".promapanything_results"

        # Try the specific file for this mode first
        primary = self._STATIC_FILE_MAP.get(mode)
        candidates = [primary] if primary else []
        # Fallback order if the specific file doesn't exist
        for name in ("depth_then_warp.png", "warp_then_depth.png", "warped_camera.png"):
            if name not in candidates:
                candidates.append(name)

        for name in candidates:
            path = results_dir / name
            if path.is_file():
                img = cv2.imread(str(path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(
                        img.astype(np.float32) / 255.0
                    ).to(self.device)
                    if name != primary:
                        logger.warning(
                            "Static frame for %s: wanted %s but fell back to %s "
                            "(re-run calibration to regenerate depth images)",
                            mode, primary, name,
                        )
                    else:
                        logger.info("Loaded static frame for %s: %s", mode, name)
                    setattr(self, cache_key, frame)
                    return frame

        # Fallback: grey frame at projector resolution
        logger.warning("No calibration result images found — using grey fallback")
        frame = torch.full(
            (self.proj_h, self.proj_w, 3), 0.5, dtype=torch.float32, device=self.device,
        )
        setattr(self, cache_key, frame)
        return frame

    def _depth_then_warp(
        self, frame_f: torch.Tensor, depth_blur: float, temporal_smoothing: float,
    ) -> torch.Tensor:
        """Estimate depth from raw camera, then warp to projector perspective."""
        self._ensure_depth_model()
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
        self._ensure_depth_model()
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
        """Convert depth tensor to grayscale RGB (near=dark, far=bright).

        Normalises on GPU, applies optional blur on CPU.
        """
        d_min = depth.min()
        d_max = depth.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = torch.zeros_like(depth)

        if blur > 0.5:
            # Gaussian blur requires CPU/numpy
            depth_np = depth_norm.cpu().numpy()
            ksize = int(blur) * 2 + 1
            depth_np = cv2.GaussianBlur(depth_np, (ksize, ksize), 0)
            return torch.from_numpy(
                np.stack([depth_np, depth_np, depth_np], axis=-1).astype(np.float32)
            ).to(self.device)

        # No blur — stay on GPU
        return depth_norm.unsqueeze(-1).expand(-1, -1, 3).to(self.device)

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

            # Determine target size for the encoder thread (resize off pipeline thread)
            target_size = None
            upscale = kwargs.get("upscale_to_projector", True)
            if upscale:
                proj_res = self._get_projector_resolution()
                if proj_res is not None:
                    proj_w, proj_h = proj_res
                    h, w = rgb_np.shape[:2]
                    if (w, h) != (proj_w, proj_h):
                        target_size = (proj_w, proj_h)

            self._streamer.submit_frame(rgb_np, target_size=target_size)

        return {"video": output}
