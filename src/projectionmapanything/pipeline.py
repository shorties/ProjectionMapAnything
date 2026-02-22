"""ProjectionMapAnything — pipelines.

Registered pipelines:
1. ProMapAnythingCalibratePipeline  — main pipeline (calibration visualization, future)
2. ProMapAnythingPipeline           — preprocessor (calibration + depth conditioning)
3. ProMapAnythingProjectorPipeline  — postprocessor (MJPEG relay to projector)

The preprocessor is the primary pipeline. It handles:
- Gray code calibration (inline, via start_calibration toggle)
- Depth estimation + projector warp
- Edge processing, effects, subject isolation, edge feathering
- All spatial masking is applied to the depth conditioning BEFORE the AI

The postprocessor relays AI output to the projector via MJPEG streaming.
Select it as the Postprocessor in Scope to see AI output on the projector page.
"""

from __future__ import annotations

import logging
import os
import threading
import time as _time
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
    _handler.setFormatter(logging.Formatter("[PMA Pipe] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

_DEFAULT_CALIBRATION_PATH = Path.home() / ".projectionmapanything_calibration.json"
_RESULTS_DIR = Path.home() / ".projectionmapanything_results"


def publish_calibration_results(
    map_x: np.ndarray,
    map_y: np.ndarray,
    rgb_frame_np: np.ndarray,
    proj_w: int,
    proj_h: int,
    proj_valid_mask: np.ndarray | None,
    coverage_pct: float,
    streamer: FrameStreamer | None,
    disparity_map: np.ndarray | None = None,
) -> dict[str, bytes]:
    """Generate calibration artifacts, push to streamer, and upload to Scope gallery.

    Parameters
    ----------
    rgb_frame_np : np.ndarray
        uint8 (H, W, 3) RGB camera image (ambient frame).
    proj_valid_mask : np.ndarray | None
        Boolean mask of valid projector pixels (before inpainting).
    streamer : FrameStreamer | None
        Shared MJPEG streamer to push results to.
    disparity_map : np.ndarray | None
        (proj_h, proj_w) float32 [0, 1] disparity from calibration.

    Returns
    -------
    dict[str, bytes]
        Mapping of filename → file bytes for all generated artifacts.
    """
    logger.info("Publishing calibration results (coverage=%.1f%%) ...", coverage_pct)
    timestamp = datetime.now(timezone.utc).isoformat()
    files: dict[str, bytes] = {}

    # 1. Calibration JSON
    try:
        files["calibration.json"] = _DEFAULT_CALIBRATION_PATH.read_bytes()
    except Exception:
        logger.warning("Could not read calibration JSON for download")

    # 2. Coverage map — green where valid, black where inpainted
    if proj_valid_mask is not None:
        coverage = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
        coverage[proj_valid_mask] = [0, 200, 100]
        ok, buf = cv2.imencode(".png", cv2.cvtColor(coverage, cv2.COLOR_RGB2BGR))
        if ok:
            files["coverage_map.png"] = buf.tobytes()

    # 3. Warped camera image
    try:
        warped = cv2.remap(
            rgb_frame_np, map_x, map_y,
            cv2.INTER_LINEAR, borderValue=0,
        )
        ok, buf = cv2.imencode(
            ".png", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
        )
        if ok:
            files["warped_camera.png"] = buf.tobytes()
    except Exception:
        logger.warning("Could not generate warped camera image", exc_info=True)

    # 4. Disparity depth map (from calibration decode)
    if disparity_map is not None:
        try:
            disp_u8 = (disparity_map * 255).clip(0, 255).astype(np.uint8)
            disp_bgr = cv2.cvtColor(disp_u8, cv2.COLOR_GRAY2BGR)
            ok, buf = cv2.imencode(".png", disp_bgr)
            if ok:
                files["depth_disparity.png"] = buf.tobytes()
                logger.info("Generated disparity depth map")
        except Exception:
            logger.warning("Could not generate disparity depth image", exc_info=True)

    # Save result images to disk
    _RESULTS_DIR.mkdir(exist_ok=True)
    for name, data in files.items():
        if name.endswith(".png"):
            (_RESULTS_DIR / name).write_bytes(data)
            logger.info("Saved %s to %s", name, _RESULTS_DIR / name)

    # Push to streamer for dashboard
    if files and streamer is not None:
        streamer.set_calibration_results(files, timestamp)
        if coverage_pct > 0:
            streamer._calibration_coverage_pct = coverage_pct
        logger.info(
            "Calibration results published for download: %s",
            list(files.keys()),
        )

    # Upload warped image + coverage to Scope's asset gallery for VACE
    _upload_to_scope_gallery(files)

    return files


def _upload_to_scope_gallery(files: dict[str, bytes]) -> None:
    """Upload calibration images to Scope's asset gallery for VACE use."""
    scope_url = "http://localhost:8000"
    upload_url = f"{scope_url}/api/v1/assets"

    for name, data in files.items():
        if not name.endswith(".png"):
            continue
        try:
            # Build multipart form data
            boundary = b"----PMABoundary"
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
# Calibration main pipeline
# =============================================================================


class ProMapAnythingCalibratePipeline(Pipeline):
    """Gray code structured light calibration.

    Select as the main pipeline, hit play. The Scope viewer shows the
    camera feed — position it on the projector. Then toggle
    **Start Calibration** to begin projecting patterns. When done, the
    calibration saves to ``~/.projectionmapanything_calibration.json``.
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

        # Calibration brightness (controls test card + WHITE pattern)
        self._cal_brightness: int = 128
        # Last camera frame captured while showing test card (used for warped RGB)
        self._ambient_frame: torch.Tensor | None = None

        # Build the grey test card (shown on projector before calibration)
        self._test_card = self._build_test_card(self._cal_brightness)

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

    def _build_test_card(self, brightness: int = 128) -> torch.Tensor:
        """Build a grey test card image at projector resolution.

        Shown on the projector before calibration starts to avoid
        camera-projector feedback loops.  Returns (H, W, 3) float32 [0,1].
        """
        card = np.full((self.proj_h, self.proj_w, 3), brightness, dtype=np.uint8)

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
        new_brightness = int(kwargs.get("calibration_brightness", self._cal_brightness))
        rebuild_card = False
        if new_pw != self.proj_w or new_ph != self.proj_h:
            self.proj_w = new_pw
            self.proj_h = new_ph
            rebuild_card = True
            logger.info("Projector resolution updated to %dx%d", self.proj_w, self.proj_h)
        if new_brightness != self._cal_brightness:
            self._cal_brightness = new_brightness
            rebuild_card = True
            logger.info("Calibration brightness updated to %d", self._cal_brightness)
        if rebuild_card:
            self._test_card = self._build_test_card(self._cal_brightness)

        # Reset calibration on rising edge (toggled ON)
        if reset and not self._reset_armed:
            self._reset_armed = True
            self._calib = None
            self._calibrating = False
            self._done = False
            self._ambient_frame = None
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

            # Store the camera frame while showing test card — this is what the
            # room looks like under the test card brightness. Used later for the
            # warped camera RGB image (instead of the white reference).
            self._ambient_frame = frame.clone()

            # Send test card to streamer (projector shows neutral grey)
            self._submit_to_streamer(self._test_card)
            return {"video": self._test_card.unsqueeze(0).clamp(0, 1)}

        # -- Start calibration on first toggle ON ----------------------------
        if start and not self._calibrating and not self._done:
            self._calib = CalibrationState(
                self.proj_w, self.proj_h,
                settle_frames=self._settle_frames,
                capture_frames=self._capture_frames,
                max_brightness=self._cal_brightness,
            )
            self._calib.start()
            self._calibrating = True
            if self._streamer is not None:
                self._streamer.clear_calibration_results()
            logger.info(
                "Calibration started (Gray code, %d patterns)",
                self._calib.total_patterns,
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
                        disparity_map=self._calib.disparity_map,
                        valid_mask=self._calib.proj_valid_mask,
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

                    # Use the stored ambient frame (captured while the test
                    # card was displayed) for the warped camera RGB image.
                    # This shows the room at the user's chosen brightness —
                    # much more natural than the white reference.  Fall back
                    # to the WHITE reference capture if no ambient was stored.
                    if self._ambient_frame is not None:
                        rgb_tensor = self._ambient_frame.clone()
                    else:
                        white_np = self._calib._get_averaged(0)  # float32 grayscale
                        white_rgb = np.stack([white_np] * 3, axis=-1)
                        white_rgb = (
                            white_rgb / max(white_rgb.max(), 1.0) * 255.0
                        ).clip(0, 255)
                        rgb_tensor = torch.from_numpy(
                            white_rgb.astype(np.float32) / 255.0
                        ).unsqueeze(0).to(self.device)

                    self._publish_calibration_results(
                        map_x, map_y, rgb_tensor, coverage_pct,
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
        pattern_info = self._calib.get_pattern_info()

        if self._calib._waiting_for_settle:
            pattern_info += f" settling {self._calib._settle_counter}/{self._calib.settle_frames}"

        self._streamer.update_calibration_progress(progress, phase, pattern_info)

    def _publish_calibration_results(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        last_frame: torch.Tensor,
        coverage_pct: float = 0.0,
    ) -> None:
        """Generate calibration artifacts, push to streamer, and upload to Scope gallery."""
        # Convert tensor to uint8 numpy
        cam_np = last_frame.squeeze(0).cpu().numpy()
        if cam_np.max() > 1.5:
            cam_np = cam_np.astype(np.uint8)
        else:
            cam_np = (cam_np * 255).clip(0, 255).astype(np.uint8)

        proj_valid_mask = (
            self._calib.proj_valid_mask if self._calib is not None else None
        )
        disparity_map = (
            self._calib.disparity_map if self._calib is not None else None
        )

        publish_calibration_results(
            map_x=map_x,
            map_y=map_y,
            rgb_frame_np=cam_np,
            proj_w=self.proj_w,
            proj_h=self.proj_h,
            proj_valid_mask=proj_valid_mask,
            coverage_pct=coverage_pct,
            streamer=self._streamer,
            disparity_map=disparity_map,
        )


# =============================================================================
# Depth preprocessor
# =============================================================================


class ProMapAnythingPipeline(Pipeline):
    """Primary preprocessor — calibration + depth conditioning.

    This is the main pipeline for ProjectionMapAnything. It handles:
    - **Calibration**: Toggle ``start_calibration`` to run Gray code
      structured light calibration inline (patterns go to projector via
      FrameStreamer, captures come from Scope's camera input).
    - **Depth conditioning**: Estimates depth, warps to projector perspective,
      applies effects/isolation/edge feathering, outputs VACE-optimized
      grayscale depth map.

    The preprocessor chain:
    ``depth → edge_processing → edge_blend → effect → isolation → edge_feather → resize``
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

        # Depth estimation — lazy-loaded on first use (avoids OOM in static mode)
        self._depth = None

        # MJPEG streamer for dashboard + projector pop-out
        port = kwargs.get("stream_port", 8765)
        self._streamer = get_or_create_streamer(port)

        # Calibration mapping
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self._proj_valid_mask: np.ndarray | None = None
        self._disparity_map: np.ndarray | None = None
        self.proj_w: int = 1920
        self.proj_h: int = 1080

        # Temporal smoothing buffer
        self._prev_depth: torch.Tensor | None = None

        # Effect animation timer
        self._effect_start_time: float = _time.time()

        # Last warped camera frame (for edge blend)
        self._last_warped_camera: np.ndarray | None = None

        # Cached depth results (invalidated on recalibration / resolution change)
        self._disparity_depth_cache: torch.Tensor | None = None
        self._ai_depth_cache: torch.Tensor | None = None
        self._static_warped_camera: torch.Tensor | None = None

        # -- Inline calibration state --
        self._calib: CalibrationState | None = None
        self._calibrating = False
        self._calib_done = False
        self._reset_armed = False
        self._ambient_frame: torch.Tensor | None = None

        # Load calibration
        cal_path = _DEFAULT_CALIBRATION_PATH
        if cal_path.is_file():
            mx, my, pw, ph, ts, disp = load_calibration(cal_path)
            self._map_x = mx
            self._map_y = my
            self._disparity_map = disp
            self.proj_w = pw
            self.proj_h = ph
            # Derive valid mask from NPZ if available, else from disparity
            npz_path = cal_path.with_suffix(".npz")
            if npz_path.is_file():
                npz = np.load(npz_path)
                if "valid_mask" in npz.files:
                    self._proj_valid_mask = npz["valid_mask"].astype(bool)
            if self._proj_valid_mask is None and disp is not None:
                # Disparity was computed only for valid pixels — derive mask
                # from non-uniform regions (inpainted areas are smooth gradients)
                self._proj_valid_mask = disp > 0.01
            logger.info(
                "Calibration loaded from %s (%dx%d, captured %s, disparity=%s, "
                "valid_mask=%s)",
                cal_path, pw, ph, ts,
                "yes" if disp is not None else "no",
                "yes" if self._proj_valid_mask is not None else "no",
            )
        else:
            logger.warning(
                "No calibration found at %s — output will be un-warped camera depth",
                cal_path,
            )

        logger.info("Preprocessor ready on port %d", port)

    def _get_generation_resolution(self, kwargs: dict | None = None) -> tuple[int, int]:
        """Compute generation resolution from projector res + preset."""
        gen_res = self._p("gen_resolution", kwargs or {}, "half")
        scale = self._RESOLUTION_SCALES.get(gen_res, 0.5)
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
                out_w, out_h = self._get_generation_resolution(kwargs)
            fallback = torch.full(
                (int(out_h), int(out_w), 3), 0.5,
                dtype=torch.float32, device=self.device,
            )
            return {"video": fallback.unsqueeze(0)}

    def _p(self, key: str, kwargs: dict, default=None):
        """Read a parameter: dashboard override > Scope kwargs > default."""
        if self._streamer is not None:
            ov = self._streamer._param_overrides.get(key)
            if ov is not None:
                return ov
        return kwargs.get(key, default)

    def _call_inner(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Depth preprocessor requires video input")

        frame = video[0]  # (1, H, W, C) [0, 255]

        # -- Reset calibration (rising edge) ----------------------------------
        reset_cal = self._p("reset_calibration", kwargs, False)
        if reset_cal and not self._reset_armed:
            self._reset_armed = True
            self._map_x = None
            self._map_y = None
            self._proj_valid_mask = None
            self._disparity_map = None
            self._prev_depth = None
            self._disparity_depth_cache = None
            self._ai_depth_cache = None
            self._static_warped_camera = None
            self._calib = None
            self._calibrating = False
            self._calib_done = False
            self._ambient_frame = None
            # Delete calibration file
            if _DEFAULT_CALIBRATION_PATH.is_file():
                _DEFAULT_CALIBRATION_PATH.unlink()
                logger.info("Deleted calibration file: %s", _DEFAULT_CALIBRATION_PATH)
            if self._streamer is not None:
                self._streamer.clear_calibration_results()
                self._streamer.calibration_active = False
            logger.info("Calibration reset — ready to recalibrate")
        elif not reset_cal:
            self._reset_armed = False

        # -- Auto-detect projector resolution from companion app/projector page
        if self._streamer is not None:
            cfg = self._streamer.client_config
            if cfg and "width" in cfg and "height" in cfg:
                new_pw, new_ph = int(cfg["width"]), int(cfg["height"])
                if new_pw != self.proj_w or new_ph != self.proj_h:
                    self.proj_w = new_pw
                    self.proj_h = new_ph
                    # Invalidate caches that depend on projector resolution
                    self._disparity_depth_cache = None
                    self._ai_depth_cache = None
                    self._static_warped_camera = None
                    logger.info("Projector resolution auto-detected: %dx%d", self.proj_w, self.proj_h)

        # -- Inline calibration -----------------------------------------------
        start_cal = self._p("start_calibration", kwargs, False)
        result = self._handle_inline_calibration(frame, start_cal, kwargs)
        if result is not None:
            return result

        # All processing params — read from dashboard overrides with sensible defaults
        # (disabled by default; advanced users tune via dashboard)
        depth_mode = self._p("depth_mode", kwargs, "ai_depth")
        temporal_smoothing = self._p("temporal_smoothing", kwargs, 0.0)
        depth_blur = self._p("depth_blur", kwargs, 0.0)
        edge_erosion = int(self._p("edge_erosion", kwargs, 0))
        depth_contrast = float(self._p("depth_contrast", kwargs, 1.0))
        near_clip = float(self._p("depth_near_clip", kwargs, 0.0))
        far_clip = float(self._p("depth_far_clip", kwargs, 1.0))
        edge_blend = float(self._p("edge_blend", kwargs, 0.0))
        edge_method = self._p("edge_method", kwargs, "sobel")
        active_effect = self._p("active_effect", kwargs, "none")
        effect_intensity = float(self._p("effect_intensity", kwargs, 0.5))
        effect_speed = float(self._p("effect_speed", kwargs, 1.0))
        isolation_mode = self._p("subject_isolation", kwargs, "none")
        subject_depth_range = float(self._p("subject_depth_range", kwargs, 0.3))
        subject_feather = float(self._p("subject_feather", kwargs, 5.0))
        invert_mask = bool(self._p("invert_subject_mask", kwargs, False))
        edge_feather = float(self._p("edge_feather", kwargs, 0.0))

        # Normalise input to [0, 1]
        frame_f = frame.squeeze(0).to(device=self.device, dtype=torch.float32)
        if frame_f.max() > 1.5:
            frame_f = frame_f / 255.0

        has_calib = self._map_x is not None and self._map_y is not None
        # All modes are static to prevent camera-projector feedback loops.
        is_static = True

        # -- Depth / source selection -----------------------------------------
        # All depth modes use static calibration data — never the live camera
        # feed (the camera sees the projector, creating feedback loops).
        if depth_mode == "ai_depth" and has_calib:
            rgb = self._ai_depth()
        elif depth_mode == "disparity" and has_calib:
            rgb = self._disparity_depth()
        elif depth_mode == "custom":
            rgb = self._get_custom_depth(frame_f)
        elif depth_mode == "canny" and has_calib:
            rgb = self._canny_edges(frame_f, is_static=True)
        elif depth_mode == "warped_rgb" and has_calib:
            rgb = self._warped_rgb_static()
        elif has_calib:
            # Any other mode with calibration: use AI depth
            rgb = self._ai_depth()
        else:
            # No calibration: neutral grey (no live depth — would cause feedback)
            rgb = self._grey_fallback(frame_f)

        # For static/custom modes, use the static warped camera image (from
        # calibration results) for edge blend and isolation instead of the
        # live camera feed. The live camera sees the projector output,
        # creating a feedback loop.
        if is_static:
            reference_frame = self._get_static_warped_camera()
        else:
            reference_frame = frame_f

        # -- Temporal smoothing -----------------------------------------------
        if temporal_smoothing > 0 and self._prev_depth is not None:
            prev = self._prev_depth
            if prev.shape == rgb.shape:
                rgb = temporal_smoothing * prev + (1 - temporal_smoothing) * rgb
        self._prev_depth = rgb.clone()

        # -- Depth blur -------------------------------------------------------
        if depth_blur > 0.5:
            rgb_np = rgb.cpu().numpy()
            ksize = int(depth_blur) * 2 + 1
            rgb_np = cv2.GaussianBlur(rgb_np, (ksize, ksize), 0)
            rgb = torch.from_numpy(rgb_np.astype(np.float32)).to(self.device)

        # -- Edge processing (erosion, contrast, clipping) --------------------
        rgb = self._apply_edge_processing(
            rgb, edge_erosion, depth_contrast, near_clip, far_clip,
        )

        # -- Edge blend (overlay surface edges from warped camera) ------------
        if edge_blend > 0 and has_calib:
            rgb = self._apply_edge_blend(
                rgb, reference_frame, edge_blend, edge_method, is_static=is_static,
            )

        # -- Depth effects (surface-masked) -----------------------------------
        if active_effect != "none" and effect_intensity > 0:
            rgb = self._apply_effect(
                rgb, active_effect, effect_intensity, effect_speed,
            )

        # -- Subject isolation ------------------------------------------------
        if isolation_mode != "none":
            rgb = self._apply_isolation(
                rgb, reference_frame, isolation_mode, subject_depth_range, subject_feather,
                invert=invert_mask,
            )

        # -- Edge feather (fade to black at projection boundaries) -----------
        if edge_feather > 0:
            rgb = self._apply_edge_feather(rgb, edge_feather)

        # -- Submit VACE input preview to dashboard ----------------------------
        if self._streamer is not None and self._streamer.is_running:
            preview_np = (rgb.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            self._streamer.submit_input_preview(preview_np)

        # -- Resize to match what Scope / the main pipeline expects -----------
        out_w = kwargs.get("width", None)
        out_h = kwargs.get("height", None)
        if out_w is None or out_h is None:
            out_w, out_h = self._get_generation_resolution(kwargs)
        else:
            out_w, out_h = int(out_w), int(out_h)
        h, w = rgb.shape[:2]
        if (w, h) != (out_w, out_h):
            rgb_nchw = rgb.permute(2, 0, 1).unsqueeze(0)
            rgb_nchw = torch.nn.functional.interpolate(
                rgb_nchw, size=(out_h, out_w), mode="area",
            )
            rgb = rgb_nchw.squeeze(0).permute(1, 2, 0)

        return {"video": rgb.unsqueeze(0).clamp(0, 1)}

    # -- Edge processing ------------------------------------------------------

    def _apply_edge_processing(
        self,
        rgb: torch.Tensor,
        erosion: int,
        contrast: float,
        near_clip: float,
        far_clip: float,
    ) -> torch.Tensor:
        """Apply edge erosion, contrast, and depth clipping."""
        if erosion <= 0 and abs(contrast - 1.0) < 0.01 and near_clip <= 0.0 and far_clip >= 1.0:
            return rgb

        img = rgb.cpu().numpy()
        gray = img.mean(axis=-1)

        if near_clip > 0.0 or far_clip < 1.0:
            mask = (gray >= near_clip) & (gray <= far_clip)
            if far_clip > near_clip:
                gray = np.where(mask, (gray - near_clip) / (far_clip - near_clip), 0.0)
            else:
                gray = np.zeros_like(gray)
            gray = gray.clip(0, 1)

        if abs(contrast - 1.0) >= 0.01:
            gray = np.where(
                gray > 0,
                np.power(gray.clip(1e-6, 1.0), 1.0 / contrast),
                0.0,
            )

        if erosion > 0:
            valid = (gray > 0.02).astype(np.uint8)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erosion * 2 + 1, erosion * 2 + 1),
            )
            eroded = cv2.erode(valid, kernel, iterations=1)
            gray = gray * eroded.astype(np.float32)

        out = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
        return torch.from_numpy(out).to(rgb.device)

    # -- Edge blend -----------------------------------------------------------

    def _apply_edge_blend(
        self,
        depth_rgb: torch.Tensor,
        reference_frame: torch.Tensor,
        blend: float,
        method: str,
        is_static: bool = False,
    ) -> torch.Tensor:
        """Blend surface edges from the warped camera into the depth map."""
        # For static modes the reference IS already the warped camera image.
        # For live modes we need to warp the raw camera through the calibration.
        if is_static:
            ref_np = reference_frame.cpu().numpy()
        else:
            ref_np = self._get_warped_camera_np(reference_frame)
            if ref_np is None:
                return depth_rgb

        # Convert to grayscale for edge detection
        if ref_np.ndim == 3 and ref_np.shape[2] == 3:
            gray = cv2.cvtColor(
                (ref_np * 255).clip(0, 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY,
            )
        else:
            gray = (ref_np * 255).clip(0, 255).astype(np.uint8)

        # Edge detection
        if method == "canny":
            edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        else:
            # Sobel
            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edges = np.sqrt(sx * sx + sy * sy)
            e_max = edges.max()
            if e_max > 1e-6:
                edges = edges / e_max

        # Resize edges to match depth_rgb if needed
        dh, dw = depth_rgb.shape[:2]
        if edges.shape[:2] != (dh, dw):
            edges = cv2.resize(edges, (dw, dh), interpolation=cv2.INTER_LINEAR)

        # Blend: depth + edges * blend
        edges_t = torch.from_numpy(edges).to(depth_rgb.device)
        edges_rgb = edges_t.unsqueeze(-1).expand(-1, -1, 3)
        return (depth_rgb + edges_rgb * blend).clamp(0, 1)

    def _get_warped_camera_np(self, frame_f: torch.Tensor) -> np.ndarray | None:
        """Get warped camera image for edge detection, with caching."""
        if self._map_x is None or self._map_y is None:
            return None
        warped = cv2.remap(
            frame_f.cpu().numpy(), self._map_x, self._map_y,
            cv2.INTER_LINEAR, borderValue=0,
        )
        self._last_warped_camera = warped
        return warped

    # -- Depth effects --------------------------------------------------------

    def _apply_effect(
        self,
        rgb: torch.Tensor,
        effect_name: str,
        intensity: float,
        speed: float,
    ) -> torch.Tensor:
        """Apply an animated effect to the depth map with surface masking.

        Effects only apply to non-black regions of the depth map (the actual
        projection surfaces). Black void regions stay black.
        """
        from . import effects

        # Extract grayscale depth for effect processing
        gray = rgb.mean(dim=-1)  # (H, W)
        t = _time.time() - self._effect_start_time

        # Build surface mask: non-black regions
        surface_mask = (gray > 0.02).float()

        # Apply effect to the grayscale depth
        effected = gray.clone()

        if effect_name == "noise_blend":
            effected = effects.apply_noise_blend(
                gray, intensity=intensity, scale=4.0, octaves=4,
                speed=speed, time=t,
            )
        elif effect_name == "flow_warp":
            effected = effects.apply_flow_warp(
                gray, intensity=intensity * 0.3, scale=4.0,
                speed=speed, time=t,
            )
        elif effect_name == "pulse":
            effected = effects.apply_pulse(
                gray, speed=speed, amount=intensity, time=t,
            )
        elif effect_name == "wave_warp":
            effected = effects.apply_wave_warp(
                gray, frequency=3.0, amplitude=intensity * 0.1,
                speed=speed, direction=0.0, time=t,
            )
        elif effect_name == "kaleido":
            effected = effects.apply_kaleido(
                gray, segments=max(2, int(intensity * 4 + 2)),
                rotation=0.0, time=t * speed,
            )
        elif effect_name == "shockwave":
            effected = effects.apply_shockwave(
                gray, origin_x=0.5, origin_y=0.5, speed=speed,
                thickness=0.15, strength=intensity, decay=1.5,
                time=t, auto_trigger_interval=max(2.0, 5.0 / max(speed, 0.1)),
            )
        elif effect_name == "wobble":
            effected = effects.apply_wobble(
                gray, intensity=intensity * 0.15, speed=speed, time=t,
            )
        elif effect_name == "geometry_edges":
            effected = effects.apply_geometry_edges(
                gray, edge_strength=intensity, glow_width=3.0,
                pulse_speed=speed, time=t,
            )
        elif effect_name == "depth_fog":
            effected = effects.apply_depth_fog(
                gray, density=intensity, near=0.0, far=0.7,
                animated=True, speed=speed, time=t,
            )
        elif effect_name == "radial_zoom":
            effected = effects.apply_radial_zoom(
                gray, origin_x=0.5, origin_y=0.5, strength=intensity * 0.2,
                speed=speed, time=t,
            )

        # Surface masking: effect only on surfaces, void stays black
        result = effected * surface_mask + gray * (1.0 - surface_mask)
        return result.clamp(0, 1).unsqueeze(-1).expand(-1, -1, 3)

    # -- Subject isolation ----------------------------------------------------

    def _apply_isolation(
        self,
        rgb: torch.Tensor,
        frame_f: torch.Tensor,
        mode: str,
        depth_range: float,
        feather: float,
        invert: bool = False,
    ) -> torch.Tensor:
        """Apply subject isolation to the depth RGB."""
        from .isolation import isolate_by_depth_band, isolate_by_mask, isolate_by_rembg

        h, w = rgb.shape[:2]
        gray_np = rgb.mean(dim=-1).cpu().numpy()
        mask_np: np.ndarray | None = None

        if mode == "depth_band":
            mask_np = isolate_by_depth_band(gray_np, band_width=depth_range, feather=feather)
        elif mode == "mask":
            mask_path = _RESULTS_DIR / "custom_mask.png"
            mask_np = isolate_by_mask(mask_path, h, w, feather=feather)
        elif mode == "rembg":
            # rembg needs the camera frame as uint8 RGB
            cam_np = (frame_f.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mask_np = isolate_by_rembg(cam_np, feather=feather)
            # Resize mask to match depth if needed
            if mask_np is not None and mask_np.shape[:2] != (h, w):
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)

        if mask_np is None:
            return rgb

        if invert:
            mask_np = 1.0 - mask_np

        mask_t = torch.from_numpy(mask_np).to(rgb.device).unsqueeze(-1)
        result = rgb * mask_t

        # Store mask on streamer for postprocessor subject mask
        if self._streamer is not None:
            self._streamer.set_isolation_mask(mask_np)

        return result

    # -- Edge feather ---------------------------------------------------------

    @staticmethod
    def _apply_edge_feather(frame: torch.Tensor, radius: float) -> torch.Tensor:
        """Fade to black at projection edges. Pure torch, vectorized."""
        h, w = frame.shape[:2]
        r = radius

        rows = torch.arange(h, device=frame.device, dtype=torch.float32)
        cols = torch.arange(w, device=frame.device, dtype=torch.float32)

        top = (rows / r).clamp(0, 1)
        bottom = ((h - 1 - rows) / r).clamp(0, 1)
        left = (cols / r).clamp(0, 1)
        right = ((w - 1 - cols) / r).clamp(0, 1)

        vert = torch.min(top, bottom).unsqueeze(1)   # (H, 1)
        horiz = torch.min(left, right).unsqueeze(0)   # (1, W)
        mask = (vert * horiz).unsqueeze(-1)            # (H, W, 1)

        return frame * mask

    # -- Inline calibration ---------------------------------------------------

    def _handle_inline_calibration(
        self,
        frame: torch.Tensor,
        start: bool,
        kwargs: dict,
    ) -> dict | None:
        """Run Gray code calibration inline, overriding projector output.

        Returns a preprocessor output dict while calibrating, or None
        to fall through to normal depth processing.
        """
        # Read calibration params via _p() (dashboard override > kwargs > default)
        cal_brightness = int(self._p("calibration_brightness", kwargs, 128))

        # Map calibration_speed (0=careful, 1=fast) to capture frames
        # settle_frames is only a timeout — change detection handles timing.
        # 0.0 = remote/RunPod (2 captures per pattern)
        # 0.85 = default (1 capture per pattern)
        # 1.0 = local fast (1 capture per pattern)
        cal_speed = float(self._p("calibration_speed", kwargs, 0.85))
        settle_frames = 60  # timeout only — change detection exits much earlier
        capture_frames = 3 if cal_speed < 0.5 else 2
        # Noise thresholds: tighter at slow speed for accuracy, relaxed at fast
        change_threshold = 5.0 if cal_speed < 0.5 else 5.0
        stability_threshold = 2.0 if cal_speed < 0.5 else 3.0

        # -- Toggled OFF or never started: cleanup and fall through ----------
        if not start:
            if self._calibrating:
                # User toggled off mid-calibration — cancel
                self._calibrating = False
                self._calib = None
                self._ambient_frame = None
                if self._streamer is not None:
                    self._streamer.calibration_active = False
                logger.info("Inline calibration cancelled")
            # Reset done flag so re-toggling starts fresh
            self._calib_done = False
            return None

        # -- First toggle ON: create CalibrationState -----------------------
        if start and not self._calibrating and not self._calib_done:
            self._calib = CalibrationState(
                self.proj_w, self.proj_h,
                settle_frames=settle_frames,
                capture_frames=capture_frames,
                max_brightness=cal_brightness,
                change_threshold=change_threshold,
                stability_threshold=stability_threshold,
            )
            self._calib.start()
            self._calibrating = True
            self._ambient_frame = frame.clone()
            if self._streamer is not None:
                self._streamer.clear_calibration_results()
            logger.info(
                "Inline calibration started (Gray code, %d patterns, %dx%d, "
                "speed=%.1f, capture=%d)",
                self._calib.total_patterns, self.proj_w, self.proj_h,
                cal_speed, capture_frames,
            )

        # -- Step calibration -----------------------------------------------
        if self._calibrating and self._calib is not None:
            pattern = self._calib.step(frame, self.device)

            # Update progress on streamer
            if self._streamer is not None:
                phase = self._calib.phase.name
                progress = self._calib.progress
                pattern_info = self._calib.get_pattern_info()
                self._streamer.update_calibration_progress(
                    progress, phase, pattern_info,
                )

            if self._calib.phase == CalibrationPhase.DONE:
                mapping = self._calib.get_mapping()
                if mapping is not None:
                    map_x, map_y = mapping
                    disp = self._calib.disparity_map
                    save_calibration(
                        map_x, map_y, _DEFAULT_CALIBRATION_PATH,
                        self.proj_w, self.proj_h,
                        disparity_map=disp,
                        valid_mask=self._calib.proj_valid_mask,
                    )
                    logger.info("Inline calibration saved to %s", _DEFAULT_CALIBRATION_PATH)

                    # Update live state
                    self._map_x = map_x
                    self._map_y = map_y
                    self._proj_valid_mask = self._calib.proj_valid_mask
                    self._disparity_map = disp

                    # Clear cached depth so all depth modes recompute
                    self._disparity_depth_cache = None
                    self._ai_depth_cache = None
                    self._static_warped_camera = None

                    # Coverage
                    coverage_pct = 0.0
                    if self._calib.proj_valid_mask is not None:
                        total = self._calib.proj_valid_mask.size
                        valid = np.count_nonzero(self._calib.proj_valid_mask)
                        coverage_pct = (valid / total) * 100.0 if total > 0 else 0.0

                    # Publish results in background (heavy: depth derivation,
                    # PNG encoding, disk I/O, gallery upload)
                    ambient_np = self._ambient_frame.squeeze(0).cpu().numpy()
                    if ambient_np.max() > 1.5:
                        ambient_np = ambient_np.astype(np.uint8)
                    else:
                        ambient_np = (ambient_np * 255).clip(0, 255).astype(np.uint8)

                    # Save raw camera frame for AI depth estimation
                    _RESULTS_DIR.mkdir(exist_ok=True)
                    try:
                        ambient_bgr = cv2.cvtColor(ambient_np, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(_RESULTS_DIR / "ambient_camera.png"), ambient_bgr)
                        logger.info("Saved ambient camera frame for AI depth")
                    except Exception:
                        logger.warning("Could not save ambient_camera.png", exc_info=True)

                    _pvm = self._calib.proj_valid_mask
                    _disp = self._calib.disparity_map
                    _str = self._streamer

                    def _publish_bg():
                        publish_calibration_results(
                            map_x=map_x, map_y=map_y,
                            rgb_frame_np=ambient_np,
                            proj_w=self.proj_w, proj_h=self.proj_h,
                            proj_valid_mask=_pvm,
                            coverage_pct=coverage_pct,
                            streamer=_str,
                            disparity_map=_disp,
                        )
                        if _str is not None:
                            _str.update_calibration_progress(
                                1.0, "DONE", coverage_pct=coverage_pct,
                            )

                    threading.Thread(
                        target=_publish_bg, daemon=True, name="calib-publish",
                    ).start()

                self._calibrating = False
                self._calib_done = True
                self._calib = None
                if self._streamer is not None:
                    self._streamer.calibration_active = False
                logger.info("Inline calibration complete — resuming normal mode")
                return None  # Fall through to normal depth processing

            elif pattern is not None:
                # Send pattern to projector via streamer
                t = pattern.squeeze(0) if pattern.ndim == 4 else pattern
                if t.max() > 1.5:
                    t = t / 255.0
                if self._streamer is not None:
                    self._streamer.calibration_active = True
                    rgb_np = (t.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
                    self._streamer.submit_calibration_frame(rgb_np)

        # While calibrating, output current depth (or grey) to the AI model.
        # The AI keeps generating — its output just won't reach the projector
        # because calibration_active=True suppresses normal streamer frames.
        out_w = kwargs.get("width", None)
        out_h = kwargs.get("height", None)
        if out_w is None or out_h is None:
            out_w, out_h = self._get_generation_resolution(kwargs)
        else:
            out_w, out_h = int(out_w), int(out_h)
        grey = torch.full(
            (int(out_h), int(out_w), 3), 0.5,
            dtype=torch.float32, device=self.device,
        )
        return {"video": grey.unsqueeze(0)}

    # -- Static warped camera -------------------------------------------------

    def _get_static_warped_camera(self) -> torch.Tensor:
        """Load the static warped camera from calibration results (cached).

        Used instead of the live camera feed for edge blend / isolation
        in static and custom depth modes to avoid feedback loops.
        """
        cached = getattr(self, "_static_warped_camera", None)
        if cached is not None:
            return cached

        path = _RESULTS_DIR / "warped_camera.png"
        if path.is_file():
            img = cv2.imread(str(path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(
                    img.astype(np.float32) / 255.0
                ).to(self.device)
                self._static_warped_camera = frame
                logger.info("Loaded static warped camera for edge blend / isolation")
                return frame

        logger.warning("No warped_camera.png found — edge blend will use live camera")
        # Return a grey placeholder so we don't crash
        frame = torch.full(
            (self.proj_h, self.proj_w, 3), 0.5,
            dtype=torch.float32, device=self.device,
        )
        self._static_warped_camera = frame
        return frame

    # -- Custom depth ---------------------------------------------------------

    def _get_custom_depth(self, frame_f: torch.Tensor) -> torch.Tensor:
        """Load custom depth map from disk (uploaded via dashboard)."""
        custom_path = _RESULTS_DIR / "custom_depth.png"
        if custom_path.is_file():
            img = cv2.imread(str(custom_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return torch.from_numpy(
                    img.astype(np.float32) / 255.0
                ).to(self.device)

        # Fallback: grey frame
        logger.warning("No custom depth map found at %s", custom_path)
        return torch.full(
            (self.proj_h, self.proj_w, 3), 0.5,
            dtype=torch.float32, device=self.device,
        )

    # -- Depth source methods -------------------------------------------------

    def _ensure_depth_model(self) -> None:
        """Lazy-load the depth model on first use."""
        if self._depth is None:
            from .depth_provider import create_depth_provider
            self._depth = create_depth_provider(self.device, model_size="small")

    _STATIC_FILE_MAP = {
        "static_depth_warped": "depth_then_warp.png",
        "static_depth_from_warped": "warp_then_depth.png",
        "static_warped_camera": "warped_camera.png",
    }

    def _get_static_frame(self, mode: str) -> torch.Tensor:
        """Load and cache a static image from calibration results."""
        cache_key = f"_static_{mode}"
        cached = getattr(self, cache_key, None)
        if cached is not None:
            return cached

        primary = self._STATIC_FILE_MAP.get(mode)
        candidates = [primary] if primary else []
        for name in ("depth_then_warp.png", "warp_then_depth.png", "warped_camera.png"):
            if name not in candidates:
                candidates.append(name)

        for name in candidates:
            path = _RESULTS_DIR / name
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

        logger.warning("No calibration result images found — using grey fallback")
        frame = torch.full(
            (self.proj_h, self.proj_w, 3), 0.5, dtype=torch.float32, device=self.device,
        )
        setattr(self, cache_key, frame)
        return frame

    def _warped_rgb_static(self) -> torch.Tensor:
        """Return the static warped camera image from calibration results."""
        return self._get_static_warped_camera()

    def _grey_fallback(self, frame_f: torch.Tensor) -> torch.Tensor:
        """Neutral grey fallback when no calibration is available."""
        h, w = frame_f.shape[:2] if frame_f.ndim >= 2 else (self.proj_h, self.proj_w)
        return torch.full(
            (h, w, 3), 0.5, dtype=torch.float32, device=self.device,
        )

    def _disparity_depth(self) -> torch.Tensor:
        """Derive depth from calibration correspondence (cached).

        Uses the disparity map from calibration if available, otherwise
        computes depth from the correspondence maps using homography-
        residual displacement: fits a perspective transform (homography)
        to the projector→camera mapping, then uses residual magnitude
        as depth-dependent parallax (closer objects deviate more from
        the planar projection model).
        """
        if self._disparity_depth_cache is not None:
            return self._disparity_depth_cache

        if self._disparity_map is not None:
            depth = self._disparity_map.copy()
            logger.info("Using saved disparity map from calibration")
        elif self._map_x is not None:
            depth = self._compute_disparity_from_maps()
        else:
            depth = np.full(
                (self.proj_h, self.proj_w), 0.5, dtype=np.float32,
            )
            logger.warning("No calibration maps — disparity is uniform grey")

        # CLAHE for local contrast (brings out surface detail)
        depth_u8 = (depth * 255).clip(0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        depth_u8 = clahe.apply(depth_u8)

        # Inpaint holes
        valid = self._proj_valid_mask
        if valid is not None:
            holes = (~valid).astype(np.uint8) * 255
            if np.any(~valid):
                depth_u8 = cv2.inpaint(depth_u8, holes, 15, cv2.INPAINT_NS)

        # Smooth
        depth_u8 = cv2.GaussianBlur(depth_u8, (7, 7), 0)
        depth = depth_u8.astype(np.float32) / 255.0

        depth_rgb = np.stack([depth, depth, depth], axis=-1)
        result = torch.from_numpy(depth_rgb).to(self.device)
        self._disparity_depth_cache = result
        logger.info(
            "Disparity depth ready: range [%.3f, %.3f]",
            depth.min(), depth.max(),
        )
        return result

    def _compute_disparity_from_maps(self) -> np.ndarray:
        """Compute depth from calibration correspondence via homography residuals.

        The projector→camera correspondence maps encode depth information
        through parallax: closer objects produce larger displacements from
        the expected projective mapping.  By fitting a homography (perspective
        transform) to the correspondence and taking the residual, we isolate
        the depth-dependent component.

        Returns (proj_h, proj_w) float32 in [0, 1], near=bright.
        """
        proj_h, proj_w = self._map_x.shape
        px = np.arange(proj_w, dtype=np.float32)[None, :].repeat(proj_h, axis=0)
        py = np.arange(proj_h, dtype=np.float32)[:, None].repeat(proj_w, axis=1)

        # Valid mask — if we have one from calibration, use it; else all valid
        if self._proj_valid_mask is not None:
            valid = self._proj_valid_mask
        else:
            # After inpainting, maps don't have -1 anymore, so use disparity hint
            valid = np.ones((proj_h, proj_w), dtype=bool)

        if not np.any(valid):
            return np.full((proj_h, proj_w), 0.5, dtype=np.float32)

        # Build source/destination point arrays for homography
        src = np.column_stack([px[valid], py[valid]]).astype(np.float32)
        dst = np.column_stack(
            [self._map_x[valid], self._map_y[valid]]
        ).astype(np.float32)

        # Subsample for performance (findHomography on 100k+ points is slow)
        rng = np.random.default_rng(42)
        if len(src) > 10000:
            idx = rng.choice(len(src), 10000, replace=False)
            H, _ = cv2.findHomography(src[idx], dst[idx], cv2.RANSAC, 5.0)
        else:
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if H is None:
            # Fallback: simple horizontal disparity (camera_x - projector_x)
            logger.warning("Homography fit failed — using simple disparity")
            disp = np.abs(self._map_x - px)
            disp[~valid] = 0.0
        else:
            # Apply homography to all projector pixels → predicted camera coords
            pts = np.column_stack(
                [px.ravel(), py.ravel()]
            ).astype(np.float32).reshape(-1, 1, 2)
            pred = cv2.perspectiveTransform(pts, H).reshape(proj_h, proj_w, 2)

            # Residual = actual - predicted correspondence
            res_x = self._map_x - pred[:, :, 0]
            res_y = self._map_y - pred[:, :, 1]
            disp = np.sqrt(res_x ** 2 + res_y ** 2)
            disp[~valid] = 0.0

        # Percentile normalization
        valid_vals = disp[valid]
        if valid_vals.size > 0:
            p2 = float(np.percentile(valid_vals, 2))
            p98 = float(np.percentile(valid_vals, 98))
            if p98 - p2 > 1e-6:
                disp = (disp - p2) / (p98 - p2)
            else:
                disp = np.full_like(disp, 0.5)
        depth = np.clip(disp, 0.0, 1.0).astype(np.float32)

        logger.info(
            "Computed disparity from correspondence maps (%dx%d, "
            "valid=%.1f%%, homography=%s)",
            proj_w, proj_h,
            100.0 * np.count_nonzero(valid) / max(valid.size, 1),
            "yes" if H is not None else "no (simple fallback)",
        )
        return depth

    def _load_camera_image(self) -> torch.Tensor | None:
        """Load the ambient camera image from calibration results.

        Returns the image as (H, W, 3) float32 [0, 1] on self.device,
        or None if the file doesn't exist or can't be decoded.
        """
        ambient_path = _RESULTS_DIR / "ambient_camera.png"
        if not ambient_path.is_file():
            return None
        img = cv2.imread(str(ambient_path))
        if img is None:
            logger.warning("AI depth: could not decode ambient_camera.png")
            return None
        logger.info("AI depth: loaded ambient camera %dx%d", img.shape[1], img.shape[0])
        return torch.from_numpy(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).to(self.device)

    def _warp_to_projector(self, depth_np: np.ndarray) -> np.ndarray:
        """Warp a camera-space depth map to projector space via calibration.

        Uses the same cv2.remap approach as the warped camera RGB (which is
        known to work correctly from publish_calibration_results).

        Pipeline:
        1. Convert depth to uint8 BGR (same format as camera RGB warp)
        2. cv2.remap with borderValue=0 (matching the RGB warp)
        3. Detect holes (all-black pixels + invalid mask)
        4. Inpaint holes with Navier-Stokes method
        5. Convert back to float32

        Parameters
        ----------
        depth_np : np.ndarray
            (H_cam, W_cam) float32 depth in [0, 1].

        Returns
        -------
        np.ndarray
            (proj_h, proj_w) float32 depth in [0, 1], holes inpainted.
        """
        cam_h, cam_w = depth_np.shape[:2]

        # Step 1: Convert to uint8 BGR — same pipeline as warped camera RGB
        depth_u8 = (depth_np * 255).clip(0, 255).astype(np.uint8)
        depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

        # Step 2: cv2.remap with borderValue=0 — same as publish_calibration_results
        warped_bgr = cv2.remap(
            depth_bgr, self._map_x, self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Step 3: Detect holes — unmapped regions + invalid calibration pixels
        holes = np.all(warped_bgr == 0, axis=2)
        if self._proj_valid_mask is not None:
            holes = holes | ~self._proj_valid_mask

        # Step 4: Inpaint holes
        if np.any(holes):
            hole_u8 = holes.astype(np.uint8) * 255
            warped_bgr = cv2.inpaint(warped_bgr, hole_u8, 15, cv2.INPAINT_NS)

        # Step 5: Convert back to float32 grayscale
        warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        warped = warped_gray.astype(np.float32) / 255.0

        logger.info(
            "Depth warped to projector %dx%d (from camera %dx%d), "
            "range [%.3f, %.3f], holes=%.1f%%",
            warped.shape[1], warped.shape[0], cam_w, cam_h,
            warped.min(), warped.max(),
            100.0 * np.count_nonzero(holes) / max(holes.size, 1),
        )
        return warped

    def _ai_depth(self) -> torch.Tensor:
        """Run Depth Anything V2 on a camera image, warp to projector space.

        Pipeline:
        1. Load ambient_camera.png (raw camera image from calibration)
           Falls back to warped_camera.png if ambient not available.
        2. Run Depth Anything V2 → (H, W) float32 [0,1], near=bright
        3. If source was camera-space, warp to projector via calibration maps
        4. Apply CLAHE for local contrast enhancement
        5. Gaussian blur for smooth output
        6. Cache result (static — recomputed only on recalibration)
        """
        if self._ai_depth_cache is not None:
            return self._ai_depth_cache

        # Load source image — prefer camera-space ambient for best quality
        camera_frame = self._load_camera_image()
        needs_warp = camera_frame is not None

        if camera_frame is None:
            # Fallback: warped camera (already in projector space)
            camera_frame = self._get_static_warped_camera()
            if camera_frame.mean() < 0.05 or camera_frame.mean() > 0.95:
                # Warped camera is all black/white — useless
                logger.warning(
                    "AI depth: no usable source image — returning grey. "
                    "Run calibration to capture an ambient frame."
                )
                result = torch.full(
                    (self.proj_h, self.proj_w, 3), 0.5,
                    dtype=torch.float32, device=self.device,
                )
                self._ai_depth_cache = result
                return result
            logger.warning(
                "AI depth: no ambient_camera.png — using warped camera fallback "
                "(re-run calibration for better results)"
            )

        # Run depth estimation
        self._ensure_depth_model()
        depth = self._depth.estimate(camera_frame)  # (H, W) [0,1], near=bright
        depth_np = depth.cpu().numpy().astype(np.float32)

        # Warp to projector space (only if source was camera-space)
        if needs_warp and self._map_x is not None and self._map_y is not None:
            depth_np = self._warp_to_projector(depth_np)
        elif not needs_warp:
            logger.info(
                "AI depth: source already projector-space, shape=%dx%d",
                depth_np.shape[1], depth_np.shape[0],
            )

        # CLAHE for local contrast enhancement — brings out surface detail
        # that percentile normalization alone misses
        depth_u8 = (depth_np * 255).clip(0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        depth_u8 = clahe.apply(depth_u8)
        depth_np = depth_u8.astype(np.float32) / 255.0

        # Light Gaussian blur for smooth output
        depth_np = cv2.GaussianBlur(depth_np, (5, 5), 0)

        # Grayscale → RGB tensor, cache
        rgb_np = np.stack([depth_np, depth_np, depth_np], axis=-1)
        result = torch.from_numpy(rgb_np).to(self.device)

        self._ai_depth_cache = result
        return result

    def _canny_edges(
        self, frame_f: torch.Tensor, is_static: bool = False,
    ) -> torch.Tensor:
        """Canny edge detection on warped camera image for VACE conditioning.

        Produces white edges on black background — excellent for VACE
        structural guidance without depth information.
        """
        # Get the warped camera image (static or live)
        if self._map_x is not None and self._map_y is not None:
            # Use static warped camera to avoid feedback loops
            ref = self._get_static_warped_camera()
            ref_np = (ref.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            # No calibration — use raw camera frame
            ref_np = (frame_f.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # Convert to grayscale
        if ref_np.ndim == 3 and ref_np.shape[2] == 3:
            gray = cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = ref_np

        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges slightly for better visibility
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Convert to float [0, 1] RGB
        edges_f = edges.astype(np.float32) / 255.0
        edges_rgb = np.stack([edges_f, edges_f, edges_f], axis=-1)
        return torch.from_numpy(edges_rgb).to(self.device)

    def _depth_to_grayscale(self, depth: torch.Tensor, blur: float) -> torch.Tensor:
        """Convert depth tensor to grayscale RGB (near=bright, far=dark)."""
        d_min = depth.min()
        d_max = depth.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = torch.zeros_like(depth)

        if blur > 0.5:
            depth_np = depth_norm.cpu().numpy()
            ksize = int(blur) * 2 + 1
            depth_np = cv2.GaussianBlur(depth_np, (ksize, ksize), 0)
            return torch.from_numpy(
                np.stack([depth_np, depth_np, depth_np], axis=-1).astype(np.float32)
            ).to(self.device)

        return depth_norm.unsqueeze(-1).expand(-1, -1, 3).to(self.device)



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

        # -- Edge feather -----------------------------------------------------
        edge_feather = float(kwargs.get("edge_feather", 0.0))
        if edge_feather > 0:
            frame = self._apply_edge_feather(frame, edge_feather)

        # -- Subject mask -----------------------------------------------------
        apply_mask = kwargs.get("apply_subject_mask", False)
        if apply_mask and self._streamer is not None:
            mask_np = self._streamer.get_isolation_mask()
            if mask_np is not None:
                h, w = frame.shape[:2]
                if mask_np.shape[:2] != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_t = torch.from_numpy(mask_np).to(frame.device).unsqueeze(-1)
                frame = frame * mask_t

        # -- Color correction -------------------------------------------------
        brightness = float(kwargs.get("brightness", 0.0))
        gamma = float(kwargs.get("gamma", 1.0))
        contrast = float(kwargs.get("contrast", 1.0))
        if abs(gamma - 1.0) > 0.01 or abs(brightness) > 0.001 or abs(contrast - 1.0) > 0.01:
            frame = self._apply_color_correction(frame, brightness, gamma, contrast)

        output = frame.unsqueeze(0).clamp(0, 1)

        if self._streamer is not None and self._streamer.is_running:
            rgb_np = (frame.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

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

    @staticmethod
    def _apply_edge_feather(frame: torch.Tensor, radius: float) -> torch.Tensor:
        """Fade to black at projection edges. Pure torch, vectorized."""
        h, w = frame.shape[:2]
        r = radius

        # Distance from each edge → ramp [0, 1] over radius pixels
        rows = torch.arange(h, device=frame.device, dtype=torch.float32)
        cols = torch.arange(w, device=frame.device, dtype=torch.float32)

        top = (rows / r).clamp(0, 1)
        bottom = ((h - 1 - rows) / r).clamp(0, 1)
        left = (cols / r).clamp(0, 1)
        right = ((w - 1 - cols) / r).clamp(0, 1)

        # Combine: minimum distance to any edge
        vert = torch.min(top, bottom).unsqueeze(1)  # (H, 1)
        horiz = torch.min(left, right).unsqueeze(0)  # (1, W)
        mask = (vert * horiz).unsqueeze(-1)  # (H, W, 1)

        return frame * mask

    @staticmethod
    def _apply_color_correction(
        frame: torch.Tensor,
        brightness: float,
        gamma: float,
        contrast: float,
    ) -> torch.Tensor:
        """Apply brightness, gamma, and contrast correction."""
        if abs(gamma - 1.0) > 0.01:
            frame = frame.clamp(1e-6, 1.0).pow(gamma)
        if abs(contrast - 1.0) > 0.01:
            frame = (frame - 0.5) * contrast + 0.5
        if abs(brightness) > 0.001:
            frame = frame + brightness
        return frame.clamp(0, 1)
