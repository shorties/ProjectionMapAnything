"""MJPEG streaming server for remote projector output.

Runs a lightweight HTTP server on a background thread.  The pipeline submits
frames via ``submit_frame(rgb_np)``; connected clients receive them as an
MJPEG stream.

A module-level **singleton** ensures both preprocessor and postprocessor share
the same server instance.  During calibration the preprocessor sends patterns
via ``submit_calibration_frame()`` while normal ``submit_frame()`` calls are
suppressed.

Endpoints
---------
``/``                        Control panel dashboard — calibration status, projector, downloads.
``/projector``               Clean fullscreen MJPEG viewer — drag to projector monitor, click for fullscreen.
``/stream``                  MJPEG multipart stream (``multipart/x-mixed-replace``).
``/frame``                   Single JPEG snapshot of the latest frame.
``POST /config``             Companion app reports its projector resolution.
``GET  /config``             Returns the current projector config (JSON).
``GET  /calibration/status`` Returns calibration progress, completion status and available files.
``GET  /calibration/download/<name>``  Download a calibration result file.
``GET  /calibration/preview/<name>``   Serve a calibration image inline (for thumbnails).
``GET  /calibration/export``           Download all calibration files as a single zip.
``POST /calibration/import``           Upload a calibration zip to restore files.

Works through RunPod's port proxy — expose the port and connect from anywhere.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import unquote

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

_BOUNDARY = b"promapframe"
_PROJECTOR_CONFIG_PATH = Path.home() / ".projectionmapanything_projector.json"
_CALIBRATION_JSON_PATH = Path.home() / ".projectionmapanything_calibration.json"
_CALIBRATION_NPZ_PATH = Path.home() / ".projectionmapanything_calibration.npz"
_RESULTS_DIR = Path.home() / ".projectionmapanything_results"


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ── HTML templates ───────────────────────────────────────────────────────────

_PROJECTOR_HTML = """\
<!DOCTYPE html>
<html><head>
<title>ProjectionMapAnything Projector</title>
<style>
  * { margin:0; padding:0; }
  body { background:#000; overflow:hidden;
         width:100vw; height:100vh;
         display:flex; justify-content:center;
         align-items:center; }
  body.fs { cursor:none; }
  video#rtc { width:100vw; height:100vh;
              object-fit:contain; display:block; }
  img#mjpeg { width:100vw; height:100vh;
              object-fit:contain; display:none;
              position:absolute; top:0; left:0; }
  #hint { position:fixed; bottom:30px;
          left:50%%; transform:translateX(-50%%);
          color:#444; font:14px sans-serif;
          pointer-events:none;
          transition:opacity 0.5s; }
  body.fs #hint { opacity:0; }
  #status { position:fixed; top:10px; right:10px;
            color:#555; font:11px monospace;
            pointer-events:none; }
  body.fs #status { opacity:0; }
</style>
</head><body>
<video id="rtc" autoplay playsinline muted></video>
<img id="mjpeg" />
<div id="hint">Click to go fullscreen &mdash; drag this window to your projector first</div>
<div id="status"></div>
<script>
const rtcVideo = document.getElementById('rtc');
const mjpegImg = document.getElementById('mjpeg');
const statusEl = document.getElementById('status');
let pc = null;
let sessionId = null;
let calibrating = false;
let rtcConnected = false;

// ---- Mode switching: WebRTC (AI output) vs MJPEG (calibration) ----
function showWebRTC() {
  rtcVideo.style.display = 'block';
  mjpegImg.style.display = 'none';
  mjpegImg.src = '';
}
function showMJPEG() {
  mjpegImg.style.display = 'block';
  rtcVideo.style.display = 'none';
  mjpegImg.src = '/stream?t=' + Date.now();
}
mjpegImg.onerror = () => { setTimeout(() => { if (calibrating) showMJPEG(); }, 1000); };

// ---- Poll calibration status to switch modes ----
async function pollCalibration() {
  try {
    const r = await fetch('/calibration/status');
    const d = await r.json();
    const wasCalibrating = calibrating;
    calibrating = d.active;
    if (calibrating && !wasCalibrating) {
      statusEl.textContent = 'CALIBRATING';
      showMJPEG();
    } else if (!calibrating && wasCalibrating) {
      statusEl.textContent = 'AI Output';
      showWebRTC();
    }
  } catch(e) {}
}
setInterval(pollCalibration, 1000);

// ---- WebRTC connection to Scope ----
async function connectWebRTC() {
  statusEl.textContent = 'Connecting WebRTC...';
  try {
    // 1. Get ICE servers via our proxy
    const iceResp = await fetch('/scope/ice-servers');
    if (!iceResp.ok) throw new Error('ICE servers: ' + iceResp.status);
    const iceData = await iceResp.json();
    const iceServers = iceData.iceServers || [{ urls: ['stun:stun.l.google.com:19302'] }];

    // 2. Create peer connection
    pc = new RTCPeerConnection({ iceServers });

    // 3. Create data channel (Scope expects this)
    const dc = pc.createDataChannel('parameters', { ordered: true });

    // 4. Send a black video track (Scope expects sendrecv)
    const canvas = document.createElement('canvas');
    canvas.width = 512; canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, 512, 512);
    const blackStream = canvas.captureStream(1);
    const blackTrack = blackStream.getVideoTracks()[0];
    const sender = pc.addTrack(blackTrack, blackStream);

    // 5. Force VP8 codec (Scope's aiortc requires VP8)
    const transceiver = pc.getTransceivers().find(t => t.sender === sender);
    if (transceiver) {
      const codecs = RTCRtpReceiver.getCapabilities('video')?.codecs || [];
      const vp8 = codecs.filter(c => c.mimeType.toLowerCase() === 'video/vp8');
      if (vp8.length > 0) transceiver.setCodecPreferences(vp8);
    }

    // 6. Handle incoming AI output video
    pc.ontrack = (e) => {
      if (e.track.kind === 'video') {
        rtcVideo.srcObject = e.streams[0] || new MediaStream([e.track]);
        rtcVideo.play().catch(() => {});
        rtcConnected = true;
        if (!calibrating) {
          statusEl.textContent = 'AI Output';
          showWebRTC();
        }
      }
    };

    // 7. Trickle ICE
    pc.onicecandidate = async (e) => {
      if (e.candidate && sessionId) {
        await fetch('/scope/ice/' + sessionId, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            candidates: [{
              candidate: e.candidate.candidate,
              sdpMid: e.candidate.sdpMid,
              sdpMLineIndex: e.candidate.sdpMLineIndex
            }]
          })
        }).catch(() => {});
      }
    };

    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
        statusEl.textContent = 'WebRTC lost, reconnecting...';
        rtcConnected = false;
        setTimeout(connectWebRTC, 3000);
      }
    };

    // 8. Create offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // 9. Send offer via our proxy
    const answerResp = await fetch('/scope/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sdp: pc.localDescription.sdp,
        type: 'offer',
        initialParameters: { input_mode: 'video' }
      })
    });
    if (!answerResp.ok) throw new Error('Offer: ' + answerResp.status);
    const answer = await answerResp.json();
    sessionId = answer.sessionId;
    await pc.setRemoteDescription({ type: 'answer', sdp: answer.sdp });

    statusEl.textContent = 'WebRTC connected';
  } catch(err) {
    console.warn('WebRTC failed, falling back to MJPEG:', err);
    statusEl.textContent = 'MJPEG mode (WebRTC unavailable)';
    rtcConnected = false;
    showMJPEG();
    // Retry WebRTC after delay
    setTimeout(connectWebRTC, 10000);
  }
}

// ---- Start ----
connectWebRTC();

// ---- Fullscreen ----
document.body.addEventListener('click', () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
  }
});
document.addEventListener('fullscreenchange', () => {
  document.body.classList.toggle('fs', !!document.fullscreenElement);
});

// ---- Report screen resolution ----
function postConfig() {
  const s = window.screen;
  fetch('/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ width: s.width, height: s.height, monitor_name: 'browser' })
  }).catch(() => {});
}
postConfig();
setInterval(postConfig, 30000);
</script>
</body></html>
"""

_CONTROL_PANEL_HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


class FrameStreamer:
    """MJPEG HTTP streaming server for projector output.

    Usage::

        streamer = FrameStreamer(port=8765)
        streamer.start()
        streamer.submit_frame(rgb_numpy)   # call every frame
        streamer.stop()
    """

    def __init__(self, port: int = 8765, jpeg_quality: int = 70) -> None:
        self._port = port
        self._quality = jpeg_quality
        self._frame_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._server: _ThreadedHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        # Non-blocking encode guard: submit_frame skips if an encode is
        # already in progress (keeps the pipeline thread fast without
        # background threads that complicate plugin lifecycle).
        self._encoding = threading.Lock()

        # Calibration priority: when True, submit_frame() is suppressed
        self._calibration_active = False

        # Calibration results for download
        self._calibration_files: dict[str, bytes] = {}
        self._calibration_complete = False
        self._calibration_timestamp: str = ""

        # Calibration progress tracking
        self._calibration_progress: float = 0.0
        self._calibration_phase: str = ""
        self._calibration_pattern_info: str = ""
        self._calibration_errors: list[str] = []
        self._calibration_coverage_pct: float = 0.0

        # Input preview (VACE conditioning) — separate from projector output
        self._input_preview_jpeg: bytes | None = None
        self._input_lock = threading.Lock()
        self._input_new_frame = threading.Event()

        # Subject isolation mask (shared between pre/postprocessor)
        self._isolation_mask: np.ndarray | None = None

        # Client-reported projector config (resolution, monitor name)
        self._client_config: dict | None = None
        self._load_persisted_config()

        # Standalone calibration (browser webcam → server CalibrationState)
        self._standalone_calib = None  # CalibrationState | None
        self._standalone_device: torch.device | None = None
        self._standalone_proj_w: int = 1920
        self._standalone_proj_h: int = 1080
        self._standalone_ambient: np.ndarray | None = None

        # Dashboard parameter overrides — set via POST /api/params,
        # read by pipelines as kwargs overrides.
        self._param_overrides: dict = {}

        # Track active MJPEG stream clients (projector pages).
        # When zero AND dashboard preview is disabled, submit_frame()
        # skips JPEG encoding entirely to save CPU.
        self._stream_client_count = 0
        self._stream_client_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def port(self) -> int:
        return self._port

    @property
    def calibration_active(self) -> bool:
        return self._calibration_active

    @calibration_active.setter
    def calibration_active(self, value: bool) -> None:
        self._calibration_active = value

    @property
    def client_config(self) -> dict | None:
        """Resolution reported by the companion app, or None."""
        return self._client_config

    def set_calibration_results(
        self, files: dict[str, bytes], timestamp: str = ""
    ) -> None:
        """Store calibration result files for download via the projector page."""
        self._calibration_files = files
        self._calibration_complete = True
        self._calibration_timestamp = timestamp

    def clear_calibration_results(self) -> None:
        """Clear stored calibration results (e.g. when starting a new calibration)."""
        self._calibration_files = {}
        self._calibration_complete = False
        self._calibration_timestamp = ""
        self._calibration_progress = 0.0
        self._calibration_phase = ""
        self._calibration_pattern_info = ""
        self._calibration_errors = []
        self._calibration_coverage_pct = 0.0

    def update_calibration_progress(
        self,
        progress: float,
        phase: str,
        pattern_info: str = "",
        errors: list[str] | None = None,
        coverage_pct: float = 0.0,
    ) -> None:
        """Update calibration progress for the control panel dashboard."""
        self._calibration_progress = progress
        self._calibration_phase = phase
        self._calibration_pattern_info = pattern_info
        if errors is not None:
            self._calibration_errors = errors
        self._calibration_coverage_pct = coverage_pct

    # -- Subject isolation mask (shared between pre/postprocessor) ----------

    def set_isolation_mask(self, mask: np.ndarray) -> None:
        """Store the subject isolation mask from the preprocessor."""
        self._isolation_mask = mask.copy()

    def get_isolation_mask(self) -> np.ndarray | None:
        """Retrieve the subject isolation mask for the postprocessor."""
        return getattr(self, "_isolation_mask", None)

    # -- Custom depth/mask upload -------------------------------------------

    def set_custom_upload(self, data: bytes, stage: str, upload_type: str) -> dict:
        """Process and save a custom depth map or mask upload.

        Parameters
        ----------
        data : bytes
            Raw image bytes (JPEG/PNG).
        stage : str
            Processing stage: raw_camera, depth_estimated, depth_warped.
        upload_type : str
            'depth' or 'mask'.

        Returns
        -------
        dict
            Status dict with 'ok', 'filename', 'stage'.
        """
        results_dir = Path.home() / ".projectionmapanything_results"
        results_dir.mkdir(exist_ok=True)

        # Decode image
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"ok": False, "error": "Could not decode image"}

        if upload_type == "mask":
            # Save as grayscale mask
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out_path = results_dir / "custom_mask.png"
            cv2.imwrite(str(out_path), gray)
            logger.info("Saved custom mask to %s (%dx%d)", out_path, gray.shape[1], gray.shape[0])
            return {"ok": True, "filename": "custom_mask.png", "stage": stage}

        # depth upload — save directly (stage processing is done by pipeline)
        out_path = results_dir / "custom_depth.png"
        cv2.imwrite(str(out_path), img)
        logger.info(
            "Saved custom depth to %s (%dx%d, stage=%s)",
            out_path, img.shape[1], img.shape[0], stage,
        )
        return {"ok": True, "filename": "custom_depth.png", "stage": stage}

    def get_upload_status(self) -> dict:
        """Return status of custom uploads."""
        results_dir = Path.home() / ".projectionmapanything_results"
        depth_path = results_dir / "custom_depth.png"
        mask_path = results_dir / "custom_mask.png"
        return {
            "has_custom_depth": depth_path.is_file(),
            "has_custom_mask": mask_path.is_file(),
        }

    # -- Standalone calibration (browser webcam) --------------------------------

    def start_standalone_calibration(
        self,
        proj_w: int = 1920,
        proj_h: int = 1080,
        settle_frames: int = 10,
        capture_frames: int = 3,
        max_brightness: int = 128,
    ) -> dict:
        """Start a standalone calibration session using browser webcam frames.

        Returns dict with 'ok' and 'total_patterns'.
        """
        if self._calibration_active:
            return {"ok": False, "error": "Calibration already in progress"}

        from .calibration import CalibrationState

        self._standalone_proj_w = proj_w
        self._standalone_proj_h = proj_h
        self._standalone_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._standalone_ambient = None

        # Lower settle for browser mode: each frame is a full HTTP round-trip
        # (~160ms) so we don't need as many settle frames as the Scope pipeline.
        self._standalone_calib = CalibrationState(
            proj_w, proj_h,
            settle_frames=settle_frames,
            capture_frames=capture_frames,
            max_brightness=max_brightness,
            min_settle=3,
        )
        self._standalone_calib.start()

        self._calibration_active = True
        self.clear_calibration_results()

        # Send initial test card to projector stream
        card = np.full((proj_h, proj_w, 3), max_brightness, dtype=np.uint8)
        cv2.rectangle(card, (2, 2), (proj_w - 3, proj_h - 3), (200, 200, 200), 1)
        self.submit_calibration_frame(card)

        total = self._standalone_calib.total_patterns
        logger.info(
            "Standalone calibration started: %dx%d, %d patterns",
            proj_w, proj_h, total,
        )
        return {"ok": True, "total_patterns": total}

    def step_standalone_calibration(self, jpeg_bytes: bytes) -> dict:
        """Process one webcam frame for standalone calibration.

        Parameters
        ----------
        jpeg_bytes : bytes
            JPEG-encoded webcam frame from browser getUserMedia.

        Returns
        -------
        dict with phase, progress, pattern_info, done, settling, and
        optionally coverage_pct.
        """
        from .calibration import CalibrationPhase

        calib = self._standalone_calib
        if calib is None:
            return {"error": "No calibration in progress", "done": True}

        # If we're in DECODING phase, run decode in this call.
        # This will block for a few seconds — the browser shows
        # "Decoding..." from the previous response while we work.
        if calib.phase == CalibrationPhase.DECODING:
            self.update_calibration_progress(
                0.98, "DECODING", "Decoding patterns...",
            )
            # step() will run _decode() and set phase=DONE
            dummy = torch.zeros(1, 1, 1, 3)
            calib.step(dummy, self._standalone_device or torch.device("cpu"))
            if calib.phase == CalibrationPhase.DONE:
                return self._finish_standalone_calibration()
            return {
                "phase": "DECODING", "progress": 0.98,
                "pattern_info": "Decoding...", "done": False,
            }

        # Decode JPEG → RGB uint8
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "Could not decode JPEG frame", "done": False}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Store first frame as ambient (for warped camera image)
        if self._standalone_ambient is None:
            self._standalone_ambient = rgb.copy()

        # Wrap in tensor: (1, H, W, 3) float32
        tensor = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(0)
        device = self._standalone_device or torch.device("cpu")

        # Track settle state before stepping
        was_settling = calib._waiting_for_settle

        # Step the calibration state machine
        pattern = calib.step(tensor, device)

        # Build progress info
        phase = calib.phase.name
        progress = calib.progress
        settling = calib._waiting_for_settle
        pattern_info = ""

        if calib.phase == CalibrationPhase.PATTERNS:
            idx = calib._pattern_index
            total_x = 2 * calib.bits_x
            if idx < total_x:
                bit = idx // 2 + 1
                pattern_info = f"bit {bit}/{calib.bits_x} X-axis"
            else:
                y_idx = idx - total_x
                bit = y_idx // 2 + 1
                pattern_info = f"bit {bit}/{calib.bits_y} Y-axis"
            captured = sum(len(s) for s in calib._captures)
            total_cap = calib.total_patterns * calib.capture_frames
            pattern_info += f" ({captured}/{total_cap} captures)"
            if settling:
                pattern_info += f" settling {calib._settle_counter}/{calib.settle_frames}"
        elif calib.phase == CalibrationPhase.WHITE:
            pattern_info = "Capturing white reference"
            if settling:
                pattern_info += f" settling {calib._settle_counter}/{calib.settle_frames}"
        elif calib.phase == CalibrationPhase.BLACK:
            pattern_info = "Capturing black reference"
            if settling:
                pattern_info += f" settling {calib._settle_counter}/{calib.settle_frames}"
        elif calib.phase == CalibrationPhase.DECODING:
            pattern_info = "Decoding patterns..."

        # Update dashboard progress
        self.update_calibration_progress(progress, phase, pattern_info)

        # Send pattern to projector stream
        if pattern is not None:
            pat = pattern.squeeze(0) if pattern.ndim == 4 else pattern
            if pat.max() > 1.5:
                pat = pat / 255.0
            pat_np = (pat.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            self.submit_calibration_frame(pat_np)

        # Check if done
        if calib.phase == CalibrationPhase.DONE:
            return self._finish_standalone_calibration()

        return {
            "phase": phase,
            "progress": progress,
            "pattern_info": pattern_info,
            "settling": settling,
            "done": False,
        }

    def _finish_standalone_calibration(self) -> dict:
        """Finalize standalone calibration: save mapping and publish results."""
        from .calibration import save_calibration

        calib = self._standalone_calib
        if calib is None:
            return {"error": "No calibration state", "done": True}

        mapping = calib.get_mapping()
        coverage_pct = 0.0

        if mapping is not None:
            map_x, map_y = mapping

            # Compute coverage
            if calib.proj_valid_mask is not None:
                total = calib.proj_valid_mask.size
                valid = np.count_nonzero(calib.proj_valid_mask)
                coverage_pct = (valid / total) * 100.0 if total > 0 else 0.0

            self.update_calibration_progress(
                0.99, "Generating results...",
                pattern_info="Saving calibration",
                coverage_pct=coverage_pct,
            )

            # Save calibration to disk
            cal_path = _CALIBRATION_JSON_PATH
            logger.info("Saving standalone calibration to %s ...", cal_path)
            save_calibration(
                map_x, map_y, cal_path,
                self._standalone_proj_w, self._standalone_proj_h,
            )

            # Publish results (warped camera, coverage map, etc.)
            from .pipeline import publish_calibration_results

            ambient = self._standalone_ambient
            if ambient is None:
                ambient = np.full(
                    (480, 640, 3), 128, dtype=np.uint8,
                )

            publish_calibration_results(
                map_x=map_x,
                map_y=map_y,
                rgb_frame_np=ambient,
                proj_w=self._standalone_proj_w,
                proj_h=self._standalone_proj_h,
                proj_valid_mask=calib.proj_valid_mask,
                coverage_pct=coverage_pct,
                streamer=self,
            )

            self.update_calibration_progress(
                1.0, "DONE", coverage_pct=coverage_pct,
            )
            logger.info(
                "Standalone calibration complete (coverage=%.1f%%)", coverage_pct,
            )
        else:
            logger.warning("Standalone calibration: mapping was None")
            self.update_calibration_progress(
                1.0, "DONE", errors=["Mapping was None — decode failed"],
            )

        # Cleanup
        self._standalone_calib = None
        self._standalone_ambient = None
        self._calibration_active = False

        return {
            "phase": "DONE",
            "progress": 1.0,
            "pattern_info": "Complete",
            "done": True,
            "coverage_pct": coverage_pct,
        }

    def stop_standalone_calibration(self) -> dict:
        """Cancel an in-progress standalone calibration."""
        was_active = self._standalone_calib is not None
        self._standalone_calib = None
        self._standalone_ambient = None
        self._calibration_active = False
        self.clear_calibration_results()
        logger.info("Standalone calibration stopped (was_active=%s)", was_active)
        return {"ok": True}

    # -- Calibration export/import ---------------------------------------------

    def export_calibration_zip(self) -> bytes | None:
        """Bundle all calibration files into a single zip archive.

        Includes the JSON metadata, NPZ binary maps, and any result
        images from the results directory.

        Returns zip bytes, or None if no calibration exists.
        """
        if not _CALIBRATION_JSON_PATH.is_file():
            return None

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Core calibration files
            zf.write(_CALIBRATION_JSON_PATH, _CALIBRATION_JSON_PATH.name)
            if _CALIBRATION_NPZ_PATH.is_file():
                zf.write(_CALIBRATION_NPZ_PATH, _CALIBRATION_NPZ_PATH.name)

            # Result images
            if _RESULTS_DIR.is_dir():
                for f in _RESULTS_DIR.iterdir():
                    if f.is_file() and f.suffix in (".png", ".json", ".npz"):
                        zf.write(f, f"results/{f.name}")

        logger.info("Exported calibration zip (%d bytes)", buf.tell())
        return buf.getvalue()

    def import_calibration_zip(self, data: bytes) -> dict:
        """Import a calibration zip archive, restoring all files.

        Returns a status dict with 'ok' and optional 'error'.
        """
        try:
            buf = io.BytesIO(data)
            with zipfile.ZipFile(buf, "r") as zf:
                names = zf.namelist()

                # Must contain the JSON metadata at minimum
                json_name = _CALIBRATION_JSON_PATH.name
                if json_name not in names:
                    return {"ok": False, "error": f"Missing {json_name} in zip"}

                # Extract core files to home directory
                for name in names:
                    if name.startswith("results/"):
                        # Result images go into the results dir
                        rel = name[len("results/"):]
                        if not rel:
                            continue
                        _RESULTS_DIR.mkdir(exist_ok=True)
                        (_RESULTS_DIR / rel).write_bytes(zf.read(name))
                    else:
                        # Core calibration files go to home dir
                        target = Path.home() / name
                        target.write_bytes(zf.read(name))

            logger.info("Imported calibration zip (%d files)", len(names))
            return {"ok": True, "files": len(names)}
        except zipfile.BadZipFile:
            return {"ok": False, "error": "Invalid zip file"}
        except Exception as exc:
            logger.warning("Calibration import failed", exc_info=True)
            return {"ok": False, "error": str(exc)}

    def _load_persisted_config(self) -> None:
        """Load last-known projector config from disk."""
        try:
            if _PROJECTOR_CONFIG_PATH.is_file():
                self._client_config = json.loads(
                    _PROJECTOR_CONFIG_PATH.read_text(encoding="utf-8")
                )
                logger.info(
                    "Loaded persisted projector config: %s", self._client_config
                )
        except Exception:
            logger.debug("No persisted projector config found")

    def _persist_config(self) -> None:
        """Save current client config to disk."""
        if self._client_config is not None:
            try:
                _PROJECTOR_CONFIG_PATH.write_text(
                    json.dumps(self._client_config, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                logger.debug("Failed to persist projector config", exc_info=True)

    def start(self) -> None:
        """Start the HTTP server on a background thread."""
        if self._running:
            return

        streamer = self  # closure reference for handler

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self_handler) -> None:  # noqa: N805
                path = self_handler.path.split("?")[0]
                if path == "/stream":
                    self_handler._handle_stream()
                elif path == "/frame":
                    self_handler._handle_frame()
                elif path == "/config":
                    self_handler._handle_get_config()
                elif path == "/projector":
                    self_handler._handle_projector()
                elif path == "/calibration/status":
                    self_handler._handle_calibration_status()
                elif path.startswith("/calibration/download/"):
                    self_handler._handle_calibration_download(path)
                elif path.startswith("/calibration/preview/"):
                    self_handler._handle_calibration_preview(path)
                elif path == "/input-frame":
                    self_handler._handle_input_frame()
                elif path == "/input-stream":
                    self_handler._handle_input_stream()
                elif path == "/upload/status":
                    self_handler._handle_upload_status()
                elif path == "/calibration/export":
                    self_handler._handle_calibration_export()
                elif path == "/api/params":
                    self_handler._handle_get_params()
                elif path == "/scope/ice-servers":
                    self_handler._handle_scope_proxy("GET", "/api/v1/webrtc/ice-servers")
                else:
                    self_handler._handle_control_panel()

            def do_POST(self_handler) -> None:  # noqa: N805
                path = self_handler.path.split("?")[0]
                if path == "/config":
                    self_handler._handle_post_config()
                elif path == "/upload":
                    self_handler._handle_upload()
                elif path == "/calibration/import":
                    self_handler._handle_calibration_import()
                elif path == "/calibrate/start":
                    self_handler._handle_calibrate_start()
                elif path == "/calibrate/frame":
                    self_handler._handle_calibrate_frame()
                elif path == "/calibrate/stop":
                    self_handler._handle_calibrate_stop()
                elif path == "/api/params":
                    self_handler._handle_post_params()
                elif path == "/scope/offer":
                    self_handler._handle_scope_proxy_post("/api/v1/webrtc/offer")
                else:
                    self_handler.send_response(404)
                    self_handler.end_headers()

            def do_PATCH(self_handler) -> None:  # noqa: N805
                path = self_handler.path.split("?")[0]
                if path.startswith("/scope/ice/"):
                    scope_session = path[len("/scope/ice/"):]
                    self_handler._handle_scope_proxy_post(
                        f"/api/v1/webrtc/offer/{scope_session}",
                        method="PATCH",
                    )
                else:
                    self_handler.send_response(404)
                    self_handler.end_headers()

            def do_OPTIONS(self_handler) -> None:  # noqa: N805
                self_handler.send_response(204)
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
                self_handler.send_header("Access-Control-Allow-Headers", "Content-Type")
                self_handler.end_headers()

            def _handle_stream(self_handler) -> None:  # noqa: N805
                """MJPEG multipart stream."""
                with streamer._stream_client_lock:
                    streamer._stream_client_count += 1
                self_handler.send_response(200)
                self_handler.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}",
                )
                self_handler.send_header("Cache-Control", "no-cache, no-store")
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.end_headers()
                try:
                    while streamer._running:
                        streamer._new_frame.wait(timeout=1.0)
                        streamer._new_frame.clear()
                        with streamer._lock:
                            jpeg = streamer._frame_jpeg
                        if jpeg is None:
                            continue
                        self_handler.wfile.write(b"--" + _BOUNDARY + b"\r\n")
                        self_handler.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self_handler.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n".encode()
                        )
                        self_handler.wfile.write(b"\r\n")
                        self_handler.wfile.write(jpeg)
                        self_handler.wfile.write(b"\r\n")
                        self_handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with streamer._stream_client_lock:
                        streamer._stream_client_count = max(0, streamer._stream_client_count - 1)

            def _handle_frame(self_handler) -> None:  # noqa: N805
                """Single JPEG snapshot."""
                with streamer._lock:
                    jpeg = streamer._frame_jpeg
                if jpeg is not None:
                    self_handler.send_response(200)
                    self_handler.send_header("Content-Type", "image/jpeg")
                    self_handler.send_header("Content-Length", str(len(jpeg)))
                    self_handler.send_header("Cache-Control", "no-cache")
                    self_handler.end_headers()
                    self_handler.wfile.write(jpeg)
                else:
                    self_handler.send_response(204)
                    self_handler.end_headers()

            def _handle_projector(self_handler) -> None:  # noqa: N805
                """Fullscreen projector viewer — drag to projector, click for fullscreen.

                Auto-POSTs screen resolution to /config every 30s.
                """
                html = _PROJECTOR_HTML
                body = html.encode()
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "text/html")
                self_handler.send_header("Content-Length", str(len(body)))
                self_handler.end_headers()
                self_handler.wfile.write(body)

            def _handle_control_panel(self_handler) -> None:  # noqa: N805
                """Control panel dashboard — calibration status, projector, downloads."""
                body = _CONTROL_PANEL_HTML.encode()
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "text/html")
                self_handler.send_header("Content-Length", str(len(body)))
                self_handler.end_headers()
                self_handler.wfile.write(body)

            def _handle_get_params(self_handler) -> None:  # noqa: N805
                """Return current dashboard parameter overrides."""
                body = json.dumps(streamer._param_overrides).encode()
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.send_header("Content-Length", str(len(body)))
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.end_headers()
                self_handler.wfile.write(body)

            def _handle_post_params(self_handler) -> None:  # noqa: N805
                """Set dashboard parameter overrides."""
                length = int(self_handler.headers.get("Content-Length", 0))
                raw = self_handler.rfile.read(length) if length > 0 else b"{}"
                try:
                    params = json.loads(raw)
                except Exception:
                    params = {}
                # Merge into overrides (None values delete keys)
                for k, v in params.items():
                    if v is None:
                        streamer._param_overrides.pop(k, None)
                    else:
                        streamer._param_overrides[k] = v
                body = json.dumps({"ok": True, "params": streamer._param_overrides}).encode()
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.send_header("Content-Length", str(len(body)))
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.end_headers()
                self_handler.wfile.write(body)

            # -- Scope WebRTC proxy endpoints ------------------------------------

            def _handle_scope_proxy(self_handler, method: str, scope_path: str) -> None:  # noqa: N805
                """Proxy a GET request to Scope's API (localhost:8000)."""
                import os
                scope_base = "http://localhost:8000"
                url = scope_base + scope_path
                try:
                    req = urllib.request.Request(url, method=method)
                    req.add_header("Accept", "application/json")
                    # Add RunPod headers if needed
                    pod_id = os.environ.get("RUNPOD_POD_ID", "")
                    if pod_id:
                        origin = f"https://{pod_id}-8000.proxy.runpod.net"
                        req.add_header("Referer", origin + "/")
                        req.add_header("Origin", origin)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        data = resp.read()
                        self_handler.send_response(resp.status)
                        self_handler.send_header("Content-Type", "application/json")
                        self_handler.send_header("Content-Length", str(len(data)))
                        self_handler.send_header("Access-Control-Allow-Origin", "*")
                        self_handler.end_headers()
                        self_handler.wfile.write(data)
                except Exception as exc:
                    logger.warning("Scope proxy GET %s failed: %s", scope_path, exc)
                    body = json.dumps({"error": str(exc)}).encode()
                    self_handler.send_response(502)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(body)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(body)

            def _handle_scope_proxy_post(self_handler, scope_path: str, method: str = "POST") -> None:  # noqa: N805
                """Proxy a POST/PATCH request to Scope's API."""
                import os
                scope_base = "http://localhost:8000"
                url = scope_base + scope_path
                length = int(self_handler.headers.get("Content-Length", 0))
                raw = self_handler.rfile.read(length) if length > 0 else b""
                try:
                    req = urllib.request.Request(url, data=raw, method=method)
                    req.add_header("Content-Type", "application/json")
                    req.add_header("Accept", "application/json")
                    pod_id = os.environ.get("RUNPOD_POD_ID", "")
                    if pod_id:
                        origin = f"https://{pod_id}-8000.proxy.runpod.net"
                        req.add_header("Referer", origin + "/")
                        req.add_header("Origin", origin)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        data = resp.read()
                        self_handler.send_response(resp.status)
                        self_handler.send_header("Content-Type", "application/json")
                        self_handler.send_header("Content-Length", str(len(data)))
                        self_handler.send_header("Access-Control-Allow-Origin", "*")
                        self_handler.end_headers()
                        self_handler.wfile.write(data)
                except Exception as exc:
                    logger.warning("Scope proxy %s %s failed: %s", method, scope_path, exc)
                    body = json.dumps({"error": str(exc)}).encode()
                    self_handler.send_response(502)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(body)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(body)

            def _handle_get_config(self_handler) -> None:  # noqa: N805
                """Return current projector config as JSON."""
                cfg = streamer._client_config or {}
                body = json.dumps(cfg).encode()
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.send_header("Content-Length", str(len(body)))
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.end_headers()
                self_handler.wfile.write(body)

            def _handle_post_config(self_handler) -> None:  # noqa: N805
                """Receive projector config from companion app."""
                try:
                    length = int(
                        self_handler.headers.get("Content-Length", 0)
                    )
                    body = self_handler.rfile.read(length)
                    data = json.loads(body)
                    streamer._client_config = data
                    streamer._persist_config()
                    logger.info("Received projector config: %s", data)
                    self_handler.send_response(200)
                    self_handler.send_header(
                        "Access-Control-Allow-Origin", "*"
                    )
                    self_handler.end_headers()
                except Exception:
                    logger.warning(
                        "Bad POST /config payload", exc_info=True
                    )
                    self_handler.send_response(400)
                    self_handler.end_headers()

            def _handle_calibration_status(self_handler) -> None:  # noqa: N805
                """Return calibration completion status, progress, and file list."""
                data = {
                    "complete": streamer._calibration_complete,
                    "active": streamer._calibration_active,
                    "progress": streamer._calibration_progress,
                    "phase": streamer._calibration_phase,
                    "pattern_info": streamer._calibration_pattern_info,
                    "errors": streamer._calibration_errors,
                    "coverage_pct": streamer._calibration_coverage_pct,
                    "files": list(streamer._calibration_files.keys()),
                    "timestamp": streamer._calibration_timestamp,
                }
                body = json.dumps(data).encode()
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.send_header("Content-Length", str(len(body)))
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.send_header("Cache-Control", "no-cache")
                self_handler.end_headers()
                self_handler.wfile.write(body)

            def _handle_calibration_download(self_handler, path: str) -> None:  # noqa: N805
                """Serve a calibration result file for download."""
                name = unquote(path.split("/calibration/download/", 1)[-1])
                data = streamer._calibration_files.get(name)
                if data is None:
                    self_handler.send_response(404)
                    self_handler.end_headers()
                    return
                # Determine content type
                if name.endswith(".json"):
                    ct = "application/json"
                elif name.endswith(".png"):
                    ct = "image/png"
                else:
                    ct = "application/octet-stream"
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", ct)
                self_handler.send_header("Content-Length", str(len(data)))
                self_handler.send_header(
                    "Content-Disposition", f'attachment; filename="{name}"'
                )
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.end_headers()
                self_handler.wfile.write(data)

            def _handle_calibration_preview(self_handler, path: str) -> None:  # noqa: N805
                """Serve a calibration result image inline (for thumbnails)."""
                name = unquote(path.split("/calibration/preview/", 1)[-1])
                data = streamer._calibration_files.get(name)
                if data is None or not name.endswith(".png"):
                    self_handler.send_response(404)
                    self_handler.end_headers()
                    return
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "image/png")
                self_handler.send_header("Content-Length", str(len(data)))
                self_handler.send_header("Content-Disposition", "inline")
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.send_header("Cache-Control", "no-cache")
                self_handler.end_headers()
                self_handler.wfile.write(data)

            def _handle_input_frame(self_handler) -> None:  # noqa: N805
                """Single JPEG snapshot of the VACE input preview."""
                with streamer._input_lock:
                    jpeg = streamer._input_preview_jpeg
                if jpeg is not None:
                    self_handler.send_response(200)
                    self_handler.send_header("Content-Type", "image/jpeg")
                    self_handler.send_header("Content-Length", str(len(jpeg)))
                    self_handler.send_header("Cache-Control", "no-cache")
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(jpeg)
                else:
                    self_handler.send_response(204)
                    self_handler.end_headers()

            def _handle_input_stream(self_handler) -> None:  # noqa: N805
                """MJPEG stream of the VACE input preview."""
                self_handler.send_response(200)
                self_handler.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}",
                )
                self_handler.send_header("Cache-Control", "no-cache, no-store")
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.end_headers()
                try:
                    while streamer._running:
                        streamer._input_new_frame.wait(timeout=1.0)
                        streamer._input_new_frame.clear()
                        with streamer._input_lock:
                            jpeg = streamer._input_preview_jpeg
                        if jpeg is None:
                            continue
                        self_handler.wfile.write(b"--" + _BOUNDARY + b"\r\n")
                        self_handler.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self_handler.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n".encode()
                        )
                        self_handler.wfile.write(b"\r\n")
                        self_handler.wfile.write(jpeg)
                        self_handler.wfile.write(b"\r\n")
                        self_handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def _handle_upload(self_handler) -> None:  # noqa: N805
                """Handle custom depth/mask upload via POST /upload."""
                try:
                    from urllib.parse import parse_qs, urlparse
                    parsed = urlparse(self_handler.path)
                    params = parse_qs(parsed.query)
                    stage = params.get("stage", ["depth_warped"])[0]
                    upload_type = params.get("type", ["depth"])[0]

                    length = int(self_handler.headers.get("Content-Length", 0))
                    if length <= 0 or length > 50 * 1024 * 1024:
                        self_handler.send_response(400)
                        self_handler.end_headers()
                        return

                    data = self_handler.rfile.read(length)
                    result = streamer.set_custom_upload(data, stage, upload_type)

                    body = json.dumps(result).encode()
                    status = 200 if result.get("ok") else 400
                    self_handler.send_response(status)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(body)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(body)
                except Exception:
                    logger.warning("Upload failed", exc_info=True)
                    self_handler.send_response(500)
                    self_handler.end_headers()

            def _handle_upload_status(self_handler) -> None:  # noqa: N805
                """Return custom upload status as JSON."""
                data = streamer.get_upload_status()
                body = json.dumps(data).encode()
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.send_header("Content-Length", str(len(body)))
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.end_headers()
                self_handler.wfile.write(body)

            def _handle_calibration_export(self_handler) -> None:  # noqa: N805
                """Export all calibration files as a single zip download."""
                zip_bytes = streamer.export_calibration_zip()
                if zip_bytes is None:
                    body = json.dumps({"error": "No calibration found"}).encode()
                    self_handler.send_response(404)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(body)))
                    self_handler.end_headers()
                    self_handler.wfile.write(body)
                    return
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "application/zip")
                self_handler.send_header("Content-Length", str(len(zip_bytes)))
                self_handler.send_header(
                    "Content-Disposition",
                    'attachment; filename="projectionmapanything_calibration.zip"',
                )
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.end_headers()
                self_handler.wfile.write(zip_bytes)

            def _handle_calibration_import(self_handler) -> None:  # noqa: N805
                """Import a calibration zip archive via POST."""
                try:
                    length = int(self_handler.headers.get("Content-Length", 0))
                    if length <= 0 or length > 200 * 1024 * 1024:
                        self_handler.send_response(400)
                        self_handler.end_headers()
                        return
                    data = self_handler.rfile.read(length)
                    result = streamer.import_calibration_zip(data)
                    body = json.dumps(result).encode()
                    status = 200 if result.get("ok") else 400
                    self_handler.send_response(status)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(body)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(body)
                except Exception:
                    logger.warning("Calibration import failed", exc_info=True)
                    self_handler.send_response(500)
                    self_handler.end_headers()

            def _handle_calibrate_start(self_handler) -> None:  # noqa: N805
                """Start standalone calibration from browser webcam."""
                try:
                    length = int(self_handler.headers.get("Content-Length", 0))
                    body = self_handler.rfile.read(length) if length > 0 else b"{}"
                    cfg = json.loads(body) if body else {}
                    result = streamer.start_standalone_calibration(
                        proj_w=int(cfg.get("proj_w", 1920)),
                        proj_h=int(cfg.get("proj_h", 1080)),
                        settle_frames=int(cfg.get("settle_frames", 15)),
                        capture_frames=int(cfg.get("capture_frames", 3)),
                        max_brightness=int(cfg.get("max_brightness", 128)),
                    )
                    resp = json.dumps(result).encode()
                    status = 200 if result.get("ok") else 409
                    self_handler.send_response(status)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(resp)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(resp)
                except Exception:
                    logger.warning("calibrate/start failed", exc_info=True)
                    self_handler.send_response(500)
                    self_handler.end_headers()

            def _handle_calibrate_frame(self_handler) -> None:  # noqa: N805
                """Process one webcam frame for standalone calibration."""
                try:
                    length = int(self_handler.headers.get("Content-Length", 0))
                    if length <= 0 or length > 10 * 1024 * 1024:
                        self_handler.send_response(400)
                        self_handler.end_headers()
                        return
                    jpeg_bytes = self_handler.rfile.read(length)
                    result = streamer.step_standalone_calibration(jpeg_bytes)
                    resp = json.dumps(result).encode()
                    self_handler.send_response(200)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(resp)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(resp)
                except Exception:
                    logger.warning("calibrate/frame failed", exc_info=True)
                    self_handler.send_response(500)
                    self_handler.end_headers()

            def _handle_calibrate_stop(self_handler) -> None:  # noqa: N805
                """Stop standalone calibration."""
                try:
                    # Read body if any (may be empty)
                    length = int(self_handler.headers.get("Content-Length", 0))
                    if length > 0:
                        self_handler.rfile.read(length)
                    result = streamer.stop_standalone_calibration()
                    resp = json.dumps(result).encode()
                    self_handler.send_response(200)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(resp)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(resp)
                except Exception:
                    logger.warning("calibrate/stop failed", exc_info=True)
                    self_handler.send_response(500)
                    self_handler.end_headers()

            def log_message(self_handler, format, *args) -> None:  # noqa: N805
                # Suppress per-request HTTP logging — the MJPEG stream fires
                # dozens of requests per second and would flood stdout.
                pass

        self._running = True
        self._server = _ThreadedHTTPServer(("0.0.0.0", self._port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="frame-streamer", daemon=True
        )
        self._thread.start()

        logger.info(
            "FrameStreamer: MJPEG server started on port %d "
            "(endpoints: /stream, /frame, /config, /)",
            self._port,
        )

    def submit_frame(
        self, rgb: np.ndarray, target_size: tuple[int, int] | None = None,
    ) -> None:
        """Submit an RGB uint8 (H, W, 3) frame for streaming.

        Uses a try-lock so the pipeline thread is never blocked: if a
        previous encode is still in progress the frame is silently dropped.
        Suppressed when ``calibration_active`` is True.

        Parameters
        ----------
        rgb : np.ndarray
            RGB uint8 (H, W, 3) frame.
        target_size : tuple[int, int] | None
            Optional (width, height) to resize to before encoding.

        Thread-safe — may be called from any thread.
        """
        if not self._running or self._calibration_active:
            return
        # Skip encoding entirely when no one needs the frames:
        # no MJPEG stream clients AND dashboard preview is off.
        if (
            self._stream_client_count == 0
            and not self._param_overrides.get("preview_enabled", True)
        ):
            return
        # Try-lock: skip this frame if we're already encoding one
        if not self._encoding.acquire(blocking=False):
            return
        try:
            if target_size is not None:
                tw, th = target_size
                h, w = rgb.shape[:2]
                if (w, h) != (tw, th):
                    rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
            jpeg = self._encode_jpeg(rgb)
            if jpeg is not None:
                with self._lock:
                    self._frame_jpeg = jpeg
                self._new_frame.set()
        finally:
            self._encoding.release()

    def submit_calibration_frame(self, rgb: np.ndarray) -> None:
        """Submit a calibration pattern frame. Always accepted.

        Encodes synchronously since calibration patterns are infrequent
        and need to arrive reliably.

        Thread-safe — may be called from any thread.
        """
        if not self._running:
            return
        jpeg = self._encode_jpeg(rgb)
        if jpeg is not None:
            with self._lock:
                self._frame_jpeg = jpeg
            self._new_frame.set()

    def submit_input_preview(self, rgb: np.ndarray) -> None:
        """Submit a VACE input preview frame (preprocessor output).

        Encodes synchronously.  Does not block the pipeline thread if the
        input lock is already held (frame is dropped instead).
        Skipped when dashboard input preview is disabled.

        Thread-safe — may be called from any thread.
        """
        if not self._running:
            return
        if not self._param_overrides.get("input_preview_enabled", True):
            return
        jpeg = self._encode_jpeg(rgb)
        if jpeg is not None:
            with self._input_lock:
                self._input_preview_jpeg = jpeg
            self._input_new_frame.set()

    def _encode_jpeg(self, rgb: np.ndarray) -> bytes | None:
        """Encode an RGB uint8 array as JPEG. Returns bytes or None on failure."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, jpeg_buf = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        )
        return jpeg_buf.tobytes() if ok else None

    def stop(self) -> None:
        """Shut down the server and join the server thread."""
        self._running = False
        self._new_frame.set()
        self._input_new_frame.set()
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("FrameStreamer: stopped")

    def __del__(self) -> None:
        if self._running:
            self.stop()


# ── Module-level singleton ──────────────────────────────────────────────────

_shared_streamer: FrameStreamer | None = None
_shared_lock = threading.Lock()


def get_or_create_streamer(port: int = 8765) -> FrameStreamer:
    """Return the shared FrameStreamer, creating it if necessary.

    If a streamer already exists on a different port, it is stopped and
    replaced.  Both preprocessor and postprocessor should call this to
    share a single MJPEG server.
    """
    global _shared_streamer
    with _shared_lock:
        if _shared_streamer is not None:
            if _shared_streamer.port == port and _shared_streamer.is_running:
                return _shared_streamer
            # Port changed or not running — tear down old one
            _shared_streamer.stop()
            _shared_streamer = None

        streamer = FrameStreamer(port=port)
        streamer.start()
        _shared_streamer = streamer
        return streamer
