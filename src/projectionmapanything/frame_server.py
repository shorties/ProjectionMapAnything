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


# ── Test card images ────────────────────────────────────────────────────────

_PKG_DIR = Path(__file__).parent
_TESTCARD_16_9_PATH = _PKG_DIR / "testcard_16_9.jpg"
_TESTCARD_4_3_PATH = _PKG_DIR / "testcard_4_3.jpg"


def _load_testcard(target_w: int, target_h: int) -> np.ndarray:
    """Load the appropriate test card for the given resolution.

    Picks 16:9 or 4:3 based on nearest aspect ratio, then center-crops
    and resizes to match the target resolution exactly.
    """
    target_ar = target_w / max(target_h, 1)
    # Pick closest aspect ratio
    if abs(target_ar - 16 / 9) <= abs(target_ar - 4 / 3):
        card_path = _TESTCARD_16_9_PATH
    else:
        card_path = _TESTCARD_4_3_PATH

    if card_path.is_file():
        bgr = cv2.imread(str(card_path))
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ch, cw = rgb.shape[:2]
            card_ar = cw / max(ch, 1)
            # Center-crop to target aspect ratio
            if card_ar > target_ar:
                # Card is wider — crop sides
                new_w = int(ch * target_ar)
                x0 = (cw - new_w) // 2
                rgb = rgb[:, x0:x0 + new_w]
            elif card_ar < target_ar:
                # Card is taller — crop top/bottom
                new_h = int(cw / target_ar)
                y0 = (ch - new_h) // 2
                rgb = rgb[y0:y0 + new_h, :]
            # Resize to exact target
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return rgb

    # Fallback: grey card with border
    card = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    cv2.rectangle(card, (2, 2), (target_w - 3, target_h - 3), (200, 200, 200), 1)
    return card


# ── HTML templates ───────────────────────────────────────────────────────────

_PROJECTOR_HTML = """\
<!DOCTYPE html>
<html><head>
<title>Projection-Map-Anything Projector</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #000; overflow: hidden; width: 100vw; height: 100vh;
    cursor: pointer;
  }
  body.fs { cursor: none; }
  #webrtc, #mjpeg {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    object-fit: contain; background: #000;
  }
  /* Default: MJPEG on top (fallback until WebRTC connects) */
  #mjpeg  { z-index: 2; }
  #webrtc { z-index: 1; }
  #restore {
    display: none; position: fixed; bottom: 0; left: 0; right: 0;
    text-align: center; padding: 6px;
    background: rgba(0,0,0,0.6); color: #888;
    font: 12px/1.2 sans-serif; z-index: 10; pointer-events: none;
  }
  #restore.show { display: block; }
  #restore.fade { opacity: 0; transition: opacity 0.5s; }
</style>
</head><body>
<video id="webrtc" autoplay playsinline muted></video>
<img id="mjpeg" src="/stream" />
<div id="restore">Click or press any key to restore fullscreen</div>

<script>
const webrtcEl = document.getElementById('webrtc');
const mjpegEl = document.getElementById('mjpeg');
const restoreBanner = document.getElementById('restore');

// ---- Layer state ----
let calibActive = false;
let webrtcConnected = false;
let pc = null;           // RTCPeerConnection
let rtcRetryTimer = null;

function updateLayers() {
  if (calibActive) {
    // Calibration: MJPEG shows patterns on top
    mjpegEl.style.zIndex = '2';
    webrtcEl.style.zIndex = '1';
  } else if (webrtcConnected) {
    // Normal + WebRTC: AI output on top
    webrtcEl.style.zIndex = '2';
    mjpegEl.style.zIndex = '1';
  } else {
    // Fallback: MJPEG on top
    mjpegEl.style.zIndex = '2';
    webrtcEl.style.zIndex = '1';
  }
}

// ---- WebRTC connection ----
async function connectWebRTC() {
  // Tear down previous connection
  if (pc) {
    pc.oniceconnectionstatechange = null;
    pc.ontrack = null;
    pc.close();
    pc = null;
  }
  webrtcConnected = false;
  updateLayers();

  try {
    // 1. Get ICE servers
    const iceResp = await fetch('/scope/ice-servers');
    if (!iceResp.ok) throw new Error('ICE servers: ' + iceResp.status);
    const iceData = await iceResp.json();

    // 2. Create peer connection
    pc = new RTCPeerConnection(iceData);

    // 3. Add recvonly video transceiver
    pc.addTransceiver('video', { direction: 'recvonly' });

    // 4. Track event — attach stream to video element
    pc.ontrack = (ev) => {
      if (ev.streams && ev.streams[0]) {
        webrtcEl.srcObject = ev.streams[0];
      } else {
        const s = new MediaStream();
        s.addTrack(ev.track);
        webrtcEl.srcObject = s;
      }
    };

    // 5. ICE connection state monitoring
    pc.oniceconnectionstatechange = () => {
      const st = pc.iceConnectionState;
      if (st === 'connected' || st === 'completed') {
        webrtcConnected = true;
        updateLayers();
      } else if (st === 'disconnected' || st === 'failed' || st === 'closed') {
        webrtcConnected = false;
        updateLayers();
        scheduleRTCRetry(3000);
      }
    };

    // 6. Create offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // 7. Send offer to Scope via our proxy
    const offerResp = await fetch('/scope/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
    });
    if (!offerResp.ok) throw new Error('Offer: ' + offerResp.status);
    const answer = await offerResp.json();

    // 8. Set remote description
    await pc.setRemoteDescription(new RTCSessionDescription({
      type: answer.type,
      sdp: answer.sdp,
    }));

    // 9. Trickle ICE candidates
    const sessionId = answer.sessionId;
    if (sessionId) {
      pc.onicecandidate = (ev) => {
        if (ev.candidate) {
          fetch('/scope/ice/' + sessionId, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              candidates: [{
                candidate: ev.candidate.candidate,
                sdpMid: ev.candidate.sdpMid,
                sdpMLineIndex: ev.candidate.sdpMLineIndex,
              }],
            }),
          }).catch(() => {});
        }
      };
    }
  } catch (err) {
    // Scope not running or network error — retry later
    webrtcConnected = false;
    updateLayers();
    scheduleRTCRetry(10000);
  }
}

function scheduleRTCRetry(ms) {
  clearTimeout(rtcRetryTimer);
  rtcRetryTimer = setTimeout(connectWebRTC, ms);
}

// Start WebRTC connection
connectWebRTC();

// ---- Calibration status polling ----
function pollCalibration() {
  fetch('/calibration/status')
    .then(r => r.json())
    .then(data => {
      const wasActive = calibActive;
      calibActive = !!data.active;
      if (calibActive !== wasActive) updateLayers();
    })
    .catch(() => {});
}
setInterval(pollCalibration, 1000);
pollCalibration();

// ---- Fullscreen state tracking ----
let wasFullscreen = false;
let fadeTimer = null;

function goFullscreen() {
  document.documentElement.requestFullscreen().catch(() => {});
}

function showBanner() {
  restoreBanner.classList.add('show');
  restoreBanner.classList.remove('fade');
  clearTimeout(fadeTimer);
  fadeTimer = setTimeout(() => { restoreBanner.classList.add('fade'); }, 4000);
}

function hideBanner() {
  restoreBanner.classList.remove('show', 'fade');
  clearTimeout(fadeTimer);
}

function isWindowFullscreen() {
  return (
    window.outerWidth >= screen.width - 2 &&
    window.outerHeight >= screen.height - 2
  );
}

function updateFSClass() {
  const apiFS = !!document.fullscreenElement;
  const windowFS = isWindowFullscreen();
  document.body.classList.toggle('fs', apiFS || windowFS);
}

// ---- Fullscreen change tracking ----
document.addEventListener('fullscreenchange', () => {
  const isFS = !!document.fullscreenElement;
  updateFSClass();
  if (isFS) {
    wasFullscreen = true;
    hideBanner();
  } else if (wasFullscreen) {
    if (!isWindowFullscreen()) showBanner();
  }
  setTimeout(postConfig, 300);
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    wasFullscreen = false;
    hideBanner();
  }
});

// ---- Restore fullscreen on user activation ----
function tryRestore(e) {
  if (wasFullscreen && !document.fullscreenElement) {
    goFullscreen();
  }
}
document.addEventListener('click', tryRestore);
document.addEventListener('keydown', tryRestore);
document.addEventListener('pointerdown', tryRestore);
document.addEventListener('touchstart', tryRestore);

document.body.addEventListener('click', () => {
  if (!document.fullscreenElement) goFullscreen();
});

window.addEventListener('focus', () => {
  updateFSClass();
  if (wasFullscreen && !document.fullscreenElement && !isWindowFullscreen()) {
    goFullscreen();
    showBanner();
  }
});

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    updateFSClass();
    if (wasFullscreen && !document.fullscreenElement && !isWindowFullscreen()) {
      goFullscreen();
      showBanner();
    }
  }
});

setInterval(updateFSClass, 2000);

// ---- MJPEG reconnect on error ----
mjpegEl.onerror = () => {
  setTimeout(() => { mjpegEl.src = '/stream?t=' + Date.now(); }, 1000);
};

// ---- Report projector resolution ----
function postConfig() {
  const apiFS = !!document.fullscreenElement;
  const windowFS = isWindowFullscreen();
  const isFS = apiFS || windowFS;
  const w = isFS ? screen.width : window.innerWidth;
  const h = isFS ? screen.height : window.innerHeight;
  fetch('/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ width: w, height: h, monitor_name: 'browser', fullscreen: isFS })
  }).catch(() => {});
}
postConfig();
setInterval(postConfig, 30000);
window.addEventListener('resize', () => {
  updateFSClass();
  setTimeout(postConfig, 500);
});
</script>
</body></html>
"""

_CONTROL_PANEL_HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


# ── Checkerboard intrinsics calibration utilities ────────────────────────────


def generate_checkerboard(
    proj_w: int,
    proj_h: int,
    board_cols: int,
    board_rows: int,
    square_px: int,
    offset_x: int,
    offset_y: int,
    white_level: int = 255,
) -> np.ndarray:
    """Generate a checkerboard image at projector resolution.

    Black background with white border around the checkerboard area.
    Returns (proj_h, proj_w, 3) uint8 RGB.
    """
    img = np.zeros((proj_h, proj_w), dtype=np.uint8)
    n_sq_x = board_cols + 1
    n_sq_y = board_rows + 1

    # White border (1 square-width padding)
    border = square_px
    bx0 = max(0, offset_x - border)
    by0 = max(0, offset_y - border)
    bx1 = min(proj_w, offset_x + n_sq_x * square_px + border)
    by1 = min(proj_h, offset_y + n_sq_y * square_px + border)
    img[by0:by1, bx0:bx1] = white_level

    # Black squares on top
    for r in range(n_sq_y):
        for c in range(n_sq_x):
            if (r + c) % 2 == 0:
                sx = offset_x + c * square_px
                sy = offset_y + r * square_px
                sx0 = max(0, min(sx, proj_w))
                sy0 = max(0, min(sy, proj_h))
                sx1 = max(0, min(sx + square_px, proj_w))
                sy1 = max(0, min(sy + square_px, proj_h))
                if sx1 > sx0 and sy1 > sy0:
                    img[sy0:sy1, sx0:sx1] = 0

    return np.stack([img] * 3, axis=-1)


def generate_checkerboard_positions(
    proj_w: int,
    proj_h: int,
    board_cols: int,
    board_rows: int,
    square_px: int,
) -> list[dict]:
    """Generate checkerboard positions spread across the projector.

    Returns list of dicts with offset_x, offset_y, and label.
    """
    n_sq_x = board_cols + 1
    n_sq_y = board_rows + 1
    board_w = n_sq_x * square_px
    board_h = n_sq_y * square_px
    border = square_px
    margin = border + square_px // 2

    min_x = margin
    max_x = max(min_x, proj_w - board_w - margin)
    min_y = margin
    max_y = max(min_y, proj_h - board_h - margin)
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2

    return [
        {"offset_x": cx, "offset_y": cy, "label": "center"},
        {"offset_x": min_x, "offset_y": min_y, "label": "top-left"},
        {"offset_x": max_x, "offset_y": min_y, "label": "top-right"},
        {"offset_x": min_x, "offset_y": max_y, "label": "bottom-left"},
        {"offset_x": max_x, "offset_y": max_y, "label": "bottom-right"},
        {"offset_x": cx, "offset_y": min_y, "label": "top-center"},
        {"offset_x": cx, "offset_y": max_y, "label": "bottom-center"},
    ]


def get_projected_corner_positions(
    board_cols: int,
    board_rows: int,
    square_px: int,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    """Get projector-pixel coordinates of inner corners.

    Returns (board_rows * board_cols, 1, 2) float32.
    """
    corners = np.zeros((board_rows * board_cols, 1, 2), dtype=np.float32)
    for r in range(board_rows):
        for c in range(board_cols):
            corners[r * board_cols + c, 0] = (
                offset_x + (c + 1) * square_px,
                offset_y + (r + 1) * square_px,
            )
    return corners


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
        self._calibration_quality = 97  # near-lossless for calibration patterns
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

    def get_calibration_file(self, name: str) -> bytes | None:
        """Get a calibration result file — in-memory first, disk fallback.

        Handles the race condition where the background publish thread
        hasn't populated ``_calibration_files`` yet but the files have
        been saved to disk by ``publish_calibration_results()``.
        """
        data = self._calibration_files.get(name)
        if data is not None:
            return data
        # Disk fallback
        if name == "calibration.json":
            if _CALIBRATION_JSON_PATH.is_file():
                return _CALIBRATION_JSON_PATH.read_bytes()
        elif name.endswith(".png"):
            path = _RESULTS_DIR / name
            if path.is_file():
                return path.read_bytes()
        return None

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

    # -- Projected checkerboard intrinsics calibration --------------------------

    def start_intrinsics_calibration(
        self,
        proj_w: int,
        proj_h: int,
        square_px: int | None = None,
    ) -> dict:
        """Begin projected checkerboard intrinsics calibration.

        Projects checkerboards at multiple positions. The pipeline calls
        ``step_intrinsics_calibration()`` each frame with the camera image.

        Returns status dict with 'ok' and calibration parameters.
        """
        if square_px is None or square_px <= 0:
            square_px = min(proj_w, proj_h) // 12

        board_cols = 7
        board_rows = 5

        positions = generate_checkerboard_positions(
            proj_w, proj_h, board_cols, board_rows, square_px,
        )

        self._intrinsics_active = True
        self._intrinsics_proj_w = proj_w
        self._intrinsics_proj_h = proj_h
        self._intrinsics_square_px = square_px
        self._intrinsics_board_cols = board_cols
        self._intrinsics_board_rows = board_rows
        self._intrinsics_positions = positions
        self._intrinsics_current_idx = 0
        self._intrinsics_captures: list[dict] = []
        self._intrinsics_settle_start = 0.0
        self._intrinsics_settled = False
        self._intrinsics_timeout_start = 0.0
        self._intrinsics_result: dict | None = None
        self._intrinsics_error: str | None = None
        self._intrinsics_detected = 0
        self._intrinsics_skipped = 0

        # Show first checkerboard pattern
        self._calibration_active = True
        self._submit_intrinsics_pattern()

        logger.info(
            "Intrinsics calibration started: %dx%d, %dx%d board, "
            "square=%dpx, %d positions",
            proj_w, proj_h, board_cols, board_rows,
            square_px, len(positions),
        )
        return {
            "ok": True,
            "square_px": square_px,
            "positions": len(positions),
            "board_size": [board_cols, board_rows],
        }

    def stop_intrinsics_calibration(self) -> None:
        """Cancel intrinsics calibration."""
        self._intrinsics_active = False
        self._calibration_active = False
        logger.info("Intrinsics calibration cancelled")

    def step_intrinsics_calibration(
        self, camera_frame: np.ndarray,
    ) -> np.ndarray | None:
        """Process one frame during intrinsics calibration.

        Parameters
        ----------
        camera_frame : np.ndarray
            uint8 (H, W, 3) RGB camera image.

        Returns
        -------
        np.ndarray | None
            The checkerboard pattern to project (H, W, 3) uint8, or None
            if calibration is complete.
        """
        import time as _t

        if not getattr(self, "_intrinsics_active", False):
            return None

        idx = self._intrinsics_current_idx
        positions = self._intrinsics_positions

        # All positions done — finish
        if idx >= len(positions):
            self._finish_intrinsics_calibration()
            return None

        now = _t.monotonic()

        # Settle check — wait 500ms after pattern change
        if not self._intrinsics_settled:
            if self._intrinsics_settle_start == 0.0:
                self._intrinsics_settle_start = now
                self._intrinsics_timeout_start = now
            if now - self._intrinsics_settle_start < 0.5:
                return self._get_intrinsics_pattern()
            self._intrinsics_settled = True
            self._intrinsics_timeout_start = now

        # Detect corners
        gray = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2GRAY)
        board_size = (self._intrinsics_board_cols, self._intrinsics_board_rows)
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        found, corners = cv2.findChessboardCorners(gray, board_size, flags)

        if found and corners is not None:
            # Sub-pixel refinement
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                30, 0.001,
            )
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria,
            )

            # Build object points
            sq = self._intrinsics_square_px
            cols = self._intrinsics_board_cols
            rows = self._intrinsics_board_rows
            obj_pts = np.zeros((rows * cols, 3), dtype=np.float32)
            for r in range(rows):
                for c in range(cols):
                    obj_pts[r * cols + c] = (c * sq, r * sq, 0.0)

            # Get projected corner positions
            pos = positions[idx]
            proj_corners = get_projected_corner_positions(
                cols, rows, sq, pos["offset_x"], pos["offset_y"],
            )

            self._intrinsics_captures.append({
                "obj_pts": obj_pts,
                "img_pts": corners,
                "proj_pts": proj_corners,
                "cam_size": (gray.shape[1], gray.shape[0]),
                "label": pos["label"],
            })
            self._intrinsics_detected += 1

            logger.info(
                "Intrinsics: position %d/%d (%s) — %d corners detected",
                idx + 1, len(positions), pos["label"], len(corners),
            )

            # Advance to next position
            self._intrinsics_current_idx += 1
            self._intrinsics_settled = False
            self._intrinsics_settle_start = 0.0
            if self._intrinsics_current_idx < len(positions):
                self._submit_intrinsics_pattern()
                return self._get_intrinsics_pattern()
            else:
                self._finish_intrinsics_calibration()
                return None

        # Timeout: skip after 3s
        if now - self._intrinsics_timeout_start > 3.0:
            pos = positions[idx]
            self._intrinsics_skipped += 1
            logger.warning(
                "Intrinsics: position %d/%d (%s) — detection timeout, skipping",
                idx + 1, len(positions), pos["label"],
            )
            self._intrinsics_current_idx += 1
            self._intrinsics_settled = False
            self._intrinsics_settle_start = 0.0
            if self._intrinsics_current_idx < len(positions):
                self._submit_intrinsics_pattern()
                return self._get_intrinsics_pattern()
            else:
                self._finish_intrinsics_calibration()
                return None

        return self._get_intrinsics_pattern()

    def _submit_intrinsics_pattern(self) -> None:
        """Submit the current checkerboard pattern to the MJPEG stream."""
        pattern = self._get_intrinsics_pattern()
        if pattern is not None:
            self.submit_calibration_frame(pattern)

    def _get_intrinsics_pattern(self) -> np.ndarray | None:
        """Generate the current checkerboard pattern image."""
        idx = self._intrinsics_current_idx
        positions = self._intrinsics_positions
        if idx >= len(positions):
            return None
        pos = positions[idx]
        return generate_checkerboard(
            self._intrinsics_proj_w,
            self._intrinsics_proj_h,
            self._intrinsics_board_cols,
            self._intrinsics_board_rows,
            self._intrinsics_square_px,
            pos["offset_x"],
            pos["offset_y"],
        )

    def _finish_intrinsics_calibration(self) -> None:
        """Run camera + projector intrinsics calibration from collected data."""
        from .calibration import save_camera_intrinsics, save_procam_intrinsics

        self._intrinsics_active = False
        self._calibration_active = False
        captures = self._intrinsics_captures

        if len(captures) < 3:
            self._intrinsics_error = (
                f"Need at least 3 detections, got {len(captures)}"
            )
            logger.warning("Intrinsics: %s", self._intrinsics_error)
            return

        # Unpack captured data
        all_obj_pts = [c["obj_pts"] for c in captures]
        all_img_pts = [c["img_pts"] for c in captures]
        all_proj_pts = [c["proj_pts"] for c in captures]
        cam_size = captures[0]["cam_size"]
        proj_size = (self._intrinsics_proj_w, self._intrinsics_proj_h)

        # Phase 1: Camera intrinsics
        try:
            ret_cam, K_cam, dist_cam, rvecs, tvecs = cv2.calibrateCamera(
                all_obj_pts, all_img_pts, cam_size, None, None,
            )
        except Exception as exc:
            self._intrinsics_error = f"Camera calibration failed: {exc}"
            logger.error("Intrinsics: %s", self._intrinsics_error)
            return

        logger.info(
            "Intrinsics: camera fx=%.1f fy=%.1f cx=%.1f cy=%.1f err=%.4f",
            K_cam[0, 0], K_cam[1, 1], K_cam[0, 2], K_cam[1, 2], ret_cam,
        )

        # Phase 2: Projector intrinsics via 3D points from camera poses
        try:
            proj_obj_pts_3d = []
            for i in range(len(rvecs)):
                R, _ = cv2.Rodrigues(rvecs[i])
                t = tvecs[i].reshape(3)
                pts_cam = (R @ all_obj_pts[i].T).T + t
                proj_obj_pts_3d.append(pts_cam.astype(np.float32))

            # Initial guess for projector intrinsics
            K_proj_init = np.eye(3, dtype=np.float64)
            K_proj_init[0, 0] = K_proj_init[1, 1] = float(
                max(proj_size[0], proj_size[1])
            )
            K_proj_init[0, 2] = proj_size[0] / 2.0
            K_proj_init[1, 2] = proj_size[1] / 2.0

            ret_proj, K_proj, dist_proj, _, _ = cv2.calibrateCamera(
                proj_obj_pts_3d,
                all_proj_pts,
                proj_size,
                K_proj_init,
                None,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            )
        except Exception as exc:
            # Projector calibration failed — save camera only
            logger.warning(
                "Intrinsics: projector calibration failed: %s — "
                "saving camera intrinsics only",
                exc,
            )
            save_camera_intrinsics(K_cam, dist_cam, cam_size, ret_cam)
            self._intrinsics_result = {
                "cam_fx": float(K_cam[0, 0]),
                "cam_fy": float(K_cam[1, 1]),
                "cam_error": float(ret_cam),
                "detections": len(captures),
            }
            return

        logger.info(
            "Intrinsics: projector fx=%.1f fy=%.1f cx=%.1f cy=%.1f err=%.4f",
            K_proj[0, 0], K_proj[1, 1], K_proj[0, 2], K_proj[1, 2], ret_proj,
        )

        # Save both camera and projector intrinsics
        save_procam_intrinsics(
            K_cam, dist_cam, cam_size, ret_cam,
            K_proj, dist_proj, proj_size, ret_proj,
        )

        self._intrinsics_result = {
            "cam_fx": float(K_cam[0, 0]),
            "cam_fy": float(K_cam[1, 1]),
            "cam_error": float(ret_cam),
            "proj_fx": float(K_proj[0, 0]),
            "proj_fy": float(K_proj[1, 1]),
            "proj_error": float(ret_proj),
            "detections": len(captures),
        }
        logger.info("Intrinsics calibration complete: %s", self._intrinsics_result)

    def get_intrinsics_status(self) -> dict:
        """Return current intrinsics calibration status."""
        active = getattr(self, "_intrinsics_active", False)
        result = getattr(self, "_intrinsics_result", None)
        error = getattr(self, "_intrinsics_error", None)
        positions = getattr(self, "_intrinsics_positions", [])
        current = getattr(self, "_intrinsics_current_idx", 0)
        detected = getattr(self, "_intrinsics_detected", 0)
        skipped = getattr(self, "_intrinsics_skipped", 0)

        status: dict = {
            "active": active,
            "current": current,
            "total": len(positions),
            "detected": detected,
            "skipped": skipped,
        }
        if active and current < len(positions):
            status["label"] = positions[current].get("label", "")
        if result is not None:
            status["result"] = result
        if error is not None:
            status["error"] = error
        return status

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
                elif path == "/calibrate/intrinsics/status":
                    self_handler._handle_intrinsics_status()
                elif path == "/api/params":
                    self_handler._handle_get_params()
                elif path == "/scope/ice-servers":
                    self_handler._handle_scope_proxy("GET", "/api/v1/webrtc/ice-servers")
                elif path == "/scope/pipeline/status":
                    self_handler._handle_scope_proxy("GET", "/api/v1/pipeline/status")
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
                elif path == "/calibrate/intrinsics/start":
                    self_handler._handle_intrinsics_start()
                elif path == "/calibrate/intrinsics/stop":
                    self_handler._handle_intrinsics_stop()
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

            def do_HEAD(self_handler) -> None:  # noqa: N805
                """Handle HEAD requests — used by dashboard to check frame availability."""
                path = self_handler.path.split("?")[0]
                if path == "/input-frame":
                    with streamer._input_lock:
                        jpeg = streamer._input_preview_jpeg
                    if jpeg is not None:
                        self_handler.send_response(200)
                        self_handler.send_header("Content-Type", "image/jpeg")
                        self_handler.send_header("Content-Length", str(len(jpeg)))
                    else:
                        self_handler.send_response(204)
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                elif path == "/frame":
                    with streamer._lock:
                        jpeg = streamer._frame_jpeg
                    if jpeg is not None:
                        self_handler.send_response(200)
                        self_handler.send_header("Content-Type", "image/jpeg")
                        self_handler.send_header("Content-Length", str(len(jpeg)))
                    else:
                        self_handler.send_response(204)
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                else:
                    self_handler.send_response(405)
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
                # Build file list from in-memory dict, with disk fallback
                file_names = list(streamer._calibration_files.keys())
                if not file_names and not streamer._calibration_active:
                    # In-memory dict is empty — check disk for saved results
                    if _CALIBRATION_JSON_PATH.is_file():
                        file_names.append("calibration.json")
                    if _RESULTS_DIR.is_dir():
                        for f in _RESULTS_DIR.iterdir():
                            if f.is_file() and f.suffix == ".png":
                                file_names.append(f.name)
                    if file_names:
                        # Mark as complete since files exist on disk
                        streamer._calibration_complete = True
                data = {
                    "complete": streamer._calibration_complete,
                    "active": streamer._calibration_active,
                    "progress": streamer._calibration_progress,
                    "phase": streamer._calibration_phase,
                    "pattern_info": streamer._calibration_pattern_info,
                    "errors": streamer._calibration_errors,
                    "coverage_pct": streamer._calibration_coverage_pct,
                    "files": file_names,
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
                data = streamer.get_calibration_file(name)
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
                if not name.endswith(".png"):
                    self_handler.send_response(404)
                    self_handler.end_headers()
                    return
                data = streamer.get_calibration_file(name)
                if data is None:
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

            # -- Intrinsics calibration endpoints ----------------------------

            def _handle_intrinsics_start(self_handler) -> None:  # noqa: N805
                """Start projected checkerboard intrinsics calibration."""
                try:
                    length = int(self_handler.headers.get("Content-Length", 0))
                    raw = self_handler.rfile.read(length) if length > 0 else b"{}"
                    params = json.loads(raw)
                    pw = int(params.get("proj_w", 1920))
                    ph = int(params.get("proj_h", 1080))
                    sq = params.get("square_px")
                    if sq is not None:
                        sq = int(sq)
                    result = streamer.start_intrinsics_calibration(pw, ph, sq)
                    body = json.dumps(result).encode()
                    self_handler.send_response(200)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(body)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(body)
                except Exception as exc:
                    body = json.dumps({"ok": False, "error": str(exc)}).encode()
                    self_handler.send_response(500)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(body)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(body)

            def _handle_intrinsics_stop(self_handler) -> None:  # noqa: N805
                """Stop intrinsics calibration."""
                streamer.stop_intrinsics_calibration()
                body = json.dumps({"ok": True}).encode()
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.send_header("Content-Length", str(len(body)))
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.end_headers()
                self_handler.wfile.write(body)

            def _handle_intrinsics_status(self_handler) -> None:  # noqa: N805
                """Return intrinsics calibration progress and results."""
                data = streamer.get_intrinsics_status()
                body = json.dumps(data).encode()
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.send_header("Content-Length", str(len(body)))
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.send_header("Cache-Control", "no-cache")
                self_handler.end_headers()
                self_handler.wfile.write(body)

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

        # Submit initial test card so /projector has something to show
        card = _load_testcard(1920, 1080)
        self.submit_calibration_frame(card)

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

        Uses high-quality JPEG encoding (quality 97) since calibration
        patterns need maximum fidelity — JPEG artifacts at quality 70
        destroy the structured light signal.

        Encodes synchronously since calibration patterns are infrequent
        and need to arrive reliably.

        Thread-safe — may be called from any thread.
        """
        if not self._running:
            return
        jpeg = self._encode_calibration_jpeg(rgb)
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

    def _encode_calibration_jpeg(self, rgb: np.ndarray) -> bytes | None:
        """Encode a calibration pattern as near-lossless JPEG (quality 97).

        Calibration patterns need maximum fidelity — the phase/Gray code
        signal is destroyed by standard JPEG compression (quality 70).
        """
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, jpeg_buf = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._calibration_quality]
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


def get_streamer() -> FrameStreamer | None:
    """Return the existing shared FrameStreamer, or None if not yet created."""
    return _shared_streamer


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
