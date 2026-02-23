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


# ── Projected checkerboard pattern generation ─────────────────────────────


def generate_projected_checkerboard(
    cols: int,
    rows: int,
    square_px: int,
    offset_x: int,
    offset_y: int,
    proj_w: int,
    proj_h: int,
    white_level: int = 255,
) -> np.ndarray:
    """Generate a checkerboard image for projector display.

    Black background with a white border region around the board, then
    black squares drawn on top.  This contrast is essential for
    ``findChessboardCorners`` to detect the board boundary.

    The board has ``(cols+1) x (rows+1)`` squares.
    Inner corners are ``cols x rows``.

    Returns ``(proj_h, proj_w, 3)`` uint8 RGB.
    """
    img = np.zeros((proj_h, proj_w), dtype=np.uint8)  # black background

    n_sq_x = cols + 1
    n_sq_y = rows + 1

    # Draw white border (1 square-width padding) around the checkerboard
    border = square_px
    bx0 = max(0, offset_x - border)
    by0 = max(0, offset_y - border)
    bx1 = min(proj_w, offset_x + n_sq_x * square_px + border)
    by1 = min(proj_h, offset_y + n_sq_y * square_px + border)
    img[by0:by1, bx0:bx1] = white_level

    # Draw black squares on top of the white border
    for sy in range(n_sq_y):
        for sx in range(n_sq_x):
            if (sx + sy) % 2 == 1:
                x0 = offset_x + sx * square_px
                y0 = offset_y + sy * square_px
                x1 = x0 + square_px
                y1 = y0 + square_px
                x0c = max(0, min(x0, proj_w))
                y0c = max(0, min(y0, proj_h))
                x1c = max(0, min(x1, proj_w))
                y1c = max(0, min(y1, proj_h))
                if x1c > x0c and y1c > y0c:
                    img[y0c:y1c, x0c:x1c] = 0

    # Convert to 3-channel RGB
    return np.stack([img, img, img], axis=-1)


def generate_checkerboard_sequence(
    cols: int, rows: int, square_px: int, proj_w: int, proj_h: int,
) -> list[dict]:
    """Generate a sequence of checkerboard positions for a single square size.

    Uses 7 well-chosen positions (center, 4 corners, 2 midpoints) matching
    Zhang's method requirements: varied positions across the frame.

    Returns a list of dicts with ``offset_x``, ``offset_y``, and ``label``.

    Parameters
    ----------
    cols, rows : int
        Inner corner counts (e.g. 7x5).
    square_px : int
        Square size in projector pixels.
    proj_w, proj_h : int
        Projector resolution.
    """
    board_w = (cols + 1) * square_px
    board_h = (rows + 1) * square_px
    border = square_px  # white border around board
    margin = border + square_px // 2  # border + extra breathing room

    # Check if board fits at all
    if board_w + 2 * margin > proj_w or board_h + 2 * margin > proj_h:
        logger.warning(
            "Checkerboard too large for %dx%d (board %dx%d + margin %d)",
            proj_w, proj_h, board_w, board_h, margin,
        )
        return []

    max_x = proj_w - board_w - margin
    max_y = proj_h - board_h - margin
    min_x = margin
    min_y = margin

    if max_x < min_x:
        max_x = min_x
    if max_y < min_y:
        max_y = min_y

    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2

    positions = [
        ("center", cx, cy),
        ("top-left", min_x, min_y),
        ("top-right", max_x, min_y),
        ("bottom-left", min_x, max_y),
        ("bottom-right", max_x, max_y),
        ("top-center", cx, min_y),
        ("bottom-center", cx, max_y),
    ]

    patterns: list[dict] = []
    for label, ox, oy in positions:
        patterns.append({
            "offset_x": ox,
            "offset_y": oy,
            "label": label,
        })

    return patterns


def get_projected_corner_positions(
    cols: int,
    rows: int,
    square_px: int,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    """Get the projector-pixel coordinates of the inner corners.

    Returns ``(cols * rows, 1, 2)`` float32 array of projector pixel positions.
    """
    corners = np.zeros((rows * cols, 1, 2), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            # Inner corners start at (1,1) square offset
            px = offset_x + (c + 1) * square_px
            py = offset_y + (r + 1) * square_px
            corners[r * cols + c, 0] = (px, py)
    return corners


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

        # Standalone calibration (browser webcam → server CalibrationState)
        self._standalone_calib = None  # CalibrationState | None
        self._standalone_device: torch.device | None = None
        self._standalone_proj_w: int = 1920
        self._standalone_proj_h: int = 1080
        self._standalone_ambient: np.ndarray | None = None

        # Dashboard parameter overrides — set via POST /api/params,
        # read by pipelines as kwargs overrides.
        self._param_overrides: dict = {}

        # Checkerboard camera intrinsics calibration state
        self._checkerboard_captures: list[tuple[np.ndarray, np.ndarray]] = []
        self._checkerboard_img_size: tuple[int, int] | None = None
        self._checkerboard_cols: int = 9
        self._checkerboard_rows: int = 6

        # Auto camera calibration (projected checkerboards)
        self._cc_auto_active: bool = False
        self._cc_auto_patterns: list[dict] = []
        self._cc_auto_images: list[np.ndarray] = []
        self._cc_auto_idx: int = 0
        self._cc_auto_cols: int = 7
        self._cc_auto_rows: int = 5
        self._cc_auto_proj_w: int = 0
        self._cc_auto_proj_h: int = 0
        self._cc_auto_square_px: int = 0
        self._cc_auto_proj_pts: list[np.ndarray] = []
        # Camera calibration results (for projector intrinsics computation)
        self._cam_K: np.ndarray | None = None
        self._cam_dist: np.ndarray | None = None
        self._cam_rvecs: list | None = None
        self._cam_tvecs: list | None = None
        # Settle state (mirrors Gray code settle)
        self._cc_settle_waiting: bool = False
        self._cc_settle_baseline: np.ndarray | None = None
        self._cc_settle_prev: np.ndarray | None = None
        self._cc_settle_counter: int = 0
        self._cc_settle_stable: int = 0
        self._cc_settle_changed: bool = False

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

    # -- Standalone calibration (browser webcam) --------------------------------

    def start_standalone_calibration(
        self,
        proj_w: int = 1920,
        proj_h: int = 1080,
        settle_frames: int = 10,
        capture_frames: int = 3,
        max_brightness: int = 128,
        method: str = "gray_code",
        refine: bool = False,
    ) -> dict:
        """Start a standalone calibration session using browser webcam frames.

        Parameters
        ----------
        refine : bool
            If True, merge new scan results with existing calibration
            instead of replacing it.

        Returns dict with 'ok' and 'total_patterns'.
        """
        if self._calibration_active:
            return {"ok": False, "error": "Calibration already in progress"}

        from .calibration import CalibrationState

        self._standalone_proj_w = proj_w
        self._standalone_proj_h = proj_h
        self._standalone_refine = refine
        self._standalone_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._standalone_ambient = None

        # Browser mode: each frame is an HTTP round-trip (~150-500ms).
        # Change-detection settle captures as soon as the camera shows the
        # new pattern.  settle_frames is just a timeout safety net — set it
        # high so we don't capture stale frames if the MJPEG round-trip
        # through RunPod proxy is slow.
        self._standalone_calib = CalibrationState(
            proj_w, proj_h,
            method=method,
            settle_frames=max(60, settle_frames),  # timeout only
            capture_frames=max(2, capture_frames),  # at least 2 for averaging
            max_brightness=max_brightness,
            change_threshold=5.0,
            stability_threshold=3.0,
        )
        self._standalone_calib.start()

        self._calibration_active = True
        self.clear_calibration_results()

        # Send test pattern to projector stream while waiting to start
        card = _load_testcard(proj_w, proj_h)
        self.submit_calibration_frame(card)

        total = self._standalone_calib.total_patterns
        logger.info(
            "Standalone calibration started: %s, %dx%d, %d patterns",
            method, proj_w, proj_h, total,
        )
        return {"ok": True, "total_patterns": total, "method": method}

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

        # Track state before stepping to detect pattern changes
        prev_idx = calib._pattern_index
        prev_phase = calib.phase
        was_settling = calib._waiting_for_settle

        # Step the calibration state machine
        pattern = calib.step(tensor, device)

        # Build progress info
        phase = calib.phase.name
        progress = calib.progress
        settling = calib._waiting_for_settle
        pattern_info = calib.get_pattern_info()

        if settling:
            pattern_info += f" settling {calib._settle_counter}/{calib.settle_frames}"

        # Update dashboard progress
        self.update_calibration_progress(progress, phase, pattern_info)

        # Only submit pattern to projector when it actually changed
        # (new phase or new pattern index).  During settling the projector
        # is already showing the correct pattern — no need to regenerate
        # and re-encode 1920x1080 JPEG every frame.
        pattern_changed = (
            calib.phase != prev_phase
            or calib._pattern_index != prev_idx
        )
        if pattern is not None and pattern_changed:
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
        """Finalize standalone calibration: save mapping and publish results.

        When ``_standalone_refine`` is True, merges new scan with existing
        calibration — averaging where both have valid data, filling gaps
        where only one has coverage.
        """
        from .calibration import load_calibration, save_calibration

        calib = self._standalone_calib
        if calib is None:
            return {"error": "No calibration state", "done": True}

        mapping = calib.get_mapping()
        coverage_pct = 0.0
        refine = getattr(self, "_standalone_refine", False)

        if mapping is not None:
            map_x, map_y = mapping
            new_valid = calib.proj_valid_mask

            # ---- Refine: merge with existing calibration ----
            if refine and _CALIBRATION_JSON_PATH.is_file():
                try:
                    old_x, old_y, old_pw, old_ph, _ = load_calibration(
                        _CALIBRATION_JSON_PATH,
                    )
                    if (old_x.shape == map_x.shape and old_y.shape == map_y.shape):
                        old_valid = (old_x >= 0) & (old_y >= 0)
                        if new_valid is None:
                            new_valid = (map_x >= 0) & (map_y >= 0)

                        # Average where both have valid data
                        both = old_valid & new_valid
                        map_x[both] = (old_x[both] + map_x[both]) / 2.0
                        map_y[both] = (old_y[both] + map_y[both]) / 2.0

                        # Fill gaps: use old data where only old is valid
                        only_old = old_valid & ~new_valid
                        map_x[only_old] = old_x[only_old]
                        map_y[only_old] = old_y[only_old]

                        # Update valid mask to union
                        new_valid = old_valid | new_valid

                        logger.info(
                            "Refine: merged new scan with existing calibration "
                            "(both=%d, old-only=%d, new-only=%d)",
                            np.count_nonzero(both),
                            np.count_nonzero(only_old),
                            np.count_nonzero(new_valid & ~old_valid),
                        )
                    else:
                        logger.warning(
                            "Refine: shape mismatch old=%s new=%s, using new only",
                            old_x.shape, map_x.shape,
                        )
                except Exception:
                    logger.warning("Refine: could not load existing calibration", exc_info=True)

            # Compute coverage
            if new_valid is not None:
                total = new_valid.size
                valid = np.count_nonzero(new_valid)
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
                valid_mask=new_valid,
                cam_w=calib.cam_w,
                cam_h=calib.cam_h,
            )

            # Save ambient camera for AI depth estimation
            from pathlib import Path as _Path
            results_dir = _Path.home() / ".projectionmapanything_results"
            results_dir.mkdir(exist_ok=True)
            ambient = self._standalone_ambient
            if ambient is None:
                ambient = np.full(
                    (480, 640, 3), 128, dtype=np.uint8,
                )
            try:
                ambient_bgr = cv2.cvtColor(ambient, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(results_dir / "ambient_camera.png"), ambient_bgr)
                logger.info("Saved ambient camera frame for AI depth")
            except Exception:
                logger.warning("Could not save ambient_camera.png", exc_info=True)

            # Publish results (warped camera, coverage map, etc.)
            from .pipeline import publish_calibration_results

            publish_calibration_results(
                map_x=map_x,
                map_y=map_y,
                rgb_frame_np=ambient,
                proj_w=self._standalone_proj_w,
                proj_h=self._standalone_proj_h,
                proj_valid_mask=new_valid,
                coverage_pct=coverage_pct,
                streamer=self,
            )

            self.update_calibration_progress(
                1.0, "DONE", coverage_pct=coverage_pct,
            )
            logger.info(
                "Standalone calibration %s (coverage=%.1f%%)",
                "refined" if refine else "complete",
                coverage_pct,
            )
        else:
            logger.warning("Standalone calibration: mapping was None")
            self.update_calibration_progress(
                1.0, "DONE", errors=["Mapping was None — decode failed"],
            )

        # Cleanup
        self._standalone_calib = None
        self._standalone_ambient = None
        self._standalone_refine = False
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

    # -- Checkerboard camera intrinsics calibration ----------------------------

    def capture_checkerboard(
        self, jpeg_bytes: bytes, cols: int = 9, rows: int = 6,
    ) -> dict:
        """Detect checkerboard corners in a JPEG frame and store if found.

        Parameters
        ----------
        jpeg_bytes : bytes
            JPEG-encoded webcam frame.
        cols, rows : int
            Inner corner counts of the checkerboard pattern.

        Returns
        -------
        dict
            ``{"ok": True, "found": True, "captures": N}`` on success,
            ``{"ok": True, "found": False}`` if no board detected.
        """
        # Decode JPEG → grayscale
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"ok": False, "error": "Could not decode image"}
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        self._checkerboard_cols = cols
        self._checkerboard_rows = rows

        # Try multiple flag sets for robustness (from standalone app)
        board_size = (cols, rows)
        flag_sets = [
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
        ]
        found = False
        corners = None
        for flags in flag_sets:
            found, corners = cv2.findChessboardCorners(gray, board_size, flags)
            if found and corners is not None:
                break

        if not found or corners is None:
            return {"ok": True, "found": False, "captures": len(self._checkerboard_captures)}

        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Build object points (3D coordinates on the board plane, z=0)
        obj_pts = np.zeros((cols * rows, 3), dtype=np.float32)
        obj_pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

        self._checkerboard_captures.append((obj_pts, corners.reshape(-1, 2)))
        self._checkerboard_img_size = (w, h)

        n = len(self._checkerboard_captures)
        logger.info(
            "Checkerboard capture %d: %dx%d board detected in %dx%d image",
            n, cols, rows, w, h,
        )
        return {"ok": True, "found": True, "captures": n}

    def compute_camera_intrinsics(self) -> dict:
        """Run cv2.calibrateCamera on captured checkerboard frames.

        Requires at least 5 captures. Saves intrinsics to disk.

        Returns
        -------
        dict
            ``{"ok": True, "focal_length": fx, "reprojection_error": err}``
            or ``{"ok": False, "error": "..."}``
        """
        n = len(self._checkerboard_captures)
        if n < 5:
            return {"ok": False, "error": f"Need at least 5 captures (have {n})"}

        if self._checkerboard_img_size is None:
            return {"ok": False, "error": "No image size recorded"}

        obj_pts_list = [c[0] for c in self._checkerboard_captures]
        img_pts_list = [c[1].reshape(-1, 1, 2).astype(np.float32)
                        for c in self._checkerboard_captures]
        w, h = self._checkerboard_img_size

        try:
            ret, K_cam, dist_cam, rvecs, tvecs = cv2.calibrateCamera(
                obj_pts_list, img_pts_list, (w, h), None, None,
            )
        except Exception as exc:
            logger.warning("calibrateCamera failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"calibrateCamera failed: {exc}"}

        # Store for projector intrinsics computation
        self._cam_K = K_cam
        self._cam_dist = dist_cam
        self._cam_rvecs = rvecs
        self._cam_tvecs = tvecs

        from .calibration import save_camera_intrinsics

        save_camera_intrinsics(K_cam, dist_cam, (w, h), ret)

        fx = float(K_cam[0, 0])
        fy = float(K_cam[1, 1])
        logger.info(
            "Camera intrinsics computed: fx=%.1f fy=%.1f, error=%.4f px, "
            "%d captures",
            fx, fy, ret, n,
        )
        return {
            "ok": True,
            "focal_length": fx,
            "focal_length_y": fy,
            "reprojection_error": float(ret),
            "image_width": w,
            "image_height": h,
            "captures": n,
        }

    def get_camera_calibration_status(self) -> dict:
        """Return current checkerboard calibration status."""
        from .calibration import load_camera_intrinsics

        intrinsics = load_camera_intrinsics()
        status: dict = {
            "captures": len(self._checkerboard_captures),
            "cols": self._checkerboard_cols,
            "rows": self._checkerboard_rows,
            "has_intrinsics": intrinsics is not None,
            "intrinsics": {
                "fx": float(intrinsics["K_cam"][0, 0]),
                "fy": float(intrinsics["K_cam"][1, 1]),
                "error": intrinsics["reprojection_error"],
                "image_size": list(intrinsics["image_size"]),
                "timestamp": intrinsics["timestamp"],
            } if intrinsics is not None else None,
            "auto_active": self._cc_auto_active,
            "auto_progress": (
                self._cc_auto_idx / len(self._cc_auto_patterns)
                if self._cc_auto_patterns else 0.0
            ),
            "auto_total": len(self._cc_auto_patterns),
        }
        return status

    def reset_camera_calibration(self) -> dict:
        """Clear checkerboard captures."""
        self._checkerboard_captures = []
        self._checkerboard_img_size = None
        logger.info("Checkerboard captures cleared")
        return {"ok": True}

    # -- Auto camera calibration (projected checkerboards) ----------------------

    def start_auto_camera_calibration(
        self,
        proj_w: int,
        proj_h: int,
        cols: int = 7,
        rows: int = 5,
        square_px: int = 0,
    ) -> dict:
        """Start automated camera intrinsics calibration using projected checkerboards.

        Projects checkerboard patterns at 7 well-chosen positions,
        detects corners in webcam frames, then runs cv2.calibrateCamera
        followed by projector intrinsics recovery.

        Parameters
        ----------
        proj_w, proj_h : int
            Projector resolution (must match the projector page window).
        cols, rows : int
            Inner corner counts for the checkerboard pattern.
        square_px : int
            Square size in projector pixels.  0 = auto-compute from projector res.
        """
        if self._cc_auto_active:
            return {"ok": False, "error": "Auto calibration already in progress"}
        if self._calibration_active:
            return {"ok": False, "error": "Gray code calibration in progress"}

        # Auto-compute square size if not specified
        if square_px <= 0:
            square_px = max(8, min(proj_w, proj_h) // 12)

        # Generate pattern sequence (single size, 7 positions)
        patterns = generate_checkerboard_sequence(
            cols, rows, square_px, proj_w, proj_h,
        )
        if not patterns:
            return {
                "ok": False,
                "error": (
                    f"No patterns fit in {proj_w}x{proj_h} for {cols}x{rows}"
                    f" board with square_px={square_px}"
                ),
            }

        # Pre-generate all pattern images
        images = []
        for p in patterns:
            img = generate_projected_checkerboard(
                cols, rows, square_px,
                p["offset_x"], p["offset_y"],
                proj_w, proj_h,
            )
            images.append(img)

        # Clear existing captures
        self._checkerboard_captures = []
        self._checkerboard_img_size = None
        self._checkerboard_cols = cols
        self._checkerboard_rows = rows

        # Set state
        self._cc_auto_active = True
        self._cc_auto_patterns = patterns
        self._cc_auto_images = images
        self._cc_auto_idx = 0
        self._cc_auto_cols = cols
        self._cc_auto_rows = rows
        self._cc_auto_proj_w = proj_w
        self._cc_auto_proj_h = proj_h
        self._cc_auto_square_px = square_px
        self._cc_auto_proj_pts: list[np.ndarray] = []  # projector corner positions
        self._calibration_active = True  # suppress normal frames

        # Project first pattern and start settling
        self.submit_calibration_frame(images[0])
        self._cc_begin_settle()

        total = len(patterns)
        logger.info(
            "Auto camera calibration started: %dx%d board, %dx%d proj, %d patterns",
            cols, rows, proj_w, proj_h, total,
        )
        return {"ok": True, "total_patterns": total}

    def step_auto_camera_calibration(self, jpeg_bytes: bytes) -> dict:
        """Process one webcam frame for auto camera calibration.

        Handles settle detection, corner finding, and pattern advancement.

        Parameters
        ----------
        jpeg_bytes : bytes
            JPEG-encoded webcam frame from browser getUserMedia.

        Returns
        -------
        dict with phase, pattern_idx, total_patterns, pattern_label,
        captures_so_far, last_found, progress, settling, done.
        """
        if not self._cc_auto_active:
            return {"error": "No auto calibration in progress", "done": True}

        idx = self._cc_auto_idx
        total = len(self._cc_auto_patterns)

        # Check if we've exhausted all patterns
        if idx >= total:
            return self._finish_auto_camera_calibration()

        # Decode JPEG
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "Could not decode JPEG frame", "done": False}
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        pattern = self._cc_auto_patterns[idx]
        last_found = False

        # Phase 1: Settling — wait for image to stabilize after pattern change
        if self._cc_settle_waiting:
            settled = self._cc_check_settle(gray)
            if not settled:
                return {
                    "phase": "settling",
                    "pattern_idx": idx,
                    "total_patterns": total,
                    "pattern_label": pattern["label"],
                    "captures_so_far": len(self._checkerboard_captures),
                    "last_found": False,
                    "progress": idx / total,
                    "settling": True,
                    "done": False,
                }
            # Settled — fall through to capture

        # Phase 2: Capture — detect corners
        h, w = gray.shape[:2]
        board_size = (self._cc_auto_cols, self._cc_auto_rows)

        flag_sets = [
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
        ]
        found = False
        corners = None
        for flags in flag_sets:
            found, corners = cv2.findChessboardCorners(gray, board_size, flags)
            if found and corners is not None:
                break

        if found and corners is not None:
            # Sub-pixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Object points with real square_px spacing (not unit squares)
            sq = self._cc_auto_square_px
            nc, nr = self._cc_auto_cols, self._cc_auto_rows
            obj_pts = np.zeros((nc * nr, 3), dtype=np.float32)
            for r in range(nr):
                for c in range(nc):
                    obj_pts[r * nc + c] = (c * sq, r * sq, 0.0)

            self._checkerboard_captures.append((obj_pts, corners.reshape(-1, 2)))
            self._checkerboard_img_size = (w, h)

            # Store projector-pixel corner positions for projector intrinsics
            proj_corners = get_projected_corner_positions(
                nc, nr, sq,
                pattern["offset_x"], pattern["offset_y"],
            )
            self._cc_auto_proj_pts.append(proj_corners)

            last_found = True
            logger.info(
                "Auto CC pattern %d/%d (%s): %d corners found, %d captures total",
                idx + 1, total, pattern["label"],
                len(corners), len(self._checkerboard_captures),
            )
        else:
            logger.info(
                "Auto CC pattern %d/%d (%s): no corners found, skipping",
                idx + 1, total, pattern["label"],
            )

        # Advance to next pattern
        self._cc_auto_idx = idx + 1
        if self._cc_auto_idx < total:
            # Project next pattern and begin settle
            self.submit_calibration_frame(self._cc_auto_images[self._cc_auto_idx])
            self._cc_begin_settle()
        else:
            # All patterns done
            return self._finish_auto_camera_calibration()

        return {
            "phase": "capturing",
            "pattern_idx": self._cc_auto_idx,
            "total_patterns": total,
            "pattern_label": self._cc_auto_patterns[min(self._cc_auto_idx, total - 1)]["label"],
            "captures_so_far": len(self._checkerboard_captures),
            "last_found": last_found,
            "progress": self._cc_auto_idx / total,
            "settling": False,
            "done": False,
        }

    def _finish_auto_camera_calibration(self) -> dict:
        """Finalize auto camera calibration: compute camera + projector intrinsics."""
        n = len(self._checkerboard_captures)
        total = len(self._cc_auto_patterns)

        result: dict
        if n < 3:
            result = {
                "phase": "done",
                "done": True,
                "ok": False,
                "error": f"Only {n} boards detected (need 3+). Try better lighting or larger patterns.",
                "captures_so_far": n,
                "total_patterns": total,
                "progress": 1.0,
            }
            logger.warning(
                "Auto CC: only %d captures (need 3+), cannot compute intrinsics", n,
            )
        else:
            # Phase 1: Camera intrinsics
            cal_result = self.compute_camera_intrinsics()
            result = {
                "phase": "done",
                "done": True,
                "captures_so_far": n,
                "total_patterns": total,
                "progress": 1.0,
                **cal_result,
            }
            if cal_result.get("ok"):
                logger.info(
                    "Auto CC camera: %d captures, fx=%.1f, error=%.4f px",
                    n, cal_result.get("focal_length", 0),
                    cal_result.get("reprojection_error", 0),
                )
                # Phase 2: Projector intrinsics
                proj_result = self._compute_projector_intrinsics()
                if proj_result.get("ok"):
                    result["proj_focal_length"] = proj_result["focal_length"]
                    result["proj_reprojection_error"] = proj_result[
                        "reprojection_error"
                    ]
                    logger.info(
                        "Auto CC projector: fx=%.1f, error=%.4f px",
                        proj_result["focal_length"],
                        proj_result["reprojection_error"],
                    )
                else:
                    result["proj_error"] = proj_result.get("error", "unknown")
                    logger.warning(
                        "Auto CC projector intrinsics failed: %s",
                        proj_result.get("error"),
                    )

        # Cleanup
        self._cc_auto_active = False
        self._calibration_active = False
        self._cc_auto_patterns = []
        self._cc_auto_images = []
        self._cc_settle_baseline = None
        self._cc_settle_prev = None

        # Show test card on projector
        pw = self._cc_auto_proj_w or 1920
        ph = self._cc_auto_proj_h or 1080
        card = _load_testcard(pw, ph)
        self.submit_calibration_frame(card)

        return result

    def _compute_projector_intrinsics(self) -> dict:
        """Compute projector intrinsics using camera rvecs/tvecs.

        Uses the camera's extrinsic parameters to transform object points
        to 3D camera coordinates, then calibrates the projector using those
        3D points and the known projector pixel positions.

        Must be called after ``compute_camera_intrinsics()`` has stored
        ``_cam_K``, ``_cam_rvecs``, ``_cam_tvecs``, and after
        ``_cc_auto_proj_pts`` has been populated during capture.
        """
        if not hasattr(self, "_cam_rvecs") or self._cam_rvecs is None:
            return {"ok": False, "error": "No camera rvecs available"}

        proj_pts = getattr(self, "_cc_auto_proj_pts", [])
        if len(proj_pts) < 3:
            return {"ok": False, "error": f"Need 3+ proj pts, have {len(proj_pts)}"}

        n = len(proj_pts)
        if n != len(self._cam_rvecs):
            return {
                "ok": False,
                "error": f"Mismatch: {n} proj pts vs {len(self._cam_rvecs)} rvecs",
            }

        # Build object point template (same as used during capture)
        sq = self._cc_auto_square_px
        nc, nr = self._cc_auto_cols, self._cc_auto_rows
        obj_template = np.zeros((nc * nr, 3), dtype=np.float32)
        for r in range(nr):
            for c in range(nc):
                obj_template[r * nc + c] = (c * sq, r * sq, 0.0)

        # Transform object points to camera coordinates for each view
        obj_3d_list = []
        for i in range(n):
            R_cam, _ = cv2.Rodrigues(self._cam_rvecs[i])
            t_cam = self._cam_tvecs[i].reshape(3)
            pts_3d = (R_cam @ obj_template.T).T + t_cam
            obj_3d_list.append(pts_3d.astype(np.float32))

        proj_w = self._cc_auto_proj_w or 1920
        proj_h = self._cc_auto_proj_h or 1080
        proj_size = (proj_w, proj_h)

        # Initial projector intrinsic guess
        fx_init = float(max(proj_w, proj_h))
        K_init = np.array([
            [fx_init, 0, proj_w / 2.0],
            [0, fx_init, proj_h / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)

        try:
            ret, K_proj, dist_proj, _, _ = cv2.calibrateCamera(
                obj_3d_list,
                proj_pts,
                proj_size,
                K_init,
                None,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS,
            )
        except Exception as exc:
            logger.warning("Projector calibrateCamera failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"Projector calibration failed: {exc}"}

        fx = float(K_proj[0, 0])
        fy = float(K_proj[1, 1])
        logger.info(
            "Projector intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f error=%.3f px",
            fx, fy, float(K_proj[0, 2]), float(K_proj[1, 2]), ret,
        )

        # Save projector intrinsics alongside camera intrinsics
        self._save_projector_intrinsics(K_proj, dist_proj, ret)

        return {
            "ok": True,
            "focal_length": fx,
            "focal_length_y": fy,
            "reprojection_error": float(ret),
        }

    def _save_projector_intrinsics(
        self,
        K_proj: np.ndarray,
        dist_proj: np.ndarray,
        reprojection_error: float,
    ) -> None:
        """Save projector intrinsics to the calibration JSON (merge with existing)."""
        try:
            if _CALIBRATION_JSON_PATH.is_file():
                data = json.loads(_CALIBRATION_JSON_PATH.read_text())
            else:
                data = {}
            data["K_proj"] = K_proj.tolist()
            data["dist_proj"] = (
                dist_proj.flatten().tolist() if dist_proj is not None
                else [0.0] * 5
            )
            data["proj_reprojection_error"] = float(reprojection_error)
            _CALIBRATION_JSON_PATH.write_text(json.dumps(data))
            logger.info("Saved projector intrinsics to %s", _CALIBRATION_JSON_PATH.name)
        except Exception:
            logger.warning(
                "Could not save projector intrinsics", exc_info=True,
            )

    def stop_auto_camera_calibration(self) -> dict:
        """Cancel an in-progress auto camera calibration."""
        was_active = self._cc_auto_active
        self._cc_auto_active = False
        self._calibration_active = False
        self._cc_auto_patterns = []
        self._cc_auto_images = []
        self._cc_settle_baseline = None
        self._cc_settle_prev = None
        logger.info("Auto camera calibration stopped (was_active=%s)", was_active)

        # Show test card
        pw = self._cc_auto_proj_w or 1920
        ph = self._cc_auto_proj_h or 1080
        card = _load_testcard(pw, ph)
        self.submit_calibration_frame(card)

        return {"ok": True}

    def _cc_begin_settle(self) -> None:
        """Reset settle state for a new pattern."""
        self._cc_settle_waiting = True
        self._cc_settle_baseline = None
        self._cc_settle_prev = None
        self._cc_settle_counter = 0
        self._cc_settle_stable = 0
        self._cc_settle_changed = False

    def _cc_check_settle(self, gray: np.ndarray) -> bool:
        """Check if the camera image has settled after a pattern change.

        Uses the same change-detection approach as Gray code calibration:
        wait for a significant change from baseline, then wait for stability.

        Returns True when settled and ready to capture.
        """
        self._cc_settle_counter += 1
        change_threshold = 5.0
        stability_threshold = 3.0
        timeout = 60

        # Downsample for fast comparison
        small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)

        if self._cc_settle_baseline is None:
            self._cc_settle_baseline = small.astype(np.float32)
            self._cc_settle_prev = small.astype(np.float32)
            return False

        curr = small.astype(np.float32)

        if not self._cc_settle_changed:
            # Wait for change from baseline
            diff_from_baseline = np.mean(np.abs(curr - self._cc_settle_baseline))
            if diff_from_baseline > change_threshold:
                self._cc_settle_changed = True
                self._cc_settle_stable = 0
            elif self._cc_settle_counter > timeout:
                # Timeout — proceed anyway
                return True
            self._cc_settle_prev = curr
            return False

        # Changed detected — now wait for stability
        diff_from_prev = np.mean(np.abs(curr - self._cc_settle_prev))
        self._cc_settle_prev = curr

        if diff_from_prev < stability_threshold:
            self._cc_settle_stable += 1
        else:
            self._cc_settle_stable = 0

        if self._cc_settle_stable >= 3:
            return True

        if self._cc_settle_counter > timeout:
            return True

        return False

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
                elif path == "/camera-calibration/status":
                    self_handler._handle_camera_cal_status()
                elif path == "/calibration/export":
                    self_handler._handle_calibration_export()
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
                elif path == "/calibrate/start":
                    self_handler._handle_calibrate_start()
                elif path == "/calibrate/frame":
                    self_handler._handle_calibrate_frame()
                elif path == "/calibrate/stop":
                    self_handler._handle_calibrate_stop()
                elif path == "/camera-calibration/capture":
                    self_handler._handle_camera_cal_capture()
                elif path == "/camera-calibration/compute":
                    self_handler._handle_camera_cal_compute()
                elif path == "/camera-calibration/reset":
                    self_handler._handle_camera_cal_reset()
                elif path == "/camera-calibration/auto/start":
                    self_handler._handle_cc_auto_start()
                elif path == "/camera-calibration/auto/frame":
                    self_handler._handle_cc_auto_frame()
                elif path == "/camera-calibration/auto/stop":
                    self_handler._handle_cc_auto_stop()
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

            def _handle_camera_cal_capture(self_handler) -> None:  # noqa: N805
                """Capture a checkerboard frame for camera intrinsics."""
                try:
                    length = int(self_handler.headers.get("Content-Length", 0))
                    if length <= 0 or length > 10 * 1024 * 1024:
                        self_handler.send_response(400)
                        self_handler.end_headers()
                        return
                    jpeg_bytes = self_handler.rfile.read(length)
                    # Parse cols/rows from query string
                    qs = self_handler.path.split("?", 1)[1] if "?" in self_handler.path else ""
                    params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
                    cols = int(params.get("cols", 9))
                    rows = int(params.get("rows", 6))
                    result = streamer.capture_checkerboard(jpeg_bytes, cols, rows)
                    resp = json.dumps(result).encode()
                    self_handler.send_response(200)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(resp)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(resp)
                except Exception:
                    logger.warning("camera-calibration/capture failed", exc_info=True)
                    self_handler.send_response(500)
                    self_handler.end_headers()

            def _handle_camera_cal_compute(self_handler) -> None:  # noqa: N805
                """Compute camera intrinsics from captured checkerboards."""
                try:
                    # Read body if any (may be empty)
                    length = int(self_handler.headers.get("Content-Length", 0))
                    if length > 0:
                        self_handler.rfile.read(length)
                    result = streamer.compute_camera_intrinsics()
                    resp = json.dumps(result).encode()
                    status = 200 if result.get("ok") else 400
                    self_handler.send_response(status)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(resp)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(resp)
                except Exception:
                    logger.warning("camera-calibration/compute failed", exc_info=True)
                    self_handler.send_response(500)
                    self_handler.end_headers()

            def _handle_camera_cal_status(self_handler) -> None:  # noqa: N805
                """Return camera intrinsics calibration status."""
                try:
                    result = streamer.get_camera_calibration_status()
                    resp = json.dumps(result).encode()
                    self_handler.send_response(200)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(resp)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(resp)
                except Exception:
                    logger.warning("camera-calibration/status failed", exc_info=True)
                    self_handler.send_response(500)
                    self_handler.end_headers()

            def _handle_camera_cal_reset(self_handler) -> None:  # noqa: N805
                """Reset checkerboard captures."""
                try:
                    length = int(self_handler.headers.get("Content-Length", 0))
                    if length > 0:
                        self_handler.rfile.read(length)
                    result = streamer.reset_camera_calibration()
                    resp = json.dumps(result).encode()
                    self_handler.send_response(200)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(resp)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(resp)
                except Exception:
                    logger.warning("camera-calibration/reset failed", exc_info=True)
                    self_handler.send_response(500)
                    self_handler.end_headers()

            def _handle_cc_auto_start(self_handler) -> None:  # noqa: N805
                """Start auto camera calibration with projected checkerboards."""
                try:
                    length = int(self_handler.headers.get("Content-Length", 0))
                    body = self_handler.rfile.read(length) if length > 0 else b"{}"
                    cfg = json.loads(body) if body else {}
                    result = streamer.start_auto_camera_calibration(
                        proj_w=int(cfg.get("proj_w", 1920)),
                        proj_h=int(cfg.get("proj_h", 1080)),
                        cols=int(cfg.get("cols", 7)),
                        rows=int(cfg.get("rows", 5)),
                        square_px=int(cfg.get("square_px", 0)),
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
                    logger.warning("camera-calibration/auto/start failed", exc_info=True)
                    self_handler.send_response(500)
                    self_handler.end_headers()

            def _handle_cc_auto_frame(self_handler) -> None:  # noqa: N805
                """Process one webcam frame for auto camera calibration."""
                try:
                    length = int(self_handler.headers.get("Content-Length", 0))
                    if length <= 0 or length > 10 * 1024 * 1024:
                        self_handler.send_response(400)
                        self_handler.end_headers()
                        return
                    jpeg_bytes = self_handler.rfile.read(length)
                    result = streamer.step_auto_camera_calibration(jpeg_bytes)
                    resp = json.dumps(result).encode()
                    self_handler.send_response(200)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(resp)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(resp)
                except Exception:
                    logger.warning("camera-calibration/auto/frame failed", exc_info=True)
                    self_handler.send_response(500)
                    self_handler.end_headers()

            def _handle_cc_auto_stop(self_handler) -> None:  # noqa: N805
                """Stop auto camera calibration."""
                try:
                    length = int(self_handler.headers.get("Content-Length", 0))
                    if length > 0:
                        self_handler.rfile.read(length)
                    result = streamer.stop_auto_camera_calibration()
                    resp = json.dumps(result).encode()
                    self_handler.send_response(200)
                    self_handler.send_header("Content-Type", "application/json")
                    self_handler.send_header("Content-Length", str(len(resp)))
                    self_handler.send_header("Access-Control-Allow-Origin", "*")
                    self_handler.end_headers()
                    self_handler.wfile.write(resp)
                except Exception:
                    logger.warning("camera-calibration/auto/stop failed", exc_info=True)
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
                        settle_frames=int(cfg.get("settle_frames", 30)),
                        capture_frames=int(cfg.get("capture_frames", 2)),
                        max_brightness=int(cfg.get("max_brightness", 128)),
                        method=str(cfg.get("method", "gray_code")),
                        refine=bool(cfg.get("refine", False)),
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
