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
``/``           Control panel — paste RunPod URL, iframe Scope UI, projector status.
``/projector``  Fullscreen MJPEG viewer — drag to projector monitor, click for fullscreen.
``/stream``     MJPEG multipart stream (``multipart/x-mixed-replace``).
``/frame``      Single JPEG snapshot of the latest frame.
``POST /config``  Companion app reports its projector resolution.
``GET  /config``  Returns the current projector config (JSON).

Works through RunPod's port proxy — expose the port and connect from anywhere.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BOUNDARY = b"promapframe"
_PROJECTOR_CONFIG_PATH = Path.home() / ".promapanything_projector.json"


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ── HTML templates ───────────────────────────────────────────────────────────

_PROJECTOR_HTML = """\
<!DOCTYPE html>
<html><head>
<title>ProMapAnything Projector</title>
<style>
  * { margin:0; padding:0; }
  body { background:#000; overflow:hidden;
         width:100vw; height:100vh;
         display:flex; justify-content:center;
         align-items:center; }
  body.fs { cursor:none; }
  img { width:100vw; height:100vh;
        object-fit:contain; display:block; }
  #hint { position:fixed; bottom:30px;
          left:50%; transform:translateX(-50%);
          color:#444; font:14px sans-serif;
          pointer-events:none;
          transition:opacity 0.5s; }
  body.fs #hint { opacity:0; }
</style>
</head><body>
<img id="stream" src="/stream" />
<div id="hint">Click to go fullscreen &mdash; drag this window to your projector first</div>
<script>
document.body.addEventListener('click', () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
  }
});
document.addEventListener('fullscreenchange', () => {
  document.body.classList.toggle('fs', !!document.fullscreenElement);
});
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

_CONTROL_PANEL_HTML = """\
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>ProMapAnything</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#1a1a2e; color:#e0e0e0;
         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         height:100vh; display:flex; flex-direction:column;
         align-items:center; justify-content:center; gap:24px; }
  h1 { font-size:22px; color:#e94560; }
  .sub { font-size:13px; color:#888; margin-top:4px; }
  .env-badge { display:inline-block; padding:2px 8px; border-radius:4px;
               font-size:11px; font-weight:600; margin-left:8px; }
  .env-local { background:#1a4a1a; color:#4ecca3; }
  .env-remote { background:#4a1a1a; color:#e94560; }

  .big-btn { display:flex; align-items:center; justify-content:center; gap:10px;
             padding:16px 32px; border:none; border-radius:10px;
             font-size:18px; font-weight:700; cursor:pointer;
             background:#e94560; color:#fff; transition:background 0.2s;
             box-shadow:0 4px 20px rgba(233,69,96,0.3); }
  .big-btn:hover { background:#c73652; }

  .preview { border-radius:8px; overflow:hidden; background:#000;
             width:480px; max-width:90vw; aspect-ratio:16/9; }
  .preview img { width:100%; height:100%; object-fit:contain; }

  .status { display:flex; align-items:center; gap:8px; font-size:13px; }
  .dot { width:10px; height:10px; border-radius:50%; }
  .dot.green { background:#4ecca3; box-shadow:0 0 6px #4ecca3; }
  .dot.yellow { background:#f0c040; box-shadow:0 0 6px #f0c040; }
  .dot.red { background:#e94560; }
  .detail { font-size:11px; color:#666; }

  .links { font-size:12px; color:#555; }
  .links a { color:#4ecca3; }

  .scope-section { display:flex; align-items:center; gap:8px; }
  .scope-section input { padding:6px 10px; border:1px solid #333; border-radius:4px;
                         background:#16213e; color:#fff; font-size:12px; width:300px; outline:none; }
  .scope-section input:focus { border-color:#e94560; }
  .scope-btn { padding:6px 12px; border:none; border-radius:4px;
               font-size:12px; font-weight:600; cursor:pointer;
               background:#0f3460; color:#ddd; transition:background 0.2s; }
  .scope-btn:hover { background:#1a4a80; }
</style>
</head><body>

<div>
  <h1>ProMapAnything <span class="env-badge" id="env-badge"></span></h1>
  <div class="sub">Drag this window to your main monitor. Pop out the projector below.</div>
</div>

<button class="big-btn" onclick="openProjector()">
  Open Projector Window
</button>

<div class="preview">
  <img id="preview" src="/frame" />
</div>

<div class="status">
  <div class="dot" id="proj-dot"></div>
  <span id="proj-status">Checking...</span>
</div>
<div class="detail" id="proj-resolution"></div>

<div class="scope-section">
  <input type="text" id="scope-url" placeholder="Scope URL (auto-detected or paste RunPod URL)" spellcheck="false" />
  <button class="scope-btn" onclick="openScope()">Open Scope</button>
</div>

<div class="links">
  <a href="/projector" target="_blank">/projector</a> &middot;
  <a href="/stream" target="_blank">/stream</a> &middot;
  <a href="/frame" target="_blank">/frame</a> &middot;
  <a href="/config" target="_blank">/config</a>
</div>

<script>
// Auto-detect environment
const host = window.location.hostname;
const isRunPod = host.includes('.proxy.runpod.net');
const badge = document.getElementById('env-badge');
badge.textContent = isRunPod ? 'RunPod' : 'Local';
badge.className = 'env-badge ' + (isRunPod ? 'env-remote' : 'env-local');

// Auto-fill Scope URL
const scopeInput = document.getElementById('scope-url');
const savedScope = localStorage.getItem('promap_scope_url');
if (savedScope) {
  scopeInput.value = savedScope;
} else if (isRunPod) {
  // Derive Scope URL: replace port in RunPod proxy URL
  // e.g. https://xxx-8765.proxy.runpod.net -> https://xxx-8000.proxy.runpod.net
  const m = host.match(/^(.+)-\\d+\\.proxy\\.runpod\\.net$/);
  if (m) scopeInput.value = 'https://' + m[1] + '-8000.proxy.runpod.net';
} else {
  scopeInput.value = 'http://localhost:8000';
}

function openScope() {
  const url = scopeInput.value.trim();
  if (url) {
    localStorage.setItem('promap_scope_url', url);
    window.open(url, '_blank');
  }
}

function openProjector() {
  // Open projector in a new window - user drags to projector monitor
  const w = window.open('/projector', 'promap-projector',
    'width=960,height=540,menubar=no,toolbar=no,location=no,status=no');
  if (w) w.focus();
}

// Preview auto-refresh
setInterval(() => {
  document.getElementById('preview').src = '/frame?t=' + Date.now();
}, 2000);

// Poll projector status
function updateStatus() {
  fetch('/config').then(r => r.json()).then(cfg => {
    const dot = document.getElementById('proj-dot');
    const st = document.getElementById('proj-status');
    const res = document.getElementById('proj-resolution');
    if (cfg && cfg.width) {
      dot.className = 'dot green';
      st.textContent = 'Projector connected';
      res.textContent = cfg.width + ' x ' + cfg.height +
        (cfg.monitor_name ? ' (' + cfg.monitor_name + ')' : '');
    } else {
      dot.className = 'dot yellow';
      st.textContent = 'Waiting for projector window...';
      res.textContent = '';
    }
  }).catch(() => {
    document.getElementById('proj-dot').className = 'dot red';
    document.getElementById('proj-status').textContent = 'Unreachable';
    document.getElementById('proj-resolution').textContent = '';
  });
}
updateStatus();
setInterval(updateStatus, 5000);
</script>
</body></html>
"""


class FrameStreamer:
    """MJPEG HTTP streaming server for projector output.

    Usage::

        streamer = FrameStreamer(port=8765)
        streamer.start()
        streamer.submit_frame(rgb_numpy)   # call every frame
        streamer.stop()
    """

    def __init__(self, port: int = 8765, jpeg_quality: int = 85) -> None:
        self._port = port
        self._quality = jpeg_quality
        self._frame_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._server: _ThreadedHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        # Calibration priority: when True, submit_frame() is suppressed
        self._calibration_active = False

        # Client-reported projector config (resolution, monitor name)
        self._client_config: dict | None = None
        self._load_persisted_config()

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
                if self_handler.path == "/stream":
                    self_handler._handle_stream()
                elif self_handler.path == "/frame":
                    self_handler._handle_frame()
                elif self_handler.path == "/config":
                    self_handler._handle_get_config()
                elif self_handler.path == "/projector":
                    self_handler._handle_projector()
                else:
                    self_handler._handle_control_panel()

            def do_POST(self_handler) -> None:  # noqa: N805
                if self_handler.path == "/config":
                    self_handler._handle_post_config()
                else:
                    self_handler.send_response(404)
                    self_handler.end_headers()

            def do_OPTIONS(self_handler) -> None:  # noqa: N805
                self_handler.send_response(204)
                self_handler.send_header("Access-Control-Allow-Origin", "*")
                self_handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self_handler.send_header("Access-Control-Allow-Headers", "Content-Type")
                self_handler.end_headers()

            def _handle_stream(self_handler) -> None:  # noqa: N805
                """MJPEG multipart stream."""
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
                """Control panel — Scope iframe, projector status, stream config."""
                cfg = streamer._client_config or {}
                cfg_json = json.dumps(cfg)
                html = _CONTROL_PANEL_HTML.replace("__CONFIG_JSON__", cfg_json)
                body = html.encode()
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "text/html")
                self_handler.send_header("Content-Length", str(len(body)))
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

            def log_message(self_handler, format, *args) -> None:  # noqa: N805
                pass  # suppress per-request logging

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

    def submit_frame(self, rgb: np.ndarray) -> None:
        """Submit an RGB uint8 (H, W, 3) frame for streaming.

        Suppressed when ``calibration_active`` is True — calibration patterns
        take priority via ``submit_calibration_frame()``.

        Thread-safe — may be called from any thread.
        """
        if not self._running or self._calibration_active:
            return
        self._encode_and_set(rgb)

    def submit_calibration_frame(self, rgb: np.ndarray) -> None:
        """Submit a calibration pattern frame. Always accepted.

        Thread-safe — may be called from any thread.
        """
        if not self._running:
            return
        self._encode_and_set(rgb)

    def _encode_and_set(self, rgb: np.ndarray) -> None:
        """Encode RGB frame as JPEG and update the current frame."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, jpeg_buf = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        )
        if ok:
            with self._lock:
                self._frame_jpeg = jpeg_buf.tobytes()
            self._new_frame.set()

    def stop(self) -> None:
        """Shut down the server and join the background thread."""
        self._running = False
        self._new_frame.set()  # wake up any waiting handlers
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
