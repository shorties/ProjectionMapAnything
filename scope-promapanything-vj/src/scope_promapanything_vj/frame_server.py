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
<title>ProMapAnything Control Panel</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#1a1a2e; color:#e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; height:100vh; display:flex; }

  /* Sidebar */
  .sidebar { width:320px; min-width:320px; background:#16213e; padding:20px;
             display:flex; flex-direction:column; gap:16px; overflow-y:auto; border-right: 1px solid #0f3460; }
  .sidebar h1 { font-size:18px; color:#e94560; margin-bottom:4px; }
  .sidebar .subtitle { font-size:12px; color:#888; margin-bottom:8px; }

  /* Cards */
  .card { background:#0f3460; border-radius:8px; padding:14px; }
  .card h2 { font-size:13px; color:#e94560; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px; }
  .card label { font-size:12px; color:#aaa; display:block; margin-bottom:4px; }
  .card input[type=text] { width:100%; padding:8px 10px; border:1px solid #1a1a4e; border-radius:4px;
                           background:#16213e; color:#fff; font-size:13px; outline:none; }
  .card input[type=text]:focus { border-color:#e94560; }
  .card input[type=text]::placeholder { color:#555; }

  /* Buttons */
  .btn { display:inline-flex; align-items:center; justify-content:center; gap:6px;
         padding:8px 14px; border:none; border-radius:6px; font-size:13px; font-weight:600;
         cursor:pointer; transition: background 0.2s; text-decoration:none; }
  .btn-primary { background:#e94560; color:#fff; }
  .btn-primary:hover { background:#c73652; }
  .btn-secondary { background:#1a1a4e; color:#ddd; border:1px solid #333; }
  .btn-secondary:hover { background:#252560; }
  .btn-sm { padding:6px 10px; font-size:12px; }
  .btn-row { display:flex; gap:8px; flex-wrap:wrap; }

  /* Status indicators */
  .status-row { display:flex; align-items:center; gap:8px; margin:6px 0; }
  .status-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
  .status-dot.green { background:#4ecca3; box-shadow:0 0 6px #4ecca3; }
  .status-dot.yellow { background:#f0c040; box-shadow:0 0 6px #f0c040; }
  .status-dot.red { background:#e94560; box-shadow:0 0 6px #e94560; }
  .status-text { font-size:13px; }
  .status-detail { font-size:11px; color:#888; margin-left:18px; }

  /* Preview */
  .preview-container { position:relative; border-radius:6px; overflow:hidden;
                       background:#000; aspect-ratio:16/9; }
  .preview-container img { width:100%; height:100%; object-fit:contain; }
  .preview-label { position:absolute; top:6px; left:8px; font-size:10px; color:#888;
                   background:rgba(0,0,0,0.6); padding:2px 6px; border-radius:3px; }

  /* Main area */
  .main { flex:1; display:flex; flex-direction:column; }
  .iframe-bar { display:flex; align-items:center; gap:10px; padding:8px 16px;
                background:#16213e; border-bottom:1px solid #0f3460; }
  .iframe-bar .url-display { flex:1; font-size:12px; color:#888; overflow:hidden;
                             text-overflow:ellipsis; white-space:nowrap; }
  .iframe-container { flex:1; position:relative; }
  .iframe-container iframe { width:100%; height:100%; border:none; }
  .iframe-placeholder { position:absolute; inset:0; display:flex; flex-direction:column;
                        align-items:center; justify-content:center; gap:16px; color:#555; }
  .iframe-placeholder .icon { font-size:48px; }
  .iframe-placeholder p { font-size:14px; max-width:400px; text-align:center; line-height:1.5; }
</style>
</head><body>

<div class="sidebar">
  <div>
    <h1>ProMapAnything</h1>
    <div class="subtitle">Control Panel</div>
  </div>

  <!-- Scope Connection -->
  <div class="card">
    <h2>Scope Instance</h2>
    <label for="scope-url">RunPod / Scope URL</label>
    <input type="text" id="scope-url" placeholder="https://xxxxx-8000.proxy.runpod.net" spellcheck="false" />
    <div style="margin-top:10px;" class="btn-row">
      <button class="btn btn-primary btn-sm" onclick="loadScope()">Load</button>
      <button class="btn btn-secondary btn-sm" onclick="openScopeTab()">Open in Tab</button>
    </div>
  </div>

  <!-- Projector Output -->
  <div class="card">
    <h2>Projector Output</h2>
    <div class="btn-row" style="margin-bottom:10px;">
      <button class="btn btn-primary" onclick="openProjector()">Open Projector Window</button>
    </div>
    <div class="status-row">
      <div class="status-dot" id="proj-dot"></div>
      <span class="status-text" id="proj-status">Checking...</span>
    </div>
    <div class="status-detail" id="proj-resolution"></div>
    <div class="status-detail" id="proj-calibration"></div>
  </div>

  <!-- Stream Preview -->
  <div class="card">
    <h2>Stream Preview</h2>
    <div class="preview-container">
      <img id="preview" src="/frame" />
      <div class="preview-label">Live</div>
    </div>
    <div style="margin-top:8px; text-align:center;">
      <button class="btn btn-secondary btn-sm" onclick="refreshPreview()">Refresh</button>
    </div>
  </div>

  <!-- Quick Links -->
  <div class="card">
    <h2>Endpoints</h2>
    <div style="font-size:12px; line-height:1.8;">
      <a href="/stream" target="_blank" style="color:#4ecca3;">/stream</a> &mdash; MJPEG stream<br>
      <a href="/frame" target="_blank" style="color:#4ecca3;">/frame</a> &mdash; JPEG snapshot<br>
      <a href="/config" target="_blank" style="color:#4ecca3;">/config</a> &mdash; Projector config (JSON)<br>
      <a href="/projector" target="_blank" style="color:#4ecca3;">/projector</a> &mdash; Fullscreen viewer
    </div>
  </div>
</div>

<div class="main">
  <div class="iframe-bar">
    <span class="url-display" id="iframe-url-display">No Scope URL configured</span>
    <button class="btn btn-secondary btn-sm" onclick="reloadIframe()">Reload</button>
  </div>
  <div class="iframe-container">
    <div class="iframe-placeholder" id="placeholder">
      <div class="icon">&#127916;</div>
      <p>Paste your RunPod Scope URL in the sidebar and click <strong>Load</strong> to embed the Scope interface here.</p>
      <p style="font-size:12px; color:#444;">e.g. https://xxxxx-8000.proxy.runpod.net</p>
    </div>
    <iframe id="scope-iframe" style="display:none;" allow="camera;microphone;fullscreen"></iframe>
  </div>
</div>

<script>
const initialConfig = __CONFIG_JSON__;

// Persist Scope URL in localStorage
const urlInput = document.getElementById('scope-url');
const saved = localStorage.getItem('promap_scope_url');
if (saved) { urlInput.value = saved; loadScope(); }

function loadScope() {
  const url = urlInput.value.trim();
  if (!url) return;
  localStorage.setItem('promap_scope_url', url);
  const iframe = document.getElementById('scope-iframe');
  const ph = document.getElementById('placeholder');
  document.getElementById('iframe-url-display').textContent = url;
  iframe.src = url;
  iframe.style.display = 'block';
  ph.style.display = 'none';
}

function openScopeTab() {
  const url = urlInput.value.trim();
  if (url) window.open(url, '_blank');
}

function reloadIframe() {
  const iframe = document.getElementById('scope-iframe');
  if (iframe.src) iframe.src = iframe.src;
}

function openProjector() {
  const w = window.open('/projector', 'promap-projector',
    'width=960,height=540,menubar=no,toolbar=no,location=no,status=no');
  if (w) w.focus();
}

// Preview refresh
let previewTimer = null;
function refreshPreview() {
  document.getElementById('preview').src = '/frame?t=' + Date.now();
}
// Auto-refresh preview every 2s
previewTimer = setInterval(refreshPreview, 2000);

// Poll projector status
function updateStatus() {
  fetch('/config').then(r => r.json()).then(cfg => {
    const dot = document.getElementById('proj-dot');
    const status = document.getElementById('proj-status');
    const res = document.getElementById('proj-resolution');
    const cal = document.getElementById('proj-calibration');
    if (cfg && cfg.width) {
      dot.className = 'status-dot green';
      status.textContent = 'Projector connected';
      res.textContent = cfg.width + ' x ' + cfg.height +
        (cfg.monitor_name ? ' (' + cfg.monitor_name + ')' : '');
    } else {
      dot.className = 'status-dot yellow';
      status.textContent = 'Waiting for projector...';
      res.textContent = 'Open the projector window and drag it to your projector monitor';
    }
  }).catch(() => {
    document.getElementById('proj-dot').className = 'status-dot red';
    document.getElementById('proj-status').textContent = 'Stream server unreachable';
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
