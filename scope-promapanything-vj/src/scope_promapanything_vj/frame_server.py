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

Works through RunPod's port proxy — expose the port and connect from anywhere.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import unquote

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
  img#view { width:100vw; height:100vh;
        object-fit:contain; display:block; }
  #hint { position:fixed; bottom:30px;
          left:50%; transform:translateX(-50%);
          color:#444; font:14px sans-serif;
          pointer-events:none;
          transition:opacity 0.5s; }
  body.fs #hint { opacity:0; }
  #status { position:fixed; top:10px; right:10px;
            color:#333; font:11px monospace;
            pointer-events:none; }
  body.fs #status { opacity:0; }
</style>
</head><body>
<img id="view" />
<div id="hint">Click to go fullscreen &mdash; drag this window to your projector first</div>
<div id="status"></div>
<script>
const img = document.getElementById('view');
const statusEl = document.getElementById('status');

// MJPEG stream with auto-reconnect on failure
function startStream() {
  img.src = '/stream?t=' + Date.now();
  statusEl.textContent = '';
}

img.onerror = () => {
  statusEl.textContent = 'Reconnecting...';
  setTimeout(startStream, 1000);
};

// Detect stalled stream (no new frame for 5s)
let lastCheck = 0;
setInterval(() => {
  // img.complete && img.naturalHeight > 0 means it has decoded at least one frame
  if (img.src.includes('/stream') && img.naturalHeight === 0) {
    // Stream never started — reconnect
    statusEl.textContent = 'Reconnecting...';
    startStream();
  }
}, 3000);

startStream();

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
         min-height:100vh; padding:24px; }
  .container { max-width:720px; margin:0 auto; display:flex;
               flex-direction:column; gap:20px; }

  /* Header */
  .header { display:flex; align-items:center; justify-content:space-between; }
  h1 { font-size:22px; color:#e94560; }
  .env-badge { display:inline-block; padding:2px 8px; border-radius:4px;
               font-size:11px; font-weight:600; margin-left:8px; }
  .env-local { background:#1a4a1a; color:#4ecca3; }
  .env-remote { background:#4a1a1a; color:#e94560; }

  /* Buttons row */
  .btn-row { display:flex; gap:10px; flex-wrap:wrap; }
  .btn { padding:10px 20px; border:none; border-radius:8px;
         font-size:14px; font-weight:600; cursor:pointer; transition:background 0.2s; }
  .btn-primary { background:#e94560; color:#fff; box-shadow:0 2px 12px rgba(233,69,96,0.3); }
  .btn-primary:hover { background:#c73652; }
  .btn-secondary { background:#0f3460; color:#ddd; }
  .btn-secondary:hover { background:#1a4a80; }

  /* Cards */
  .card { background:#16213e; border-radius:10px; padding:16px; }
  .card h2 { font-size:14px; color:#888; text-transform:uppercase;
             letter-spacing:1px; margin-bottom:12px; }

  /* Live preview */
  .preview { border-radius:8px; overflow:hidden; background:#000;
             width:100%; aspect-ratio:16/9; }
  .preview img { width:100%; height:100%; object-fit:contain; display:block; }

  /* Progress bar */
  .progress-wrap { background:#0d1b2e; border-radius:6px; height:22px;
                   overflow:hidden; position:relative; }
  .progress-bar { height:100%; background:linear-gradient(90deg, #4ecca3, #36b88e);
                  border-radius:6px; transition:width 0.3s ease; min-width:0; }
  .progress-text { position:absolute; inset:0; display:flex;
                   align-items:center; justify-content:center;
                   font-size:11px; font-weight:600; color:#fff; }
  .calib-detail { font-size:12px; color:#aaa; margin-top:8px; line-height:1.6; }
  .calib-detail .label { color:#666; }
  .calib-errors { color:#e94560; font-size:12px; margin-top:6px; }
  .calib-idle { color:#555; font-size:13px; font-style:italic; }

  /* Results */
  .result-meta { font-size:12px; color:#888; margin-bottom:12px; }
  .file-list { display:flex; flex-direction:column; gap:6px; }
  .file-row { display:flex; align-items:center; justify-content:space-between;
              background:#0d1b2e; border-radius:6px; padding:8px 12px; }
  .file-name { font-size:13px; font-family:monospace; color:#ddd; }
  .file-icon { margin-right:6px; }
  .dl-btn { padding:4px 12px; border:none; border-radius:4px;
            background:#4ecca3; color:#000; font:bold 11px sans-serif;
            cursor:pointer; transition:background 0.2s; }
  .dl-btn:hover { background:#3db893; }
  .result-actions { display:flex; gap:8px; margin-top:10px; }
  .btn-dl-all { padding:8px 18px; border:none; border-radius:6px;
                background:#e94560; color:#fff; font:bold 13px sans-serif;
                cursor:pointer; }
  .btn-dl-all:hover { background:#c73652; }

  /* Thumbnails */
  .thumb-row { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
  .thumb { width:140px; height:80px; border-radius:6px; overflow:hidden;
           background:#0d1b2e; cursor:pointer; border:2px solid transparent;
           transition:border-color 0.2s; }
  .thumb:hover { border-color:#4ecca3; }
  .thumb img { width:100%; height:100%; object-fit:cover; }

  /* Projector status */
  .status-row { display:flex; align-items:center; gap:8px; font-size:13px; }
  .dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
  .dot.green { background:#4ecca3; box-shadow:0 0 6px #4ecca3; }
  .dot.yellow { background:#f0c040; box-shadow:0 0 6px #f0c040; }
  .dot.red { background:#e94560; }
  .status-detail { font-size:11px; color:#666; margin-top:2px; }

  /* Scope section */
  .scope-row { display:flex; align-items:center; gap:8px; }
  .scope-row input { flex:1; padding:6px 10px; border:1px solid #333; border-radius:4px;
                     background:#0d1b2e; color:#fff; font-size:12px; outline:none; }
  .scope-row input:focus { border-color:#e94560; }

  /* Footer links */
  .links { font-size:12px; color:#555; text-align:center; }
  .links a { color:#4ecca3; }

  /* Hidden */
  .hidden { display:none !important; }
</style>
</head><body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>ProMapAnything <span class="env-badge" id="env-badge"></span></h1>
  </div>

  <!-- Action buttons -->
  <div class="btn-row">
    <button class="btn btn-primary" onclick="openProjector()">Open Projector Window</button>
    <button class="btn btn-secondary" onclick="openScope()">Open Scope</button>
  </div>

  <!-- Live Preview -->
  <div class="card">
    <h2>Live Preview</h2>
    <div class="preview">
      <img id="preview" src="/frame" />
    </div>
  </div>

  <!-- VACE Input Preview -->
  <div class="card">
    <h2>VACE Input (preprocessor output)</h2>
    <div class="preview">
      <img id="input-preview" src="/input-frame" />
    </div>
  </div>

  <!-- Calibration Status -->
  <div class="card" id="calib-status-card">
    <h2>Calibration Status</h2>
    <div id="calib-idle" class="calib-idle">Idle &mdash; toggle Start Calibration in Scope to begin</div>
    <div id="calib-active" class="hidden">
      <div class="progress-wrap">
        <div class="progress-bar" id="calib-bar" style="width:0%"></div>
        <div class="progress-text" id="calib-pct">0%</div>
      </div>
      <div class="calib-detail">
        <div><span class="label">Phase:</span> <span id="calib-phase">---</span></div>
        <div><span class="label">Pattern:</span> <span id="calib-pattern">---</span></div>
      </div>
      <div class="calib-errors hidden" id="calib-errors"></div>
    </div>
  </div>

  <!-- Calibration Results -->
  <div class="card hidden" id="calib-results-card">
    <h2>Calibration Results</h2>
    <div class="result-meta" id="result-meta"></div>
    <div class="file-list" id="result-files"></div>
    <div class="result-actions">
      <button class="btn-dl-all" onclick="downloadAll()">Download All</button>
    </div>
    <div class="thumb-row" id="result-thumbs"></div>
  </div>

  <!-- Projector Status -->
  <div class="card">
    <h2>Projector Status</h2>
    <div class="status-row">
      <div class="dot" id="proj-dot"></div>
      <span id="proj-status">Checking...</span>
    </div>
    <div class="status-detail" id="proj-resolution"></div>
  </div>

  <!-- Scope URL -->
  <div class="card">
    <h2>Scope Connection</h2>
    <div class="scope-row">
      <input type="text" id="scope-url"
             placeholder="Scope URL (auto-detected or paste RunPod URL)" spellcheck="false" />
      <button class="btn btn-secondary" onclick="openScope()">Open</button>
    </div>
  </div>

  <!-- Links -->
  <div class="links">
    <a href="/projector" target="_blank">/projector</a> &middot;
    <a href="/stream" target="_blank">/stream</a> &middot;
    <a href="/frame" target="_blank">/frame</a> &middot;
    <a href="/config" target="_blank">/config</a> &middot;
    <a href="/calibration/status" target="_blank">/calibration/status</a> &middot;
    <a href="/input-stream" target="_blank">/input-stream</a>
  </div>

</div>

<script>
// -- Environment detection --
const host = window.location.hostname;
const isRunPod = host.includes('.proxy.runpod.net');
const badge = document.getElementById('env-badge');
badge.textContent = isRunPod ? 'RunPod' : 'Local';
badge.className = 'env-badge ' + (isRunPod ? 'env-remote' : 'env-local');

// -- Scope URL auto-fill --
const scopeInput = document.getElementById('scope-url');
const savedScope = localStorage.getItem('promap_scope_url');
if (savedScope) {
  scopeInput.value = savedScope;
} else if (isRunPod) {
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
  const w = window.open('/projector', 'promap-projector',
    'width=960,height=540,menubar=no,toolbar=no,location=no,status=no');
  if (w) w.focus();
}

// -- Preview auto-refresh --
setInterval(() => {
  document.getElementById('preview').src = '/frame?t=' + Date.now();
  document.getElementById('input-preview').src = '/input-frame?t=' + Date.now();
}, 2000);

// -- Projector status polling --
function updateProjectorStatus() {
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
updateProjectorStatus();
setInterval(updateProjectorStatus, 5000);

// -- Calibration status polling --
let lastCalibTs = '';
let resultFiles = [];

function triggerDownload(name) {
  const a = document.createElement('a');
  a.href = '/calibration/download/' + encodeURIComponent(name);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function downloadAll() {
  let delay = 0;
  resultFiles.forEach(name => {
    setTimeout(() => triggerDownload(name), delay);
    delay += 300;
  });
}

function updateCalibrationUI(data) {
  const idleEl = document.getElementById('calib-idle');
  const activeEl = document.getElementById('calib-active');
  const resultsCard = document.getElementById('calib-results-card');

  // Active calibration
  if (data.active && !data.complete) {
    idleEl.classList.add('hidden');
    activeEl.classList.remove('hidden');

    const pct = Math.round(data.progress * 100);
    document.getElementById('calib-bar').style.width = pct + '%';
    document.getElementById('calib-pct').textContent = pct + '%';
    document.getElementById('calib-phase').textContent = data.phase || '---';
    document.getElementById('calib-pattern').textContent = data.pattern_info || '---';

    const errEl = document.getElementById('calib-errors');
    if (data.errors && data.errors.length > 0) {
      errEl.classList.remove('hidden');
      errEl.textContent = data.errors.join('; ');
    } else {
      errEl.classList.add('hidden');
    }
  } else if (!data.active && !data.complete) {
    // Idle
    idleEl.classList.remove('hidden');
    activeEl.classList.add('hidden');

    idleEl.textContent = 'Idle \\u2014 toggle Start Calibration in Scope to begin';
  } else {
    // Complete
    idleEl.classList.remove('hidden');
    activeEl.classList.add('hidden');
    idleEl.textContent = 'Calibration complete';
  }

  // Results section
  if (data.complete && data.files && data.files.length > 0) {
    resultsCard.classList.remove('hidden');
    resultFiles = data.files;

    // Meta
    const meta = document.getElementById('result-meta');
    let metaText = '';
    if (data.timestamp) {
      metaText += 'Captured: ' + new Date(data.timestamp).toLocaleString();
    }
    if (data.coverage_pct > 0) {
      metaText += ' \\u2014 Coverage: ' + data.coverage_pct.toFixed(1) + '%';
    }
    meta.textContent = metaText;

    // File list (only rebuild if timestamp changed)
    if (data.timestamp !== lastCalibTs) {
      lastCalibTs = data.timestamp;
      const filesDiv = document.getElementById('result-files');
      filesDiv.innerHTML = '';
      data.files.forEach(name => {
        const row = document.createElement('div');
        row.className = 'file-row';
        const icon = name.endsWith('.json') ? '\\ud83d\\udcc4' : '\\ud83d\\uddbc';
        row.innerHTML = '<span class="file-name"><span class="file-icon">' + icon +
          '</span>' + name + '</span>' +
          '<button class="dl-btn" onclick="triggerDownload(\\'' +
          name.replace(/'/g, "\\\\'") + '\\')">Download</button>';
        filesDiv.appendChild(row);
      });

      // Thumbnails
      const thumbDiv = document.getElementById('result-thumbs');
      thumbDiv.innerHTML = '';
      data.files.forEach(name => {
        if (!name.endsWith('.png')) return;
        const thumb = document.createElement('div');
        thumb.className = 'thumb';
        thumb.title = name;
        thumb.onclick = () => window.open('/calibration/preview/' + encodeURIComponent(name), '_blank');
        thumb.innerHTML = '<img src="/calibration/preview/' +
          encodeURIComponent(name) + '?t=' + Date.now() + '" />';
        thumbDiv.appendChild(thumb);
      });
    }
  } else {
    resultsCard.classList.add('hidden');
  }
}

async function pollCalibration() {
  try {
    const r = await fetch('/calibration/status');
    const data = await r.json();
    updateCalibrationUI(data);
  } catch {}
}

// Adaptive polling: 1s during calibration, 3s otherwise
let calibActive = false;
async function calibPoll() {
  await pollCalibration();
  setTimeout(calibPoll, calibActive ? 1000 : 3000);
}

// Override to track active state
const origUpdate = updateCalibrationUI;
updateCalibrationUI = function(data) {
  calibActive = data.active && !data.complete;
  origUpdate(data);
};

calibPoll();
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
                """Control panel dashboard — calibration status, projector, downloads."""
                body = _CONTROL_PANEL_HTML.encode()
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

        Thread-safe — may be called from any thread.
        """
        if not self._running:
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
