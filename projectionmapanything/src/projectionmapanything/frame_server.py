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
<title>ProjectionMapAnything</title>
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

  /* Standalone calibration */
  .sc-video-wrap { position:relative; border-radius:8px; overflow:hidden;
                   background:#000; width:100%; aspect-ratio:16/9; }
  .sc-video-wrap video { width:100%; height:100%; object-fit:contain; display:block; }
  .sc-video-wrap canvas { display:none; }
  .sc-controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:10px; }
  .sc-controls label { font-size:12px; color:#aaa; }
  .sc-controls input[type=number] { width:70px; padding:4px 6px; background:#0d1b2e;
    color:#fff; border:1px solid #333; border-radius:4px; font-size:12px; }
  .sc-status { font-size:12px; color:#888; margin-top:8px; min-height:18px; }
  .sc-status.error { color:#e94560; }
  .sc-status.success { color:#4ecca3; }
</style>
</head><body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>ProjectionMapAnything <span class="env-badge" id="env-badge"></span></h1>
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

  <!-- VACE Input Preview (hidden until preprocessor feeds frames) -->
  <div class="card hidden" id="input-preview-card">
    <h2>VACE Input (preprocessor output)</h2>
    <div class="preview">
      <img id="input-preview" />
    </div>
  </div>

  <!-- Standalone Calibration (browser webcam) -->
  <div class="card">
    <h2>Standalone Calibration</h2>
    <div class="sc-video-wrap" id="sc-video-wrap" style="display:none;">
      <video id="sc-video" autoplay playsinline muted></video>
      <canvas id="sc-canvas"></canvas>
    </div>
    <div class="sc-controls">
      <button class="btn btn-secondary" id="sc-webcam-btn" onclick="scToggleWebcam()">Enable Webcam</button>
      <button class="btn btn-primary" id="sc-start-btn" onclick="scStartCalibration()" disabled>Start Calibration</button>
      <button class="btn btn-secondary" id="sc-stop-btn" onclick="scStopCalibration()" style="display:none;">Cancel</button>
    </div>
    <div class="sc-controls">
      <label>Proj W: <input type="number" id="sc-proj-w" value="1920" min="320" max="7680" /></label>
      <label>Proj H: <input type="number" id="sc-proj-h" value="1080" min="240" max="4320" /></label>
      <label>Brightness: <input type="number" id="sc-brightness" value="128" min="10" max="255" /></label>
    </div>
    <div class="sc-status" id="sc-status">Enable webcam to begin standalone calibration (no Scope required)</div>
  </div>

  <!-- Calibration Status -->
  <div class="card" id="calib-status-card">
    <h2>Calibration Status</h2>
    <div id="calib-idle" class="calib-idle">Idle &mdash; use Standalone Calibration above or toggle Start Calibration in Scope</div>
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

  <!-- Calibration Transfer -->
  <div class="card">
    <h2>Calibration Transfer</h2>
    <div style="display:flex; flex-direction:column; gap:10px;">
      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
        <button class="btn btn-secondary" onclick="exportCalibration()" style="white-space:nowrap;">Export Calibration (.zip)</button>
        <label class="btn btn-secondary" style="white-space:nowrap; cursor:pointer;">
          Import Calibration (.zip)
          <input type="file" id="import-file" accept=".zip" style="display:none;" onchange="importCalibration()" />
        </label>
      </div>
      <div id="transfer-status" style="font-size:12px; color:#666;"></div>
    </div>
  </div>

  <!-- Custom Depth Upload -->
  <div class="card">
    <h2>Custom Depth Map</h2>
    <div style="display:flex; flex-direction:column; gap:10px;">
      <div style="display:flex; gap:8px; align-items:center;">
        <select id="upload-stage" style="padding:6px 10px; background:#0d1b2e; color:#fff; border:1px solid #333; border-radius:4px; font-size:12px;">
          <option value="depth_warped">Depth (warped, ready)</option>
          <option value="depth_estimated">Depth (needs warp)</option>
          <option value="raw_camera">Raw camera (needs depth+warp)</option>
        </select>
        <select id="upload-type" style="padding:6px 10px; background:#0d1b2e; color:#fff; border:1px solid #333; border-radius:4px; font-size:12px;">
          <option value="depth">Depth Map</option>
          <option value="mask">Isolation Mask</option>
        </select>
      </div>
      <div style="display:flex; gap:8px; align-items:center;">
        <input type="file" id="upload-file" accept="image/*" style="flex:1; font-size:12px; color:#aaa;" />
        <button class="btn btn-secondary" onclick="uploadCustom()" style="white-space:nowrap;">Upload</button>
      </div>
      <div id="upload-status" style="font-size:12px; color:#666;"></div>
      <div id="upload-preview" style="display:none; max-width:200px;">
        <img id="upload-thumb" style="width:100%; border-radius:4px;" />
      </div>
    </div>
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

  <!-- Dashboard URL (for sharing) -->
  <div class="card">
    <h2>Dashboard URL</h2>
    <div style="font-size:12px; color:#aaa; line-height:1.6;">
      <div id="dashboard-url-hint"></div>
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
const savedScope = localStorage.getItem('pma_scope_url');
if (savedScope) {
  scopeInput.value = savedScope;
} else if (isRunPod) {
  const m = host.match(/^(.+)-\\d+\\.proxy\\.runpod\\.net$/);
  if (m) scopeInput.value = 'https://' + m[1] + '-8000.proxy.runpod.net';
} else {
  scopeInput.value = 'http://localhost:8000';
}

// -- Dashboard URL hint (dynamic) --
(function() {
  const hintEl = document.getElementById('dashboard-url-hint');
  const curUrl = window.location.href.replace(/\\/+$/, '');
  if (isRunPod) {
    // Build the Scope URL by replacing port in the RunPod proxy URL
    const scopeUrl = curUrl.replace(/-\\d+\\.proxy\\.runpod\\.net/, '-8000.proxy.runpod.net');
    hintEl.innerHTML = 'This dashboard: <a href="' + curUrl + '" style="color:#4ecca3;word-break:break-all;">' + curUrl + '</a>' +
      '<br>Scope UI: <a href="' + scopeUrl + '" target="_blank" style="color:#4ecca3;word-break:break-all;">' + scopeUrl + '</a>' +
      '<br><span style="color:#666;">Tip: change <b>-8000.</b> to <b>-8765.</b> in the Scope URL to get here.</span>';
  } else {
    const dashPort = window.location.port || '8765';
    hintEl.innerHTML = 'This dashboard: <a href="' + curUrl + '" style="color:#4ecca3;">' + curUrl + '</a>' +
      '<br>Scope UI: <a href="http://localhost:8000" target="_blank" style="color:#4ecca3;">http://localhost:8000</a>' +
      '<br><span style="color:#666;">Tip: change <b>:8000</b> to <b>:' + dashPort + '</b> in the Scope URL to get here.</span>';
  }
})();

function openScope() {
  const url = scopeInput.value.trim();
  if (url) {
    localStorage.setItem('pma_scope_url', url);
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
}, 2000);

// Input preview: only poll if the endpoint has content (avoids 204 flicker)
let inputPreviewActive = false;
async function checkInputPreview() {
  try {
    const r = await fetch('/input-frame', { method: 'HEAD' });
    if (r.status === 200) {
      document.getElementById('input-preview-card').classList.remove('hidden');
      document.getElementById('input-preview').src = '/input-frame?t=' + Date.now();
      inputPreviewActive = true;
    } else if (inputPreviewActive) {
      inputPreviewActive = false;
      document.getElementById('input-preview-card').classList.add('hidden');
    }
  } catch {}
}
setInterval(checkInputPreview, 3000);

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

    idleEl.textContent = 'Idle \\u2014 use Standalone Calibration above or toggle Start Calibration in Scope';
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

// -- Custom depth/mask upload --
async function uploadCustom() {
  const fileInput = document.getElementById('upload-file');
  const stage = document.getElementById('upload-stage').value;
  const type = document.getElementById('upload-type').value;
  const statusEl = document.getElementById('upload-status');
  const previewEl = document.getElementById('upload-preview');
  const thumbEl = document.getElementById('upload-thumb');

  if (!fileInput.files || !fileInput.files[0]) {
    statusEl.textContent = 'No file selected';
    statusEl.style.color = '#e94560';
    return;
  }

  statusEl.textContent = 'Uploading...';
  statusEl.style.color = '#aaa';

  try {
    const file = fileInput.files[0];
    const arrayBuf = await file.arrayBuffer();
    const resp = await fetch('/upload?stage=' + stage + '&type=' + type, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: arrayBuf
    });
    const data = await resp.json();
    if (data.ok) {
      statusEl.textContent = 'Uploaded: ' + data.filename + ' (stage: ' + data.stage + '). Set depth_mode to "custom" in Scope.';
      statusEl.style.color = '#4ecca3';
      // Show preview
      thumbEl.src = URL.createObjectURL(file);
      previewEl.style.display = 'block';
    } else {
      statusEl.textContent = 'Upload failed: ' + (data.error || 'unknown');
      statusEl.style.color = '#e94560';
    }
  } catch (err) {
    statusEl.textContent = 'Upload error: ' + err.message;
    statusEl.style.color = '#e94560';
  }
}

// -- Standalone calibration (browser webcam) --
let scStream = null;       // MediaStream from getUserMedia
let scRunning = false;     // capture loop active
let scWebcamOn = false;

async function scToggleWebcam() {
  const video = document.getElementById('sc-video');
  const wrap = document.getElementById('sc-video-wrap');
  const btn = document.getElementById('sc-webcam-btn');
  const startBtn = document.getElementById('sc-start-btn');
  const statusEl = document.getElementById('sc-status');

  if (scWebcamOn) {
    // Turn off
    if (scStream) { scStream.getTracks().forEach(t => t.stop()); scStream = null; }
    video.srcObject = null;
    wrap.style.display = 'none';
    btn.textContent = 'Enable Webcam';
    startBtn.disabled = true;
    scWebcamOn = false;
    statusEl.textContent = 'Webcam disabled';
    statusEl.className = 'sc-status';
    return;
  }

  try {
    statusEl.textContent = 'Requesting camera access...';
    scStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = scStream;
    wrap.style.display = 'block';
    btn.textContent = 'Disable Webcam';
    startBtn.disabled = false;
    scWebcamOn = true;
    statusEl.textContent = 'Webcam ready. Open Projector Window, position it on the projector, then click Start Calibration.';
    statusEl.className = 'sc-status';

    // Auto-fill projector resolution from /config if available
    try {
      const cfg = await (await fetch('/config')).json();
      if (cfg && cfg.width) {
        document.getElementById('sc-proj-w').value = cfg.width;
        document.getElementById('sc-proj-h').value = cfg.height;
      }
    } catch {}
  } catch (err) {
    statusEl.textContent = 'Camera error: ' + err.message;
    statusEl.className = 'sc-status error';
  }
}

async function scStartCalibration() {
  if (!scWebcamOn || scRunning) return;
  const statusEl = document.getElementById('sc-status');
  const startBtn = document.getElementById('sc-start-btn');
  const stopBtn = document.getElementById('sc-stop-btn');

  const config = {
    proj_w: parseInt(document.getElementById('sc-proj-w').value) || 1920,
    proj_h: parseInt(document.getElementById('sc-proj-h').value) || 1080,
    max_brightness: parseInt(document.getElementById('sc-brightness').value) || 128,
  };

  statusEl.textContent = 'Starting calibration...';
  statusEl.className = 'sc-status';

  try {
    const resp = await fetch('/calibrate/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    const data = await resp.json();
    if (!data.ok) {
      statusEl.textContent = 'Start failed: ' + (data.error || 'unknown');
      statusEl.className = 'sc-status error';
      return;
    }

    // Open projector pop-out
    openProjector();

    scRunning = true;
    startBtn.disabled = true;
    stopBtn.style.display = '';
    statusEl.textContent = 'Calibrating... ' + data.total_patterns + ' patterns';
    statusEl.className = 'sc-status';

    // Start capture loop
    scCaptureLoop();
  } catch (err) {
    statusEl.textContent = 'Start error: ' + err.message;
    statusEl.className = 'sc-status error';
  }
}

async function scCaptureLoop() {
  if (!scRunning) return;
  const video = document.getElementById('sc-video');
  const canvas = document.getElementById('sc-canvas');
  const statusEl = document.getElementById('sc-status');

  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
  const ctx = canvas.getContext('2d');

  while (scRunning) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    try {
      const blob = await new Promise(resolve =>
        canvas.toBlob(resolve, 'image/jpeg', 0.85)
      );
      if (!blob || !scRunning) break;

      const resp = await fetch('/calibrate/frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: blob,
      });
      const data = await resp.json();

      if (data.error) {
        statusEl.textContent = 'Error: ' + data.error;
        statusEl.className = 'sc-status error';
      } else if (data.done) {
        scRunning = false;
        const cov = data.coverage_pct ? ' (coverage: ' + data.coverage_pct.toFixed(1) + '%)' : '';
        statusEl.textContent = 'Calibration complete!' + cov + ' Check results below.';
        statusEl.className = 'sc-status success';
        document.getElementById('sc-start-btn').disabled = false;
        document.getElementById('sc-stop-btn').style.display = 'none';
        break;
      } else {
        statusEl.textContent = data.phase + ' — ' + (data.pattern_info || '') +
          ' (' + Math.round((data.progress || 0) * 100) + '%)';
        statusEl.className = 'sc-status';
      }
    } catch (err) {
      statusEl.textContent = 'Frame error: ' + err.message;
      statusEl.className = 'sc-status error';
    }

    // Throttle to ~12fps (80ms between frames)
    await new Promise(r => setTimeout(r, 80));
  }
}

async function scStopCalibration() {
  scRunning = false;
  const statusEl = document.getElementById('sc-status');
  try {
    await fetch('/calibrate/stop', { method: 'POST' });
    statusEl.textContent = 'Calibration cancelled.';
    statusEl.className = 'sc-status';
  } catch {}
  document.getElementById('sc-start-btn').disabled = !scWebcamOn;
  document.getElementById('sc-stop-btn').style.display = 'none';
}

// -- Calibration export/import --
function exportCalibration() {
  const statusEl = document.getElementById('transfer-status');
  statusEl.textContent = 'Exporting...';
  statusEl.style.color = '#aaa';
  fetch('/calibration/export')
    .then(r => {
      if (!r.ok) throw new Error('No calibration found');
      return r.blob();
    })
    .then(blob => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'projectionmapanything_calibration.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(a.href);
      statusEl.textContent = 'Calibration exported successfully';
      statusEl.style.color = '#4ecca3';
    })
    .catch(err => {
      statusEl.textContent = 'Export failed: ' + err.message;
      statusEl.style.color = '#e94560';
    });
}

async function importCalibration() {
  const fileInput = document.getElementById('import-file');
  const statusEl = document.getElementById('transfer-status');
  if (!fileInput.files || !fileInput.files[0]) return;

  statusEl.textContent = 'Importing...';
  statusEl.style.color = '#aaa';

  try {
    const file = fileInput.files[0];
    const arrayBuf = await file.arrayBuffer();
    const resp = await fetch('/calibration/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/zip' },
      body: arrayBuf
    });
    const data = await resp.json();
    if (data.ok) {
      statusEl.textContent = 'Imported ' + data.files + ' files. Reload the depth preprocessor in Scope to use the new calibration.';
      statusEl.style.color = '#4ecca3';
    } else {
      statusEl.textContent = 'Import failed: ' + (data.error || 'unknown');
      statusEl.style.color = '#e94560';
    }
  } catch (err) {
    statusEl.textContent = 'Import error: ' + err.message;
    statusEl.style.color = '#e94560';
  }
  fileInput.value = '';
}
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
        settle_frames: int = 15,
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

        self._standalone_calib = CalibrationState(
            proj_w, proj_h,
            settle_frames=settle_frames,
            capture_frames=capture_frames,
            max_brightness=max_brightness,
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
        dict with phase, progress, pattern_info, done, and optionally coverage_pct.
        """
        from .calibration import CalibrationPhase

        calib = self._standalone_calib
        if calib is None:
            return {"error": "No calibration in progress", "done": True}

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

        # Step the calibration state machine
        pattern = calib.step(tensor, device)

        # Build progress info
        phase = calib.phase.name
        progress = calib.progress
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
        elif calib.phase == CalibrationPhase.WHITE:
            pattern_info = "Capturing white reference"
        elif calib.phase == CalibrationPhase.BLACK:
            pattern_info = "Capturing black reference"
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
