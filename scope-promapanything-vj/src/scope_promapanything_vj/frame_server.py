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
``/``        Status page with an embedded ``<img>`` viewer.
``/stream``  MJPEG multipart stream (``multipart/x-mixed-replace``).
``/frame``   Single JPEG snapshot of the latest frame.
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
                else:
                    self_handler._handle_status()

            def do_POST(self_handler) -> None:  # noqa: N805
                if self_handler.path == "/config":
                    self_handler._handle_post_config()
                else:
                    self_handler.send_response(404)
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

            def _handle_status(self_handler) -> None:  # noqa: N805
                """Projector viewer page — acts as a browser-based companion app.

                Features:
                - Click anywhere to enter fullscreen
                - ESC to exit fullscreen
                - Auto-POSTs screen resolution to /config
                - Re-POSTs every 30s (handles server restarts)
                - Hidden cursor in fullscreen
                - Black background, no UI chrome
                """
                html = (
                    "<!DOCTYPE html>\n"
                    "<html><head>\n"
                    "<title>ProMapAnything Projector</title>\n"
                    "<style>\n"
                    "  * { margin:0; padding:0; }\n"
                    "  body { background:#000; overflow:hidden;\n"
                    "         width:100vw; height:100vh;\n"
                    "         display:flex; justify-content:center;\n"
                    "         align-items:center; }\n"
                    "  body.fs { cursor:none; }\n"
                    "  img { width:100vw; height:100vh;\n"
                    "        object-fit:contain; display:block; }\n"
                    "  #hint { position:fixed; bottom:30px;\n"
                    "          left:50%; transform:translateX(-50%);\n"
                    "          color:#444; font:14px sans-serif;\n"
                    "          pointer-events:none;\n"
                    "          transition:opacity 0.5s; }\n"
                    "  body.fs #hint { opacity:0; }\n"
                    "</style>\n"
                    "</head><body>\n"
                    '<img id="stream" src="/stream" />\n'
                    '<div id="hint">Click to go fullscreen &mdash; '
                    "drag this window to your projector first</div>\n"
                    "<script>\n"
                    "// Click to enter fullscreen\n"
                    "document.body.addEventListener('click', () => {\n"
                    "  if (!document.fullscreenElement) {\n"
                    "    document.documentElement.requestFullscreen()\n"
                    "      .catch(() => {});\n"
                    "  }\n"
                    "});\n"
                    "document.addEventListener('fullscreenchange', () => {\n"
                    "  document.body.classList.toggle('fs',\n"
                    "    !!document.fullscreenElement);\n"
                    "});\n"
                    "\n"
                    "// POST screen resolution to /config\n"
                    "function postConfig() {\n"
                    "  const s = window.screen;\n"
                    "  fetch('/config', {\n"
                    "    method: 'POST',\n"
                    "    headers: {'Content-Type': 'application/json'},\n"
                    "    body: JSON.stringify({\n"
                    "      width: s.width, height: s.height,\n"
                    "      monitor_name: 'browser'\n"
                    "    })\n"
                    "  }).catch(() => {});\n"
                    "}\n"
                    "postConfig();\n"
                    "setInterval(postConfig, 30000);\n"
                    "</script>\n"
                    "</body></html>"
                )
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "text/html")
                self_handler.send_header("Content-Length", str(len(html)))
                self_handler.end_headers()
                self_handler.wfile.write(html.encode())

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
