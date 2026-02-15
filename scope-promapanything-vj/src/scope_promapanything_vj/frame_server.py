"""MJPEG streaming server for remote projector output.

Runs a lightweight HTTP server on a background thread.  The pipeline submits
frames via ``submit_frame(rgb_np)``; connected clients receive them as an
MJPEG stream.

Endpoints
---------
``/``        Status page with an embedded ``<img>`` viewer.
``/stream``  MJPEG multipart stream (``multipart/x-mixed-replace``).
``/frame``   Single JPEG snapshot of the latest frame.

Works through RunPod's port proxy — expose the port and connect from anywhere.
"""

from __future__ import annotations

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_BOUNDARY = b"promapframe"


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

    @property
    def is_running(self) -> bool:
        return self._running

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
                else:
                    self_handler._handle_status()

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
                """Status page with embedded MJPEG viewer."""
                html = (
                    "<!DOCTYPE html>\n"
                    "<html><head><title>ProMapAnything Projector Stream</title>\n"
                    "<style>\n"
                    "  body { margin:0; background:#000; display:flex;\n"
                    "         justify-content:center; align-items:center;\n"
                    "         height:100vh; font-family:sans-serif; }\n"
                    "  img  { max-width:100%; max-height:100vh;\n"
                    "         object-fit:contain; }\n"
                    "  .info { position:fixed; top:10px; left:10px;\n"
                    "          color:#555; font-size:12px; }\n"
                    "</style></head><body>\n"
                    '<div class="info">ProMapAnything Projector Stream '
                    "&mdash; fullscreen: F11</div>\n"
                    '<img src="/stream" />\n'
                    "</body></html>"
                )
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "text/html")
                self_handler.send_header("Content-Length", str(len(html)))
                self_handler.end_headers()
                self_handler.wfile.write(html.encode())

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
            "(endpoints: /stream, /frame, /)",
            self._port,
        )

    def submit_frame(self, rgb: np.ndarray) -> None:
        """Submit an RGB uint8 (H, W, 3) frame for streaming.

        Thread-safe — may be called from any thread.
        """
        if not self._running:
            return
        # Encode as JPEG
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
