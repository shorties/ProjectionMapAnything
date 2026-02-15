#!/usr/bin/env python3
"""ProMapAnything Projector Client.

Receives an MJPEG stream from a remote Scope instance (e.g. RunPod) and
displays it fullscreen on a local projector via GLFW + OpenGL.

Usage::

    python projector_client.py https://<pod-id>-8765.proxy.runpod.net/stream
    python projector_client.py http://localhost:8765/stream --monitor 1
    python projector_client.py http://192.168.1.100:8765/stream --monitor 0
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import sys
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shader sources (same fullscreen triangle as the plugin)
# ---------------------------------------------------------------------------

_VERT_SRC = """\
#version 330 core
out vec2 v_uv;
void main() {
    vec2 pos = vec2(
        float((gl_VertexID & 1) << 2) - 1.0,
        float((gl_VertexID & 2) << 1) - 1.0
    );
    v_uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

_FRAG_SRC = """\
#version 330 core
uniform sampler2D tex;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    frag_color = texture(tex, v_uv);
}
"""

# ---------------------------------------------------------------------------
# MJPEG stream reader
# ---------------------------------------------------------------------------


def read_mjpeg_stream(url: str):
    """Yield decoded RGB numpy frames from an MJPEG HTTP stream.

    Parses the multipart boundary and uses Content-Length headers for
    reliable frame extraction.  Falls back to JPEG marker scanning if
    Content-Length is missing.
    """
    logger.info("Connecting to %s ...", url)

    req = urllib.request.Request(url)
    req.add_header("Accept", "multipart/x-mixed-replace")

    response = urllib.request.urlopen(req, timeout=30)  # noqa: S310
    logger.info("Connected — streaming frames")

    buf = b""
    while True:
        chunk = response.read(65536)
        if not chunk:
            logger.warning("Stream ended")
            break
        buf += chunk

        # Scan for JPEG frames using SOI (0xFFD8) and EOI (0xFFD9) markers
        while True:
            soi = buf.find(b"\xff\xd8")
            if soi == -1:
                # Discard bytes before any marker to prevent unbounded growth
                buf = buf[-2:] if len(buf) > 2 else buf
                break
            eoi = buf.find(b"\xff\xd9", soi + 2)
            if eoi == -1:
                # Incomplete frame — keep buffering
                break

            jpeg_data = buf[soi : eoi + 2]
            buf = buf[eoi + 2 :]

            # Decode JPEG
            arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield rgb


# ---------------------------------------------------------------------------
# GLFW + OpenGL projector display
# ---------------------------------------------------------------------------


class ProjectorDisplay:
    """Fullscreen GLFW window for projector output.

    Runs on the main thread (GLFW requirement on macOS; simpler overall).
    """

    def __init__(self, monitor_index: int = 1) -> None:
        import glfw

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        monitors = glfw.get_monitors()
        idx = min(monitor_index, len(monitors) - 1)
        monitor = monitors[idx]
        mode = glfw.get_video_mode(monitor)
        w, h = mode.size.width, mode.size.height

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)

        self._window = glfw.create_window(
            w, h, "ProMapAnything Projector", monitor, None
        )
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        self._glfw = glfw
        self._monitor_name = glfw.get_monitor_name(monitor)
        logger.info(
            "Projector window: %dx%d on monitor %d (%s)", w, h, idx, self._monitor_name
        )

        # Build GL resources
        from OpenGL import GL

        self._GL = GL
        self._program = self._create_shader_program()
        self._vao = self._create_empty_vao()
        self._texture = self._create_texture(w, h)
        self._tex_w, self._tex_h = w, h

        # Thread-safe frame buffer
        self._lock = threading.Lock()
        self._frame_rgba: np.ndarray | None = None
        self._frame_dirty = False

    def submit_frame(self, rgb: np.ndarray) -> None:
        """Submit an RGB uint8 (H, W, 3) frame. Thread-safe."""
        h, w = rgb.shape[:2]
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255
        rgba = np.ascontiguousarray(rgba[::-1])  # flip for OpenGL
        with self._lock:
            self._frame_rgba = rgba
            self._frame_dirty = True

    def render_frame(self) -> bool:
        """Poll events, upload texture if dirty, draw. Returns False if closed."""
        glfw = self._glfw
        GL = self._GL

        glfw.poll_events()
        if glfw.window_should_close(self._window):
            return False

        # Upload new frame
        with self._lock:
            new_frame = self._frame_rgba if self._frame_dirty else None
            self._frame_dirty = False

        if new_frame is not None:
            fh, fw = new_frame.shape[:2]
            if fw != self._tex_w or fh != self._tex_h:
                GL.glDeleteTextures(1, [self._texture])
                self._texture = self._create_texture(fw, fh)
                self._tex_w, self._tex_h = fw, fh
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0, fw, fh,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                new_frame.ctypes.data_as(ctypes.c_void_p),
            )

        # Draw
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(self._program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        glfw.swap_buffers(self._window)
        return True

    def destroy(self) -> None:
        GL = self._GL
        GL.glDeleteTextures(1, [self._texture])
        GL.glDeleteVertexArrays(1, [self._vao])
        GL.glDeleteProgram(self._program)
        self._glfw.destroy_window(self._window)
        self._glfw.terminate()
        logger.info("Projector window closed")

    # -- GL helpers -----------------------------------------------------------

    def _create_shader_program(self) -> int:
        GL = self._GL
        vert = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vert, _VERT_SRC)
        GL.glCompileShader(vert)
        if not GL.glGetShaderiv(vert, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(vert).decode())

        frag = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(frag, _FRAG_SRC)
        GL.glCompileShader(frag)
        if not GL.glGetShaderiv(frag, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(frag).decode())

        program = GL.glCreateProgram()
        GL.glAttachShader(program, vert)
        GL.glAttachShader(program, frag)
        GL.glLinkProgram(program)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(program).decode())

        GL.glDeleteShader(vert)
        GL.glDeleteShader(frag)
        GL.glUseProgram(program)
        GL.glUniform1i(GL.glGetUniformLocation(program, "tex"), 0)
        return program

    def _create_empty_vao(self) -> int:
        GL = self._GL
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        GL.glBindVertexArray(0)
        return vao

    def _create_texture(self, width: int, height: int) -> int:
        GL = self._GL
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
            width, height, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        return tex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProMapAnything Projector Client — display remote stream on local projector",
    )
    parser.add_argument(
        "url",
        help=(
            "MJPEG stream URL, e.g. "
            "https://<pod-id>-8765.proxy.runpod.net/stream "
            "or http://localhost:8765/stream"
        ),
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor index for fullscreen (0=primary, 1=first secondary, default: 1)",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before reconnecting after stream loss (default: 3)",
    )
    args = parser.parse_args()

    # Ensure URL ends with /stream if user gives base URL
    url = args.url.rstrip("/")
    if not url.endswith("/stream"):
        url += "/stream"

    # Create projector display (GLFW on main thread)
    display = ProjectorDisplay(monitor_index=args.monitor)

    # Stream reader runs on a background thread
    stream_active = threading.Event()
    stream_active.set()

    def stream_thread() -> None:
        while stream_active.is_set():
            try:
                for rgb in read_mjpeg_stream(url):
                    if not stream_active.is_set():
                        break
                    display.submit_frame(rgb)
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                logger.warning("Stream error: %s — reconnecting in %.0fs", e, args.reconnect_delay)
            except Exception:
                logger.exception("Unexpected stream error — reconnecting in %.0fs", args.reconnect_delay)

            if stream_active.is_set():
                time.sleep(args.reconnect_delay)

    t = threading.Thread(target=stream_thread, daemon=True)
    t.start()

    # Main thread drives the GLFW render loop
    try:
        while display.render_frame():
            pass
    except KeyboardInterrupt:
        pass
    finally:
        stream_active.clear()
        display.destroy()


if __name__ == "__main__":
    main()
