"""Fullscreen projector output window on a background thread.

Uses GLFW + raw PyOpenGL (OpenGL 3.3 Core) so the render loop is entirely
self-contained and never touches the host application's GL context.

Usage::

    proj = ProjectorOutput()
    proj.start(monitor_index=1)      # open fullscreen on second monitor
    proj.submit_frame(rgb_numpy)     # thread-safe, call from any thread
    proj.stop()                      # close window and join thread
"""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Shader sources ---------------------------------------------------------------

_VERT_SRC = """\
#version 330 core
out vec2 v_uv;
void main() {
    // Fullscreen triangle — 3 vertices, no VBO needed
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


class ProjectorOutput:
    """Manages a fullscreen GLFW window on a selectable monitor.

    All OpenGL calls happen on the background render thread.  The only
    cross-thread interaction is ``submit_frame()`` which copies an RGBA
    buffer under a lock.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._monitor_index: int = 1
        self._lock = threading.Lock()
        self._frame_rgba: np.ndarray | None = None  # RGBA uint8, vertically flipped
        self._frame_dirty = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- Public API ------------------------------------------------------------

    def start(self, monitor_index: int = 1) -> None:
        """Spawn the render thread and open a fullscreen window."""
        if self.is_running:
            logger.warning("ProjectorOutput already running")
            return
        self._monitor_index = monitor_index
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._render_loop, name="projector-output", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the render thread to exit and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def submit_frame(self, rgb: np.ndarray) -> None:
        """Submit an RGB uint8 (H, W, 3) numpy array for display.

        Thread-safe — may be called from any thread.  The frame is converted
        to RGBA and vertically flipped (OpenGL convention) before storage.
        """
        if not self.is_running:
            return
        h, w = rgb.shape[:2]
        # Convert RGB -> RGBA and flip vertically for OpenGL
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255
        rgba = np.ascontiguousarray(rgba[::-1])
        with self._lock:
            self._frame_rgba = rgba
            self._frame_dirty = True

    # -- Render thread ---------------------------------------------------------

    def _render_loop(self) -> None:
        """GLFW init -> window -> render loop -> cleanup.  Runs on its own thread."""
        import glfw
        from OpenGL import GL

        if not glfw.init():
            logger.error("ProjectorOutput: glfw.init() failed")
            return

        try:
            self._run_gl(glfw, GL)
        except Exception:
            logger.exception("ProjectorOutput render loop crashed")
        finally:
            glfw.terminate()

    def _run_gl(self, glfw, GL) -> None:  # noqa: N803
        """Core GL loop — separated for clean finally/terminate."""
        monitors = glfw.get_monitors()
        idx = min(self._monitor_index, len(monitors) - 1)
        monitor = monitors[idx]
        mode = glfw.get_video_mode(monitor)
        w, h = mode.size.width, mode.size.height

        # Window hints: OpenGL 3.3 core, no decoration, no auto-iconify
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)

        window = glfw.create_window(w, h, "ProMapAnything Projector", monitor, None)
        if not window:
            logger.error("ProjectorOutput: failed to create GLFW window")
            return

        glfw.make_context_current(window)
        glfw.swap_interval(1)  # vsync

        logger.info(
            "ProjectorOutput: opened %dx%d on monitor %d (%s)",
            w, h, idx, glfw.get_monitor_name(monitor),
        )

        # Build GL resources
        program = self._create_shader_program(GL)
        vao = self._create_empty_vao(GL)
        texture = self._create_texture(GL, w, h)
        tex_w, tex_h = w, h

        try:
            while not self._stop_event.is_set() and not glfw.window_should_close(window):
                glfw.poll_events()

                # Upload new frame if available
                with self._lock:
                    new_frame = self._frame_rgba if self._frame_dirty else None
                    self._frame_dirty = False

                if new_frame is not None:
                    fh, fw = new_frame.shape[:2]
                    if fw != tex_w or fh != tex_h:
                        GL.glDeleteTextures(1, [texture])
                        texture = self._create_texture(GL, fw, fh)
                        tex_w, tex_h = fw, fh
                    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                    GL.glTexSubImage2D(
                        GL.GL_TEXTURE_2D, 0, 0, 0, fw, fh,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                        new_frame.ctypes.data_as(ctypes.c_void_p),
                    )

                # Draw
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                GL.glUseProgram(program)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                GL.glBindVertexArray(vao)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

                glfw.swap_buffers(window)
        finally:
            GL.glDeleteTextures(1, [texture])
            GL.glDeleteVertexArrays(1, [vao])
            GL.glDeleteProgram(program)
            glfw.destroy_window(window)
            logger.info("ProjectorOutput: window closed")

    # -- GL helpers (called on render thread only) -----------------------------

    @staticmethod
    def _create_shader_program(GL) -> int:  # noqa: N803
        """Compile and link the fullscreen passthrough shader."""
        vert = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vert, _VERT_SRC)
        GL.glCompileShader(vert)
        if not GL.glGetShaderiv(vert, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(vert).decode()
            raise RuntimeError(f"Vertex shader compile error:\n{log}")

        frag = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(frag, _FRAG_SRC)
        GL.glCompileShader(frag)
        if not GL.glGetShaderiv(frag, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(frag).decode()
            raise RuntimeError(f"Fragment shader compile error:\n{log}")

        program = GL.glCreateProgram()
        GL.glAttachShader(program, vert)
        GL.glAttachShader(program, frag)
        GL.glLinkProgram(program)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(program).decode()
            raise RuntimeError(f"Shader link error:\n{log}")

        GL.glDeleteShader(vert)
        GL.glDeleteShader(frag)

        # Set texture uniform
        GL.glUseProgram(program)
        loc = GL.glGetUniformLocation(program, "tex")
        GL.glUniform1i(loc, 0)

        return program

    @staticmethod
    def _create_empty_vao(GL) -> int:  # noqa: N803
        """Create an empty VAO for attributeless fullscreen triangle rendering."""
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        GL.glBindVertexArray(0)
        return vao

    @staticmethod
    def _create_texture(GL, width: int, height: int) -> int:  # noqa: N803
        """Create an RGBA texture sized to the given dimensions."""
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

    def __del__(self) -> None:
        if self.is_running:
            self.stop()
