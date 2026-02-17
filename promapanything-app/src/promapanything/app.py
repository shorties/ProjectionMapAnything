"""ProMapAnything — main application.

Two GLFW windows:
- Main window (primary monitor): ImGui controls + 3D preview
- Projector window (secondary monitor): fullscreen calibration/effects output

Three Spout outputs: depth map, color map, projector output.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import glfw
import moderngl
import numpy as np
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer as ImGuiGlfwRenderer

from .calibration import CalibrationRunner, save_calibration
from .camera import CameraCapture, enumerate_cameras
from .depth import DepthEstimator
from .ndi_camera import NDICamera, discover_ndi_sources, ndi_available
from .procam_calibration import ProCamCalibrator, load_procam, save_procam
from .projector_depth import ProjectorDepthBuilder
from .renderer.scene import SceneRenderer
from .reproject import DepthReprojector
from .spout_output import SpoutOutputManager, SpoutReceiver
from .state import AppMode, AppState, CalibrationPhase, CameraSource, MainViewMode, ProjectorSource, ViewMode

logger = logging.getLogger(__name__)

SHADER_DIR = Path(__file__).parent / "renderer" / "shaders"


def _read_shader(name: str) -> str:
    return (SHADER_DIR / name).read_text()


class ProMapAnythingApp:
    """Main application class."""

    def __init__(self):
        self.state = AppState()
        self.camera = CameraCapture(self.state)
        self.depth_estimator: DepthEstimator | None = None

        # GLFW windows
        self._main_window = None
        self._proj_window = None

        # ModernGL contexts
        self._main_ctx: moderngl.Context | None = None
        self._proj_ctx: moderngl.Context | None = None

        # OpenGL resources (main window)
        self._depth_texture = None
        self._color_texture = None
        self._effects_program = None
        self._effects_vao = None
        self._fullscreen_vao = None

        # OpenGL resources (projector window)
        self._proj_passthrough_prog = None
        self._proj_fullscreen_vao = None
        self._proj_texture = None

        # Effects FBO for Spout readback
        self._effects_fbo = None
        self._effects_output_tex = None
        self._effects_fbo_size = (0, 0)

        # Calibration (Gray codes)
        self._calib_runner: CalibrationRunner | None = None
        self._calib_pattern_idx: int = 0
        self._calib_last_pattern_time: float = 0.0
        self._calib_waiting: bool = False
        self._calib_pattern_texture = None
        self._calib_frame_count: int = 0         # frames captured for current pattern
        self._calib_interframe_settle: bool = False  # True = short inter-frame wait

        # ProCam calibration (multi-step)
        self._procam_calibrator: ProCamCalibrator | None = None
        self._checkerboard_image: np.ndarray | None = None  # current board to project
        self._cam_reproj_error: float | None = None
        self._proj_reproj_error: float | None = None
        self._stereo_reproj_error: float | None = None

        # Depth reprojection
        self._depth_reprojector: DepthReprojector | None = None

        # Calibration preview (camera feed with corner overlay)
        self._calib_preview_texture = None
        self._calib_corners_found: bool = False
        self._calib_corners_count: int = 0

        # Scene renderer
        self._scene: SceneRenderer | None = None

        # Spout output
        self._spout = SpoutOutputManager()
        self._spout_initialized = False

        # Spout input (receive from Scope)
        self._spout_receiver: SpoutReceiver | None = None
        self._spout_receive_texture = None
        self._spout_receive_frame: np.ndarray | None = None

        # ImGui
        self._imgui_renderer: ImGuiGlfwRenderer | None = None

        # NDI camera
        self.ndi_camera = NDICamera(self.state)
        self._ndi_source_list: list[str] = []
        self._ndi_discovering = False

        # Camera enumeration cache (not auto-enumerated — user clicks Refresh)
        self._camera_list: list[tuple[int, str]] = [
            (0, "[0] Camera (active)"),
        ]
        self._camera_refreshing = False

        # Latency measurement for calibration (pattern→camera delay)
        self._latency_measured = False
        self._latency_ms: float = 0.0
        self._latency_measuring = False
        self._latency_white_time: float = 0.0
        self._latency_frame_before: np.ndarray | None = None

        # Gray code decode stats (persisted after calibration completes)
        self._decode_valid_cam_pixels: int = 0
        self._decode_total_cam_pixels: int = 0
        self._decode_cam_coverage_pct: float = 0.0
        self._decode_valid_proj_pixels: int = 0
        self._decode_total_proj_pixels: int = 0
        self._decode_proj_coverage_pct: float = 0.0

        # Always-on camera preview texture (for UI sidebar + viewport)
        self._camera_preview_texture = None

        # Depth visualization texture (colormapped for display)
        self._depth_vis_texture = None

        # Orbit camera drag state
        self._dragging_orbit = False
        self._panning = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Timing
        self._last_frame_time = 0.0
        self._start_time = 0.0
        self._frame_count = 0

        # Background depth inference
        self._depth_thread: threading.Thread | None = None
        self._depth_running = False
        self._depth_lock = threading.Lock()
        self._depth_input_frame: np.ndarray | None = None  # latest camera frame for depth
        self._depth_new_input = threading.Event()

    # ── Initialization ───────────────────────────────────────────────────

    def init(self) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Get monitors
        monitors = glfw.get_monitors()
        logger.info("Found %d monitors", len(monitors))
        for i, mon in enumerate(monitors):
            mode = glfw.get_video_mode(mon)
            name = glfw.get_monitor_name(mon)
            logger.info("  [%d] %s: %dx%d", i, name, mode.size.width, mode.size.height)

        # ── Main window (windowed, primary monitor) ──────────────────
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        self._main_window = glfw.create_window(
            1280, 800, "ProMapAnything", None, None
        )
        if not self._main_window:
            raise RuntimeError("Failed to create main window")

        glfw.make_context_current(self._main_window)
        self._main_ctx = moderngl.create_context()

        # ── Create shared OpenGL resources on main context ───────────
        self._create_gl_resources()

        # ── Scene renderer ────────────────────────────────────────────
        self._scene = SceneRenderer(self._main_ctx)

        # ── ImGui setup (using imgui-bundle's GLFW backend) ──────────
        imgui.create_context()
        self._imgui_renderer = ImGuiGlfwRenderer(self._main_window)

        # ── Input callbacks (orbit camera — installed after ImGui) ────
        # ImGui's GLFW backend installs its own callbacks; we chain ours
        # by storing the previous callbacks and calling them.
        self._prev_scroll_cb = glfw.set_scroll_callback(
            self._main_window, self._scroll_callback
        )
        self._prev_mouse_btn_cb = glfw.set_mouse_button_callback(
            self._main_window, self._mouse_button_callback
        )
        self._prev_cursor_cb = glfw.set_cursor_pos_callback(
            self._main_window, self._cursor_pos_callback
        )
        self._prev_key_cb = glfw.set_key_callback(
            self._main_window, self._key_callback
        )

        # ── Depth estimator (loads model) ────────────────────────────
        logger.info("Loading depth estimation model...")
        try:
            self.depth_estimator = DepthEstimator()
            logger.info("Depth estimator ready")
        except Exception as e:
            logger.warning("Failed to load depth model: %s", e)

        # ── Camera ───────────────────────────────────────────────────
        try:
            self.camera.start(self.state.camera_device_idx)
            logger.info("Camera started (device %d)", self.state.camera_device_idx)
        except Exception as e:
            logger.warning("Failed to start camera: %s", e)

        self._start_time = time.monotonic()

        # Start background depth thread
        self._start_depth_thread()

    def _create_gl_resources(self) -> None:
        ctx = self._main_ctx

        # Passthrough fullscreen triangle
        self._passthrough_prog = ctx.program(
            vertex_shader=_read_shader("fullscreen.vert"),
            fragment_shader=_read_shader("passthrough.frag"),
        )
        self._fullscreen_vao = ctx.vertex_array(self._passthrough_prog, [])

        # Effects program
        self._effects_program = ctx.program(
            vertex_shader=_read_shader("fullscreen.vert"),
            fragment_shader=_read_shader("effects.frag"),
        )
        self._effects_vao = ctx.vertex_array(self._effects_program, [])

        # Textures (placeholder 1x1, will be resized on first frame)
        self._depth_texture = ctx.texture((1, 1), 1, dtype="f4")
        self._color_texture = ctx.texture((1, 1), 3, dtype="f1")

    def _ensure_texture_size(self, tex, w, h, components, dtype):
        """Recreate texture if size changed."""
        if tex.width != w or tex.height != h:
            tex.release()
            return self._main_ctx.texture((w, h), components, dtype=dtype)
        return tex

    def _ensure_effects_fbo(self, w: int, h: int) -> None:
        """Ensure effects FBO matches the target size."""
        if self._effects_fbo_size == (w, h) and self._effects_fbo is not None:
            return
        if self._effects_fbo is not None:
            self._effects_fbo.release()
            self._effects_output_tex.release()
        self._effects_output_tex = self._main_ctx.texture((w, h), 4, dtype="f1")
        self._effects_fbo = self._main_ctx.framebuffer(
            color_attachments=[self._effects_output_tex]
        )
        self._effects_fbo_size = (w, h)

    # ── Projector window management ──────────────────────────────────────

    def open_projector_window(self, monitor_idx: int) -> None:
        """Open a fullscreen window on the specified monitor."""
        if self._proj_window is not None:
            self.close_projector_window()

        monitors = glfw.get_monitors()
        if monitor_idx >= len(monitors):
            logger.warning("Monitor %d not found", monitor_idx)
            return

        monitor = monitors[monitor_idx]
        mode = glfw.get_video_mode(monitor)

        glfw.window_hint(glfw.AUTO_ICONIFY, False)
        glfw.window_hint(glfw.DECORATED, False)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        # Separate window with its own context (no sharing)
        self._proj_window = glfw.create_window(
            mode.size.width, mode.size.height,
            "ProMapAnything Projector",
            monitor,  # Fullscreen on this monitor
            None,
        )

        if self._proj_window:
            self.state.calibration.projector_width = mode.size.width
            self.state.calibration.projector_height = mode.size.height

            # Create projector's own moderngl context + resources
            glfw.make_context_current(self._proj_window)
            self._proj_ctx = moderngl.create_context()
            self._proj_passthrough_prog = self._proj_ctx.program(
                vertex_shader=_read_shader("fullscreen.vert"),
                fragment_shader=_read_shader("passthrough.frag"),
            )
            self._proj_fullscreen_vao = self._proj_ctx.vertex_array(
                self._proj_passthrough_prog, []
            )
            # Switch back to main
            glfw.make_context_current(self._main_window)

            logger.info(
                "Projector window opened on monitor %d (%dx%d)",
                monitor_idx, mode.size.width, mode.size.height,
            )

    def close_projector_window(self) -> None:
        if self._proj_window is not None:
            if self._proj_ctx is not None:
                glfw.make_context_current(self._proj_window)
                if self._proj_texture is not None:
                    self._proj_texture.release()
                    self._proj_texture = None
                self._proj_fullscreen_vao = None
                self._proj_passthrough_prog = None
                self._proj_ctx.release()
                self._proj_ctx = None
                glfw.make_context_current(self._main_window)
            glfw.destroy_window(self._proj_window)
            self._proj_window = None

    # ── Spout ─────────────────────────────────────────────────────────────

    def _init_spout(self) -> None:
        if not self._spout_initialized and self.state.spout_enabled:
            self._spout.init(
                depth_name=self.state.spout_depth_name,
                color_name=self.state.spout_color_name,
                projector_name=self.state.spout_projector_name,
            )
            self._spout_initialized = True
            logger.info("Spout output initialized")

    def _init_spout_receiver(self) -> None:
        """Create or recreate the Spout receiver when settings change."""
        if self._spout_receiver is not None:
            self._spout_receiver.release()
            self._spout_receiver = None
        self._spout_receiver = SpoutReceiver(self.state.spout_receive_name)
        self.state.spout_receive_enabled = True
        logger.info("Spout receiver started (source=%s)", self.state.spout_receive_name or "auto")

    def _receive_spout_frame(self) -> None:
        """Pull the latest frame from the Spout receiver."""
        if self._spout_receiver is None:
            return
        frame = self._spout_receiver.receive()
        if frame is not None:
            self._spout_receive_frame = frame

    def _send_spout_frames(self) -> None:
        if not self.state.spout_enabled or not self._spout_initialized:
            return

        depth = self.state.depth_frame
        if depth is not None:
            self._spout.send_depth(depth)

        frame = self.state.camera_frame
        if frame is not None:
            self._spout.send_color(frame)

        # Projector output: read back from effects FBO
        if (
            self._effects_output_tex is not None
            and self._effects_fbo_size[0] > 0
        ):
            w, h = self._effects_fbo_size
            raw = self._effects_output_tex.read()
            rgba = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
            rgb = rgba[:, :, :3]
            self._spout.send_projector(rgb)

    # ── Calibration ──────────────────────────────────────────────────────

    def cancel_calibration(self) -> None:
        """Cancel any active calibration phase."""
        self.state.mode = AppMode.IDLE
        self.state.calib_phase = CalibrationPhase.IDLE
        self._calib_runner = None
        self._procam_calibrator = None
        self._checkerboard_image = None
        self.state.calibration_progress = 0.0
        logger.info("Calibration cancelled")

    # ── Full auto calibration ──────────────────────────────────────

    def start_calibration(self) -> None:
        """Start the full automatic calibration pipeline.

        Flow: measure latency → [project checkerboards] → Gray codes → compute everything.
        
        If skip_checkerboard is True, goes straight to Gray codes (2D correspondence only).
        """
        if self._proj_window is None:
            self.open_projector_window(self.state.projector_monitor_idx)

        # Measure camera latency to auto-set settle time
        if not self._latency_measured:
            self._measure_latency()

        # Use measured latency + safety margin for settle time
        auto_settle = self._latency_ms + 100.0  # add 100ms margin
        if auto_settle > self.state.calibration.settle_ms:
            logger.info("Auto-adjusting settle time: %.0fms → %.0fms (latency=%.0fms)",
                       self.state.calibration.settle_ms, auto_settle, self._latency_ms)
            self.state.calibration.settle_ms = auto_settle

        # Check if user wants to skip checkerboard phase
        if self.state.calibration.skip_checkerboard:
            logger.info("Skipping checkerboard phase (user preference)")
            self._start_gray_code_phase()
            self.state.mode = AppMode.CALIBRATING
            return

        pw = self.state.calibration.projector_width
        ph = self.state.calibration.projector_height
        cal = self.state.calibration
        self._procam_calibrator = ProCamCalibrator(
            proj_w=pw, proj_h=ph,
            board_cols=cal.board_cols,
            board_rows=cal.board_rows,
            square_px=cal.square_px,
            white_level=cal.board_white,
            black_level=cal.board_black,
        )
        self._checkerboard_image = self._procam_calibrator.get_current_board_image()
        self._cb_phase = 0  # 0=display, 1=settling, 2=detect+capture
        self._cb_settle_time = 0.0
        self._cb_detect_attempts = 0
        self._cb_max_detect_attempts = 30  # frames to try before skipping

        self.state.mode = AppMode.CALIBRATING
        self.state.calib_phase = CalibrationPhase.CAMERA_INTRINSICS
        self.state.calibration_progress = 0.0
        logger.info("Auto-calibration started (%d board positions, settle=%.0fms)",
                     self._procam_calibrator.total_positions,
                     self.state.calibration.settle_ms)

    def _update_checkerboard_phase(self) -> None:
        """Auto state machine for projected checkerboard capture.

        States:
          0 = "display" — board pattern was just set, start settle timer
          1 = "settling" — wait for projector + camera to settle
          2 = "detect"  — try to detect corners, capture if found, retry or skip
        """
        if self._procam_calibrator is None:
            return

        calibrator = self._procam_calibrator
        now = time.monotonic()
        settle_s = max(self.state.calibration.settle_ms, 500.0) / 1000.0  # at least 500ms for checkerboard

        if self._cb_phase == 0:
            self._cb_phase = 1
            self._cb_settle_time = now
            return

        if self._cb_phase == 1:
            if now - self._cb_settle_time >= settle_s:
                self._cb_phase = 2
                self._cb_detect_attempts = 0
            return

        if self._cb_phase == 2:
            frame = self.state.camera_frame
            if frame is None:
                return

            self._cb_detect_attempts += 1
            success, corners = calibrator.capture_board(frame.copy())

            if success:
                logger.info("Board %d captured at position %d",
                            calibrator.capture_count,
                            calibrator.current_position_idx)
                self._advance_checkerboard()
            elif self._cb_detect_attempts >= self._cb_max_detect_attempts:
                logger.warning("Skipping position %d (detection failed after %d attempts)",
                               calibrator.current_position_idx, self._cb_detect_attempts)
                calibrator._current_position_idx += 1
                self._advance_checkerboard()

    def _advance_checkerboard(self) -> None:
        """Move to next board position or finish the checkerboard phase."""
        calibrator = self._procam_calibrator
        total = calibrator.total_positions
        captures = calibrator.capture_count
        pos_idx = calibrator.current_position_idx

        self.state.calibration_progress = pos_idx / max(total, 1)

        if pos_idx >= total:
            # All positions done — compute intrinsics and move to Gray codes
            if captures >= 3:
                self._finish_checkerboard_phase()
            else:
                logger.error("Only %d checkerboard captures — need at least 3. Skipping intrinsics.", captures)
                self._start_gray_code_phase()
        else:
            # Next position
            self._checkerboard_image = calibrator.get_current_board_image()
            self._cb_phase = 0  # back to display

    def _finish_checkerboard_phase(self) -> None:
        """Compute camera + projector intrinsics, then auto-start Gray codes."""
        if self._procam_calibrator is None:
            return

        # Camera intrinsics
        try:
            K_cam, dist_cam, cam_err = self._procam_calibrator.compute_camera_intrinsics()
            self._cam_reproj_error = cam_err
            self.state.procam.K_cam = K_cam
            self.state.procam.dist_cam = dist_cam
        except Exception as e:
            logger.error("Camera intrinsics failed: %s — continuing with Gray codes only", e)
            self._start_gray_code_phase()
            return

        # Projector intrinsics (requires camera intrinsics)
        try:
            K_proj, dist_proj, proj_err = self._procam_calibrator.compute_projector_intrinsics(K_cam, dist_cam)
            self._proj_reproj_error = proj_err
            self.state.procam.K_proj = K_proj
            self.state.procam.dist_proj = dist_proj
            logger.info("Intrinsics computed (cam_err=%.3f, proj_err=%.3f)",
                         cam_err, proj_err)
        except Exception as e:
            logger.error("Projector intrinsics failed: %s — will estimate from stereo", e)
            # Provide a reasonable default K_proj so stereo calibration can proceed
            pw, ph = self.state.calibration.projector_width, self.state.calibration.projector_height
            f_init = float(max(pw, ph))
            self.state.procam.K_proj = np.array([
                [f_init, 0, pw / 2.0],
                [0, f_init, ph / 2.0],
                [0, 0, 1],
            ], dtype=np.float64)
            self.state.procam.dist_proj = np.zeros((1, 5), dtype=np.float64)

        self._start_gray_code_phase()

    def _estimate_intrinsics_from_gray_code(self) -> None:
        """Estimate camera and projector intrinsics from Gray code correspondences.
        
        Fallback when checkerboard phase is skipped. Uses the correspondence map
        to estimate reasonable intrinsics for depth reprojection.
        """
        pw = self.state.calibration.projector_width
        ph = self.state.calibration.projector_height
        
        # Estimate projector intrinsics assuming centered principal point
        # and focal length based on typical projector throw ratio (~1.5:1)
        f_proj = float(max(pw, ph)) * 1.2  # rough estimate
        K_proj = np.array([
            [f_proj, 0, pw / 2.0],
            [0, f_proj, ph / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)
        
        self.state.procam.K_proj = K_proj
        self.state.procam.dist_proj = np.zeros((1, 5), dtype=np.float64)
        
        # Camera intrinsics: get from current frame size if available
        frame = self.state.camera_frame
        if frame is not None:
            ch, cw = frame.shape[:2]
            f_cam = float(max(cw, ch)) * 1.0  # rough estimate
            K_cam = np.array([
                [f_cam, 0, cw / 2.0],
                [0, f_cam, ch / 2.0],
                [0, 0, 1],
            ], dtype=np.float64)
            self.state.procam.K_cam = K_cam
            self.state.procam.dist_cam = np.zeros((1, 5), dtype=np.float64)
            logger.info("Estimated intrinsics from frame size: cam %dx%d, proj %dx%d", cw, ch, pw, ph)
        else:
            logger.warning("No camera frame available — intrinsics not estimated")

    # ── Phase 2: Gray codes ─────────────────────────────────────────

    def _start_gray_code_phase(self) -> None:
        """Start the Gray code structured light phase."""
        pw = self.state.calibration.projector_width
        ph = self.state.calibration.projector_height
        cal = self.state.calibration
        self._calib_runner = CalibrationRunner(
            pw, ph,
            settle_ms=cal.settle_ms,
            capture_frames=cal.capture_frames,
            decode_threshold=cal.decode_threshold,
            morph_cleanup=cal.morph_cleanup,
            morph_kernel_size=cal.morph_kernel_size,
            spatial_consistency=cal.spatial_consistency,
            consistency_max_diff=cal.consistency_max_diff,
            bit_threshold=cal.bit_threshold,
            fill_kernel_size=cal.fill_kernel_size,
        )
        self._calib_pattern_idx = 0
        self._calib_frame_count = 0
        self._calib_interframe_settle = False
        self._calib_gc_phase = 0  # 0=display, 1=settling, 2=capture
        self._calib_last_pattern_time = time.monotonic()
        self._checkerboard_image = None
        self.state.calib_phase = CalibrationPhase.GRAY_CODES
        self.state.calibration_progress = 0.0
        logger.info("Gray code phase started — white warm-up for camera exposure (%d patterns, %d frames/pattern)",
                     self._calib_runner.total_patterns, cal.capture_frames)

    def start_gray_codes_only(self) -> None:
        """Legacy: start Gray codes only (no intrinsics step)."""
        self._start_gray_code_phase()
        self.state.mode = AppMode.CALIBRATING

    def _update_calibration(self) -> None:
        """Calibration state machine for Gray code phase.

        States (tracked by _calib_gc_phase):
          0 = "display" — pattern was just set, render it this frame
          1 = "settling" — pattern is on screen, wait for settle time
          2 = "capture"  — settle done, grab camera frame and feed it

        Captures multiple frames per pattern (capture_frames setting).
        Uses a short inter-frame delay (~50ms) between frames of the same
        pattern, and the full settle_ms between different patterns.
        """
        if self._calib_runner is None:
            return

        runner = self._calib_runner
        target_frames = runner.capture_frames
        now = time.monotonic()

        phase = getattr(self, "_calib_gc_phase", 0)

        if phase == 0:
            self._calib_gc_phase = 1
            self._calib_last_pattern_time = now
            return

        if phase == 1:
            # Short wait for inter-frame (same pattern), full settle for new pattern
            if self._calib_interframe_settle:
                wait_s = 0.05  # 50ms between frames of same pattern
            else:
                wait_s = self.state.calibration.settle_ms / 1000.0
            if now - self._calib_last_pattern_time >= wait_s:
                self._calib_gc_phase = 2
            return

        if phase == 2:
            capture = self.state.camera_frame
            if capture is not None:
                capture = capture.copy()
                runner.feed_capture(self._calib_pattern_idx, capture)
                self._calib_frame_count += 1

                # Update progress including sub-frame progress
                self.state.calibration_progress = (
                    (self._calib_pattern_idx + self._calib_frame_count / target_frames)
                    / runner.total_patterns
                )

                if self._calib_frame_count >= target_frames:
                    # All frames for this pattern captured — advance
                    self._calib_frame_count = 0
                    self._calib_pattern_idx += 1
                    self._calib_interframe_settle = False

                    if self._calib_pattern_idx >= runner.total_patterns:
                        self._finish_gray_codes()
                        return
                    # New pattern: go to display phase with full settle
                    self._calib_gc_phase = 0
                else:
                    # More frames needed for same pattern: short inter-frame wait
                    self._calib_interframe_settle = True
                    self._calib_gc_phase = 1
                    self._calib_last_pattern_time = now

    def _get_current_calibration_pattern(self) -> np.ndarray | None:
        """Get the current pattern to display on the projector."""
        # During checkerboard phase, show the checkerboard
        if self.state.calib_phase == CalibrationPhase.CAMERA_INTRINSICS:
            return self._checkerboard_image

        # During Gray code phase, show the Gray code pattern
        if self._calib_runner is None:
            return None
        if self._calib_pattern_idx >= self._calib_runner.total_patterns:
            return None
        return self._calib_runner.get_pattern_image(self._calib_pattern_idx)

    def _finish_gray_codes(self) -> None:
        """Decode Gray codes, compute stereo, save calibration.

        Uses RoomAlive-style dense correspondences + depth for robust
        projector calibration when depth data is available.
        """
        runner = self._calib_runner
        self.state.calib_phase = CalibrationPhase.COMPUTING
        try:
            map_x, map_y = runner.decode()
            self.state.calib_map_x = map_x
            self.state.calib_map_y = map_y
            self.state.procam.map_x = map_x
            self.state.procam.map_y = map_y

            # Store decode stats for UI display
            self._decode_valid_cam_pixels = runner.valid_camera_pixels
            self._decode_total_cam_pixels = runner.total_camera_pixels
            self._decode_cam_coverage_pct = runner.camera_coverage_pct
            self._decode_valid_proj_pixels = runner.valid_projector_pixels
            self._decode_total_proj_pixels = runner.total_projector_pixels
            self._decode_proj_coverage_pct = runner.projector_coverage_pct
            self._decode_sparse_proj_pixels = getattr(runner, 'sparse_projector_pixels', 0)
            logger.info(
                "Gray code decode: %d/%d camera pixels (%.1f%%), %d/%d projector pixels (%.1f%%)",
                runner.valid_camera_pixels, runner.total_camera_pixels, runner.camera_coverage_pct,
                runner.valid_projector_pixels, runner.total_projector_pixels, runner.projector_coverage_pct,
            )

            procam = self.state.procam
            pw = self.state.calibration.projector_width
            ph = self.state.calibration.projector_height

            # Get the real projector validity mask (before inpainting filled holes)
            proj_valid_mask = runner.proj_valid_mask

            # Wait for camera to capture a clean scene frame
            # (projector just finished showing patterns)
            import time
            time.sleep(0.5)

            # PRIMARY OUTPUT: Warp camera depth + RGB to projector perspective
            # using the correspondence map. Works with any webcam, no stereo needed.
            self._build_warped_projector_depth(map_x, map_y, proj_valid_mask)

            # If we skipped checkerboard, estimate intrinsics now
            if self.state.calibration.skip_checkerboard and procam.K_cam is None:
                self._estimate_intrinsics_from_gray_code()

            # OPTIONAL: Stereo calibration (for live depth reprojection, 3D preview, etc.)
            if procam.K_cam is not None and self._procam_calibrator is not None:
                stereo_ok = False
                proj_intrinsics_calibrated = self._proj_reproj_error is not None
                if self._procam_calibrator.capture_count >= 3:
                    try:
                        R, T, stereo_err = self._procam_calibrator.compute_stereo(
                            procam.K_cam, procam.dist_cam,
                            procam.K_proj, procam.dist_proj,
                            fix_intrinsics=proj_intrinsics_calibrated,
                        )
                        procam.R = R
                        procam.T = T
                        self._stereo_reproj_error = stereo_err
                        stereo_ok = True
                        if not proj_intrinsics_calibrated and hasattr(self._procam_calibrator, '_refined_K_proj'):
                            procam.K_proj = self._procam_calibrator._refined_K_proj
                            logger.info("Projector intrinsics refined by stereo calibration")
                        logger.info("Stereo calibration succeeded (error=%.3f px)", stereo_err)
                    except Exception as e:
                        logger.warning("Stereo calibration failed: %s", e)

                if procam.R is not None:
                    save_path = Path(self.state.output_dir) / "calibration.json"
                    save_procam(
                        save_path,
                        procam.K_cam, procam.dist_cam,
                        procam.K_proj, procam.dist_proj,
                        procam.R, procam.T, map_x, map_y, pw, ph,
                    )
                    self.state.calibration_file = str(save_path)
                    logger.info("Full ProCam calibration saved to %s", save_path)

                    # Also try stereo triangulation depth (secondary — only if stereo is good)
                    if stereo_ok and stereo_err < 5.0:
                        self._build_metric_projector_depth(map_x, map_y, proj_valid_mask)

                    if self._scene is not None:
                        self._scene.build_frustum_from_intrinsics(
                            procam.K_proj, procam.R, procam.T, pw, ph,
                        )
                else:
                    self._save_2d_calibration(map_x, map_y, pw, ph)
            else:
                self._save_2d_calibration(map_x, map_y, pw, ph)

            # Build depth reprojector from correspondence maps (works for all calibration types)
            self._build_depth_reprojector()

        except Exception as e:
            logger.error("Calibration decode/stereo failed: %s", e)

        self.state.mode = AppMode.LIVE
        self.state.calib_phase = CalibrationPhase.DONE
        self.state.calibration_progress = 1.0
        self._calib_runner = None

        # Open preview if stereo depth was successfully built
        if self.state.static_depth_captured:
            if self.state.depth.auto_open_reprojection:
                self._save_and_open_reprojection_preview()

    def _save_2d_calibration(
        self, map_x: np.ndarray, map_y: np.ndarray, pw: int, ph: int,
    ) -> None:
        """Save 2D-only calibration (no intrinsics/extrinsics)."""
        save_path = Path(self.state.output_dir) / "calibration.json"
        save_calibration(map_x, map_y, save_path, pw, ph)
        self.state.calibration_file = str(save_path)

        if self._scene is not None:
            dw = self._depth_texture.width if self._depth_texture else 640
            focal = max(dw, 480) * 0.8
            self._scene.build_frustum(pw, ph, focal)

    def _build_depth_reprojector(self) -> None:
        """Create the DepthReprojector from Gray code correspondence maps."""
        map_x = self.state.calib_map_x
        map_y = self.state.calib_map_y
        if map_x is None or map_y is None:
            logger.warning("Cannot build depth reprojector: no correspondence maps")
            return

        proj_w = self.state.calibration.projector_width
        proj_h = self.state.calibration.projector_height

        logger.info(f"Building depth reprojector (cv2.remap):")
        logger.info(f"  Projector: {proj_w}x{proj_h}")
        logger.info(f"  Map shape: {map_x.shape}")

        try:
            self._depth_reprojector = DepthReprojector(
                map_x, map_y, proj_w, proj_h,
            )
            logger.info("Depth reprojector initialized successfully (%dx%d)",
                         proj_w, proj_h)
        except Exception as e:
            logger.error(f"Failed to build depth reprojector: {e}")
            self._depth_reprojector = None

    def _build_warped_projector_depth(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        proj_valid_mask: np.ndarray,
    ) -> None:
        """Build projector-perspective depth + RGB by warping camera images.

        Uses the Gray code correspondence map to remap camera depth and color
        to projector pixel space.  No stereo calibration or projector intrinsics
        needed — just the 2D correspondence from structured light.

        This is the PRIMARY depth map output for ControlNet.
        """
        import cv2

        logger.info("Building warped projector-perspective depth + RGB ...")

        color_frame = self.state.camera_frame   # (cam_h, cam_w, 3) uint8 BGR
        if color_frame is None:
            logger.error("Cannot build depth map — no camera frame")
            return

        # Always run a fresh depth estimate from the current camera frame
        # (the cached depth_frame may be stale — e.g. from a calibration pattern)
        if self.depth_estimator is not None:
            logger.info("  Running fresh depth estimate on current camera frame (%dx%d) ...",
                        color_frame.shape[1], color_frame.shape[0])
            depth_frame = self.depth_estimator.estimate(color_frame)
            self.state.depth_frame = depth_frame
        else:
            depth_frame = self.state.depth_frame
            if depth_frame is None:
                logger.error("Cannot build depth map — no depth estimator or cached frame")
                return

        # Ensure depth matches camera resolution (map coords are in camera pixel space)
        cam_h, cam_w = color_frame.shape[:2]
        if depth_frame.shape[:2] != (cam_h, cam_w):
            logger.info("  Resizing depth %s → %s to match camera",
                        depth_frame.shape[:2], (cam_h, cam_w))
            depth_frame = cv2.resize(depth_frame, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)

        save_dir = Path(self.state.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Store the raw camera-space depth for live adjustment via UI sliders
        self.state.raw_warped_depth = depth_frame.copy()

        # ── Step 1: Create depth IMAGE in camera space ────────────────
        # Apply output settings (percentile clip, gamma, contrast) to make
        # a proper uint8 depth image, THEN warp it like RGB.
        depth_img = self._apply_depth_output_settings(depth_frame)
        depth_uint8_cam = (depth_img * 255).clip(0, 255).astype(np.uint8)
        # Make it 3-channel so it warps identically to RGB
        depth_bgr_cam = cv2.cvtColor(depth_uint8_cam, cv2.COLOR_GRAY2BGR)

        # Save camera-space depth for reference
        cv2.imwrite(str(save_dir / "camera_depth.png"), depth_uint8_cam)

        # ── Step 2: Warp depth image to projector (same as RGB) ───────
        depth_proj_bgr = cv2.remap(depth_bgr_cam, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Only inpaint small internal holes — large uncovered regions stay black
        depth_holes = np.all(depth_proj_bgr == 0, axis=2).astype(np.uint8) * 255
        if np.any(depth_holes):
            depth_proj_bgr = self._inpaint_small_holes(depth_proj_bgr, depth_holes)

        # Extract single-channel result
        depth_proj_uint8 = depth_proj_bgr[:, :, 0]  # all channels are the same

        # Save ControlNet-ready grayscale depth
        controlnet_path = save_dir / "projector_depth_controlnet.png"
        cv2.imwrite(str(controlnet_path), depth_proj_uint8)
        logger.info("ControlNet depth saved: %s", controlnet_path)

        # Save turbo colormap visualization
        turbo = cv2.applyColorMap(depth_proj_uint8, cv2.COLORMAP_TURBO)
        turbo_path = save_dir / "projector_depth_turbo.png"
        cv2.imwrite(str(turbo_path), turbo)

        # Store as static depth map for the app
        depth_norm = depth_proj_uint8.astype(np.float32) / 255.0
        self.state.static_depth_map = depth_norm
        self.state.static_depth_captured = True
        np.save(str(save_dir / "static_depth_projector.npy"), depth_norm)

        # ── Step 3: Warp RGB to projector (unchanged) ─────────────────
        rgb_proj = cv2.remap(color_frame, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rgb_holes = np.all(rgb_proj == 0, axis=2).astype(np.uint8) * 255
        if np.any(rgb_holes):
            rgb_proj = self._inpaint_small_holes(rgb_proj, rgb_holes)
        rgb_path = save_dir / "projector_rgb_view.png"
        cv2.imwrite(str(rgb_path), rgb_proj)
        logger.info("Projector RGB view saved: %s", rgb_path)

        # Switch to depth view so user can see and adjust the result
        self.state.main_view = MainViewMode.DEPTH

        coverage = float(np.count_nonzero(depth_proj_uint8 > 0)) / max(depth_proj_uint8.size, 1)
        logger.info("Warped depth map complete: %.1f%% coverage", coverage * 100)

    @staticmethod
    def _inpaint_small_holes(
        image: np.ndarray, hole_mask: np.ndarray, max_pct: float = 0.005,
    ) -> np.ndarray:
        """Inpaint only small holes; leave large uncovered regions as-is.

        Prevents fold-over artifacts from extrapolating into regions the
        camera never covered.
        """
        import cv2

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            hole_mask, connectivity=8,
        )
        max_area = max(int(image.shape[0] * image.shape[1] * max_pct), 100)
        small_mask = np.zeros_like(hole_mask)
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] < max_area:
                small_mask[labels == label_id] = 255
        if np.any(small_mask):
            return cv2.inpaint(image, small_mask, 5, cv2.INPAINT_NS)
        return image

    def _apply_depth_output_settings(self, depth_raw: np.ndarray) -> np.ndarray:
        """Apply user-adjustable output settings to raw depth.

        Produces a ControlNet-optimized depth image with maximum usable
        contrast. Returns float32 array normalized to [0, 1].
        """
        import cv2

        d = self.state.depth
        valid = depth_raw > 0

        if not np.any(valid):
            return depth_raw.copy()

        valid_vals = depth_raw[valid]

        # Percentile clipping — removes outliers so the main depth range
        # fills the full 0-255 output range
        lo_pct = max(0.0, min(d.output_clip_lo, 49.0))
        hi_pct = max(51.0, min(d.output_clip_hi, 100.0))
        d_lo = np.percentile(valid_vals, lo_pct)
        d_hi = np.percentile(valid_vals, hi_pct)

        if d_hi - d_lo < 1e-6:
            return depth_raw.copy()

        # Normalize to [0, 1] using percentile range
        depth_norm = (depth_raw - d_lo) / (d_hi - d_lo)
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        depth_norm[~valid] = 0

        # CLAHE histogram equalization — spreads depth values to use the
        # full range, giving ControlNet more detail to work with
        if d.output_equalize:
            depth_u8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            depth_u8 = clahe.apply(depth_u8)
            depth_norm = depth_u8.astype(np.float32) / 255.0
            depth_norm[~valid] = 0

        # Gamma correction
        if d.output_gamma != 1.0:
            gamma = max(0.2, d.output_gamma)
            mask = depth_norm > 0
            depth_norm[mask] = np.power(depth_norm[mask], 1.0 / gamma)

        # Brightness / contrast
        if d.output_brightness != 0.0 or d.output_contrast != 1.0:
            mask = depth_norm > 0
            depth_norm[mask] = d.output_contrast * depth_norm[mask] + d.output_brightness
            depth_norm = np.clip(depth_norm, 0.0, 1.0)
            depth_norm[~valid] = 0

        logger.info("  Depth output: clip=[%.1f%%, %.1f%%] eq=%s gamma=%.2f bright=%.2f contrast=%.2f",
                     lo_pct, hi_pct, d.output_equalize, d.output_gamma, d.output_brightness, d.output_contrast)

        return depth_norm.astype(np.float32)

    def _save_depth_outputs(self, depth_norm: np.ndarray, save_dir: Path) -> None:
        """Save ControlNet depth PNG and turbo visualization."""
        import cv2

        depth_uint8 = (depth_norm * 255).clip(0, 255).astype(np.uint8)

        controlnet_path = save_dir / "projector_depth_controlnet.png"
        cv2.imwrite(str(controlnet_path), depth_uint8)
        logger.info("ControlNet depth saved: %s", controlnet_path)

        turbo = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
        turbo_path = save_dir / "projector_depth_turbo.png"
        cv2.imwrite(str(turbo_path), turbo)

        np.save(str(save_dir / "static_depth_projector.npy"), depth_norm)

    def reapply_depth_output(self) -> None:
        """Re-apply output settings to camera-space depth, re-warp, update preview."""
        import cv2

        if self.state.raw_warped_depth is None:
            return
        map_x = self.state.calib_map_x
        map_y = self.state.calib_map_y
        if map_x is None or map_y is None:
            return

        # Re-process camera-space depth with current settings
        depth_img = self._apply_depth_output_settings(self.state.raw_warped_depth)
        depth_uint8_cam = (depth_img * 255).clip(0, 255).astype(np.uint8)
        depth_bgr_cam = cv2.cvtColor(depth_uint8_cam, cv2.COLOR_GRAY2BGR)

        # Re-warp to projector space (same as RGB)
        depth_proj_bgr = cv2.remap(depth_bgr_cam, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        depth_holes = np.all(depth_proj_bgr == 0, axis=2).astype(np.uint8) * 255
        if np.any(depth_holes):
            depth_proj_bgr = self._inpaint_small_holes(depth_proj_bgr, depth_holes)

        depth_proj_uint8 = depth_proj_bgr[:, :, 0]
        depth_norm = depth_proj_uint8.astype(np.float32) / 255.0
        self.state.static_depth_map = depth_norm
        self.state.depth_frame = depth_norm.copy()

    def _build_metric_projector_depth(
        self,
        map_x: np.ndarray,
        map_y: np.ndarray,
        proj_valid_mask: np.ndarray | None = None,
    ) -> None:
        """Build metric depth map using stereo triangulation (secondary).

        Uses the calibrated projector+camera stereo pair and Gray code
        correspondences to triangulate 3D points.  Only useful when stereo
        calibration has low reprojection error.
        """
        import cv2

        logger.info("Building metric depth map using stereo triangulation...")

        procam = self.state.procam
        pw = self.state.calibration.projector_width
        ph = self.state.calibration.projector_height

        try:
            builder = ProjectorDepthBuilder(
                K_cam=procam.K_cam,
                dist_cam=procam.dist_cam,
                K_proj=procam.K_proj,
                dist_proj=procam.dist_proj,
                R=procam.R,
                T=procam.T,
                proj_w=pw,
                proj_h=ph,
            )

            # Use the REAL validity mask (not map_x >= 0 which is true everywhere after inpainting)
            if proj_valid_mask is None:
                proj_valid_mask = (map_x >= 0) & (map_y >= 0)

            color_frame = self.state.camera_frame
            depth_map, rgb_proj, valid_mask = builder.build_depth_with_color(
                map_x, map_y, color_frame, proj_valid_mask
            )

            depth_filled = builder.fill_holes(depth_map, valid_mask)

            # Save stereo-triangulated outputs (separate prefix from warped outputs)
            save_dir = Path(self.state.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            builder.save_depth_map(depth_filled, valid_mask, save_dir, prefix="stereo_depth")

            logger.info("Stereo triangulation depth map complete!")

        except Exception as e:
            logger.error(f"Failed to build stereo depth map: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _capture_static_depth_map(self) -> None:
        """Recapture static depth using the stored correspondence map.
        
        This re-runs the stereo triangulation with a fresh camera frame
to update the RGB texture while keeping the same geometry.
        """
        logger.info("Recapturing static depth map...")
        
        if self.state.calib_map_x is None:
            logger.warning("No correspondence map available - calibrate first")
            return
        
        # Just rebuild using the existing correspondence
        self._build_metric_projector_depth(self.state.calib_map_x, self.state.calib_map_y)

    def _load_static_depth_map(self) -> bool:
        """Load the static depth map from disk if available.
        
        Returns True if successfully loaded.
        """
        depth_path = Path(self.state.output_dir) / "static_depth_projector.npy"
        if not depth_path.exists():
            return False
        
        try:
            depth_proj = np.load(str(depth_path))
            self.state.static_depth_map = depth_proj
            self.state.static_depth_captured = True
            logger.info(f"Static depth map loaded: {depth_proj.shape[1]}x{depth_proj.shape[0]}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load static depth map: {e}")
            return False

    def _save_and_open_reprojection_preview(self) -> None:
        """Save a visualization of the reprojected depth and open it.
        
        Called automatically when calibration completes with 3D reprojection.
        """
        import cv2
        import subprocess
        import os
        
        # Use the static depth map if available
        if self.state.static_depth_map is not None:
            depth_reproj = self.state.static_depth_map
        else:
            # Fallback: capture fresh
            frame = self.state.camera_frame
            if frame is None:
                logger.warning("Cannot save reprojection preview: no camera frame")
                return
            try:
                depth = self.depth_estimator.estimate(frame)
                depth_reproj = self._depth_reprojector.reproject(depth)
            except Exception as e:
                logger.warning(f"Failed to generate reprojection preview: {e}")
                return
        
        # Create visualization (turbo colormap)
        h, w = depth_reproj.shape
        depth_u8 = (depth_reproj * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        
        # Add info text overlay
        info_text = f"Projector Depth: {w}x{h}  range: [{depth_reproj.min():.3f}, {depth_reproj.max():.3f}]"
        cv2.putText(depth_color, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_color, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Save to file
        save_dir = Path(self.state.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / "reprojected_depth.png"
        cv2.imwrite(str(output_path), depth_color)
        logger.info(f"Reprojection preview saved to: {output_path}")
        
        # Open with default image viewer
        try:
            if os.name == 'nt':  # Windows
                os.startfile(str(output_path))
            elif os.name == 'posix':  # macOS/Linux
                subprocess.run(['open', str(output_path)], check=False)
            logger.info(f"Opened reprojection preview: {output_path}")
        except Exception as e:
            logger.warning(f"Could not open image viewer: {e}")

    def load_calibration_file(self) -> None:
        """Load calibration from disk."""
        path = Path(self.state.output_dir) / "calibration.json"
        if not path.exists():
            logger.warning("No calibration file found at %s", path)
            return
        try:
            procam = load_procam(path)
            self.state.procam = procam
            self.state.calib_map_x = procam.map_x
            self.state.calib_map_y = procam.map_y
            self.state.calibration_file = str(path)
            self.state.calib_phase = CalibrationPhase.DONE

            # Build depth reprojector from correspondence maps
            if procam.map_x is not None:
                self._build_depth_reprojector()
                self._load_static_depth_map()

            # Build 3D frustum for scene preview
            if procam.K_cam is not None and procam.R is not None and self._scene is not None:
                self._scene.build_frustum_from_intrinsics(
                    procam.K_proj, procam.R, procam.T,
                    self.state.calibration.projector_width,
                    self.state.calibration.projector_height,
                )
            elif procam.map_x is not None and self._scene is not None:
                dw = self._depth_texture.width if self._depth_texture else 640
                focal = max(dw, 480) * 0.8
                self._scene.build_frustum(
                    self.state.calibration.projector_width,
                    self.state.calibration.projector_height,
                    focal,
                )

            self.state.mode = AppMode.LIVE
            logger.info("Calibration loaded from %s", path)
        except Exception as e:
            logger.error("Failed to load calibration: %s", e)

    # ── Calibration preview ────────────────────────────────────────────

    def _update_calib_preview(self) -> None:
        """Run corner detection on camera frame and build annotated preview."""
        import cv2

        frame = self.state.camera_frame
        if frame is None:
            self._calib_corners_found = False
            return

        preview = frame.copy()
        phase = self.state.calib_phase

        if phase == CalibrationPhase.CAMERA_INTRINSICS and self._procam_calibrator is not None:
            # Detect projected checkerboard corners for live overlay
            found, corners = self._procam_calibrator.detect_current_board(frame)
            self._calib_corners_found = found
            self._calib_corners_count = len(corners) if corners is not None else 0

            # Draw corners on preview
            board_size = self._procam_calibrator.board_size
            cv2.drawChessboardCorners(preview, board_size, corners, found)

            # Add status text overlay
            pos_idx = self._procam_calibrator.current_position_idx
            total = self._procam_calibrator.total_positions
            captures = self._procam_calibrator.capture_count

            status = "DETECTED" if found else "NOT FOUND"
            color = (0, 255, 0) if found else (0, 0, 255)
            cv2.putText(preview, f"Board: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(preview, f"Position {pos_idx + 1}/{total}  Captures: {captures}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if found:
                cv2.putText(preview, "Press [Capture] in panel", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        elif phase == CalibrationPhase.GRAY_CODES:
            # Show progress overlay during Gray code capture
            if self._calib_runner is not None:
                progress = self._calib_pattern_idx / max(self._calib_runner.total_patterns, 1)
                cv2.putText(preview, f"Gray Codes: {self._calib_pattern_idx}/{self._calib_runner.total_patterns}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                # Draw progress bar
                bar_w = preview.shape[1] - 20
                bar_h = 20
                bar_y = 50
                cv2.rectangle(preview, (10, bar_y), (10 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                cv2.rectangle(preview, (10, bar_y), (10 + int(bar_w * progress), bar_y + bar_h), (0, 200, 255), -1)

        elif phase == CalibrationPhase.COMPUTING:
            cv2.putText(preview, "Computing calibration...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        # Upload preview to texture
        h, w = preview.shape[:2]
        rgb = preview[::-1, :, ::-1].copy()  # BGR→RGB + flip vertical
        if self._calib_preview_texture is None:
            self._calib_preview_texture = self._main_ctx.texture((w, h), 3, dtype="f1")
        else:
            self._calib_preview_texture = self._ensure_texture_size(
                self._calib_preview_texture, w, h, 3, "f1"
            )
        self._calib_preview_texture.write(rgb.tobytes())

    def _render_calib_preview(self, viewport: tuple[int, int, int, int]) -> None:
        """Render the calibration camera preview in the main viewport."""
        if self._calib_preview_texture is None:
            return

        ctx = self._main_ctx
        x, y, w, h = viewport
        ctx.viewport = viewport

        self._calib_preview_texture.use(0)
        self._passthrough_prog["tex"].value = 0
        self._fullscreen_vao.render(moderngl.TRIANGLES, vertices=3)

    # ── Camera enumeration ─────────────────────────────────────────────

    def refresh_camera_list(self) -> list[tuple[int, str]]:
        """Enumerate available cameras in a background thread.

        Skips the currently active camera to avoid interfering with capture.
        Only call when the user explicitly clicks Refresh.
        """
        if self._camera_refreshing:
            return self._camera_list

        import threading
        self._camera_refreshing = True

        def _enum():
            try:
                result = enumerate_cameras(
                    max_index=10, skip_index=self.state.camera_device_idx,
                )
                self._camera_list = result
                logger.info("Camera enumeration found %d devices", len(result))
            except Exception as e:
                logger.warning("Camera enumeration error: %s", e)
            finally:
                self._camera_refreshing = False

        threading.Thread(target=_enum, daemon=True).start()
        return self._camera_list

    def get_camera_list(self) -> list[tuple[int, str]]:
        """Return cached camera list (no auto-enumeration)."""
        return self._camera_list

    def switch_camera(self, device_idx: int) -> None:
        """Switch to a different camera device."""
        if device_idx == self.state.camera_device_idx:
            return
        self.state.camera_device_idx = device_idx
        self.camera.stop()
        try:
            self.camera.start(device_idx)
            logger.info("Switched to camera device %d", device_idx)
        except Exception as e:
            logger.warning("Failed to start camera %d: %s", device_idx, e)

    # ── NDI camera management ──────────────────────────────────────────

    def discover_ndi_sources(self) -> list[str]:
        """Discover NDI sources on the network in a background thread.

        Returns current cached list immediately; updates list when discovery completes.
        """
        if self._ndi_discovering:
            return self._ndi_source_list

        import threading
        self._ndi_discovering = True

        def _discover():
            try:
                result = discover_ndi_sources(timeout=5.0)
                self._ndi_source_list = result
                logger.info("NDI discovery complete: %d sources", len(result))
            except Exception as e:
                logger.warning("NDI discovery error: %s", e)
            finally:
                self._ndi_discovering = False

        threading.Thread(target=_discover, daemon=True).start()
        return self._ndi_source_list

    def get_ndi_source_list(self) -> list[str]:
        """Return cached NDI source list."""
        return self._ndi_source_list

    def switch_to_ndi(self, source_name: str) -> None:
        """Switch camera input to an NDI source."""
        # Stop local camera
        self.camera.stop()
        self.state.camera_source = CameraSource.NDI
        self.state.ndi_source_name = source_name
        self.ndi_camera.start(source_name)
        self._latency_measured = False  # need re-measure for NDI
        logger.info("Switched to NDI source: %s", source_name)

    def switch_to_local_camera(self, device_idx: int | None = None) -> None:
        """Switch camera input back to local webcam."""
        self.ndi_camera.stop()
        self.state.camera_source = CameraSource.LOCAL
        idx = device_idx if device_idx is not None else self.state.camera_device_idx
        self.state.camera_device_idx = idx
        try:
            self.camera.start(idx)
            logger.info("Switched to local camera %d", idx)
        except Exception as e:
            logger.warning("Failed to start local camera %d: %s", idx, e)
        self._latency_measured = False

    # ── Latency measurement ──────────────────────────────────────────

    def _measure_latency(self) -> None:
        """Measure pattern→camera latency by flashing white and timing response.

        Called at the start of calibration. Projects a white frame and measures
        how long until the camera sees a brightness jump.
        """
        import cv2

        if self._proj_window is None:
            self._latency_ms = 200.0  # fallback
            self._latency_measured = True
            return

        # Capture baseline brightness (with projector dark)
        glfw.make_context_current(self._proj_window)
        self._proj_ctx.screen.use()
        w, h = glfw.get_framebuffer_size(self._proj_window)
        self._proj_ctx.viewport = (0, 0, w, h)
        self._proj_ctx.clear(0.0, 0.0, 0.0)
        glfw.swap_buffers(self._proj_window)
        glfw.make_context_current(self._main_window)

        time.sleep(0.3)  # let camera settle on dark

        dark_frame = self.state.camera_frame
        if dark_frame is None:
            self._latency_ms = 200.0
            self._latency_measured = True
            return

        dark_mean = float(np.mean(cv2.cvtColor(dark_frame, cv2.COLOR_BGR2GRAY)))

        # Flash white
        glfw.make_context_current(self._proj_window)
        self._proj_ctx.screen.use()
        self._proj_ctx.viewport = (0, 0, w, h)
        self._proj_ctx.clear(1.0, 1.0, 1.0)
        glfw.swap_buffers(self._proj_window)
        glfw.make_context_current(self._main_window)

        flash_time = time.monotonic()
        threshold = dark_mean + max(20.0, dark_mean * 0.3)

        # Poll camera for brightness jump
        measured = False
        for _ in range(100):  # up to ~1s
            time.sleep(0.01)
            glfw.poll_events()
            frame = self.state.camera_frame
            if frame is None:
                continue
            brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            if brightness > threshold:
                delay = (time.monotonic() - flash_time) * 1000.0
                self._latency_ms = delay
                measured = True
                logger.info("Measured camera latency: %.0f ms (dark=%.1f, bright=%.1f, thresh=%.1f)",
                           delay, dark_mean, brightness, threshold)
                break

        if not measured:
            self._latency_ms = 200.0
            logger.warning("Latency measurement failed — using default 200ms")

        # Clear projector back to dark
        glfw.make_context_current(self._proj_window)
        self._proj_ctx.screen.use()
        self._proj_ctx.clear(0.0, 0.0, 0.0)
        glfw.swap_buffers(self._proj_window)
        glfw.make_context_current(self._main_window)

        self._latency_measured = True

    def _update_camera_preview(self) -> None:
        """Upload camera frame to a preview texture for UI display."""
        frame = self.state.camera_frame
        if frame is None:
            return
        h, w = frame.shape[:2]
        rgb = frame[::-1, :, ::-1].copy()  # BGR→RGB + flip vertical
        if self._camera_preview_texture is None:
            self._camera_preview_texture = self._main_ctx.texture((w, h), 3, dtype="f1")
        else:
            self._camera_preview_texture = self._ensure_texture_size(
                self._camera_preview_texture, w, h, 3, "f1"
            )
        self._camera_preview_texture.write(rgb.tobytes())

    def _update_depth_vis(self) -> None:
        """Upload a colormapped depth visualization texture."""
        import cv2 as _cv2

        depth = self.state.depth_frame
        if depth is None:
            return
        h, w = depth.shape

        # Apply colormap for visualization
        gray_u8 = (depth * 255).clip(0, 255).astype(np.uint8)
        colormap_id = {
            "grayscale": -1,
            "turbo": _cv2.COLORMAP_TURBO,
            "viridis": _cv2.COLORMAP_VIRIDIS,
            "magma": _cv2.COLORMAP_MAGMA,
        }.get(self.state.depth.colormap, _cv2.COLORMAP_TURBO)

        if colormap_id == -1:
            rgb = np.stack([gray_u8] * 3, axis=-1)
        else:
            bgr = _cv2.applyColorMap(gray_u8, colormap_id)
            rgb = bgr[:, :, ::-1]  # BGR→RGB

        rgb_flipped = rgb[::-1].copy()  # flip vertical for OpenGL

        if self._depth_vis_texture is None:
            self._depth_vis_texture = self._main_ctx.texture((w, h), 3, dtype="f1")
        else:
            self._depth_vis_texture = self._ensure_texture_size(
                self._depth_vis_texture, w, h, 3, "f1"
            )
        self._depth_vis_texture.write(rgb_flipped.tobytes())

    # ── Frame update ─────────────────────────────────────────────────────

    def _start_depth_thread(self) -> None:
        """Start background thread for depth inference."""
        if self.depth_estimator is None:
            return
        self._depth_running = True
        self._depth_thread = threading.Thread(target=self._depth_loop, daemon=True)
        self._depth_thread.start()
        logger.info("Depth inference thread started")

    def _stop_depth_thread(self) -> None:
        self._depth_running = False
        self._depth_new_input.set()  # wake thread so it can exit
        if self._depth_thread is not None:
            self._depth_thread.join(timeout=3.0)
            self._depth_thread = None

    def _depth_loop(self) -> None:
        """Background loop: wait for new frames, run inference, store result."""
        while self._depth_running:
            # Check if we should use static depth (skip live inference)
            if self.state.depth.use_static_depth and self.state.static_depth_captured:
                # Use the static depth map - just copy it to depth_frame
                if self.state.static_depth_map is not None:
                    self.state.depth_frame = self.state.static_depth_map.copy()
                # Still wait for the event to avoid busy-looping, but don't process
                self._depth_new_input.wait(timeout=0.1)
                self._depth_new_input.clear()
                continue

            # Wait for a new frame (with timeout so we can check _running)
            self._depth_new_input.wait(timeout=0.5)
            self._depth_new_input.clear()

            if not self._depth_running:
                break

            with self._depth_lock:
                frame = self._depth_input_frame
            if frame is None:
                continue

            try:
                depth = self.depth_estimator.estimate(frame)
                h, w = depth.shape
                logger.debug(f"Depth estimated: {w}x{h}  range: [{depth.min():.3f}, {depth.max():.3f}]")

                # Reproject to projector perspective if calibrated and enabled
                if self._depth_reprojector is not None and self.state.depth.use_projector_perspective:
                    depth = self._depth_reprojector.reproject(depth)
                    logger.debug(f"Depth reprojected: {depth.shape[1]}x{depth.shape[0]}  range: [{depth.min():.3f}, {depth.max():.3f}]")
                elif self._depth_reprojector is not None:
                    logger.debug("Projector perspective available but disabled")
                else:
                    logger.debug("No depth reprojector available")

                self.state.depth_frame = depth

                if self.state.live_depth_export:
                    self._export_live_depth(depth)
            except Exception as e:
                logger.warning("Depth estimation failed: %s", e)
                import traceback
                logger.debug(traceback.format_exc())

    def _update_depth(self) -> None:
        """Feed latest camera frame to the background depth thread."""
        if self.depth_estimator is None:
            return
        frame = self.state.camera_frame
        if frame is None:
            return
        with self._depth_lock:
            self._depth_input_frame = frame
        self._depth_new_input.set()

    _last_export_time: float = 0.0

    def _export_live_depth(self, depth: np.ndarray) -> None:
        """Write depth maps to shared directory for the Scope plugin to read.

        Exports three feeds:
        - depth_bw.npy     — (H, W) float32 [0,1] B&W depth
        - depth_color.npy  — (H, W, 3) uint8 colormapped depth
        - projector_rgb.npy — (H, W, 3) uint8 color-corrected projector output
        - meta.json        — timestamp + availability flags
        """
        import json

        now = time.monotonic()
        # Throttle to ~10 Hz to avoid disk thrashing
        if now - self._last_export_time < 0.1:
            return
        self._last_export_time = now

        try:
            (Path(self.state.output_dir) / "live").mkdir(parents=True, exist_ok=True)

            # Depth B&W
            self._atomic_save(Path(self.state.output_dir) / "live" / "depth_bw.npy", depth)

            # Depth colormapped (turbo colormap via OpenCV)
            import cv2
            gray_u8 = (depth * 255).clip(0, 255).astype(np.uint8)
            color = cv2.applyColorMap(gray_u8, cv2.COLORMAP_TURBO)
            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            self._atomic_save(Path(self.state.output_dir) / "live" / "depth_color.npy", color_rgb)

            # Projector RGB (camera frame color-corrected by calibration)
            frame = self.state.camera_frame
            if frame is not None and self.state.calib_map_x is not None:
                warped = cv2.remap(
                    frame, self.state.calib_map_x, self.state.calib_map_y,
                    cv2.INTER_LINEAR,
                )
                warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                self._atomic_save(Path(self.state.output_dir) / "live" / "projector_rgb.npy", warped_rgb)
                has_projector = True
            else:
                has_projector = False

            # Metadata
            meta = {
                "timestamp": time.time(),
                "width": depth.shape[1],
                "height": depth.shape[0],
                "has_calibration": self.state.calib_map_x is not None,
                "feeds": {
                    "depth_bw": True,
                    "depth_color": True,
                    "projector_rgb": has_projector,
                },
            }
            (Path(self.state.output_dir) / "live" / "meta.json").write_text(json.dumps(meta))
        except Exception as e:
            logger.debug("Failed to export live depth: %s", e)

    @staticmethod
    def _atomic_save(path: Path, arr: np.ndarray) -> None:
        tmp = path.with_suffix(".tmp")
        np.save(str(tmp), arr)
        tmp.rename(path)

    def _upload_textures(self) -> None:
        """Upload camera and depth frames to OpenGL textures."""
        # Camera texture
        frame = self.state.camera_frame
        if frame is not None:
            h, w = frame.shape[:2]
            rgb = frame[::-1, :, ::-1].copy()  # BGR→RGB + flip vertical
            self._color_texture = self._ensure_texture_size(
                self._color_texture, w, h, 3, "f1"
            )
            self._color_texture.write(rgb.tobytes())

        # Depth texture
        depth = self.state.depth_frame
        if depth is not None:
            h, w = depth.shape
            flipped = depth[::-1].copy()
            self._depth_texture = self._ensure_texture_size(
                self._depth_texture, w, h, 1, "f4"
            )
            self._depth_texture.write(flipped.tobytes())

    # ── Rendering ────────────────────────────────────────────────────────

    def _render_main_window(self) -> None:
        """Render the main window: viewport (left) + ImGui UI (right panel)."""
        glfw.make_context_current(self._main_window)
        w, h = glfw.get_framebuffer_size(self._main_window)

        self._main_ctx.screen.use()
        self._main_ctx.viewport = (0, 0, w, h)
        self._main_ctx.clear(0.12, 0.12, 0.15, 1.0)

        # Viewport occupies the left portion (full width minus UI panel)
        panel_w = 340
        scene_w = max(w - panel_w, 100)
        scene_viewport = (0, 0, scene_w, h)

        if self.state.mode == AppMode.CALIBRATING:
            # During calibration, always show annotated camera preview
            self._render_calib_preview(scene_viewport)
        else:
            view = self.state.main_view
            if view == MainViewMode.CAMERA:
                self._render_fullscreen_texture(
                    self._camera_preview_texture, scene_viewport
                )
            elif view == MainViewMode.DEPTH:
                self._render_fullscreen_texture(
                    self._depth_vis_texture, scene_viewport
                )
            elif view == MainViewMode.PROJECTOR:
                self._render_fullscreen_texture(
                    self._effects_output_tex, scene_viewport
                )
            elif view == MainViewMode.SCENE_3D:
                if self.state.depth_frame is not None and self._scene is not None:
                    self._scene.render(
                        self.state,
                        self._depth_texture,
                        self._color_texture,
                        viewport=scene_viewport,
                    )

        # Reset viewport and render ImGui
        self._main_ctx.viewport = (0, 0, w, h)
        self._render_imgui()

        glfw.swap_buffers(self._main_window)

    def _render_fullscreen_texture(
        self,
        tex: moderngl.Texture | None,
        viewport: tuple[int, int, int, int],
    ) -> None:
        """Render a texture fullscreen in the given viewport."""
        if tex is None or tex.width <= 1:
            return
        ctx = self._main_ctx
        ctx.viewport = viewport
        tex.use(0)
        self._passthrough_prog["tex"].value = 0
        self._fullscreen_vao.render(moderngl.TRIANGLES, vertices=3)

    def _render_imgui(self) -> None:
        """Process ImGui input and render the draw data."""
        self._imgui_renderer.process_inputs()

        from .ui import render_ui
        render_ui(self)

        draw_data = imgui.get_draw_data()
        self._imgui_renderer.render(draw_data)

    def _proj_upload_and_render(self, data: np.ndarray) -> None:
        """Upload RGBA uint8 image to projector texture and render fullscreen."""
        ctx = self._proj_ctx
        h, w = data.shape[:2]

        if (self._proj_texture is None or
                self._proj_texture.width != w or
                self._proj_texture.height != h):
            if self._proj_texture is not None:
                self._proj_texture.release()
            self._proj_texture = ctx.texture((w, h), 4, dtype="f1")
            self._proj_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self._proj_texture.write(data.tobytes())
        self._proj_texture.use(0)
        self._proj_passthrough_prog["tex"].value = 0
        self._proj_fullscreen_vao.render(moderngl.TRIANGLES, vertices=3)

    def _render_projector_window(self) -> None:
        """Render the projector window: calibration patterns or effects."""
        if self._proj_window is None or self._proj_ctx is None:
            return

        glfw.make_context_current(self._proj_window)
        ctx = self._proj_ctx
        w, h = glfw.get_framebuffer_size(self._proj_window)
        ctx.screen.use()
        ctx.viewport = (0, 0, w, h)

        if self.state.mode == AppMode.CALIBRATING:
            pattern = self._get_current_calibration_pattern()
            if pattern is not None:
                if pattern.ndim == 2:
                    ph, pw = pattern.shape
                    rgba = np.zeros((ph, pw, 4), dtype=np.uint8)
                    rgba[:, :, 0] = pattern
                    rgba[:, :, 1] = pattern
                    rgba[:, :, 2] = pattern
                    rgba[:, :, 3] = 255
                else:
                    ph, pw = pattern.shape[:2]
                    rgba = np.zeros((ph, pw, 4), dtype=np.uint8)
                    rgba[:, :, :3] = pattern
                    rgba[:, :, 3] = 255
                rgba = rgba[::-1].copy()
                self._proj_upload_and_render(rgba)
            else:
                ctx.clear(0.0, 0.0, 0.0)
        elif self.state.projector_source == ProjectorSource.SPOUT_INPUT:
            frame = self._spout_receive_frame
            if frame is not None:
                flipped = frame[::-1].copy()
                self._proj_upload_and_render(flipped)
            else:
                ctx.clear(0.0, 0.0, 0.0)
        else:
            # Effects on depth — only when calibrated and effects are active
            any_effect = self._any_effect_enabled()
            if self.state.depth_frame is not None and any_effect and self.state.calib_map_x is not None:
                # Render effects on main context to FBO
                glfw.make_context_current(self._main_window)
                self._ensure_effects_fbo(w, h)
                self._effects_fbo.use()
                self._main_ctx.viewport = (0, 0, w, h)
                self._render_effects(w, h)

                # Read back pixels from FBO
                raw = self._effects_fbo.read(components=4)
                effects_rgba = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)

                # Upload to projector
                glfw.make_context_current(self._proj_window)
                ctx.screen.use()
                ctx.viewport = (0, 0, w, h)
                self._proj_upload_and_render(effects_rgba)
            else:
                # Idle: show mid-grey so the camera settles to a balanced
                # exposure before calibration starts
                ctx.clear(0.5, 0.5, 0.5)

        glfw.swap_buffers(self._proj_window)

        # Switch context back to main for next frame
        glfw.make_context_current(self._main_window)

    def _any_effect_enabled(self) -> bool:
        """Check if any visual effect is currently enabled."""
        eff = self.state.effects
        return any([
            eff.noise_enabled, eff.flow_enabled, eff.pulse_enabled,
            eff.wave_enabled, eff.kaleido_enabled, eff.shockwave_enabled,
            eff.wobble_enabled, eff.edges_enabled, eff.fog_enabled,
            eff.zoom_enabled,
        ])

    @staticmethod
    def _set_uniform(prog, name, value):
        """Set a shader uniform, silently skipping if optimized out by GLSL."""
        if name in prog:
            prog[name].value = value

    def _render_effects(self, w: int, h: int) -> None:
        """Render depth map through effects shader."""
        prog = self._effects_program
        eff = self.state.effects
        t = time.monotonic() - self._start_time
        su = self._set_uniform

        self._depth_texture.use(0)
        su(prog, "depth_tex", 0)
        su(prog, "u_time", t)
        su(prog, "u_resolution", (float(w), float(h)))

        # Noise
        su(prog, "u_noise_enabled", eff.noise_enabled)
        su(prog, "u_noise_intensity", eff.noise_intensity)
        su(prog, "u_noise_scale", eff.noise_scale)
        su(prog, "u_noise_octaves", eff.noise_octaves)
        su(prog, "u_noise_speed", eff.noise_speed)

        # Flow
        su(prog, "u_flow_enabled", eff.flow_enabled)
        su(prog, "u_flow_intensity", eff.flow_intensity)
        su(prog, "u_flow_scale", eff.flow_scale)
        su(prog, "u_flow_speed", eff.flow_speed)

        # Pulse
        su(prog, "u_pulse_enabled", eff.pulse_enabled)
        su(prog, "u_pulse_speed", eff.pulse_speed)
        su(prog, "u_pulse_amount", eff.pulse_amount)

        # Wave
        su(prog, "u_wave_enabled", eff.wave_enabled)
        su(prog, "u_wave_frequency", eff.wave_frequency)
        su(prog, "u_wave_amplitude", eff.wave_amplitude)
        su(prog, "u_wave_speed", eff.wave_speed)
        su(prog, "u_wave_direction", eff.wave_direction)

        # Shockwave
        su(prog, "u_shockwave_enabled", eff.shockwave_enabled)
        su(prog, "u_shockwave_origin", (eff.shockwave_origin_x, eff.shockwave_origin_y))
        su(prog, "u_shockwave_speed", eff.shockwave_speed)
        su(prog, "u_shockwave_thickness", eff.shockwave_thickness)
        su(prog, "u_shockwave_strength", eff.shockwave_strength)
        su(prog, "u_shockwave_decay", eff.shockwave_decay)
        su(prog, "u_shockwave_interval", eff.shockwave_interval)

        # Wobble
        su(prog, "u_wobble_enabled", eff.wobble_enabled)
        su(prog, "u_wobble_intensity", eff.wobble_intensity)
        su(prog, "u_wobble_speed", eff.wobble_speed)

        # Edges
        su(prog, "u_edges_enabled", eff.edges_enabled)
        su(prog, "u_edges_strength", eff.edges_strength)
        su(prog, "u_edges_glow_width", eff.edges_glow_width)
        su(prog, "u_edges_pulse_speed", eff.edges_pulse_speed)

        # Fog
        su(prog, "u_fog_enabled", eff.fog_enabled)
        su(prog, "u_fog_density", eff.fog_density)
        su(prog, "u_fog_near", eff.fog_near)
        su(prog, "u_fog_far", eff.fog_far)
        su(prog, "u_fog_animated", eff.fog_animated)
        su(prog, "u_fog_speed", eff.fog_speed)

        # Zoom
        su(prog, "u_zoom_enabled", eff.zoom_enabled)
        su(prog, "u_zoom_origin", (eff.zoom_origin_x, eff.zoom_origin_y))
        su(prog, "u_zoom_strength", eff.zoom_strength)
        su(prog, "u_zoom_speed", eff.zoom_speed)

        # Kaleidoscope
        su(prog, "u_kaleido_enabled", eff.kaleido_enabled)
        su(prog, "u_kaleido_segments", eff.kaleido_segments)
        su(prog, "u_kaleido_rotation", eff.kaleido_rotation)
        su(prog, "u_kaleido_spin_speed", eff.kaleido_spin_speed)

        self._effects_vao.render(moderngl.TRIANGLES, vertices=3)

    # ── Input callbacks ──────────────────────────────────────────────────

    def _scroll_callback(self, window, xoff, yoff):
        # Let ImGui handle scroll if it wants focus
        io = imgui.get_io()
        if io.want_capture_mouse:
            if self._prev_scroll_cb:
                self._prev_scroll_cb(window, xoff, yoff)
            return
        self.state.orbit.distance = max(0.5, self.state.orbit.distance - yoff * 0.3)

    def _mouse_button_callback(self, window, button, action, mods):
        io = imgui.get_io()
        # Forward to ImGui backend
        if self._prev_mouse_btn_cb:
            self._prev_mouse_btn_cb(window, button, action, mods)

        if io.want_capture_mouse:
            return

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self._dragging_orbit = True
                self._last_mouse_x, self._last_mouse_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                self._dragging_orbit = False

        if button == glfw.MOUSE_BUTTON_MIDDLE:
            if action == glfw.PRESS:
                self._panning = True
                self._last_mouse_x, self._last_mouse_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                self._panning = False

    def _cursor_pos_callback(self, window, xpos, ypos):
        io = imgui.get_io()
        # Forward to ImGui backend
        if self._prev_cursor_cb:
            self._prev_cursor_cb(window, xpos, ypos)

        if io.want_capture_mouse:
            return

        dx = xpos - self._last_mouse_x
        dy = ypos - self._last_mouse_y
        self._last_mouse_x = xpos
        self._last_mouse_y = ypos

        if self._dragging_orbit:
            self.state.orbit.yaw += dx * 0.3
            self.state.orbit.pitch = max(-89.0, min(89.0, self.state.orbit.pitch - dy * 0.3))

        if self._panning:
            import math
            speed = self.state.orbit.distance * 0.002
            yaw_rad = math.radians(self.state.orbit.yaw)
            # Pan in the camera's local right/up directions
            self.state.orbit.target_x -= math.cos(yaw_rad) * dx * speed
            self.state.orbit.target_z -= math.sin(yaw_rad) * dx * speed
            self.state.orbit.target_y += dy * speed

    def _key_callback(self, window, key, scancode, action, mods):
        # Forward to ImGui backend first
        if self._prev_key_cb:
            self._prev_key_cb(window, key, scancode, action, mods)

        if action != glfw.PRESS:
            return

        ctrl = mods & glfw.MOD_CONTROL

        # Ctrl+0..9 — switch projector to that monitor
        if ctrl and glfw.KEY_0 <= key <= glfw.KEY_9:
            idx = key - glfw.KEY_0
            monitors = glfw.get_monitors()
            if idx < len(monitors):
                self.state.projector_monitor_idx = idx
                self.open_projector_window(idx)
                logger.info("Switched projector to monitor %d (Ctrl+%d)", idx, idx)
            return

        # F11 — toggle projector window on/off
        if key == glfw.KEY_F11:
            if self._proj_window is not None:
                self.close_projector_window()
            else:
                self.open_projector_window(self.state.projector_monitor_idx)
            return

        # Escape — close projector window (not the app)
        if key == glfw.KEY_ESCAPE and self._proj_window is not None:
            self.close_projector_window()
            return

    # ── Main loop ────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main render loop."""
        self._last_frame_time = time.monotonic()

        # Initialize Spout (needs GL context)
        self._init_spout()

        while not glfw.window_should_close(self._main_window):
            glfw.poll_events()

            # Timing
            now = time.monotonic()
            dt = now - self._last_frame_time
            self._last_frame_time = now
            self._frame_count += 1
            self.state.frame_time = dt
            self.state.fps = 1.0 / max(dt, 0.001)

            # Update calibration if active
            if self.state.mode == AppMode.CALIBRATING:
                if self.state.calib_phase == CalibrationPhase.CAMERA_INTRINSICS:
                    self._update_checkerboard_phase()
                elif self.state.calib_phase == CalibrationPhase.GRAY_CODES:
                    self._update_calibration()
                self._update_calib_preview()
                # Keep depth estimation running during calibration (needed for dense correspondences)
                self._update_depth()
            else:
                self._update_depth()

            # Receive Spout input if in that mode
            if self.state.projector_source == ProjectorSource.SPOUT_INPUT:
                if self._spout_receiver is None:
                    self._init_spout_receiver()
                self._receive_spout_frame()

            # Upload textures (main context)
            glfw.make_context_current(self._main_window)
            self._update_camera_preview()
            self._update_depth_vis()
            self._upload_textures()

            # Render both windows
            self._render_main_window()
            self._render_projector_window()

            # Send Spout frames
            self._send_spout_frames()

            # Check if projector window was closed
            if self._proj_window and glfw.window_should_close(self._proj_window):
                self.close_projector_window()

    def cleanup(self) -> None:
        self._stop_depth_thread()
        self.camera.stop()
        self.ndi_camera.stop()
        self._spout.release()
        if self._spout_receiver is not None:
            self._spout_receiver.release()
        if self._scene is not None:
            self._scene.release()
        if self._imgui_renderer is not None:
            self._imgui_renderer.shutdown()
        self.close_projector_window()
        if self._main_window:
            glfw.destroy_window(self._main_window)
        glfw.terminate()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app = ProMapAnythingApp()
    try:
        app.init()

        # Auto-populate projector resolution from selected monitor
        monitors = glfw.get_monitors()
        if monitors:
            idx = min(app.state.projector_monitor_idx, len(monitors) - 1)
            app.state.projector_monitor_idx = idx
            mode = glfw.get_video_mode(monitors[idx])
            app.state.calibration.projector_width = mode.size.width
            app.state.calibration.projector_height = mode.size.height

        # Auto-open projector on second monitor if available
        if len(monitors) > 1:
            app.open_projector_window(1)
            app.state.projector_monitor_idx = 1

        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
