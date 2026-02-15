"""ImGui control panel for ProMapAnything."""

from __future__ import annotations

from typing import TYPE_CHECKING

import glfw
from imgui_bundle import imgui

if TYPE_CHECKING:
    from .app import ProMapAnythingApp

from .ndi_camera import ndi_available
from .state import AppMode, CalibrationPhase, CameraSource, MainViewMode, ProjectorSource, ViewMode


def render_ui(app: ProMapAnythingApp) -> None:
    """Render the full ImGui control panel.  Called each frame."""
    state = app.state

    imgui.new_frame()

    # ── Main control window ──────────────────────────────────────────────
    flags = (
        imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_resize
        | imgui.WindowFlags_.no_collapse
    )
    main_w, main_h = glfw.get_framebuffer_size(app._main_window)
    panel_w = 340.0
    imgui.set_next_window_pos(imgui.ImVec2(main_w - panel_w, 0))
    imgui.set_next_window_size(imgui.ImVec2(panel_w, float(main_h)))

    imgui.begin("Controls", flags=flags)

    # Status bar
    imgui.text(f"FPS: {state.fps:.0f}")
    imgui.same_line()
    mode_labels = {AppMode.IDLE: "Idle", AppMode.CALIBRATING: "Calibrating", AppMode.LIVE: "Live"}
    imgui.text(f"| Mode: {mode_labels.get(state.mode, '?')}")

    # ── Main viewport selector ────────────────────────────────────────
    view_modes = [
        (MainViewMode.CAMERA, "Camera"),
        (MainViewMode.DEPTH, "Depth"),
        (MainViewMode.SCENE_3D, "3D Scene"),
        (MainViewMode.PROJECTOR, "Projector"),
    ]
    for i, (vm, label) in enumerate(view_modes):
        if i > 0:
            imgui.same_line()
        selected = state.main_view == vm
        if selected:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.5, 0.8, 1.0))
        if imgui.small_button(label):
            state.main_view = vm
        if selected:
            imgui.pop_style_color()

    imgui.separator()

    # ── Monitor & Camera ─────────────────────────────────────────────
    if imgui.collapsing_header("Monitor & Camera", imgui.TreeNodeFlags_.default_open):
        monitors = glfw.get_monitors()
        monitor_names = [
            f"[{i}] {glfw.get_monitor_name(m)}" for i, m in enumerate(monitors)
        ]

        # Clamp index to valid range
        if state.projector_monitor_idx >= len(monitors):
            state.projector_monitor_idx = 0

        changed, idx = imgui.combo(
            "Projector##monitor", state.projector_monitor_idx, monitor_names
        )
        if changed:
            state.projector_monitor_idx = idx
            app.open_projector_window(idx)

        # Show selected monitor resolution
        sel_mon = monitors[state.projector_monitor_idx]
        sel_mode = glfw.get_video_mode(sel_mon)
        imgui.text(
            f"Monitor res: {sel_mode.size.width}x{sel_mode.size.height}"
        )

        # Open / close projector button
        if app._proj_window is None:
            if imgui.button("Open Projector Window"):
                app.open_projector_window(state.projector_monitor_idx)
        else:
            if imgui.button("Close Projector Window"):
                app.close_projector_window()

        imgui.separator()

        # ── Camera source type (Local / NDI) ───────────────────
        source_types = ["Local Webcam", "NDI (Network)"]
        has_ndi = ndi_available()
        if not has_ndi:
            source_types[1] += " [not installed]"
        cur_src = 0 if state.camera_source == CameraSource.LOCAL else 1
        changed, sel_src = imgui.combo("Source##cam", cur_src, source_types)
        if changed:
            if sel_src == 0 and state.camera_source != CameraSource.LOCAL:
                app.switch_to_local_camera()
            elif sel_src == 1 and has_ndi and state.camera_source != CameraSource.NDI:
                # Switch to NDI — need a source selected first
                ndi_list = app.get_ndi_source_list()
                if ndi_list:
                    app.switch_to_ndi(ndi_list[0])
                else:
                    # Trigger discovery
                    app.discover_ndi_sources()
                    ndi_list = app.get_ndi_source_list()
                    if ndi_list:
                        app.switch_to_ndi(ndi_list[0])

        if state.camera_source == CameraSource.LOCAL:
            # ── Local camera selection dropdown ────────────────
            camera_list = app.get_camera_list()
            if camera_list:
                cam_labels = [label for _, label in camera_list]
                cam_indices = [idx for idx, _ in camera_list]
                cur_sel = 0
                for i, ci in enumerate(cam_indices):
                    if ci == state.camera_device_idx:
                        cur_sel = i
                        break
                changed, sel = imgui.combo("Camera##device", cur_sel, cam_labels)
                if changed and sel < len(cam_indices):
                    app.switch_camera(cam_indices[sel])
            else:
                imgui.text("No cameras found")

            imgui.same_line()
            if app._camera_refreshing:
                imgui.text_colored(imgui.ImVec4(1.0, 1.0, 0.0, 1.0), "(scanning...)")
            elif imgui.small_button("Refresh##cam"):
                app.refresh_camera_list()

        elif state.camera_source == CameraSource.NDI:
            # ── NDI source selection ───────────────────────────
            ndi_list = app.get_ndi_source_list()
            # Auto-trigger discovery if we have no sources yet
            if not ndi_list and not app._ndi_discovering:
                app.discover_ndi_sources()
            if ndi_list:
                cur_ndi = 0
                for i, name in enumerate(ndi_list):
                    if name == state.ndi_source_name:
                        cur_ndi = i
                        break
                changed, sel_ndi = imgui.combo("NDI Source", cur_ndi, ndi_list)
                if changed and sel_ndi < len(ndi_list):
                    app.switch_to_ndi(ndi_list[sel_ndi])
            elif app._ndi_discovering:
                imgui.text("Discovering NDI sources...")
            else:
                imgui.text("No NDI sources found")

            if app._ndi_discovering:
                imgui.same_line()
                imgui.text_colored(imgui.ImVec4(1.0, 1.0, 0.0, 1.0), "(scanning...)")
            else:
                imgui.same_line()
                if imgui.small_button("Refresh##ndi"):
                    app.discover_ndi_sources()

            # NDI connection status
            if app.ndi_camera.connected:
                imgui.text_colored(
                    imgui.ImVec4(0.0, 1.0, 0.3, 1.0),
                    f"Connected ({app.ndi_camera.width}x{app.ndi_camera.height})"
                )
            elif state.ndi_source_name:
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.5, 0.0, 1.0), "Connecting..."
                )

        # ── Camera brightness / contrast ───────────────────────
        cam = state.camera_settings
        _, cam.brightness = imgui.slider_float(
            "Brightness##cam", cam.brightness, -1.0, 1.0
        )
        _, cam.contrast = imgui.slider_float(
            "Contrast##cam", cam.contrast, 0.5, 3.0
        )

        # ── Live camera preview ────────────────────────────────
        tex = app._camera_preview_texture
        if tex is not None and tex.width > 1:
            # Fit preview into the panel width
            avail_w = imgui.get_content_region_avail().x
            aspect = tex.height / max(tex.width, 1)
            preview_w = avail_w
            preview_h = preview_w * aspect
            imgui.image(
                imgui.ImTextureRef(tex.glo),
                imgui.ImVec2(preview_w, preview_h),
                imgui.ImVec2(0, 1),  # UV flipped: OpenGL origin is bottom-left
                imgui.ImVec2(1, 0),
            )
        else:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.5, 0.0, 1.0), "No camera feed"
            )

    # ── Calibration ──────────────────────────────────────────────────
    if imgui.collapsing_header("Calibration", imgui.TreeNodeFlags_.default_open):
        _render_calibration_wizard(app)

    # ── Projector Source ────────────────────────────────────────────
    if imgui.collapsing_header("Projector Source", imgui.TreeNodeFlags_.default_open):
        sources = ["Effects (local)", "Spout Input (from Scope)"]
        cur_src = 0 if state.projector_source == ProjectorSource.EFFECTS else 1
        changed, idx = imgui.combo("Source##proj", cur_src, sources)
        if changed:
            state.projector_source = (
                ProjectorSource.EFFECTS if idx == 0 else ProjectorSource.SPOUT_INPUT
            )

        if state.projector_source == ProjectorSource.SPOUT_INPUT:
            changed, name = imgui.input_text(
                "Spout source name", state.spout_receive_name
            )
            if changed:
                state.spout_receive_name = name
                app._init_spout_receiver()

            receiver = app._spout_receiver
            if receiver is not None and receiver.connected:
                sender = receiver.sender_name
                imgui.text_colored(
                    imgui.ImVec4(0.0, 1.0, 0.3, 1.0),
                    f"Connected: {sender} ({receiver.width}x{receiver.height})"
                )
            elif receiver is not None:
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.5, 0.0, 1.0),
                    "Waiting for Spout sender..."
                )
            imgui.text_wrapped(
                "Tip: Leave name empty to auto-detect. "
                "Or enter the Scope output name."
            )

    # ── Depth Settings ───────────────────────────────────────────────
    if imgui.collapsing_header("Depth", imgui.TreeNodeFlags_.default_open):
        d = state.depth
        # Show current depth map info
        depth_frame = state.depth_frame
        # Show metric depth map status
        if state.static_depth_captured and state.static_depth_map is not None:
            h, w = state.static_depth_map.shape
            imgui.text_colored(
                imgui.ImVec4(0.0, 1.0, 0.3, 1.0),
                f"Metric Depth (Stereo Triangulation): {w}x{h}"
            )
            imgui.text("Pure geometric - no depth sensor")
            imgui.text("Aligns with physical objects")
            _, d.use_static_depth = imgui.checkbox(
                "Use for ControlNet", d.use_static_depth
            )
            if imgui.button("Rebuild Depth Map"):
                app._capture_static_depth_map()
        elif app._depth_reprojector is not None:
            imgui.text_colored(
                imgui.ImVec4(1.0, 1.0, 0.0, 1.0),
                "Stereo calibrated - build depth map"
            )
            if imgui.button("Build Metric Depth Map"):
                app._capture_static_depth_map()
        else:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.5, 0.0, 1.0),
                "Calibrate to enable metric depth"
            )

        imgui.separator()

        # Show current depth frame info
        depth_frame = state.depth_frame
        if depth_frame is not None:
            h, w = depth_frame.shape
            d_min, d_max = depth_frame.min(), depth_frame.max()
            has_reproj = app._depth_reprojector is not None

            if has_reproj:
                if d.use_projector_perspective:
                    imgui.text(
                        f"Projector-perspective: {w}x{h}"
                    )
                else:
                    imgui.text(
                        f"Camera-perspective: {w}x{h}"
                    )
                _, d.use_projector_perspective = imgui.checkbox(
                    "Use projector perspective (live)", d.use_projector_perspective
                )
            else:
                imgui.text(
                    f"Camera-perspective depth: {w}x{h}"
                )
            imgui.text(f"Range: [{d_min:.3f}, {d_max:.3f}]")
        else:
            imgui.text_colored(imgui.ImVec4(1.0, 0.5, 0.0, 1.0), "No depth data")

        _, d.scale = imgui.slider_float("Scale", d.scale, 0.1, 5.0)
        _, d.offset = imgui.slider_float("Offset", d.offset, -1.0, 1.0)
        _, d.blur = imgui.slider_float("Blur", d.blur, 0.0, 20.0)
        _, d.invert = imgui.checkbox("Invert", d.invert)
        _, d.temporal_smoothing = imgui.slider_float(
            "Temporal smooth", d.temporal_smoothing, 0.0, 1.0
        )

        _, d.auto_open_reprojection = imgui.checkbox(
            "Auto-open reprojection after calib", d.auto_open_reprojection
        )

        colormaps = ["grayscale", "turbo", "viridis", "magma"]
        cur = colormaps.index(d.colormap) if d.colormap in colormaps else 0
        changed, idx = imgui.combo("Colormap", cur, colormaps)
        if changed:
            d.colormap = colormaps[idx]

        # ── ControlNet Output Adjustments ─────────────────────
        if state.raw_warped_depth is not None:
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(0.4, 0.8, 1.0, 1.0), "ControlNet Output")

            output_changed = False

            ch, val = imgui.checkbox("CLAHE Equalize##out", d.output_equalize)
            if ch:
                d.output_equalize = val
                output_changed = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Adaptive histogram equalization.\nSpreads depth values across the full 0-255 range\nfor maximum ControlNet detail.")

            ch, val = imgui.slider_float("Clip Low %##out", d.output_clip_lo, 0.0, 49.0)
            if ch:
                d.output_clip_lo = val
                output_changed = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Clip darkest N%% of depth values to black.\nHigher = more contrast, less near-depth detail.")

            ch, val = imgui.slider_float("Clip High %##out", d.output_clip_hi, 51.0, 100.0)
            if ch:
                d.output_clip_hi = val
                output_changed = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Clip brightest N%% of depth values to white.\nLower = more contrast, less far-depth detail.")

            ch, val = imgui.slider_float("Gamma##out", d.output_gamma, 0.2, 5.0)
            if ch:
                d.output_gamma = val
                output_changed = True
            if imgui.is_item_hovered():
                imgui.set_tooltip("Gamma curve: <1 brightens dark areas, >1 darkens them.\n1.0 = linear (no change).")

            ch, val = imgui.slider_float("Brightness##out", d.output_brightness, -1.0, 1.0)
            if ch:
                d.output_brightness = val
                output_changed = True

            ch, val = imgui.slider_float("Contrast##out", d.output_contrast, 0.5, 3.0)
            if ch:
                d.output_contrast = val
                output_changed = True

            if output_changed:
                app.reapply_depth_output()

            if imgui.button("Save ControlNet Depth"):
                from pathlib import Path as _P
                app.reapply_depth_output()
                save_dir = _P(state.output_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                app._save_depth_outputs(state.static_depth_map, save_dir)
            imgui.same_line()
            if imgui.button("Reset##out"):
                d.output_clip_lo = 2.0
                d.output_clip_hi = 98.0
                d.output_brightness = 0.0
                d.output_contrast = 1.0
                d.output_gamma = 1.0
                d.output_equalize = True
                app.reapply_depth_output()

    # ── 3D View (only when 3D Scene viewport is active) ─────────────
    if state.main_view == MainViewMode.SCENE_3D and imgui.collapsing_header("3D View", imgui.TreeNodeFlags_.default_open):
        is_pc = state.view_mode == ViewMode.POINT_CLOUD
        if imgui.radio_button("Point Cloud", is_pc):
            state.view_mode = ViewMode.POINT_CLOUD
        imgui.same_line()
        if imgui.radio_button("Mesh", not is_pc):
            state.view_mode = ViewMode.MESH

        orbit = state.orbit
        _, orbit.fov = imgui.slider_float("FOV", orbit.fov, 10.0, 120.0)
        _, orbit.distance = imgui.slider_float("Distance", orbit.distance, 0.5, 20.0)
        if imgui.button("Reset View"):
            orbit.yaw = 0.0
            orbit.pitch = -30.0
            orbit.distance = 3.0
            orbit.target_x = 0.0
            orbit.target_y = 0.0
            orbit.target_z = 0.0

    # ── Output / Sharing ────────────────────────────────────────────
    if imgui.collapsing_header("Output & Sharing"):
        # Output directory
        changed, new_dir = imgui.input_text("Output folder", state.output_dir)
        if changed:
            state.output_dir = new_dir
        if imgui.small_button("Open folder"):
            import subprocess, os
            out = state.output_dir
            if os.path.isdir(out):
                if os.name == 'nt':
                    os.startfile(out)
                else:
                    subprocess.run(['open', out], check=False)

        # Show saved files list
        from pathlib import Path as _P
        out_path = _P(state.output_dir)
        if out_path.is_dir():
            saved_files = sorted(out_path.glob("projector_depth*")) + sorted(out_path.glob("projector_rgb*"))
            if saved_files:
                imgui.text_colored(imgui.ImVec4(0.5, 0.8, 0.5, 1.0), "Saved files:")
                for f in saved_files:
                    imgui.bullet_text(f.name)

        imgui.separator()
        _, state.live_depth_export = imgui.checkbox(
            "Export to Scope plugin (live)", state.live_depth_export
        )
        if state.live_depth_export:
            imgui.text_colored(
                imgui.ImVec4(0.0, 1.0, 0.3, 1.0),
                "Exporting depth to Scope plugin"
            )
            live_dir = _P(state.output_dir) / "live"
            imgui.text_wrapped(f"Live dir: {live_dir}")

        imgui.separator()
        _, state.spout_enabled = imgui.checkbox("Enable Spout Output", state.spout_enabled)
        if state.spout_enabled:
            _, state.spout_depth_name = imgui.input_text(
                "Depth name", state.spout_depth_name
            )
            _, state.spout_color_name = imgui.input_text(
                "Color name", state.spout_color_name
            )
            _, state.spout_projector_name = imgui.input_text(
                "Projector name", state.spout_projector_name
            )

    # ── Effects ──────────────────────────────────────────────────────
    _render_effects_ui(state)

    imgui.end()
    imgui.render()


def _render_calibration_wizard(app: ProMapAnythingApp) -> None:
    """Render the automatic ProCam calibration UI."""
    state = app.state
    calib = state.calibration
    phase = state.calib_phase

    is_running = phase not in (CalibrationPhase.IDLE, CalibrationPhase.DONE)

    # ── Start / Cancel ──────────────────────────────────────
    if not is_running:
        if phase == CalibrationPhase.DONE:
            if imgui.button("Re-Calibrate"):
                app.start_calibration()
        else:
            if imgui.button("Start Calibration"):
                app.start_calibration()
            if app._proj_window is None:
                imgui.same_line()
                imgui.text_colored(
                    imgui.ImVec4(1.0, 1.0, 0.0, 1.0), "(will open projector)"
                )
    else:
        if imgui.button("Cancel"):
            app.cancel_calibration()

    # ── Settings (only when idle) ───────────────────────────
    if not is_running:
        if imgui.button("Match Monitor"):
            monitors = glfw.get_monitors()
            if state.projector_monitor_idx < len(monitors):
                mode = glfw.get_video_mode(monitors[state.projector_monitor_idx])
                calib.projector_width = mode.size.width
                calib.projector_height = mode.size.height
        imgui.same_line()
        imgui.text(f"{calib.projector_width}x{calib.projector_height}")
        _, calib.settle_ms = imgui.slider_float("Settle (ms)", calib.settle_ms, 50.0, 1000.0)
        if app._latency_measured:
            imgui.text(f"Measured latency: {app._latency_ms:.0f}ms")
        if imgui.small_button("Re-measure latency"):
            app._latency_measured = False

        # ── Decode quality ────────────────────────────────────
        imgui.separator()
        imgui.text("Decode Quality")
        _, calib.capture_frames = imgui.slider_int(
            "Frames/pattern", calib.capture_frames, 1, 10
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Capture N frames per pattern and average them.\nMore frames = less noise, but slower calibration.")
        _, calib.decode_threshold = imgui.slider_float(
            "Decode threshold", calib.decode_threshold, 5.0, 100.0
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Minimum white-black brightness difference to accept a pixel.\nLower = more coverage but more noise.")
        _, calib.morph_cleanup = imgui.checkbox("Morphological cleanup", calib.morph_cleanup)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Apply morphological close/open + median filter\nto fill holes and smooth decoded coordinates.")
        if calib.morph_cleanup:
            _, calib.morph_kernel_size = imgui.slider_int(
                "Kernel size##morph", calib.morph_kernel_size, 3, 15
            )
            calib.morph_kernel_size = calib.morph_kernel_size | 1  # ensure odd
        _, calib.spatial_consistency = imgui.checkbox("Spatial consistency", calib.spatial_consistency)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Reject pixels whose decoded coordinate differs\nfrom local median by more than the max deviation.")
        if calib.spatial_consistency:
            _, calib.consistency_max_diff = imgui.slider_float(
                "Max deviation (px)", calib.consistency_max_diff, 1.0, 20.0
            )
        _, calib.bit_threshold = imgui.slider_float(
            "Bit threshold", calib.bit_threshold, 0.0, 20.0
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Min |positive - negative| brightness per Gray code bit.\nRejects pixels at stripe boundaries where the camera\ncan't clearly see the pattern. Higher = stricter.")
        _, calib.fill_kernel_size = imgui.slider_int(
            "Fill kernel", calib.fill_kernel_size, 1, 31
        )
        calib.fill_kernel_size = calib.fill_kernel_size | 1  # ensure odd
        if imgui.is_item_hovered():
            imgui.set_tooltip("Gaussian splat size to fill gaps in projector space.\nLarger = more filled coverage, softer edges.\n1 = disabled (sparse mapping only).")
        imgui.separator()

        # Board pattern settings
        _, calib.skip_checkerboard = imgui.checkbox("Skip checkerboard (Gray codes only)", calib.skip_checkerboard)
        if calib.skip_checkerboard:
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.5, 1.0), "Only 2D correspondence will be calibrated")
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.5, 1.0), "(no intrinsics/extrinsics)")
        else:
            _, calib.board_cols = imgui.slider_int("Board cols", calib.board_cols, 3, 9)
            _, calib.board_rows = imgui.slider_int("Board rows", calib.board_rows, 3, 7)
            _, calib.square_px = imgui.slider_int("Square size (px)", calib.square_px, 30, 150)
            _, calib.board_white = imgui.slider_int("White level", calib.board_white, 128, 255)
            _, calib.board_black = imgui.slider_int("Black level", calib.board_black, 0, 127)

    imgui.separator()

    # ── Step 1: Projected Checkerboard ──────────────────────
    step1_color = _phase_color(phase, CalibrationPhase.CAMERA_INTRINSICS)
    if calib.skip_checkerboard:
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Step 1: Checkerboard (Skipped)")
    else:
        imgui.text_colored(step1_color, "Step 1: Projected Checkerboard")

    if phase == CalibrationPhase.CAMERA_INTRINSICS:
        calibrator = app._procam_calibrator
        if calibrator is not None:
            pos_idx = calibrator.current_position_idx
            total = calibrator.total_positions
            captures = calibrator.capture_count
            min_captures = 3

            # Live detection indicator
            if app._calib_corners_found:
                imgui.text_colored(imgui.ImVec4(0.0, 1.0, 0.3, 1.0), "Corners detected")
            else:
                imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.0, 1.0), "Searching...")

            # Capture count with color: red if below min, yellow if at min, green if good
            cap_color = (
                imgui.ImVec4(0.0, 1.0, 0.3, 1.0) if captures >= 5 else
                imgui.ImVec4(1.0, 1.0, 0.0, 1.0) if captures >= min_captures else
                imgui.ImVec4(1.0, 0.5, 0.3, 1.0)
            )
            imgui.text(f"Position {min(pos_idx + 1, total)}/{total}")
            imgui.same_line()
            imgui.text_colored(cap_color, f"Captured: {captures} (min {min_captures})")
            imgui.progress_bar(
                pos_idx / max(total, 1), imgui.ImVec2(-1, 0),
            )
    elif phase.value > CalibrationPhase.CAMERA_INTRINSICS.value:
        if calib.skip_checkerboard:
            imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Skipped by user")
        else:
            cam_err = getattr(app, "_cam_reproj_error", None)
            proj_err = getattr(app, "_proj_reproj_error", None)
            if cam_err is not None:
                imgui.text(f"Cam err: {cam_err:.2f}px  Proj err: {proj_err:.2f}px" if proj_err else f"Cam err: {cam_err:.2f}px")
            imgui.text_colored(imgui.ImVec4(0.0, 1.0, 0.3, 1.0), "Done")
    else:
        if calib.skip_checkerboard:
            imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Will be skipped")
        else:
            imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Pending")

    imgui.separator()

    # ── Step 2: Gray Codes ──────────────────────────────────
    step2_color = _phase_color(phase, CalibrationPhase.GRAY_CODES)
    imgui.text_colored(step2_color, "Step 2: Structured Light")

    if phase == CalibrationPhase.GRAY_CODES:
        runner = app._calib_runner
        if runner is not None:
            idx = app._calib_pattern_idx
            total = runner.total_patterns
            frames_target = runner.capture_frames
            frames_done = app._calib_frame_count
            if frames_target > 1:
                imgui.text(f"Pattern {idx}/{total}  frame {frames_done}/{frames_target}")
            else:
                imgui.text(f"Pattern {idx}/{total}")
            imgui.progress_bar(state.calibration_progress, imgui.ImVec2(-1, 0))
    elif phase.value > CalibrationPhase.GRAY_CODES.value:
        # Show decode stats
        if app._decode_valid_cam_pixels > 0:
            cam_pct = app._decode_cam_coverage_pct
            proj_pct = app._decode_proj_coverage_pct
            # Color code: green if good (>20%), yellow if marginal (5-20%), red if poor (<5%)
            cam_color = (
                imgui.ImVec4(0.0, 1.0, 0.3, 1.0) if cam_pct > 20 else
                imgui.ImVec4(1.0, 1.0, 0.0, 1.0) if cam_pct > 5 else
                imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
            )
            proj_color = (
                imgui.ImVec4(0.0, 1.0, 0.3, 1.0) if proj_pct > 20 else
                imgui.ImVec4(1.0, 1.0, 0.0, 1.0) if proj_pct > 5 else
                imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
            )
            imgui.text_colored(cam_color,
                f"Camera: {app._decode_valid_cam_pixels:,}/{app._decode_total_cam_pixels:,} px ({cam_pct:.1f}%)")
            sparse_proj = getattr(app, "_decode_sparse_proj_pixels", 0)
            if sparse_proj > 0 and sparse_proj != app._decode_valid_proj_pixels:
                imgui.text_colored(proj_color,
                    f"Projector: {app._decode_valid_proj_pixels:,}/{app._decode_total_proj_pixels:,} px ({proj_pct:.1f}%) [sparse: {sparse_proj:,}]")
            else:
                imgui.text_colored(proj_color,
                    f"Projector: {app._decode_valid_proj_pixels:,}/{app._decode_total_proj_pixels:,} px ({proj_pct:.1f}%)")
            if cam_pct < 5 or proj_pct < 5:
                imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
                    "Poor coverage! Need >5% (min), >20% recommended")
        imgui.text_colored(imgui.ImVec4(0.0, 1.0, 0.3, 1.0), "Done")
    else:
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Pending")

    imgui.separator()

    # ── Step 3: Stereo + Results ────────────────────────────
    step3_color = _phase_color(phase, CalibrationPhase.COMPUTING)
    imgui.text_colored(step3_color, "Step 3: Compute Stereo")

    if phase == CalibrationPhase.COMPUTING:
        imgui.text("Computing...")
    elif phase == CalibrationPhase.DONE:
        stereo_err = getattr(app, "_stereo_reproj_error", None)
        if stereo_err is not None:
            imgui.text(f"Stereo error: {stereo_err:.3f} px")
        has_reproj = app._depth_reprojector is not None
        if has_reproj:
            imgui.text_colored(imgui.ImVec4(0.0, 1.0, 0.3, 1.0), "3D reprojection active")
            # Show depth map info
            depth_frame = state.depth_frame
            if depth_frame is not None:
                h, w = depth_frame.shape
                d_min, d_max = depth_frame.min(), depth_frame.max()
                imgui.text(f"  Depth: {w}x{h}  range: [{d_min:.3f}, {d_max:.3f}]")
            # Button to manually save/open reprojection preview
            if imgui.button("Save Reprojection Preview"):
                app._save_and_open_reprojection_preview()
        else:
            imgui.text("2D remap only (no depth reprojection)")
        imgui.text_colored(imgui.ImVec4(0.0, 1.0, 0.3, 1.0), "Calibration complete!")
    else:
        imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), "Pending")

    imgui.separator()

    # ── Status / Load ───────────────────────────────────────
    if state.calibration_file:
        imgui.text_wrapped(f"File: {state.calibration_file}")

    has_full = state.procam.K_cam is not None
    has_map = state.calib_map_x is not None
    status = "Full ProCam" if has_full else ("2D maps" if has_map else "None")
    imgui.text(f"Loaded: {status}")

    if imgui.button("Load From Disk"):
        app.load_calibration_file()


def _phase_color(
    current: CalibrationPhase, target: CalibrationPhase,
) -> imgui.ImVec4:
    """Color for a calibration step label based on current phase."""
    if current == target:
        return imgui.ImVec4(1.0, 1.0, 0.0, 1.0)  # Yellow = active
    elif current.value > target.value:
        return imgui.ImVec4(0.0, 1.0, 0.3, 1.0)  # Green = done
    else:
        return imgui.ImVec4(0.7, 0.7, 0.7, 1.0)  # Gray = pending


def _render_effects_ui(state) -> None:
    """Render the effects section of the UI."""
    if not imgui.collapsing_header("Effects"):
        return

    eff = state.effects

    # -- Noise
    if imgui.tree_node("Fractal Noise"):
        _, eff.noise_enabled = imgui.checkbox("Enable##noise", eff.noise_enabled)
        _, eff.noise_intensity = imgui.slider_float("Intensity##noise", eff.noise_intensity, 0.0, 1.0)
        _, eff.noise_scale = imgui.slider_float("Scale##noise", eff.noise_scale, 0.5, 20.0)
        _, eff.noise_octaves = imgui.slider_int("Octaves##noise", eff.noise_octaves, 1, 8)
        _, eff.noise_speed = imgui.slider_float("Speed##noise", eff.noise_speed, 0.0, 5.0)
        imgui.tree_pop()

    # -- Flow warp
    if imgui.tree_node("Flow Warp"):
        _, eff.flow_enabled = imgui.checkbox("Enable##flow", eff.flow_enabled)
        _, eff.flow_intensity = imgui.slider_float("Intensity##flow", eff.flow_intensity, 0.0, 1.0)
        _, eff.flow_scale = imgui.slider_float("Scale##flow", eff.flow_scale, 0.5, 20.0)
        _, eff.flow_speed = imgui.slider_float("Speed##flow", eff.flow_speed, 0.0, 5.0)
        imgui.tree_pop()

    # -- Pulse
    if imgui.tree_node("Pulse"):
        _, eff.pulse_enabled = imgui.checkbox("Enable##pulse", eff.pulse_enabled)
        _, eff.pulse_speed = imgui.slider_float("Speed##pulse", eff.pulse_speed, 0.01, 5.0)
        _, eff.pulse_amount = imgui.slider_float("Amount##pulse", eff.pulse_amount, 0.0, 1.0)
        imgui.tree_pop()

    # -- Wave
    if imgui.tree_node("Wave"):
        _, eff.wave_enabled = imgui.checkbox("Enable##wave", eff.wave_enabled)
        _, eff.wave_frequency = imgui.slider_float("Frequency##wave", eff.wave_frequency, 0.5, 20.0)
        _, eff.wave_amplitude = imgui.slider_float("Amplitude##wave", eff.wave_amplitude, 0.0, 0.3)
        _, eff.wave_speed = imgui.slider_float("Speed##wave", eff.wave_speed, 0.0, 5.0)
        _, eff.wave_direction = imgui.slider_float("Direction##wave", eff.wave_direction, 0.0, 360.0)
        imgui.tree_pop()

    # -- Shockwave
    if imgui.tree_node("Shockwave"):
        _, eff.shockwave_enabled = imgui.checkbox("Enable##shock", eff.shockwave_enabled)
        _, eff.shockwave_origin_x = imgui.slider_float("Origin X##shock", eff.shockwave_origin_x, 0.0, 1.0)
        _, eff.shockwave_origin_y = imgui.slider_float("Origin Y##shock", eff.shockwave_origin_y, 0.0, 1.0)
        _, eff.shockwave_speed = imgui.slider_float("Speed##shock", eff.shockwave_speed, 0.1, 3.0)
        _, eff.shockwave_thickness = imgui.slider_float("Thickness##shock", eff.shockwave_thickness, 0.01, 0.5)
        _, eff.shockwave_strength = imgui.slider_float("Strength##shock", eff.shockwave_strength, 0.0, 1.0)
        _, eff.shockwave_decay = imgui.slider_float("Decay##shock", eff.shockwave_decay, 0.0, 5.0)
        _, eff.shockwave_interval = imgui.slider_float("Interval##shock", eff.shockwave_interval, 0.5, 10.0)
        imgui.tree_pop()

    # -- Wobble
    if imgui.tree_node("Wobble"):
        _, eff.wobble_enabled = imgui.checkbox("Enable##wobble", eff.wobble_enabled)
        _, eff.wobble_intensity = imgui.slider_float("Intensity##wobble", eff.wobble_intensity, 0.0, 0.5)
        _, eff.wobble_speed = imgui.slider_float("Speed##wobble", eff.wobble_speed, 0.0, 5.0)
        imgui.tree_pop()

    # -- Edges
    if imgui.tree_node("Geometry Edges"):
        _, eff.edges_enabled = imgui.checkbox("Enable##edges", eff.edges_enabled)
        _, eff.edges_strength = imgui.slider_float("Strength##edges", eff.edges_strength, 0.0, 2.0)
        _, eff.edges_glow_width = imgui.slider_float("Glow width##edges", eff.edges_glow_width, 1.0, 10.0)
        _, eff.edges_pulse_speed = imgui.slider_float("Pulse speed##edges", eff.edges_pulse_speed, 0.0, 5.0)
        imgui.tree_pop()

    # -- Fog
    if imgui.tree_node("Depth Fog"):
        _, eff.fog_enabled = imgui.checkbox("Enable##fog", eff.fog_enabled)
        _, eff.fog_density = imgui.slider_float("Density##fog", eff.fog_density, 0.0, 1.0)
        _, eff.fog_near = imgui.slider_float("Near##fog", eff.fog_near, 0.0, 1.0)
        _, eff.fog_far = imgui.slider_float("Far##fog", eff.fog_far, 0.0, 1.0)
        _, eff.fog_animated = imgui.checkbox("Animated##fog", eff.fog_animated)
        _, eff.fog_speed = imgui.slider_float("Speed##fog", eff.fog_speed, 0.0, 5.0)
        imgui.tree_pop()

    # -- Zoom
    if imgui.tree_node("Radial Zoom"):
        _, eff.zoom_enabled = imgui.checkbox("Enable##zoom", eff.zoom_enabled)
        _, eff.zoom_origin_x = imgui.slider_float("Origin X##zoom", eff.zoom_origin_x, 0.0, 1.0)
        _, eff.zoom_origin_y = imgui.slider_float("Origin Y##zoom", eff.zoom_origin_y, 0.0, 1.0)
        _, eff.zoom_strength = imgui.slider_float("Strength##zoom", eff.zoom_strength, 0.0, 1.0)
        _, eff.zoom_speed = imgui.slider_float("Speed##zoom", eff.zoom_speed, 0.0, 5.0)
        imgui.tree_pop()

    # -- Kaleidoscope
    if imgui.tree_node("Kaleidoscope"):
        _, eff.kaleido_enabled = imgui.checkbox("Enable##kaleido", eff.kaleido_enabled)
        _, eff.kaleido_segments = imgui.slider_int("Segments##kaleido", eff.kaleido_segments, 2, 16)
        _, eff.kaleido_rotation = imgui.slider_float("Rotation##kaleido", eff.kaleido_rotation, 0.0, 6.28)
        _, eff.kaleido_spin_speed = imgui.slider_float("Spin speed##kaleido", eff.kaleido_spin_speed, 0.0, 5.0)
        imgui.tree_pop()
