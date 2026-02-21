"""Azure Kinect Exposure Control Utility

Live camera preview with full GUI controls for exposure, white balance,
brightness, contrast, saturation, sharpness, gain, and more.

Requirements:
    pip install pyk4a opencv-python numpy pillow

Usage:
    python tools/kinect_exposure.py
    python tools/kinect_exposure.py --resolution 1080p
    python tools/kinect_exposure.py --load settings.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from pyk4a import Config, PyK4A, ColorControlCommand, ColorControlMode
    from pyk4a import ColorResolution, DepthMode, FPS as K4aFPS
except ImportError:
    print("pyk4a not installed. Install with: pip install pyk4a")
    print("Also requires Azure Kinect SDK:")
    print("  https://learn.microsoft.com/en-us/azure/kinect-dk/sensor-sdk-download")
    sys.exit(1)


# ── Constants ──────────────────────────────────────────────────────

EXPOSURE_STEPS_US = [500, 1250, 2500, 8330, 16670, 33330]
EXPOSURE_LABELS = ["500us (1/2000s)", "1250us (1/800s)", "2500us (1/400s)",
                   "8330us (1/120s)", "16670us (1/60s)", "33330us (1/30s)"]

SETTINGS_FILE = "kinect_settings.json"

RESOLUTION_MAP = {
    "720p":  ColorResolution.RES_720P,
    "1080p": ColorResolution.RES_1080P,
    "1440p": ColorResolution.RES_1440P,
    "1536p": ColorResolution.RES_1536P,
    "2160p": ColorResolution.RES_2160P,
    "3072p": ColorResolution.RES_3072P,
}

# (ColorControlCommand, label, min, max, default, step)
SLIDER_CONTROLS = [
    (ColorControlCommand.BRIGHTNESS,  "Brightness",  0, 255, 128, 1),
    (ColorControlCommand.CONTRAST,    "Contrast",    0, 10,  5,   1),
    (ColorControlCommand.SATURATION,  "Saturation",  0, 63,  32,  1),
    (ColorControlCommand.SHARPNESS,   "Sharpness",   0, 4,   2,   1),
    (ColorControlCommand.GAIN,        "Gain",        0, 255, 0,   1),
]


# ── Kinect Controller ─────────────────────────────────────────────

class KinectController:
    """Wraps pyk4a with typed accessors for every color control."""

    def __init__(self, resolution: str = "720p", device_id: int = 0):
        color_res = RESOLUTION_MAP.get(resolution, ColorResolution.RES_720P)
        self.k4a = PyK4A(
            Config(color_resolution=color_res,
                   depth_mode=DepthMode.NFOV_UNBINNED,
                   camera_fps=K4aFPS.FPS_30),
            device_id=device_id,
        )
        self.k4a.start()

    def get(self, cmd: ColorControlCommand) -> tuple[bool, int]:
        """Return (is_auto, value)."""
        try:
            mode, val = self.k4a._device.get_color_control(cmd)
            return mode == ColorControlMode.AUTO, val
        except Exception:
            return True, 0

    def set(self, cmd: ColorControlCommand, value: int, auto: bool = False):
        mode = ColorControlMode.AUTO if auto else ColorControlMode.MANUAL
        try:
            self.k4a._device.set_color_control(cmd, mode, value)
        except Exception as e:
            print(f"  set {cmd.name} failed: {e}")

    def capture_color(self) -> np.ndarray | None:
        try:
            cap = self.k4a.get_capture()
            if cap.color is not None:
                return cv2.cvtColor(cap.color[:, :, :3], cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        return None

    def stop(self):
        self.k4a.stop()


# ── GUI App ────────────────────────────────────────────────────────

class App:
    PREVIEW_W = 800
    PREVIEW_H = 450
    UPDATE_MS = 33  # ~30 fps

    def __init__(self, root: tk.Tk, ctrl: KinectController):
        self.root = root
        self.ctrl = ctrl
        self.running = True
        self._fps_count = 0
        self._fps_time = time.monotonic()
        self._fps = 0.0
        self._photo: ImageTk.PhotoImage | None = None

        root.title("Azure Kinect Exposure Control")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#1e1e1e", foreground="#e0e0e0",
                        troughcolor="#2d2d2d", fieldbackground="#2d2d2d")
        style.configure("TLabel", background="#1e1e1e", foreground="#e0e0e0")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabelframe", background="#1e1e1e", foreground="#aaa")
        style.configure("TLabelframe.Label", background="#1e1e1e", foreground="#aaa")
        style.configure("TButton", padding=4)
        style.configure("Toggle.TButton", padding=4)
        style.configure("Auto.TLabel", foreground="#4ec9b0")
        style.configure("Manual.TLabel", foreground="#d19a66")
        style.configure("Status.TLabel", background="#181818", foreground="#888",
                        padding=(8, 4))
        style.configure("TScale", background="#1e1e1e")
        style.map("TButton",
                  background=[("active", "#3a3a3a"), ("!active", "#2d2d2d")])

        # ── Layout: left = preview, right = controls ──
        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(main, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right.pack_propagate(False)

        # ── Preview ──
        self.canvas = tk.Canvas(left, bg="#000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ── Scrollable controls panel ──
        canvas_scroll = tk.Canvas(right, bg="#1e1e1e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(right, orient=tk.VERTICAL, command=canvas_scroll.yview)
        self.controls_frame = ttk.Frame(canvas_scroll)
        self.controls_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")),
        )
        canvas_scroll.create_window((0, 0), window=self.controls_frame, anchor=tk.NW)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas_scroll.yview_scroll(-1 * (event.delta // 120), "units")
        canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_controls(self.controls_frame)

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Starting...")
        status = ttk.Label(root, textvariable=self.status_var, style="Status.TLabel")
        status.pack(fill=tk.X, side=tk.BOTTOM)

        # Read initial state from camera
        self._sync_from_camera()

        # Start update loop
        self.root.after(self.UPDATE_MS, self._update)

    # ── Build control widgets ──────────────────────────────────────

    def _build_controls(self, parent: ttk.Frame):
        # ── Exposure ──
        exp_frame = ttk.LabelFrame(parent, text=" Exposure ", padding=8)
        exp_frame.pack(fill=tk.X, pady=(0, 6))

        self.auto_exp_var = tk.BooleanVar(value=True)
        self.auto_exp_label = ttk.Label(exp_frame, text="AUTO", style="Auto.TLabel")
        self.auto_exp_label.pack(anchor=tk.W)

        btn_row = ttk.Frame(exp_frame)
        btn_row.pack(fill=tk.X, pady=(2, 6))
        ttk.Button(btn_row, text="Auto", width=8,
                   command=lambda: self._toggle_auto_exposure(True)).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="Manual", width=8,
                   command=lambda: self._toggle_auto_exposure(False)).pack(side=tk.LEFT)

        self.exp_var = tk.IntVar(value=3)
        self.exp_combo = ttk.Combobox(exp_frame, values=EXPOSURE_LABELS,
                                       state="disabled", width=24)
        self.exp_combo.current(3)
        self.exp_combo.pack(fill=tk.X, pady=(0, 4))
        self.exp_combo.bind("<<ComboboxSelected>>", self._on_exposure_change)

        self.exp_readout = ttk.Label(exp_frame, text="8330 us")
        self.exp_readout.pack(anchor=tk.W)

        # ── White Balance ──
        wb_frame = ttk.LabelFrame(parent, text=" White Balance ", padding=8)
        wb_frame.pack(fill=tk.X, pady=(0, 6))

        self.auto_wb_var = tk.BooleanVar(value=True)
        self.auto_wb_label = ttk.Label(wb_frame, text="AUTO", style="Auto.TLabel")
        self.auto_wb_label.pack(anchor=tk.W)

        btn_row2 = ttk.Frame(wb_frame)
        btn_row2.pack(fill=tk.X, pady=(2, 6))
        ttk.Button(btn_row2, text="Auto", width=8,
                   command=lambda: self._toggle_auto_wb(True)).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row2, text="Manual", width=8,
                   command=lambda: self._toggle_auto_wb(False)).pack(side=tk.LEFT)

        self.wb_var = tk.IntVar(value=4500)
        self.wb_scale = ttk.Scale(wb_frame, from_=2500, to=12500,
                                   variable=self.wb_var, orient=tk.HORIZONTAL,
                                   command=self._on_wb_change, state="disabled")
        self.wb_scale.pack(fill=tk.X, pady=(0, 2))
        self.wb_readout = ttk.Label(wb_frame, text="4500 K")
        self.wb_readout.pack(anchor=tk.W)

        # ── Gain ──
        gain_frame = ttk.LabelFrame(parent, text=" Gain ", padding=8)
        gain_frame.pack(fill=tk.X, pady=(0, 6))
        self.gain_var = tk.IntVar(value=0)
        self.gain_scale = ttk.Scale(gain_frame, from_=0, to=255,
                                     variable=self.gain_var, orient=tk.HORIZONTAL,
                                     command=lambda v: self._on_slider(
                                         ColorControlCommand.GAIN, self.gain_var, self.gain_readout, ""))
        self.gain_scale.pack(fill=tk.X, pady=(0, 2))
        self.gain_readout = ttk.Label(gain_frame, text="0")
        self.gain_readout.pack(anchor=tk.W)

        # ── Image Adjustments ──
        adj_frame = ttk.LabelFrame(parent, text=" Image Adjustments ", padding=8)
        adj_frame.pack(fill=tk.X, pady=(0, 6))

        self._slider_vars: dict[ColorControlCommand, tuple[tk.IntVar, ttk.Label]] = {}
        for cmd, label, lo, hi, default, step in SLIDER_CONTROLS:
            if cmd == ColorControlCommand.GAIN:
                continue  # already built above
            row = ttk.Frame(adj_frame)
            row.pack(fill=tk.X, pady=(0, 6))
            ttk.Label(row, text=label, width=12).pack(anchor=tk.W)
            var = tk.IntVar(value=default)
            readout = ttk.Label(row, text=str(default), width=6)
            readout.pack(side=tk.RIGHT)
            scale = ttk.Scale(row, from_=lo, to=hi, variable=var,
                              orient=tk.HORIZONTAL,
                              command=lambda v, c=cmd, vr=var, r=readout: self._on_slider(c, vr, r, ""))
            scale.pack(fill=tk.X, expand=True)
            self._slider_vars[cmd] = (var, readout)

        # ── Powerline ──
        pl_frame = ttk.LabelFrame(parent, text=" Powerline Frequency ", padding=8)
        pl_frame.pack(fill=tk.X, pady=(0, 6))
        self.powerline_var = tk.IntVar(value=2)
        ttk.Radiobutton(pl_frame, text="50 Hz", variable=self.powerline_var, value=1,
                        command=self._on_powerline).pack(anchor=tk.W)
        ttk.Radiobutton(pl_frame, text="60 Hz", variable=self.powerline_var, value=2,
                        command=self._on_powerline).pack(anchor=tk.W)

        # ── Backlight Compensation ──
        bl_frame = ttk.LabelFrame(parent, text=" Backlight Compensation ", padding=8)
        bl_frame.pack(fill=tk.X, pady=(0, 6))
        self.backlight_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bl_frame, text="Enabled", variable=self.backlight_var,
                        command=self._on_backlight).pack(anchor=tk.W)

        # ── Buttons ──
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(6, 0))

        ttk.Button(btn_frame, text="Save Settings", command=self._save).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Load Settings", command=self._load).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Reset Defaults", command=self._reset).pack(fill=tk.X, pady=2)

        # Add gain to slider_vars so _sync_from_camera can reach it
        self._slider_vars[ColorControlCommand.GAIN] = (self.gain_var, self.gain_readout)

    # ── Event handlers ─────────────────────────────────────────────

    def _toggle_auto_exposure(self, auto: bool):
        self.auto_exp_var.set(auto)
        if auto:
            self.ctrl.set(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, 0, auto=True)
            self.exp_combo.configure(state="disabled")
            self.auto_exp_label.configure(text="AUTO", style="Auto.TLabel")
        else:
            idx = self.exp_combo.current()
            self.ctrl.set(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE,
                          EXPOSURE_STEPS_US[idx], auto=False)
            self.exp_combo.configure(state="readonly")
            self.auto_exp_label.configure(text="MANUAL", style="Manual.TLabel")

    def _on_exposure_change(self, _event=None):
        if not self.auto_exp_var.get():
            idx = self.exp_combo.current()
            us = EXPOSURE_STEPS_US[idx]
            self.ctrl.set(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, us, auto=False)
            self.exp_readout.configure(text=f"{us} us")

    def _toggle_auto_wb(self, auto: bool):
        self.auto_wb_var.set(auto)
        if auto:
            self.ctrl.set(ColorControlCommand.WHITEBALANCE, 0, auto=True)
            self.wb_scale.configure(state="disabled")
            self.auto_wb_label.configure(text="AUTO", style="Auto.TLabel")
        else:
            val = (self.wb_var.get() // 10) * 10
            self.ctrl.set(ColorControlCommand.WHITEBALANCE, val, auto=False)
            self.wb_scale.configure(state="!disabled")
            self.auto_wb_label.configure(text="MANUAL", style="Manual.TLabel")

    def _on_wb_change(self, _val=None):
        val = (self.wb_var.get() // 10) * 10
        self.wb_readout.configure(text=f"{val} K")
        if not self.auto_wb_var.get():
            self.ctrl.set(ColorControlCommand.WHITEBALANCE, val, auto=False)

    def _on_slider(self, cmd: ColorControlCommand, var: tk.IntVar, readout: ttk.Label, _val):
        v = var.get()
        readout.configure(text=str(v))
        self.ctrl.set(cmd, v)

    def _on_powerline(self):
        self.ctrl.set(ColorControlCommand.POWERLINE_FREQUENCY, self.powerline_var.get())

    def _on_backlight(self):
        self.ctrl.set(ColorControlCommand.BACKLIGHT_COMPENSATION,
                      1 if self.backlight_var.get() else 0)

    # ── Sync UI ← camera ──────────────────────────────────────────

    def _sync_from_camera(self):
        """Read all current values from camera and update widgets."""
        # Exposure
        is_auto, val = self.ctrl.get(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)
        self.auto_exp_var.set(is_auto)
        if is_auto:
            self.exp_combo.configure(state="disabled")
            self.auto_exp_label.configure(text="AUTO", style="Auto.TLabel")
        else:
            self.exp_combo.configure(state="readonly")
            self.auto_exp_label.configure(text="MANUAL", style="Manual.TLabel")
        # Find closest step
        idx = min(range(len(EXPOSURE_STEPS_US)),
                  key=lambda i: abs(EXPOSURE_STEPS_US[i] - val))
        self.exp_combo.current(idx)
        self.exp_readout.configure(text=f"{val} us")

        # White balance
        is_auto, val = self.ctrl.get(ColorControlCommand.WHITEBALANCE)
        self.auto_wb_var.set(is_auto)
        if is_auto:
            self.wb_scale.configure(state="disabled")
            self.auto_wb_label.configure(text="AUTO", style="Auto.TLabel")
        else:
            self.wb_scale.configure(state="!disabled")
            self.auto_wb_label.configure(text="MANUAL", style="Manual.TLabel")
        self.wb_var.set(val)
        self.wb_readout.configure(text=f"{val} K")

        # Sliders
        for cmd, (var, readout) in self._slider_vars.items():
            _, v = self.ctrl.get(cmd)
            var.set(v)
            readout.configure(text=str(v))

        # Powerline
        _, v = self.ctrl.get(ColorControlCommand.POWERLINE_FREQUENCY)
        self.powerline_var.set(v)

        # Backlight
        _, v = self.ctrl.get(ColorControlCommand.BACKLIGHT_COMPENSATION)
        self.backlight_var.set(bool(v))

    # ── Save / Load / Reset ────────────────────────────────────────

    def _get_all_settings(self) -> dict:
        settings = {}
        for cmd, name in [
            (ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, "exposure"),
            (ColorControlCommand.BRIGHTNESS, "brightness"),
            (ColorControlCommand.CONTRAST, "contrast"),
            (ColorControlCommand.SATURATION, "saturation"),
            (ColorControlCommand.SHARPNESS, "sharpness"),
            (ColorControlCommand.GAIN, "gain"),
            (ColorControlCommand.WHITEBALANCE, "whitebalance"),
            (ColorControlCommand.BACKLIGHT_COMPENSATION, "backlight_comp"),
            (ColorControlCommand.POWERLINE_FREQUENCY, "powerline"),
        ]:
            is_auto, val = self.ctrl.get(cmd)
            settings[name] = {"value": val, "auto": is_auto}
        return settings

    def _save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=SETTINGS_FILE,
        )
        if not path:
            return
        settings = self._get_all_settings()
        Path(path).write_text(json.dumps(settings, indent=2))
        self.status_var.set(f"Saved to {Path(path).name}")

    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json")],
            initialfile=SETTINGS_FILE,
        )
        if not path or not Path(path).exists():
            return
        settings = json.loads(Path(path).read_text())

        # Apply exposure
        exp = settings.get("exposure", {})
        if exp.get("auto", True):
            self._toggle_auto_exposure(True)
        else:
            self._toggle_auto_exposure(False)
            val = exp.get("value", 8330)
            idx = min(range(len(EXPOSURE_STEPS_US)),
                      key=lambda i: abs(EXPOSURE_STEPS_US[i] - val))
            self.exp_combo.current(idx)
            self.ctrl.set(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE,
                          EXPOSURE_STEPS_US[idx], auto=False)

        # Apply white balance
        wb = settings.get("whitebalance", {})
        if wb.get("auto", True):
            self._toggle_auto_wb(True)
        else:
            self._toggle_auto_wb(False)
            self.wb_var.set(wb.get("value", 4500))
            self._on_wb_change()

        # Simple sliders
        mapping = {
            "brightness": ColorControlCommand.BRIGHTNESS,
            "contrast": ColorControlCommand.CONTRAST,
            "saturation": ColorControlCommand.SATURATION,
            "sharpness": ColorControlCommand.SHARPNESS,
            "gain": ColorControlCommand.GAIN,
        }
        for name, cmd in mapping.items():
            if name in settings:
                v = settings[name]["value"]
                self.ctrl.set(cmd, v)
                if cmd in self._slider_vars:
                    var, readout = self._slider_vars[cmd]
                    var.set(v)
                    readout.configure(text=str(v))

        # Powerline
        if "powerline" in settings:
            v = settings["powerline"]["value"]
            self.powerline_var.set(v)
            self.ctrl.set(ColorControlCommand.POWERLINE_FREQUENCY, v)

        # Backlight
        if "backlight_comp" in settings:
            v = settings["backlight_comp"]["value"]
            self.backlight_var.set(bool(v))
            self.ctrl.set(ColorControlCommand.BACKLIGHT_COMPENSATION, v)

        self.status_var.set(f"Loaded from {Path(path).name}")

    def _reset(self):
        self._toggle_auto_exposure(True)
        self._toggle_auto_wb(True)
        defaults = {
            ColorControlCommand.BRIGHTNESS: 128,
            ColorControlCommand.CONTRAST: 5,
            ColorControlCommand.SATURATION: 32,
            ColorControlCommand.SHARPNESS: 2,
            ColorControlCommand.GAIN: 0,
        }
        for cmd, val in defaults.items():
            self.ctrl.set(cmd, val)
            if cmd in self._slider_vars:
                var, readout = self._slider_vars[cmd]
                var.set(val)
                readout.configure(text=str(val))

        self.powerline_var.set(2)
        self.ctrl.set(ColorControlCommand.POWERLINE_FREQUENCY, 2)
        self.backlight_var.set(False)
        self.ctrl.set(ColorControlCommand.BACKLIGHT_COMPENSATION, 0)
        self.exp_combo.current(3)
        self.status_var.set("Reset to defaults")

    # ── Main loop ──────────────────────────────────────────────────

    def _update(self):
        if not self.running:
            return

        frame = self.ctrl.capture_color()
        if frame is not None:
            # FPS
            self._fps_count += 1
            now = time.monotonic()
            if now - self._fps_time >= 1.0:
                self._fps = self._fps_count / (now - self._fps_time)
                self._fps_count = 0
                self._fps_time = now

            # Fit frame to canvas
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 1 and ch > 1:
                fh, fw = frame.shape[:2]
                scale = min(cw / fw, ch / fh)
                nw, nh = int(fw * scale), int(fh * scale)
                resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                img = Image.fromarray(resized)
                self._photo = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.create_image(cw // 2, ch // 2, image=self._photo,
                                         anchor=tk.CENTER)

            # Update status with live readouts
            is_auto_exp, exp_val = self.ctrl.get(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE)
            _, gain_val = self.ctrl.get(ColorControlCommand.GAIN)
            is_auto_wb, wb_val = self.ctrl.get(ColorControlCommand.WHITEBALANCE)
            exp_str = f"Exp: AUTO {exp_val}us" if is_auto_exp else f"Exp: {exp_val}us"
            wb_str = f"WB: AUTO {wb_val}K" if is_auto_wb else f"WB: {wb_val}K"
            self.status_var.set(
                f"{exp_str}  |  {wb_str}  |  Gain: {gain_val}  |  {self._fps:.0f} fps"
            )

            # Update exposure readout when in auto mode
            if is_auto_exp:
                self.exp_readout.configure(text=f"{exp_val} us")
            if is_auto_wb:
                self.wb_readout.configure(text=f"{wb_val} K")

        self.root.after(self.UPDATE_MS, self._update)

    def _on_close(self):
        self.running = False
        self.ctrl.stop()
        self.root.destroy()


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Azure Kinect Exposure Control")
    parser.add_argument("--resolution", default="720p",
                        choices=list(RESOLUTION_MAP.keys()),
                        help="Color camera resolution (default: 720p)")
    parser.add_argument("--device", type=int, default=0,
                        help="Device index if multiple Kinects (default: 0)")
    parser.add_argument("--load", type=str, default=None,
                        help="Load settings from JSON file on startup")
    args = parser.parse_args()

    ctrl = KinectController(resolution=args.resolution, device_id=args.device)

    root = tk.Tk()
    root.geometry("1200x700")
    root.minsize(800, 500)
    app = App(root, ctrl)

    if args.load:
        # Defer load until mainloop starts
        root.after(100, lambda: app._load_file(args.load))

    root.mainloop()


if __name__ == "__main__":
    main()
