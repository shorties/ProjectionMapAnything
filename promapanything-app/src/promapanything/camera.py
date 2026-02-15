"""Camera capture in a background thread."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .state import AppState


def enumerate_cameras(
    max_index: int = 10, skip_index: int = -1,
) -> list[tuple[int, str]]:
    """Probe camera indices and return a list of (index, label) for available cameras.

    Uses DirectShow backend on Windows to find virtual cameras (OBS, etc.).
    Call sparingly (only on explicit user request via Refresh button).

    Args:
        max_index: highest device index to probe (exclusive)
        skip_index: skip this index (e.g. the currently active camera)
    """
    import sys

    available: list[tuple[int, str]] = []
    seen_indices: set[int] = set()

    # On Windows, use DirectShow backend which finds OBS Virtual Camera
    backends = [cv2.CAP_DSHOW] if sys.platform == "win32" else []
    backends.append(cv2.CAP_ANY)  # fallback

    for backend in backends:
        for idx in range(max_index):
            if idx in seen_indices:
                continue
            if idx == skip_index:
                if idx not in seen_indices:
                    available.append((idx, f"[{idx}] Camera (active)"))
                    seen_indices.add(idx)
                continue
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                backend_name = "DSHOW" if backend == cv2.CAP_DSHOW else ""
                label = f"[{idx}] Camera ({w}x{h})"
                if backend_name:
                    label += f" [{backend_name}]"
                available.append((idx, label))
                seen_indices.add(idx)
                cap.release()
    return available


class CameraCapture:
    """Captures frames from a webcam in a background thread.

    Frames are stored in ``self.frame`` as (H, W, 3) BGR uint8 numpy arrays
    and also written to ``state.camera_frame`` for other modules to consume.
    """

    def __init__(self, state: AppState):
        self.state = state
        self.frame: np.ndarray | None = None
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self, device_idx: int = 0) -> None:
        import sys

        self.stop()
        # Try DirectShow first on Windows (needed for OBS Virtual Camera etc.)
        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(device_idx)
        else:
            self._cap = cv2.VideoCapture(device_idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device_idx}")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def capture_single(self) -> np.ndarray | None:
        """Capture a single frame synchronously (for calibration)."""
        if self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                return frame
        return None

    @staticmethod
    def _apply_brightness_contrast(
        frame: np.ndarray, brightness: float, contrast: float,
    ) -> np.ndarray:
        """Apply brightness and contrast adjustment.

        brightness: -1.0 to 1.0 (0 = no change)
        contrast: 0.5 to 3.0 (1.0 = no change)
        """
        if brightness == 0.0 and contrast == 1.0:
            return frame
        # contrast * pixel + brightness_offset
        f = frame.astype(np.float32)
        f = contrast * f + brightness * 255.0
        return np.clip(f, 0, 255).astype(np.uint8)

    def _capture_loop(self) -> None:
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if ret:
                cam = self.state.camera_settings
                frame = self._apply_brightness_contrast(
                    frame, cam.brightness, cam.contrast,
                )
                self.frame = frame
                self.state.camera_frame = frame
            else:
                time.sleep(0.01)
