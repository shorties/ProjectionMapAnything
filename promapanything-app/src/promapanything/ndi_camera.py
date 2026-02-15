"""NDI camera receiver — receive video from NDI sources (e.g. iPhone NDI HX Camera).

Uses cyndilib for NDI SDK access. Optional dependency — if not installed,
NDI features are silently unavailable.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .state import AppState

logger = logging.getLogger(__name__)

# Check if cyndilib is available
try:
    from cyndilib.finder import Finder
    from cyndilib.receiver import Receiver, ReceiveFrameType
    from cyndilib.video_frame import VideoRecvFrame
    from cyndilib.wrapper.ndi_structs import FourCC
    NDI_AVAILABLE = True
except ImportError:
    NDI_AVAILABLE = False


def ndi_available() -> bool:
    return NDI_AVAILABLE


def discover_ndi_sources(timeout: float = 4.0) -> list[str]:
    """Discover NDI sources on the network.

    Returns a list of source names (e.g. ['IPHONE (NDI HX Camera)']).
    Polls multiple times during the timeout period for more reliable discovery.
    """
    if not NDI_AVAILABLE:
        return []
    try:
        finder = Finder()
        finder.open()
        names = set()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            for name in finder.get_source_names():
                names.add(name)
        finder.close()
        result = sorted(names)
        logger.info("NDI discovery found %d sources: %s", len(result), result)
        return result
    except Exception as e:
        logger.warning("NDI source discovery failed: %s", e)
        return []


class NDICamera:
    """Receives video frames from an NDI source in a background thread.

    Writes frames to state.camera_frame, same as CameraCapture.
    """

    def __init__(self, state: AppState):
        self.state = state
        self.frame: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._source_name: str = ""
        self.connected = False
        self.width = 0
        self.height = 0
        self._format_logged = False

    def start(self, source_name: str) -> None:
        """Start receiving from the named NDI source."""
        if not NDI_AVAILABLE:
            raise RuntimeError("cyndilib not installed — NDI unavailable")
        self.stop()
        self._source_name = source_name
        self._running = True
        self._format_logged = False
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.connected = False

    def _receive_loop(self) -> None:
        """Background loop: find source, connect, receive frames."""
        finder = None
        receiver = None
        try:
            # Find the source
            finder = Finder()
            finder.open()

            source = None
            for _ in range(20):  # try for up to 10 seconds
                if not self._running:
                    return
                time.sleep(0.5)
                names = finder.get_source_names()
                for name in names:
                    if name == self._source_name:
                        source = name
                        break
                if source is not None:
                    break

            if source is None:
                logger.error("NDI source '%s' not found", self._source_name)
                return

            src_obj = finder.get_source(source)

            # Create receiver and connect — request highest bandwidth for full resolution
            from cyndilib.wrapper.ndi_recv import RecvBandwidth
            receiver = Receiver(bandwidth=RecvBandwidth.highest)
            video_frame = VideoRecvFrame()
            receiver.set_video_frame(video_frame)
            receiver.connect_to(src_obj)

            logger.info("NDI receiver connected to '%s'", self._source_name)
            self.connected = True

            while self._running:
                try:
                    result = receiver.receive(
                        ReceiveFrameType.recv_video, 100,
                    )
                    if result == ReceiveFrameType.recv_video:
                        xres = video_frame.xres
                        yres = video_frame.yres
                        if xres <= 0 or yres <= 0:
                            continue

                        self.width = xres
                        self.height = yres

                        fourcc = video_frame.fourcc
                        stride = video_frame.get_line_stride()

                        with video_frame.read_lock:
                            raw = bytes(video_frame.current_frame_data)

                        data = np.frombuffer(raw, dtype=np.uint8)

                        if not self._format_logged:
                            logger.info("NDI frame: %dx%d fourcc=%s stride=%d len=%d",
                                       xres, yres, fourcc, stride, len(data))
                            self._format_logged = True

                        if fourcc == FourCC.UYVY:
                            padded = data[:yres * stride].reshape(yres, stride)
                            uyvy = padded[:, :xres * 2]  # 2D: (H, W*2)
                            bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
                        elif fourcc in (FourCC.BGRA, FourCC.BGRX):
                            padded = data[:yres * stride].reshape(yres, stride)
                            bgra = padded[:, :xres * 4].reshape(yres, xres, 4)
                            bgr = bgra[:, :, :3].copy()
                        elif fourcc in (FourCC.RGBA, FourCC.RGBX):
                            padded = data[:yres * stride].reshape(yres, stride)
                            rgba = padded[:, :xres * 4].reshape(yres, xres, 4)
                            bgr = rgba[:, :, :3][:, :, ::-1].copy()  # RGB→BGR
                        elif fourcc == FourCC.NV12:
                            # NV12: Y plane (yres rows) + UV interleaved (yres/2 rows)
                            expected = stride * yres * 3 // 2
                            nv12 = data[:expected].reshape(yres * 3 // 2, stride)
                            # Crop to actual width
                            nv12_cropped = nv12[:, :xres]
                            bgr = cv2.cvtColor(
                                nv12_cropped.reshape(yres * 3 // 2, xres),
                                cv2.COLOR_YUV2BGR_NV12,
                            )
                        elif fourcc == FourCC.I420:
                            expected = stride * yres * 3 // 2
                            i420 = data[:expected].reshape(yres * 3 // 2, stride)
                            i420_cropped = i420[:, :xres]
                            bgr = cv2.cvtColor(
                                i420_cropped.reshape(yres * 3 // 2, xres),
                                cv2.COLOR_YUV2BGR_I420,
                            )
                        elif fourcc == FourCC.P216:
                            # P216: 16-bit 4:2:2 planar — take high bytes
                            # Y plane: yres * xres * 2 bytes, then UV
                            y16 = data[:yres * xres * 2].view(np.uint16).reshape(yres, xres)
                            y8 = (y16 >> 8).astype(np.uint8)
                            bgr = cv2.cvtColor(y8, cv2.COLOR_GRAY2BGR)
                        else:
                            # Fallback: guess from stride
                            bpp = stride // xres if xres > 0 else 0
                            if bpp == 4:
                                padded = data[:yres * stride].reshape(yres, stride)
                                bgra = padded[:, :xres * 4].reshape(yres, xres, 4)
                                bgr = bgra[:, :, :3].copy()
                            elif bpp == 3:
                                padded = data[:yres * stride].reshape(yres, stride)
                                bgr = padded[:, :xres * 3].reshape(yres, xres, 3).copy()
                            elif bpp == 2:
                                padded = data[:yres * stride].reshape(yres, stride)
                                uyvy = padded[:, :xres * 2]  # 2D: (H, W*2)
                                bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
                            else:
                                logger.debug("Unknown NDI format: fourcc=%s stride=%d xres=%d", fourcc, stride, xres)
                                continue

                        # Apply brightness/contrast
                        cam = self.state.camera_settings
                        if cam.brightness != 0.0 or cam.contrast != 1.0:
                            f = bgr.astype(np.float32)
                            f = cam.contrast * f + cam.brightness * 255.0
                            bgr = np.clip(f, 0, 255).astype(np.uint8)

                        self.frame = bgr
                        self.state.camera_frame = bgr
                    else:
                        time.sleep(0.001)
                except Exception as e:
                    if self._running:
                        logger.debug("NDI receive error: %s", e)
                    time.sleep(0.01)

        except Exception as e:
            logger.error("NDI receiver failed: %s", e)
        finally:
            self.connected = False
            if receiver is not None:
                try:
                    receiver.disconnect()
                except Exception:
                    pass
            if finder is not None:
                try:
                    finder.close()
                except Exception:
                    pass
