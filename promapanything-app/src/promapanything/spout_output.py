"""Spout texture I/O — sends depth/color/projector and receives from Scope."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_SPOUT_AVAILABLE = False
_GL_RGBA = 0x1908  # fallback constant
try:
    import SpoutGL
    from SpoutGL.enums import GL_RGBA as _GL_RGBA
    _SPOUT_AVAILABLE = True
except ImportError:
    logger.warning("SpoutGL not available — Spout output disabled")


class SpoutSender:
    """Wraps a single Spout sender."""

    def __init__(self, name: str, width: int, height: int):
        self.name = name
        self.width = width
        self.height = height
        self._sender = None

        if not _SPOUT_AVAILABLE:
            return

        self._sender = SpoutGL.SpoutSender()
        self._sender.setSenderName(name)
        logger.info("Spout sender created: %s (%dx%d)", name, width, height)

    def send(self, data: np.ndarray) -> None:
        """Send a frame via Spout.

        Parameters
        ----------
        data : np.ndarray
            (H, W, 3) or (H, W, 4) uint8 RGB(A) image.
        """
        if self._sender is None:
            return

        h, w = data.shape[:2]
        if w != self.width or h != self.height:
            self.width = w
            self.height = h

        # SpoutGL expects RGBA
        if data.shape[2] == 3:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = data
            rgba[:, :, 3] = 255
            data = rgba

        # Flip vertically (OpenGL convention)
        data = data[::-1].copy()

        self._sender.sendImage(
            data.tobytes(), w, h, _GL_RGBA, False, 0
        )

    def release(self) -> None:
        if self._sender is not None:
            self._sender.releaseSender()
            self._sender = None


class SpoutReceiver:
    """Wraps a single Spout receiver — pulls frames from another app."""

    def __init__(self, name: str = ""):
        self.name = name
        self._receiver = None
        self.width = 0
        self.height = 0
        self._connected = False

        if not _SPOUT_AVAILABLE:
            return

        self._receiver = SpoutGL.SpoutReceiver()
        if name:
            self._receiver.setReceiverName(name)
        logger.info("Spout receiver created (source=%s)", name or "<auto>")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def sender_name(self) -> str:
        """Return the name of the sender we're currently connected to."""
        if self._receiver is None:
            return ""
        try:
            return self._receiver.getSenderName()
        except Exception:
            return self.name

    def receive(self) -> np.ndarray | None:
        """Try to receive a frame. Returns (H, W, 4) RGBA uint8 or None."""
        if self._receiver is None:
            return None

        # Check for a sender — receiveImage returns True when data is ready
        # SpoutGL.receiveImage needs a buffer to write into.
        # We first check if there's a sender, then get its size.
        try:
            result = self._receiver.receiveImage(
                b"",  # empty buffer triggers size query
                _GL_RGBA, False, 0
            )
        except Exception:
            result = None

        if not result:
            # Try isConnected / check sender
            try:
                if self._receiver.isUpdated():
                    self.width = self._receiver.getSenderWidth()
                    self.height = self._receiver.getSenderHeight()
            except Exception:
                pass

        if self.width == 0 or self.height == 0:
            self._connected = False
            return None

        buf_size = self.width * self.height * 4
        buf = bytearray(buf_size)
        success = self._receiver.receiveImage(
            buf, _GL_RGBA, False, 0
        )
        if not success:
            self._connected = False
            return None

        self._connected = True

        # Check for size updates
        if self._receiver.isUpdated():
            self.width = self._receiver.getSenderWidth()
            self.height = self._receiver.getSenderHeight()

        frame = np.frombuffer(buf, dtype=np.uint8).reshape(
            self.height, self.width, 4
        )
        return frame

    def release(self) -> None:
        if self._receiver is not None:
            self._receiver.releaseReceiver()
            self._receiver = None


class SpoutOutputManager:
    """Manages the three Spout output feeds."""

    def __init__(self):
        self.depth_sender: SpoutSender | None = None
        self.color_sender: SpoutSender | None = None
        self.projector_sender: SpoutSender | None = None

    def init(
        self,
        depth_name: str = "ProMap-DepthMap",
        color_name: str = "ProMap-ColorMap",
        projector_name: str = "ProMap-Projector",
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        if not _SPOUT_AVAILABLE:
            return
        self.depth_sender = SpoutSender(depth_name, width, height)
        self.color_sender = SpoutSender(color_name, width, height)
        self.projector_sender = SpoutSender(projector_name, width, height)

    def send_depth(self, depth: np.ndarray) -> None:
        """Send a depth map. Expects (H, W) float32 [0,1]."""
        if self.depth_sender is None:
            return
        gray = (depth * 255).clip(0, 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        self.depth_sender.send(rgb)

    def send_color(self, frame_bgr: np.ndarray) -> None:
        """Send a camera frame. Expects (H, W, 3) BGR uint8."""
        if self.color_sender is None:
            return
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.color_sender.send(rgb)

    def send_projector(self, frame_rgb: np.ndarray) -> None:
        """Send projector output. Expects (H, W, 3) RGB uint8."""
        if self.projector_sender is None:
            return
        self.projector_sender.send(frame_rgb)

    def release(self) -> None:
        for sender in [self.depth_sender, self.color_sender, self.projector_sender]:
            if sender is not None:
                sender.release()
