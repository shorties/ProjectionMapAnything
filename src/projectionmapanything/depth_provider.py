"""Depth estimation via Depth Anything V2 (HuggingFace transformers).

Single concrete provider — no ABC, no built-in Scope depth fallback.
Uses percentile normalization for robust contrast.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DepthProvider:
    """Monocular depth estimation using Depth Anything V2.

    Loads the model via HuggingFace transformers.  Supports Small and Base
    model sizes.  Falls back to CPU automatically on GPU OOM.
    """

    MODELS = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
    }

    def __init__(self, device: torch.device, model_size: str = "small"):
        self.device = device
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required for depth estimation "
                "but is not installed.  Install it with: pip install transformers"
            ) from exc

        model_id = self.MODELS.get(model_size, self.MODELS["small"])
        logger.info("Loading Depth Anything V2 %s via transformers ...", model_size.title())
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            model_id, device_map=None,
        )
        try:
            self._model.to(device).eval()
            logger.info("Depth model loaded on %s (%s)", device, model_size)
        except (torch.cuda.OutOfMemoryError, RuntimeError, NotImplementedError):
            logger.warning("Cannot load depth model on %s — falling back to CPU", device)
            self.device = torch.device("cpu")
            try:
                self._model.to(self.device).eval()
            except NotImplementedError:
                # Meta tensor: must reload from scratch
                self._model = AutoModelForDepthEstimation.from_pretrained(
                    model_id, device_map=None,
                )
                self._model.to(self.device).eval()
            logger.info("Depth model loaded on CPU (%s)", model_size)

    @torch.no_grad()
    def estimate(self, frame: torch.Tensor) -> torch.Tensor:
        """Estimate depth from a single RGB frame.

        Parameters
        ----------
        frame : torch.Tensor
            Shape (H, W, 3), float32, [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (H, W), float32, values normalised to [0, 1].
            Convention: 1 = near (bright), 0 = far (dark).
            This is inverse depth (disparity) matching VACE / Depth Anything V2.
        """
        from PIL import Image

        h, w = frame.shape[:2]

        img_np = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        # predicted_depth is (1, Hm, Wm) — use explicit indexing, not squeeze
        depth = outputs.predicted_depth[0, :, :]  # (Hm, Wm)

        # Interpolate to original resolution
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0),  # (1, 1, Hm, Wm)
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]  # (H, W) — explicit indexing

        # Percentile normalization (2nd/98th) — robust to outliers
        flat = depth.flatten()
        p2 = torch.quantile(flat.float(), 0.02)
        p98 = torch.quantile(flat.float(), 0.98)
        if p98 - p2 > 1e-6:
            depth = (depth - p2) / (p98 - p2)
        else:
            depth = torch.zeros_like(depth)
        depth = depth.clamp(0.0, 1.0)

        return depth.to(self.device)


def create_depth_provider(
    device: torch.device,
    model_size: str = "small",
) -> DepthProvider:
    """Create a depth provider.

    Parameters
    ----------
    device : torch.device
        Compute device.
    model_size : str
        ``"small"`` or ``"base"`` for Depth Anything V2.
    """
    return DepthProvider(device, model_size=model_size)
