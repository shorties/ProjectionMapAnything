"""Depth estimation provider with automatic fallback.

Tries to use Scope's built-in Depth Anything pipeline first.  If that is not
available (plugin not installed), falls back to loading the model directly via
the ``transformers`` library.

Supports both Depth Anything V2 Small and Base models.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DepthProvider(ABC):
    """Abstract interface for monocular depth estimation."""

    @abstractmethod
    def estimate(self, frame: torch.Tensor) -> torch.Tensor:
        """Estimate depth from a single RGB frame.

        Parameters
        ----------
        frame : torch.Tensor
            Shape (H, W, 3), float32, [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (H, W), float32, values normalised to [0, 1] where
            0 = near, 1 = far (or vice-versa depending on model).
        """


class BuiltinDepthProvider(DepthProvider):
    """Uses Scope's built-in Depth Anything pipeline."""

    def __init__(self, device: torch.device):
        self.device = device
        from scope_depth_anything.pipeline import DepthAnythingPipeline  # type: ignore[import-not-found]

        self._pipeline = DepthAnythingPipeline(device=device)
        logger.info("Using Scope's built-in Depth Anything pipeline")

    def estimate(self, frame: torch.Tensor) -> torch.Tensor:
        frame_255 = (frame * 255.0).unsqueeze(0)  # (1, H, W, C)
        result = self._pipeline(video=[frame_255])
        depth_out = result["video"]  # (T, H, W, C) [0,1]
        depth = depth_out[0]  # (H, W, C)
        if depth.shape[-1] == 3:
            depth = depth.mean(dim=-1)  # (H, W)
        return depth


class TransformersDepthProvider(DepthProvider):
    """Loads Depth Anything V2 via HuggingFace transformers.

    Supports Small and Base model sizes.
    """

    MODELS = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
    }

    def __init__(self, device: torch.device, model_size: str = "small"):
        self.device = device
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required for the bundled depth "
                "provider but is not installed.  Install it with: "
                "pip install transformers"
            ) from exc

        model_id = self.MODELS.get(model_size, self.MODELS["small"])
        logger.info("Loading Depth Anything V2 %s via transformers ...", model_size.title())
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self._model.to(device).eval()
        logger.info("Depth model loaded (%s)", model_size)

    @torch.no_grad()
    def estimate(self, frame: torch.Tensor) -> torch.Tensor:
        from transformers.image_utils import to_pil_image  # type: ignore[import-not-found]

        h, w = frame.shape[:2]

        img_np = (frame.cpu().numpy() * 255).astype(np.uint8)
        pil_img = to_pil_image(img_np)

        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        depth = outputs.predicted_depth  # (1, Hm, Wm)

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # (H, W)

        # Normalise to [0, 1]
        d_min = depth.min()
        d_max = depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = torch.zeros_like(depth)

        return depth.to(self.device)


def create_depth_provider(
    mode: str,
    device: torch.device,
    model_size: str = "small",
) -> DepthProvider:
    """Factory that creates the appropriate depth provider.

    Parameters
    ----------
    mode : str
        ``"auto"`` -- try built-in first, fall back to transformers.
        ``"bundled"`` -- use transformers directly.
    device : torch.device
        Compute device.
    model_size : str
        ``"small"`` or ``"base"`` for Depth Anything V2.
    """
    if mode == "auto":
        try:
            return BuiltinDepthProvider(device)
        except Exception:
            logger.info(
                "Scope's built-in Depth Anything not available, "
                "falling back to bundled model (%s)", model_size
            )
            return TransformersDepthProvider(device, model_size=model_size)
    else:
        return TransformersDepthProvider(device, model_size=model_size)
