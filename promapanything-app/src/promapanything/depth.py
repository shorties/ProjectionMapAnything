"""Depth Anything V2 Small inference wrapper.

Downloads the model weights on first run and caches them locally.
Uses direct PyTorch loading (no transformers dependency).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
MODEL_FILE = "depth_anything_v2_vits.pth"
HF_REPO = "depth-anything/Depth-Anything-V2-Small"


def _download_model(dest: Path) -> None:
    """Download model weights from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

        logger.info("Downloading Depth Anything V2 Small weights...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=MODEL_FILE,
            local_dir=str(dest.parent),
        )
        logger.info("Download complete: %s", dest)
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required to download the depth model. "
            "Install it with: pip install huggingface_hub\n"
            f"Or manually download {MODEL_FILE} from "
            f"https://huggingface.co/{HF_REPO} and place it in {dest.parent}"
        )


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2 Small."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_path = ASSETS_DIR / MODEL_FILE
        if not model_path.is_file():
            ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            _download_model(model_path)

        # Load model using the transformers-based approach for simplicity
        # Falls back to direct loading if available
        self._model = self._load_model(model_path)
        self._prev_depth: np.ndarray | None = None

    def _load_model(self, path: Path):
        """Load the depth model."""
        try:
            # Try using transformers (most reliable)
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            logger.info("Loading Depth Anything V2 Small via transformers...")
            model_id = "depth-anything/Depth-Anything-V2-Small-hf"
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id)
            model.to(self.device).eval()
            self._use_transformers = True
            logger.info("Depth model loaded (transformers)")
            return model
        except ImportError:
            pass

        # Fallback: try direct model loading with timm-based architecture
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

            model_config = {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            }
            model = DepthAnythingV2(**model_config)
            model.load_state_dict(torch.load(str(path), map_location="cpu"))
            model.to(self.device).eval()
            self._use_transformers = False
            logger.info("Depth model loaded (direct)")
            return model
        except ImportError:
            raise RuntimeError(
                "Neither 'transformers' nor 'depth_anything_v2' packages are available. "
                "Install one: pip install transformers"
            )

    @torch.no_grad()
    def estimate(self, frame_bgr: np.ndarray, max_side: int = 518) -> np.ndarray:
        """Estimate depth from a BGR uint8 frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            (H, W, 3) BGR uint8
        max_side : int
            Downscale so the longest side is at most this many pixels.

        Returns
        -------
        np.ndarray
            (H, W) float32 normalised to [0, 1]
        """
        h, w = frame_bgr.shape[:2]

        # Downscale for speed
        scale = min(max_side / max(h, w), 1.0)
        if scale < 1.0:
            frame_bgr = cv2.resize(
                frame_bgr, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        if self._use_transformers:
            from PIL import Image

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            inputs = self._processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self._model(**inputs)
            depth = outputs.predicted_depth.squeeze()

            # Resize to original
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            depth = depth.cpu().numpy()
        else:
            depth = self._model.infer_image(frame_bgr)

        # Normalise to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)
