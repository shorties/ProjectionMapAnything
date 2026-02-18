import logging
import threading
import time

from scope.core.plugins.hookspecs import hookimpl

logger = logging.getLogger(__name__)

# Seconds to wait after registration before pre-caching the depth model,
# giving Scope time to finish importing all plugins (concurrent imports of
# transformers cause partial-module errors).
_PRECACHE_DELAY_SECONDS = 15


def _precache_depth_model():
    """Download Depth Anything V2 Small in the background so it's ready on first use."""
    time.sleep(_PRECACHE_DELAY_SECONDS)
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        AutoImageProcessor.from_pretrained(model_id)
        AutoModelForDepthEstimation.from_pretrained(model_id)
    except Exception as exc:
        logger.debug("Depth model precache failed: %s", exc)


@hookimpl
def register_pipelines(register):
    from .pipeline import (
        ProMapAnythingCalibratePipeline,
        ProMapAnythingPipeline,
        ProMapAnythingProjectorPipeline,
    )

    register(ProMapAnythingCalibratePipeline)
    register(ProMapAnythingPipeline)
    register(ProMapAnythingProjectorPipeline)

    # Start pre-cache AFTER registration completes
    threading.Thread(target=_precache_depth_model, daemon=True, name="depth-precache").start()
