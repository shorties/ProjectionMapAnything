import threading

from scope.core.plugins.hookspecs import hookimpl


def _precache_depth_model():
    """Download Depth Anything V2 Small in the background so it's ready on first use."""
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_id = "depth-anything/Depth-Anything-V2-Small-hf"
        AutoImageProcessor.from_pretrained(model_id)
        AutoModelForDepthEstimation.from_pretrained(model_id)
    except Exception:
        pass  # non-critical — will retry when pipeline loads


# Start download as soon as Scope loads this plugin
threading.Thread(target=_precache_depth_model, daemon=True, name="depth-precache").start()


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
