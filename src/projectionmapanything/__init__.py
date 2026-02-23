import logging

from scope.core.plugins.hookspecs import hookimpl

logger = logging.getLogger(__name__)


@hookimpl
def register_pipelines(register):
    from .pipeline import ProMapAnythingPipeline

    # Primary pipeline — handles calibration + depth conditioning.
    # WebRTC projector page (/projector) replaces the old MJPEG postprocessor.
    register(ProMapAnythingPipeline)
