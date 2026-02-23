import logging

from scope.core.plugins.hookspecs import hookimpl

logger = logging.getLogger(__name__)


@hookimpl
def register_pipelines(register):
    from .pipeline import ProMapAnythingPipeline, ProMapAnythingProjectorPipeline

    # Primary preprocessor — handles calibration + depth conditioning.
    register(ProMapAnythingPipeline)

    # Postprocessor — relays AI output to the projector via MJPEG stream.
    # Required for projector output on /projector page.
    register(ProMapAnythingProjectorPipeline)
