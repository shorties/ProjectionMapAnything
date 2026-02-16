from scope.core.plugins.hookspecs import hookimpl


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
