from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipeline import (
        ProMapAnythingEffectsPipeline,
        ProMapAnythingPipeline,
        ProMapAnythingPreviewPipeline,
    )

    register(ProMapAnythingPipeline)
    register(ProMapAnythingPreviewPipeline)
    register(ProMapAnythingEffectsPipeline)
