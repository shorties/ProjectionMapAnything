from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipeline import ProMapAnythingPipeline, ProMapAnythingProjectorPipeline

    register(ProMapAnythingPipeline)
    register(ProMapAnythingProjectorPipeline)
