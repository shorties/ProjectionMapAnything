from typing import Literal

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


# =============================================================================
# Calibration + Depth preprocessor
# =============================================================================


class ProMapAnythingConfig(BasePipelineConfig):
    """Projection mapping depth preprocessor (VACE-optimized).

    Appears in the **Preprocessor** dropdown.  Output feeds directly into the
    main pipeline as VACE depth conditioning.  Outputs grayscale depth maps
    with near=dark, far=bright — matching VACE's training data distribution.

    All depth processing parameters (CLAHE, percentile clip, gamma, colormap,
    scale, offset, invert) are hardcoded to VACE-optimal values.  Pipeline
    mode is always ``depth_warp_ai``.
    """

    pipeline_id = "promapanything-vj-depth"
    pipeline_name = "ProMapAnything VJ Tools"
    pipeline_description = (
        "Projection mapping depth preprocessor — calibrates projector-camera "
        "geometry via Gray code structured light patterns and outputs a "
        "VACE-optimized grayscale depth map re-projected to the projector's "
        "perspective.  Near=dark, far=bright."
    )

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.PREPROCESSOR]

    # -- Load-time parameters -------------------------------------------------

    calibration_file: str = Field(
        default="",
        description=(
            "Path to a calibration JSON file. If provided, the plugin loads "
            "the projector-camera mapping from this file instead of running "
            "the built-in calibration. Leave empty to calibrate interactively."
        ),
        json_schema_extra=ui_field_config(
            order=0,
            label="Calibration File",
            is_load_param=True,
            category="configuration",
        ),
    )

    projector_width: int = Field(
        default=1920,
        ge=640,
        le=3840,
        description=(
            "Projector output resolution width in pixels. "
            "Auto-filled when the companion app reports its resolution."
        ),
        json_schema_extra=ui_field_config(
            order=1,
            label="Projector Width",
            is_load_param=True,
            category="configuration",
        ),
    )

    projector_height: int = Field(
        default=1080,
        ge=480,
        le=2160,
        description=(
            "Projector output resolution height in pixels. "
            "Auto-filled when the companion app reports its resolution."
        ),
        json_schema_extra=ui_field_config(
            order=2,
            label="Projector Height",
            is_load_param=True,
            category="configuration",
        ),
    )

    generation_resolution: Literal["quarter", "half", "native"] = Field(
        default="half",
        description=(
            "Resolution for AI generation relative to the projector. "
            "'quarter' = 1/4 resolution (fast, lower quality), "
            "'half' = 1/2 resolution (balanced), "
            "'native' = full projector resolution (slow, highest quality). "
            "The depth map output is resized to this resolution."
        ),
        json_schema_extra=ui_field_config(
            order=3,
            label="Generation Resolution",
            is_load_param=True,
            category="configuration",
        ),
    )

    # -- Runtime controls -----------------------------------------------------

    calibrate: bool = Field(
        default=False,
        description=(
            "Toggle calibration mode. When enabled, the plugin projects Gray "
            "code patterns and captures the camera response to build a "
            "projector-camera pixel mapping. Disable when calibration is done."
        ),
        json_schema_extra=ui_field_config(order=10, label="Calibrate"),
    )

    temporal_smoothing: float = Field(
        default=0.5,
        ge=0.0,
        le=0.99,
        description=(
            "Blend factor with the previous frame's depth map. Higher values "
            "produce smoother but more latent output."
        ),
        json_schema_extra=ui_field_config(order=11, label="Temporal Smoothing"),
    )

    depth_blur: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description=(
            "Gaussian blur radius applied to the depth map. "
            "Reduces flicker between frames. 0 = sharp."
        ),
        json_schema_extra=ui_field_config(order=12, label="Depth Blur"),
    )


# =============================================================================
# Projector Output postprocessor
# =============================================================================


class ProMapAnythingProjectorConfig(BasePipelineConfig):
    """Projector output postprocessor.

    Appears in the **Postprocessor** dropdown.  Receives the final output
    from the main pipeline (e.g. Krea) and streams it to the companion app
    via MJPEG.  The stream starts automatically when this postprocessor is
    selected — no toggle needed.

    Optionally upscales the output to the projector's native resolution
    before streaming (useful when generating at 1/4 or 1/2 res).
    """

    pipeline_id = "promapanything-projector"
    pipeline_name = "ProMapAnything Projector"
    pipeline_description = (
        "Streams the final pipeline output to a projector via the companion "
        "app.  Selecting this postprocessor auto-starts the MJPEG stream.  "
        "Optionally upscales to projector resolution."
    )

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.POSTPROCESSOR]

    stream_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description=(
            "Port for the MJPEG streaming server. The companion app connects "
            "to this port (localhost for local, or via RunPod proxy for remote)."
        ),
        json_schema_extra=ui_field_config(
            order=0,
            label="Stream Port",
            is_load_param=True,
            category="configuration",
        ),
    )

    upscale_to_projector: bool = Field(
        default=True,
        description=(
            "Upscale the output to the projector's native resolution before "
            "streaming. Uses the resolution reported by the companion app. "
            "Disable if you want to stream at the generation resolution."
        ),
        json_schema_extra=ui_field_config(order=1, label="Upscale to Projector"),
    )
