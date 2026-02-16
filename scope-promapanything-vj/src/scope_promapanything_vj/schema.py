from typing import Literal

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


# =============================================================================
# Calibration main pipeline
# =============================================================================


class ProMapAnythingCalibrateConfig(BasePipelineConfig):
    """Projector-camera calibration via Gray code structured light.

    Appears in the **main pipeline** selector.  Select it, hit play, position
    the Scope viewer on the projector, then toggle **Start Calibration**.
    Patterns are displayed in the viewer and captured by the camera.
    When done, the calibration mapping saves automatically.
    """

    pipeline_id = "promapanything-calibrate"
    pipeline_name = "ProMapAnything Calibrate"
    pipeline_description = (
        "Projector-camera calibration via Gray code structured light. "
        "Displays patterns in the Scope viewer (projector), captures them "
        "with the camera, and saves the pixel mapping for depth warping."
    )

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}

    # -- Load-time parameters -------------------------------------------------

    projector_width: int = Field(
        default=1920,
        ge=1,
        le=7680,
        description="Projector output resolution width in pixels.",
        json_schema_extra=ui_field_config(
            order=0,
            label="Projector Width",
            is_load_param=True,
            category="configuration",
        ),
    )

    projector_height: int = Field(
        default=1080,
        ge=1,
        le=4320,
        description="Projector output resolution height in pixels.",
        json_schema_extra=ui_field_config(
            order=1,
            label="Projector Height",
            is_load_param=True,
            category="configuration",
        ),
    )

    stream_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description=(
            "MJPEG stream port. Open the control panel in your browser "
            "to pop out a fullscreen projector window."
        ),
        json_schema_extra=ui_field_config(
            order=2,
            label="Stream Port",
            is_load_param=True,
            category="configuration",
        ),
    )

    # -- Runtime controls -----------------------------------------------------

    start_calibration: bool = Field(
        default=False,
        description=(
            "Toggle to start calibration. First open the projector window "
            "from the control panel, then flip this ON."
        ),
        json_schema_extra=ui_field_config(order=10, label="Start Calibration"),
    )


# =============================================================================
# Depth preprocessor
# =============================================================================


class ProMapAnythingConfig(BasePipelineConfig):
    """Projection mapping depth preprocessor (VACE-optimized).

    Appears in the **Preprocessor** dropdown.  Estimates depth from the
    camera, warps it to the projector's perspective using saved calibration,
    and outputs a VACE-optimized grayscale depth map (near=dark, far=bright).
    """

    pipeline_id = "promapanything-vj-depth"
    pipeline_name = "ProMapAnything Depth"
    pipeline_description = (
        "Depth preprocessor — estimates depth and warps to projector "
        "perspective using saved calibration.  Outputs VACE-optimized "
        "grayscale depth (near=dark, far=bright)."
    )

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.PREPROCESSOR]

    # -- Load-time parameters -------------------------------------------------

    calibration_file: str = Field(
        default="",
        description=(
            "Path to a calibration JSON file.  Leave empty to use the "
            "default (~/.promapanything_calibration.json)."
        ),
        json_schema_extra=ui_field_config(
            order=0,
            label="Calibration File",
            is_load_param=True,
            category="configuration",
        ),
    )

    generation_resolution: Literal["quarter", "half", "native"] = Field(
        default="half",
        description=(
            "Resolution for AI generation relative to the projector. "
            "'quarter' = 1/4 res (fast), 'half' = 1/2 res (balanced), "
            "'native' = full projector res (slow, highest quality)."
        ),
        json_schema_extra=ui_field_config(
            order=1,
            label="Generation Resolution",
            is_load_param=True,
            category="configuration",
        ),
    )

    # -- Runtime controls -----------------------------------------------------

    temporal_smoothing: float = Field(
        default=0.5,
        ge=0.0,
        le=0.99,
        description="Blend factor with the previous depth frame. Higher = smoother.",
        json_schema_extra=ui_field_config(order=10, label="Temporal Smoothing"),
    )

    depth_blur: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="Gaussian blur radius on the depth map. 0 = sharp.",
        json_schema_extra=ui_field_config(order=11, label="Depth Blur"),
    )


# =============================================================================
# Projector Output postprocessor
# =============================================================================


class ProMapAnythingProjectorConfig(BasePipelineConfig):
    """Projector output postprocessor.

    Appears in the **Postprocessor** dropdown.  Streams the final output
    to the companion app via MJPEG.  Auto-starts the stream when selected.
    """

    pipeline_id = "promapanything-projector"
    pipeline_name = "ProMapAnything Projector"
    pipeline_description = (
        "Streams the final pipeline output to a projector via MJPEG. "
        "Optionally upscales to projector resolution."
    )

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.POSTPROCESSOR]

    stream_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description="Port for the MJPEG streaming server.",
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
            "Upscale output to projector resolution before streaming. "
            "Uses resolution from the companion app or calibration."
        ),
        json_schema_extra=ui_field_config(order=1, label="Upscale to Projector"),
    )
