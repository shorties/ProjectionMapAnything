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
    When done, the calibration mapping saves automatically and the result
    is uploaded to the media gallery for VACE conditioning.
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

    # -- Input-side controls (left panel) -------------------------------------

    start_calibration: bool = Field(
        default=False,
        description=(
            "Toggle to start calibration. Position the viewer on the "
            "projector first, then flip this ON."
        ),
        json_schema_extra=ui_field_config(
            order=0,
            label="Start Calibration",
            category="input",
        ),
    )

    open_dashboard: bool = Field(
        default=False,
        description="Enable to open the control panel dashboard in your browser.",
        json_schema_extra=ui_field_config(
            order=1,
            label="Open Dashboard",
            category="input",
        ),
    )

    reset_calibration: bool = Field(
        default=False,
        description=(
            "Enable to reset calibration state and start fresh. "
            "Clears all captured data so you can recalibrate without "
            "reloading the pipeline."
        ),
        json_schema_extra=ui_field_config(
            order=3,
            label="Reset Calibration",
            category="input",
        ),
    )

    # -- Configuration (settings panel) ---------------------------------------

    projector_width: int = Field(
        default=1920,
        ge=1,
        le=7680,
        description="Projector output resolution width in pixels.",
        json_schema_extra=ui_field_config(
            order=4,
            label="Projector Width",
            category="input",
        ),
    )

    projector_height: int = Field(
        default=1080,
        ge=1,
        le=4320,
        description="Projector output resolution height in pixels.",
        json_schema_extra=ui_field_config(
            order=5,
            label="Projector Height",
            category="input",
        ),
    )

    stream_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description="MJPEG stream port for projector pop-out window.",
        json_schema_extra=ui_field_config(
            order=2,
            label="Stream Port",
            is_load_param=True,
            category="configuration",
        ),
    )

    projector_url: str = Field(
        default="http://localhost:8765/",
        description=(
            "Open this URL in a browser to get the projector pop-out. "
            "On RunPod this is auto-detected. Drag the window to your "
            "projector, click to fullscreen, then start calibration."
        ),
        json_schema_extra=ui_field_config(
            order=3,
            label="Projector URL (open in browser)",
            is_load_param=True,
            category="configuration",
        ),
    )

    settle_frames: int = Field(
        default=60,
        ge=1,
        le=120,
        description=(
            "Frames to wait after each pattern change before capturing. "
            "Increase for high-latency setups (remote/RunPod). "
            "Local: 6–10, Remote: 30–60."
        ),
        json_schema_extra=ui_field_config(
            order=4,
            label="Settle Frames",
            is_load_param=True,
            category="configuration",
        ),
    )

    capture_frames: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Frames to average per pattern for noise rejection. "
            "More = cleaner but slower calibration."
        ),
        json_schema_extra=ui_field_config(
            order=5,
            label="Capture Frames",
            is_load_param=True,
            category="configuration",
        ),
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

    # -- Configuration (settings panel) ---------------------------------------
    # NOTE: Scope only renders "configuration" fields for preprocessors.
    # "input" category fields are invisible for preprocessor pipelines.

    depth_mode: Literal[
        "depth_then_warp",
        "warp_then_depth",
        "warped_rgb",
        "static_depth_warped",
        "static_depth_from_warped",
        "static_warped_camera",
    ] = Field(
        default="static_depth_warped",
        description=(
            "Conditioning input for the AI model. "
            "LIVE: depth_then_warp, warp_then_depth, warped_rgb (need GPU depth model). "
            "STATIC: static_depth_warped, static_depth_from_warped, "
            "static_warped_camera (from calibration, no GPU needed)."
        ),
        json_schema_extra=ui_field_config(
            order=0,
            label="Depth Mode",
            is_load_param=True,
            category="configuration",
        ),
    )

    temporal_smoothing: float = Field(
        default=0.5,
        ge=0.0,
        le=0.99,
        description="Blend factor with the previous depth frame. Higher = smoother.",
        json_schema_extra=ui_field_config(
            order=1,
            label="Temporal Smoothing",
            category="configuration",
        ),
    )

    depth_blur: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="Gaussian blur radius on the depth map. 0 = sharp.",
        json_schema_extra=ui_field_config(
            order=2,
            label="Depth Blur",
            category="configuration",
        ),
    )

    edge_erosion: int = Field(
        default=0,
        ge=0,
        le=50,
        description=(
            "Erode the valid depth region by this many pixels. "
            "Trims overshoot at object edges where depth bleeds "
            "onto background surfaces. 0 = no erosion."
        ),
        json_schema_extra=ui_field_config(
            order=3,
            label="Edge Erosion",
            category="configuration",
        ),
    )

    depth_contrast: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        description=(
            "Contrast multiplier for the depth map. Values > 1 sharpen "
            "depth transitions at object edges. 1.0 = no change."
        ),
        json_schema_extra=ui_field_config(
            order=4,
            label="Depth Contrast",
            category="configuration",
        ),
    )

    depth_near_clip: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Clip depth values below this threshold (0-1 normalized). "
            "Pixels closer than this become black. Useful for isolating "
            "foreground objects."
        ),
        json_schema_extra=ui_field_config(
            order=5,
            label="Near Clip",
            category="configuration",
        ),
    )

    depth_far_clip: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Clip depth values above this threshold (0-1 normalized). "
            "Pixels farther than this become black. Useful for removing "
            "background surfaces."
        ),
        json_schema_extra=ui_field_config(
            order=6,
            label="Far Clip",
            category="configuration",
        ),
    )

    stream_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description="MJPEG stream port for the dashboard input preview.",
        json_schema_extra=ui_field_config(
            order=10,
            label="Stream Port",
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
            order=11,
            label="Generation Resolution",
            is_load_param=True,
            category="configuration",
        ),
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

    # -- Configuration (settings panel) ---------------------------------------
    # NOTE: Scope only renders "configuration" fields for postprocessors.
    # "input" category fields are invisible for postprocessor pipelines.

    upscale_to_projector: bool = Field(
        default=True,
        description=(
            "Upscale output to projector resolution before streaming. "
            "Uses resolution from the companion app or calibration."
        ),
        json_schema_extra=ui_field_config(
            order=0,
            label="Upscale to Projector",
            category="configuration",
        ),
    )

    stream_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description="Port for the MJPEG streaming server.",
        json_schema_extra=ui_field_config(
            order=1,
            label="Stream Port",
            is_load_param=True,
            category="configuration",
        ),
    )
