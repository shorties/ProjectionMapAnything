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

    pipeline_id = "projectionmapanything-calibrate"
    pipeline_name = "ProjectionMapAnything Calibrate"
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

    calibration_brightness: int = Field(
        default=128,
        ge=10,
        le=255,
        description=(
            "Brightness of the projector during the test card and WHITE "
            "calibration phase (0-255). Lower values reduce glare and give "
            "a more natural RGB capture. Adjust before starting calibration."
        ),
        json_schema_extra=ui_field_config(
            order=1,
            label="Calibration Brightness",
            category="input",
        ),
    )

    open_dashboard: bool = Field(
        default=False,
        description="Enable to open the control panel dashboard in your browser.",
        json_schema_extra=ui_field_config(
            order=2,
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
        default="Change 8000 to 8765 in this page's URL to open the dashboard",
        description=(
            "To open the projector dashboard, change the port in your "
            "current Scope URL from 8000 to 8765. On RunPod: replace "
            "-8000. in the URL with -8765. — the dashboard will "
            "auto-detect the environment."
        ),
        json_schema_extra=ui_field_config(
            order=3,
            label="Dashboard URL Hint",
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
            "Local: 6-10, Remote: 30-60."
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
    """Projection mapping preprocessor — calibration + depth conditioning.

    Select as the **Preprocessor**.  Handles calibration (toggle Start
    Calibration) and depth conditioning for VACE/ControlNet.  All depth
    processing controls are available via the dashboard (port 8765) —
    only essential controls appear here.
    """

    pipeline_id = "Projection-Map-Anything-VJ-Tools"
    pipeline_name = "Projection-Map-Anything (VJ.Tools)"
    pipeline_description = (
        "Projection mapping preprocessor — calibration + depth conditioning. "
        "Toggle Start Calibration to run Gray code calibration. "
        "Open the dashboard (port 8765) for advanced depth controls."
    )

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.PREPROCESSOR]

    # -- Calibration controls -------------------------------------------------

    start_calibration: bool = Field(
        default=False,
        description=(
            "Start Gray code calibration. Open the projector page first "
            "(dashboard port 8765 → Projector button), then toggle ON. "
            "Patterns are projected and captured automatically. "
            "Toggle OFF to cancel."
        ),
        json_schema_extra=ui_field_config(
            order=0,
            label="Start Calibration",
            category="configuration",
        ),
    )

    reset_calibration: bool = Field(
        default=False,
        description=(
            "Clear saved calibration and start fresh. Toggle ON to reset, "
            "then toggle OFF. The next calibration will overwrite the file."
        ),
        json_schema_extra=ui_field_config(
            order=1,
            label="Reset Calibration",
            category="configuration",
        ),
    )

    calibration_speed: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Calibration speed. 0 = careful (60 settle frames, 5 captures "
            "per pattern — best for remote/RunPod). 1 = fast (8 settle "
            "frames, 2 captures — for local low-latency setups)."
        ),
        json_schema_extra=ui_field_config(
            order=2,
            label="Calibration Speed",
            category="configuration",
        ),
    )

    # -- Depth mode selector ---------------------------------------------------

    depth_mode: Literal[
        "ai_depth",
        "disparity",
        "canny",
        "warped_rgb",
        "custom",
    ] = Field(
        default="ai_depth",
        description=(
            "Conditioning input for the AI model. All modes use static "
            "calibration data — never live camera processing (avoids "
            "camera-projector feedback loops). "
            "ai_depth: Depth Anything V2 on the raw camera image, warped "
            "to projector perspective (best quality, recommended). "
            "disparity: horizontal disparity from calibration — simple, "
            "fast, no AI model needed. "
            "canny: edge detection on warped camera image. "
            "warped_rgb: warped camera image from calibration. "
            "custom: upload your own via the dashboard."
        ),
        json_schema_extra=ui_field_config(
            order=4,
            label="Depth Mode",
            category="configuration",
        ),
    )

    # -- Dashboard link -------------------------------------------------------

    dashboard_url: str = Field(
        default="Change 8000 to 8765 in this page's URL",
        description=(
            "Open the dashboard for projector pop-out, advanced depth "
            "controls, calibration results, and custom uploads."
        ),
        json_schema_extra=ui_field_config(
            order=3,
            label="Dashboard",
            is_load_param=True,
            category="configuration",
        ),
    )

    # -- Load-time config (rarely changed) ------------------------------------

    stream_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description="HTTP port for the dashboard and MJPEG projector stream.",
        json_schema_extra=ui_field_config(
            order=10,
            label="Stream Port",
            is_load_param=True,
            category="configuration",
        ),
    )


# =============================================================================
# Projector Output postprocessor
# =============================================================================


class ProMapAnythingProjectorConfig(BasePipelineConfig):
    """Projector output postprocessor — MJPEG stream to projector.

    Appears in the **Postprocessor** dropdown.  Select this to stream the
    AI-generated output to the projector page (dashboard port 8765 →
    Projector button).  Includes optional edge feathering, subject masking,
    and color correction before streaming.
    """

    pipeline_id = "projectionmapanything-projector"
    pipeline_name = "ProjectionMapAnything Projector"
    pipeline_description = (
        "Streams AI output to the projector via MJPEG. Select this as the "
        "postprocessor to see the AI canvas on the projector page. "
        "Includes edge feathering, subject masking, and color correction."
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

    edge_feather: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description=(
            "Fade to black at the projection edges (pixels). "
            "Softens the hard boundary where the projection meets the wall."
        ),
        json_schema_extra=ui_field_config(
            order=1,
            label="Edge Feather",
            category="configuration",
        ),
    )

    apply_subject_mask: bool = Field(
        default=False,
        description=(
            "Apply the preprocessor's subject isolation mask to the output. "
            "Blacks out areas outside the isolated subject."
        ),
        json_schema_extra=ui_field_config(
            order=2,
            label="Apply Subject Mask",
            category="configuration",
        ),
    )

    brightness: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Brightness adjustment. 0 = no change.",
        json_schema_extra=ui_field_config(
            order=3,
            label="Brightness",
            category="configuration",
        ),
    )

    gamma: float = Field(
        default=1.0,
        ge=0.2,
        le=3.0,
        description="Gamma correction. <1 = brighter midtones, >1 = darker midtones.",
        json_schema_extra=ui_field_config(
            order=4,
            label="Gamma",
            category="configuration",
        ),
    )

    contrast: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Contrast multiplier. 1.0 = no change.",
        json_schema_extra=ui_field_config(
            order=5,
            label="Contrast",
            category="configuration",
        ),
    )

    stream_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description="Port for the MJPEG streaming server.",
        json_schema_extra=ui_field_config(
            order=10,
            label="Stream Port",
            is_load_param=True,
            category="configuration",
        ),
    )
