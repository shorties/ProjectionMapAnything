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
    """Projection mapping depth preprocessor.

    Appears in the **Preprocessor** dropdown.  Output feeds directly into the
    main ControlNet pipeline as depth conditioning.
    """

    pipeline_id = "promapanything-vj-depth"
    pipeline_name = "ProMapAnything VJ Tools"
    pipeline_description = (
        "Projection mapping depth preprocessor — calibrates projector-camera "
        "geometry via Gray code structured light patterns and outputs a depth "
        "map re-projected to the projector's perspective for ControlNet conditioning."
    )

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.PREPROCESSOR]

    # -- Load-time parameters -------------------------------------------------

    depth_source: Literal["camera", "external_app"] = Field(
        default="camera",
        description=(
            "Where to get the depth map from. 'camera' runs depth estimation "
            "on the camera feed (normal mode). 'external_app' auto-detects "
            "a running ProMapAnything standalone app and reads its depth map "
            "in real time — no camera or depth model needed."
        ),
        json_schema_extra=ui_field_config(
            order=0,
            label="Depth Source",
            is_load_param=True,
            category="configuration",
        ),
    )

    depth_provider: Literal["auto", "bundled"] = Field(
        default="auto",
        description=(
            "Depth estimation backend (only used when Depth Source = 'camera'). "
            "'auto' tries Scope's built-in Depth Anything first and falls "
            "back to a bundled model. 'bundled' always uses the plugin's "
            "own model via transformers."
        ),
        json_schema_extra=ui_field_config(
            order=1,
            label="Depth Provider",
            is_load_param=True,
            category="configuration",
        ),
    )

    depth_model_size: Literal["small", "base"] = Field(
        default="small",
        description=(
            "Depth Anything V2 model size. 'small' is faster with lower VRAM, "
            "'base' produces higher quality depth but uses more memory."
        ),
        json_schema_extra=ui_field_config(
            order=2,
            label="Model Size",
            is_load_param=True,
            category="configuration",
        ),
    )

    projector_width: int = Field(
        default=1920,
        ge=640,
        le=3840,
        description="Projector output resolution width in pixels",
        json_schema_extra=ui_field_config(
            order=3,
            label="Projector Width",
            is_load_param=True,
            category="configuration",
        ),
    )

    projector_height: int = Field(
        default=1080,
        ge=480,
        le=2160,
        description="Projector output resolution height in pixels",
        json_schema_extra=ui_field_config(
            order=4,
            label="Projector Height",
            is_load_param=True,
            category="configuration",
        ),
    )

    calibration_file: str = Field(
        default="",
        description=(
            "Path to a calibration JSON file. If provided, the plugin loads "
            "the projector-camera mapping from this file instead of running "
            "the built-in calibration. Leave empty to calibrate interactively."
        ),
        json_schema_extra=ui_field_config(
            order=5,
            label="Calibration File",
            is_load_param=True,
            category="configuration",
        ),
    )

    # -- Pipeline ordering ----------------------------------------------------

    pipeline_mode: Literal["depth_warp_ai", "depth_ai_warp"] = Field(
        default="depth_warp_ai",
        description=(
            "Pipeline ordering. 'Depth -> Warp -> AI': warp depth to projector "
            "perspective before AI sees it (default). 'Depth -> AI -> Warp': AI "
            "generates from camera-space depth, then warp AI output to projector "
            "perspective in the effects pipeline."
        ),
        json_schema_extra=ui_field_config(
            order=8,
            label="Pipeline Mode",
            is_load_param=True,
            category="configuration",
        ),
    )

    # -- External app controls ------------------------------------------------

    external_app_feed: Literal["depth_bw", "depth_color", "projector_rgb"] = Field(
        default="depth_bw",
        description=(
            "Which output from the standalone app to use as the depth map. "
            "'depth_bw' = B&W depth map, 'depth_color' = colormapped depth, "
            "'projector_rgb' = color-corrected projector output. "
            "Only available when Depth Source = 'external_app' and the app is running."
        ),
        json_schema_extra=ui_field_config(order=6, label="App Feed"),
    )

    external_app_status: str = Field(
        default="",
        description="Status of the external app connection (read-only)",
        json_schema_extra=ui_field_config(order=7, label="App Status"),
    )

    # -- Calibration runtime controls -----------------------------------------

    calibrate: bool = Field(
        default=False,
        description=(
            "Toggle calibration mode. When enabled, the plugin projects Gray "
            "code patterns and captures the camera response to build a "
            "projector-camera pixel mapping. Disable when calibration is done."
        ),
        json_schema_extra=ui_field_config(order=10, label="Calibrate"),
    )

    calibration_settle_frames: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Number of frames to wait after projecting each pattern before "
            "capturing. Increase if the camera has high latency."
        ),
        json_schema_extra=ui_field_config(order=11, label="Settle Frames"),
    )

    capture_frames: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Number of frames to capture and average per pattern. Higher "
            "values reduce noise at the cost of longer calibration time."
        ),
        json_schema_extra=ui_field_config(order=12, label="Frames/Pattern"),
    )

    decode_threshold: float = Field(
        default=20.0,
        ge=5.0,
        le=100.0,
        description=(
            "Minimum white-black difference for a pixel to be considered valid. "
            "Lower = more aggressive coverage, higher = stricter quality."
        ),
        json_schema_extra=ui_field_config(order=13, label="Decode Threshold"),
    )

    bit_threshold: float = Field(
        default=3.0,
        ge=0.0,
        le=30.0,
        description=(
            "Minimum |positive - negative| per Gray code bit to trust the "
            "decode. Rejects pixels at bit boundaries. 0 = disabled."
        ),
        json_schema_extra=ui_field_config(order=14, label="Bit Threshold"),
    )

    # -- ControlNet depth output settings -------------------------------------

    depth_equalize: bool = Field(
        default=True,
        description=(
            "Apply CLAHE histogram equalization for maximum contrast in the "
            "depth output — optimized for ControlNet consumption."
        ),
        json_schema_extra=ui_field_config(order=20, label="CLAHE Equalize"),
    )

    depth_clip_lo: float = Field(
        default=2.0,
        ge=0.0,
        le=49.0,
        description="Lower percentile clip — removes darkest outliers",
        json_schema_extra=ui_field_config(order=21, label="Clip Low %"),
    )

    depth_clip_hi: float = Field(
        default=98.0,
        ge=51.0,
        le=100.0,
        description="Upper percentile clip — removes brightest outliers",
        json_schema_extra=ui_field_config(order=22, label="Clip High %"),
    )

    depth_gamma: float = Field(
        default=1.0,
        ge=0.2,
        le=5.0,
        description="Gamma correction (< 1 brightens darks, > 1 darkens darks)",
        json_schema_extra=ui_field_config(order=23, label="Gamma"),
    )

    depth_scale: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Contrast multiplier for the depth map",
        json_schema_extra=ui_field_config(order=24, label="Depth Scale"),
    )

    depth_offset: float = Field(
        default=0.0,
        ge=-0.5,
        le=0.5,
        description="Brightness offset added to the depth map",
        json_schema_extra=ui_field_config(order=25, label="Depth Offset"),
    )

    depth_blur: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description=(
            "Gaussian blur radius applied to the depth map. Reduces noise "
            "and flickering between frames."
        ),
        json_schema_extra=ui_field_config(order=26, label="Depth Blur"),
    )

    depth_invert: bool = Field(
        default=False,
        description="Invert the depth map (swap near and far)",
        json_schema_extra=ui_field_config(order=27, label="Invert Depth"),
    )

    colormap: Literal["grayscale", "turbo", "viridis", "magma"] = Field(
        default="grayscale",
        description="Colormap applied to the depth visualization",
        json_schema_extra=ui_field_config(order=28, label="Colormap"),
    )

    temporal_smoothing: float = Field(
        default=0.5,
        ge=0.0,
        le=0.99,
        description=(
            "Blend factor with the previous frame's depth map. Higher values "
            "produce smoother but more latent output."
        ),
        json_schema_extra=ui_field_config(order=29, label="Temporal Smoothing"),
    )

    # -- Projector output -----------------------------------------------------

    projector_output: bool = Field(
        default=False,
        description=(
            "Open a fullscreen window on the selected monitor to display the "
            "depth output directly on the projector."
        ),
        json_schema_extra=ui_field_config(order=30, label="Projector Output"),
    )

    projector_monitor: int = Field(
        default=1,
        ge=0,
        le=8,
        description=(
            "Monitor index for the projector output window. 0 = primary monitor, "
            "1 = first secondary monitor, etc."
        ),
        json_schema_extra=ui_field_config(order=31, label="Projector Monitor"),
    )

    # -- Projector stream (remote) --------------------------------------------

    projector_stream: bool = Field(
        default=False,
        description=(
            "Start an MJPEG streaming server so a remote client (or browser) "
            "can display the output on a projector. Use this when running "
            "Scope on a remote GPU (e.g. RunPod)."
        ),
        json_schema_extra=ui_field_config(order=32, label="Projector Stream"),
    )

    projector_stream_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description=(
            "Port for the MJPEG streaming server. On RunPod, expose this port "
            "and connect to https://<pod-id>-<port>.proxy.runpod.net/"
        ),
        json_schema_extra=ui_field_config(order=33, label="Stream Port"),
    )


class ProMapAnythingPreviewConfig(ProMapAnythingConfig):
    """Preview variant — appears in the **main pipeline selector**.

    Outputs the depth map directly to screen so you can see exactly what the
    preprocessor would feed into ControlNet.
    """

    pipeline_id = "promapanything-vj-preview"
    pipeline_name = "ProMapAnything VJ Preview"
    pipeline_description = (
        "Preview mode — displays the re-projected depth map on screen. "
        "Use this to verify calibration and depth settings before switching "
        "to the preprocessor variant for ControlNet conditioning."
    )

    usage = []


# =============================================================================
# VJ Effects pipeline (standalone, not a preprocessor)
# =============================================================================


class ProMapAnythingEffectsConfig(BasePipelineConfig):
    """VJ effects pipeline — animated depth map distortions.

    Appears in the **main pipeline selector**.  Takes video input (typically
    a depth map from the preprocessor or any camera feed) and applies
    real-time animated effects.

    When ``depth_source`` is set to ``'external_app'``, it auto-detects the
    ProMapAnything standalone app and reads depth/color feeds from it instead
    of using the Scope video input.
    """

    pipeline_id = "promapanything-vj-effects"
    pipeline_name = "ProMapAnything VJ Effects"
    pipeline_description = (
        "Real-time animated effects for depth maps and video — fractal noise, "
        "flow warp, shockwave, room wobble, geometry edges, and more. "
        "Set Depth Source to 'external_app' to receive live depth from the "
        "ProMapAnything standalone app, or use any Scope video input."
    )

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}

    # -- Depth source ---------------------------------------------------------

    depth_source: Literal["video_input", "external_app"] = Field(
        default="video_input",
        description=(
            "Where to get the input from. 'video_input' uses whatever Scope "
            "feeds in (camera, another pipeline). 'external_app' auto-detects "
            "the ProMapAnything standalone app and reads its live depth maps."
        ),
        json_schema_extra=ui_field_config(
            order=1,
            label="Depth Source",
            is_load_param=True,
            category="configuration",
        ),
    )

    external_app_feed: Literal["depth_bw", "depth_color", "projector_rgb"] = Field(
        default="depth_bw",
        description=(
            "Which feed to read from the standalone app. "
            "'depth_bw' = B&W depth, 'depth_color' = colormapped depth, "
            "'projector_rgb' = color-corrected projector image. "
            "Only used when Depth Source = 'external_app'."
        ),
        json_schema_extra=ui_field_config(order=2, label="App Feed"),
    )

    external_app_status: str = Field(
        default="",
        description="Connection status (read-only)",
        json_schema_extra=ui_field_config(order=3, label="App Status"),
    )

    # -- Projector warp -------------------------------------------------------

    projector_warp: bool = Field(
        default=False,
        description=(
            "Warp the input frame to projector perspective before applying "
            "effects. Requires a calibration file. Used in 'Depth -> AI -> Warp' "
            "pipeline mode where the AI output needs perspective correction."
        ),
        json_schema_extra=ui_field_config(order=4, label="Projector Warp"),
    )

    calibration_file: str = Field(
        default="",
        description=(
            "Path to a calibration JSON file for projector warping. If empty, "
            "falls back to the default calibration at "
            "~/.promapanything_calibration.json."
        ),
        json_schema_extra=ui_field_config(
            order=5,
            label="Calibration File",
            is_load_param=True,
            category="configuration",
        ),
    )

    # -- Fractal noise blend --------------------------------------------------

    noise_enabled: bool = Field(
        default=False,
        description="Blend animated fractal noise into the depth map for organic variation",
        json_schema_extra=ui_field_config(order=10, label="Fractal Noise"),
    )

    noise_intensity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="How strongly the noise affects depth values (0 = none, 1 = full)",
        json_schema_extra=ui_field_config(order=11, label="Noise Intensity"),
    )

    noise_scale: float = Field(
        default=4.0,
        ge=0.5,
        le=20.0,
        description="Zoom level of the noise pattern (lower = larger blobs, higher = finer detail)",
        json_schema_extra=ui_field_config(order=12, label="Noise Scale"),
    )

    noise_octaves: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Number of fractal layers stacked together (more = richer detail)",
        json_schema_extra=ui_field_config(order=13, label="Noise Octaves"),
    )

    noise_speed: float = Field(
        default=0.5,
        ge=0.0,
        le=3.0,
        description="Animation speed of the noise pattern",
        json_schema_extra=ui_field_config(order=14, label="Noise Speed"),
    )

    # -- Flow warp ------------------------------------------------------------

    flow_enabled: bool = Field(
        default=False,
        description="Warp the image with an animated noise displacement field",
        json_schema_extra=ui_field_config(order=20, label="Flow Warp"),
    )

    flow_intensity: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Displacement strength (fraction of image size)",
        json_schema_extra=ui_field_config(order=21, label="Flow Intensity"),
    )

    flow_scale: float = Field(
        default=3.0,
        ge=0.5,
        le=15.0,
        description="Scale of the flow displacement pattern",
        json_schema_extra=ui_field_config(order=22, label="Flow Scale"),
    )

    flow_speed: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Animation speed of the flow field",
        json_schema_extra=ui_field_config(order=23, label="Flow Speed"),
    )

    # -- Pulse ----------------------------------------------------------------

    pulse_enabled: bool = Field(
        default=False,
        description="Rhythmic global brightness oscillation — makes the scene breathe",
        json_schema_extra=ui_field_config(order=30, label="Pulse"),
    )

    pulse_speed: float = Field(
        default=0.5,
        ge=0.05,
        le=5.0,
        description="Pulse rate in cycles per second",
        json_schema_extra=ui_field_config(order=31, label="Pulse Speed"),
    )

    pulse_amount: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Pulse amplitude — how much the values swing",
        json_schema_extra=ui_field_config(order=32, label="Pulse Amount"),
    )

    # -- Wave warp ------------------------------------------------------------

    wave_enabled: bool = Field(
        default=False,
        description="Sinusoidal ripple displacement across the image",
        json_schema_extra=ui_field_config(order=40, label="Wave Warp"),
    )

    wave_frequency: float = Field(
        default=3.0,
        ge=0.5,
        le=15.0,
        description="Number of wave cycles across the image",
        json_schema_extra=ui_field_config(order=41, label="Wave Frequency"),
    )

    wave_amplitude: float = Field(
        default=0.05,
        ge=0.0,
        le=0.3,
        description="Wave displacement strength (fraction of image size)",
        json_schema_extra=ui_field_config(order=42, label="Wave Amplitude"),
    )

    wave_speed: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="How fast the wave travels across the image",
        json_schema_extra=ui_field_config(order=43, label="Wave Speed"),
    )

    wave_direction: float = Field(
        default=0.0,
        ge=0.0,
        le=360.0,
        description="Direction the wave travels in degrees (0 = right, 90 = down)",
        json_schema_extra=ui_field_config(order=44, label="Wave Direction"),
    )

    # -- Kaleidoscope ---------------------------------------------------------

    kaleido_enabled: bool = Field(
        default=False,
        description="Mirror the image into radial segments for mandala-like patterns",
        json_schema_extra=ui_field_config(order=50, label="Kaleidoscope"),
    )

    kaleido_segments: int = Field(
        default=6,
        ge=2,
        le=16,
        description="Number of radial mirror segments",
        json_schema_extra=ui_field_config(order=51, label="Segments"),
    )

    kaleido_rotation: float = Field(
        default=0.0,
        ge=0.0,
        le=6.28,
        description="Static rotation offset of the kaleidoscope pattern (radians)",
        json_schema_extra=ui_field_config(order=52, label="Rotation"),
    )

    kaleido_spin_speed: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Auto-rotation speed (radians per second, 0 = static)",
        json_schema_extra=ui_field_config(order=53, label="Spin Speed"),
    )

    # -- Shockwave ------------------------------------------------------------

    shockwave_enabled: bool = Field(
        default=False,
        description=(
            "Radial shockwave that expands from a point. Depth-aware — "
            "nearer surfaces react more, creating parallax."
        ),
        json_schema_extra=ui_field_config(order=60, label="Shockwave"),
    )

    shockwave_origin_x: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Horizontal origin of the shockwave (0 = left, 1 = right)",
        json_schema_extra=ui_field_config(order=61, label="Shock Origin X"),
    )

    shockwave_origin_y: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Vertical origin of the shockwave (0 = top, 1 = bottom)",
        json_schema_extra=ui_field_config(order=62, label="Shock Origin Y"),
    )

    shockwave_speed: float = Field(
        default=0.8,
        ge=0.1,
        le=3.0,
        description="How fast the shockwave ring expands",
        json_schema_extra=ui_field_config(order=63, label="Shock Speed"),
    )

    shockwave_thickness: float = Field(
        default=0.15,
        ge=0.02,
        le=0.5,
        description="Width of the shockwave ring (thinner = sharper)",
        json_schema_extra=ui_field_config(order=64, label="Shock Thickness"),
    )

    shockwave_strength: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Intensity of the depth and spatial displacement",
        json_schema_extra=ui_field_config(order=65, label="Shock Strength"),
    )

    shockwave_decay: float = Field(
        default=1.5,
        ge=0.0,
        le=5.0,
        description="How quickly the shockwave fades with distance",
        json_schema_extra=ui_field_config(order=66, label="Shock Decay"),
    )

    shockwave_interval: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="Auto-trigger interval in seconds (0 = single shot, continuous)",
        json_schema_extra=ui_field_config(order=67, label="Shock Interval"),
    )

    # -- Room Wobble ----------------------------------------------------------

    wobble_enabled: bool = Field(
        default=False,
        description=(
            "Room wobble — surfaces warp like jelly. Depth-aware so nearby "
            "objects wobble more than distant walls."
        ),
        json_schema_extra=ui_field_config(order=70, label="Room Wobble"),
    )

    wobble_intensity: float = Field(
        default=0.08,
        ge=0.0,
        le=0.3,
        description="Wobble displacement strength",
        json_schema_extra=ui_field_config(order=71, label="Wobble Intensity"),
    )

    wobble_speed: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Speed of the wobble animation",
        json_schema_extra=ui_field_config(order=72, label="Wobble Speed"),
    )

    # -- Geometry Edges (Tron lines) ------------------------------------------

    edges_enabled: bool = Field(
        default=False,
        description=(
            "Highlight depth discontinuities (room edges, furniture outlines) "
            "with glowing lines — Tron-style edge tracing."
        ),
        json_schema_extra=ui_field_config(order=80, label="Geometry Edges"),
    )

    edges_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Brightness of the edge highlights",
        json_schema_extra=ui_field_config(order=81, label="Edge Strength"),
    )

    edges_glow_width: float = Field(
        default=3.0,
        ge=0.0,
        le=15.0,
        description="Width of the edge glow (0 = sharp lines, higher = soft glow)",
        json_schema_extra=ui_field_config(order=82, label="Edge Glow Width"),
    )

    edges_pulse_speed: float = Field(
        default=0.0,
        ge=0.0,
        le=3.0,
        description="Pulsing animation speed on the edges (0 = static)",
        json_schema_extra=ui_field_config(order=83, label="Edge Pulse"),
    )

    # -- Depth Fog ------------------------------------------------------------

    fog_enabled: bool = Field(
        default=False,
        description=(
            "Volumetric depth fog — distant surfaces fade into mist, "
            "near surfaces emerge clearly."
        ),
        json_schema_extra=ui_field_config(order=90, label="Depth Fog"),
    )

    fog_density: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Overall fog thickness",
        json_schema_extra=ui_field_config(order=91, label="Fog Density"),
    )

    fog_near: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Depth value where fog begins (0 = closest)",
        json_schema_extra=ui_field_config(order=92, label="Fog Near"),
    )

    fog_far: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Depth value where fog is fully opaque",
        json_schema_extra=ui_field_config(order=93, label="Fog Far"),
    )

    fog_animated: bool = Field(
        default=True,
        description="Animate the fog boundary for a rolling mist effect",
        json_schema_extra=ui_field_config(order=94, label="Fog Animated"),
    )

    fog_speed: float = Field(
        default=0.5,
        ge=0.0,
        le=3.0,
        description="Animation speed of the rolling fog",
        json_schema_extra=ui_field_config(order=95, label="Fog Speed"),
    )

    # -- Radial Zoom ----------------------------------------------------------

    zoom_enabled: bool = Field(
        default=False,
        description=(
            "Radial zoom burst — pixels fly outward from a point. "
            "Depth-aware for parallax tunnel/vortex effect."
        ),
        json_schema_extra=ui_field_config(order=100, label="Radial Zoom"),
    )

    zoom_origin_x: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Horizontal center of the zoom effect",
        json_schema_extra=ui_field_config(order=101, label="Zoom Origin X"),
    )

    zoom_origin_y: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Vertical center of the zoom effect",
        json_schema_extra=ui_field_config(order=102, label="Zoom Origin Y"),
    )

    zoom_strength: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Maximum zoom displacement",
        json_schema_extra=ui_field_config(order=103, label="Zoom Strength"),
    )

    zoom_speed: float = Field(
        default=0.5,
        ge=0.0,
        le=3.0,
        description="Oscillation speed of the zoom effect",
        json_schema_extra=ui_field_config(order=104, label="Zoom Speed"),
    )

    # -- Projector output -----------------------------------------------------

    projector_output: bool = Field(
        default=False,
        description=(
            "Open a fullscreen window on the selected monitor to display the "
            "effects output directly on the projector."
        ),
        json_schema_extra=ui_field_config(order=110, label="Projector Output"),
    )

    projector_monitor: int = Field(
        default=1,
        ge=0,
        le=8,
        description=(
            "Monitor index for the projector output window. 0 = primary monitor, "
            "1 = first secondary monitor, etc."
        ),
        json_schema_extra=ui_field_config(order=111, label="Projector Monitor"),
    )

    # -- Projector stream (remote) --------------------------------------------

    projector_stream: bool = Field(
        default=False,
        description=(
            "Start an MJPEG streaming server so a remote client (or browser) "
            "can display the output on a projector. Use this when running "
            "Scope on a remote GPU (e.g. RunPod)."
        ),
        json_schema_extra=ui_field_config(order=112, label="Projector Stream"),
    )

    projector_stream_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description=(
            "Port for the MJPEG streaming server. On RunPod, expose this port "
            "and connect to https://<pod-id>-<port>.proxy.runpod.net/"
        ),
        json_schema_extra=ui_field_config(order=113, label="Stream Port"),
    )


# =============================================================================
# Projector Output postprocessor
# =============================================================================


class ProMapAnythingProjectorConfig(BasePipelineConfig):
    """Projector output postprocessor.

    Appears in the **Postprocessor** dropdown.  Receives the final output
    from the main pipeline (e.g. Krea) and streams it to a local projector
    via MJPEG.  Pass-through — does not modify the image unless projector
    warp is enabled.
    """

    pipeline_id = "promapanything-projector"
    pipeline_name = "ProMapAnything Projector"
    pipeline_description = (
        "Streams the final pipeline output to a projector. Enable the MJPEG "
        "stream for remote use (RunPod) or the local projector window for "
        "direct display. Optionally warps the output to projector perspective."
    )

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.POSTPROCESSOR]

    # -- Projector warp -------------------------------------------------------

    projector_warp: bool = Field(
        default=False,
        description=(
            "Warp the output to projector perspective using calibration data. "
            "Use this in 'Depth -> AI -> Warp' mode where the AI generates in "
            "camera space and the output needs perspective correction."
        ),
        json_schema_extra=ui_field_config(order=1, label="Projector Warp"),
    )

    calibration_file: str = Field(
        default="",
        description=(
            "Path to a calibration JSON file. If empty, falls back to "
            "~/.promapanything_calibration.json."
        ),
        json_schema_extra=ui_field_config(
            order=2,
            label="Calibration File",
            is_load_param=True,
            category="configuration",
        ),
    )

    # -- Local projector output -----------------------------------------------

    projector_output: bool = Field(
        default=False,
        description=(
            "Open a fullscreen window on the selected monitor to display the "
            "output directly on a locally connected projector."
        ),
        json_schema_extra=ui_field_config(order=10, label="Projector Output"),
    )

    projector_monitor: int = Field(
        default=1,
        ge=0,
        le=8,
        description=(
            "Monitor index for the projector output window. 0 = primary, "
            "1 = first secondary, etc."
        ),
        json_schema_extra=ui_field_config(order=11, label="Projector Monitor"),
    )

    # -- Remote projector stream ----------------------------------------------

    projector_stream: bool = Field(
        default=False,
        description=(
            "Start an MJPEG streaming server so a remote client (or browser) "
            "can display the output on a projector. Use this when running "
            "Scope on a remote GPU (e.g. RunPod)."
        ),
        json_schema_extra=ui_field_config(order=12, label="Projector Stream"),
    )

    projector_stream_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description=(
            "Port for the MJPEG streaming server. On RunPod, expose this port "
            "and connect to https://<pod-id>-<port>.proxy.runpod.net/"
        ),
        json_schema_extra=ui_field_config(order=13, label="Stream Port"),
    )
