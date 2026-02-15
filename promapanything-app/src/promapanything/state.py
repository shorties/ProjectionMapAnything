"""Shared application state — single source of truth for all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import numpy as np

# Default output directory
DEFAULT_OUTPUT_DIR = str(Path.home() / ".promapanything_output")


class AppMode(Enum):
    IDLE = auto()
    CALIBRATING = auto()
    LIVE = auto()


class CalibrationPhase(Enum):
    """Phase within the multi-step ProCam calibration wizard."""
    IDLE = auto()
    CAMERA_INTRINSICS = auto()    # Project checkerboard, detect in camera
    GRAY_CODES = auto()           # Structured light patterns on projector
    COMPUTING = auto()            # Stereo calibration running
    DONE = auto()


class MainViewMode(Enum):
    """What to show in the main viewport."""
    SCENE_3D = auto()      # Point cloud / mesh
    CAMERA = auto()        # Raw webcam feed
    DEPTH = auto()         # Depth map visualization
    PROJECTOR = auto()     # What the projector outputs (effects/remap)


class ViewMode(Enum):
    POINT_CLOUD = auto()
    MESH = auto()


class CameraSource(Enum):
    """Where the camera feed comes from."""
    LOCAL = auto()         # Local webcam (OpenCV VideoCapture)
    NDI = auto()           # NDI network source (e.g. iPhone NDI HX Camera)


class ProjectorSource(Enum):
    """What to display on the projector window."""
    EFFECTS = auto()       # Local depth + effects shader
    SPOUT_INPUT = auto()   # Incoming Spout source (e.g. Scope output)


@dataclass
class ProCamCalibration:
    """Full ProCam calibration results."""
    # Camera intrinsics
    K_cam: np.ndarray | None = None        # 3x3
    dist_cam: np.ndarray | None = None     # 1x5

    # Projector intrinsics
    K_proj: np.ndarray | None = None       # 3x3
    dist_proj: np.ndarray | None = None    # 1x5

    # Stereo extrinsics (camera → projector)
    R: np.ndarray | None = None            # 3x3 rotation
    T: np.ndarray | None = None            # 3x1 translation

    # Gray code correspondence (kept for fast 2D fallback)
    map_x: np.ndarray | None = None
    map_y: np.ndarray | None = None


@dataclass
class CameraSettings:
    """Camera image adjustments."""
    brightness: float = 0.0    # -1.0 to 1.0
    contrast: float = 1.0      # 0.5 to 3.0
    exposure: float = 0.0      # camera-dependent, 0 = auto


@dataclass
class CalibrationSettings:
    settle_ms: float = 200.0  # Milliseconds between patterns
    projector_width: int = 1920
    projector_height: int = 1080
    board_cols: int = 5       # Inner corners (columns) — lower for low-res projectors
    board_rows: int = 3       # Inner corners (rows)
    board_white: int = 255    # White square brightness (0-255)
    board_black: int = 0      # Black square brightness (0-255)
    square_px: int = 80       # Square size in projector pixels
    skip_checkerboard: bool = True   # Skip checkerboard intrinsics, go straight to Gray codes
    # Decode quality
    capture_frames: int = 3            # Frames to average per pattern (1-10)
    decode_threshold: float = 20.0     # White-black diff threshold (was hardcoded 30)
    morph_cleanup: bool = True         # Morphological close + median filter after decode
    morph_kernel_size: int = 5         # Kernel size for cleanup (odd, 3-15)
    spatial_consistency: bool = True   # Reject spatially inconsistent decoded coords
    consistency_max_diff: float = 3.0  # Max deviation from local median (projector pixels)
    bit_threshold: float = 3.0         # Min |pos-neg| per bit to trust the decode (rejects bit-boundary pixels)
    fill_kernel_size: int = 11         # Gaussian splat kernel to fill gaps in projector space (odd, 1=off)


@dataclass
class DepthSettings:
    scale: float = 1.0
    offset: float = 0.0
    blur: float = 0.0
    invert: bool = False
    colormap: str = "grayscale"  # grayscale, turbo, viridis, magma
    temporal_smoothing: float = 0.5
    use_projector_perspective: bool = True  # If False, show camera-perspective depth even when calibrated
    auto_open_reprojection: bool = True  # Auto-open reprojection image after calibration
    use_static_depth: bool = True  # Use the static depth map captured at calibration time
    # ControlNet output adjustments (applied when saving/previewing)
    output_clip_lo: float = 2.0    # Percentile low clip (0-49) — clip darkest 2%
    output_clip_hi: float = 98.0   # Percentile high clip (51-100) — clip brightest 2%
    output_brightness: float = 0.0  # -1.0 to 1.0
    output_contrast: float = 1.0   # 0.5 to 3.0
    output_gamma: float = 1.0     # 0.2 to 5.0 (< 1 brightens darks, > 1 darkens)
    output_equalize: bool = True   # Apply CLAHE histogram equalization for max contrast


@dataclass
class EffectSettings:
    # Fractal noise
    noise_enabled: bool = False
    noise_intensity: float = 0.3
    noise_scale: float = 4.0
    noise_octaves: int = 4
    noise_speed: float = 0.5

    # Flow warp
    flow_enabled: bool = False
    flow_intensity: float = 0.15
    flow_scale: float = 3.0
    flow_speed: float = 0.3

    # Pulse
    pulse_enabled: bool = False
    pulse_speed: float = 0.5
    pulse_amount: float = 0.3

    # Wave warp
    wave_enabled: bool = False
    wave_frequency: float = 3.0
    wave_amplitude: float = 0.05
    wave_speed: float = 1.0
    wave_direction: float = 0.0

    # Kaleidoscope
    kaleido_enabled: bool = False
    kaleido_segments: int = 6
    kaleido_rotation: float = 0.0
    kaleido_spin_speed: float = 0.0

    # Shockwave
    shockwave_enabled: bool = False
    shockwave_origin_x: float = 0.5
    shockwave_origin_y: float = 0.5
    shockwave_speed: float = 0.8
    shockwave_thickness: float = 0.15
    shockwave_strength: float = 0.4
    shockwave_decay: float = 1.5
    shockwave_interval: float = 2.0

    # Room wobble
    wobble_enabled: bool = False
    wobble_intensity: float = 0.08
    wobble_speed: float = 1.0

    # Geometry edges
    edges_enabled: bool = False
    edges_strength: float = 0.5
    edges_glow_width: float = 3.0
    edges_pulse_speed: float = 0.0

    # Depth fog
    fog_enabled: bool = False
    fog_density: float = 0.6
    fog_near: float = 0.3
    fog_far: float = 0.9
    fog_animated: bool = True
    fog_speed: float = 0.5

    # Radial zoom
    zoom_enabled: bool = False
    zoom_origin_x: float = 0.5
    zoom_origin_y: float = 0.5
    zoom_strength: float = 0.15
    zoom_speed: float = 0.5


@dataclass
class OrbitCamera:
    """Simple orbit camera for 3D preview."""
    yaw: float = 0.0
    pitch: float = -30.0
    distance: float = 3.0
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0
    fov: float = 60.0


@dataclass
class AppState:
    mode: AppMode = AppMode.IDLE
    main_view: MainViewMode = MainViewMode.CAMERA
    view_mode: ViewMode = ViewMode.POINT_CLOUD

    # Monitor selection (index into glfw.get_monitors())
    projector_monitor_idx: int = 0
    camera_device_idx: int = 0

    # Camera source
    camera_source: CameraSource = CameraSource.LOCAL
    ndi_source_name: str = ""         # Selected NDI source name

    # Settings
    camera_settings: CameraSettings = field(default_factory=CameraSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    depth: DepthSettings = field(default_factory=DepthSettings)
    effects: EffectSettings = field(default_factory=EffectSettings)
    orbit: OrbitCamera = field(default_factory=OrbitCamera)

    # Projector source
    projector_source: ProjectorSource = ProjectorSource.EFFECTS

    # Spout output
    spout_enabled: bool = True
    spout_depth_name: str = "ProMap-DepthMap"
    spout_color_name: str = "ProMap-ColorMap"
    spout_projector_name: str = "ProMap-Projector"

    # Spout input (receive from Scope/other app)
    spout_receive_enabled: bool = False
    spout_receive_name: str = ""  # empty = auto-detect first sender

    # Output directory for all saved files (depth maps, RGB, calibration)
    output_dir: str = DEFAULT_OUTPUT_DIR

    # Live depth export (shared file for Scope plugin to read)
    live_depth_export: bool = True

    # ProCam calibration state
    calib_phase: CalibrationPhase = CalibrationPhase.IDLE
    procam: ProCamCalibration = field(default_factory=ProCamCalibration)

    # Calibration data (populated after calibration)
    calib_map_x: np.ndarray | None = None
    calib_map_y: np.ndarray | None = None
    calibration_progress: float = 0.0
    calibration_file: str = ""

    # Runtime textures (OpenGL texture IDs, set by renderers)
    camera_texture_id: int = 0
    depth_texture_id: int = 0

    # Frame data (numpy arrays, updated by camera/depth threads)
    camera_frame: np.ndarray | None = None  # (H, W, 3) BGR uint8
    depth_frame: np.ndarray | None = None   # (H, W) float32 [0, 1]
    
    # Static depth map captured at calibration time (projector perspective)
    static_depth_map: np.ndarray | None = None  # (proj_h, proj_w) float32 [0, 1]
    static_depth_captured: bool = False
    # Raw warped depth (before normalization) for live adjustment
    raw_warped_depth: np.ndarray | None = None  # (proj_h, proj_w) float32, raw values

    # Timing
    fps: float = 0.0
    frame_time: float = 0.0
