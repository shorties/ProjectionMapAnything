# ProMapAnything — Projection Mapping System

A complete projection mapping solution for projector-camera calibration, depth estimation, and real-time VJ effects. Inspired by Microsoft Research's RoomAlive Toolkit and IllumiRoom projects.

## Project Overview

This repository contains two main components that work together:

1. **promapanything-app** — Standalone desktop application with dual-window OpenGL rendering, structured light calibration, and Spout output
2. **scope-promapanything-vj** — Daydream Scope plugin for AI-powered video pipelines

### What It Does

- **Structured Light Calibration**: Uses Gray code patterns to establish precise pixel correspondence between projector and camera
- **Depth Estimation**: Monocular depth estimation using Depth Anything V2
- **3D Preview**: Real-time point cloud and mesh visualization with orbit camera
- **Reprojection**: Warps depth maps from camera perspective to projector perspective
- **VJ Effects**: Real-time animated effects (wobble, shockwave, flow warp, kaleidoscope, etc.)
- **Spout Integration**: Shares depth maps, color frames, and projector output with other apps

## Repository Structure

```
.
├── promapanything-app/           # Standalone desktop application
│   ├── src/promapanything/       # Main Python package
│   │   ├── app.py                # Main application (dual-window GLFW setup)
│   │   ├── state.py              # Central state management (dataclasses)
│   │   ├── ui.py                 # ImGui interface panels
│   │   ├── calibration.py        # Gray code pattern generation/decoding
│   │   ├── procam_calibration.py # Camera/projector intrinsics + stereo
│   │   ├── depth.py              # Depth Anything model wrapper
│   │   ├── reproject.py          # Depth map reprojection to projector view
│   │   ├── camera.py             # OpenCV camera capture
│   │   ├── ndi_camera.py         # NDI network camera support
│   │   ├── spout_output.py       # Spout SDK integration
│   │   ├── renderer/             # OpenGL rendering modules
│   │   │   ├── scene.py          # 3D scene (point cloud, mesh, frustum)
│   │   │   └── shaders/          # GLSL shaders
│   │   └── __init__.py
│   ├── assets/                   # Model weights (depth_anything_v2_vits.pth)
│   └── pyproject.toml
│
├── scope-promapanything-vj/      # Daydream Scope plugin
│   ├── src/scope_promapanything_vj/
│   │   ├── __init__.py           # Plugin entry point (hookimpl)
│   │   ├── schema.py             # Pydantic config with UI field definitions
│   │   ├── pipeline.py           # Three pipeline implementations
│   │   ├── calibration.py        # Calibration state machine
│   │   ├── depth_provider.py     # Depth estimation abstraction
│   │   ├── remap.py              # Depth warping/adjustments
│   │   └── effects.py            # GPU-accelerated VJ effects
│   ├── README.md                 # User-facing documentation
│   └── pyproject.toml
│
└── .claude/skills/               # Claude Code skills
    ├── structured-light/         # Comprehensive calibration reference
    └── create-scope-plugin/      # Plugin scaffolding skill
```

## Technology Stack

### promapanything-app
- **Windowing**: GLFW with dual OpenGL contexts
- **Rendering**: ModernGL (OpenGL 3.3 Core)
- **UI**: imgui-bundle (Dear ImGui)
- **Depth Model**: Depth Anything V2 (PyTorch)
- **Inter-process**: Spout (DirectX/OpenGL texture sharing)
- **Camera**: OpenCV VideoCapture + NDI HX Camera support

### scope-promapanything-vj
- **Framework**: Daydream Scope plugin system
- **Config**: Pydantic with custom UI annotations
- **Tensors**: PyTorch (THWC format, [0,1] range)
- **Minimal deps**: Only `opencv-python-headless` (Scope provides torch, numpy, etc.)

## Key Architecture Patterns

### State Management
Central state in `promapanything-app/src/promapanything/state.py`:
- `AppState` — Single source of truth
- `AppMode` — IDLE, CALIBRATING, LIVE
- `CalibrationPhase` — IDLE, CAMERA_INTRINSICS, GRAY_CODES, COMPUTING, DONE
- `ProCamCalibration` — Stores K_cam, dist_cam, K_proj, dist_proj, R, T + 2D maps

### Dual-Window Architecture
- **Main window**: ImGui controls + 3D viewport (1280x800)
- **Projector window**: Fullscreen output on secondary monitor
- Separate OpenGL contexts (no sharing) — resources created per-context

### Calibration Pipeline
1. **Checkerboard Phase** (optional): Project checkerboards, detect corners → compute intrinsics
2. **Gray Code Phase**: Project structured light patterns → decode correspondence
3. **Stereo Phase**: Use dense correspondences + depth → compute R, T via RANSAC + LM

### Depth Pipeline
```
Camera Frame → Depth Estimator → Reprojector → Projector-view Depth Map
                                    ↑
                         (uses calibration maps)
```

## Build and Development

### Install promapanything-app (development)
```bash
cd promapanything-app
pip install -e ".[dev]"
```

### Run the app
```bash
promapanything
# or
python -m promapanything.app
```

### Install Scope plugin
```bash
cd scope-promapanything-vj
pip install -e .
```

Then in Scope: Settings → Plugins → Browse → select `scope-promapanything-vj` folder → Install

## Coding Conventions

### Python Style
- **Type hints**: Full annotations, use `from __future__ import annotations`
- **Imports**: Stdlib → third-party → local (each group alphabetically)
- **Strings**: Double quotes for UI text, single quotes for internal keys
- **Line length**: 100 characters soft limit

### Project-Specific Patterns

**Load-time vs Runtime Parameters** (Scope plugin):
```python
# Load-time: passed to __init__, requires reload to change
json_schema_extra=ui_field_config(is_load_param=True, category="configuration")

# Runtime: read from kwargs in __call__
calibrate = kwargs.get("calibrate", False)
```

**Lazy Imports in Plugin**:
```python
@hookimpl
def register_pipelines(register):
    from .pipeline import MyPipeline  # Import inside function
    register(MyPipeline)
```

**Video Tensor Format**:
- Input: List of tensors `(1, H, W, C)` uint8 [0, 255]
- Output: `{"video": tensor}` where tensor is `(1, H, W, C)` float32 [0, 1]

### GPU Code Style
- All effect functions in `effects.py` are pure torch (no Python loops over pixels)
- Use `torch.nn.functional.grid_sample` for warping
- Hash-based noise for GPU-friendly procedural generation

### Shader Organization
- `fullscreen.vert` — Standard full-screen pass-through vertex shader
- `effects.frag` — Main effects composition shader
- `pointcloud.vert/frag` — Point cloud rendering
- `mesh.vert/frag` — Mesh/depth map rendering
- `frustum.vert/frag` — Projector frustum visualization

## Testing Strategy

- **Manual testing**: Run app, verify calibration flow, check Spout output in OBS/Resolume
- **Scope plugin**: Test in Scope with real camera feed
- **Calibration validation**: Check decode coverage % and reprojection errors in UI

## Configuration Files

### Calibration File Format
`~/.promapanything_calibration.json`:
```json
{
  "version": 1,
  "projector_width": 1920,
  "projector_height": 1080,
  "K_cam": [[...]],           // 3x3 camera intrinsics
  "dist_cam": [...],          // 1x5 distortion
  "K_proj": [[...]],          // 3x3 projector intrinsics
  "dist_proj": [...],         // 1x5 distortion
  "R": [[...]],               // 3x3 rotation (cam→proj)
  "T": [...],                 // 3x1 translation
  "map_x": [[...]],           // projector_height x projector_width
  "map_y": [[...]]            // maps projector pixels to camera pixels
}
```

### Live Export Directory
`~/.promapanything_live/` — Shared files for Scope plugin:
- `depth_bw.npy` — B&W depth map (H, W) float32
- `depth_color.npy` — Colormapped depth (H, W, 3) uint8
- `projector_rgb.npy` — Projector output (H, W, 3) uint8
- `meta.json` — Timestamp and feed availability

## Common Tasks

### Adding a New VJ Effect
1. Add effect parameters to `EffectSettings` in `state.py`
2. Add UI controls in `ui.py` (or `schema.py` for Scope)
3. Implement effect function in `effects.py` (pure torch, no loops)
4. Add to effects chain in `pipeline.py` or `effects.frag`

### Adding a New Shader
1. Create `.vert` and `.frag` files in `shaders/`
2. Load in `app.py` `_create_gl_resources()` or `scene.py`
3. Use standard uniforms: `mvp`, `tex`, `time`, `resolution`

### Modifying Calibration
1. Gray code logic: `calibration.py`
2. Intrinsics/stereo: `procam_calibration.py`
3. UI feedback: `app.py` `_update_calib_preview()`

## External Dependencies

### Required Hardware
- Projector (secondary display)
- Camera (webcam or NDI source)
- Windows PC with OpenGL 3.3+ support

### Spout Receivers
The app outputs to named Spout sources:
- `ProMap-DepthMap` — B&W depth (normalized 0-1)
- `ProMap-ColorMap` — Original camera feed
- `ProMap-Projector` — Final projector output

## Troubleshooting

**Calibration fails to decode**: Increase settle time (camera latency)
**NDI not connecting**: Ensure source is on same network, try manual source entry
**Spout not visible**: Install Spout runtime, check sender exists with Spout demo receiver
**Depth model slow**: First run downloads weights; subsequent runs use cached model

## References

- **RoomAlive Toolkit** — Microsoft Research structured light calibration
- **IllumiRoom** — Projection mapping effects research
- **Depth Anything V2** — Monocular depth estimation (Yang et al.)
- **Gray Code** — Robust binary encoding for structured light
