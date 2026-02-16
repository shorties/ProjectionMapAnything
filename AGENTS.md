# ProMapAnything — Projection Mapping System

A complete projection mapping solution for projector-camera calibration, depth estimation, and real-time VJ effects. Inspired by Microsoft Research's RoomAlive Toolkit and IllumiRoom projects.

## Project Overview

This repository contains three main components:

1. **promapanything-app** — Standalone desktop application with dual-window OpenGL rendering, structured light calibration, and Spout output
2. **scope-promapanything-vj** — Daydream Scope plugin with 3 pipelines (calibrate, depth, projector)
3. **projector-client** — Companion app that receives MJPEG stream and displays fullscreen on a local projector

### What It Does

- **Structured Light Calibration**: Uses Gray code patterns to establish precise pixel correspondence between projector and camera
- **Depth Estimation**: Monocular depth estimation using Depth Anything V2
- **3D Preview**: Real-time point cloud and mesh visualization with orbit camera (standalone app)
- **Reprojection**: Warps depth maps from camera perspective to projector perspective
- **VJ Effects**: Real-time animated effects (wobble, shockwave, flow warp, kaleidoscope, etc.)
- **Spout Integration**: Shares depth maps, color frames, and projector output with other apps
- **MJPEG Streaming**: Streams output to remote projectors via HTTP (RunPod compatible)

## Repository Structure

```
.
├── promapanything-app/              # Standalone desktop application
│   ├── src/promapanything/          # Main Python package
│   │   ├── app.py                   # Main application (dual-window GLFW setup)
│   │   ├── state.py                 # Central state management (dataclasses)
│   │   ├── ui.py                    # ImGui interface panels
│   │   ├── calibration.py           # Gray code pattern generation/decoding
│   │   ├── procam_calibration.py    # Camera/projector intrinsics + stereo
│   │   ├── depth.py                 # Depth Anything model wrapper
│   │   ├── reproject.py             # Depth map reprojection to projector view
│   │   ├── camera.py                # OpenCV camera capture
│   │   ├── ndi_camera.py            # NDI network camera support
│   │   ├── spout_output.py          # Spout SDK integration
│   │   ├── renderer/                # OpenGL rendering modules
│   │   │   ├── scene.py             # 3D scene (point cloud, mesh, frustum)
│   │   │   └── shaders/             # GLSL shaders
│   │   └── __init__.py
│   └── pyproject.toml
│
├── scope-promapanything-vj/         # Daydream Scope plugin (3 pipelines)
│   ├── src/scope_promapanything_vj/
│   │   ├── __init__.py              # Plugin entry point (hookimpl, registers 3 pipelines)
│   │   ├── schema.py                # Pydantic configs with UI field annotations
│   │   ├── pipeline.py              # CalibratePipeline, DepthPipeline, ProjectorPipeline
│   │   ├── calibration.py           # Gray code calibration state machine
│   │   ├── depth_provider.py        # Depth estimation abstraction (built-in + fallback)
│   │   ├── remap.py                 # Depth warping/adjustments camera→projector
│   │   ├── effects.py               # GPU-accelerated VJ effects (pure torch)
│   │   └── frame_server.py          # MJPEG HTTP server (singleton FrameStreamer)
│   ├── README.md                    # User-facing documentation
│   └── pyproject.toml
│
├── projector-client/                # Companion projector display app
│   ├── projector_client.py          # MJPEG receiver + fullscreen GLFW window
│   └── pyproject.toml
│
├── ProCamCalibration/               # Microsoft RoomAlive Toolkit reference (C#)
│
└── .claude/skills/                  # Claude Code skills
    ├── structured-light/            # Comprehensive calibration theory reference
    └── create-scope-plugin/         # Plugin scaffolding skill + API reference
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
- **Framework**: Daydream Scope plugin system (3 pipelines)
- **Config**: Pydantic with custom UI annotations (`category="input"` / `"configuration"`)
- **Tensors**: PyTorch (THWC format, [0,1] range)
- **Streaming**: MJPEG HTTP server (singleton FrameStreamer)
- **Deps**: opencv-python-headless, transformers (Scope provides torch, numpy, etc.)

### projector-client
- **Display**: GLFW + PyOpenGL fullscreen window
- **Stream**: OpenCV MJPEG decoding from HTTP

## Scope Plugin — 3 Pipeline Architecture

### Pipelines

| Pipeline | Type | ID | Purpose |
|----------|------|-----|---------|
| ProMapAnything Calibrate | Main | `promapanything-calibrate` | Gray code calibration |
| ProMapAnything Depth | Preprocessor | `promapanything-vj-depth` | Depth estimation + projector warp |
| ProMapAnything Projector | Postprocessor | `promapanything-projector` | MJPEG stream to projector |

### Workflow
```
CALIBRATION (standalone — no pre/post needed):
  Main: ProMapAnything Calibrate | Pre: (none) | Post: (none)
  Grey test card → Start Calibration toggle → patterns → decode → save

NORMAL VJ MODE:
  Main: Krea/VACE | Pre: ProMapAnything Depth | Post: ProMapAnything Projector
  Camera → depth → warp → AI generation → MJPEG → projector
```

### Frame Server Endpoints
| Endpoint | Purpose |
|----------|---------|
| `GET /` | Control panel (RunPod auto-detect, Scope URL, projector pop-out) |
| `GET /projector` | Fullscreen MJPEG viewer with calibration download overlay |
| `GET /stream` | MJPEG multipart stream |
| `GET /frame` | Single JPEG snapshot |
| `GET/POST /config` | Projector resolution from browser |
| `GET /calibration/status` | Calibration completion + file list (JSON) |
| `GET /calibration/download/<name>` | Download calibration result files |

### Schema Field Categories
- `category="input"` → Left panel (runtime controls: Start Calibration, smoothing, blur, upscale)
- `category="configuration"` → Right panel (load-time: resolution, ports, paths)

## Key Architecture Patterns

### State Management (standalone app)
Central state in `state.py`:
- `AppState` — Single source of truth
- `AppMode` — IDLE, CALIBRATING, LIVE
- `CalibrationPhase` — IDLE, CAMERA_INTRINSICS, GRAY_CODES, COMPUTING, DONE
- `ProCamCalibration` — K_cam, dist_cam, K_proj, dist_proj, R, T + 2D maps

### Calibration State Machine (Scope plugin)
Phases: `IDLE → WHITE → BLACK → PATTERNS → DECODING → DONE`
- Grey test card before start (prevents feedback loop)
- 6 settle frames between patterns (~200ms at 30fps)
- 3 capture frames per pattern (averaged)
- Per-bit reliability + spatial consistency filtering
- Gaussian splat fill + inpainting for dense correspondence
- Saves to `~/.promapanything_calibration.json` with UTC timestamp
- Uploads results to Scope asset gallery for VACE conditioning

### Dual-Window Architecture (standalone app)
- **Main window**: ImGui controls + 3D viewport (1280x800)
- **Projector window**: Fullscreen output on secondary monitor
- Separate OpenGL contexts — resources created per-context

### Depth Pipeline
```
Camera Frame → Depth Anything V2 → Temporal Smoothing → Projector Warp → Grayscale Output
                                                              ↑
                                                   calibration map_x/map_y
```
Output: grayscale (near=dark, far=bright) matching VACE training data.

## Build and Development

### Install promapanything-app
```bash
cd promapanything-app
pip install -e ".[dev]"
promapanything  # or: python -m promapanything.app
```

### Install Scope plugin (local)
```bash
cd scope-promapanything-vj
pip install -e .
```
Then in Scope: Settings → Plugins → Browse → select folder → Install

### Install Scope plugin (RunPod / remote)
```
git+https://github.com/shorties/ProjectionMapAnything.git#subdirectory=scope-promapanything-vj
```

### RunPod Deployment
- SSH: `mkdir -p /run/sshd && /usr/sbin/sshd` (add to `.bashrc`)
- Expose port 8765 in RunPod settings for MJPEG streamer
- URL pattern: `https://{pod-id}-{port}.proxy.runpod.net/`
- Auto-detected via `RUNPOD_POD_ID` env var
- Killing Scope process restarts the entire container

## Coding Conventions

### Python Style
- **Type hints**: Full annotations, use `from __future__ import annotations`
- **Imports**: Stdlib → third-party → local (alphabetical within groups)
- **Strings**: Double quotes for UI text, single quotes for internal keys
- **Line length**: 100 characters soft limit

### Plugin-Specific Patterns

**Frame format**: Scope may feed float [0,1] or uint8 [0,255] — always detect and convert:
```python
img = tensor.numpy()
if img.dtype != np.uint8:
    if img.max() <= 1.5:
        img = (img * 255.0).clip(0, 255)
    img = img.astype(np.uint8)
```

**Load-time vs Runtime**:
```python
# Load-time: passed to __init__, requires reload
json_schema_extra=ui_field_config(is_load_param=True, category="configuration")

# Runtime: read from kwargs in __call__(), never from self
value = kwargs.get("param_name", default)
```

**Lazy Imports**: Import pipeline classes inside `register_pipelines()`, not at module level.

**Video Tensor Format**:
- Input: List of tensors `(1, H, W, C)` — may be float or uint8
- Output: `{"video": tensor}` — `(T, H, W, C)` float32 [0, 1]

**Validation**: Use `ge=1` for resolution fields (not `ge=640`) — digit-by-digit typing triggers validation.

### GPU Code Style
- All effect functions in `effects.py` are pure torch (no Python loops over pixels)
- Use `torch.nn.functional.grid_sample` for warping
- Hash-based noise for GPU-friendly procedural generation

## Configuration Files

### Calibration (`~/.promapanything_calibration.json`)
```json
{
  "version": 1,
  "projector_width": 1920, "projector_height": 1080,
  "timestamp": "2026-02-15T12:00:00+00:00",
  "map_x": [[...]], "map_y": [[...]]
}
```
Standalone app version also includes K_cam, dist_cam, K_proj, dist_proj, R, T.

### Live Export (`~/.promapanything_live/`)
- `depth_bw.npy`, `depth_color.npy`, `projector_rgb.npy`, `meta.json`

### Spout Outputs (standalone app)
- `ProMap-DepthMap`, `ProMap-ColorMap`, `ProMap-Projector`

## Testing

- **Standalone app**: Run → calibrate → check Spout in OBS/Resolume
- **Scope plugin**: Install → select pipeline → test with camera
- **Calibration**: Check coverage % and inspect warped_camera.png

## Troubleshooting

- **Calibration fails / blank output**: Check frame format (float vs uint8). The `_camera_to_gray` bug (fixed) truncated float [0,1] to all zeros.
- **Feedback loop on projector**: Grey test card should show before calibration, not camera feed.
- **NDI not connecting**: Same network required, try manual source entry.
- **Spout not visible**: Install Spout runtime, verify with Spout demo receiver.
- **SSH on RunPod**: `mkdir -p /run/sshd && /usr/sbin/sshd`, add public key to `~/.ssh/authorized_keys`.

## References

- **RoomAlive Toolkit** — Microsoft Research structured light calibration
- **IllumiRoom** — Projection mapping effects research
- **Depth Anything V2** — Monocular depth estimation (Yang et al.)
- **Daydream Scope** — Real-time AI video pipeline framework
- **VACE** — Video Anything Conditioning Engine (depth conditioning)
