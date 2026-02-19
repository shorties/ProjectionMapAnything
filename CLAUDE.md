# ProjectionMapAnything — Projection Mapping System

A complete projection mapping solution: camera-projector calibration via structured light, real-time depth estimation, and AI-powered VJ effects projected onto physical surfaces. Inspired by Microsoft's RoomAlive Toolkit.

## Repository Structure

```
ProjectionMapAnything/
├── projectionmapanything/              # Daydream Scope plugin (3 pipelines)
│   └── src/projectionmapanything/
│       ├── __init__.py              # Plugin entry (hookimpl, registers 3 pipelines)
│       ├── schema.py                # Pydantic configs with UI annotations
│       ├── pipeline.py              # 3 pipeline classes (calibrate, depth, projector)
│       ├── calibration.py           # Gray code state machine (CalibrationState)
│       ├── depth_provider.py        # Depth estimation (built-in or transformers)
│       ├── remap.py                 # Depth warp camera→projector
│       ├── effects.py               # GPU effects (pure torch, no loops)
│       ├── isolation.py             # Subject isolation (depth_band, mask, rembg)
│       └── frame_server.py          # MJPEG HTTP server (singleton FrameStreamer)
│
├── promapanything-app/              # Standalone desktop app (Windows, gitignored)
│   └── src/promapanything/
│       ├── app.py                   # Main app (dual-window GLFW, event loop)
│       ├── state.py                 # Central state (AppState, AppMode, CalibrationPhase)
│       ├── ui.py                    # ImGui interface panels
│       ├── calibration.py           # Gray code generation/decoding (CalibrationRunner)
│       ├── procam_calibration.py    # Intrinsics + stereo calibration (RANSAC + LM)
│       ├── depth.py                 # Depth Anything V2 model wrapper
│       ├── camera.py                # OpenCV VideoCapture wrapper
│       ├── ndi_camera.py            # NDI network camera support
│       ├── reproject.py             # Depth reprojection camera→projector
│       ├── spout_output.py          # SpoutGL texture sharing
│       └── renderer/
│           ├── scene.py             # 3D scene (point cloud, mesh, frustum)
│           └── shaders/             # GLSL shaders (effects, pointcloud, mesh, etc.)
│
├── projector-client/                # Companion app for local projector display (gitignored)
│   └── projector_client.py          # MJPEG receiver + fullscreen GLFW window
│
├── ProCamCalibration/               # Microsoft RoomAlive Toolkit reference (gitignored)
│
└── .claude/skills/
    ├── structured-light/SKILL.md    # 61KB calibration theory reference
    └── create-scope-plugin/         # Scope plugin scaffolding skill
        ├── SKILL.md                 # Plugin creation guide
        ├── reference.md             # Complete Scope plugin API spec
        └── examples/vfx-pack.md     # Working plugin example
```

## Scope Plugin Architecture

### Pipelines

| Pipeline | Role | Config Class | Pipeline Class |
|----------|------|--------------|----------------|
| ProjectionMapAnything Depth | **Preprocessor** (primary) | `ProMapAnythingConfig` | `ProMapAnythingPipeline` |
| ProjectionMapAnything Calibrate | **Main pipeline** (visualization) | `ProMapAnythingCalibrateConfig` | `ProMapAnythingCalibratePipeline` |
| ProjectionMapAnything Projector (Legacy) | **Postprocessor** (deprecated) | `ProMapAnythingProjectorConfig` | `ProMapAnythingProjectorPipeline` |

### Pipeline IDs

- `projectionmapanything-depth` — **primary, use this**
- `projectionmapanything-calibrate` — calibration visualization
- `projectionmapanything-projector` — legacy MJPEG delivery

### Workflow

```
CALIBRATION (from within preprocessor):
  Select preprocessor → toggle "Start Calibration" in Scope settings
  Gray code patterns → camera captures → decode → save calibration
  Auto-resumes depth mode when done

NORMAL VJ MODE:
  Main: Krea/VACE    Pre: ProjectionMapAnything Depth    Post: (none needed)
  Camera → depth estimation → warp → effects → isolation → edge feather → AI generation
  AI output delivered via Scope's WebRTC (or legacy postprocessor for MJPEG)
```

### Preprocessor Details (Primary Pipeline)

The preprocessor handles **everything** — calibration, depth conditioning, and all spatial masking:

- **Calibration**: Inline Gray code calibration via `start_calibration` toggle
- Loads calibration automatically from `~/.projectionmapanything_calibration.json`
- Uses Depth Anything V2 (tries Scope built-in first, falls back to transformers)
- Warps depth from camera→projector perspective using calibration maps
- Output: grayscale RGB (near=dark, far=bright) matching VACE training data
- Temporal smoothing + Gaussian blur for flicker/noise reduction
- Edge blend: Sobel/Canny edge detection blended into depth
- Depth effects: Surface-masked animated effects (noise, flow, pulse, wave, kaleido, etc.)
- Subject isolation: depth_band, custom mask, or rembg AI background removal
- Edge feathering: fade depth conditioning to black at projection boundaries
- Custom depth: User-uploaded depth map via dashboard
- Resizes to generation resolution (quarter/half/native of projector res)

**Preprocessor chain**: `depth → edge_processing → edge_blend → surface-masked effect → isolation → edge_feather → resize`

**Calibration state machine**: `IDLE → WHITE → BLACK → PATTERNS → DECODING → DONE`

### Projector Postprocessor (DEPRECATED)

The postprocessor is deprecated. It only provides MJPEG delivery of AI output:
- Edge feathering has **moved to the preprocessor** (applied to depth conditioning)
- Subject masking was **always in the preprocessor** (isolation)
- Color correction (brightness, gamma, contrast) is a client-side concern
- Use Scope's WebRTC output instead of MJPEG for lower latency

## Frame Server (MJPEG HTTP)

Module-level singleton `FrameStreamer` in `frame_server.py`. Both calibration and projector pipelines share the same instance via `get_or_create_streamer()`.

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Control panel (Scope URL, projector button, preview, status, custom upload) |
| `GET /projector` | Fullscreen MJPEG viewer (drag to projector, click fullscreen) |
| `GET /stream` | MJPEG multipart stream |
| `GET /frame` | Single JPEG snapshot |
| `GET/POST /config` | Projector resolution from browser |
| `GET /calibration/status` | Calibration completion status + file list |
| `GET /calibration/download/<name>` | Download calibration result files |
| `POST /upload` | Custom depth map or mask upload (query: stage, type) |
| `GET /upload/status` | Custom upload status (has_custom_depth, has_custom_mask) |

**Calibration priority**: When `calibration_active=True`, normal `submit_frame()` is suppressed — only `submit_calibration_frame()` goes through.

**Auto-download overlay**: Projector page (`/projector`) polls `/calibration/status` every 2s and shows download overlay with calibration.json, coverage_map.png, warped_camera.png when calibration completes.

## Schema Field Placement

Fields use `category` in `ui_field_config()` to control panel placement:

- `category="input"` → **Left panel** (input side) — runtime controls users adjust frequently
- `category="configuration"` → **Right panel** (settings) — load-time config set once

Current preprocessor layout (all "configuration" category — Scope limitation for preprocessors):
- Depth Mode, Temporal Smoothing, Depth Blur, Edge Erosion, Depth Contrast, Near/Far Clip
- Edge Blend, Edge Method
- Active Effect, Effect Intensity, Effect Speed
- Subject Isolation, Depth Range, Feather, Invert Mask
- Edge Feather
- Start Calibration, Calibration Brightness, Projector Width/Height
- Stream Port, Generation Resolution (load-time)

## Critical Implementation Notes

### Frame Format Ambiguity
Scope video input may arrive as **float [0,1] OR uint8 [0,255]**. Always detect and convert:
```python
# WRONG: .astype(np.uint8) on float [0,1] produces ALL ZEROS
img = tensor.numpy().astype(np.uint8)  # BUG!

# RIGHT: check and scale first
img = tensor.numpy()
if img.dtype != np.uint8:
    if img.max() <= 1.5:
        img = (img * 255.0).clip(0, 255)
    img = img.astype(np.uint8)
```

### Feedback Loop Prevention
Never show camera feed on the projector before calibration — the camera sees the projector which shows the camera, creating infinite feedback. Show a neutral grey test card instead.

### Scope Plugin Conventions
- **Input format**: List of tensors `(1, H, W, C)` — may be float [0,1] or uint8 [0,255]
- **Output format**: `{"video": tensor}` — THWC `(T, H, W, C)` float32 [0, 1]
- **Runtime params**: Read from `kwargs.get("param_name", default)` in `__call__()`, never from `self`
- **Load-time params**: Passed to `__init__()` via `**kwargs`, require pipeline reload
- **Lazy imports**: Import pipeline classes inside `register_pipelines()`, not at module level
- **`**kwargs` everywhere**: Always accept in `__init__()`, `prepare()`, `__call__()`

### Scope Asset Gallery / VACE Integration
- Scope has `ref_images` field and `supports_vace` flag in `BasePipelineConfig`
- Asset upload via `POST /api/v1/assets` (multipart form data)
- After calibration, warped images are uploaded to gallery for VACE conditioning
- `vace_context_scale` controls VACE hint injection strength (0.0–2.0)

## Data Files

### Calibration (`~/.projectionmapanything_calibration.json`)
```json
{
  "version": 1,
  "projector_width": 1920,
  "projector_height": 1080,
  "timestamp": "2026-02-15T12:00:00+00:00",
  "map_x": [[...]],     // proj_h × proj_w — maps projector pixels to camera X
  "map_y": [[...]]      // proj_h × proj_w — maps projector pixels to camera Y
}
```
Note: Standalone app version also includes `K_cam`, `dist_cam`, `K_proj`, `dist_proj`, `R`, `T`.

### Results (`~/.projectionmapanything_results/`)
- `custom_depth.png` — User-uploaded custom depth map
- `custom_mask.png` — User-uploaded custom isolation mask
- Calibration result images (coverage map, warped camera, etc.)

### Live Export (`~/.promapanything_live/`)
- `depth_bw.npy` — (H, W) float32 grayscale depth
- `depth_color.npy` — (H, W, 3) uint8 colormapped depth
- `projector_rgb.npy` — (H, W, 3) uint8 projector output
- `meta.json` — timestamp + availability

### Spout Outputs (standalone app only)
- `ProMap-DepthMap` — Normalized B&W depth [0-1]
- `ProMap-ColorMap` — Camera feed
- `ProMap-Projector` — Final projector output

## RunPod Deployment

- **SSH setup**: `mkdir -p /run/sshd && /usr/sbin/sshd` (add to `.bashrc` for persistence)
- **Port 8765**: Must be explicitly exposed in RunPod settings for MJPEG streamer
- **URL pattern**: `https://{pod-id}-{port}.proxy.runpod.net/`
- **Auto-detect**: Uses `RUNPOD_POD_ID` env var for URL construction
- **Plugin install**: `git+https://github.com/shorties/ProjectionMapAnything.git#subdirectory=projectionmapanything`
- **Caution**: Killing Scope process (PID) restarts the entire container

## Technology Stack

| Component | Stack |
|-----------|-------|
| Standalone App | Python 3.12+, GLFW, ModernGL, imgui-bundle, PyTorch, SpoutGL, OpenCV, PyGLM |
| Scope Plugin | Python 3.12+, PyTorch, OpenCV, Pydantic, transformers (fallback depth), rembg |
| Projector Client | Python 3.10+, GLFW, PyOpenGL, OpenCV |
| Reference (C#) | .NET, DirectX, SharpDX, Kinect v2 SDK |

## Coding Conventions

- **Type hints**: Full annotations, use `from __future__ import annotations`
- **Imports**: Stdlib → third-party → local, alphabetical within groups
- **Line length**: 100 chars soft limit
- **GPU code**: Pure torch operations in `effects.py`, no Python loops over pixels
- **Shaders**: GLSL 330 core, standard uniforms: `mvp`, `tex`, `time`, `resolution`
- **Validation**: Use `ge=1` (not `ge=640`) for resolution fields — digit-by-digit typing triggers validation on every keystroke

## Common Tasks

### Adding a VJ Effect
1. Add params to `EffectSettings` in `state.py` (standalone) or `schema.py` (Scope)
2. Add UI controls in `ui.py` or schema field definitions
3. Implement in `effects.py` (pure torch)
4. Add to effects chain in `pipeline.py` or `effects.frag`

### Modifying Calibration
- Gray code logic: `calibration.py` in either app
- Intrinsics/stereo: `procam_calibration.py` (standalone only)
- State machine: `CalibrationState` (plugin) or `CalibrationRunner` (standalone)
- The standalone app uses time-based settling (200ms), the plugin uses frame-counting (6 frames)

### Testing
- **Standalone**: Run app → verify calibration → check Spout output in OBS/Resolume
- **Scope plugin**: Install in Scope → select pipeline → test with camera
- **Calibration validation**: Check decode coverage % and inspect warped_camera.png output
