# ProMapAnything VJ Tools

A Daydream Scope plugin for AI-powered projection mapping. Calibrates projector-camera geometry using structured light (Gray code patterns), estimates scene depth, and outputs a depth map re-projected to the projector's perspective — ready for VACE depth conditioning.

## How it works

1. **Calibrate** — The calibration pipeline projects Gray code stripe patterns through Scope's output. Your camera captures each pattern, and the plugin decodes the projector-camera pixel correspondence automatically.
2. **Estimate depth** — Uses monocular depth estimation (Scope's built-in Depth Anything or a bundled fallback) to compute per-pixel depth from the camera feed.
3. **Re-project** — Warps the camera-perspective depth map into the projector's perspective using the calibration mapping.
4. **Condition** — The re-projected depth map feeds into the AI model as VACE conditioning, so generated visuals align with the physical scene geometry.

## Installation

**From local path (development):**
1. Open Scope Settings Plugins
2. Click **Browse** and select this folder
3. Click **Install**

**From Git URL:**
```
git+https://github.com/shorties/ProjectionMapAnything.git#subdirectory=scope-promapanything-vj
```

## Quick start

### Calibration

1. Install the plugin and restart Scope
2. Select **ProMapAnything Calibrate** as the main pipeline, hit play
3. Open the projector pop-out window (auto-opens or visit the Stream Port URL)
4. Drag the pop-out to your projector, click to go fullscreen
5. Toggle **Start Calibration** — the plugin projects stripe patterns
6. Wait for all patterns to be captured and decoded
7. Calibration saves automatically; result images upload to the Scope gallery

### Normal VJ mode

1. Select your AI pipeline (e.g. Krea/VACE) as the main pipeline
2. Select **ProMapAnything Depth** as the preprocessor
3. Select **ProMapAnything Projector** as the postprocessor
4. The preprocessor outputs a depth map conditioned to the projector's perspective
5. The postprocessor streams the AI output to your projector via MJPEG

## Three pipelines

| Pipeline | Role | Description |
|----------|------|-------------|
| **ProMapAnything Calibrate** | Main pipeline | Gray code structured light calibration |
| **ProMapAnything Depth** | Preprocessor | Depth estimation + projector warp |
| **ProMapAnything Projector** | Postprocessor | MJPEG stream to projector |

## Depth preprocessor parameters

### Load-time (set before streaming)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Depth Mode | `static_depth_warped` | Conditioning mode (see below) |
| Stream Port | 8765 | MJPEG stream port for dashboard |
| Generation Resolution | `half` | Output resolution relative to projector |

### Depth modes

| Mode | GPU depth? | Description |
|------|-----------|-------------|
| `depth_then_warp` | Yes | Live depth from camera, warped to projector |
| `warp_then_depth` | Yes | Camera warped to projector, then depth |
| `warped_rgb` | No | Live camera RGB warped to projector |
| `static_depth_warped` | No | Saved depth-then-warp image from calibration |
| `static_depth_from_warped` | No | Saved warp-then-depth image from calibration |
| `static_warped_camera` | No | Saved warped camera RGB from calibration |

### Runtime (adjustable during streaming)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Temporal Smoothing | 0.5 | Blend with previous depth frame (0-0.99) |
| Depth Blur | 0 | Gaussian blur radius on depth map |
| Edge Erosion | 0 | Erode depth edges inward (pixels) |
| Depth Contrast | 1.0 | Power curve contrast (0.5-5.0) |
| Near Clip | 0.0 | Black out pixels closer than this (0-1) |
| Far Clip | 1.0 | Black out pixels farther than this (0-1) |

## Calibration file format

The plugin saves calibration to `~/.promapanything_calibration.json` (metadata) + `.npz` (binary maps). Legacy v1 JSON-only format is also supported for loading.

`map_x` and `map_y` are 2D arrays of shape `(projector_height, projector_width)` containing the camera pixel coordinates that correspond to each projector pixel.

## Development

Edit code, then click **Reload** next to the plugin in Scope's Settings Plugins. No reinstall needed.

## Dependencies

- `opencv-python-headless` — calibration pattern detection and image warping
- `transformers` — bundled depth model fallback (only if Scope's Depth Anything is not available)

Scope provides: `torch`, `numpy`, `pydantic`, `pillow`.
