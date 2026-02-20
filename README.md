# ProjectionMapAnything

AI-powered projection mapping with structured light calibration, real-time depth estimation, and VJ effects. Built as a [Daydream Scope](https://scope.daydream.live) plugin.

## What it does

1. **Calibrate** your projector-camera setup using structured light patterns (phase shifting or Gray code)
2. **Estimate depth** from calibration data or live camera feed using Depth Anything V2
3. **Warp** the depth map from camera perspective to projector perspective
4. **Apply effects** — edge blending, animated depth effects, subject isolation
5. **Condition** the AI model (VACE/ControlNet) so generated visuals match your room geometry
6. **Project** the AI output onto physical surfaces via MJPEG streaming

## Installation

**In Scope:**
1. Open Settings > Plugins
2. Click **Browse** and select the `projectionmapanything/` folder, or install from Git URL:
```
git+https://github.com/shorties/ProjectionMapAnything.git#subdirectory=projectionmapanything
```
3. Restart Scope

## Quick Start

### Calibration

There are two ways to calibrate:

**From the preprocessor (recommended):**
1. Select your AI pipeline (e.g. Krea/VACE) as main
2. Select **ProjectionMapAnything** as preprocessor
3. Open the dashboard (change port 8000 to 8765 in the Scope URL) and click the **Projector** button
4. Drag the projector window to your projector, click to go fullscreen
5. Toggle **Start Calibration** in the Scope settings panel
6. Calibration runs automatically and saves when done

**From the calibrate pipeline:**
1. Select **ProjectionMapAnything Calibrate** as the main pipeline, hit play
2. Open the projector pop-out (dashboard port 8765 → Projector button)
3. Drag to your projector, go fullscreen
4. Toggle **Start Calibration**

### VJ Mode

1. Select your AI pipeline (e.g. Krea/VACE) as main
2. Select **ProjectionMapAnything** as preprocessor
3. Select **ProjectionMapAnything Projector** as postprocessor
4. The depth map conditions the AI; the output streams to your projector

## Three Pipelines

| Pipeline | Role | Description |
|----------|------|-------------|
| **ProjectionMapAnything Calibrate** | Main | Structured light calibration (phase shift / Gray code) |
| **ProjectionMapAnything** | Preprocessor | Depth estimation + projector warp + effects + inline calibration |
| **ProjectionMapAnything Projector** | Postprocessor | MJPEG stream with color correction |

## Calibration Methods

| Method | Patterns | Description |
|--------|----------|-------------|
| **Phase Shift** (default) | 32 | Smooth sinusoidal patterns. Survives JPEG compression, sub-pixel precision. |
| **Gray Code** | 46 | Binary stripe patterns. Requires high-quality capture. |

Phase shifting is recommended for all setups, especially remote/RunPod where the MJPEG stream uses JPEG compression.

## Depth Modes

All modes use static calibration data — no live camera processing on the projector output (avoids feedback loops).

| Mode | Description |
|------|-------------|
| `structured_light` (default) | Displacement-based depth from calibration |
| `structured_light_jacobian` | Jacobian-based depth from calibration |
| `canny` | Edge detection on warped camera image from calibration |
| `warped_rgb` | Warped camera image from calibration |
| `custom` | Upload your own depth map via the dashboard |

## Depth Effects

Animated effects that modify the depth conditioning signal. All effects are surface-masked — they only animate on projection surfaces, leaving void regions black.

| Effect | Description |
|--------|-------------|
| `noise_blend` | Animated fractal noise blended into depth |
| `flow_warp` | Organic morphing distortion via noise field |
| `pulse` | Rhythmic depth oscillation (breathing) |
| `wave_warp` | Sinusoidal ripple distortion |
| `kaleido` | Kaleidoscope symmetry (mandala patterns) |
| `shockwave` | Radial shockwave from center (depth-aware) |
| `wobble` | Room jelly wobble (depth-aware) |
| `geometry_edges` | Glowing outlines on depth edges |
| `depth_fog` | Animated fog rolling through depth layers |
| `radial_zoom` | Tunnel zoom burst from center (depth-aware) |

## Subject Isolation

| Mode | Description |
|------|-------------|
| `depth_band` | Auto-detect nearest surfaces via histogram peak |
| `mask` | Upload a custom mask image via dashboard |
| `rembg` | AI background removal (requires `rembg` optional dependency) |

## Dashboard

Web-based control panel on the stream port (default 8765):
- Live preview of projector output
- Projector pop-out window (fullscreen on your projector)
- Calibration progress and result downloads
- Custom depth map / mask upload
- Parameter overrides

Access it by changing the port in your Scope URL from 8000 to 8765. On RunPod, replace `-8000.` with `-8765.` in the URL.

## RunPod Deployment

- Port 8765 must be explicitly exposed in RunPod settings
- URL pattern: `https://{pod-id}-8765.proxy.runpod.net/`
- Install from Git URL in Scope's plugin settings:
```
git+https://github.com/shorties/ProjectionMapAnything.git#subdirectory=projectionmapanything
```

## Requirements

- [Daydream Scope](https://scope.daydream.live)
- Python 3.12+
- A camera and a projector

## License

MIT
