# ProjectionMapAnything

AI-powered projection mapping with structured light calibration, real-time depth estimation, and VJ effects. Built as a [Daydream Scope](https://scope.daydream.live) plugin.

Inspired by Microsoft's RoomAlive Toolkit / IllumiRoom.

## What it does

1. **Calibrate** your projector-camera setup using Gray code structured light patterns
2. **Estimate depth** from calibration disparity or AI (Depth Anything V2)
3. **Warp** depth maps from camera perspective to projector perspective
4. **Apply effects** — edge blending, animated depth effects, subject isolation
5. **Condition** the AI model (VACE/ControlNet) so generated visuals match your room geometry
6. **Project** the AI output onto physical surfaces via MJPEG streaming

## Installation

**In Scope:**
1. Open Settings > Plugins
2. Install from Git URL:
```
git+https://github.com/shorties/ProjectionMapAnything.git#subdirectory=projectionmapanything
```
3. Restart Scope

## Quick Start

### Calibration

1. Select your AI pipeline (e.g. Krea/VACE) as main
2. Select **Projection-Map-Anything (VJ.Tools)** as preprocessor
3. Open the dashboard (change port 8000 to 8765 in the Scope URL) and click the **Projector** button
4. Drag the projector window to your projector, click to go fullscreen
5. Toggle **Start Calibration** in the Scope settings panel
6. Gray code patterns project automatically — calibration saves when done

### VJ Mode

1. Select your AI pipeline (e.g. Krea/VACE) as main
2. Select **Projection-Map-Anything (VJ.Tools)** as preprocessor
3. Select **ProjectionMapAnything Projector** as postprocessor
4. The depth map conditions the AI; the AI output streams to your projector

## Three Pipelines

| Pipeline | Role | Description |
|----------|------|-------------|
| **ProjectionMapAnything Calibrate** | Main | Gray code structured light calibration |
| **Projection-Map-Anything (VJ.Tools)** | Preprocessor | Depth conditioning + projector warp + effects + inline calibration |
| **ProjectionMapAnything Projector** | Postprocessor | MJPEG stream to projector with color correction |

## Depth Modes

All modes use static calibration data — no live camera processing on the projector output (avoids feedback loops).

| Mode | Description |
|------|-------------|
| **AI Depth** (default) | Depth Anything V2 on raw camera image, warped to projector perspective |
| **Calibration Disparity** | Horizontal disparity from calibration — simple, fast, no AI needed |
| **Edge Detection** | Canny edges on warped camera image |
| **Warped Camera** | RGB camera image warped to projector perspective |
| **Custom Upload** | Upload your own depth map via the dashboard |

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
- Live preview of depth conditioning and projector output
- Projector pop-out window (fullscreen on your projector)
- Calibration progress and result downloads
- Depth mode, effects, isolation controls
- Custom depth map / mask upload

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
