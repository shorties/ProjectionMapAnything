# ProjectionMapAnything

AI-powered projection mapping with structured light calibration, real-time depth estimation, and VJ effects. Built as a [Daydream Scope](https://scope.daydream.live) plugin.

## What it does

1. **Calibrate** your projector-camera setup using Gray code structured light patterns
2. **Estimate depth** from the camera feed using Depth Anything V2
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

1. Select **ProjectionMapAnything Calibrate** as the main pipeline, hit play
2. Open the projector pop-out window (auto-opens or visit the Stream Port URL)
3. Drag the pop-out to your projector, click to go fullscreen
4. Toggle **Start Calibration** — stripe patterns are projected and captured
5. Calibration saves automatically when done

### VJ Mode

1. Select your AI pipeline (e.g. Krea/VACE) as main
2. Select **ProjectionMapAnything Depth** as preprocessor
3. Select **ProjectionMapAnything Projector** as postprocessor
4. The depth map conditions the AI; the output streams to your projector

## Three Pipelines

| Pipeline | Role | Description |
|----------|------|-------------|
| **ProjectionMapAnything Calibrate** | Main | Gray code structured light calibration |
| **ProjectionMapAnything Depth** | Preprocessor | Depth estimation + projector warp + effects |
| **ProjectionMapAnything Projector** | Postprocessor | MJPEG stream with color correction |

## Features

### Depth Modes
- **Live**: `depth_then_warp`, `warp_then_depth`, `warped_rgb` (GPU depth model)
- **Static**: `static_depth_warped`, `static_depth_from_warped`, `static_warped_camera` (from calibration)
- **Custom**: Upload your own depth map via the dashboard

### Depth Effects
Animated effects that modify the depth conditioning signal: noise blend, flow warp, pulse, wave warp, kaleidoscope, shockwave, wobble, geometry edges, depth fog, radial zoom. All effects are surface-masked — they only animate on projection surfaces, leaving void regions black.

### Subject Isolation
- **depth_band** — Auto-detect nearest surfaces via histogram peak
- **mask** — Upload a custom mask image
- **rembg** — AI background removal

### Output Optimization
Edge feathering, subject masking, brightness/gamma/contrast correction on the postprocessor output.

### Dashboard
Web-based control panel at the stream port with live preview, calibration progress, custom depth upload, projector status, and calibration result downloads.

## Requirements

- [Daydream Scope](https://scope.daydream.live)
- Python 3.12+
- A camera and a projector

## License

MIT
