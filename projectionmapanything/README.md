# ProjectionMapAnything — Scope Plugin

A Daydream Scope plugin for AI-powered projection mapping. Calibrates projector-camera geometry using structured light (Gray code patterns), estimates scene depth, and outputs depth maps re-projected to the projector's perspective — ready for VACE depth conditioning.

## Installation

**From local path (development):**
1. Open Scope Settings > Plugins
2. Click **Browse** and select this folder
3. Click **Install**

**From Git URL:**
```
git+https://github.com/shorties/ProjectionMapAnything.git#subdirectory=projectionmapanything
```

## Quick Start

### Calibration

1. Install the plugin and restart Scope
2. Select **ProjectionMapAnything Calibrate** as the main pipeline, hit play
3. Open the projector pop-out window (auto-opens or visit the Stream Port URL)
4. Drag the pop-out to your projector, click to go fullscreen
5. Toggle **Start Calibration** — the plugin projects stripe patterns
6. Wait for all patterns to be captured and decoded
7. Calibration saves automatically; result images upload to the Scope gallery

### Normal VJ Mode

1. Select your AI pipeline (e.g. Krea/VACE) as the main pipeline
2. Select **ProjectionMapAnything Depth** as the preprocessor
3. Select **ProjectionMapAnything Projector** as the postprocessor
4. The preprocessor outputs a depth map conditioned to the projector's perspective
5. The postprocessor streams the AI output to your projector via MJPEG

## Three Pipelines

| Pipeline | Role | Description |
|----------|------|-------------|
| **ProjectionMapAnything Calibrate** | Main pipeline | Gray code structured light calibration |
| **ProjectionMapAnything Depth** | Preprocessor | Depth estimation + projector warp + effects |
| **ProjectionMapAnything Projector** | Postprocessor | MJPEG stream with color correction |

## Depth Preprocessor Parameters

### Load-time (set before streaming)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Depth Mode | `static_depth_warped` | Conditioning mode (see below) |
| Subject Isolation | `none` | Subject isolation strategy |
| Stream Port | 8765 | MJPEG stream port for dashboard |
| Generation Resolution | `half` | Output resolution relative to projector |

### Depth Modes

| Mode | GPU depth? | Description |
|------|-----------|-------------|
| `depth_then_warp` | Yes | Live depth from camera, warped to projector |
| `warp_then_depth` | Yes | Camera warped to projector, then depth |
| `warped_rgb` | No | Live camera RGB warped to projector |
| `static_depth_warped` | No | Saved depth image from calibration |
| `static_depth_from_warped` | No | Saved warp-then-depth image |
| `static_warped_camera` | No | Saved warped camera RGB |
| `custom` | No | User-uploaded depth map via dashboard |

### Runtime (adjustable during streaming)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Temporal Smoothing | 0.5 | Blend with previous depth frame (0-0.99) |
| Depth Blur | 0 | Gaussian blur radius on depth map |
| Edge Erosion | 0 | Erode depth edges inward (pixels) |
| Depth Contrast | 1.0 | Power curve contrast (0.5-5.0) |
| Near Clip | 0.0 | Black out pixels closer than this (0-1) |
| Far Clip | 1.0 | Black out pixels farther than this (0-1) |
| Edge Blend | 0.0 | Blend surface edges into depth (0-1) |
| Edge Method | `sobel` | Edge detection: sobel or canny |
| Active Effect | `none` | Animated depth effect |
| Effect Intensity | 0.5 | Effect strength (0-2) |
| Effect Speed | 1.0 | Animation speed (0-5) |
| Subject Depth Range | 0.3 | Band width for depth_band isolation |
| Subject Feather | 5.0 | Feather radius for isolation mask |

### Depth Effects

All effects are surface-masked: they only animate on projection surfaces (non-black depth regions), giving VACE a clean signal.

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

## Projector Postprocessor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Upscale to Projector | true | Upscale output to projector resolution |
| Edge Feather | 0 | Fade to black at projection edges (pixels) |
| Apply Subject Mask | false | Apply preprocessor isolation mask |
| Brightness | 0.0 | Brightness adjustment (-0.5 to 0.5) |
| Gamma | 1.0 | Gamma correction (0.2-3.0) |
| Contrast | 1.0 | Contrast multiplier (0.5-2.0) |

## Dashboard

The web dashboard runs on the configured stream port (default 8765) with:
- Live preview of projector output and VACE input
- Calibration progress and result downloads
- Custom depth map / mask upload
- Projector connection status
- Scope connection shortcut

## Calibration File Format

Calibration saves to `~/.projectionmapanything_calibration.json` (metadata) + `.npz` (binary maps). Legacy v1 JSON-only format is also supported for loading.

## Dependencies

- `opencv-python-headless` — calibration, image warping
- `transformers` — bundled depth model fallback
- `rembg` — AI background removal (optional, for subject isolation)

Scope provides: `torch`, `numpy`, `pydantic`, `pillow`.

## Development

Edit code, then click **Reload** next to the plugin in Scope's Settings > Plugins. No reinstall needed.
