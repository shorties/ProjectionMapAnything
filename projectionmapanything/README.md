# ProjectionMapAnything — Scope Plugin

A [Daydream Scope](https://scope.daydream.live) plugin for AI-powered projection mapping. Calibrates projector-camera geometry using structured light (phase shifting or Gray code), estimates scene depth, and outputs depth maps re-projected to the projector's perspective — ready for VACE/ControlNet depth conditioning.

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

### Calibration (from preprocessor)

1. Install the plugin and restart Scope
2. Select your AI pipeline as main, **ProjectionMapAnything** as preprocessor
3. Open the dashboard (change port 8000 to 8765 in the Scope URL), click **Projector**
4. Drag the projector window to your projector, go fullscreen
5. Toggle **Start Calibration** in the settings panel
6. Patterns are projected and captured automatically
7. Calibration saves when done; result images upload to the Scope gallery

### Normal VJ Mode

1. Select your AI pipeline (e.g. Krea/VACE) as the main pipeline
2. Select **ProjectionMapAnything** as the preprocessor
3. Select **ProjectionMapAnything Projector** as the postprocessor
4. The preprocessor outputs a depth map conditioned to the projector's perspective
5. The postprocessor streams the AI output to your projector via MJPEG

## Three Pipelines

| Pipeline | Role | Description |
|----------|------|-------------|
| **ProjectionMapAnything Calibrate** | Main pipeline | Structured light calibration |
| **ProjectionMapAnything** | Preprocessor | Depth estimation + projector warp + effects + inline calibration |
| **ProjectionMapAnything Projector** | Postprocessor | MJPEG stream with color correction |

## Calibration

### Methods

| Method | Patterns | Best for |
|--------|----------|----------|
| **Phase Shift** (default) | 32 | All setups. Smooth sinusoidal patterns survive JPEG compression, sub-pixel precision. |
| **Gray Code** | 46 | High-quality lossless capture only. Binary stripe patterns. |

### Calibrate Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Start Calibration | off | Toggle to begin calibration sequence |
| Calibration Brightness | 128 | Pattern brightness (0-255) |
| Open Dashboard | off | Open dashboard in browser |
| Reset Calibration | off | Clear saved calibration |
| Calibration Method | `phase_shift` | `phase_shift` or `gray_code` |
| Projector Width | 1920 | Projector output resolution |
| Projector Height | 1080 | Projector output resolution |
| Stream Port | 8765 | Dashboard and MJPEG stream port |
| Settle Frames | 60 | Max frames to wait per pattern (timeout) |
| Capture Frames | 3 | Frames averaged per pattern |

## Depth Preprocessor

### Calibration Controls

| Parameter | Default | Description |
|-----------|---------|-------------|
| Start Calibration | off | Start inline calibration from preprocessor |
| Reset Calibration | off | Clear saved calibration |
| Calibration Speed | 0.85 | 0 = careful (remote), 1 = fast (local) |
| Calibration Method | `phase_shift` | `phase_shift` or `gray_code` |

### Depth Modes

All modes use static calibration data — no live camera processing (avoids camera-projector feedback loops).

| Mode | Description |
|------|-------------|
| `structured_light` (default) | Displacement-based depth from calibration |
| `structured_light_jacobian` | Jacobian-based depth from calibration |
| `canny` | Edge detection on warped camera from calibration |
| `warped_rgb` | Warped camera image from calibration |
| `custom` | Upload your own via the dashboard |

### Depth Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| Temporal Smoothing | 0.0 | Blend with previous depth frame (0-0.99) |
| Depth Blur | 0.0 | Gaussian blur radius on depth map |
| Edge Erosion | 0 | Erode depth edges inward (pixels) |
| Depth Contrast | 1.0 | Power curve contrast (0.5-5.0) |
| Near Clip | 0.0 | Black out closer pixels (0-1) |
| Far Clip | 1.0 | Black out farther pixels (0-1) |
| Edge Blend | 0.0 | Blend surface edges into depth (0-1) |
| Edge Method | `sobel` | Edge detection: `sobel` or `canny` |

### Depth Effects

Surface-masked animated effects — only animate on projection surfaces, leaving void regions black.

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| Active Effect | `none` | Which effect to apply |
| Effect Intensity | 0.5 | Effect strength (0-2) |
| Effect Speed | 1.0 | Animation speed (0-5) |

### Subject Isolation

| Mode | Description |
|------|-------------|
| `none` | No isolation |
| `depth_band` | Auto-detect nearest surfaces via histogram peak |
| `mask` | Upload a custom mask via dashboard |
| `rembg` | AI background removal (requires optional `rembg` dep) |

| Parameter | Default | Description |
|-----------|---------|-------------|
| Subject Depth Range | 0.3 | Band width for depth_band isolation |
| Subject Feather | 5.0 | Feather radius for isolation mask |
| Invert Subject Mask | off | Invert the isolation mask |
| Edge Feather | 0.0 | Fade to black at projection edges |

### Load-time Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| Stream Port | 8765 | Dashboard and MJPEG stream port |

## Projector Postprocessor

| Parameter | Default | Description |
|-----------|---------|-------------|
| Upscale to Projector | true | Upscale output to projector resolution |
| Edge Feather | 0.0 | Fade to black at projection edges (pixels) |
| Apply Subject Mask | false | Apply preprocessor isolation mask |
| Brightness | 0.0 | Brightness adjustment (-0.5 to 0.5) |
| Gamma | 1.0 | Gamma correction (0.2-3.0) |
| Contrast | 1.0 | Contrast multiplier (0.5-2.0) |
| Stream Port | 8765 | MJPEG stream port |

## Dashboard

The web dashboard runs on the configured stream port (default 8765):
- Live preview of projector output
- Projector pop-out window (drag to projector, go fullscreen)
- Calibration progress and result downloads
- Custom depth map / mask upload
- Parameter overrides API

Access by changing port 8000 to 8765 in the Scope URL. On RunPod: replace `-8000.` with `-8765.` in the URL.

## Dependencies

- `opencv-python-headless` — calibration, image warping
- `transformers` — depth model fallback (Depth Anything V2)
- `rembg` — AI background removal (optional)

Scope provides: `torch`, `numpy`, `pydantic`, `pillow`.

## Development

Edit code, then click **Reload** next to the plugin in Scope's Settings > Plugins. No reinstall needed.
