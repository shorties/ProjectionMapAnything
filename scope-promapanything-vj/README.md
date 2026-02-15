# ProMapAnything VJ Tools

A Daydream Scope preprocessor plugin for AI-powered projection mapping. Calibrates projector-camera geometry using structured light (Gray code patterns), estimates scene depth, and outputs a depth map re-projected to the projector's perspective — ready for ControlNet depth conditioning.

## How it works

1. **Calibrate** — Toggle calibration mode. The plugin projects Gray code stripe patterns through Scope's output. Your camera captures each pattern, and the plugin decodes the projector↔camera pixel correspondence automatically.
2. **Estimate depth** — Uses monocular depth estimation (Scope's built-in Depth Anything or a bundled fallback) to compute per-pixel depth from the camera feed.
3. **Re-project** — Warps the camera-perspective depth map into the projector's perspective using the calibration mapping.
4. **Condition** — The re-projected depth map feeds into ControlNet as depth conditioning, so AI-generated visuals align with the physical scene geometry.

## Installation

**From local path (development):**
1. Open Scope → Settings → Plugins
2. Click **Browse** and select this folder
3. Click **Install**

**From Git URL:**
```
git+https://github.com/YOUR_USER/scope-promapanything-vj.git
```

## Quick start

1. Install the plugin and restart Scope
2. Select **ProMapAnything VJ Tools** from the Preprocessor dropdown
3. Set your **Projector Width** and **Projector Height** to match your projector
4. Connect your camera as the video source
5. Toggle **Calibrate** on — the plugin projects stripe patterns
6. Wait for all patterns to be captured (~2-3 seconds at 30fps)
7. Toggle **Calibrate** off — the plugin now outputs the re-projected depth map
8. Select a ControlNet depth pipeline as your main pipeline

## Parameters

### Load-time (set before streaming)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Depth Provider | `auto` / `bundled` | `auto` | `auto` tries Scope's built-in Depth Anything first |
| Projector Width | 640–3840 | 1920 | Projector output resolution width |
| Projector Height | 480–2160 | 1080 | Projector output resolution height |
| Calibration File | path | _(empty)_ | Load a pre-computed calibration JSON instead of calibrating interactively |

### Runtime (adjustable during streaming)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Calibrate | toggle | off | Enter/exit calibration mode |
| Settle Frames | 1–10 | 3 | Frames to wait between projected patterns (increase for high-latency setups) |
| Depth Scale | 0.1–3.0 | 1.0 | Depth map contrast multiplier |
| Depth Offset | -0.5–0.5 | 0.0 | Depth map brightness offset |
| Depth Blur | 0–20 | 0 | Gaussian blur to reduce noise/flicker |
| Invert Depth | toggle | off | Swap near and far |
| Colormap | dropdown | grayscale | Depth visualization: grayscale, turbo, viridis, magma |
| Temporal Smoothing | 0–0.99 | 0.5 | Blend with previous frame (higher = smoother but more latent) |

## Calibration file format

The plugin auto-saves calibration to `~/.promapanything_calibration.json`. You can also provide your own file via the **Calibration File** parameter. Format:

```json
{
  "version": 1,
  "projector_width": 1920,
  "projector_height": 1080,
  "map_x": [[...], ...],
  "map_y": [[...], ...]
}
```

`map_x` and `map_y` are 2D arrays of shape `(projector_height, projector_width)` containing the camera pixel coordinates that correspond to each projector pixel.

## Development

Edit code, then click **Reload** next to the plugin in Scope's Settings → Plugins. No reinstall needed.

## Dependencies

- `opencv-python-headless` — calibration pattern detection and image warping
- `transformers` — bundled depth model fallback (only if Scope's Depth Anything is not available)

Scope provides: `torch`, `numpy`, `pydantic`, `pillow`.
