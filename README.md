# ProjectionMapAnything

**Turn any surface into a living AI canvas.**

ProjectionMapAnything is a [Daydream Scope](https://scope.daydream.live) plugin that calibrates a camera-projector pair, maps room geometry, and feeds depth data to AI models so their output wraps onto your physical space in real time. Point a projector at a wall, a stage set, a sculpture — the AI sees the shape and generates visuals that follow every contour.

Inspired by Microsoft's [RoomAlive Toolkit](https://www.microsoft.com/en-us/research/project/roomalive-toolkit/) and [IllumiRoom](https://www.microsoft.com/en-us/research/project/illumiroom/).

---

## How it works

```
Camera ──> Calibration (Gray code) ──> Depth Map ──> AI Model (VACE) ──> Projector
              one-time setup          every frame     Krea / Longlive     onto walls
```

1. **Calibrate** — Gray code structured light patterns establish a pixel-perfect mapping between your camera and projector
2. **Depth** — Compute scene depth from calibration disparity or Depth Anything V2
3. **Condition** — Feed the depth map to a VACE-enabled AI model so it generates geometry-aware visuals
4. **Project** — Stream the AI output to your projector via the built-in dashboard

Everything runs from a single Scope preprocessor — calibration, depth, effects, isolation, and projector streaming.

---

## Quick Start

### Install

In Scope, go to **Settings > Plugins** and install from Git URL:

```
git+https://github.com/shorties/ProjectionMapAnything.git#subdirectory=projectionmapanything
```

### Configure

You need a main AI pipeline with VACE (in-context depth adapter) enabled. Pipelines that work:

- **Krea Realtime** with VACE enabled
- **Longlive** with VACE enabled

1. Select your AI pipeline as main, enable VACE
2. Select **Projection-Map-Anything (VJ.Tools)** as preprocessor
3. Open the dashboard — change port `8000` to `8765` in your Scope URL

### Calibrate

Calibration runs inline while the AI keeps generating — no pipeline switching needed.

1. On the dashboard, click **Projector** and drag the window to your projector
2. Click to go fullscreen
3. Toggle **Start Calibration** in Scope settings
4. Gray code patterns fire automatically — takes about 30 seconds
5. Done. Depth conditioning kicks in immediately.

### Perform

Once calibrated, the AI sees your room's geometry and generates accordingly. Tweak everything live from the dashboard: depth mode, VJ effects, subject isolation, edge blending.

---

## Depth Modes

All modes use static calibration data — never the live camera feed (avoids projector-camera feedback loops).

| Mode | Description |
|------|-------------|
| **AI Depth** | Depth Anything V2 on the raw camera image, warped to projector space. Best quality. |
| **Calibration Disparity** | Horizontal disparity computed during calibration. Fast, no AI model needed. |
| **Edge Detection** | Canny edges on the warped camera image. Great for structural VACE guidance. |
| **Warped Camera** | The camera image warped to projector perspective. |
| **Custom Upload** | Your own depth map, uploaded via the dashboard. |

## VJ Effects

Animated effects that modulate the depth conditioning signal. All effects are **surface-masked** — they only animate on projection surfaces, void stays black.

| | | |
|---|---|---|
| **Noise Blend** — fractal noise | **Flow Warp** — organic morphing | **Pulse** — rhythmic breathing |
| **Wave Warp** — sinusoidal ripples | **Kaleido** — mandala symmetry | **Shockwave** — radial burst |
| **Wobble** — jelly room distortion | **Geometry Edges** — glowing outlines | **Depth Fog** — rolling fog layers |
| **Radial Zoom** — tunnel burst | | |

## Subject Isolation

| Mode | Description |
|------|-------------|
| **Depth Band** | Auto-detect nearest surfaces via histogram peak |
| **Custom Mask** | Upload a mask image via dashboard |
| **rembg** | AI background removal (optional `rembg` dependency) |

---

## Dashboard

All calibration and depth controls are available directly in Scope's settings panel. The dashboard is only needed for the **projector output window** — it hosts the MJPEG stream you drag to your projector and fullscreen.

It also provides some extras: live preview, calibration result downloads, and custom depth/mask uploads.

Access: change `8000` to `8765` in your Scope URL. On RunPod, replace `-8000.` with `-8765.` in the proxy URL.

---

## RunPod

Works on RunPod — expose port **8765** in your pod settings.

```
git+https://github.com/shorties/ProjectionMapAnything.git#subdirectory=projectionmapanything
```

Dashboard URL: `https://{pod-id}-8765.proxy.runpod.net/`

---

## Requirements

- [Daydream Scope](https://scope.daydream.live)
- Python 3.12+
- A camera and a projector

## License

MIT
