# ProMapAnything: Vibe Coding /Vibe Engineering a Projection Mapping System with AI in a week

**How I built a structured light calibration system, depth estimation pipeline, and AI-powered VJ tool as a Scope plugin — mostly by describing what I wanted and letting Claude figure out the math.**

---

## The Idea

I wanted to project AI-generated visuals onto physical surfaces — walls, furniture, corners, ceilings — and have them actually *stick* to the geometry. Not just a flat rectangle of video splashed onto a wall, but imagery that understands the 3D structure of the room and wraps around it.

Microsoft Research did this years ago with [RoomAlive](https://www.microsoft.com/en-us/research/project/roomalive/) — a system of Kinect cameras and projectors that could turn any room into an interactive display. Their approach: project Gray code patterns, decode them with a camera, solve for the projector-camera geometry, then render from the projector's exact viewpoint. Brilliant, but it was C#/DirectX, needed a Kinect, and was very much a research prototype.

I wanted to build something similar but with modern tools: monocular depth estimation (Depth Anything V2) instead of a Kinect, AI generation (Krea/VACE) instead of pre-rendered content, and the whole thing running as a [Daydream Scope](https://www.daydream.live/) plugin so I could use it for live VJ sets. And I wanted to build it by vibe coding — describing the system I wanted and collaborating with Claude to implement it.



## Starting Point: The Standalone App

The first version was a standalone desktop app. Dual windows (one for the UI, one fullscreen on the projector), ModernGL for rendering, imgui-bundle for the interface. It talked to the projector directly and used Spout to share textures with other tools like Resolume and Scope using Cloud Infrence. 

This worked, but it was I just had to prove out the concept to make sure it would work. What I really wanted was to plug it into Scope's pipeline architecture: camera in, depth estimation + calibration in the middle, AI generation, projector out. Three distinct stages that each do one thing well.

## The Three-Pipeline Architecture

Scope plugins can register as main pipelines, preprocessors, or postprocessors. I realized the projection mapping system naturally splits into three:

1. **ProMapAnything Calibrate** — A main pipeline that takes over the entire viewer to project Gray code patterns and calibrate the projector-camera geometry.
2. **ProMapAnything Depth** — A preprocessor that estimates depth from the camera, warps it to the projector's perspective using the calibration data, and feeds it to whatever AI model you're running (Krea, VACE, etc.) as a conditioning signal.
3. **ProMapAnything Projector** — A postprocessor that takes the AI-generated output and streams it to the projector via MJPEG.

The workflow: calibrate once, then swap the main pipeline to your AI model of choice. The preprocessor gives it depth-aware conditioning, the postprocessor gets it to the projector. Clean separation of concerns.

```
CALIBRATION:   Main: Calibrate  |  Pre: —         |  Post: —
PERFORMING:    Main: Krea/VACE  |  Pre: Depth     |  Post: Projector
```

## Gray Code Calibration: The Hard Part

Structured light calibration sounds simple on paper. Project binary stripe patterns onto the scene. For each camera pixel, decode which projector pixel illuminated it. Build a mapping. Done.

In practice? A nightmare of timing, synchronization, and edge cases.

### The Feedback Loop

My first attempt: display camera feed on the projector while setting up, then switch to patterns when calibration starts. Seemed reasonable. What actually happened: the camera sees the projector, which shows the camera, which sees the projector... infinite feedback spiral. The image oscillates wildly and the Gray code decode gets garbage data.

The fix is embarrassingly simple: show a neutral grey test card on the projector until calibration actually starts. But it took me a while to figure out *why* calibration was producing nonsense results.

### Timing Hell

The fundamental problem with structured light over a network: you project a pattern, but the camera doesn't see it instantly. There's projector response time, camera exposure time, network latency (if you're running on RunPod), MJPEG encoding/decoding latency. Project a pattern and capture too early? You get the *previous* pattern's image. Your entire calibration is shifted by one frame and everything decodes to garbage.

My first approach: fixed delay (`settle_frames=6`). Works locally where latency is ~200ms. Completely fails on RunPod where the round trip can be 500ms+.

I cranked it to `settle_frames=60` for remote setups, but that made calibration take forever — 22+ patterns × 60 frames × 2 (normal + inverse) = several minutes of staring at stripes.

The solution I landed on: **adaptive settling with structural baseline comparison**. When a new pattern starts displaying, I capture the camera frame as a baseline (it's still showing the *old* pattern). Then I watch for structural change — mean absolute pixel difference across the whole image. Once I detect the change AND see the image stabilize (frame-to-frame diff drops below threshold for 3 consecutive frames), I know the new pattern has fully arrived. Early exit when conditions are met, hard cap as a safety net.

This cut calibration time roughly in half for local setups while still being reliable over high-latency connections.

### The Frame Format Trap

This one burned me multiple times. Scope can feed video frames as either float32 [0,1] or uint8 [0,255]. Seems like a minor detail. Here's the trap:

```python
# This produces ALL ZEROS and you will stare at black frames for hours
img = tensor.numpy().astype(np.uint8)  # when tensor is float [0,1]
```

Casting float 0.7 to uint8 gives you 0, not 178. You need to multiply by 255 first. I wrote this bug at least three separate times before it became a reflex to always check:

```python
if img.max() <= 1.5:
    img = (img * 255.0).clip(0, 255)
img = img.astype(np.uint8)
```

## MJPEG Streaming: Simpler Than You'd Think (Until It Isn't)

I needed to get frames from Scope (running on a GPU server) to a physical projector (connected to a laptop). My options were Spout (Windows-only, same machine), NDI (network, but heavy), or MJPEG (HTTP, works everywhere, dead simple).

MJPEG won. It's just a `multipart/x-mixed-replace` HTTP response where each part is a JPEG. I wrote a `FrameStreamer` class that runs an HTTP server on a background thread. The pipeline calls `submit_frame(rgb_numpy)`, the server encodes it to JPEG and pushes it to any connected clients. Open a browser, point it at the URL, you get a live video feed. Click for fullscreen, drag the browser window to the projector monitor — done.

The wrinkle: JPEG encoding blocks the pipeline thread. At 1080p, that's maybe 10-15ms per frame. Doesn't sound like much, but Scope's pipeline runs at whatever frame rate it can manage, and every millisecond counts for real-time generation.

I tried background encoder threads. Seemed like the obvious fix — submit the frame, let a background thread encode it, pipeline keeps running. What actually happened: the background threads had lifecycle issues with Scope's plugin reload mechanism. The plugin would unload, the threads would keep running with stale references, and everything would crash or deadlock on the next reload.

The solution that actually works: synchronous encoding with a try-lock. If encoding is already in progress when a new frame arrives, just drop it. The pipeline thread never blocks, the projector gets the latest available frame, and there are no background threads to manage.

```python
def submit_frame(self, rgb):
    if not self._encoding.acquire(blocking=False):
        return  # drop frame — encoder is busy
    try:
        jpeg = self._encode_jpeg(rgb)
        with self._lock:
            self._frame_jpeg = jpeg
        self._new_frame.set()
    finally:
        self._encoding.release()
```

Simple, robust, no threads. Sometimes the boring solution is the right one.

## The 99% Problem

Calibration would run perfectly — all patterns captured, progress bar climbing steadily — and then freeze at 99%. The dashboard shows "Decoding..." but nothing happens. The pipeline is alive (Scope isn't crashed), it's just... stuck.

The root cause was a chain of blocking operations all running synchronously inside the pipeline's `__call__` method:

1. **Gray code decoding**: CPU-intensive vectorized numpy operations. Takes 2-5 seconds depending on resolution. During this time, no frames are being processed, so no progress updates reach the dashboard.

2. **Neural depth model loading**: After decode, I was loading Depth Anything V2 to generate depth images from the calibration results. This downloads the model from HuggingFace on first run (~30 seconds) and loads it into GPU memory (~10 seconds). All blocking the pipeline thread.

3. **Scope gallery upload**: HTTP POST requests to upload result images. 5-second timeout per file, 4 files = potential 20 seconds.

The fix was surgical:

- Split the DECODING phase into its own frame. Instead of decoding inline when the last pattern completes, set `phase = DECODING` and return. On the *next* frame, the pipeline pushes a "Decoding..." progress update, *then* runs the decode. The dashboard actually shows what's happening.

- Ripped out the neural depth model entirely from calibration. Replaced it with a fast grayscale luminance conversion (bilateral filter on the warped camera image). Takes milliseconds instead of 30+ seconds. If you want real neural depth, use it via the preprocessor's live depth modes — that's where it belongs anyway.

- Gallery uploads already had a 5-second timeout and exception handling, so those were acceptable.

## Depth Estimation: Six Modes

The preprocessor supports six different ways to generate the depth conditioning signal:

**Live modes** (need GPU depth model):
- `depth_then_warp` — Estimate depth from raw camera, then warp to projector perspective
- `warp_then_depth` — Warp camera to projector perspective first, then estimate depth
- `warped_rgb` — Just warp the camera RGB (no depth estimation)

**Static modes** (from calibration, no GPU needed):
- `static_depth_warped` — Saved depth-then-warp image
- `static_depth_from_warped` — Saved warp-then-depth image
- `static_warped_camera` — Saved warped camera RGB

The static modes are the key insight for RunPod deployment. Running the depth model on every frame eats into your generation budget. But the room geometry doesn't change during a performance. Calibrate once, save the depth map, use it as a static conditioning signal for every frame. Zero GPU overhead for the depth pipeline, all compute goes to the AI generation.

## RunPod: Cloud GPUs for Live Performance

Running Scope on RunPod means access to serious GPUs (A40, A100) for real-time AI generation. But it also means:

- The camera is local, the GPU is remote. Video has to stream over the internet.
- The projector is local. Generated frames have to stream back.
- Port 8765 needs to be explicitly exposed in RunPod settings.
- Killing the Scope process restarts the entire container (learned that the hard way).
- The MJPEG URL changes every time you spin up a new pod.

The plugin auto-detects RunPod via the `RUNPOD_POD_ID` environment variable and constructs the correct proxy URL:

```
https://{pod-id}-8765.proxy.runpod.net/
```

Open that in a browser on your local machine, drag to projector, fullscreen. It just works through RunPod's port proxy with no special configuration.

## What I Learned Vibe Coding This

### Let the AI handle the math

I didn't write the Gray code generation, the vectorized decode, the Gaussian splat fill, or the adaptive settle algorithm by hand. I described what I wanted ("generate Gray code patterns for a 1920x1080 projector", "decode them with per-bit reliability thresholding", "fill gaps using a weighted Gaussian blur") and Claude implemented it. The structured light math is well-documented in papers — Claude knows it cold.

What I *did* do: decide the architecture (three pipelines), choose the constraints (MJPEG over Spout for network portability), debug the integration issues (frame format, timing, feedback loops), and test on real hardware.

### Debug logging saves your life on remote systems

When your code is running on a RunPod GPU server and you can't attach a debugger, `logger.info()` is your only friend. I went through a phase of adding `/tmp/promap_debug.log` file writes everywhere, which I eventually cleaned up into proper Python logging. But having those logs is what let me diagnose the 99% freeze, the black frame bug, the frame format issues.

### The boring solution usually wins

Background encoder threads? Crashed on plugin reload. Neural depth in calibration? Froze at 99%. Complex adaptive frame dropping? Race conditions. The solutions that survived: synchronous encoding with a try-lock, grayscale luminance instead of neural depth, frame counting with a hard cap.

### Scope's plugin API is simple but has gotchas

- `__init__` params need `is_load_param=True` and require a pipeline reload.
- Runtime params come through `kwargs` in `__call__`, not `self`.
- Preprocessors only show `category="configuration"` fields, never `category="input"`.
- The video tensor format is ambiguous (float vs uint8). Always check.
- Lazy imports in `register_pipelines()` — don't import pipeline classes at module level.

### Calibration is a solved problem (the integration isn't)

The math of structured light calibration has been known for decades. Gray codes, correspondence maps, camera-projector geometry — it's all well-documented. What's *hard* is making it work reliably in a plugin that runs on cloud GPUs, streams video over the internet, and needs to handle arbitrary timing and frame formats. The novel part isn't the algorithm — it's the system around it.

## What's Next

- **Real-time effects**: The standalone app has GPU shaders for wobble/shockwave effects (displacement mapping using the depth map). These need to be ported to the Scope plugin as a postprocessor effect chain.
- **Multi-projector support**: The architecture supports it in theory (multiple calibrations, separate MJPEG streams), but it hasn't been tested.
- **Better depth conditioning**: The grayscale luminance fallback works but isn't true depth. Exploring running the depth model once after calibration and caching the result, rather than running it live every frame.
- **Latency optimization**: The end-to-end camera→AI→projector latency is still noticeable. Looking at WebRTC as an alternative to MJPEG for lower latency.

## Try It

The plugin is open source:

```bash
pip install "git+https://github.com/shorties/ProjectionMapAnything.git#subdirectory=scope-promapanything-vj"
```

You need a camera, a projector, and a Scope instance (local or RunPod). Calibrate once, then use any AI model as your main pipeline with ProMapAnything Depth as the preprocessor and ProMapAnything Projector as the postprocessor.

Point a projector at your wall. Let the AI paint on it.

---

*Built with [Daydream Scope](https://www.daydream.live/) and [Claude Code](https://claude.ai/claude-code). 38 commits, 8 files, and several hundred conversations with an AI that knows more about projective geometry than I do.*
