"""Animated depth map effects — fractal noise, flow warp, pulse, waves,
and IllumiRoom-style geometry-aware projection effects.

All functions operate on (H, W) float32 depth tensors in [0, 1] and are
GPU-friendly (pure torch, no Python loops over pixels).
"""

from __future__ import annotations

import math

import torch


# ── GPU-friendly gradient noise ──────────────────────────────────────────────
#
# Hash-based 2D/3D gradient noise implemented entirely in torch so it runs as
# fused GPU kernels.  Not as high-quality as OpenSimplex but fast and smooth
# enough for real-time VJ depth effects.


def _hash(p: torch.Tensor) -> torch.Tensor:
    """Pseudo-random hash for integer coordinates.  Returns [0, 1)."""
    # Based on the classic sin-hash from GLSL shader land
    return torch.frac(torch.sin(p * 127.1 + 311.7) * 43758.5453123)


def _hash2d(ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
    return torch.frac(torch.sin(ix * 127.1 + iy * 269.5) * 43758.5453123)


def _hash3d(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
    return torch.frac(
        torch.sin(ix * 127.1 + iy * 269.5 + iz * 419.2) * 43758.5453123
    )


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    return t * t * (3.0 - 2.0 * t)


def value_noise_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    z: float,
) -> torch.Tensor:
    """3D value noise.  x, y are (H, W) coordinate grids, z is the time axis.

    Returns (H, W) float32 in approximately [-0.5, 0.5].
    """
    ix = x.floor()
    iy = y.floor()
    iz = torch.tensor(z, device=x.device).floor()

    fx = _smoothstep(x - ix)
    fy = _smoothstep(y - iy)
    fz_val = z - iz.item()
    fz = fz_val * fz_val * (3.0 - 2.0 * fz_val)

    iz_f = iz.float()

    # 8 corner hashes
    n000 = _hash3d(ix, iy, iz_f)
    n100 = _hash3d(ix + 1, iy, iz_f)
    n010 = _hash3d(ix, iy + 1, iz_f)
    n110 = _hash3d(ix + 1, iy + 1, iz_f)
    n001 = _hash3d(ix, iy, iz_f + 1)
    n101 = _hash3d(ix + 1, iy, iz_f + 1)
    n011 = _hash3d(ix, iy + 1, iz_f + 1)
    n111 = _hash3d(ix + 1, iy + 1, iz_f + 1)

    # Trilinear interpolation
    nx00 = torch.lerp(n000, n100, fx)
    nx10 = torch.lerp(n010, n110, fx)
    nx01 = torch.lerp(n001, n101, fx)
    nx11 = torch.lerp(n011, n111, fx)

    nxy0 = torch.lerp(nx00, nx10, fy)
    nxy1 = torch.lerp(nx01, nx11, fy)

    return torch.lerp(nxy0, nxy1, fz) - 0.5  # Centre around 0


def fractal_noise_3d(
    h: int,
    w: int,
    *,
    scale: float = 4.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    time: float = 0.0,
    device: torch.device,
) -> torch.Tensor:
    """Generate animated fractal (fBm) noise.

    Returns (H, W) float32 approximately in [-0.5, 0.5].
    """
    gy = torch.linspace(0, scale, h, device=device)
    gx = torch.linspace(0, scale, w, device=device)
    y, x = torch.meshgrid(gy, gx, indexing="ij")

    total = torch.zeros(h, w, device=device)
    amplitude = 1.0
    frequency = 1.0
    max_amp = 0.0

    for _ in range(octaves):
        total += amplitude * value_noise_3d(
            x * frequency, y * frequency, time * frequency
        )
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    if max_amp > 0:
        total /= max_amp

    return total


# ── Effect functions ─────────────────────────────────────────────────────────


def apply_noise_blend(
    depth: torch.Tensor,
    *,
    intensity: float,
    scale: float,
    octaves: int,
    speed: float,
    time: float,
) -> torch.Tensor:
    """Blend animated fractal noise into the depth map.

    Adds organic, evolving texture to the depth values so the AI model
    receives constantly changing conditioning.
    """
    if intensity <= 0:
        return depth

    h, w = depth.shape
    noise = fractal_noise_3d(
        h, w,
        scale=scale,
        octaves=octaves,
        time=time * speed,
        device=depth.device,
    )

    return depth + noise * intensity


def apply_flow_warp(
    depth: torch.Tensor,
    *,
    intensity: float,
    scale: float,
    speed: float,
    time: float,
) -> torch.Tensor:
    """Warp the depth map using an animated noise-based displacement field.

    Pixels flow and shift according to fractal noise, creating organic
    morphing distortion in the depth map.
    """
    if intensity <= 0:
        return depth

    h, w = depth.shape

    # Generate two noise fields for x and y displacement
    dx = fractal_noise_3d(
        h, w,
        scale=scale,
        octaves=3,
        time=time * speed,
        device=depth.device,
    )
    dy = fractal_noise_3d(
        h, w,
        scale=scale,
        octaves=3,
        time=time * speed + 100.0,  # Offset so dx and dy are uncorrelated
        device=depth.device,
    )

    # Build sampling grid
    gy = torch.linspace(-1, 1, h, device=depth.device)
    gx = torch.linspace(-1, 1, w, device=depth.device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

    # Apply displacement (intensity controls max pixel shift as fraction of image)
    grid_x = grid_x + dx * intensity
    grid_y = grid_y + dy * intensity

    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    depth_nchw = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    warped = torch.nn.functional.grid_sample(
        depth_nchw, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped.squeeze()  # (H, W)


def apply_pulse(
    depth: torch.Tensor,
    *,
    speed: float,
    amount: float,
    time: float,
) -> torch.Tensor:
    """Rhythmic global depth oscillation — makes the depth "breathe".

    Modulates the overall depth values with a smooth sine wave over time.
    """
    if amount <= 0:
        return depth

    wave = math.sin(time * speed * 2.0 * math.pi) * 0.5 + 0.5  # [0, 1]
    modulation = 1.0 + (wave - 0.5) * amount * 2.0  # [1-amount, 1+amount]

    return depth * modulation


def apply_wave_warp(
    depth: torch.Tensor,
    *,
    frequency: float,
    amplitude: float,
    speed: float,
    direction: float,
    time: float,
) -> torch.Tensor:
    """Sinusoidal spatial displacement — ripples across the depth map.

    Creates wave-like distortion that travels across the image, like a
    flag waving or water ripple effect.
    """
    if amplitude <= 0:
        return depth

    h, w = depth.shape
    rad = math.radians(direction)
    cos_d = math.cos(rad)
    sin_d = math.sin(rad)

    gy = torch.linspace(-1, 1, h, device=depth.device)
    gx = torch.linspace(-1, 1, w, device=depth.device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

    # Project coordinates along wave direction
    projected = grid_x * cos_d + grid_y * sin_d

    # Compute displacement perpendicular to wave direction
    phase = projected * frequency * math.pi * 2.0 + time * speed * math.pi * 2.0
    displacement = torch.sin(phase) * amplitude

    # Displace perpendicular to the wave direction
    perp_x = -sin_d
    perp_y = cos_d
    sample_x = grid_x + displacement * perp_x
    sample_y = grid_y + displacement * perp_y

    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
    depth_nchw = depth.unsqueeze(0).unsqueeze(0)

    warped = torch.nn.functional.grid_sample(
        depth_nchw, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped.squeeze()


def apply_kaleido(
    depth: torch.Tensor,
    *,
    segments: int,
    rotation: float,
    time: float,
) -> torch.Tensor:
    """Kaleidoscope symmetry — mirrors the depth map into radial segments.

    Creates mandala-like patterns from the depth data, which produces
    wild symmetric AI generations.
    """
    if segments < 2:
        return depth

    h, w = depth.shape

    # Coordinate system centred at image middle
    gy = torch.linspace(-1, 1, h, device=depth.device)
    gx = torch.linspace(-1, 1, w, device=depth.device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

    # Convert to polar
    angle = torch.atan2(grid_y, grid_x) + rotation + time
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)

    # Fold angle into segment
    segment_angle = 2.0 * math.pi / segments
    angle = angle % segment_angle
    # Mirror alternate segments
    angle = torch.where(
        (angle / segment_angle * 2).long() % 2 == 1,
        segment_angle - angle,
        angle,
    )

    # Convert back to cartesian
    sample_x = radius * torch.cos(angle)
    sample_y = radius * torch.sin(angle)

    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
    depth_nchw = depth.unsqueeze(0).unsqueeze(0)

    warped = torch.nn.functional.grid_sample(
        depth_nchw, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped.squeeze()


# ── IllumiRoom-style geometry-aware effects ──────────────────────────────────
#
# These effects use the depth map not just as data to modify, but as actual
# 3D geometry information.  Distortions respect the room's surface so they
# look physically plausible when projected — the key insight from IllumiRoom.


def apply_shockwave(
    depth: torch.Tensor,
    *,
    origin_x: float,
    origin_y: float,
    speed: float,
    thickness: float,
    strength: float,
    decay: float,
    time: float,
    auto_trigger_interval: float,
) -> torch.Tensor:
    """Radial shockwave that expands from a point — IllumiRoom's signature effect.

    A ring of displacement radiates outward from the origin.  Because it
    displaces depth values, the AI generates content that appears to bulge
    and ripple through the physical room geometry.

    The shockwave is depth-aware: surfaces closer to the camera (lower depth)
    react more strongly, creating a parallax effect that sells the illusion
    of a physical shockwave passing through 3D space.
    """
    if strength <= 0:
        return depth

    h, w = depth.shape

    # Coordinate system [-1, 1]
    gy = torch.linspace(-1, 1, h, device=depth.device)
    gx = torch.linspace(-1, 1, w, device=depth.device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

    # Distance from origin
    dx = grid_x - (origin_x * 2.0 - 1.0)
    dy = grid_y - (origin_y * 2.0 - 1.0)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)

    # Repeating shockwave: wave front position cycles based on interval
    if auto_trigger_interval > 0:
        cycle_time = time % auto_trigger_interval
    else:
        cycle_time = time
    wave_front = cycle_time * speed

    # Ring shape: Gaussian around the wave front
    ring = torch.exp(-((dist - wave_front) ** 2) / (thickness * thickness + 1e-6))

    # Decay with distance from origin
    ring = ring * torch.exp(-dist * decay)

    # Depth-aware: closer surfaces (lower depth) react more
    depth_influence = 1.0 - depth * 0.5

    # Radial displacement of depth values
    displacement = ring * strength * depth_influence
    result = depth + displacement

    # Also apply spatial displacement (pixels shift radially)
    radial_x = dx / dist
    radial_y = dy / dist

    spatial_strength = strength * 0.3
    sample_x = grid_x + radial_x * ring * spatial_strength
    sample_y = grid_y + radial_y * ring * spatial_strength

    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
    depth_nchw = result.unsqueeze(0).unsqueeze(0)

    warped = torch.nn.functional.grid_sample(
        depth_nchw, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped.squeeze()


def apply_wobble(
    depth: torch.Tensor,
    *,
    intensity: float,
    speed: float,
    time: float,
) -> torch.Tensor:
    """Room wobble — the entire projected geometry shakes and distorts.

    Simulates the room's surfaces warping as if made of jelly.  Uses
    multiple sine waves at different frequencies and phases to create
    a complex, organic wobbling motion.  Depth-aware so nearby surfaces
    wobble more than far ones.
    """
    if intensity <= 0:
        return depth

    h, w = depth.shape
    t = time * speed

    gy = torch.linspace(-1, 1, h, device=depth.device)
    gx = torch.linspace(-1, 1, w, device=depth.device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

    # Multiple wobble frequencies for organic feel
    wobble_x = (
        torch.sin(grid_y * 3.0 + t * 2.1) * 0.4
        + torch.sin(grid_y * 7.0 - t * 1.7) * 0.2
        + torch.sin(grid_x * 5.0 + t * 3.3) * 0.15
    )
    wobble_y = (
        torch.sin(grid_x * 4.0 + t * 1.9) * 0.4
        + torch.sin(grid_x * 6.0 - t * 2.3) * 0.2
        + torch.sin(grid_y * 8.0 + t * 2.7) * 0.15
    )

    # Depth-aware: closer surfaces wobble more
    depth_factor = (1.0 - depth * 0.6)

    sample_x = grid_x + wobble_x * intensity * depth_factor
    sample_y = grid_y + wobble_y * intensity * depth_factor

    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
    depth_nchw = depth.unsqueeze(0).unsqueeze(0)

    warped = torch.nn.functional.grid_sample(
        depth_nchw, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped.squeeze()


def apply_geometry_edges(
    depth: torch.Tensor,
    *,
    edge_strength: float,
    glow_width: float,
    pulse_speed: float,
    time: float,
) -> torch.Tensor:
    """Highlight room geometry edges with glowing outlines.

    Detects depth discontinuities (corners, edges of furniture, walls) and
    adds bright outlines.  With ControlNet depth conditioning, this makes
    the AI generate content that traces the room's physical edges — like
    Tron lines mapped onto real architecture.
    """
    if edge_strength <= 0:
        return depth

    # Sobel-style edge detection on the depth map (pure torch)
    d = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Horizontal and vertical gradients via conv2d
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32, device=depth.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32, device=depth.device,
    ).view(1, 1, 3, 3)

    gx = torch.nn.functional.conv2d(d, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(d, sobel_y, padding=1)
    edges = torch.sqrt(gx * gx + gy * gy).squeeze()  # (H, W)

    # Normalize edges
    e_max = edges.max()
    if e_max > 1e-6:
        edges = edges / e_max

    # Expand edges with Gaussian-like blur (approximate with average pool)
    if glow_width > 0:
        k = int(glow_width) * 2 + 1
        edges_4d = edges.unsqueeze(0).unsqueeze(0)
        edges_4d = torch.nn.functional.avg_pool2d(
            edges_4d, kernel_size=k, stride=1, padding=k // 2
        )
        edges = edges_4d.squeeze()
        # Re-normalize
        e_max = edges.max()
        if e_max > 1e-6:
            edges = edges / e_max

    # Animated pulse on the edge brightness
    if pulse_speed > 0:
        pulse = math.sin(time * pulse_speed * 2.0 * math.pi) * 0.3 + 0.7
    else:
        pulse = 1.0

    # Blend edges into depth
    return depth + edges * edge_strength * pulse


def apply_depth_fog(
    depth: torch.Tensor,
    *,
    density: float,
    near: float,
    far: float,
    animated: bool,
    speed: float,
    time: float,
) -> torch.Tensor:
    """Animated depth fog — surfaces fade into rolling mist.

    Attenuates depth values based on distance, simulating fog or mist
    rolling through the room.  When projected, surfaces at different
    depths fade in and out, creating an atmospheric volumetric effect.
    """
    if density <= 0:
        return depth

    # Fog ramp: 0 at near, 1 at far
    fog_range = max(far - near, 0.01)
    fog_factor = ((depth - near) / fog_range).clamp(0, 1)

    # Animated fog boundary
    if animated and speed > 0:
        wave = math.sin(time * speed) * 0.15
        fog_factor = (fog_factor + wave).clamp(0, 1)

    # Apply fog: blend toward uniform grey (0.5) based on fog factor
    fog_value = 0.5
    fogged = torch.lerp(depth, torch.full_like(depth, fog_value), fog_factor * density)

    return fogged


def apply_radial_zoom(
    depth: torch.Tensor,
    *,
    origin_x: float,
    origin_y: float,
    strength: float,
    speed: float,
    time: float,
) -> torch.Tensor:
    """Radial zoom / tunnel effect from a focal point.

    Pixels appear to fly outward from (or inward toward) the origin,
    creating a zoom-burst or vortex look.  Depth-aware so nearer surfaces
    zoom faster, adding parallax.
    """
    if strength <= 0:
        return depth

    h, w = depth.shape

    gy = torch.linspace(-1, 1, h, device=depth.device)
    gx = torch.linspace(-1, 1, w, device=depth.device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

    cx = origin_x * 2.0 - 1.0
    cy = origin_y * 2.0 - 1.0
    dx = grid_x - cx
    dy = grid_y - cy

    # Oscillating zoom
    zoom = math.sin(time * speed * 2.0 * math.pi) * strength

    # Depth-aware: nearer surfaces zoom faster
    depth_factor = 1.0 - depth * 0.5

    # Radial displacement
    scale = 1.0 + zoom * depth_factor
    sample_x = cx + dx * scale
    sample_y = cy + dy * scale

    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
    depth_nchw = depth.unsqueeze(0).unsqueeze(0)

    warped = torch.nn.functional.grid_sample(
        depth_nchw, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped.squeeze()
