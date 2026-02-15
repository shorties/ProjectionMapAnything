#version 330 core

uniform sampler2D depth_tex;
uniform float u_time;
uniform vec2 u_resolution;

// Effect toggles and parameters (uniforms set from Python)
// -- Noise
uniform bool u_noise_enabled;
uniform float u_noise_intensity;
uniform float u_noise_scale;
uniform int u_noise_octaves;
uniform float u_noise_speed;

// -- Flow warp
uniform bool u_flow_enabled;
uniform float u_flow_intensity;
uniform float u_flow_scale;
uniform float u_flow_speed;

// -- Pulse
uniform bool u_pulse_enabled;
uniform float u_pulse_speed;
uniform float u_pulse_amount;

// -- Wave
uniform bool u_wave_enabled;
uniform float u_wave_frequency;
uniform float u_wave_amplitude;
uniform float u_wave_speed;
uniform float u_wave_direction;

// -- Shockwave
uniform bool u_shockwave_enabled;
uniform vec2 u_shockwave_origin;
uniform float u_shockwave_speed;
uniform float u_shockwave_thickness;
uniform float u_shockwave_strength;
uniform float u_shockwave_decay;
uniform float u_shockwave_interval;

// -- Wobble
uniform bool u_wobble_enabled;
uniform float u_wobble_intensity;
uniform float u_wobble_speed;

// -- Edges
uniform bool u_edges_enabled;
uniform float u_edges_strength;
uniform float u_edges_glow_width;
uniform float u_edges_pulse_speed;

// -- Fog
uniform bool u_fog_enabled;
uniform float u_fog_density;
uniform float u_fog_near;
uniform float u_fog_far;
uniform bool u_fog_animated;
uniform float u_fog_speed;

// -- Zoom
uniform bool u_zoom_enabled;
uniform vec2 u_zoom_origin;
uniform float u_zoom_strength;
uniform float u_zoom_speed;

// -- Kaleidoscope
uniform bool u_kaleido_enabled;
uniform int u_kaleido_segments;
uniform float u_kaleido_rotation;
uniform float u_kaleido_spin_speed;

in vec2 v_uv;
out vec4 frag_color;

// ── Noise functions ─────────────────────────────────────────────────────────

float hash3d(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 269.5, 419.2))) * 43758.5453123);
}

float value_noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);  // smoothstep

    float n000 = hash3d(i);
    float n100 = hash3d(i + vec3(1, 0, 0));
    float n010 = hash3d(i + vec3(0, 1, 0));
    float n110 = hash3d(i + vec3(1, 1, 0));
    float n001 = hash3d(i + vec3(0, 0, 1));
    float n101 = hash3d(i + vec3(1, 0, 1));
    float n011 = hash3d(i + vec3(0, 1, 1));
    float n111 = hash3d(i + vec3(1, 1, 1));

    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);

    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);

    return mix(nxy0, nxy1, f.z) - 0.5;
}

float fbm(vec3 p, int octaves) {
    float total = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float max_amp = 0.0;
    for (int i = 0; i < octaves; i++) {
        total += amplitude * value_noise(p * frequency);
        max_amp += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return total / max(max_amp, 0.001);
}

// ── Main ────────────────────────────────────────────────────────────────────

void main() {
    vec2 uv = v_uv;
    float t = u_time;

    // Start with a sampling offset for spatial warps
    vec2 sample_uv = uv;

    // -- Flow warp (spatial displacement)
    if (u_flow_enabled) {
        float dx = fbm(vec3(uv * u_flow_scale, t * u_flow_speed), 3);
        float dy = fbm(vec3(uv * u_flow_scale + 100.0, t * u_flow_speed), 3);
        sample_uv += vec2(dx, dy) * u_flow_intensity;
    }

    // -- Wave warp
    if (u_wave_enabled) {
        float rad = radians(u_wave_direction);
        vec2 dir = vec2(cos(rad), sin(rad));
        float projected = dot(sample_uv * 2.0 - 1.0, dir);
        float phase = projected * u_wave_frequency * 6.2832 + t * u_wave_speed * 6.2832;
        vec2 perp = vec2(-sin(rad), cos(rad));
        sample_uv += perp * sin(phase) * u_wave_amplitude;
    }

    // -- Wobble
    if (u_wobble_enabled) {
        float wt = t * u_wobble_speed;
        vec2 sc = sample_uv * 2.0 - 1.0;
        float wx = sin(sc.y * 3.0 + wt * 2.1) * 0.4
                  + sin(sc.y * 7.0 - wt * 1.7) * 0.2
                  + sin(sc.x * 5.0 + wt * 3.3) * 0.15;
        float wy = sin(sc.x * 4.0 + wt * 1.9) * 0.4
                  + sin(sc.x * 6.0 - wt * 2.3) * 0.2
                  + sin(sc.y * 8.0 + wt * 2.7) * 0.15;
        sample_uv += vec2(wx, wy) * u_wobble_intensity;
    }

    // -- Shockwave (spatial)
    if (u_shockwave_enabled && u_shockwave_strength > 0.0) {
        vec2 sc = sample_uv * 2.0 - 1.0;
        vec2 origin = u_shockwave_origin * 2.0 - 1.0;
        vec2 delta = sc - origin;
        float dist = length(delta) + 0.0001;

        float cycle_time = u_shockwave_interval > 0.0
            ? mod(t, u_shockwave_interval) : t;
        float wave_front = cycle_time * u_shockwave_speed;

        float ring = exp(-pow(dist - wave_front, 2.0) /
                        (u_shockwave_thickness * u_shockwave_thickness + 0.000001));
        ring *= exp(-dist * u_shockwave_decay);

        vec2 radial = delta / dist;
        sample_uv += radial * ring * u_shockwave_strength * 0.3;
    }

    // -- Kaleidoscope
    if (u_kaleido_enabled && u_kaleido_segments >= 2) {
        vec2 sc = sample_uv * 2.0 - 1.0;
        float angle = atan(sc.y, sc.x) + u_kaleido_rotation + t * u_kaleido_spin_speed;
        float radius = length(sc);
        float seg = 6.2832 / float(u_kaleido_segments);
        angle = mod(angle, seg);
        if (mod(angle / seg * 2.0, 2.0) > 1.0) angle = seg - angle;
        sample_uv = vec2(radius * cos(angle), radius * sin(angle)) * 0.5 + 0.5;
    }

    // -- Zoom (spatial)
    if (u_zoom_enabled && u_zoom_strength > 0.0) {
        vec2 sc = sample_uv * 2.0 - 1.0;
        vec2 center = u_zoom_origin * 2.0 - 1.0;
        vec2 delta = sc - center;
        float zoom = sin(t * u_zoom_speed * 6.2832) * u_zoom_strength;
        sample_uv = (center + delta * (1.0 + zoom)) * 0.5 + 0.5;
    }

    // Clamp UVs
    sample_uv = clamp(sample_uv, 0.0, 1.0);

    // Sample depth
    float d = texture(depth_tex, sample_uv).r;

    // -- Noise blend
    if (u_noise_enabled) {
        float n = fbm(vec3(uv * u_noise_scale, t * u_noise_speed), u_noise_octaves);
        d += n * u_noise_intensity;
    }

    // -- Pulse
    if (u_pulse_enabled) {
        float wave = sin(t * u_pulse_speed * 6.2832) * 0.5 + 0.5;
        float mod_val = 1.0 + (wave - 0.5) * u_pulse_amount * 2.0;
        d *= mod_val;
    }

    // -- Shockwave (depth displacement)
    if (u_shockwave_enabled && u_shockwave_strength > 0.0) {
        vec2 sc = uv * 2.0 - 1.0;
        vec2 origin = u_shockwave_origin * 2.0 - 1.0;
        float dist = length(sc - origin) + 0.0001;
        float cycle_time = u_shockwave_interval > 0.0
            ? mod(t, u_shockwave_interval) : t;
        float wave_front = cycle_time * u_shockwave_speed;
        float ring = exp(-pow(dist - wave_front, 2.0) /
                        (u_shockwave_thickness * u_shockwave_thickness + 0.000001));
        ring *= exp(-dist * u_shockwave_decay);
        d += ring * u_shockwave_strength * (1.0 - d * 0.5);
    }

    // -- Edges
    if (u_edges_enabled) {
        vec2 texel = 1.0 / u_resolution;
        float tl = texture(depth_tex, sample_uv + vec2(-texel.x, -texel.y)).r;
        float tr = texture(depth_tex, sample_uv + vec2( texel.x, -texel.y)).r;
        float bl = texture(depth_tex, sample_uv + vec2(-texel.x,  texel.y)).r;
        float br = texture(depth_tex, sample_uv + vec2( texel.x,  texel.y)).r;
        float gx = (tr + br) - (tl + bl);
        float gy = (bl + br) - (tl + tr);
        float edge = sqrt(gx * gx + gy * gy);
        float pulse = u_edges_pulse_speed > 0.0
            ? sin(t * u_edges_pulse_speed * 6.2832) * 0.3 + 0.7 : 1.0;
        d += edge * u_edges_strength * pulse;
    }

    // -- Fog
    if (u_fog_enabled) {
        float fog_range = max(u_fog_far - u_fog_near, 0.01);
        float fog_factor = clamp((d - u_fog_near) / fog_range, 0.0, 1.0);
        if (u_fog_animated && u_fog_speed > 0.0) {
            fog_factor = clamp(fog_factor + sin(t * u_fog_speed) * 0.15, 0.0, 1.0);
        }
        d = mix(d, 0.5, fog_factor * u_fog_density);
    }

    d = clamp(d, 0.0, 1.0);
    frag_color = vec4(d, d, d, 1.0);
}
