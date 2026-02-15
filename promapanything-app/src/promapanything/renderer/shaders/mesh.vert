#version 330 core

uniform sampler2D depth_tex;
uniform sampler2D color_tex;
uniform mat4 mvp;
uniform float depth_scale;
uniform float depth_offset;
uniform bool depth_invert;
uniform vec2 resolution;       // depth texture resolution
uniform float focal_length;
uniform vec2 principal;        // principal point (cx, cy)

out vec3 v_color;

void main() {
    int px = gl_VertexID % int(resolution.x);
    int py = gl_VertexID / int(resolution.x);
    vec2 uv = vec2(float(px) + 0.5, float(py) + 0.5) / resolution;

    float d = texture(depth_tex, uv).r;
    d = d * depth_scale + depth_offset;
    if (depth_invert) d = 1.0 - d;
    d = clamp(d, 0.0, 1.0);

    float z = d * 5.0 + 0.1;
    float x = (float(px) - principal.x) / focal_length * z;
    float y = (float(py) - principal.y) / focal_length * z;

    gl_Position = mvp * vec4(x, -y, -z, 1.0);

    v_color = texture(color_tex, uv).rgb;
}
