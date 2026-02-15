#version 330 core

in vec3 v_color;
out vec4 frag_color;

void main() {
    // Circular point shape
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    if (dot(pc, pc) > 1.0) discard;
    frag_color = vec4(v_color, 1.0);
}
