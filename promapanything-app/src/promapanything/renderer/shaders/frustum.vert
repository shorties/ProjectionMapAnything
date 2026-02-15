#version 330 core

uniform mat4 mvp;
in vec3 in_position;
out vec3 v_color;

uniform vec3 line_color;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = line_color;
}
