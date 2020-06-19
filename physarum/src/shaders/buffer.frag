#version 450
precision highp float;

// Changed
layout(location=0) in vec2 tex_coords;
layout(location=0) out vec4 f_color;

layout(set=0, binding=0) 
buffer positions{
    float pos[];
};

void main() {
    int x = int(tex_coords.x*1024.0);
    int y = int(tex_coords.y*1024.0);
    int index = x*1024+y;

    f_color = vec4(pos[index])*10.0;
}
