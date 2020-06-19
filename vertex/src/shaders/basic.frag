// shader.frag
#version 450
precision highp float;

// Changed
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_tex_coords;
layout(location = 2) in vec4 v_color;
layout(location = 3) in vec4 v_pos;
layout(location = 4) in vec4 v_light_pos;

layout(location=0) out vec4 f_color;

// New
//layout(set = 0, binding = 0) uniform texture2D t_diffuse;
//layout(set = 0, binding = 1) uniform sampler s_diffuse;

void main() {
    //vec2 st = vec2(tex_coords.x, 1. - tex_coords.y);
    //f_color = texture(sampler2D(t_diffuse, s_diffuse), st);
    vec3 norm = normalize(v_normal);
    vec3 lightDir = normalize(v_light_pos.xyz - v_pos.xyz);
    //vec3 lightDir = normalize(- v_pos.xyz);
    float diffuse = 0.3*max(dot(norm, lightDir), 0.0);

    // specular
    float specularStrength = 0.0+max(0.0,1.2-v_color.w)*5.0;
    vec3 viewDir = normalize(-v_pos.xyz); // the viewer is always at (0,0,0) in view-space, so viewDir is (0,0,0) - Position => -Position
    vec3 col = v_color.xyz;
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 6);
    vec3 specular = specularStrength * spec * vec3(1.0,1.0,1.0);

    float ambient = 0.5;
    //vec3 result = int(((ambient+diffuse+specular)*4.0))*0.25 * col;
    vec3 result = (ambient+diffuse+specular)* col;
    //result = vec3(1.0,0.0,v_pos.z*-0.01);
    //if(abs(dot(norm,viewDir)) < 0.1) {
        //result = vec3(1.0);
    //}
    //f_color =vec4(lightDir,1.0);
    f_color = vec4(result,1.0);
    //f_color = v_color;
}
