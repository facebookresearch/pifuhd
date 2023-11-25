#version 330 core

out vec4 FragColor;

in vec3 CamNormal;
in vec3 CamPos;

void main() 
{
    vec3 light_direction = vec3(0, 0, 1);
    vec3 f_normal = normalize(CamNormal.xyz);
    vec4 specular_reflection = vec4(0.2) * pow(max(0.0, dot(reflect(-light_direction, f_normal), vec3(0, 0, -1))), 16.f);
    FragColor = vec4(dot(f_normal, light_direction)*vec3(1.0)+specular_reflection.xyz, 1.0);
}
