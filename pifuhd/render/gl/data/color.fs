#version 330 core

out vec4 FragColor;

in vec3 Color;

void main() 
{
    FragColor = vec4(Color,1.0);
}
