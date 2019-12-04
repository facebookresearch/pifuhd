#version 330

layout (location = 0) in vec2 UV;
layout (location = 1) in vec3 Color;

out vec3 CamColor;


void main()
{
	gl_Position = vec4(UV * 2 - 1, 0.0, 1.0);
	CamColor = Color;
}
