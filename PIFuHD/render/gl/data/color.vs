#version 330 core

layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Color;

out vec3 CamNormal;
out vec3 CamPos;
out vec3 Color;

uniform mat4 ModelMat;
uniform mat4 PerspMat;

void main()
{
	gl_Position = PerspMat * ModelMat * vec4(a_Position, 1.0);
    Color = a_Color;
}