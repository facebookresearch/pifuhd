#version 330

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 Normal;

out vec3 CamNormal;

uniform mat4 ModelMat;
uniform mat4 PerspMat;

void main()
{
	gl_Position = PerspMat * ModelMat * vec4(Position, 1.0);
	CamNormal = (ModelMat * vec4(Normal, 0.0)).xyz;
}
