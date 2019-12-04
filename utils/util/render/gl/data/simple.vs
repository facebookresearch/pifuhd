#version 330

layout (location = 0) in vec3 Position;

uniform mat4 ModelMat;
uniform mat4 PerspMat;

void main()
{
	gl_Position = PerspMat * ModelMat * vec4(Position, 1.0);
}
