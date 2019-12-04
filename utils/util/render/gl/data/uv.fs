#version 330

out vec4 FragColor;

in vec3 CamColor;

void main()
{
	FragColor = vec4(CamColor, 1.0);
}
