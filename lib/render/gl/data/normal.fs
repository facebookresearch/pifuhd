#version 330

out vec4 FragColor;

in vec3 CamNormal;

void main()
{
    vec3 cam_norm_normalized = normalize(CamNormal);
    vec3 rgb = (cam_norm_normalized + 1.0) / 2.0;
	FragColor = vec4(rgb, 1.0);
}
