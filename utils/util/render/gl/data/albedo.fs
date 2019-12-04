#version 330 core
out vec4 FragColor;

in vec2 TextureCoord;

uniform sampler2D TargetTexture;

void main()
{
    FragColor = texture(TargetTexture, TextureCoord);
    //FragColor = vec4(TextureCoord,1,1);
}