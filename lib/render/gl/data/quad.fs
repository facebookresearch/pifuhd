#version 330 core
// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenTexture;

void main()
{
    FragColor = texture(screenTexture, TexCoord);
}