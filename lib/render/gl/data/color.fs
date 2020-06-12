#version 330 core

// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
out vec4 FragColor;

in vec3 Color;

void main() 
{
    FragColor = vec4(Color,1.0);
}
