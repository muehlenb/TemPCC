#version 330

out vec4 fragment_color;

uniform vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

void main()
{
    fragment_color = color;
}
