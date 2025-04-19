#version 330

uniform vec4 color = vec4(0.0, 0.0, 0.0, 0.0);

/**
 * This first defined out variable of the fragment shader defines
 * with which color the fragment is rendered
 */
out vec4 fragment_color;

void main()
{
    fragment_color = color;
}
