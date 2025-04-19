#version 330

uniform vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

in float progress;

/**
 * This first defined out variable of the fragment shader defines
 * with which color the fragment is rendered
 */
out vec4 fragment_color;


void main()
{
    fragment_color = vec4((progress * 0.5 + 0.5) * color.rgb, color.a);
}
