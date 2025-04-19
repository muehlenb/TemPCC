#version 330

/**
 * This first defined out variable of the fragment shader defines
 * with which color the fragment is rendered
 */
out vec4 fragment_color;

in vec4 color_from_vs;


void main()
{
    fragment_color = color_from_vs.bgra;
}
