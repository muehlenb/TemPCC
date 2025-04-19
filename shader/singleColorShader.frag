#version 330

/**
 * This first defined out variable of the fragment shader defines
 * with which color the fragment is rendered
 */
out vec4 fragment_color;

/**
 * This is the color you can define in the IMGUI menu.
 *
 * (In this shader, we don't use the vertex_color attributes or
 * a texture, we just define a single color for the whole mesh
 * in the C++ code)
 */
uniform vec4 color;

/**
 * The main function of the fragment shader, which is executed
 * for every fragment.
 */
void main()
{
    // Set the output color variable to the color of the vertex:
    fragment_color = color;
}

