#version 330

/** Input variable (the position, which is vertex attribute 0) */
layout(location = 0) in vec4 vertex_position;

/** Input variable (the normal, which is vertex attribute 1) */
layout(location = 1) in vec4 vertex_normal;

/** Input variable (the color, which is vertex attribute 2) (NOT USED HERE) */
layout(location = 2) in vec4 vertex_color;

/** Input variable (the tex coord, which is vertex attribute 3) */
layout(location = 3) in vec4 vertex_texCoord;

/** Outputs the position of the vertex in camera space */
out vec3 position_from_vs;

/** Outputs the normal of the vertex in camera space */
out vec3 normal_from_vs;

/** Outputs the texture coordinate */
out vec2 texCoord_from_vs;

/**
 * Projection matrix which performs a perspective transformation.
 */
uniform mat4 projection;

/**
 * View matrix which transforms the vertices from world space
 * into camera space.
 */
uniform mat4 view;

/**
 * Model matrix which transforms the vertices from model space
 * into world space.
 */
uniform mat4 model;

/**
 * Since this is the vertex shader, the following program is
 * executed for each vertex (i.e. triangle point) individually.
 */
void main()
{
    // Forward vertex_texCoord to fragmentShader:
    texCoord_from_vs = vertex_texCoord.st;

    // Sets the position in camera space:
    position_from_vs = (view * model * vertex_position).xyz;

    // Set the normal in camera space:
    normal_from_vs = normalize(mat3(view * model) * vertex_normal.xyz);

    // Calculate gl_Position:
    gl_Position = projection * vec4(position_from_vs, 1.0);
}
