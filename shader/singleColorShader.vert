#version 330

/** Input variable (the position, which is vertex attribute 0) */
layout(location = 0) in vec4 vertex_position;

/** Input variable (the normal, which is vertex attribute 1) (NOT USED HERE)  */
layout(location = 1) in vec4 vertex_normal;

/** Input variable (the color, which is vertex attribute 2) (NOT USED HERE) */
layout(location = 2) in vec4 vertex_color;

/** Input variable (the tex coord, which is vertex attribute 3) (NOT USED HERE) */
layout(location = 3) in vec4 vertex_texCoord;

/**
 * Transformation matrix which transforms the vertices of this
 * mesh into the world coordinate system.
 */
uniform mat4 model;

/**
 * Transformation matrix which transforms the vertices in the
 * world coordinate system into the camera coordinate system.
 */
uniform mat4 view;

/**
 * Projection matrix which performs a perspective transformation.
 */
uniform mat4 projection;


/**
 * Since this is the vertex shader, the following program is
 * executed for each vertex (i.e. triangle point) individually.
 */
void main()
{

    // Calculate gl_Position using the projection, view and
    // model matrix:
    gl_Position = projection * view * model * vertex_position;
}
