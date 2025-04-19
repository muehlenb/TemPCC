#version 330

/** Position */
layout(location = 0) in vec3 position;

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


uniform int pointSize = 3;
/**
 * Since this is the vertex shader, the following program is
 * executed for each vertex (i.e. triangle point) individually.
 */
void main()
{
	// Set point size:
	gl_PointSize = pointSize;
	
    // Calculate gl_Position:
    gl_Position = projection * view * model * vec4(position, 1.0);
}
