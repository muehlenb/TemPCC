#version 330

/** Position */
layout(location = 0) in vec3 flow;

uniform int temporalSize;

out vec4 vPosition;
out vec4 vFlow;
out float vProgress;

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
	vProgress = float(gl_VertexID) / float(temporalSize);
	float xPos = (0.5 - vProgress) * 0.66; 

	vPosition = vec4(xPos, 0.0, 0.0, 1.0);
	vFlow = vec4(flow, 0.0);

    // Calculate gl_Position:
    gl_Position = projection * view * model * vPosition;
}
