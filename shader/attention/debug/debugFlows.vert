#version 330

/** Position */
layout(location = 0) in vec3 position;

/** Flow */
layout(location = 1) in vec3 flow;

out vec4 vPosition;
out vec4 vFlow;

/**
 * Since this is the vertex shader, the following program is
 * executed for each vertex (i.e. triangle point) individually.
 */
void main()
{
	vPosition = vec4(position, 1.0);
	vFlow = vec4(flow, 0.0);

    // Calculate gl_Position:
    gl_Position = vec4(position, 1.0);
}
