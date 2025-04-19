#version 330

/** Position */
layout(location = 0) in vec4 position;

/** Flow */
layout(location = 1) in vec4 nextPosition;

out vec4 vPosition;
out vec4 vNextPosition;

/**
 * Since this is the vertex shader, the following program is
 * executed for each vertex (i.e. triangle point) individually.
 */
void main()
{
	vPosition = vec4(position.xyz, 1.0);
	vNextPosition = vec4(nextPosition.xyz, 1.0);

    // Calculate gl_Position:
    gl_Position = vec4(position.xyz, 1.0);
}
