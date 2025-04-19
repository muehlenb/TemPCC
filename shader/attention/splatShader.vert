#version 330

/** Position */
layout(location = 0) in vec4 position;

/** Flow */
layout(location = 1) in vec4 flow;

/** Colors */
layout(location = 2) in vec4 color;

/** Ground Truth Index */
layout(location = 3) in int groundTruthIdx;

/** Lifetime */
layout(location = 4) in int lifetime;

/** Valid */
layout(location = 5) in int state;

/** Cam ID */
layout(location = 6) in int camID;

flat out int state_from_vs;
flat out int camID_from_vs;
flat out int lifetime_from_vs;
flat out int groundTruthIdx_from_vs;

flat out int vertexId_from_vs;

/** Color from vertex shader */
out vec4 color_from_vs;

/** World position */
out vec4 world_position;


/**
 * Projection matrix which performs a perspective transformation.
 */
uniform mat4 projection;

/**
 * View matrix which transforms the vertices from world space
 * into camera space.
 */
uniform mat4 view;

uniform int debugID = 0;
uniform float pointSize;


/**
 * Since this is the vertex shader, the following program is
 * executed for each vertex (i.e. triangle point) individually.
 */
void main()
{	
	color_from_vs = color;
	world_position = position;
	
	state_from_vs = state;
	lifetime_from_vs = lifetime;
	groundTruthIdx_from_vs = groundTruthIdx;
	camID_from_vs = camID;
	
	vertexId_from_vs = gl_VertexID;

	// Set point size:
	gl_PointSize = pointSize;
	
	if(gl_VertexID == debugID)
		gl_PointSize = 10;
	
    // Calculate gl_Position:
    gl_Position = projection * view * world_position;
}
