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

/** Color from vertex shader */
out float line_length;

/** Flow from vertex shader */
out vec4 line_end_from_vs;

/** CamID */
out int camID_from_vs;

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
 * Since this is the vertex shader, the following program is
 * executed for each vertex (i.e. triangle point) individually.
 */
void main()
{
	camID_from_vs = camID;
	state_from_vs = state;
	
	line_length = length(flow);
	
	if(/*gl_VertexID % 2 == 0 && */line_length < 1.0){
		line_end_from_vs = projection * view * (position + flow * 2);
	} else {
		line_end_from_vs = projection * view * position;
	}

    // Calculate gl_Position:
    gl_Position = projection * view * position;
}
