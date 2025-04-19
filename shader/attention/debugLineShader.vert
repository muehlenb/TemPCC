#version 330

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
 * Start Pos
 */
uniform vec4 startPos;

/**
 * End Pos
 */
uniform vec4 endPos;

/**
 * Since this is the vertex shader, the following program is
 * executed for each vertex (i.e. triangle point) individually.
 */
void main()
{
	vec4 position = vec4(0,0,0,1);
	
	if(gl_VertexID == 0)
		position = startPos;
	else if(gl_VertexID == 1)
		position = endPos;
		
    // Calculate gl_Position:
    gl_Position = projection * view * position;
}
