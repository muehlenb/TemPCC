#version 330

/** Color from vertex shader */
out vec4 color_from_vs;

/** World position */
out vec4 world_position;

/** Position texture */
uniform sampler2D positionTexture;

/** Color texture */
uniform sampler2D colorTexture;

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
	ivec2 texSize = textureSize(positionTexture, 0);
	
	float u = float(gl_VertexID % texSize.x + 0.5) / texSize.x;
	float v = float(gl_VertexID / texSize.x + 0.5) / texSize.y;
	
	vec4 position = texture(positionTexture, vec2(u, v));
	vec4 color = texture(colorTexture, vec2(u, v));
	
	color_from_vs = color;
	world_position = model * position;

	// Set point size:
	gl_PointSize = 3;
	
    // Calculate gl_Position:
    gl_Position = projection * view * world_position;
}
