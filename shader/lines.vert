#version 330

/** Color from vertex shader */
out vec4 color_from_vs;

/** World position */
out vec4 world_position;

/** Position texture */
uniform sampler2D positionTexture;

/** Color texture */
uniform sampler2D directionTexture;

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
	ivec2 dirTexSize = textureSize(directionTexture, 0);
	
	int sX = (gl_VertexID / 2) % dirTexSize.x;
	int sY = (gl_VertexID / 2) / dirTexSize.x;
	
	vec4 position = vec4(0.0, 0.0, 0.0, 1.0);
	vec4 color = vec4(0.0, 0.0, 1.0, 1.0);
		
	int factor =  texSize.y / dirTexSize.y;
		
	int lX = sX * factor;
	int lY = sY * factor;

	float u = float(lX) / texSize.x;
	float v = float(lY) / texSize.y;

	position = texture(positionTexture, vec2(u, v));
		
	if(gl_VertexID % 2 == 1){	
		float sU = float(sX) / dirTexSize.x;
		float sV = float(sY) / dirTexSize.y;
	
		position += vec4(texture(directionTexture, vec2(sU, sV)).xyz * 2.5, 0);
		color = vec4(1.0, 1.0, 1.0, 1.0);
	}
	
	color_from_vs = color;
	world_position = model * position;

	
    // Calculate gl_Position:
    gl_Position = projection * view * world_position;
}
