#version 330 core

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


layout (points) in;
layout (line_strip, max_vertices = 2) out;

in vec4 vPosition[];
in vec4 vNextPosition[];

void main() {
	gl_Position =  projection * view * model * vPosition[0];
	EmitVertex();
	
	gl_Position =  projection * view * model * vNextPosition[0];
	EmitVertex();
	
    EndPrimitive();
}  