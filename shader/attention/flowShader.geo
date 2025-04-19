#version 330 core

layout (points) in;
layout (line_strip, max_vertices = 2) out;

/** Flow from Vertex Shader */
in vec4 line_end_from_vs[];

/** Color from Vertex Shader */
in float line_length[];

/** CamID from Vertex Shader */
in int camID_from_vs[];

/** State from Vertex Shader */
flat in int state_from_vs[];

/** Color from Geometry Shader */
out vec4 color_from_gs;

/** State from Geometry Shader */
flat out int state_from_gs;

/** State from Geometry Shader */
flat out int camID_from_gs;

void main() {
	int camID = camID_from_vs[0];
	int state = state_from_vs[0];
	
	if(camID == 0){
		gl_Position = gl_in[0].gl_Position;
		color_from_gs = vec4(0.0, 0.0, 1.0, 1.0);
		state_from_gs = state;
		camID_from_gs = camID;
		EmitVertex();
		
		gl_Position = line_end_from_vs[0];
		color_from_gs = vec4(line_length[0] * 40, line_length[0] * 40, 1.0, 1.0);
		state_from_gs = state;
		camID_from_gs = camID;
		EmitVertex();
	} else if(camID == 1){
		gl_Position = gl_in[0].gl_Position;
		color_from_gs = vec4(0.0, 1.0, 0.0, 1.0);
		state_from_gs = state;
		camID_from_gs = camID;
		EmitVertex();
		
		gl_Position = line_end_from_vs[0];
		color_from_gs = vec4(line_length[0] * 40, 1.0, line_length[0] * 40, 1.0);
		state_from_gs = state;
		camID_from_gs = camID;
		EmitVertex();
	} else if(camID == 2){
		gl_Position = gl_in[0].gl_Position;
		color_from_gs = vec4(1.0, 0.0, 0.0, 1.0);
		state_from_gs = state;
		camID_from_gs = camID;
		EmitVertex();
		
		gl_Position = line_end_from_vs[0];
		color_from_gs = vec4(1.0, line_length[0] * 40, line_length[0] * 40, 1.0);
		state_from_gs = state;
		camID_from_gs = camID;
		EmitVertex();
	}
    
    EndPrimitive();
}