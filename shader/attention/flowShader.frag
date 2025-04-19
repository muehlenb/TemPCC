#version 330

/**
 * This first defined out variable of the fragment shader defines
 * with which color the fragment is rendered
 */
out vec4 fragment_color;

in vec4 color_from_gs;
flat in int state_from_gs;
flat in int camID_from_gs;

uniform bool onlyDrawHiddenPoints = false;
uniform bool onlyDrawVisiblePoints = false;

uniform bool showCam1 = false;
uniform bool showCam2 = false;
uniform bool showCam3 = false;

void main()
{
	if(onlyDrawHiddenPoints && state_from_gs != 2)
		discard;
		
	if(onlyDrawVisiblePoints && state_from_gs != 1)
		discard;
		
	if((camID_from_gs == 0 && !showCam1) || (camID_from_gs == 1 && !showCam2) || (camID_from_gs == 2 && !showCam3))
		discard;
		
    fragment_color = color_from_gs.bgra;
}
