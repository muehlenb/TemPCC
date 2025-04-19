#version 330

/**
 * Precalculated gaussians with a variance of 0.2 and µ=0:
 */
const float gaussianValues[21] = float[21](
    0.8920620580763855, 0.8865040570590268, 0.8700369673862929, 
    0.8432687440138061, 0.8071711293576809, 0.7630211130438789,
    0.712326021513863, 0.6567383581775923, 0.5979670798364098,
    0.5376918905659376, 0.4774864115335566, 0.4187548654238963,
    0.3626853674196145, 0.31022123430514303, 0.26205009872639673,
    0.21860920899267267, 0.18010422336076826, 0.146538135035783,
    0.11774669940754752, 0.09343684018496816, 0.07322491280963243
);


out vec4 fragment_color;

in vec4 world_position;
in vec4 color_from_vs;

flat in int lifetime_from_vs;
flat in int state_from_vs;
flat in int groundTruthIdx_from_vs;
flat in int camID_from_vs;

flat in int vertexId_from_vs;

uniform vec4 overrideColor = vec4(0.0, 0.0, 0.0, 0.0);

uniform bool onlyDrawHiddenPoints = false;
uniform bool onlyDrawVisiblePoints = false;
uniform bool drawLifeTimeColor = false;
uniform bool colorizeHiddenPoints = false;

uniform bool showCam1 = false;
uniform bool showCam2 = false;
uniform bool showCam3 = false;

uniform int debugID;

void main()
{
	float factor = 1;
	
	//if(world_position.z > 0.0)
	//	factor = 0.35;


	float distFromMid = length(gl_PointCoord.xy - vec2(0.5)) * 2;
	
	if(distFromMid > 1)
		discard;
		
	if(vertexId_from_vs == debugID){
		fragment_color = vec4(1.0, 0.0, 0.0, 1.0);
		return;
	}
	
/*
	if(length(color_from_vs.xyz) == 0 && discardBlackPixels)
		discard;
*/

    fragment_color = color_from_vs.bgra * factor;
	
	if(length(overrideColor) > 0.01){
		fragment_color = overrideColor;
	}
	
	if(colorizeHiddenPoints && state_from_vs == 2)
		fragment_color.r *= 0;
	
	
	/*
	if(groundTruthIdx_from_vs >= 0){
		fragment_color = vec4(1.0,0.0,0.0,1.0);	
	} else {
		fragment_color = vec4(0.0,0.0,0.0,1.0);
	}*/
	
	if(drawLifeTimeColor && state_from_vs == 2 && lifetime_from_vs > 0){
		fragment_color = vec4(0, lifetime_from_vs * 0.05, 0, 1);
	}
	
	fragment_color.rgb += vec3(0.1,0.1,0.1);
	fragment_color.a = 1.0;//gaussianValues[int(distFromMid * 20)]/gaussianValues[0];
	
	if(onlyDrawHiddenPoints && state_from_vs != 2)
		discard;
		
	if(onlyDrawVisiblePoints && state_from_vs != 1)
		discard;
		
	if((camID_from_vs == 0 && !showCam1) || (camID_from_vs == 1 && !showCam2) || (camID_from_vs == 2 && !showCam3))
		discard;
}
