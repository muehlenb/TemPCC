#version 330

/**
 * This first defined out variable of the fragment shader defines
 * with which color the fragment is rendered
 */
out vec4 fragment_color;

/**
 * The texture of the mesh which is used for rendering:
 */
uniform sampler2D materialColorTexture;

/**
 * Should the texture be used?
 */
uniform bool useMaterialColorTexture;

/**
 * The color of the mesh:
 */
uniform vec3 materialColor;

/**
 * The shininess of the mesh:
 */
uniform float materialShininess;

/**
 * Position of this fragment in camera space.
 */
in vec3 position_from_vs;

/**
 * Normal of this fragment in camera space.
 */
in vec3 normal_from_vs;

/**
 * Texture coordinate from the vertex shader.
 */
in vec2 texCoord_from_vs;

/**
 * Ambient light color (I_amb):
 */
vec3 ambientColor = vec3(0.02, 0.02, 0.02);

// Structure for point lights:
struct Light {
    // Position of the point light in camera space:
    vec3 position;

    // Color of the point light:
    vec3 color;

    // Range of the point light:
    float range;
};

// Predefine 100 point lights (since arrays must have a fixed size in glsl):
uniform Light lights[100];

// Number of lights that should actually be used:
uniform int lightNum;

/**
 * The main function of the fragment shader, which is executed
 * for every fragment.
 */
void main()
{
    // Define parameter k:
    vec3 k = materialColor;

    // Multiply the materialColorTexture with k, if it's available:s
    if(useMaterialColorTexture){
        k *= texture(materialColorTexture, texCoord_from_vs).rgb;
    }

    // TODO
    // (1a) Calculate I_out and write that value (instead of k) into fragment_color.
    // (1b) Implement the given attenuation function.

    // Define iOut variable:
    vec3 iOut = ambientColor * k;

    // Addiere das Licht aller Lichter zu iOut hinzu:
    for(int i = 0; i < lightNum; ++i){
        Light light = lights[i];

        // Lichtvektor:
        vec3 l = light.position - position_from_vs;
        float lightDist = length(l);
        l /= lightDist;

        // Normal (normalisiert):
        vec3 n = normalize(normal_from_vs);

        float nDotL = dot(n, l);

        // Eye-Vektor (vom Licht zur Kamera):
        vec3 e = normalize(-position_from_vs);

        // Half-Vektor:
        vec3 h = (l + e) / length(l + e);

        // Wenn Winkel größer 90, nicht beleuchten:
        if(nDotL < 0.0)
           continue;

        // Abschattung:
        float intensity = lightDist > light.range ? 0 : pow(1 - (lightDist / light.range), 2);

        // Diffuser Lichtanteil:
        iOut += k * light.color * nDotL * intensity;

        // Specularer Lichtanteil:
        iOut += k * light.color * pow(dot(h, n), materialShininess) * intensity;
    }

    fragment_color = vec4(iOut, 1.0);
}
