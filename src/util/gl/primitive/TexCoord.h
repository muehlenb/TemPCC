// © 2022, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

// Include this file only once when compiling:
#pragma once

/**
 * Define a struct for a texture coordinate.
 *
 * A texture coordinate defines a 2D location on an image
 * (e.g. a texture), so that an texture can be mapped
 * to the mesh.
 *
 * Texture coordinates are relative to the image size,
 * this means that both values are between 0 and 1.
 *
 * Note that this coordinates are often also called
 * UV-Coordates (even if they are accessible in glsl
 * via the st components ;-)).
 */

struct TexCoord
{
    /** The u component of the uv coordinate */
    float u;

    /** The v component of the uv coordinate */
    float v;
	
    TexCoord(float u=0.0f, float v=0.0f)
        : u(u)
        , v(v)
	{}
};
