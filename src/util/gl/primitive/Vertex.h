// © 2022, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

// Include this file only once when compiling:
#pragma once

// Include our Vec4f class of the last exercise:
#include "Vec4.h"

// Include a struct for storing uv coordinates:
#include "TexCoord.h"

/**
 * Represents a Vertex with different types of information.
 */
struct Vertex {
    /** Stores the position (X,Y,Z,1) of this vertex */
    Vec4f position;

    /** Stores the normal (X,Y,Z,0) of this vertex */
    Vec4f normal;

    /** Stores the color (R,G,B,A) of this vertex */
    Vec4f color;

    /** Texture coordinate of this vertex */
    TexCoord uv;

    /** Initializes the attributes with the given parameter values */
    Vertex(Vec4f position = Vec4f(), Vec4f normal = Vec4f(0.0f,0.0f,0.0f,0.0f), Vec4f color = Vec4f(), TexCoord uv = TexCoord(0,0))
        : position(position)
        , normal(normal)
        , color(color)
        , uv(uv)
    {}
};

// Define Vertex as Point:
typedef Vertex Point;
