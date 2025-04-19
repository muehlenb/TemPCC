// © 2022, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

// Include this file only once when compiling:
#pragma once

// Include our Vertex struct:
#include "Vertex.h"

/**
 * Represents a Triangle with three vertices a, b and c.
 */

struct Triangle {
    /** Create three variables for the vertices */
    Vertex a, b, c;

    /** Initialize the triangle with the given vertex data */
    Triangle(Vertex a = Vertex(), Vertex b = Vertex(), Vertex c = Vertex())
        : a(a)
        , b(b)
        , c(c)
    {}

    /**
     * Override the []-Operator to allow the access of the
     * vertices a, b and c via [0], [1] and [2].
     */
    Vertex& operator[](const int i){
        return i == 0 ? a : i == 1 ? b : c;
    }
};
