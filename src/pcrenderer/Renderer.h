// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)
#pragma once

#include "src/util/OrganizedPointCloud.h"

#include <memory>
#include <vector>

/**
 * Represents a class which performs a point cloud rendering (and dynamic fusion algorithm).
 */
class Renderer {
public:
    /**
     * Integrate new RGB XYZ images.
     */
    virtual void integratePointClouds(std::vector<std::shared_ptr<OrganizedPointCloud>> pointClouds) = 0;

    /**
     * Renders the point cloud
     */
    virtual void render(Mat4f projection, Mat4f view) = 0;

    /**
     * Stores the names of available pc fusion algorithms.
     */
    static const char* availableAlgorithmNames[];

    /**
     * Returns an array with names of available pc streamers.
     */
    static const unsigned int availableAlgorithmNum;

    /**
     * Constructs a PCFusion object with an algorithm of the given type
     * (which corresponds to the index of getAvailableNames()).
     */
    static std::shared_ptr<Renderer> constructAlgorithmInstance(int type);
};
