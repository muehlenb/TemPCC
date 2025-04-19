// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)#pragma once
#pragma once

#include "pcfilter/Filter.h"

#include <cuda_runtime.h>

class SpatialHoleFiller : public Filter {
public:
    float requiredValidNeighborRatio = 0.1f;
    int intensity = 3;
    float maxDistance = 0.1f;

    /**
     * Applies this noise filter.
     */
    virtual void applyFilter(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds) override;
};

/**
 * An erosion filter factory that can create a clipping filter.
 */
class SpatialHoleFillerFactory : public FilterFactory {
public:
    SpatialHoleFillerFactory(){
        FilterFactory::availableFilterFactories.push_back(this);
    }

    virtual std::shared_ptr<Filter> createInstance() override{ return std::make_shared<SpatialHoleFiller>(); };
    virtual std::string getDisplayName() override{ return "Spatial Hole Filler"; };
};
