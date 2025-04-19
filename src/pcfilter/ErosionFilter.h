// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)#pragma once
#pragma once

#include "pcfilter/Filter.h"

#include <cuda_runtime.h>

class ErosionFilter : public Filter {
public:
    int intensity = 2;
    float distanceThresholdPerMeter = 0.03f;

    ~ErosionFilter(){}

    virtual void applyFilter(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds) override;
};

/**
 * An erosion filter factory that can create a clipping filter.
 */
class ErosionFilterFactory : public FilterFactory {
public:
    ErosionFilterFactory(){
        FilterFactory::availableFilterFactories.push_back(this);
    }

    virtual std::shared_ptr<Filter> createInstance() override{ return std::make_shared<ErosionFilter>(); };
    virtual std::string getDisplayName() override{ return "Erosion Filter"; };
};
