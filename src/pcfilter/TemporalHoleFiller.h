// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)#pragma once
#pragma once

#include "pcfilter/Filter.h"

#include <cuda_runtime.h>

class TemporalHoleFiller : public Filter {
private:
    bool initialized = false;

    std::vector<uchar4*> gpuColorImages;
    std::vector<float4*> gpuPositionImages;

public:
    ~TemporalHoleFiller(){
        for(uchar4* gpuColorImg : gpuColorImages)
            cudaFree(gpuColorImg);

        for(float4* gpuPositionImg : gpuPositionImages)
            cudaFree(gpuPositionImg);
    }

    /**
     * Applies this noise filter.
     */
    virtual void applyFilter(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds) override;
};

/**
 * An erosion filter factory that can create a clipping filter.
 */
class TemporalHoleFillerFactory : public FilterFactory {
public:
    TemporalHoleFillerFactory(){
        FilterFactory::availableFilterFactories.push_back(this);
    }

    virtual std::shared_ptr<Filter> createInstance() override{ return std::make_shared<TemporalHoleFiller>(); };
    virtual std::string getDisplayName() override{ return "Temporal Hole Filler"; };
};
