// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)#pragma once
#pragma once

#include "pcfilter/Filter.h"

#include <cuda_runtime.h>

/**
 * A clipping filter which clips point that lie out of a volume.
 */
class ClippingFilter : public Filter {
private:
public:
    Vec4f min = Vec4f(-1.5f, 0.01f, -2.f);
    Vec4f max = Vec4f(1.5f, 2.f, 2.f);

    /**
     * Applies this noise filter.
     */
    virtual void applyFilter(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds) override;
};

/**
 * A clipping filter factory that can create a clipping filter.
 */
class ClippingFilterFactory : public FilterFactory {
public:
    ClippingFilterFactory(){
        FilterFactory::availableFilterFactories.push_back(this);
    }

    virtual std::shared_ptr<Filter> createInstance() override{ return std::make_shared<ClippingFilter>(); };
    virtual std::string getDisplayName() override{ return "Clipping Filter"; };
};
