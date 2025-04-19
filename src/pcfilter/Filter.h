// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)#pragma once
#pragma once

#include "util/OrganizedPointCloud.h"

#include <vector>

class FilterFactory;

/**
 * Represents a class which applies a filter to the given input point cloud.
 */
class Filter {
    static int instanceCounter;

public:
    /**
     * An ID to identify the filter.
     */
    const int instanceID;

    /**
     * Should the apply Filter function be called from the pipeline / main function?
     */
    bool isActive = true;

    /**
     * Default filter constructor.
     */
    Filter() : instanceID(instanceCounter++){}

    /**
     * Applies the filter to the point cloud.
     */
    virtual void applyFilter(std::vector<std::shared_ptr<OrganizedPointCloud>>& images) = 0;
};

/**
 * A filter factory that can instanciate a specific filter. All available filters should
 * have a filter factory implemented and registered.
 */
class FilterFactory {
public:
    virtual std::shared_ptr<Filter> createInstance() = 0;
    virtual std::string getDisplayName() = 0;

    /**
     * Stores all available filters that can be instanciated.
     */
    static std::vector<FilterFactory*> availableFilterFactories;
};
