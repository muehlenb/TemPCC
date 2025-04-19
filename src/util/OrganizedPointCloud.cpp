// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "util/OrganizedPointCloud.h"

/** Unused initialized gpu memory */
std::vector<OrganizedPointCloud::GPUMemory> OrganizedPointCloud::unusedInitializedGPUMemory = std::vector<OrganizedPointCloud::GPUMemory>();

/** Uploaded lookup tables */
std::map<float*, float2*> OrganizedPointCloud::uploadedLookup3DToImageTables = std::map<float*, float2*>();
std::map<float*, float2*> OrganizedPointCloud::uploadedLookupImageTo3DTables = std::map<float*, float2*>();

/** Clean up static memory */
void OrganizedPointCloud::cleanupStaticMemory(){
    for(const OrganizedPointCloud::GPUMemory& memory : unusedInitializedGPUMemory){
        cudaFree(memory.pointer);
    }

    for (auto const& entry : uploadedLookup3DToImageTables)
        cudaFree(entry.second);

    for (auto const& entry : uploadedLookupImageTo3DTables)
        cudaFree(entry.second);
}

