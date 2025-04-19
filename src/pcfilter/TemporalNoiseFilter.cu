// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "pcfilter/TemporalNoiseFilter.h"

#include <device_launch_parameters.h>

// Register this filter globally in the application:
TemporalNoiseFilterFactory globalTemporalNoiseFilter;

namespace cuda_spacial_hole_filler {
    __global__ void kernel(float4* currentPositions, float4* smoothedPositions, float smoothFactor, int width, int height, int pcID) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;

        if (idx < width && idy < height) {
            int id = idy * width + idx;

            float distX = (smoothedPositions[id].x - currentPositions[id].x);
            float distY = (smoothedPositions[id].y - currentPositions[id].y);
            float distZ = (smoothedPositions[id].z - currentPositions[id].z);
            float dist = sqrt(distX*distX + distY*distY + distZ*distZ);

            if(dist < 0.03f){
                smoothedPositions[id].x = smoothedPositions[id].x * smoothFactor + currentPositions[id].x * (1 - smoothFactor);
                smoothedPositions[id].y = smoothedPositions[id].y * smoothFactor + currentPositions[id].y * (1 - smoothFactor);
                smoothedPositions[id].z = smoothedPositions[id].z * smoothFactor + currentPositions[id].z * (1 - smoothFactor);
                smoothedPositions[id].w = 1.f;
                currentPositions[id] = smoothedPositions[id];
            } else {
                smoothedPositions[id] = currentPositions[id];
            }
        }
    }
}

/**
 * Applies this noise filter.
 */
void TemporalNoiseFilter::applyFilter(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds) {
    // Store pointer to the last point cloud:
    if(!initialized){
        for(std::shared_ptr<OrganizedPointCloud>& pc : pointClouds){
            // Position image:
            float4* gpuPositionImg;
            cudaMallocManaged(&gpuPositionImg, pc->width * pc->height * sizeof(float4));
            cudaMemcpy(gpuPositionImg, pc->positions, pc->width * pc->height * sizeof(float4), cudaMemcpyHostToDevice);

            gpuPositionImages.push_back(gpuPositionImg);
        }
        initialized = true;
    }

    for(unsigned int i = 0; i < pointClouds.size(); ++i){
        std::shared_ptr<OrganizedPointCloud>& pointCloud = pointClouds[i];

        pointCloud->toGPU();

        int height = int(pointCloud->height);
        int width = int(pointCloud->width);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        float4* pos = pointCloud->gpuPositions;
        float4* storedPos = gpuPositionImages[i];

        cuda_spacial_hole_filler::kernel<<<numBlocks, threadsPerBlock>>>(pos, storedPos, smoothFactor, width, height, i);
        cudaDeviceSynchronize();
    }
};
