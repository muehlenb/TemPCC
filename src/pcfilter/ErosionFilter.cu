// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "pcfilter/ErosionFilter.h"

#include <device_launch_parameters.h>

// Register this filter globally in the application:
ErosionFilterFactory globalErosionFilterFactory;

namespace cuda_erosion_filter {
    __device__ float4 Vec4SubVec4(float4 vec, float4 vec2){
        float4 result;
        result.x = vec.x - vec2.x;
        result.y = vec.y - vec2.y;
        result.z = vec.z - vec2.z;
        result.w = vec.w - vec2.w;
        return result;
    }

    __device__ float Vec4Length(float4 vec){
        return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
    }

    __global__ void applyPCErosionFilterKernel(float4* currentPositions, int intensity, float distanceThresholdPerMeter, int width, int height, int pcID) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;

        if (idx < width && idy < height) {
            int id = idy * width + idx;

            int indicator = 0;

            float4& p = currentPositions[id];
            for(int dY = -intensity; dY <= intensity; ++dY){
                for(int dX = -intensity; dX <= intensity; ++dX){
                    int qX = idx + dX;
                    int qY = idy + dY;

                    if(qX < 0 || qX >= width || qY < 0 || qY >= height)
                        continue;

                    int qId = qY * width + qX;

                    float4& q = currentPositions[qId];

                    float len = Vec4Length(Vec4SubVec4(p,q));

                    if(isnan(q.x) || len > distanceThresholdPerMeter * p.z)
                        --indicator;
                    else
                        ++indicator;
                }
            }

            if(indicator < 0){
                p.x = NAN;
                p.y = NAN;
                p.z = NAN;
                p.w = NAN;
            }
        }
    }
}

/**
 * Applies this noise filter.
 */
void ErosionFilter::applyFilter(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds) {
    for(unsigned int i = 0; i < pointClouds.size(); ++i){
        std::shared_ptr<OrganizedPointCloud>& pointCloud = pointClouds[i];
        pointCloud->toGPU();

        int height = int(pointCloud->height);
        int width = int(pointCloud->width);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        float4* pos = pointCloud->gpuPositions;
        cuda_erosion_filter::applyPCErosionFilterKernel<<<numBlocks, threadsPerBlock>>>(pos, intensity, distanceThresholdPerMeter, width, height, i);
        cudaDeviceSynchronize();
    }
};
