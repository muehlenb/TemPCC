// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "pcfilter/SpatialHoleFiller.h"

#include <device_launch_parameters.h>

// Register this filter globally in the application:
SpatialHoleFillerFactory globalSpatialHoleFillerFactory;

namespace cuda_spacial_hole_filler {
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

    __global__ void kernel(float4* currentPositions, uchar4* currentColors, float2* lookupImageTo3D, int intensity, float maxDistance, float requiredValidNeighborRatio, int width, int height, int pcID) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;

        if (idx < width && idy < height) {
            int id = idy * width + idx;

            float4& p = currentPositions[id];
            uchar4& pCol = currentColors[id];

            // If point is valid, we don't need to fill it:
            if(!isnan(p.x) && p.z >= 0.01f)
                return;

            float sumDepth = 0.f;
            float3 sumCol = {0, 0, 0};
            float sumWeight = 0;

            int validNeighbors = 0;
            int totalNeighbors = 0;

            for(int dY = -intensity; dY <= intensity; ++dY){
                for(int dX = -intensity; dX <= intensity; ++dX){
                    if(abs(dX) + abs(dY) > intensity)
                        continue;

                    int qX = idx + dX;
                    int qY = idy + dY;

                    if(qX < 0 || qX >= width || qY < 0 || qY >= height)
                        continue;

                    int qId = qY * width + qX;

                    float4& q = currentPositions[qId];
                    uchar4& qCol = currentColors[qId];

                    if(!isnan(q.x) && q.z >= 0.01f){
                        // IT IS TOTALLY STRANGE THAT IT SEEMS TO WORK TO FILL HOLES
                        // WHEN THIS IF CONDITION IS REMOVED. I HAVE TO DEBUG IT
                        // TOMORROW.
                        float weight = 1.f;

                        sumDepth += Vec4Length(q) * weight;
                        sumWeight += weight;

                        sumCol.x += qCol.x * weight;
                        sumCol.y += qCol.y * weight;
                        sumCol.z += qCol.z * weight;

                        ++validNeighbors;
                    }

                    ++totalNeighbors;
                }
            }

            float2 xyPart = lookupImageTo3D[id];


            if(validNeighbors / float(totalNeighbors) >= requiredValidNeighborRatio){
                float repairedLength = sumDepth / sumWeight;
                float repairedDepth = repairedLength / sqrt(xyPart.x*xyPart.x + xyPart.y*xyPart.y + 1);

                p.x = xyPart.x * repairedDepth;
                p.y = xyPart.y * repairedDepth;
                p.z = repairedDepth;

                pCol.x = sumCol.x / sumWeight;
                pCol.y = sumCol.y / sumWeight;
                pCol.z = sumCol.z / sumWeight;
            }
        }
    }
}

/**
 * Applies this noise filter.
 */
void SpatialHoleFiller::applyFilter(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds) {
    for(unsigned int i = 0; i < pointClouds.size(); ++i){
        std::shared_ptr<OrganizedPointCloud>& pointCloud = pointClouds[i];

        pointCloud->toGPU();

        int height = int(pointCloud->height);
        int width = int(pointCloud->width);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        float4* pos = pointCloud->gpuPositions;
        uchar4* color = pointCloud->gpuColors;
        float2* lookupImageTo3D = pointCloud->gpuLookupImageTo3D;

        cuda_spacial_hole_filler::kernel<<<numBlocks, threadsPerBlock>>>(pos, color, lookupImageTo3D, intensity, maxDistance, requiredValidNeighborRatio, width, height, i);
        cudaDeviceSynchronize();
    }
};
