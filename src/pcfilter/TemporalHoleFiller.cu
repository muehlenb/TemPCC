// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "pcfilter/TemporalHoleFiller.h"

#include <device_launch_parameters.h>

// Register this filter globally in the application:
TemporalHoleFillerFactory globalTemporalHoleFillerFactory;

namespace cuda_temporal_hole_filler {
    __global__ void kernel(float4* currentPositions, uchar4* currentColors, float4* previousPositions, uchar4* previousColors, int width, int height, int pcID) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;

        if (idx < width && idy < height) {
            int id = idy * width + idx;

            if(!(currentPositions[id].x > 0.01f || currentPositions[id].y > 0.01f || currentPositions[id].z > 0.01f)){
                float4 position = currentPositions[id];
                uchar4 color = currentColors[id];
                currentPositions[id] = previousPositions[id] ;
                currentColors[id] = previousColors[id];
                previousPositions[id] = position;
                previousColors[id] = color;
            } else {
                previousPositions[id] = currentPositions[id];
                previousColors[id] = currentColors[id];
            }
        }
    }
}

/**
 * Applies this noise filter.
 */
void TemporalHoleFiller::applyFilter(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds) {
    // Store pointer to the last point cloud:
    if(!initialized){
        for(std::shared_ptr<OrganizedPointCloud>& pc : pointClouds){
            // Color image:
            uchar4* gpuColorImg;
            cudaMallocManaged(&gpuColorImg, pc->width * pc->height * sizeof(uchar4));
            cudaMemcpy(gpuColorImg, pc->colors, pc->width * pc->height * sizeof(uchar4), cudaMemcpyHostToDevice);

            // Position image:
            float4* gpuPositionImg;
            cudaMallocManaged(&gpuPositionImg, pc->width * pc->height * sizeof(float4));
            cudaMemcpy(gpuPositionImg, pc->positions, pc->width * pc->height * sizeof(float4), cudaMemcpyHostToDevice);

            gpuColorImages.push_back(gpuColorImg);
            gpuPositionImages.push_back(gpuPositionImg);
        }
        initialized = true;
    }

    for(unsigned int i = 0; i < pointClouds.size(); ++i){
        std::shared_ptr<OrganizedPointCloud>& pointCloud = pointClouds[i];

        /*
        std::cout << "Start:" << std::endl;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "\t Error: %s\n", cudaGetErrorString(err));
        }*/

        pointCloud->toGPU();

        int height = int(pointCloud->height);
        int width = int(pointCloud->width);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        float4* pos = pointCloud->gpuPositions;
        uchar4* color = pointCloud->gpuColors;
        float4* prevPos = gpuPositionImages[i];
        uchar4* prevColor = gpuColorImages[i];

        /*
        std::cout << "Before Kernel Application:" << std::endl;
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "\t Error: %s\n", cudaGetErrorString(err));
        }*/

        cuda_temporal_hole_filler::kernel<<<numBlocks, threadsPerBlock>>>(pos, color, prevPos, prevColor, width, height, i);
        cudaDeviceSynchronize();

        /*
        float* test = new float[16];
        std::cout << "After Kernel Application:" << std::endl;
        err = cudaMemcpy(test, pos, 16*4, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "\t GPU->CPU %s\n", cudaGetErrorString(err));
        }*/
    }
};
