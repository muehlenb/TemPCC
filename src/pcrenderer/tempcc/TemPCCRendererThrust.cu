#include "src/pcrenderer/tempcc/TemPCCRendererThrust.h"

#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>

/**
 * CUDA FUNCTIONS for the attention renderer.
 *
 * This namespace especially contains the kernels.
 */

namespace cuda_tempcc {
    APoint* thrustPartitionByGTIndex(APoint* gpuPoints, int pointsNum){
        return thrust::partition(thrust::device, gpuPoints, gpuPoints + pointsNum,  [] __device__ (const APoint& p){
            return p.groundTruthIdx >= 0 && p.state != 0;
        });
    }

    APoint* thrustRemoveIfInvalid(APoint* gpuPoints, int pointsNum){
        return thrust::remove_if(thrust::device, gpuPoints, gpuPoints + pointsNum,  [] __device__ (const APoint& p){
            return p.state == 0 || p.state >= 10;
        });
    }

    APoint* thrustPartitionValid(APoint* gpuPoints, int pointsNum){
        return thrust::partition(thrust::device, gpuPoints, gpuPoints + pointsNum,  [] __device__ (const APoint& p){
            return p.state != 0;
        });
    }

    void thrustSortByCamDistance(APoint* points, int pointsNum, float4* deviceLastCamPos){
        thrust::sort(thrust::device, points, points + pointsNum, [deviceLastCamPos] __device__ (const APoint& a, const APoint& b) {
            float a_dX = a.position.x - deviceLastCamPos->x;
            float a_dY = a.position.y - deviceLastCamPos->y;
            float a_dZ = a.position.z - deviceLastCamPos->z;
            float b_dX = b.position.x - deviceLastCamPos->x;
            float b_dY = b.position.y - deviceLastCamPos->y;
            float b_dZ = b.position.z - deviceLastCamPos->z;

            return a_dX * a_dX + a_dY * a_dY + a_dZ * a_dZ > b_dX * b_dX + b_dY * b_dY + b_dZ * b_dZ;
        });
    }
}
