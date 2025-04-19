// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)
#pragma once

#include "src/pcrenderer/tempcc/TemPCCStructs.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace cuda_tempcc {
    void setConstantCamMatrix(float* data, int index);
    void setConstantInvCamMatrix(float* data, int index);
    void setConstantCamInfos(CamInfo& camInfo, int index);
    void setConstantModelMatrix(float* data);

    void cudaSetupRandStates(int numBlocks, int blockSize, curandState* states, unsigned long seed);

    void cudaSetupAPoints(int numBlocks, int blockSize, APoint* gpuPoints, int pointsNum);

    void cudaAssignGroundTruthData(int numBlocks, int blockSize, APoint* convertedCamPoints, float4* groundTruthPoints, int groundTruthPointNum);

    void cudaCopyCamPointsWOStaticIdx(int numBlocks, int blockSize, APoint* gpuPoints, APoint* convertedCamPoints, int camPointNum);

    void cudaCopyImageToPC(dim3 numBlocks, dim3 blockSize, APoint* target, int camId, int pcWidth, int pcHeight, float4* pcPositions, uchar4* pcColors, bool usePDFlow, int flowCols, int flowRows, float4* flow, int everyNPoint);
    void cudaWidenNearestKernel(dim3 numBlocks, dim3 blockSize, float* zResult, float4* gpuPositions, int pcWidth, int pcHeight, int kernelRadius);
    void cudaBorderDistanceKernel(dim3 numBlocks, dim3 blockSize, float* totalResult, float4* gpuPositions, int pcWidth, int pcHeight, int kernelRadius);

    void cudaCopyIntoTraningBuffer(int TrainingPointNum, int numBlocks, int blockSize, float4* pointNNTemporalInput, float3* pointNNSpatialInput, float4* pointNNOutput, float4* trainingNNTemporalInput, float3* trainingNNSpatialInput, float4* trainingNNOutput, int* traningShuffledIndices, curandState* randStates, int startOffset, int stride, int validPointsNum);

    void cudaMainPassKernelPartA(int numBlocks, int blockSize, APoint* gpuPoints, float4* gpuTemporalFlows, APoint* camOrderedPoints, float4* pointNNTemporalInput, float3* pointNNSpatialInput, float4* pointNNOutput, int* shuffledIndices, float4* groundTruthPoints, int groundTruthSampleCount, int validPointsNum, curandState* randStates, float scheduledSampling);
    void cudaMainPassKernelPartB(int numBlocks, int blockSize, APoint* gpuPoints, float4* gpuTemporalFlows, APoint* camOrderedPoints, float4* groundTruthPoints, float4* pointNNOutput, int* shuffledIndices, int validPointsNum, curandState* randStates, bool isTraining, float allowedTrainingDivergence, bool applyFlow);

    void cudaClearAllPoints(int numBlocks, int blockSize, APoint* gpuPoints, int pointNum);

    void cudaDensityFiltering(APoint* gpuPoints, int pointsNum, int* occupied, int occupiedSize);

    void cudaClearProjectedGTFlows(float4* gtPositions, float4* gtFlows, int width, int height);
    void cudaProjectGTFlowIntoImage(int camId, float4* gtPositions, float4* gtFlows, float4* pcPositions, float4* groundTruthPoints, int groundTruthPointNum);
    void cudaMakeProjectedGTFlowDense(int camId, float4* gtPositions, float4* gtFlows, float4* gtFlowsDense, float4* pcPositions, int width, int height);

    __global__ void clearProjectedGTFlows(float4* gtPositions, float4* gtFlows, int width, int height);
    __global__ void projectGTFlowIntoImage(int camId, float4* gtPositions, float4* gtFlows, float4* pcPositions, float4* groundTruthPoints, int groundTruthPointNum);
    __global__ void makeProjectedGTFlowDense(int camId, float4* gtPositions, float4* gtFlows, float4* gtFlowsDense, float4* pcPositions, int width, int height);

    __global__ void setupAPoints(float4* gtPositions, float4* gtFlows, int pointsNum);

    __device__ inline void predict(int id, APoint& point, float3* temporalFlows, float4* pointNNOutput, int* shuffledIndices, int validPointsNum, int applyFlow);
    __device__ inline int2 projectPointToCam(float4 point, int camId);

    __device__ inline void processWithCamPoints(APoint& point, APoint* camOrderedPoints, int pointId, int camId, curandState* randStates, bool isTraining, float allowedTrainingDivergence);
    __device__ inline float4 clampXYZ(float4& v, float minV = -1, float maxV = 1);

    __device__ float4 MatMultVec4(float mat[16], float4 vec);
    __device__ float4 Vec4Add(float4 vec, float4 vec2);
    __device__ float4 Vec4Subtract(float4 a, float4 b);
    __device__ float Vec4Length(float4 vec);

    __global__ void copyCamPointsWOStaticIdx(APoint* gpuPoints, APoint* convertedCamPoints, int pointsNum);

    __global__ void densityFiltering(APoint* gpuPoints, int pointsNum, int* occupied, int occupiedSize);
    __global__ void setupAPoints(APoint* gpuPoints, int pointsNum);
    __global__ void setupRandStates(curandState* states, unsigned long seed);
    __global__ void clearAllPoints(APoint *data, int num);
    __global__ void assignGroundTruthData(APoint* convertedCamPoints, float4* groundTruthPoints, int groundTruthPointNum);
    __global__ void copyImageToPC(APoint* target, int camId, int pcWidth, int pcHeight, float4* pcPositions, uchar4* pcColors, bool usePDFlow, int flowCols, int flowRows, float4* flow, int everyNPoint);
    __global__ void widenNearestKernel(float* zResult, float4* gpuPositions, int pcWidth, int pcHeight, int kernelRadius);
    __global__ void borderDistanceKernel(float* totalResult, float4* gpuPositions, int pcWidth, int pcHeight, int kernelRadius);

    __global__ void copyIntoTraningBuffer(int TrainingPointNum, float4* pointNNTemporalInput, float3* pointNNSpatialInput, float4* pointNNOutput, float4* trainingNNTemporalInput, float3* trainingNNSpatialInput, float4* trainingNNOutput, int* traningShuffledIndices, curandState* randStates, int startOffset, int stride, int validPointsNum);

    __global__ void mainPassKernelPartA(APoint* gpuPoints, float4* temporalFlows, APoint* camOrderedPoints, float4* pointNNTemporalInput, float3* pointNNSpatialInput, float4* pointNNOutput, int* shuffledIndices, float4* groundTruthPoints, int groundTruthSampleCount, int validPointsNum, curandState* randStates, float scheduledSampling);
    __global__ void mainPassKernelPartB(APoint* gpuPoints, float4* temporalFlows, APoint* camOrderedPoints, float4* groundTruthPoints, float4* pointNNOutput, int* shuffledIndices, int validPointsNum, curandState* randStates, bool isTraining, float allowedTrainingDivergence, bool applyFlow);
}
