#include "src/pcrenderer/tempcc/TemPCCRendererKernels.h"

#include <iostream>
#include "src/util/cuda/CuHashSet.h"

/**
 * CUDA FUNCTIONS for the attention renderer.
 *
 * This namespace especially contains the kernels.
 */

namespace cuda_tempcc {
    /**
     * Stores the camera matrices during the main pass.
     */
    __constant__ float camMatrices[3][16];
    __constant__ float invCamMatrices[3][16];

    /**
         * Stores information about the camera (resolution, lookupImageTo3D, etc.).
         */
    __constant__ CamInfo camInfos[3];

    void setConstantCamMatrix(float* data, int index){
        cudaMemcpyToSymbol(camMatrices, data, sizeof(float) * 16, sizeof(float) * 16 * index);
    }

    void setConstantInvCamMatrix(float* data, int index){
        cudaMemcpyToSymbol(invCamMatrices, data, sizeof(float) * 16, sizeof(float) * 16 * index);
    }

    void setConstantCamInfos(CamInfo& camInfo, int index){
        cudaMemcpyToSymbol(camInfos, &camInfo, sizeof(CamInfo), sizeof(CamInfo)*index);
    }

    __device__ float4 MatMultVec4(float mat[16], float4 vec){
        float4 result;
        result.x = mat[0] * vec.x + mat[4] * vec.y + mat[8] * vec.z + mat[12] * vec.w;
        result.y = mat[1] * vec.x + mat[5] * vec.y + mat[9] * vec.z + mat[13] * vec.w;
        result.z = mat[2] * vec.x + mat[6] * vec.y + mat[10] * vec.z + mat[14] * vec.w;
        result.w = mat[3] * vec.x + mat[7] * vec.y + mat[11] * vec.z + mat[15] * vec.w;
        return result;
    }

    __device__ float4 Vec4Add(float4 vec, float4 vec2){
        float4 result;
        result.x = vec.x + vec2.x;
        result.y = vec.y + vec2.y;
        result.z = vec.z + vec2.z;
        result.w = vec.w + vec2.w;
        return result;
    }

    __device__ float4 Vec4Subtract(float4 a, float4 b){
        return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
    }

    __device__ float Vec4Length(float4 vec){
        return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    }

    __global__ void setupRandStates(curandState* states, unsigned long seed) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, idx, 0, &states[idx]);
    }

    void cudaClearProjectedGTFlows(float4* gtPositions, float4* gtFlows, int width, int height){
        dim3 blockSize(16, 16);
        dim3 numBlocks(
            int(std::ceil(width / float(blockSize.x))),
            int(std::ceil(height / float(blockSize.y)))
        );

        cuda_tempcc::clearProjectedGTFlows<<<numBlocks, blockSize>>>(gtPositions, gtFlows, width, height);
    }

    void cudaProjectGTFlowIntoImage(int camId, float4* gtPositions, float4* gtFlows, float4* pcPositions, float4* groundTruthPoints, int groundTruthSampleCount){
        int blockSize = 256;
        int numBlocks = (groundTruthSampleCount + blockSize - 1) / blockSize;

        cuda_tempcc::projectGTFlowIntoImage<<<numBlocks, blockSize>>>(camId, gtPositions, gtFlows, pcPositions, groundTruthPoints, groundTruthSampleCount);
    }

    void cudaMakeProjectedGTFlowDense(int camId, float4* gtPositions, float4* gtFlows, float4* gtFlowsDense, float4* pcPositions, int width, int height){
        dim3 blockSize(16, 16);
        dim3 numBlocks(
            int(std::ceil(width / float(blockSize.x))),
            int(std::ceil(height / float(blockSize.y)))
        );

        cuda_tempcc::makeProjectedGTFlowDense<<<numBlocks, blockSize>>>(camId, gtPositions, gtFlows, gtFlowsDense, pcPositions, width, height);
    }

    __global__ void clearProjectedGTFlows(float4* gtPositions, float4* gtFlows, int width, int height){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height){
            gtPositions[x + y * width] = {0.f, 0.f, 0.f};
            gtFlows[x + y * width] = {0.f, 0.f, 0.f};
        }
    }

    __global__ void projectGTFlowIntoImage(int camId, float4* gtPositions, float4* gtFlows, float4* pcPositions, float4* groundTruthPoints, int groundTruthPointNum){
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < groundTruthPointNum){
            CamInfo& camInfo = camInfos[camId];
            float4 gtPoint = groundTruthPoints[index];

            int2 prj = projectPointToCam(gtPoint, camId);
            int prjIdx = prj.x + prj.y * camInfo.width;

            if(prj.x < camInfo.width && prj.y < camInfo.height && prj.x >= 0 && prj.y >= 0){
                float4& pos = gtPositions[prjIdx];
                float4& flow = gtFlows[prjIdx];
                float4& gtNext = groundTruthPoints[index + groundTruthPointNum];

                float4 p = MatMultVec4(camMatrices[camId], pcPositions[prjIdx]);
                float dist = Vec4Length(Vec4Subtract(p, gtPoint));

                if(dist < 0.1f){
                    // TODO: Sicherstellen, dass nicht geschrieben wird, falls eine Position näher ist.
                    pos = gtPoint;
                    flow = {gtNext.x - gtPoint.x, gtNext.y - gtPoint.y, gtNext.z - gtPoint.z, 0.f};
                }
            }
        }
    }

    __global__ void makeProjectedGTFlowDense(int camId, float4* gtPositions, float4* gtFlows, float4* gtFlowsDense, float4* pcPositions, int width, int height){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        int idx = x + y * width;

        gtFlowsDense[idx] = gtFlows[idx];

        float4 pcPos = MatMultVec4(camMatrices[camId], pcPositions[idx]);
        float3 sum = {0.f, 0.f, 0.f};
        float weight = 0;

        if (x < width && y < height){
            for(int dY = -5; dY <= 5; ++dY){
                for(int dX = -5; dX <= 5; ++dX){
                    int dIdx = (x + dX) + (y + dY) * width;

                    float4& dPos = gtPositions[dIdx];
                    float4& dFlow = gtFlows[dIdx];

                    float dDist3D = Vec4Length(Vec4Subtract(pcPos, dPos));

                    if(dDist3D < 0.2f){
                        if(dFlow.x != 0.f || dFlow.y != 0.f || dFlow.z != 0.f){
                        float w = 0.2f - dDist3D;
                            sum = {sum.x + dFlow.x * w, sum.y + dFlow.y * w, sum.z + dFlow.z * w};
                            weight += w;
                        }
                    }
                }
            }
        }

        gtFlowsDense[idx] = {sum.x / weight, sum.y / weight, sum.z / weight, 0.f};
    }

    void cudaClearAllPoints(int numBlocks, int blockSize, APoint* gpuPoints, int pointNum){
        cuda_tempcc::clearAllPoints<<<numBlocks, blockSize>>>(gpuPoints, pointNum);
    }

    void cudaSetupRandStates(int numBlocks, int blockSize, curandState* states, unsigned long seed){
        cuda_tempcc::setupRandStates<<<numBlocks, blockSize>>>(states, seed);
    }

    void cudaSetupAPoints(int numBlocks, int blockSize, APoint* gpuPoints, int pointsNum){
        cuda_tempcc::setupAPoints<<<numBlocks, blockSize>>>(gpuPoints, pointsNum);
    }

    void cudaAssignGroundTruthData(int numBlocks, int blockSize, APoint* convertedCamPoints, float4* groundTruthPoints, int groundTruthPointNum){
        cuda_tempcc::assignGroundTruthData<<<numBlocks, blockSize>>>(convertedCamPoints, groundTruthPoints, groundTruthPointNum);
    }

    void cudaCopyImageToPC(dim3 numBlocks, dim3 blockSize, APoint* target, int camId, int pcWidth, int pcHeight, float4* pcPositions, uchar4* pcColors, bool usePDFlow, int flowCols, int flowRows, float4* flow, int everyNPoint){
        cuda_tempcc::copyImageToPC<<<numBlocks, blockSize>>>(target, camId, pcWidth, pcHeight, pcPositions, pcColors, usePDFlow, flowCols, flowRows, flow, everyNPoint);
    }

    void cudaWidenNearestKernel(dim3 numBlocks, dim3 blockSize, float* zResult, float4* gpuPositions, int pcWidth, int pcHeight, int kernelRadius){
        cuda_tempcc::widenNearestKernel<<<numBlocks, blockSize>>>(zResult, gpuPositions, pcWidth, pcHeight, kernelRadius);
    }

    void cudaBorderDistanceKernel(dim3 numBlocks, dim3 blockSize, float* totalResult, float4* gpuPositions, int pcWidth, int pcHeight, int kernelRadius){
        cuda_tempcc::borderDistanceKernel<<<numBlocks, blockSize>>>(totalResult, gpuPositions, pcWidth, pcHeight, kernelRadius);
    }

    void cudaCopyIntoTraningBuffer(int TrainingPointNum, int numBlocks, int blockSize, float4* pointNNTemporalInput, float3* pointNNSpatialInput, float4* pointNNOutput, float4* trainingNNTemporalInput, float3* trainingNNSpatialInput, float4* trainingNNOutput, int* traningShuffledIndices, curandState* randStates,int startOffset, int stride, int validPointsNum){
        cuda_tempcc::copyIntoTraningBuffer<<<numBlocks, blockSize>>>(TrainingPointNum, pointNNTemporalInput, pointNNSpatialInput, pointNNOutput, trainingNNTemporalInput, trainingNNSpatialInput, trainingNNOutput, traningShuffledIndices, randStates, startOffset, stride, validPointsNum);
    }

    void cudaMainPassKernelPartA(int numBlocks, int blockSize, APoint* gpuPoints, float4* gpuTemporalFlows, APoint* camOrderedPoints, float4* pointNNTemporalInput, float3* pointNNSpatialInput, float4* pointNNOutput, int* shuffledIndices, float4* groundTruthPoints, int groundTruthSampleCount, int validPointsNum, curandState* randStates, float scheduledSampling){
        cuda_tempcc::mainPassKernelPartA<<<numBlocks, blockSize>>>(gpuPoints, gpuTemporalFlows, camOrderedPoints, pointNNTemporalInput, pointNNSpatialInput, pointNNOutput, shuffledIndices, groundTruthPoints, groundTruthSampleCount, validPointsNum, randStates, scheduledSampling);
    }

    void cudaMainPassKernelPartB(int numBlocks, int blockSize, APoint* gpuPoints, float4* gpuTemporalFlows, APoint* camOrderedPoints, float4* groundTruthPoints, float4* pointNNOutput, int* shuffledIndices, int validPointsNum, curandState* randStates, bool isTraining, float allowedTrainingDivergence, bool applyFlow){
        cuda_tempcc::mainPassKernelPartB<<<numBlocks, blockSize>>>(gpuPoints, gpuTemporalFlows, camOrderedPoints, groundTruthPoints, pointNNOutput, shuffledIndices, validPointsNum, randStates, isTraining, allowedTrainingDivergence, applyFlow);
    }

    void cudaCopyCamPointsWOStaticIdx(int numBlocks, int blockSize, APoint* gpuPoints, APoint* convertedCamPoints, int camPointNum){
        cuda_tempcc::copyCamPointsWOStaticIdx<<<numBlocks, blockSize>>>(gpuPoints, convertedCamPoints, camPointNum);
    }

    void cudaDensityFiltering(APoint* gpuPoints, int pointsNum, int* occupied, int occupiedSize){
        int blockSize = 256;
        int numBlocks = (pointsNum + blockSize - 1) / blockSize;
        cuda_hashset::initialize(occupied, occupiedSize);
        cuda_tempcc::densityFiltering<<<numBlocks, blockSize>>>(gpuPoints, pointsNum, occupied, occupiedSize);
    }

    __global__ void densityFiltering(APoint* gpuPoints, int pointsNum, int* occupied, int occupiedSize){
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        //int density = 250;
        //int sideHalfLength = 100;
        //int sideLength = 200;

        if (index < pointsNum){

            int densityPerSide = 500;
            float sideLengthX = 2.0;
            float sideLengthY = 2.0;
            float sideLengthZ = 2.0;

            float4 pos = gpuPoints[index].position;

            int xId = max(min(int(((pos.x + sideLengthX / 2) / sideLengthX) * densityPerSide), densityPerSide - 1), 0);
            int yId = max(min(int((pos.y / sideLengthY) * densityPerSide), densityPerSide - 1), 0);
            int zId = max(min(int(((pos.z + sideLengthZ / 2) / sideLengthZ) * densityPerSide), densityPerSide - 1), 0);

            int voxelIdx = xId + densityPerSide * (yId + densityPerSide * zId);

            if(!cuda_hashset::insert(occupied, occupiedSize, voxelIdx)){
                gpuPoints[index].groundTruthIdx = -1;
                gpuPoints[index].state = 0;
            }
        }
    }

    __global__ void clearAllPoints(APoint *data, int pointsNum){
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < pointsNum){
            data[index].groundTruthIdx = -1;
            data[index].state = 0;
        }
    }

    __global__ void copyCamPointsWOStaticIdx(APoint* gpuPoints, APoint* convertedCamPoints, int pointsNum){
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < pointsNum){
            gpuPoints[index].camId = convertedCamPoints[index].camId;
            gpuPoints[index].color = convertedCamPoints[index].color;
            gpuPoints[index].flow = convertedCamPoints[index].flow;
            gpuPoints[index].newFlowIdx = convertedCamPoints[index].newFlowIdx;
            gpuPoints[index].numberOfFlows = convertedCamPoints[index].numberOfFlows;
            gpuPoints[index].groundTruthIdx = convertedCamPoints[index].groundTruthIdx;
            gpuPoints[index].lifetime = convertedCamPoints[index].lifetime;
            gpuPoints[index].position = convertedCamPoints[index].position;
            gpuPoints[index].state = convertedCamPoints[index].state;
        }
    }

    __global__ void setupAPoints(APoint* gpuPoints, int pointsNum){
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < pointsNum){
            gpuPoints[index].staticIdx = index;
        }
    }

    __global__ void assignGroundTruthData(APoint* convertedCamPoints, float4* groundTruthPoints, int groundTruthPointNum){
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < groundTruthPointNum){
            float4 p = groundTruthPoints[index];

            for(int camID = 0; camID < ACTIVE_CAMS; ++camID){
                int2 imageCoords = projectPointToCam(p, camID);

                if(imageCoords.x >= 0 && imageCoords.y >= 0 && imageCoords.x < 640 && imageCoords.y < 576){
                    int idx = imageCoords.x + imageCoords.y * 640 + 368640 * camID;

                    if(Vec4Length(Vec4Subtract(convertedCamPoints[idx].position, p)) < 0.02){
                        convertedCamPoints[idx].groundTruthIdx = index;
                    }
                }
            }
        }
    }

    __global__ void copyImageToPC(APoint* target, int camId, int pcWidth, int pcHeight, float4* pcPositions, uchar4* pcColors, bool isPDFlow, int flowCols, int flowRows, float4* flow, int everyNPoint) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < pcWidth && y < pcHeight){
            int h = x + y * pcWidth;
            int destIdx = h + camId * pcWidth * pcHeight;

            float4 camPos = pcPositions[h];

            target[destIdx].state = camPos.z > 0.01f && !isinf(camPos.z);

            // Mark for removal AFTER neural nets were infered.
            if(x % everyNPoint != 0 || y % everyNPoint != 0)
                target[destIdx].state = 10;

            target[destIdx].numberOfFlows = 0;
            target[destIdx].newFlowIdx = 0;
            target[destIdx].groundTruthIdx = -1;

            if(target[destIdx].state > 0){
                float4 p = MatMultVec4(camMatrices[camId], pcPositions[h]);
                target[destIdx].position = p;
                target[destIdx].color = pcColors[h];
                target[destIdx].camId = camId;

                if(isPDFlow){
                    int factor = 576 / flowRows;
                    int sX = (h%pcWidth) / factor;
                    int sY = (h/pcWidth) / factor;
                    int sID = sY * flowCols + sX;

                    float4 dir = MatMultVec4(camMatrices[camId], float4{flow[sID].x, flow[sID].y, flow[sID].z, 0});
                    target[destIdx].flow = dir;
                } else {
                    target[destIdx].flow = flow[h];
                }
            } else if(isinf(camPos.z)){
                // If camPos.z is -inf or inf, we require this info
                // in the main pass kernel:
                target[destIdx].position = pcPositions[h];
            }
        }
    }

    __global__ void widenNearestKernel(float* zResult, float4* gpuPositions, int pcWidth, int pcHeight, int kernelRadius){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < pcWidth && y < pcHeight) {
            float result = 99999.f;

            int yMin = max(y-kernelRadius, 0);
            int yMax = min(y+kernelRadius, pcHeight - 1);

            int xMin = max(x-kernelRadius, 0);
            int xMax = min(x+kernelRadius, pcWidth - 1);

            for(int cY = yMin; cY <= yMax; ++cY){
                for(int cX = xMin; cX <= xMax; ++cX){
                    float z = gpuPositions[cX + cY * pcWidth].z;

                    if(!isnan(z) && z < result){
                        result = z;
                    }
                }
            }

            zResult[x + y * pcWidth] = result;
        }
    }

    __global__ void borderDistanceKernel(float* totalResult, float4* gpuPositions, int pcWidth, int pcHeight, int kernelRadius){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < pcWidth && y < pcHeight) {
            float result = 0;

            int yMin = max(y-kernelRadius, 0);
            int yMax = min(y+kernelRadius, pcHeight - 1);

            int xMin = max(x-kernelRadius, 0);
            int xMax = min(x+kernelRadius, pcWidth - 1);

            float midZ = gpuPositions[x + y * pcWidth].z;

            for(int cY = yMin; cY <= yMax; ++cY){
                for(int cX = xMin; cX <= xMax; ++cX){
                    float z = gpuPositions[cX + cY * pcWidth].z;

                    if(isnan(z) || abs(z - midZ) > 0.1){
                        ++result;
                    }
                }
            }

            totalResult[x + y * pcWidth] = result;
        }
    }

    __device__ inline float2 lerp(const float2& a, const float2& b, float t) {
        return {
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t
        };
    }

    __device__ inline int2 projectPointToCam(float4 point, int camId){
        CamInfo& camInfo = camInfos[camId];

        // Transform point data into camera space:
        float4 posCS = MatMultVec4(invCamMatrices[camId], point);

        float u = ((posCS.x / posCS.z) * 0.5f + 0.5f) * camInfo.lookup3DToImageSize - 0.5f;
        float v = ((posCS.y / posCS.z) * 0.5f + 0.5f) * camInfo.lookup3DToImageSize - 0.5f;
        int u_base = int(u);
        int v_base = int(v);
        float u_offset = u - u_base;
        float v_offset = v - v_base;

        if(u >= 0  && v >= 0  && u + 1 < camInfo.lookup3DToImageSize && v + 1 < camInfo.lookup3DToImageSize){

            float2 topLeft = camInfo.lookup3DToImage[u_base + v_base * camInfo.lookup3DToImageSize];
            float2 topRight = camInfo.lookup3DToImage[(u_base + 1) + v_base * camInfo.lookup3DToImageSize];
            float2 bottomLeft = camInfo.lookup3DToImage[u_base + (v_base + 1) * camInfo.lookup3DToImageSize];
            float2 bottomRight = camInfo.lookup3DToImage[(u_base + 1) + (v_base + 1) * camInfo.lookup3DToImageSize];

            float2 interpTop = lerp(topLeft, topRight, u_offset);
            float2 interpBottom = lerp(bottomLeft, bottomRight, u_offset);

            float2 uv = lerp(interpTop, interpBottom, v_offset);

            int2 pixelCoord = {
                int(uv.x * camInfo.width),
                int(uv.y * camInfo.height)
            };

            if(pixelCoord.x >= 0 && pixelCoord.x < int(camInfo.width) && pixelCoord.y >= 0 && pixelCoord.y < int(camInfo.height)){
                return pixelCoord;
            }
        }

        return {-1,-1};
    }

    /**
     * Processes the given (predicted) point of the last frames with the
     * point cloud data of the current frame for a single camera. The given
     * point is projected into the camera's image and compared with the
     * position of that point there.
     *
     * Depending on it, the point is either merged with the cam point,
     * deleted or unaltered (by marking it for deletion).
     *
     * This method is used by mainPassKernelPartB!
     */
    __device__ inline void processWithCamPoints(APoint& point, APoint* camOrderedPoints, float4* groundTruthPoints, int pointId, int camId, curandState* randStates, bool isTraining, float allowedTrainingDivergence){
        if(point.state == 0)
            return;

        CamInfo& camInfo = camInfos[camId];

        // Transform point data into camera space:
        float4 posCS = MatMultVec4(invCamMatrices[camId], point.position);
        float4 flowCS = MatMultVec4(invCamMatrices[camId], point.flow);

        // Get lookupUV:
        float2 lookupUV = {
            ((posCS.x / posCS.z) * 0.5f + 0.5f) * camInfo.lookup3DToImageSize,
            ((posCS.y / posCS.z) * 0.5f + 0.5f) * camInfo.lookup3DToImageSize
        };

        float2 fracPart = {
            lookupUV.x - floor(lookupUV.x),
            lookupUV.y - floor(lookupUV.y)
        };

        // Check if point is in lookup table:
        if(lookupUV.x >= 0 && lookupUV.x < int(camInfo.lookup3DToImageSize) && lookupUV.y >= 0 && lookupUV.y < int(camInfo.lookup3DToImageSize))
        {
            float2 uvAA = camInfo.lookup3DToImage[int(floor(lookupUV.x)) + int(floor(lookupUV.y)) * camInfo.lookup3DToImageSize];
            float2 uvAB = camInfo.lookup3DToImage[int(floor(lookupUV.x)) + int(ceil(lookupUV.y)) * camInfo.lookup3DToImageSize];
            float2 uvBA = camInfo.lookup3DToImage[int(ceil(lookupUV.x)) + int(floor(lookupUV.y)) * camInfo.lookup3DToImageSize];
            float2 uvBB = camInfo.lookup3DToImage[int(ceil(lookupUV.x)) + int(ceil(lookupUV.y)) * camInfo.lookup3DToImageSize];

            float2 interpX1 = {
                uvAA.x + fracPart.x * (uvBA.x - uvAA.x),
                uvAA.y + fracPart.x * (uvBA.y - uvAA.y)
            };
            float2 interpX2 = {
                uvAB.x + fracPart.x * (uvBB.x - uvAB.x),
                uvAB.y + fracPart.x * (uvBB.y - uvAB.y)
            };
            float2 uv = {
                interpX1.x + fracPart.y * (interpX2.x - interpX1.x),
                interpX1.y + fracPart.y * (interpX2.y - interpX1.y)
            };

            int2 pixelCoord = {
                int(uv.x * camInfo.width),
                int(uv.y * camInfo.height)
            };

            // Check if pixel coord is in image borders:
            if(pixelCoord.x >= 0 && pixelCoord.x < int(camInfo.width) && pixelCoord.y >= 0 && pixelCoord.y < int(camInfo.height)){
                int camPixelIdx = pixelCoord.x + pixelCoord.y * camInfo.width;

                // Get camera point (where 'point' was projected on):
                APoint& camPoint = camOrderedPoints[camPixelIdx];

                float4 camPointCS = camInfos[camId].gpuPositions[camPixelIdx];
                float nearestWidenedZ = camInfos[camId].nearestWidenedZ[camPixelIdx];
                float borderDistance = camInfos[camId].borderDistance[camPixelIdx];

                if(camPoint.state != 10 && abs(camPointCS.z - posCS.z) < 0.03f && borderDistance < 5){
                    point.flow = camPoint.flow;
                    point.position = camPoint.position;
                    point.color = camPoint.color;
                    point.lifetime = 0;
                    point.state = 1;

                    if(camPoint.groundTruthIdx != -1)
                        point.groundTruthIdx = camPoint.groundTruthIdx;
                    //camPoint.state = 0;
                } else {
                    // If camPoint is behind current point, remove the point (because it cannot exists):
                    if(nearestWidenedZ > posCS.z){
                        point.groundTruthIdx = -1;
                        point.state = 0;
                    } else {
                        point.state = 2;
                    }
                }
            } else {
                point.groundTruthIdx = -1;
                point.state = 0;
            }
        } else {
            // If point exceeds the area of a single camera, we can't track it anymore:
            point.groundTruthIdx = -1;
            point.state = 0;
        }

        if(abs(point.position.x) > 4 || abs(point.position.y) > 4 || abs(point.position.z) > 4){
            point.groundTruthIdx = -1;
            point.state = 0;
        }


        if(isTraining){
            // Exactly positioning points to GT position for precise sampling:
            if(point.groundTruthIdx >= 0){
                float4 gtPos = groundTruthPoints[point.groundTruthIdx];

                float length = Vec4Length(Vec4Subtract(gtPos, point.position));
                float randNum = curand_normal(&randStates[pointId]) * allowedTrainingDivergence;
                if(randNum < length)
                    point.position = gtPos;
            }

            // For faster training, remove all points that got not assigned to ground truth points:
            if(point.groundTruthIdx == -1)
                point.state = 0;
        }

        //float randNum = curand_uniform(&randStates[pointId]);

        //if(randNum > 0.95)
         //   point.state = 0;

        // TODO: Ich sollte noch einmal genau durchgehen, wie mit der Löschung der Punkte vorgegangen
        // werden soll. Ein Punkt sollte nur gelöscht werden, wenn er aus dem Frustum aller drei Kameras
        // fliegt, weil der betrachtete Bereich im Allgemeinen nicht von allen Kameras, sondern nur eine
        // Teilmenge erfast wird (die letzten beiden point.state = 0 sollten da berücksichtigt werden.
    }

    __device__
    inline float4 clampXYZ(float4& v, float minV, float maxV){
        v.x = max(min(v.x, maxV), minV);
        v.y = max(min(v.y, maxV), minV);
        v.z = max(min(v.z, maxV), minV);
    }

    __global__ void copyIntoTraningBuffer(int TrainingPointNum, float4* pointNNTemporalInput, float3* pointNNSpatialInput, float4* pointNNOutput, float4* trainingNNTemporalInput, float3* trainingNNSpatialInput, float4* trainingNNOutput, int* traningShuffledIndices, curandState* randStates, int startOffset, int stride, int validPointsNum){
        int pointID = blockIdx.x * blockDim.x + threadIdx.x  * stride;

        for(int camID = 0; camID < ACTIVE_CAMS; ++camID){
            float randNum = curand_uniform(&randStates[blockIdx.x * blockDim.x + threadIdx.x]);
            int startTempNum = int(NN_TEMPORAL_INPUT_SIZE * randNum + 0.5);

            if(pointID < validPointsNum){
                int originID = pointID;
                int targetID = traningShuffledIndices[(startOffset + pointID / stride) % 500000 /*TraningPointNum*/];

                for(int i=0; i < NN_TEMPORAL_INPUT_SIZE; ++i){
                    if(i < startTempNum)
                        trainingNNTemporalInput[targetID *  NN_TEMPORAL_INPUT_SIZE + TrainingPointNum * NN_TEMPORAL_INPUT_SIZE * camID + i] = float4{0,0,0,1};
                    else
                        trainingNNTemporalInput[targetID *  NN_TEMPORAL_INPUT_SIZE + TrainingPointNum * NN_TEMPORAL_INPUT_SIZE * camID + i] = pointNNTemporalInput[originID * NN_TEMPORAL_INPUT_SIZE + validPointsNum * NN_TEMPORAL_INPUT_SIZE * camID + i];
                }

                for(int i=0; i < NN_SPATIAL_INPUT_SIZE; ++i){
                    trainingNNSpatialInput[targetID *  NN_SPATIAL_INPUT_SIZE + TrainingPointNum * NN_SPATIAL_INPUT_SIZE * camID + i] = pointNNSpatialInput[originID * NN_SPATIAL_INPUT_SIZE + validPointsNum * NN_SPATIAL_INPUT_SIZE * camID + i];
                }

                trainingNNOutput[targetID + TrainingPointNum * camID] = pointNNOutput[originID + validPointsNum * camID];
            }
        }
    }

    /**
     * The function `mainPassKernelPartA` prepares the buffers for LibTorch inference
     * and training. E.g. the random sampling of surrounding points is performed.
     *
     * And, if available, ground truth data is copied into the output array in case
     * of training (so no extra buffers has to be used).
     */
    __global__ void mainPassKernelPartA(APoint* gpuPoints, float4* temporalFlows, APoint* camOrderedPoints, float4* pointNNTemporalInput, float3* pointNNSpatialInput, float4* pointNNOutput, int* shuffledIndices, float4* groundTruthPoints, int groundTruthSampleCount, int validPointsNum, curandState* randStates, float scheduledSampling){
        int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        if(threadId < validPointsNum){
            APoint& point = gpuPoints[threadId];

            for(int camID = 0; camID < ACTIVE_CAMS; ++camID){
                int2 prj = projectPointToCam(point.position, camID);

                const int spacing = 5;
                int ooIndex = threadId + validPointsNum * camID;

                int spatialStartId = ooIndex * NN_SPATIAL_INPUT_SIZE;
                int temporalStartId = ooIndex * NN_TEMPORAL_INPUT_SIZE;

                float meanDist = 0;
                float meanDistCount = 0;

                // Sampling spatial data:
                for(int y=-3; y <= 3; ++y){
                    for(int x=-3; x <= 3; ++x){
                        int sID = (3+y) * 7 + (3+x);


                        float2 offset = curand_normal2(&randStates[threadId]);

                        int pX = max(0, min(639, prj.x + int(offset.x * 14)));
                        int pY = max(0, min(575, prj.y + int(offset.y * 14)));

                        float3& samplePos = pointNNSpatialInput[spatialStartId + sID * 2];
                        float3& sampleFlow = pointNNSpatialInput[spatialStartId + sID * 2 + 1];

                        APoint& camP = camOrderedPoints[pY * 640 + pX + 368640 * camID];
                        float4 relCamPosWS = {camP.position.x - point.position.x, camP.position.y - point.position.y, camP.position.z - point.position.z, 0};

                        float4 relCamFlowWS = {camP.flow.x * 100, camP.flow.y * 100, camP.flow.z * 100, 0};

                        float4 relCamPosCS = MatMultVec4(invCamMatrices[camID], relCamPosWS);
                        float4 relCamFlowCS = MatMultVec4(invCamMatrices[camID], relCamFlowWS);

                        if(isnan(relCamPosCS.x) || isnan(relCamPosCS.y) || isnan(relCamPosCS.z) || isnan(relCamPosCS.w) ||
                            isnan(relCamFlowCS.x) || isnan(relCamFlowCS.y) || isnan(relCamFlowCS.z) || isnan(relCamFlowCS.w) ||
                            sqrt(relCamPosCS.x*relCamPosCS.x + relCamPosCS.y*relCamPosCS.y + relCamPosCS.z*relCamPosCS.z) > 1){
                            samplePos = {1,1,1};
                            sampleFlow = {0,0,0};
                        } else {
                            clampXYZ(relCamPosCS);
                            clampXYZ(relCamFlowCS);
                            samplePos = {1-relCamPosCS.x, 1-relCamPosCS.y, 1-relCamPosCS.z};
                            sampleFlow = {relCamFlowCS.x, relCamFlowCS.y, relCamFlowCS.z};
                            meanDist += Vec4Length(relCamPosWS);
                            ++meanDistCount;
                        }
                    }
                }

                // Set temporal data to zero:
                for(int i=0; i < NN_TEMPORAL_INPUT_SIZE; ++i){
                    pointNNTemporalInput[temporalStartId + i] = {0, 0, 0, 1};
                }

                float randNum = curand_uniform(&randStates[threadId]);

                // Copy temporal data from ring buffer:
                {
                    for(int i=0; i < point.numberOfFlows; ++i){
                        int srcIdx = (point.newFlowIdx - 1 - i + NN_TEMPORAL_INPUT_SIZE + NN_TEMPORAL_INPUT_SIZE) % NN_TEMPORAL_INPUT_SIZE;
                        int destIdx = NN_TEMPORAL_INPUT_SIZE - i - 1;

                        float4& prevFlowWS = temporalFlows[point.staticIdx * NN_TEMPORAL_INPUT_SIZE + srcIdx];
                        float4 prevFlowCS = MatMultVec4(invCamMatrices[camID], {prevFlowWS.x, prevFlowWS.y, prevFlowWS.z, 0.f});

                        float randNum = curand_uniform(&randStates[threadId]);

                        if(point.groundTruthIdx < 0 || scheduledSampling > 0.99f || scheduledSampling > randNum){
                            pointNNTemporalInput[temporalStartId + destIdx] = {prevFlowCS.x * 100, prevFlowCS.y * 100, prevFlowCS.z * 100, prevFlowWS.w};
                        } else {
                            int gtIdx = point.groundTruthIdx;
                            float4 gtFlowWS = Vec4Subtract((groundTruthPoints - groundTruthSampleCount * i)[gtIdx], (groundTruthPoints - groundTruthSampleCount * (i+1))[gtIdx]);
                            float4 gtFlowCS = MatMultVec4(invCamMatrices[camID], {gtFlowWS.x, gtFlowWS.y, gtFlowWS.z, 0.f});

                            float len = sqrt(prevFlowCS.x*prevFlowCS.x + prevFlowCS.y*prevFlowCS.y + prevFlowCS.z*prevFlowCS.z) * 10;

                            float noiseDX = curand_normal(&randStates[threadId]) * len;
                            float noiseDY = curand_normal(&randStates[threadId+1]) * len;
                            float noiseDZ = curand_normal(&randStates[threadId+2]) * len;

                            pointNNTemporalInput[temporalStartId + destIdx] = {gtFlowCS.x * 100 + noiseDX, gtFlowCS.y * 100 + noiseDY, gtFlowCS.z * 100 + noiseDZ,  prevFlowWS.w};
                        }

                        float4& p = pointNNTemporalInput[temporalStartId + destIdx];
                        float l = p.x * p.x + p.y * p.y + p.z * p.z;

                        if(isnan(pointNNTemporalInput[temporalStartId + destIdx].x)
                            || isnan(pointNNTemporalInput[temporalStartId + destIdx].y)
                            || isnan(pointNNTemporalInput[temporalStartId + destIdx].z)
                            || l > 15)
                            p = float4{0,0,0,1};
                    }
                }

                // Insert Ground Truth Values for training
                if(point.groundTruthIdx >= 0){
                    int gtIdx = point.groundTruthIdx;
                    float4 nextGtPoint = groundTruthPoints[gtIdx];
                    float4 gtPoint = groundTruthPoints[gtIdx - groundTruthSampleCount];

                    float4 toNextGTPosWS = Vec4Subtract(nextGtPoint, gtPoint);
                    float4 toNextGTPosCS = MatMultVec4(invCamMatrices[camID], {toNextGTPosWS.x, toNextGTPosWS.y, toNextGTPosWS.z, 0});
                    float currentError = Vec4Length(Vec4Subtract(gtPoint, point.position));
                    float toNextLength = Vec4Length(toNextGTPosCS);

                    if(isnan(toNextGTPosCS.x) || isnan(toNextGTPosCS.y) || isnan(toNextGTPosCS.z) || isnan(toNextGTPosCS.w) || toNextLength > 0.5){
                        pointNNOutput[ooIndex] = {0.0f, 0.0f, 0.0f, 1.0f};
                    } else {
                        pointNNOutput[ooIndex] = {
                            toNextGTPosCS.x * 100,
                            toNextGTPosCS.y * 100,
                            toNextGTPosCS.z * 100,
                            meanDistCount > 0 ? meanDist / meanDistCount : 0.f
                        };
                    }
                } else {
                    pointNNOutput[ooIndex] = {0, 0, 0, 0};
                }
            }
        }
    }

    __device__ inline void predict(int id, APoint& point, float4* temporalFlows, float4* pointNNOutput, int* shuffledIndices, int validPointsNum, bool applyFlow){
        float3 predictedFlowWS = {0.0f, 0.0f, 0.0f};
        float predictedFlowWeight = 0.f;

        float predictedError = 10000;
        float sumWeight = 0;
        for(int camID = 0; camID < ACTIVE_CAMS; ++camID){
            int ooIndex = id + camID * validPointsNum;
            float4 nnoutput = pointNNOutput[ooIndex];

            float4 flowCS = {nnoutput.x / 100.f, nnoutput.y / 100.f, nnoutput.z / 100.f, 0};
            float4 flowWS = MatMultVec4(camMatrices[camID], flowCS);

            float weight = pow(2.0, -20.0 * nnoutput.w);

            predictedFlowWS.x += flowWS.x * weight;
            predictedFlowWS.y += flowWS.y * weight;
            predictedFlowWS.z += flowWS.z * weight;
            sumWeight += weight;
            predictedError = min(predictedError, nnoutput.w);
        }

        predictedFlowWS.x /= sumWeight;
        predictedFlowWS.y /= sumWeight;
        predictedFlowWS.z /= sumWeight;

        predictedError /= ACTIVE_CAMS;

        if(applyFlow){
            point.position.x += predictedFlowWS.x;
            point.position.y += predictedFlowWS.y;
            point.position.z += predictedFlowWS.z;
            point.flow.x = predictedFlowWS.x;
            point.flow.y = predictedFlowWS.y;
            point.flow.z = predictedFlowWS.z;

            // w Value is correctly set in next mainFramePass (is 1 frame ahead in time):
            temporalFlows[point.staticIdx * NN_TEMPORAL_INPUT_SIZE + point.newFlowIdx] = {predictedFlowWS.x, predictedFlowWS.y, predictedFlowWS.z, predictedError};

            ++point.newFlowIdx;
            if(point.newFlowIdx >= NN_TEMPORAL_INPUT_SIZE)
                point.newFlowIdx = 0;

            ++point.numberOfFlows;
            if(point.numberOfFlows >= NN_TEMPORAL_INPUT_SIZE)
                point.numberOfFlows = NN_TEMPORAL_INPUT_SIZE;
        }

        ++point.lifetime;

    }

    /**
     * Compared to `mainPassKernelPartA`, `mainPassKernelPartB` is executed
     * after the LibTorch training / inference step. It applies the predicted
     * flows and also implements the point removal (here, it is only marked
     * for removal, so that the points can be removed using thrust).
     */
    __global__ void mainPassKernelPartB(APoint* gpuPoints, float4* temporalFlows, APoint* camOrderedPoints, float4* groundTruthPoints, float4* pointNNOutput, int* shuffledIndices, int validPointsNum, curandState* randStates, bool isTraining, float allowedTrainingDivergence, bool applyFlow){
        int threadId = blockIdx.x * blockDim.x + threadIdx.x;

        if(threadId < validPointsNum){
            APoint& point = gpuPoints[threadId];
            // Predict next point position:
            predict(threadId, point, temporalFlows, pointNNOutput, shuffledIndices, validPointsNum, applyFlow);

            // Offset to the first point in the current considered camera:
            int camPointOffset = 0;

            // Iterate over all cameras:
            for(int i=0; i < ACTIVE_CAMS; ++i){
                // Process the point with the current camera:
                processWithCamPoints(point, camOrderedPoints + camPointOffset, groundTruthPoints, threadId, i, randStates, isTraining, allowedTrainingDivergence);

                // Calculate the new offset:
                camPointOffset += camInfos[i].width*camInfos[i].height;
            }

            // Remove points that exceeds a reasonably area (TODO: SHOULD
            // BE REMOVED WHEN THE ALGORITHM IS FINALLY IMPLEMENTED):
            if(abs(point.position.x) > 2.0 || abs(point.position.y - 1) > 2 || abs(point.position.z) > 2.0){
                point.state = 0;
                return;
            }
        }
    }
}
