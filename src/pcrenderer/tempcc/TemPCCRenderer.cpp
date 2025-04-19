#include "src/pch.h"

#include "src/pcrenderer/tempcc/TemPCCRenderer.h"

#include "src/Data.h"

#include <imgui_internal.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include "src/util/math/Mat4.h"

#include <cstdlib>
#include <chrono>

#include <random>

#include <iostream>
#include <fstream>

#include "src/pcrenderer/tempcc/TemPCCRendererKernels.h"
#include "src/pcrenderer/tempcc/TemPCCRendererThrust.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

using namespace std::chrono_literals;

float3 ensureNonNan(float3 val, int type = 0){
    if(isnan(val.x) || isnan(val.y) || isnan(val.z)){
        std::cout << "NAN3 detected!! " << type << std::endl;
        return {0.f, 0.f, 0.f};
    }
    if(isinf(val.x) || isinf(val.y) || isinf(val.z)){
        std::cout << "INF3 detected!! " << type << std::endl;
        return {0.f, 0.f, 0.f};
    }
    return val;
}

float4 ensureNonNan(float4 val, int type = 0){
    if(isnan(val.x) || isnan(val.y) || isnan(val.z) || isnan(val.w)){
        std::cout << "NAN4 detected!! " << type << std::endl;
        return {0.f, 0.f, 0.f, 0.f};
    }
    if(isinf(val.x) || isinf(val.y) || isinf(val.z) || isinf(val.w)){
        std::cout << "INF4 detected!! " << type << std::endl;
        return {0.f, 0.f, 0.f, 0.f};
    }
    return val;
}

/**
 * Attention Renderer Constructor
 */
TemPCCRenderer::TemPCCRenderer(unsigned int flowRows)
    : flowRows(flowRows)
    , flowCols(static_cast<unsigned int>(std::ceil(flowRows*10.0/9.0)))
{
    std::cout << "Constructing the TemPCC Renderer..." << std::endl;

    int MaxPointNum = Data::instance->TemPCC_MaxPointNum;
    points = new APoint[MaxPointNum];

    // Create points on GPU (for the algorithm with fusion and completion):
    cudaMalloc(&gpuPoints, MaxPointNum * sizeof(APoint));
    cudaMalloc(&gpuTemporalFlows, MaxPointNum * NN_TEMPORAL_INPUT_SIZE * sizeof(float4));

    cudaMalloc(&convertedCamPoints, MAX_CONVCAMPOINTS_NUM * sizeof(APoint));

    std::cout << "    Memory for gpuPoints and ConvertedCamPoints created." << std::endl;

    // Setup Random Number states (curand):
    cudaMalloc(&randStates, MaxPointNum * sizeof(curandState));
    int threadsPerBlock = 256;
    int numBlocks = (MaxPointNum + threadsPerBlock - 1) / threadsPerBlock;
    cuda_tempcc::cudaSetupRandStates(numBlocks, threadsPerBlock, randStates, time(0));
    cuda_tempcc::cudaSetupAPoints(numBlocks, threadsPerBlock, gpuPoints, MaxPointNum);

    debugInputSpatialBuffer = new float3[NN_SPATIAL_INPUT_SIZE];
    debugInputTemporalBuffer = new float4[NN_TEMPORAL_INPUT_SIZE];

    std::cout << "    Debug Arrays created" << std::endl;

    // Neural Net samples:
    cudaMalloc(&pointNNTemporalInput, MaxPointNum * NN_TEMPORAL_INPUT_SIZE * sizeof(float4) * ACTIVE_CAMS);
    cudaMalloc(&pointNNSpatialInput, MaxPointNum * NN_SPATIAL_INPUT_SIZE * sizeof(float3) * ACTIVE_CAMS);
    cudaMalloc(&pointNNOutput, MaxPointNum * sizeof(float4) * ACTIVE_CAMS);

    int TrainingPointNum = Data::instance->TemPCC_TrainingPointNum;
    cudaMalloc(&trainingNNTemporalInput, TrainingPointNum * NN_TEMPORAL_INPUT_SIZE * ACTIVE_CAMS * sizeof(float4));
    cudaMalloc(&trainingNNSpatialInput, TrainingPointNum * NN_SPATIAL_INPUT_SIZE * ACTIVE_CAMS * sizeof(float3));
    cudaMalloc(&trainingNNGroundTruth, TrainingPointNum * ACTIVE_CAMS * sizeof(float4));
    cudaMalloc(&trainingShuffledIndices, TrainingPointNum * sizeof(int));

    cudaMemset(trainingNNTemporalInput, 0, TrainingPointNum * ACTIVE_CAMS * NN_TEMPORAL_INPUT_SIZE * sizeof(float4));
    cudaMemset(trainingNNSpatialInput, 0, TrainingPointNum * ACTIVE_CAMS * NN_SPATIAL_INPUT_SIZE * sizeof(float3));
    cudaMemset(trainingNNGroundTruth, 0, TrainingPointNum * ACTIVE_CAMS * sizeof(float4));

    std::vector<int> indices(TrainingPointNum);
    // Shuffle Indices:
    {
        for(int i=0;i < TrainingPointNum; ++i){
            indices[i] = i;
        }
        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(indices.begin(), indices.begin() + TrainingPointNum, g);

        cudaMemcpy(trainingShuffledIndices, &indices[0], TrainingPointNum * sizeof(int), cudaMemcpyHostToDevice);
    }

    std::cout << "    Neural Net Buffers created." << std::endl;

    for(int i=0; i < ACTIVE_CAMS; ++i){
        cudaMalloc(&camOrderedWidenedNearestZ[i], sizeof(float) * 640 * 576);
        cudaMalloc(&borderDistanceImage[i], sizeof(float) * 640 * 576);

        cudaMalloc(&projectedGTPositions[i], sizeof(float3) * 640 * 576);
        cudaMalloc(&projectedGTFlows[i], sizeof(float3) * 640 * 576);
        cudaMalloc(&projectedGTFlowsDense[i], sizeof(float3) * 640 * 576);
    }

    // Generate buffers for locations, colors:
    glGenBuffers(1, &vbo);
    glGenVertexArrays(1, &vao);

    // Dummy VAO erstellen:
    glGenVertexArrays(1, &dummyVAO);

    nativeModule.to(torch::kCUDA);

    std::cout << "Parameter Num of used Neural Net: " << nativeModule.getParamNum() << std::endl;
}


/*
 * Attention renderer destructor.
 */
TemPCCRenderer::~TemPCCRenderer(){
    if(flowOutputStream.is_open())
        flowOutputStream.close();

    delete[] points;
    cudaFree(gpuPoints);
    cudaFree(gpuTemporalFlows);
    cudaFree(convertedCamPoints);
    cudaFree(pointNNTemporalInput);
    cudaFree(pointNNSpatialInput);
    cudaFree(pointNNOutput);
    cudaFree(randStates);

    delete[] debugInputSpatialBuffer;
    delete[] debugInputTemporalBuffer;

    for(int i=0; i < ACTIVE_CAMS; ++i){
        cudaFree(camOrderedWidenedNearestZ[i]);
        cudaFree(borderDistanceImage[i]);
    }

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

// Funktion zum Überprüfen auf NaN-Werte in einem Tensor
bool contains_nan(const at::Tensor& tensor) {
    auto nan_tensor = torch::isnan(tensor).any().item<bool>();
    return nan_tensor;
}

/*
 * Generates / updates the flow for all point cloud images.
 */
void TemPCCRenderer::generateFlow(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds){
    unsigned int pointCloudNum = pointClouds.size();

    // Initialize CUDA:
    for(unsigned int i=0; i < pointCloudNum && i < ACTIVE_CAMS; ++i){
        if(pdFlows.size() <= i){
            pdFlows.push_back(std::make_shared<PDFlow>(flowCols, flowRows));
            pdFlows[i]->initializeCUDA();
        }
    }

    // Apply Flow:
    for(unsigned int i=0; i < pointCloudNum && i < ACTIVE_CAMS; ++i){
        pdFlows[i]->integratePointCloud(pointClouds[i]);
    }
}


void TemPCCRenderer::inferencePass(int validPointsNum){
    nativeModule.eval();
    torch::NoGradGuard no_grad;

    int pointsPerBatch = Data::instance->TemPCC_InferenceBatchSize;
    int splitNum = (validPointsNum * ACTIVE_CAMS) / pointsPerBatch + 1;

    splitNum = std::min(splitNum, (Data::instance->TemPCC_MaxPointNum * ACTIVE_CAMS) / pointsPerBatch);

    std::cout << pointsPerBatch << " x " << splitNum << std::endl;

    for(int i=0; i < splitNum; ++i){
        torch::Tensor temporalInput = torch::from_blob(pointNNTemporalInput + (pointsPerBatch * i * NN_TEMPORAL_INPUT_SIZE), {pointsPerBatch, NN_TEMPORAL_INPUT_SIZE, 4}, noop_deleter, torch::TensorOptions().device(torch::kCUDA));
        torch::Tensor spatialInput = torch::from_blob(pointNNSpatialInput + (pointsPerBatch * i * NN_SPATIAL_INPUT_SIZE), {pointsPerBatch, NN_SPATIAL_INPUT_SIZE * 3}, noop_deleter, torch::TensorOptions().device(torch::kCUDA));

        // Ausgabe vorhersagen
        try {
            at::Tensor output = nativeModule.forward(temporalInput, spatialInput);
            cudaMemcpy(pointNNOutput + (pointsPerBatch * i), output.data_ptr<float>(), pointsPerBatch * sizeof(float4), cudaMemcpyDeviceToDevice);
        } catch (const std::exception& e) {
            std::cerr << "Exception caught: " << e.what() << std::endl;
            // Optionally handle the error, e.g., by returning an error code or handling the exception further
            throw; // Re-throwing the exception if you want to propagate the error upwards
        }

    }
}

void TemPCCRenderer::trainingPass(float timestamp){
    nativeModule.train();

    int batchSize = Data::instance->TemPCC_TrainingBatchSize;
    int batches = (Data::instance->TemPCC_TrainingPointNum * ACTIVE_CAMS) / batchSize;

    if(optimizer == nullptr){
        optimizer = std::make_shared<torch::optim::Adam>(nativeModule.parameters(), torch::optim::AdamOptions(learningRate));
    } else {
        for (auto &group : optimizer->param_groups())
        {
            if(group.has_options())
            {
                auto &options = static_cast<torch::optim::OptimizerOptions&>(group.options());
                options.set_lr(learningRate);
            }
        }
    }

    for(int epoch = 0; epoch < 1; ++epoch) {
        for(int i=0; i < batches; ++i){
            torch::Tensor temporalInput = torch::from_blob(trainingNNTemporalInput + (batchSize * i * NN_TEMPORAL_INPUT_SIZE), {batchSize, NN_TEMPORAL_INPUT_SIZE, 4}, noop_deleter, torch::TensorOptions().device(torch::kCUDA));
            torch::Tensor spatialInput = torch::from_blob(trainingNNSpatialInput + (batchSize * i * NN_SPATIAL_INPUT_SIZE), {batchSize, NN_SPATIAL_INPUT_SIZE * 3}, noop_deleter, torch::TensorOptions().device(torch::kCUDA));
            torch::Tensor groundTruth = torch::from_blob(trainingNNGroundTruth + (batchSize * i), {batchSize, 4}, noop_deleter, torch::TensorOptions().device(torch::kCUDA));

            optimizer->zero_grad();

            at::Tensor prediction = nativeModule.forward(temporalInput, spatialInput);

            at::Tensor loss = torch::nn::functional::l1_loss(prediction, groundTruth);
            loss.backward();
            optimizer->step();

            if(i + 1 == batches){
                lastLoss = loss.item<float>();
            }
        }
    }

    std::cerr << "" << std::time(0) << "," << timestamp << "," << validTrainingNum << "," << batches << "," << learningRate << "," << scheduledSampling << "," << lastLoss << std::endl;
    //nativeModule.saveNet();

    if(allowedTrainingDivergence > 0.2f)
        allowedTrainingDivergence = 0.2f;

    learningRate *= 0.99f;
    scheduledSampling += 0.002f;

    if(scheduledSampling > 1.f)
        scheduledSampling = 1.f;

    if(timestamp > 29.66667 && !previousExceededInTrainingEnd){
        clearPointsInNextFrame = true;

        nativeModule.saveNet(std::to_string(std::time(0)) + "_" + std::to_string(learningRate) + ".pt");
        previousExceededInTrainingEnd = timestamp > 29.66667;
    }
}

void TemPCCRenderer::assignGroundTruthData(int currentFrameID){
    int blockSize = 256; // Anzahl der Threads pro Block
    int numBlocks = (Data::instance->groundTruthSampleCount + blockSize - 1) / blockSize; // Berechne die benötigte Anzahl von Blöcken

    cuda_tempcc::cudaAssignGroundTruthData(numBlocks, blockSize, convertedCamPoints, Data::instance->gpuGroundTruthPoints + (currentFrameID * Data::instance->groundTruthSampleCount), Data::instance->groundTruthSampleCount);
}

/*
 * Copies the organized point cloud into an array of APoints.
 */
int TemPCCRenderer::copyImagesToPointArray(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds){
    unsigned int pointCloudNum = pointClouds.size();

    unsigned int result = 0;
    for(unsigned int i=0; i < pointCloudNum && i < ACTIVE_CAMS; ++i){
        std::shared_ptr<OrganizedPointCloud> pc = pointClouds[i];

        // Upload camera infos:
        {
            // Upload camera matrices:
            cuda_tempcc::setConstantCamMatrix(pc->modelMatrix.data, i);
            cuda_tempcc::setConstantInvCamMatrix(pc->modelMatrix.inverse().data, i);

            // Upload camera info:
            CamInfo camInfo;
            camInfo.width = pc->width;
            camInfo.height = pc->height;

            camInfo.nearestWidenedZ = camOrderedWidenedNearestZ[i];
            camInfo.borderDistance = borderDistanceImage[i];
            camInfo.gpuPositions = pc->gpuPositions;

            camInfo.lookupImageTo3D = pc->gpuLookupImageTo3D;
            camInfo.lookup3DToImage = pc->gpuLookup3DToImage;
            camInfo.lookup3DToImageSize = pc->lookup3DToImageSize;

            cuda_tempcc::setConstantCamInfos(camInfo, i);
        }
    }

    for(unsigned int i=0; i < pointCloudNum && i < ACTIVE_CAMS; ++i){
        std::shared_ptr<OrganizedPointCloud> pc = pointClouds[i];

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks(
            int(std::ceil(pc->width / float(threadsPerBlock.x))),
            int(std::ceil(pc->height / float(threadsPerBlock.y)))
        );

        // Copy images:
        cuda_tempcc::cudaCopyImageToPC(numBlocks, threadsPerBlock, convertedCamPoints, i, pc->width, pc->height, pc->gpuPositions, pc->gpuColors, usePDFlow, flowCols, flowRows, usePDFlow ? pdFlows[i]->D : projectedGTFlowsDense[i], everyNCamPoint);

        // Create 'widenNearest' kernel:
        cuda_tempcc::cudaWidenNearestKernel(numBlocks, threadsPerBlock, camOrderedWidenedNearestZ[i], pc->gpuPositions, pc->width, pc->height, 5);
        cuda_tempcc::cudaBorderDistanceKernel(numBlocks, threadsPerBlock, borderDistanceImage[i], pc->gpuPositions, pc->width, pc->height, 5);

        result += pc->width * pc->height;
    }

    cudaDeviceSynchronize();

    return result;
}

/*
 * Integrates the given depth camera images (OrganizedPointClouds) into the point set (which includes hidden points).
 */
void TemPCCRenderer::integratePointClouds(std::vector<std::shared_ptr<OrganizedPointCloud>> pointClouds){
    if(clearPointsInNextFrame){
        //int threadsPerBlock = 256;
        //int numBlocks = (validPointsNum + threadsPerBlock - 1) / threadsPerBlock;
        //cuda_tempcc::cudaClearAllPoints(numBlocks, threadsPerBlock, gpuPoints, MAX_APOINT_NUM);
        validPointsNum = 0;
        validCamPointsNum = 0;
        clearPointsInNextFrame = false;
    }

    //timer.startTimeMeasure("1. Point Clouds to GPU");
    // Ensure the point clouds are updated and available on the gpu:
    for(std::shared_ptr<OrganizedPointCloud> pc : pointClouds){
        pc->toGPU();
    }
    //timer.endTimeMeasure("1. Point Clouds to GPU");

    timer.startTimeMeasure("TemPCCEval");

    timer.startTimeMeasure("Image Flow");
    // Generate the flow of all point clouds:

    if(usePDFlow){
        generateFlow(pointClouds);
    } else {
        for(int i=0; i < ACTIVE_CAMS; ++i){
            cuda_tempcc::cudaClearProjectedGTFlows(projectedGTPositions[i], projectedGTFlows[i], pointClouds[i]->width, pointClouds[i]->height);
            //std::cout << "Ground Truth Sample Count: " << groundTruthSampleCount << std::endl;
            cuda_tempcc::cudaProjectGTFlowIntoImage(i, projectedGTPositions[i], projectedGTFlows[i], pointClouds[i]->gpuPositions, Data::instance->gpuGroundTruthPoints + (currentFrameID * Data::instance->groundTruthSampleCount), Data::instance->groundTruthSampleCount);
            cuda_tempcc::cudaMakeProjectedGTFlowDense(i, projectedGTPositions[i], projectedGTFlows[i], projectedGTFlowsDense[i], pointClouds[i]->gpuPositions, pointClouds[i]->width, pointClouds[i]->height);
        }
    }

    timer.endTimeMeasure("Image Flow");

    // If previous point clouds doesn't exist, simply set it to the current one to avoid memory access error:
    if(lastIntegratedPointCloud.size() != pointClouds.size())
        lastIntegratedPointCloud = pointClouds;

    //timer.startTimeMeasure("3. Copy Images To Array");
    // Copy all points of the given point clouds into the 'convertedCamPoints' array:
    copyImagesToPointArray(lastIntegratedPointCloud);
    //timer.endTimeMeasure("3. Copy Images To Array");

    //timer.startTimeMeasure("4. Assign Ground Truth Data");
    assignGroundTruthData(lastIntegratedPointCloud[0]->frameID);
    cudaDeviceSynchronize();
    //timer.endTimeMeasure("4. Assign Ground Truth Data");

    timer.startTimeMeasure("Thrust GT-Patition");
    // Move GT-assigned points to the front of the list:
    if(shouldTrain){
        APoint* endOfGTReferencedPoints = cuda_tempcc::thrustPartitionByGTIndex(gpuPoints, validPointsNum);
        validTrainingNum = (unsigned int)(endOfGTReferencedPoints - gpuPoints);
    }
    cudaDeviceSynchronize();
    timer.endTimeMeasure("Thrust GT-Patition");

    // Main Pass A:
    timer.startTimeMeasure("Main Pass A");
    {
        int currentFrameID = pointClouds[0]->frameID;

        int threadsPerBlock = 256;
        int numBlocks = (validPointsNum + threadsPerBlock - 1) / threadsPerBlock;

        cuda_tempcc::cudaMainPassKernelPartA(numBlocks, threadsPerBlock, gpuPoints, gpuTemporalFlows, convertedCamPoints, pointNNTemporalInput, pointNNSpatialInput, pointNNOutput, nullptr, Data::instance->gpuGroundTruthPoints + (currentFrameID * Data::instance->groundTruthSampleCount), Data::instance->groundTruthSampleCount, validPointsNum, randStates, scheduledSampling);
        cudaDeviceSynchronize();


        --currentNoTrainingSteps;
        // Trainings pass:
        if(shouldTrain && validTrainingNum > 0 && currentNoTrainingSteps <= 0){
            int threadsPerBlock = 256;
            int numBlocks = (validTrainingNum / trainingStride + threadsPerBlock - 1) / threadsPerBlock;

            cuda_tempcc::cudaCopyIntoTraningBuffer(Data::instance->TemPCC_TrainingPointNum, numBlocks, threadsPerBlock, pointNNTemporalInput, pointNNSpatialInput, pointNNOutput, trainingNNTemporalInput, trainingNNSpatialInput, trainingNNGroundTruth, trainingShuffledIndices, randStates, currentTrainingOffset, trainingStride, validTrainingNum);
            currentTrainingOffset = (currentTrainingOffset + validTrainingNum / trainingStride);

            int TrainingPointNum = Data::instance->TemPCC_TrainingPointNum;
            if(currentTrainingOffset >= TrainingPointNum){
                trainingBufferFull = true;
                currentTrainingOffset = currentTrainingOffset % TrainingPointNum;
            }

            // Perform training pass:
            if(trainingBufferFull)
                trainingPass(pointClouds[0]->timestamp);

            {
                std::random_device rd;  // Zufallsgerät zur Initialisierung
                std::mt19937 gen(rd()); // Mersenne-Twister Zufallsgenerator
                std::uniform_int_distribution<> distrib(-simStepsWithoutTrainingVariance, simStepsWithoutTrainingVariance);
                int random_number = distrib(gen);
                currentNoTrainingSteps = simStepsWithoutTraining + random_number;
            }
        }

        // Debugdraw:
        //if(debugViewOpened){
            // Write out debug training data:
            cudaMemcpy(debugInputSpatialBuffer, pointNNSpatialInput + (selectedDebugPointID * NN_SPATIAL_INPUT_SIZE), NN_SPATIAL_INPUT_SIZE * sizeof(float3), cudaMemcpyDeviceToHost);
            cudaMemcpy(debugInputTemporalBuffer, pointNNTemporalInput + (selectedDebugPointID * NN_TEMPORAL_INPUT_SIZE), NN_TEMPORAL_INPUT_SIZE * sizeof(float4), cudaMemcpyDeviceToHost);
            cudaMemcpy(&debugGroundTruthBuffer, pointNNOutput + selectedDebugPointID, sizeof(float4), cudaMemcpyDeviceToHost);
        //}

        // Inference pass:
        int inferencePointNum = shouldTrain ? validTrainingNum : validPointsNum;
        if(inferencePointNum > 0){
            timer.startTimeMeasure("Inference");
            inferencePass(inferencePointNum);
            cudaDeviceSynchronize();
            timer.endTimeMeasure("Inference");
        }

        // Debugdraw:
        //if(debugViewOpened){
        cudaMemcpy(&debugOutputBuffer1, pointNNOutput + selectedDebugPointID, sizeof(float4), cudaMemcpyDeviceToHost);
        cudaMemcpy(&debugOutputBuffer2, pointNNOutput + selectedDebugPointID + validPointsNum, sizeof(float4), cudaMemcpyDeviceToHost);
        cudaMemcpy(&debugOutputBuffer3, pointNNOutput + selectedDebugPointID + validPointsNum * 2, sizeof(float4), cudaMemcpyDeviceToHost);
        //}
    }
    timer.endTimeMeasure("Main Pass A");

    timer.startTimeMeasure("Main Pass B");
    // Main Pass B
    {
        int currentFrameID = pointClouds[0]->frameID;

        int threadsPerBlock = 256;
        int numBlocks = (validPointsNum + threadsPerBlock - 1) / threadsPerBlock;

        /*
         * TODO: PRÜFEN, OB DAS WIRKLICH GEHT.
         * (auch im Hinblick darauf, dass convertedCamPoints ja entfernt werden sollten).
         */
        copyImagesToPointArray(pointClouds);

        assignGroundTruthData(pointClouds[0]->frameID);

        cuda_tempcc::cudaMainPassKernelPartB(numBlocks, threadsPerBlock, gpuPoints, gpuTemporalFlows, convertedCamPoints, Data::instance->gpuGroundTruthPoints + (currentFrameID * Data::instance->groundTruthSampleCount), pointNNOutput, nullptr, validPointsNum, randStates, shouldTrain, allowedTrainingDivergence, applyFlow);
        cudaDeviceSynchronize();

        //cudaMemcpy(&debugOutputBufferMerged, (void*)(gpuPoints + selectedDebugPointID), sizeof(float4), cudaMemcpyDeviceToHost);
    }
    timer.endTimeMeasure("Main Pass B");

    timer.startTimeMeasure("Clean GPU Points");
    // Remove all gpuPoints those state is 0:
    APoint* lastGpuPointOfRemove = cuda_tempcc::thrustPartitionValid(gpuPoints, validPointsNum);
    timer.endTimeMeasure("Clean GPU Points");

    unsigned int validPointsAfterRemove = (unsigned int)(lastGpuPointOfRemove - gpuPoints);
    //unsigned int validPointsAfterRemove = 0;

    // The new points should be copied to the end:
    APoint* gpuPointsCopyStartPtr = lastGpuPointOfRemove;

    // Remove all convertedCamPoints those state is 0:
    timer.startTimeMeasure("Clean Cam Points");
    APoint* onlyValidCamPoints = cuda_tempcc::thrustRemoveIfInvalid(convertedCamPoints, MAX_CONVCAMPOINTS_NUM);
    timer.endTimeMeasure("Clean Cam Points");

    // Valid cam points:
    unsigned int validCamPoints = (unsigned int)(onlyValidCamPoints - convertedCamPoints);

    std::cout << "Valid Cam Points" << std::endl;

    // If there might be more points to be copied than space is available, we have to move
    // the gpuPointsCopyStartPtr to avoid copying into non existing memory:
    int MaxPointNum = Data::instance->TemPCC_MaxPointNum;
    if(validPointsAfterRemove + validCamPoints > MaxPointNum)
        gpuPointsCopyStartPtr = gpuPoints + MaxPointNum - validCamPoints;

    //cudaMemcpy(gpuPointsCopyStartPtr, convertedCamPoints, validCamPoints * sizeof(APoint), cudaMemcpyDeviceToDevice);
    int threadsPerBlock = 256;
    int numBlocks = (validCamPoints + threadsPerBlock - 1) / threadsPerBlock;
    cuda_tempcc::cudaCopyCamPointsWOStaticIdx(numBlocks, threadsPerBlock, gpuPointsCopyStartPtr, convertedCamPoints, validCamPoints);
    //timer.startTimeMeasure("10. Last Synchronize");
    cudaDeviceSynchronize();
    //timer.endTimeMeasure("10. Last Synchronize");

    // Valid points after copy:
    unsigned int onlyExistingPointNum = (unsigned int)(gpuPointsCopyStartPtr - gpuPoints);
    unsigned int validPointsAfterCopy = (unsigned int)(gpuPointsCopyStartPtr - gpuPoints + validCamPoints);

    timer.startTimeMeasure("Density Filtering");
    cuda_tempcc::cudaDensityFiltering(gpuPoints, onlyExistingPointNum, densityHashSet.occupied, densityHashSet.size);
    APoint* lastGpuPointAfterDensityFilter = cuda_tempcc::thrustPartitionValid(gpuPoints, validPointsAfterCopy);
    validPointsAfterCopy = (unsigned int)(lastGpuPointAfterDensityFilter - gpuPoints);
    timer.endTimeMeasure("Density Filtering");
    timer.endTimeMeasure("TemPCCEval");

    timer.startTimeMeasure("Sort Points");

    if(shouldRender){
        /*if(!shouldTrain && validPointsAfterCopy > 0){
            float4* deviceLastCamPos;
            cudaMalloc(&deviceLastCamPos, sizeof(float4));
            cudaMemcpy(deviceLastCamPos, &lastCamPos, sizeof(float4), cudaMemcpyHostToDevice);

            // Sort gpuPoints:
            cuda_tempcc::thrustSortByCamDistance(gpuPoints, validPointsAfterCopy, deviceLastCamPos);
        }*/
        cudaMemcpy(points, gpuPoints, validPointsAfterCopy * sizeof(APoint), cudaMemcpyDeviceToHost);
    }
    timer.endTimeMeasure("Sort Points");

    if(shouldEvaluate && currentFrameID >= 29){
        if(!evalInitializationDone){
            loadNet();
            clearPointsInNextFrame = true;
            evalInitializationDone = true;
        } else {
            int currentGroundTruthOffset = (currentFrameID * Data::instance->groundTruthSampleCount);

            APoint* endOfGTReferencedPoints = cuda_tempcc::thrustPartitionByGTIndex(gpuPoints, validPointsNum);
            int validEvaluationNum = (unsigned int)(endOfGTReferencedPoints - gpuPoints);

            cudaMemcpy(points, gpuPoints, validPointsAfterCopy * sizeof(APoint), cudaMemcpyDeviceToHost);

            std::unordered_set<int> seenGroundTruthIds;

            for(int i = 0; i < validEvaluationNum; ++i){
                APoint& p = points[i];

                //if(p.state != 2)
                //    continue;

                EvalAPoint pEval;

                pEval.state = p.state;
                pEval.predictedFlowLength = sqrt(p.flow.x * p.flow.x + p.flow.y * p.flow.y + p.flow.z * p.flow.z);
                pEval.groundTruthId = p.groundTruthIdx;
                seenGroundTruthIds.insert(p.groundTruthIdx);

                if(isnan(pEval.predictedFlowLength))
                    continue;

                float4 thisGTPos = Data::instance->cpuGroundTruthPoints[currentGroundTruthOffset + p.groundTruthIdx];
                float4 nextGTPos = Data::instance->cpuGroundTruthPoints[currentGroundTruthOffset + Data::instance->groundTruthSampleCount + p.groundTruthIdx];
                float4 gtFlow = {nextGTPos.x - thisGTPos.x, nextGTPos.y - thisGTPos.y, nextGTPos.z - thisGTPos.z, 0};
                pEval.groundTruthFlowLength = sqrt(gtFlow.x * gtFlow.x + gtFlow.y * gtFlow.y + gtFlow.z * gtFlow.z);

                float4 flowError = {p.flow.x - gtFlow.x, p.flow.y - gtFlow.y, p.flow.z - gtFlow.z};
                pEval.flowError =  sqrt(flowError.x * flowError.x + flowError.y * flowError.y + flowError.z * flowError.z);

                float4 pathDiff = {p.position.x - thisGTPos.x, p.position.y - thisGTPos.y, p.position.z - thisGTPos.z, 0};
                pEval.positionalDivergence = sqrt(pathDiff.x * pathDiff.x + pathDiff.y * pathDiff.y + pathDiff.z * pathDiff.z);

                float pathLength = 0.f;

                for(int l=0; l < p.lifetime; ++l){
                    float4 thisP = Data::instance->cpuGroundTruthPoints[currentGroundTruthOffset - Data::instance->groundTruthSampleCount * (l) + p.groundTruthIdx];
                    float4 nextP = Data::instance->cpuGroundTruthPoints[currentGroundTruthOffset - Data::instance->groundTruthSampleCount * (l-1) + p.groundTruthIdx];
                    float4 flow = {nextP.x - thisP.x, nextP.y - thisP.y, nextP.z - thisP.z, 0};
                    pathLength += sqrt(flow.x * flow.x + flow.y * flow.y + flow.z * flow.z);
                };

                pEval.groundTruthPathLength = pathLength;
                pEval.lifetime = p.lifetime;
                pEval.frameId = currentFrameID;

                evalAPoints.push_back(pEval);
            }

            int numberOfSeenGTSamples = 0;
            for(int i=0; i < Data::instance->groundTruthSampleCount; ++i){
                if(seenGroundTruthIds.count(i)){
                    ++numberOfSeenGTSamples;
                }
            }

            EvalFrame frameEval;
            frameEval.frameId = currentFrameID;
            frameEval.groundTruthPoints = validEvaluationNum;
            frameEval.camPoints = validCamPointsNum;
            frameEval.totalPoints = validPointsNum;
            frameEval.seenGroundTruthSamples = numberOfSeenGTSamples;
            frameEval.totalGroundTruthSamples = Data::instance->groundTruthSampleCount;
            frameEval.timerTotal = timer.getTimeMeasureInMilliSec("TemPCCEval");
            frameEval.timerImageFlow = timer.getTimeMeasureInMilliSec("Image Flow");
            frameEval.timerMainPassA = timer.getTimeMeasureInMilliSec("Main Pass A");
            frameEval.timerInference = timer.getTimeMeasureInMilliSec("Inference");
            frameEval.timerMainPassB = timer.getTimeMeasureInMilliSec("Main Pass B");
            frameEval.timerCleanGPUPoints = timer.getTimeMeasureInMilliSec("Clean GPU Points");
            frameEval.timerCleanCamPoints = timer.getTimeMeasureInMilliSec("Clean Cam Points");
            frameEval.timerDensityFilter = timer.getTimeMeasureInMilliSec("Density Filtering");
            evalFrames.push_back(frameEval);

            std::cout << "EvalPoints: " << evalAPoints.size() << " | evalFrames: " << evalFrames.size() << std::endl;

            if(pointClouds[0]->timestamp > 29.66667 && !previousExceededEnd){
                std::ofstream pointsFile("evalPoints.csv");
                for(EvalAPoint& pEval : evalAPoints){
                    pointsFile << pEval << std::endl;
                }
                pointsFile.close();

                std::ofstream frameFile("evalFrames.csv");
                for(EvalFrame& frame : evalFrames){
                    frameFile << frame << std::endl;
                }
                frameFile.close();
                std::exit(0);
            }
        }
    }

    previousExceededEnd = pointClouds[0]->timestamp > 29.66667;
    validPointsNum = validPointsAfterCopy;
    validCamPointsNum = validCamPoints;

    lastIntegratedPointCloud = pointClouds;
    currentFrameID = pointClouds[0]->frameID;
    ++integratedPointCloudNum;
};

void TemPCCRenderer::drawDebugLine(Mat4f& projection, Mat4f& view, Vec4f start, Vec4f end, Vec4f color){
    debugLineShader.bind();
    debugLineShader.setUniform("projection", projection);
    debugLineShader.setUniform("view", view);
    debugLineShader.setUniform("startPos", start);
    debugLineShader.setUniform("endPos", end);
    debugLineShader.setUniform("color", color);
    glBindVertexArray(dummyVAO);
    glDrawArrays(GL_LINES, 0, 2);
}


void TemPCCRenderer::drawDebugVectors(Mat4f& projection, Mat4f& view){
    APoint point = points[selectedDebugPointID];
    Vec4f pointPos = Vec4f(point.position.x, point.position.y, point.position.z);

    Mat4f camModel = lastIntegratedPointCloud[0]->modelMatrix;

    glLineWidth(3);
    // Temporal Debug Data
    {
        Vec4f start = pointPos;
        Vec4f end = pointPos;
        for(int i=NN_TEMPORAL_INPUT_SIZE-1; i >= 0; --i){
            float4 tempF3 = debugInputTemporalBuffer[i];

            //std::cout << tempF3.w << ", ";

            Vec4f tempVec = camModel * Vec4f(tempF3.x / 100, tempF3.y / 100, tempF3.z / 100, 0);

            end = end - tempVec;

            float alphaColor = (30.f + i) / 60.f;
            drawDebugLine(projection, view, start, end, i % 2 == 0 ? Vec4f(0.0, 0.0, 1.0  * alphaColor) : Vec4f(0.5 * alphaColor, 0.5 * alphaColor, 1.0 * alphaColor, alphaColor));

            start = end;
        }
        //std::cout << std::endl;
    }

    // Spatial Debug Data
    {
        for(int i=0; i < NN_SPATIAL_INPUT_SIZE / 2; ++i){
            float3 spatialPosF3 = debugInputSpatialBuffer[i*2];
            float3 spatialFlowF3 = debugInputSpatialBuffer[i*2+1];

            Vec4f spatialPosWS = camModel * Vec4f(1 - spatialPosF3.x, 1 - spatialPosF3.y, 1 - spatialPosF3.z, 0);
            Vec4f spatialFlowWS = camModel * Vec4f(spatialFlowF3.x / 100, spatialFlowF3.y / 100, spatialFlowF3.z / 100, 0);

            Vec4f start = pointPos + spatialPosWS;
            Vec4f end = pointPos + spatialPosWS + spatialFlowWS * 2;

            drawDebugLine(projection, view, start, end, Vec4f(0.f, 1.f, 0.f));
        }
    }

    // Predicted Flow 1
    {
        Mat4f camModel = lastIntegratedPointCloud[0]->modelMatrix;
        Vec4f predFlow = camModel * Vec4f(debugOutputBuffer1.x / 100, debugOutputBuffer1.y / 100, debugOutputBuffer1.z / 100, 0);
        Vec4f start = pointPos;
        Vec4f end = pointPos + predFlow;

        drawDebugLine(projection, view, start, end, Vec4f(1.f, 0.5f, 0.f));
    }

    // Predicted Flow 2
    if(ACTIVE_CAMS > 1){
        Mat4f camModel = lastIntegratedPointCloud[1]->modelMatrix;
        Vec4f predFlow = camModel * Vec4f(debugOutputBuffer2.x / 100, debugOutputBuffer2.y / 100, debugOutputBuffer2.z / 100, 0);
        Vec4f start = pointPos;
        Vec4f end = pointPos + predFlow;

        drawDebugLine(projection, view, start, end, Vec4f(1.f, 0.5f, 0.5f));
    }

    // Predicted Flow 3
    if(ACTIVE_CAMS > 2){
        Mat4f camModel = lastIntegratedPointCloud[2]->modelMatrix;
        Vec4f predFlow = camModel * Vec4f(debugOutputBuffer3.x / 100, debugOutputBuffer3.y / 100, debugOutputBuffer3.z / 100, 0);
        Vec4f start = pointPos;
        Vec4f end = pointPos + predFlow;

        drawDebugLine(projection, view, start, end, Vec4f(1.f, 0.f, 0.5f));
    }

    // Predicted Flow Merged
    {
        Vec4f predFlow = Vec4f(point.flow.x, point.flow.y, point.flow.z, 0.f);
        Vec4f start = pointPos;
        Vec4f end = pointPos + predFlow;

        drawDebugLine(projection, view, start, end, Vec4f(1.f, 0.f, 0.f));
    }

    // Ground Truth Flow
    {
        Vec4f gtFlow = camModel * Vec4f(debugGroundTruthBuffer.x / 100, debugGroundTruthBuffer.y / 100, debugGroundTruthBuffer.z / 100, 0);
        Vec4f start = pointPos;
        Vec4f end = pointPos + gtFlow;

        drawDebugLine(projection, view, start, end, Vec4f(1.f, 1.f, 0.f));
    }

    // Error Prediction
    {
        Vec4f start = pointPos;
        Vec4f end = pointPos + Vec4f(0, debugOutputBuffer1.w, 0, 0);;

        drawDebugLine(projection, view, start, end, Vec4f(0.f, 1.f, 1.f));
    }
}

void TemPCCRenderer::render(Mat4f projection, Mat4f view){
    if(!shouldRender || validPointsNum == 0)
        return;

    // To sort the points in integrateStep (done for performance reasons there...):
    lastCamPos = view.inverse().getPosition();

    glPointSize(pointSize);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    int pointNumToRender = validPointsNum;

    glBufferData(GL_ARRAY_BUFFER, validPointsNum * sizeof(APoint), &points[0], GL_DYNAMIC_DRAW);

    //std::cout << "Valid Points Num: " << validPointsNum << std::endl;

    splatShader.bind();
    splatShader.setUniform("projection", projection);
    splatShader.setUniform("view", view);
    splatShader.setUniform("discardBlackPixels", false);
    splatShader.setUniform("onlyDrawHiddenPoints", onlyDrawHiddenPoints);
    splatShader.setUniform("onlyDrawVisiblePoints", onlyDrawVisiblePoints);
    splatShader.setUniform("drawLifeTimeColor", drawLifeTimeColor);
    splatShader.setUniform("colorizeHiddenPoints", colorizeHiddenPoints);

    splatShader.setUniform("showCam1", showCam1);
    splatShader.setUniform("showCam2", showCam2);
    splatShader.setUniform("showCam3", showCam3);
    splatShader.setUniform("debugID", selectedDebugPointID);
    splatShader.setUniform("pointSize", pointSize);

    APoint pp;

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(APoint), 0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(APoint), (GLvoid*)((unsigned char*)&pp.flow - (unsigned char*)&pp.position));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(APoint), (GLvoid*)((unsigned char*)&pp.color - (unsigned char*)&pp.position));
    glEnableVertexAttribArray(2);

    glVertexAttribIPointer(3, 1, GL_INT, sizeof(APoint), (GLvoid*)((unsigned char*)&pp.groundTruthIdx - (unsigned char*)&pp.position));
    glEnableVertexAttribArray(3);

    glVertexAttribIPointer(4, 1, GL_INT, sizeof(APoint), (GLvoid*)((unsigned char*)&pp.lifetime - (unsigned char*)&pp.position));
    glEnableVertexAttribArray(4);

    glVertexAttribIPointer(5, 1, GL_UNSIGNED_BYTE, sizeof(APoint), (GLvoid*)((unsigned char*)&pp.state - (unsigned char*)&pp.position));
    glEnableVertexAttribArray(5);

    glVertexAttribIPointer(6, 1, GL_UNSIGNED_BYTE, sizeof(APoint), (GLvoid*)((unsigned char*)&pp.camId - (unsigned char*)&pp.position));
    glEnableVertexAttribArray(6);

    glDrawArrays(GL_POINTS, 0, pointNumToRender);
    glDisable(GL_PROGRAM_POINT_SIZE);

    if(drawFlow){
        flowShader.bind();
        flowShader.setUniform("projection", projection);
        flowShader.setUniform("view", view);
        flowShader.setUniform("discardBlackPixels", false);
        flowShader.setUniform("onlyDrawHiddenPoints", onlyDrawHiddenPoints);
        flowShader.setUniform("onlyDrawVisiblePoints", onlyDrawVisiblePoints);

        flowShader.setUniform("showCam1", showCam1);
        flowShader.setUniform("showCam2", showCam2);
        flowShader.setUniform("showCam3", showCam3);

        glDrawArrays(GL_POINTS, 0, pointNumToRender);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    //drawDebugVectors(projection, view);

    // Ground Truth data:

    if(drawGTData && Data::instance->gpuGroundTruthPoints != nullptr && currentFrameID >= 0 && currentFrameID < Data::instance->groundTruthFrameCount){
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Berechnen der Größe der Punktdaten für zwei Frames
        int dataSize = Data::instance->groundTruthSampleCount * sizeof(float4);

        // Puffer für die Punktdaten des aktuellen und nächsten Frames
        float4* tmpGroundTruthPoints = new float4[Data::instance->groundTruthSampleCount * 2];
        // Kopieren der Punktdaten des aktuellen Frames
        cudaMemcpy(tmpGroundTruthPoints, Data::instance->gpuGroundTruthPoints + (currentFrameID * Data::instance->groundTruthSampleCount), dataSize, cudaMemcpyDeviceToHost);
        // Kopieren der Punktdaten des nächsten Frames
        cudaMemcpy(tmpGroundTruthPoints + Data::instance->groundTruthSampleCount, Data::instance->gpuGroundTruthPoints + ((currentFrameID + 1) * Data::instance->groundTruthSampleCount), dataSize, cudaMemcpyDeviceToHost);

        // Erstellen eines neuen Puffers mit den Daten beider Frames
        glBufferData(GL_ARRAY_BUFFER, Data::instance->groundTruthSampleCount * sizeof(float4) * 2, tmpGroundTruthPoints, GL_DYNAMIC_DRAW);

        delete[] tmpGroundTruthPoints;
        tmpGroundTruthPoints = nullptr;

        // Zuweisen des VertexAttributs für den aktuellen Frame (VertexAttribute 0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);
        glEnableVertexAttribArray(0);

        // Zuweisen des VertexAttributs für den nächsten Frame (VertexAttribute 1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)(Data::instance->groundTruthSampleCount * sizeof(float4)));
        glEnableVertexAttribArray(1);

        groundTruthFlowShader.bind();
        groundTruthFlowShader.setUniform("projection", projection);
        groundTruthFlowShader.setUniform("view", view);
        groundTruthFlowShader.setUniform("model", Mat4f());
        groundTruthFlowShader.setUniform("pointSize", 3);
        groundTruthFlowShader.setUniform("color", Vec4f(1.f, 1.f, 1.f, 1.f));

        glDrawArrays(GL_POINTS, 0, Data::instance->groundTruthSampleCount);
    }
};
