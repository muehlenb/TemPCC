// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "src/pch.h"

#include "src/pcrenderer/tempcc/TemPCCRenderer.h"

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

torch::Tensor weighted_loss(torch::Tensor prediction, torch::Tensor groundTruth) {
    // Extrahieren der Komponenten von 'prediction'
    torch::Tensor x = prediction.select(1, 0);
    torch::Tensor y = prediction.select(1, 1);
    torch::Tensor z = prediction.select(1, 2);
    torch::Tensor w = prediction.select(1, 3);

    // Extrahieren der Komponenten von 'groundTruth'
    torch::Tensor gt_x = groundTruth.select(1, 0);
    torch::Tensor gt_y = groundTruth.select(1, 1);
    torch::Tensor gt_z = groundTruth.select(1, 2);
    torch::Tensor gt_w = groundTruth.select(1, 3);

    // Berechnen des Gewichtsfaktors
    //torch::Tensor weight_factor = torch::abs(gt_w * 10);

    // Berechnung des gewichteten MSE für x, y, z
    torch::Tensor loss_xyz = (torch::pow(x - gt_x, 2) + torch::pow(y - gt_y, 2) + torch::pow(z - gt_z, 2));

    torch::Tensor loss_xyzlength = (torch::pow(x, 2) + torch::pow(y, 2) + torch::pow(z, 2)) * 0.1;

    // Konstanter MSE für w
    torch::Tensor loss_w = torch::pow(w - gt_w, 2);

    // Gesamtloss als Durchschnitt der einzelnen Losses
    torch::Tensor total_loss = (loss_xyz.sum() + loss_w.sum() + loss_xyzlength.sum()) / (prediction.size(0) * 5.5);
    //torch::Tensor total_loss = (loss_xyz.sum()) / (prediction.size(0) * 3);

    return total_loss;
}

void TemPCCRenderer::loadTestSet(){
    std::ifstream testSetReader = std::ifstream(CMAKE_SOURCE_DIR "/tools/testSet.bin", std::ios::binary);


    std::vector<float3> spatialData;
    std::vector<float4> temporalData;
    std::vector<float4> groundTruthData;

    float num = 0;
    while (!testSetReader.eof()) {
        std::vector<float3> thisSpatialData;
        for(int i=0; i < NN_SPATIAL_INPUT_SIZE; ++i){
            float3 val;
            testSetReader.read(reinterpret_cast<char*>(&val.x), sizeof(float)*3);
            thisSpatialData.push_back(ensureNonNan(val, 0));
        }
        testSetReader.read(reinterpret_cast<char*>(&num), sizeof(float));

        std::vector<float4> thisTemporalData;
        for(int i=0; i < NN_TEMPORAL_INPUT_SIZE; ++i){
            float4 val;
            testSetReader.read(reinterpret_cast<char*>(&val.x), sizeof(float)*3);
            thisTemporalData.push_back(ensureNonNan(val, 1));
        }
        testSetReader.read(reinterpret_cast<char*>(&num), sizeof(float));

        float4 gt;
        testSetReader.read(reinterpret_cast<char*>(&gt.x), sizeof(float)*4);

        testSetReader.read(reinterpret_cast<char*>(&num), sizeof(float));

        if(num != 42.f){
            std::cout << "ASSERT ERROR! 42!" << std::endl;
        }

        bool gtCheck = !isnan(gt.x) && !isnan(gt.y) && !isnan(gt.z) && !isnan(gt.w) && abs(gt.x) < 5 && abs(gt.y) < 5 && abs(gt.z) < 5 && abs(gt.w) < 5;
        bool spatialCheck = true;
        bool temporalCheck = true;

        for(float3 val : thisSpatialData)
            spatialCheck &= !isnan(val.x) && !isnan(val.y) && !isnan(val.z) && abs(val.x) < 5 && abs(val.y) < 5 && abs(val.z) < 5;
        for(float4 val : thisTemporalData)
            temporalCheck &= !isnan(val.x) && !isnan(val.y) && !isnan(val.z) && abs(val.x) < 5 && abs(val.y) < 5 && abs(val.z) < 5;


        if(gtCheck && spatialCheck && temporalCheck){
            for(float3 val : thisSpatialData)
                spatialData.push_back(val);

            for(float4 val : thisTemporalData)
                temporalData.push_back(val);

            groundTruthData.push_back(gt);
        }
    }

    int size = groundTruthData.size();

    std::cout << "    TestDataSet: Spatial Size: " << spatialData.size() << " | Temporal Size: " << temporalData.size() << " | Ground Truth Size: " << groundTruthData.size() << std::endl;

    torch::Tensor testSetSpatialTensorCPU = torch::from_blob(spatialData.data(), {size, NN_SPATIAL_INPUT_SIZE * 3}, torch::dtype(torch::kFloat32));
    torch::Tensor testSetTemporalTensorCPU = torch::from_blob(temporalData.data(), {size, NN_TEMPORAL_INPUT_SIZE, 4}, torch::dtype(torch::kFloat32));
    torch::Tensor testSetGTTensorCPU = torch::from_blob(groundTruthData.data(), {size, 4}, torch::dtype(torch::kFloat32));

    testSetSpatialTensor = testSetSpatialTensorCPU.to(torch::kCUDA);
    testSetTemporalTensor = testSetTemporalTensorCPU.to(torch::kCUDA);
    testSetGTTensor = testSetGTTensorCPU.to(torch::kCUDA);
}

/**
 * Attention Renderer Constructor
 */
TemPCCRenderer::TemPCCRenderer(unsigned int flowRows)
    : flowRows(flowRows)
    , flowCols(static_cast<unsigned int>(std::ceil(flowRows*10.0/9.0)))
{
    std::cout << "Constructing the TemPCC Renderer..." << std::endl;
    loadTestSet();
    std::cout << "    Testset loaded." << std::endl;

    // Create points on CPU (for rendering):
    points = new APoint[MAX_APOINT_NUM];

    // Create points on GPU (for the algorithm with fusion and completion):
    cudaMalloc(&gpuPoints, MAX_APOINT_NUM * sizeof(APoint));
    cudaMalloc(&gpuTemporalFlows, MAX_APOINT_NUM * NN_TEMPORAL_INPUT_SIZE * sizeof(float4));

    cudaMalloc(&convertedCamPoints, MAX_CONVCAMPOINTS_NUM * sizeof(APoint));

    std::cout << "    Memory for gpuPoints and ConvertedCamPoints created." << std::endl;

    // Setup Random Number states (curand):
    cudaMalloc(&randStates, MAX_APOINT_NUM * sizeof(curandState));
    int threadsPerBlock = 256;
    int numBlocks = (MAX_APOINT_NUM + threadsPerBlock - 1) / threadsPerBlock;
    cuda_tempcc::cudaSetupRandStates(numBlocks, threadsPerBlock, randStates, time(0));
    cuda_tempcc::cudaSetupAPoints(numBlocks, threadsPerBlock, gpuPoints, MAX_APOINT_NUM);


    debugInputSpatialBuffer = new float3[NN_SPATIAL_INPUT_SIZE];
    debugInputTemporalBuffer = new float4[NN_TEMPORAL_INPUT_SIZE];

    std::cout << "    Debug Arrays created" << std::endl;
    // Neural Net samples:
    cudaMalloc(&pointNNTemporalInput, MAX_APOINT_NUM * NN_TEMPORAL_INPUT_SIZE * sizeof(float4) * ACTIVE_CAMS);
    cudaMalloc(&pointNNSpatialInput, MAX_APOINT_NUM * NN_SPATIAL_INPUT_SIZE * sizeof(float3) * ACTIVE_CAMS);
    cudaMalloc(&pointNNOutput, MAX_APOINT_NUM * sizeof(float4) * ACTIVE_CAMS);

    cudaMalloc(&trainingNNTemporalInput, TRAINING_POINT_NUM * NN_TEMPORAL_INPUT_SIZE * ACTIVE_CAMS * sizeof(float4));
    cudaMalloc(&trainingNNSpatialInput, TRAINING_POINT_NUM * NN_SPATIAL_INPUT_SIZE * ACTIVE_CAMS * sizeof(float3));
    cudaMalloc(&trainingNNGroundTruth, TRAINING_POINT_NUM * ACTIVE_CAMS * sizeof(float4));
    cudaMalloc(&trainingShuffledIndices, TRAINING_POINT_NUM * sizeof(int));

    cudaMemset(trainingNNTemporalInput, 0, TRAINING_POINT_NUM * ACTIVE_CAMS * NN_TEMPORAL_INPUT_SIZE * sizeof(float4));
    cudaMemset(trainingNNSpatialInput, 0, TRAINING_POINT_NUM * ACTIVE_CAMS * NN_SPATIAL_INPUT_SIZE * sizeof(float3));
    cudaMemset(trainingNNGroundTruth, 0, TRAINING_POINT_NUM * ACTIVE_CAMS * sizeof(float4));

    std::vector<int> indices(TRAINING_POINT_NUM);
    // Shuffle Indices:
    {
        for(int i=0;i < TRAINING_POINT_NUM; ++i){
            indices[i] = i;
        }
        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(indices.begin(), indices.begin() + TRAINING_POINT_NUM, g);

        cudaMemcpy(trainingShuffledIndices, &indices[0], TRAINING_POINT_NUM * sizeof(int), cudaMemcpyHostToDevice);
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

    std::cout << "    Further buffers created." << std::endl;
    loadGroundTruthData("D:\\AttRen_TrainingData_final\\NiagaraSamples.bin", 28, 900);
    std::cout << "    Ground Truth Data loaded." << std::endl;
    std::cout << "    Finished." << std::endl;
}

void TemPCCRenderer::loadGroundTruthData(std::string path, int startOffset, int totalFrameCount) {
    std::ifstream gtStream(path, std::ios::binary);
    if (!gtStream.good()) {
        std::cout << "Ground Truth Stream could not be opened!" << std::endl;
        return;
    }

    groundTruthFrameCount = totalFrameCount;
    gtStream.read(reinterpret_cast<char*>(&groundTruthSampleCount), sizeof(int));

    // Reserve space on GPU:
    cudaMalloc(&gpuGroundTruthPoints, groundTruthFrameCount * groundTruthSampleCount * sizeof(float4));
    cpuGroundTruthPoints = new float4[groundTruthFrameCount * groundTruthSampleCount];

    float4* emptyFrame = new float4[groundTruthSampleCount];
    std::fill(emptyFrame, emptyFrame + groundTruthSampleCount, float4{0.0f, 0.0f, 0.0f, 0.0f});

    // Fill up the start:
    cudaMemset(gpuGroundTruthPoints, 0, startOffset * groundTruthSampleCount * sizeof(float4));

    // Allocate buffer for reading data
    double* buffer = new double[groundTruthSampleCount * 3];
    float4* frame = new float4[groundTruthSampleCount];

    int frameID = startOffset;
    float timestamp;

    while (frameID < totalFrameCount && gtStream.read(reinterpret_cast<char*>(&timestamp), sizeof(float))) {
        gtStream.read(reinterpret_cast<char*>(buffer), sizeof(double) * 3 * groundTruthSampleCount);

        #pragma omp parallel for
        for (int i = 0; i < groundTruthSampleCount; ++i) {
            frame[i] = float4{float(buffer[i * 3 + 1]) / 100, float(buffer[i * 3 + 2]) / 100, float(buffer[i * 3]) / 100, 1.0f};
        }

        cudaMemcpy(gpuGroundTruthPoints + (frameID * groundTruthSampleCount), frame, groundTruthSampleCount * sizeof(float4), cudaMemcpyHostToDevice);

        frameID++;
    }

    // Fill up the end:
    if (frameID < totalFrameCount) {
        cudaMemset(gpuGroundTruthPoints + (frameID * groundTruthSampleCount), 0, (totalFrameCount - frameID) * groundTruthSampleCount * sizeof(float4));
    }

    cudaMemcpy(cpuGroundTruthPoints, gpuGroundTruthPoints, groundTruthFrameCount * groundTruthSampleCount * sizeof(float4), cudaMemcpyDeviceToHost);

    delete[] frame;
    delete[] emptyFrame;
    delete[] buffer;
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

    delete[]cpuGroundTruthPoints;

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

int hiddenFlowCounter = 0;


void TemPCCRenderer::inferencePass(int validPointsNum){
    nativeModule.eval();
    torch::NoGradGuard no_grad;

    int pointsPerBatch = 50000;
    int splitNum = (validPointsNum * ACTIVE_CAMS) / pointsPerBatch + 1;

    splitNum = std::min(splitNum, (MAX_APOINT_NUM * ACTIVE_CAMS) / pointsPerBatch);

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

    int batchSize = TempSettings::TrainingBatchSize;
    int batches = (TRAINING_POINT_NUM * ACTIVE_CAMS) / batchSize;

    // Optimierer definieren (z.B. Adam, SGD, etc.)
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

            // Training durchführen
            optimizer->zero_grad(); // Gradienten zurücksetzen

            // Ausgabe vorhersagen
            at::Tensor prediction = nativeModule.forward(temporalInput, spatialInput);

            // Verlust berechnen (Annahme: MSE)
            at::Tensor loss = torch::nn::functional::l1_loss(prediction, groundTruth);

            // Backpropagation
            loss.backward();

            // Optimizer-Schritt
            optimizer->step();

            // Optional: Ausgabe des Verlustes für Monitoring/Zwecke der Verfolgung
            //if(i % 10 == 0) { // Beispielsweise alle 10 Schritte
            //    std::cout << "Epoch [" << epoch << "], Step [" << i << "], Loss: " << loss.item<float>() << std::endl;
            //}

            if(i + 1 == batches){
                lastLoss = loss.item<float>();
            }
        }
    }


    // Calculate test set loss:
    float testLoss = 0.f;
    float loss3DFloat = 0.f;
    float lossErrFloat = 0.f;
    /*
    {
        nativeModule.eval();
        torch::NoGradGuard no_grad;

        at::Tensor prediction = nativeModule.forward(testSetTemporalTensor, testSetSpatialTensor);

        // Überprüfen auf NaN-Werte
        if (contains_nan(testSetTemporalTensor)) {
            std::cout << "NaN found in testSetTemporalTensor" << std::endl;
        }
        if (contains_nan(testSetSpatialTensor)) {
            std::cout << "NaN found in testSetSpatialTensor" << std::endl;
        }
        if (contains_nan(testSetGTTensor)) {
            std::cout << "NaN found in testSetGTTensor" << std::endl;
        }
        if (contains_nan(prediction)) {
            std::cout << "NaN found in prediction" << std::endl;
        }
        auto prediction_3d = prediction.narrow(1, 0, 3); // Beginne bei Index 0 und nimm 3 Elemente in der 2. Dimension
        auto prediction_4th = prediction.narrow(1, 3, 1).squeeze(-1); // Nimm nur das Element an Position 3
        auto testSetGT_3d = testSetGTTensor.narrow(1, 0, 3); // Beginne bei Index 0 und nimm 3 Elemente in der 2. Dimension
        auto testSetGT_4th = testSetGTTensor.narrow(1, 3, 1).squeeze(-1); // Nimm nur das Element an Position 3

        // Verlust berechnen (Annahme: MSE)

        at::Tensor loss = torch::mse_loss(prediction, testSetGTTensor);
        testLoss = loss.item<float>();

        at::Tensor loss3D = torch::mse_loss(prediction_3d, testSetGT_3d);
        loss3DFloat = loss3D.item<float>();

        at::Tensor lossErr = torch::mse_loss(prediction_4th, testSetGT_4th);
        lossErrFloat = lossErr.item<float>();
    }*/

    std::cerr << "" << std::time(0) << "," << timestamp << "," << validTrainingNum << "," << batches << "," << learningRate << "," << scheduledSampling << "," << lastLoss << "," << testLoss << "," << loss3DFloat << "," << lossErrFloat << "" << std::endl;
    nativeModule.saveNet();

    //if(nativeModule.attention_enabled){
        //scheduledSampling += 0.0000f;
        //if(scheduledSampling > 0.5f)
        //    scheduledSampling = 0.5f;

        //allowedTrainingDivergence += 0.0004f;
        if(allowedTrainingDivergence > 0.2f)
            allowedTrainingDivergence = 0.2f;

    //}

    learningRate *= 0.99f;
    scheduledSampling += 0.002f;

    if(scheduledSampling > 1.f)
        scheduledSampling = 1.f;

    if(timestamp > 29.66667 && !previousExceededInTrainingEnd){
        clearPointsInNextFrame = true;

        //if(learningRate < TempSettings::MinimalLearningRate){
        //    nativeModule.saveNet(TempSettings::NetName);
            //std::exit(0);
        //}

        nativeModule.saveNet(std::to_string(std::time(0)) + "_" + std::to_string(learningRate) + ".pt");


        if(TempSettings::AttentionActive)
            nativeModule.attention_enabled = true;

        previousExceededInTrainingEnd = timestamp > 29.66667;
    }
}


void TemPCCRenderer::assignGroundTruthData(int currentFrameID){
    int blockSize = 256; // Anzahl der Threads pro Block
    int numBlocks = (groundTruthSampleCount + blockSize - 1) / blockSize; // Berechne die benötigte Anzahl von Blöcken

    cuda_tempcc::cudaAssignGroundTruthData(numBlocks, blockSize, convertedCamPoints, gpuGroundTruthPoints + (currentFrameID * groundTruthSampleCount), groundTruthSampleCount);
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
            cuda_tempcc::cudaProjectGTFlowIntoImage(i, projectedGTPositions[i], projectedGTFlows[i], pointClouds[i]->gpuPositions, gpuGroundTruthPoints + (currentFrameID * groundTruthSampleCount), groundTruthSampleCount);
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

        cuda_tempcc::cudaMainPassKernelPartA(numBlocks, threadsPerBlock, gpuPoints, gpuTemporalFlows, convertedCamPoints, pointNNTemporalInput, pointNNSpatialInput, pointNNOutput, nullptr, gpuGroundTruthPoints + (currentFrameID * groundTruthSampleCount), groundTruthSampleCount, validPointsNum, randStates, scheduledSampling);
        cudaDeviceSynchronize();


        --currentNoTrainingSteps;
        // Trainings pass:
        if(shouldTrain && validTrainingNum > 0 && currentNoTrainingSteps <= 0){
            int threadsPerBlock = 256;
            int numBlocks = (validTrainingNum / trainingStride + threadsPerBlock - 1) / threadsPerBlock;

            cuda_tempcc::cudaCopyIntoTraningBuffer(numBlocks, threadsPerBlock, pointNNTemporalInput, pointNNSpatialInput, pointNNOutput, trainingNNTemporalInput, trainingNNSpatialInput, trainingNNGroundTruth, trainingShuffledIndices, randStates, currentTrainingOffset, trainingStride, validTrainingNum);
            currentTrainingOffset = (currentTrainingOffset + validTrainingNum / trainingStride);

            if(currentTrainingOffset >= TRAINING_POINT_NUM){
                trainingBufferFull = true;
                currentTrainingOffset = currentTrainingOffset % TRAINING_POINT_NUM;
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

        cuda_tempcc::cudaMainPassKernelPartB(numBlocks, threadsPerBlock, gpuPoints, gpuTemporalFlows, convertedCamPoints, gpuGroundTruthPoints + (currentFrameID * groundTruthSampleCount), pointNNOutput, nullptr, validPointsNum, randStates, shouldTrain, allowedTrainingDivergence, applyFlow);
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

    // If there might be more points to be copied than space is available, we have to move
    // the gpuPointsCopyStartPtr to avoid copying into non existing memory:
    if(validPointsAfterRemove + validCamPoints > MAX_APOINT_NUM)
        gpuPointsCopyStartPtr = gpuPoints + MAX_APOINT_NUM - validCamPoints;

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
            int currentGroundTruthOffset = (currentFrameID * groundTruthSampleCount);

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

                float4 thisGTPos = cpuGroundTruthPoints[currentGroundTruthOffset + p.groundTruthIdx];
                float4 nextGTPos = cpuGroundTruthPoints[currentGroundTruthOffset + groundTruthSampleCount + p.groundTruthIdx];
                float4 gtFlow = {nextGTPos.x - thisGTPos.x, nextGTPos.y - thisGTPos.y, nextGTPos.z - thisGTPos.z, 0};
                pEval.groundTruthFlowLength = sqrt(gtFlow.x * gtFlow.x + gtFlow.y * gtFlow.y + gtFlow.z * gtFlow.z);

                float4 flowError = {p.flow.x - gtFlow.x, p.flow.y - gtFlow.y, p.flow.z - gtFlow.z};
                pEval.flowError =  sqrt(flowError.x * flowError.x + flowError.y * flowError.y + flowError.z * flowError.z);

                float4 pathDiff = {p.position.x - thisGTPos.x, p.position.y - thisGTPos.y, p.position.z - thisGTPos.z, 0};
                pEval.positionalDivergence = sqrt(pathDiff.x * pathDiff.x + pathDiff.y * pathDiff.y + pathDiff.z * pathDiff.z);

                float pathLength = 0.f;

                for(int l=0; l < p.lifetime; ++l){
                    float4 thisP = cpuGroundTruthPoints[currentGroundTruthOffset - groundTruthSampleCount * (l) + p.groundTruthIdx];
                    float4 nextP = cpuGroundTruthPoints[currentGroundTruthOffset - groundTruthSampleCount * (l-1) + p.groundTruthIdx];
                    float4 flow = {nextP.x - thisP.x, nextP.y - thisP.y, nextP.z - thisP.z, 0};
                    pathLength += sqrt(flow.x * flow.x + flow.y * flow.y + flow.z * flow.z);
                };

                pEval.groundTruthPathLength = pathLength;
                pEval.lifetime = p.lifetime;
                pEval.frameId = currentFrameID;

                evalAPoints.push_back(pEval);
            }

            int numberOfSeenGTSamples = 0;
            for(int i=0; i < groundTruthSampleCount; ++i){
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
            frameEval.totalGroundTruthSamples = groundTruthSampleCount;
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

    if(drawGTData && gpuGroundTruthPoints != nullptr && currentFrameID >= 0 && currentFrameID < groundTruthFrameCount){
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Berechnen der Größe der Punktdaten für zwei Frames
        int dataSize = groundTruthSampleCount * sizeof(float4);

        // Puffer für die Punktdaten des aktuellen und nächsten Frames
        float4* tmpGroundTruthPoints = new float4[groundTruthSampleCount * 2];
        // Kopieren der Punktdaten des aktuellen Frames
        cudaMemcpy(tmpGroundTruthPoints, gpuGroundTruthPoints + (currentFrameID * groundTruthSampleCount), dataSize, cudaMemcpyDeviceToHost);
        // Kopieren der Punktdaten des nächsten Frames
        cudaMemcpy(tmpGroundTruthPoints + groundTruthSampleCount, gpuGroundTruthPoints + ((currentFrameID + 1) * groundTruthSampleCount), dataSize, cudaMemcpyDeviceToHost);

        // Erstellen eines neuen Puffers mit den Daten beider Frames
        glBufferData(GL_ARRAY_BUFFER, groundTruthSampleCount * sizeof(float4) * 2, tmpGroundTruthPoints, GL_DYNAMIC_DRAW);

        delete[] tmpGroundTruthPoints;
        tmpGroundTruthPoints = nullptr;

        // Zuweisen des VertexAttributs für den aktuellen Frame (VertexAttribute 0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);
        glEnableVertexAttribArray(0);

        // Zuweisen des VertexAttributs für den nächsten Frame (VertexAttribute 1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)(groundTruthSampleCount * sizeof(float4)));
        glEnableVertexAttribArray(1);

        groundTruthFlowShader.bind();
        groundTruthFlowShader.setUniform("projection", projection);
        groundTruthFlowShader.setUniform("view", view);
        groundTruthFlowShader.setUniform("model", Mat4f());
        groundTruthFlowShader.setUniform("pointSize", 3);
        groundTruthFlowShader.setUniform("color", Vec4f(1.f, 1.f, 1.f, 1.f));

        glDrawArrays(GL_POINTS, 0, groundTruthSampleCount);
    }
};
