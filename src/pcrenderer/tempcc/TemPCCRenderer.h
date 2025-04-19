// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#pragma once

#include "src/pcrenderer/Renderer.h"
#include "src/pcrenderer/tempcc/TemPCCStructs.h"
#include "src/util/gl/Texture2D.h"
#include "src/util/gl/TextureFBO.h"
#include "src/util/gl/objects/GLCoordinateSystem.h"
#include "src/util/gl/Shader.h"
#include "src/util/gl/GLMesh.h"
#include "src/pcrenderer/tempcc/TinyFlowNet.h"
#include "src/util/TimeMeasurement.h"
#include "src/util/cuda/CuHashSet.h"
#include "src/Data.h"

/**
 * Some conventions:
 * - The position of an 'APoint' is always in world space (also 'camOrderedPoints')!
 * - The position of a 'gpuPositions' is always in camera space!
 */
class TemPCCRenderer : public Renderer {
    TextureFBO debugFBO1 = TextureFBO({TextureType(GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE)});
    TextureFBO debugFBO2 = TextureFBO({TextureType(GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE)});
    TextureFBO debugFBO3 = TextureFBO({TextureType(GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE)});

    // Coordinate System:
    GLCoordinateSystem coordinateSystem;

    // Splat shader:
    Shader splatShader = Shader(CMAKE_SOURCE_DIR "/shader/attention/SplatShader.vert", CMAKE_SOURCE_DIR "/shader/attention/SplatShader.frag");
    Shader flowShader = Shader(CMAKE_SOURCE_DIR "/shader/attention/flowShader.vert", CMAKE_SOURCE_DIR "/shader/attention/flowShader.frag", CMAKE_SOURCE_DIR "/shader/attention/flowShader.geo");
    Shader groundTruthFlowShader = Shader(CMAKE_SOURCE_DIR "/shader/attention/groundTruthFlowShader.vert", CMAKE_SOURCE_DIR "/shader/attention/groundTruthFlowShader.frag", CMAKE_SOURCE_DIR "/shader/attention/groundTruthFlowShader.geo");

    Shader debugLineShader = Shader(CMAKE_SOURCE_DIR "/shader/attention/debugLineShader.vert", CMAKE_SOURCE_DIR "/shader/attention/debugLineShader.frag");

    // Pointer to Vertex Array Object (on GPU):
    GLuint vao = 0;

    // Pointer to vertex buffer of points:
    GLuint vbo = 0;

    // Empty dummy VAO:
    GLuint dummyVAO = 0;

    int selectedDebugPointID = 0;

    // Points:
    APoint* points = nullptr;
    APoint* gpuPoints = nullptr;
    float4* gpuTemporalFlows = nullptr;
    unsigned int gpuPointsNum = 0;

    float3* pointNNSpatialInput = nullptr;
    float4* pointNNTemporalInput = nullptr;
    float4* pointNNOutput = nullptr;

    float3* trainingNNSpatialInput = nullptr;
    float4* trainingNNTemporalInput = nullptr;
    float4* trainingNNGroundTruth = nullptr;

    int trainingStride = 5;

    int simStepsWithoutTraining = 5;
    int simStepsWithoutTrainingVariance = 2;
    int currentNoTrainingSteps = 0;

    int* trainingShuffledIndices = nullptr;

    float3* debugInputSpatialBuffer = nullptr;
    float4* debugInputTemporalBuffer = nullptr;
    float4 debugOutputBuffer1;
    float4 debugOutputBuffer2;
    float4 debugOutputBuffer3;
    float4 debugGroundTruthBuffer;

    APoint* convertedCamPoints = nullptr;

    float* camOrderedWidenedNearestZ[ACTIVE_CAMS];
    float* borderDistanceImage[ACTIVE_CAMS];

    curandState* randStates;

    int integratedPointCloudNum = 0;

    int currentFrameID = -1;

    // Avoid deletion of data that is used by the tensor:
    torch::Deleter noop_deleter = [](void*) {};

    void loadTestSet();

    // Spatial tensor of test set:
    torch::Tensor testSetSpatialTensor;

    // Temporal tensor of test set:
    torch::Tensor testSetTemporalTensor;

    // Ground Truth tensor of test set:
    torch::Tensor testSetGTTensor;

    // Previous point clouds:
    std::vector<std::shared_ptr<OrganizedPointCloud>> lastIntegratedPointCloud;

    // TimeMeasurement:
    TimeMeasurement timer;

    cuda_hashset::HashSet densityHashSet = cuda_hashset::HashSet(Data::instance->TemPCC_MaxPointNum * 10);

    float4* projectedGTPositions[ACTIVE_CAMS];
    float4* projectedGTFlows[ACTIVE_CAMS];
    float4* projectedGTFlowsDense[ACTIVE_CAMS];

    std::vector<EvalAPoint> evalAPoints;
    std::vector<EvalFrame> evalFrames;
    bool evalInitializationDone = false;

public:
    // Clear points in next frame?
    bool clearPointsInNextFrame = false;

    // Neural Network module (native):
    TinyFlowNet nativeModule = TinyFlowNet();

    int flowRows = 30;
    int flowCols = 40;

    std::vector<std::shared_ptr<PDFlow>> pdFlows;

    // Point size of splats:
    float pointSize = 15.f;

    float learningRate = 1e-4;

    // Real input for lstm during training (0: only GT data, 1: only real data):
    float scheduledSampling = 0.2f;
    float allowedTrainingDivergence = 0.f;

    int everyNCamPoint = 1;

    bool drawFlow = false;
    bool drawGTData = false;
    bool drawLifeTimeColor = false;
    bool onlyDrawHiddenPoints = false;
    bool onlyDrawVisiblePoints = false;
    bool shouldStoreTrainingData = false;
    bool flowGenerationEnabled = true;
    bool shouldTrain = false;
    bool shouldEvaluate = false;
    bool colorizeHiddenPoints = false;
    bool shouldRender = !shouldEvaluate;
    bool usePDFlow = Data::instance->TemPCC_UsePDFlow;
    bool applyFlow = true;

    bool previousExceededEnd = false;
    bool previousExceededInTrainingEnd = false;

    bool showCam1 = true;
    bool showCam2 = true;
    bool showCam3 = true;

    unsigned int pidx = 0;

    int validPointsNum = 0;
    int validCamPointsNum = 0;
    int validTrainingNum = 0;

    bool trainingBufferFull = false;

    int currentTrainingOffset = 0;

    float lastLoss = 0;

    std::ofstream flowOutputStream;

    std::shared_ptr<torch::optim::Adam> optimizer = nullptr;

    bool debugViewOpened = false;

    /**
     * Creates the attention renderer.
     *
     * The given 'flowRows' parameter defines the precision of the flow estimation on
     * camera images. The maximal value may be the camera image height divided by 2 or
     * a part of it.
     */
    TemPCCRenderer(unsigned int flowRows = 288);

    /**
     * Destructor of the attention renderer.
     */
    ~TemPCCRenderer();

    /**
     *
     */
    void drawDebugLine(Mat4f& projection, Mat4f& view, Vec4f start, Vec4f end, Vec4f color = Vec4f(1.0, 1.0, 1.0));

    /**
     *
     */
    void drawDebugVectors(Mat4f& projection, Mat4f& view);

    /**
     * Integrate new RGB XYZ images.
     */
    virtual void integratePointClouds(std::vector<std::shared_ptr<OrganizedPointCloud>> pointClouds) override;

    /**
     * Copies each OrganizedPointCloud (which is seperate per camera) into a continous
     * block.
     */
    int copyImagesToPointArray(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds);

    /**
     * Estimates the flow for every single depth camera image (OrganizedPointCloud).
     */
    void generateFlow(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds);

    /**
     * Estimates the flow of hidden points in the total point set.
     */
    void inferencePass(int validPointsNum);

    /**
     * Generate training data
     */
    void trainingPass(float timestamp);

    void loadNet(){
        nativeModule.loadNet();
    };

    void saveNet(){
        nativeModule.saveNet();
    }

    /**
     * Assigns ground truth data to camera points.
     */
    void assignGroundTruthData(int currentFrameID);

    /**
     * Renders the point cloud.
     */
    virtual void render(Mat4f projection, Mat4f view) override;

    /**
     * Stores training data if required.
     */
    void storeTrainingData(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds);

    std::map<std::string, float> getTimings(){
        return timer.getTimeMeasuresInMilliSec();
    }

    void evaluate();

    Vec4f lastCamPos;
};
