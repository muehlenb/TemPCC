// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#pragma once
#include <iostream>

#define MAX_APOINT_NUM 1000000
#define TRAINING_POINT_NUM 1000000

#define NN_SPATIAL_INPUT_SIZE 98
#define NN_TEMPORAL_INPUT_SIZE 30

#define PIXELS_PER_CAM 368640
#define ACTIVE_CAMS 3

#define MAX_CONVCAMPOINTS_NUM PIXELS_PER_CAM * ACTIVE_CAMS

/**
 * A point structure for this attention renderer that can be
 * stored in ram and video ram.
 */
struct APoint {
    float4 position;
    float4 flow;
    uchar4 color;
    int groundTruthIdx;
    int lifetime;
    unsigned char state;
    unsigned char camId;

    /**
     * A static Idx which is maintained for this point during execution to
     * maintain the reference to external memory (e.g. gpuTemporalFlows).
     */
    int staticIdx;

    unsigned char newFlowIdx;
    unsigned char numberOfFlows;

    __host__ __device__
    APoint()
        : position({0,0,0,0})
        , flow({0,0,0,0})
        , color({0,0,0,0})
        , groundTruthIdx(-1)
        , lifetime(0)
        , state(0)
        , camId(-1)
        , staticIdx(-1)
        , newFlowIdx(0)
        , numberOfFlows(0)
    {}

    __host__ __device__
    APoint(float4 pos, uchar4 rgb, int camId)
        : position(pos)
        , flow({0,0,0,0})
        , color(rgb)
        , groundTruthIdx(-1)
        , lifetime(0)
        , state(1)
        , camId(camId)
        , newFlowIdx(0)
        , numberOfFlows(0)
    {}
};


/**
 * A cam info to project points into the camera image. It is assumed that
 * organized point clouds are aligned in row order (image).
 */
struct CamInfo {
    unsigned int width = 0;
    unsigned int height = 0;

    float* nearestWidenedZ = nullptr;
    float* borderDistance = nullptr;
    float4* gpuPositions = nullptr;

    float2* lookupImageTo3D = nullptr;
    float2* lookup3DToImage = nullptr;
    unsigned int lookup3DToImageSize = 0;
};


struct EvalAPoint {
    float predictedFlowLength = 0.f;
    float groundTruthFlowLength = 0.f;
    float flowError = 0.f;
    float positionalDivergence = 0.f;
    float groundTruthPathLength = 0.f;
    uint8_t state = 0;
    int frameId = 0;
    int groundTruthId = 0;
    int lifetime = 0;

    friend std::ostream& operator<<(std::ostream& os, const EvalAPoint& p) {
        os << p.predictedFlowLength << ","
           << p.groundTruthFlowLength << ","
           << p.flowError << ","
           << p.positionalDivergence << ","
           << p.groundTruthPathLength << ","
           << int(p.state) << ","
           << p.frameId << ","
           << p.groundTruthId << ","
           << p.lifetime;
        return os;
    }
};

struct EvalFrame {
    int groundTruthPoints = 0;
    int frameId = 0;
    int camPoints = 0;
    int totalPoints = 0;
    int seenGroundTruthSamples = 0;
    int totalGroundTruthSamples = 0;
    float timerTotal = 0.f;
    float timerImageFlow = 0.f;
    float timerMainPassA = 0.f;
    float timerInference = 0.f;
    float timerMainPassB = 0.f;
    float timerCleanGPUPoints = 0.f;
    float timerCleanCamPoints = 0.f;
    float timerDensityFilter = 0.f;

    friend std::ostream& operator<<(std::ostream& os, const EvalFrame& frame) {
        os << frame.groundTruthPoints << ","
           << frame.frameId << ","
           << frame.camPoints << ","
           << frame.totalPoints << ","
           << frame.seenGroundTruthSamples << ","
           << frame.totalGroundTruthSamples << ","
           << frame.timerTotal << ","
           << frame.timerImageFlow << ","
           << frame.timerMainPassA << ","
           << frame.timerInference << ","
           << frame.timerMainPassB << ","
           << frame.timerCleanGPUPoints << ","
           << frame.timerCleanCamPoints << ","
           << frame.timerDensityFilter;
        return os;
    }
};
