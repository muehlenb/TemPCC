#pragma once

#include<string>

/**
 * Global data for TemPCC.
 *
 * Should probably be refactored...
 */
struct Data {
    int groundTruthFrameCount = 0;
    int groundTruthSampleCount = 0;
    float4* gpuGroundTruthPoints = nullptr;
    float4* cpuGroundTruthPoints = nullptr;

    int TemPCC_MaxPointNum = 500000;
    int TemPCC_TrainingPointNum = 100000;
    int TemPCC_InferenceBatchSize = 1024;
    int TemPCC_TrainingBatchSize = 512;
    double TemPCC_MinimalLearningRate = 0.0000001;

    bool TemPCC_UsePDFlow = false;

    static Data* instance;
};
