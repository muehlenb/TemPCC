// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)
#pragma once

#include "src/pcrenderer/tempcc/TemPCCStructs.h"

/**
 * Defining thrust functions for the attention renderer.
 */
namespace cuda_tempcc {
    APoint* thrustPartitionByGTIndex(APoint* gpuPoints, int pointsNum);
    APoint* thrustRemoveIfInvalid(APoint* gpuPoints, int pointsNum);
    APoint* thrustPartitionValid(APoint* gpuPoints, int pointsNum);
    void thrustSortByCamDistance(APoint* gpuPoints, int pointsNum, float4* deviceLastCamPos);
}
