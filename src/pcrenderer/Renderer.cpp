// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "src/pcrenderer/Renderer.h"
#include "src/pcrenderer/SplatRenderer.h"
#include "src/pcrenderer/tempcc/TemPCCRenderer.h"


const char* Renderer::availableAlgorithmNames[] = { "Rendering Only", "TemPCC"};

const unsigned int Renderer::availableAlgorithmNum = 2;

std::shared_ptr<Renderer> Renderer::constructAlgorithmInstance(int type){
    if(type == 0){
        return std::make_shared<SplatRenderer>();
    } else if(type == 1){
        return std::make_shared<TemPCCRenderer>();
    }

    std::cout << "Created " << availableAlgorithmNames[type] << std::endl;
    return nullptr;
};
