// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "src/pcstreamer/Streamer.h"
#include "src/pcstreamer/BinaryStreamer.h"

#ifdef USE_KINECT
#include "src/pcstreamer/AzureKinectMKVStreamer.h"
#endif

const char* Streamer::availableStreamerNames[] = { "- No streamer selected -", "UE5 Recording (Binary)", "Azure Kinect (mkv)"};

const unsigned int Streamer::availableStreamerNum = 3;

std::shared_ptr<Streamer> Streamer::constructStreamerInstance(int type, std::string path){
    if(type == 1){
        return std::make_shared<BinaryStreamer>(path);
    } else if(type == 2){
        return std::make_shared<AzureKinectMKVStreamer>(path);
    }

    if(type >= 1 && type < int(sizeof(availableStreamerNames)))
        std::cout << "Created " << availableStreamerNames[type] << std::endl;

    return nullptr;
};
