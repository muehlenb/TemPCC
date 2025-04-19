// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)
#pragma once

#define GROUNDTRUTH_START 28
#define GROUNDTRUTH_END 900

#include "src/Data.h"

#include <thread>

#include "src/pcstreamer/Streamer.h"
#include "src/pcstreamer/binary/BinaryStream.h"

#include "src/util/OrganizedPointCloud.h"

#include <chrono>
using namespace std::chrono;

#include <filesystem>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

/**
 * A streamer when using a single or multiple HDF5 recordings.
 */
class BinaryStreamer : public FileStreamer {
private:
    std::vector<std::shared_ptr<BinaryStream>> streams;

    std::thread readingThread;
    bool shouldStop = false;

    float processingTime = 0.f;
    float lastTimeWhileStopped = -1.f;

    float fps = 30.f;

    std::chrono::time_point<std::chrono::high_resolution_clock> lastFrameTime;

public:
    bool stream1Active = true;
    bool stream2Active = true;
    bool stream3Active = true;

    /**
     * Opens the given HDF5 files and
     */
    BinaryStreamer(std::string datasetPath){
        std::filesystem::path datasetFilePath(datasetPath);
        std::filesystem::path baseDir = datasetFilePath.parent_path();

        std::ifstream file(datasetPath);
        if (!file.is_open()) {
            std::cerr << "Error while opening file!" << std::endl;
            return;
        }

        json j;
        file >> j;

        std::string ground_truth = j["ground_truth"];
        std::cout << "Load Ground Truth File: " << baseDir / ground_truth << std::endl;
        loadGroundTruthData((baseDir / ground_truth).u8string(), GROUNDTRUTH_START, GROUNDTRUTH_END);
        std::cout << "Ground Truth loaded." << std::endl;

        for (const auto& camera : j["cameras"]) {
            streams.push_back(std::make_shared<BinaryStream>(
                (baseDir / camera["rgb_path"].get<std::string>()).u8string(),
                (baseDir / camera["depth_path"].get<std::string>()).u8string(),
                Mat4f(camera["transformation_matrix"]),
                0.0
            ));
        }

        readingThread = std::thread([this](){
            lastFrameTime = high_resolution_clock::now();
            while(!shouldStop){
                if(!isPlaying){
                    std::this_thread::sleep_for(50ms);

                    // If currentTime was changed manually, update the point cloud even when paused:
                    if(lastTimeWhileStopped != currentTime){
                        std::vector<std::shared_ptr<OrganizedPointCloud>> pointClouds;

                        int frame = int(currentTime * 30.000001f);

                        for(int i = 0; i < int(streams.size()); ++i){
                            if(!stream1Active && i==0)
                                continue;
                            if(!stream2Active && i==1)
                                continue;
                            if(!stream3Active && i==2)
                                continue;


                            streams[i]->readImage(frame);
                            streams[i]->generatePointCloud();
                            pointClouds.push_back(streams[i]->getCurrentPointCloud());
                        }

                        if(callback)
                            callback(pointClouds);

                        lastTimeWhileStopped = currentTime;
                    }

                    lastFrameTime = high_resolution_clock::now();
                    continue;
                }

                auto startTime = high_resolution_clock::now();

                // Point Clouds of this frame:
                std::vector<std::shared_ptr<OrganizedPointCloud>> pointClouds;

                auto now = high_resolution_clock::now();
                if(allowFrameSkipping){
                    currentTime += float(duration_cast<microseconds>(now - lastFrameTime).count() * 0.000001);
                } else {
                    currentTime += float(1/30.f) + 0.000001f;
                }
                lastFrameTime = now;

                // If playtime exceeds the total time:
                if(currentTime + 0.1f >= 30.f/*streams[0]->getTotalTime()*/){
                    currentTime = 3;
                    streams[0]->readImage(3);

                    if(!loop){
                        isPlaying = false;
                        continue;
                    }
                }
                lastTimeWhileStopped = currentTime;

                int frame = int(currentTime * 30.000001f);

                bool s1Active = stream1Active;
                bool s2Active = stream2Active;
                bool s3Active = stream3Active;

                for(unsigned int i = 0; i < streams.size(); ++i){
                    if(!s1Active && i==0)
                        continue;
                    if(!s2Active && i==1)
                        continue;
                    if(!s3Active && i==2)
                        continue;

                    streams[i]->readImage(frame);
                }

                for(int i = 0; i < int(streams.size()); ++i){
                    if(!s1Active && i==0)
                        continue;
                    if(!s2Active && i==1)
                        continue;
                    if(!s3Active && i==2)
                        continue;

                    streams[i]->generatePointCloud();
                    pointClouds.push_back(streams[i]->getCurrentPointCloud());
                }

                processingTime = (duration_cast<microseconds>(high_resolution_clock::now() - startTime).count()*0.001f) * 0.1f + processingTime * 0.9f;

                if(callback)
                    callback(pointClouds);
            }
        });
    }

    ~BinaryStreamer(){
        shouldStop = true;
        readingThread.join();

        if(Data::instance->cpuGroundTruthPoints != nullptr)
            delete[] Data::instance->cpuGroundTruthPoints;

        if(Data::instance->gpuGroundTruthPoints != nullptr)
            cudaFree(Data::instance->gpuGroundTruthPoints);
    }

    void loadGroundTruthData(std::string path, int startOffset, int totalFrameCount) {
        std::ifstream gtStream(path, std::ios::binary);
        if (!gtStream.good()) {
            std::cout << "Ground Truth Stream could not be opened!" << std::endl;
            return;
        }

        Data::instance->groundTruthFrameCount = totalFrameCount;
        gtStream.read(reinterpret_cast<char*>(&Data::instance->groundTruthSampleCount), sizeof(int));

        // Reserve space on GPU:
        cudaMalloc(&Data::instance->gpuGroundTruthPoints, Data::instance->groundTruthFrameCount * Data::instance->groundTruthSampleCount * sizeof(float4));
        Data::instance->cpuGroundTruthPoints = new float4[Data::instance->groundTruthFrameCount * Data::instance->groundTruthSampleCount];

        float4* emptyFrame = new float4[Data::instance->groundTruthSampleCount];
        std::fill(emptyFrame, emptyFrame + Data::instance->groundTruthSampleCount, float4{0.0f, 0.0f, 0.0f, 0.0f});

        // Fill up the start:
        cudaMemset(Data::instance->gpuGroundTruthPoints, 0, startOffset * Data::instance->groundTruthSampleCount * sizeof(float4));

        // Allocate buffer for reading data
        double* buffer = new double[Data::instance->groundTruthSampleCount * 3];
        float4* frame = new float4[Data::instance->groundTruthSampleCount];

        int frameID = startOffset;
        float timestamp;

        while (frameID < totalFrameCount && gtStream.read(reinterpret_cast<char*>(&timestamp), sizeof(float))) {
            gtStream.read(reinterpret_cast<char*>(buffer), sizeof(double) * 3 * Data::instance->groundTruthSampleCount);

            #pragma omp parallel for
            for (int i = 0; i < Data::instance->groundTruthSampleCount; ++i) {
                frame[i] = float4{float(buffer[i * 3 + 1]) / 100, float(buffer[i * 3 + 2]) / 100, float(buffer[i * 3]) / 100, 1.0f};
            }

            cudaMemcpy(Data::instance->gpuGroundTruthPoints + (frameID * Data::instance->groundTruthSampleCount), frame, Data::instance->groundTruthSampleCount * sizeof(float4), cudaMemcpyHostToDevice);

            frameID++;
        }

        // Fill up the end:
        if (frameID < totalFrameCount) {
            cudaMemset(Data::instance->gpuGroundTruthPoints + (frameID * Data::instance->groundTruthSampleCount), 0, (totalFrameCount - frameID) * Data::instance->groundTruthSampleCount * sizeof(float4));
        }

        cudaMemcpy(Data::instance->cpuGroundTruthPoints, Data::instance->gpuGroundTruthPoints, Data::instance->groundTruthFrameCount * Data::instance->groundTruthSampleCount * sizeof(float4), cudaMemcpyDeviceToHost);

        delete[] frame;
        delete[] emptyFrame;
        delete[] buffer;
    }


    /**
     * Steps a frame forward. If the parameter 'frameDelta' is given, it steps
     * the number of frames forward (or backward, when negative).
     */
    virtual void step(int frameDelta = 1) override {
        int newFrameID = streams[0]->getCurrentFrame() + frameDelta;

        if(newFrameID >= 0 && newFrameID < int(streams[0]->getTotalFrameCount()))
            currentTime = newFrameID / fps;

        streams[0]->readImage(newFrameID);
    };

    /**
     * Returns the number of frames of the point cloud recording for the
     * master depth sensor (idx 0).
     */
    virtual float getTotalTime() override {
        return float(streams[0]->getTotalTime());
    };

    /**
     * Returns the CPU processing time in milliseconds per read frame of
     * sensor 0).
     */
    virtual float getProcessingTime() override {
        return processingTime;
    };
};
