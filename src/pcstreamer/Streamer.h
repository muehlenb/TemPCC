// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)
#pragma once

#include <memory>
#include <functional>
#include <vector>

#include "src/util/OrganizedPointCloud.h"

/**
 * Represents a class that streams a pointcloud from file, network or
 * device.
 *
 * Note that when using multiple sensors, it is currently assumed that
 * all sensors use the same frame rate and provide new images together/
 * synchronized. So the Callback is only called when all sensors have a
 * new image.
 */
class Streamer {
protected:
    /**
     * Function that receives a shared ptr with a vector of point clouds when they
     * are available by this streamer.
     */
    std::function<void(std::vector<std::shared_ptr<OrganizedPointCloud>>)> callback;

public:
    /** Registers a callback for new images */
    void setCallback(std::function<void(std::vector<std::shared_ptr<OrganizedPointCloud>>)> cb){
        callback = cb;
    }

    /**
     * Returns an array with names of available pc streamers.
     */
    static const char* availableStreamerNames[];

    /**
     * Number of available pc streamers.
     */
    static const unsigned int availableStreamerNum;

    /**
     * Constructs a PCStreamer object with a streamer of the given type
     * (which corresponds to the index of getAvailableNames()).
     */
    static std::shared_ptr<Streamer> constructStreamerInstance(int type, std::string path);

    /**
     * Returns the CPU processing time in milliseconds per read frame of
     * sensor 0).
     */
    virtual float getProcessingTime() = 0;
};

/**
 * Abstract PCStreamer class that is used when streaming a point cloud
 * recording from file. It offers different functions for start/stop the
 * stream but also to get the length and to jump to a specific position
 * in time.
 */
class FileStreamer : public Streamer {
public:
    /**
     * Is currently playing?
     */
    bool isPlaying = true;

    /**
     * Current time of playing.
     */
    float currentTime = 0.0;

    /**
     * Should the recording be looped? (or stop at the end)
     */
    bool loop = true;

    /**
     * Should frame skipping be allowed to enforce realtime?
     *
     * If false, every available frame is read one by one without skipping even if
     * playing slows down.
     */
    bool allowFrameSkipping = false;

    /**
     * Steps a frame forward. If the parameter 'frameDelta' is given, it steps
     * the number of frames forward (or backward, when negative).
     */
    virtual void step(int frameDelta = 1) = 0;

    /**
     * Returns the total number of frames of the point cloud recording for the
     * master depth sensor (idx 0).
     */
    virtual float getTotalTime() = 0;
};
