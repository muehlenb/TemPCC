// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)
#pragma once

#include "src/util/math/Mat4.h"

#include "src/util/OrganizedPointCloud.h"

#include <fstream>

/**
 * Represents a Binary Stream from a binary file. This type of file was
 * used for exporting virtual / simulated depth cameras from Unreal Engine 5.
 */
class BinaryStream {
public:
    /**
     * Opens an hdf5 stream from the given file.
     */
    BinaryStream(std::string rgbFile, std::string depthFile, Mat4f modelMatrix = Mat4f(), double syncOffset = 0);

    /**
     * Frees used memory.
     */
    ~BinaryStream();

    /**
     *
     */
    void generatePointCloud();

    /**
     * Reads the image with the given frame number.
     */
    bool readImage(unsigned int frame);

    /**
     * Get total frame count.
     */
    unsigned int getTotalFrameCount(){ return totalFrameCount; }

    /**
     * Returns the current point cloud.
     */
    std::shared_ptr<OrganizedPointCloud> getCurrentPointCloud(){ return currentPointCloud; }

    /**
     * Returns the total time:
     */
    double getTotalTime(){ return totalFrameCount / fps; }

    int getCurrentFrame(){
        return currentFrame;
    }

private:
    /** Is the hdf5 opened successfully? */
    bool isReady = false;

    /** Current loaded point cloud */
    std::shared_ptr<OrganizedPointCloud> currentPointCloud = nullptr;

    /** Current <Depth field> to <Camera Space> map. */
    float* lookup2DTo3D = nullptr;

    /** Is the depth field to camera space map initialized? */
    bool lookupTablesInitialized = false;

    /** Current <Depth field> to <Camera Space> map. */
    float* lookup3DTo2D = nullptr;

    /** Filepath to the file. */
    std::string rgbFilePath;
    std::string depthFilePath;

    /** Width of the image image */
    unsigned int imageWidth;
    /** Height of the image image */
    unsigned int imageHeight;
    /** PixelSize */
    unsigned int pixelSize;

    float halfFovX;
    float halfFovY;

    /** Total frame count of the recording */
    unsigned int totalFrameCount;

    /** Current loaded frame */
    int currentFrame = 0;

    float fps = 30;

    Mat4f loadedModelMatrix;

    double syncOffset = 0;

    uint16_t* currentDepthImage;
    uint8_t* currentColorImage;

    float currentRGBTime;
    std::ifstream rgbStream;
    std::ifstream depthStream;

    /**
     * Converts the raw depth image data by an azure kinect to an
     * organized point cloud.
     */
    void convertRawToOrganizedPointCloud();

    /**
     *
     */
    void createLookupTables();

    /**
     *
     */
    void loadGroundTruthFrames();

    /**
     *
     */
    inline Vec4f Gen3DPointByDepth(int sX, int sY, float z);

    /**
     *
     */
    inline Vec4f InverseGen3DPointByDepth(const Vec4f& point);
};
