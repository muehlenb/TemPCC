// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#pragma once

#include <map>
#include <cuda_runtime.h>

#include "src/util/math/Mat4.h"
#include "src/util/gl/primitive/TexCoord.h"

// Include OpenGL3.3 Core functions:
#include <glad/glad.h>

/**
 * Represents an organized point cloud ordered by the original camera image
 * (Row major order).
 *
 * Note that this organized point cloud
 */
struct OrganizedPointCloud {
private:
    /**
     * Stores all (statically) lookup3DToImage-pointers that were uploaded to gpu for use with CUDA.
     * Since they are static for each camera (pointer of lookup3DToImage doesn't change), it is not
     * wanted to upload them every frame for performance reasons.
     *
     * Note: Here, we have a memory leak (the GPU memory of these pointers are not deleted) - this
     * deletion could be implemented in the PCStreamer classes.
     */
    static std::map<float*, float2*> uploadedLookup3DToImageTables;
    static std::map<float*, float2*> uploadedLookupImageTo3DTables;

    /** Defines a piece of GPU memory */
    struct GPUMemory {
        size_t size;
        void* pointer;

        GPUMemory(size_t size, void* pointer)
            : size(size)
            , pointer(pointer)
        {}
    };

    /**
     * For performance reasons, we want to reuse gpu / shared memory instead recreating it all the
     * time. This is less a problem because the most time, we need a memory block of the same size
     * again (e.g. in the next frame for the next created organized point cloud.
     */
    void initializeMemory(void** pointer, size_t size){
        for(unsigned int i = 0; i < unusedInitializedGPUMemory.size(); ++i){
            if(unusedInitializedGPUMemory[i].size == size){
                *pointer = unusedInitializedGPUMemory[i].pointer;
                unusedInitializedGPUMemory.erase(unusedInitializedGPUMemory.begin() + i);
                return;
            }
        }

        cudaMalloc(pointer, size);
    }

    /** Defines initialized GPU memory which is currently not in use */
    static std::vector<GPUMemory> unusedInitializedGPUMemory;

public:
    OrganizedPointCloud(unsigned int width, unsigned int height)
        : width(width)
        , height(height){};

    /** Stores the xyzw coordiantes of all points */
    Vec4f* positions = nullptr;

    /** Stores the rgba coordinates of all points */
    Vec4b* colors = nullptr;

    /** Stores the normal of all points */
    Vec4f* normals = nullptr;

    /** Texture coordinates */
    TexCoord* texCoords = nullptr;

    /**
     *
     */
    int frameID = -1;

    /**
     * 3D-to-image Lookup Table (memory is managed by READER to avoid copying for every point cloud
     * since the values doesn't change between different images!)
     */
    float* lookup3DToImage = nullptr;

    /**
     * 3D-to-image Lookup Table (memory is managed by READER to avoid copying for every point cloud
     * since the values doesn't change between different images!)
     */
    float* lookupImageTo3D = nullptr;

    /** Stores the xyzw coordiantes of all points */
    float4* gpuPositions = nullptr;

    /** Stores the rgba coordinates of all points */
    uchar4* gpuColors = nullptr;

    /** Stores the normal of all points */
    float4* gpuNormals = nullptr;

    /** Texture coordinates */
    float2* gpuTexCoords = nullptr;

    float2* gpuLookup3DToImage = nullptr;

    float2* gpuLookupImageTo3D = nullptr;

    /** High resolution colors */
    unsigned int highResWidth = 0;
    unsigned int highResHeight = 0;
    Vec4b* highResColors = 0;

    /**
     * 3D-to-image Lookup Table Size for one dimension (which results in a square
     * resolution)
     */
    unsigned int lookup3DToImageSize = 0;

    /** Transforms the point cloud positions from camera space into world space */
    Mat4f modelMatrix;

    /** Width of the original depth image */
    unsigned int width = 0;

    /** Height of the original depth iamge */
    unsigned int height = 0;

    /**
     * Usually not used / needed for processing, implemented for writing synchronized
     * training data
     */
    float timestamp = -1;

    /**
     * Indicates whether the last change was performed on GPU (or CPU) and whether the
     * the (positions, colors, ...) pointer are up to date or whether the (gpuPositions,
     * gpuColors, ...) pointer are up to date.
     */
    bool gpu = false;

    /**
     * Initializes the gpu... variables (if nessecary) and copies the content from the
     * corresponding position, colors, ... arrays.
     */

    void toGPU(){
        if(gpu)
            return;

        if(colors != nullptr){
            if(gpuColors == nullptr)
                initializeMemory((void**)&gpuColors, width * height * sizeof(uchar4));
            cudaMemcpy(gpuColors, colors, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);
        }

        if(positions != nullptr){
            if(gpuPositions == nullptr)
                initializeMemory((void**)&gpuPositions, width * height * sizeof(float4));
            cudaMemcpy(gpuPositions, positions, width * height * sizeof(float4), cudaMemcpyHostToDevice);
        }

        if(normals != nullptr){
            if(gpuNormals == nullptr)
                initializeMemory((void**)&gpuNormals, width * height * sizeof(float4));
            cudaMemcpy(gpuNormals, normals, width * height * sizeof(float4), cudaMemcpyHostToDevice);
        }

        if(texCoords != nullptr){
            if(gpuTexCoords == nullptr)
                initializeMemory((void**)&gpuTexCoords, width * height * sizeof(float2));
            cudaMemcpy(gpuTexCoords, gpuTexCoords, width * height * sizeof(float2), cudaMemcpyHostToDevice);
        }

        if(lookup3DToImage != nullptr){
            // If there is a key in the map:
            if (uploadedLookup3DToImageTables.find(lookup3DToImage) != uploadedLookup3DToImageTables.end()) {
                gpuLookup3DToImage = uploadedLookup3DToImageTables[lookup3DToImage];
            } else {
                cudaMalloc(&gpuLookup3DToImage, lookup3DToImageSize * lookup3DToImageSize * sizeof(float) * 2);
                cudaMemcpy(gpuLookup3DToImage, lookup3DToImage, lookup3DToImageSize * lookup3DToImageSize * sizeof(float) * 2, cudaMemcpyHostToDevice);
                uploadedLookup3DToImageTables[lookup3DToImage] = gpuLookup3DToImage;
            }
        }

        if(lookupImageTo3D != nullptr){
            // If there is a key in the map:
            if (uploadedLookupImageTo3DTables.find(lookupImageTo3D) != uploadedLookupImageTo3DTables.end()) {
                gpuLookupImageTo3D = uploadedLookupImageTo3DTables[lookupImageTo3D];
            } else {
                cudaMalloc(&gpuLookupImageTo3D, width * height * sizeof(float) * 2);
                cudaMemcpy(gpuLookupImageTo3D, lookupImageTo3D, width * height * sizeof(float) * 2, cudaMemcpyHostToDevice);
                uploadedLookupImageTo3DTables[lookupImageTo3D] = gpuLookupImageTo3D;
            }
        }

        gpu = true;
    }

    /**
     * Copies the data back to the cpu (gpuPositions -> positions, gpuColors -> colors, ...).
     */

    void toCPU(){
        if(!gpu)
            return;

        if(colors != nullptr){
            cudaError_t err = cudaMemcpy(colors, gpuColors, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "GPU->CPU copy error in pc->colors: %s\n", cudaGetErrorString(err));
            }
        }

        if(positions != nullptr){
            cudaError_t err = cudaMemcpy(positions, gpuPositions, width * height * sizeof(float4), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "GPU->CPU copy error in pc->positions:  %s\n", cudaGetErrorString(err));
            }
        }

        if(normals != nullptr)
            cudaMemcpy(normals, gpuNormals, width * height * sizeof(float4), cudaMemcpyDeviceToHost);

        if(texCoords != nullptr)
            cudaMemcpy(texCoords, gpuTexCoords, width * height * sizeof(float2), cudaMemcpyDeviceToHost);

        gpu = false;
    }

    ~OrganizedPointCloud(){
        if(gpuPositions != nullptr){
            unusedInitializedGPUMemory.emplace_back(sizeof(Vec4f) * width * height, gpuPositions);
            gpuPositions = nullptr;
        }

        if(gpuColors != nullptr){
            unusedInitializedGPUMemory.emplace_back(sizeof(Vec4b) * width * height, gpuColors);
            gpuColors = nullptr;
        }

        if(gpuNormals != nullptr){
            unusedInitializedGPUMemory.emplace_back(sizeof(Vec4f) * width * height, gpuNormals);
            gpuNormals = nullptr;
        }

        if(gpuTexCoords != nullptr){
            unusedInitializedGPUMemory.emplace_back(sizeof(TexCoord) * width * height, gpuTexCoords);
            gpuTexCoords = nullptr;
        }

        if(positions != nullptr){
            delete[] positions;
            positions = nullptr;
        }

        if(colors != nullptr){
            delete[] colors;
            colors = nullptr;
        }

        if(normals != nullptr){
            delete[] normals;
            normals = nullptr;
        }

        if(texCoords != nullptr){
            delete[] texCoords;
            texCoords = nullptr;
        }

        if(highResColors != nullptr){
            delete[] highResColors;
            highResColors = nullptr;
        }
    }

    static void cleanupStaticMemory();
};
