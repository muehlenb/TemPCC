// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)
#pragma once

#include <k4a/k4a.h>
#include <k4arecord/playback.h>

// Include OpenGL3.3 Core functions:
#include <glad/glad.h>

#define LOOKUP_TABLE_SIZE 1024

/**
 * A streamer when using a single or multiple Azure Kinect devices:
 */
class AzureKinectMKVStream {
    std::vector<double> allTimestamps;

    uint64_t totalFrameCount = 0;
    int currentFrame = 0;

    Mat4f transformation;
    Mat4f depthToColorTransform;

    k4a_playback_t playback_handle = nullptr;
    k4a_record_configuration_t record_config;
    k4a_calibration_t calibration;
    k4a_transformation_t transformation_handle;

    bool successfullyOpened = false;

    float* DFToCS = nullptr;
    float* lookupTable3DToImage = nullptr;

public:
    AzureKinectMKVStream(std::string filepath, Mat4f transformation)
        : transformation(transformation)
    {
        if (k4a_playback_open(filepath.c_str(), &playback_handle) != K4A_RESULT_SUCCEEDED)
        {
            std::cerr << "Failed to open recording\n";
            return;
        }

        if (k4a_playback_get_record_configuration(playback_handle, &record_config) != K4A_RESULT_SUCCEEDED)
        {
            std::cerr << "Failed to get record configuration\n";
            k4a_playback_close(playback_handle);
            return;
        }

        if (k4a_playback_get_calibration(playback_handle, &calibration) != K4A_RESULT_SUCCEEDED)
        {
            std::cerr << "Failed to get calibration\n";
            k4a_playback_close(playback_handle);
            return;
        }

        // Get depth to color transform:
        {
            k4a_calibration_extrinsics_t extr = calibration.extrinsics[0][1];
            float *rotation = extr.rotation;
            float *translation = extr.translation;
            float extrMat[16] = {
                rotation[0], rotation[3], rotation[6], 0,
                rotation[1], rotation[4], rotation[7], 0,
                rotation[2], rotation[5], rotation[8], 0,
                translation[0]/1000.f, translation[1]/1000.f, translation[2]/1000.f, 1
            };
            depthToColorTransform = Mat4f(extrMat);
        }

        transformation_handle = k4a_transformation_create(&calibration);

        std::vector<double> timestamps;
        k4a_capture_t capture = NULL;
        while (k4a_playback_get_next_capture(playback_handle, &capture) == K4A_STREAM_RESULT_SUCCEEDED)
        {

            k4a_image_t depth_image = k4a_capture_get_depth_image(capture);
            if (depth_image != NULL) {
                // Erhalte den Timestamp des Tiefenbildes
                uint64_t timestamp_usec = k4a_image_get_device_timestamp_usec(depth_image);

                timestamps.push_back(timestamp_usec / 1000000.0);

                // Ressourcen freigeben
                k4a_image_release(depth_image);
            } else {
                //std::cerr << "No depth image available in this capture." << std::endl;
            }

            k4a_capture_release(capture); // Wichtig, um Speicherlecks zu vermeiden
        }

        allTimestamps = timestamps;
        totalFrameCount = timestamps.size();

        std::cout << "Total Frame Count: " << totalFrameCount << " | Start TS: " << allTimestamps[0] << std::endl;

        k4a_result_t seek_result = k4a_playback_seek_timestamp(playback_handle, 0, K4A_PLAYBACK_SEEK_BEGIN);
        if (seek_result != K4A_RESULT_SUCCEEDED)
        {
            std::cerr << "Failed to seek to beginning of the recording\n";
        }

        k4a_playback_set_color_conversion(playback_handle, K4A_IMAGE_FORMAT_COLOR_BGRA32);

        createLookupTables();

        successfullyOpened = true;
    }

    ~AzureKinectMKVStream(){
        k4a_transformation_destroy(transformation_handle);
        k4a_playback_close(playback_handle);
        delete[] DFToCS;
        delete[] lookupTable3DToImage;
    }

    int getCurrentFrame(){
        return currentFrame;
    }

    int getTotalFrameCount(){
        return int(totalFrameCount);
    }

    std::vector<double>& getAllTimestamps(){
        return allTimestamps;
    }

    float getTotalTime(){
        return float(allTimestamps[totalFrameCount-1] - allTimestamps[0]);
    }

    double getTimeDeltaToNextFrame() {
        if(currentFrame + 1 >= int(totalFrameCount))
            return 0.f;

        return allTimestamps[currentFrame + 1] - allTimestamps[currentFrame];
    }

    std::shared_ptr<OrganizedPointCloud> readImage(unsigned int frame){
        if(frame < 0 || frame >= totalFrameCount)
            return nullptr;

        return syncImage(allTimestamps[frame]);
    }

    double getTimeStampAtFrame(int frame){
        return allTimestamps[frame];
    }


    void print_image_format(k4a_image_t image) {
        k4a_image_format_t format = k4a_image_get_format(image);

        switch (format) {
        case K4A_IMAGE_FORMAT_COLOR_MJPG:
            std::cout << "Image format: MJPEG" << std::endl;
            break;
        case K4A_IMAGE_FORMAT_COLOR_NV12:
            std::cout << "Image format: NV12" << std::endl;
            break;
        case K4A_IMAGE_FORMAT_COLOR_YUY2:
            std::cout << "Image format: YUY2" << std::endl;
            break;
        case K4A_IMAGE_FORMAT_COLOR_BGRA32:
            std::cout << "Image format: BGRA32" << std::endl;
            break;
        case K4A_IMAGE_FORMAT_DEPTH16:
            std::cout << "Image format: Depth 16" << std::endl;
            break;
        case K4A_IMAGE_FORMAT_IR16:
            std::cout << "Image format: IR 16" << std::endl;
            break;
        case K4A_IMAGE_FORMAT_CUSTOM:
            std::cout << "Image format: Custom" << std::endl;
            break;
        default:
            std::cout << "Unknown image format" << std::endl;
            break;
        }
    }

    void createLookupTables(){
        lookupTable3DToImage = new float[LOOKUP_TABLE_SIZE * LOOKUP_TABLE_SIZE * 2];

        for(unsigned int y = 0; y < LOOKUP_TABLE_SIZE; ++y){
            for(unsigned int x = 0; x < LOOKUP_TABLE_SIZE; ++x){
                k4a_float3_t p;
                k4a_float2_t img;

                p.xyz.x = ((x / float(LOOKUP_TABLE_SIZE)) * 2 - 1) * 1000;
                p.xyz.y = ((y / float(LOOKUP_TABLE_SIZE)) * 2 - 1) * 1000;
                p.xyz.z = 1000;

                int valid;
                k4a_calibration_3d_to_2d(&calibration, &p, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &img, &valid);
                if (valid == 1) {
                    float relImgX = img.xy.x / float(640);
                    float relImgY = img.xy.y / float(576);

                    if(relImgX >= 0 && relImgX <= 1 && relImgY >= 0 && relImgY <= 1){
                        lookupTable3DToImage[(x + y * LOOKUP_TABLE_SIZE) * 2] = relImgX;
                        lookupTable3DToImage[(x + y * LOOKUP_TABLE_SIZE) * 2 + 1] = relImgY;
                    } else {
                        lookupTable3DToImage[(x + y * LOOKUP_TABLE_SIZE) * 2] = -1;
                        lookupTable3DToImage[(x + y * LOOKUP_TABLE_SIZE) * 2 + 1] = -1;
                    }
                } else {
                    lookupTable3DToImage[(x + y * LOOKUP_TABLE_SIZE) * 2] = -1;
                    lookupTable3DToImage[(x + y * LOOKUP_TABLE_SIZE) * 2 + 1] = -1;
                }
            }
        }

        k4a_float2_t p;
        k4a_float3_t ray;

        DFToCS = new float[640 * 576 * 2];

        for (unsigned int y = 0, idx = 0; y < 576; y++)
        {
            p.xy.y = (float)y;

            for (unsigned int x = 0; x < 640; x++, idx++)
            {
                p.xy.x = (float)x;

                int valid;
                k4a_calibration_2d_to_3d(&calibration, &p, 1.f, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &ray, &valid);
                if (valid) {
                    DFToCS[idx * 2] = ray.xyz.x;
                    DFToCS[idx * 2 + 1] = ray.xyz.y;
                }
                else {
                    DFToCS[idx * 2] = nanf("");
                    DFToCS[idx * 2 + 1] = nanf("");
                }
            }
        }
    }


    std::shared_ptr<OrganizedPointCloud> syncImage(double timeStamp){
        k4a_result_t seek_result = k4a_playback_seek_timestamp(playback_handle, uint64_t(timeStamp * 1000000), K4A_PLAYBACK_SEEK_BEGIN);
        if (seek_result != K4A_RESULT_SUCCEEDED)
        {
            return nullptr;
        }

        for(int i=0; i < allTimestamps.size(); ++i){
            if(timeStamp - 0.016f < allTimestamps[i]){
                currentFrame = i;
                break;
            }
        }

        k4a_capture_t capture = NULL;
        if (k4a_playback_get_next_capture(playback_handle, &capture) == K4A_STREAM_RESULT_SUCCEEDED) {
            k4a_image_t depth_image = k4a_capture_get_depth_image(capture);
            k4a_image_t color_image = k4a_capture_get_color_image(capture);
            k4a_image_t transformed_color_image;

            if (depth_image != nullptr && color_image != nullptr) {
                int width = k4a_image_get_width_pixels(depth_image);
                int height = k4a_image_get_height_pixels(depth_image);

                std::shared_ptr<OrganizedPointCloud> pc = std::make_shared<OrganizedPointCloud>(width, height);

                k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32, width, height, width * 4 * sizeof(uint8_t), &transformed_color_image);
                if (k4a_transformation_color_image_to_depth_camera(transformation_handle, depth_image, color_image, transformed_color_image) != K4A_RESULT_SUCCEEDED)
                {
                    std::cout << "A color image could not be transformed to the depth image." << std::endl;

                    k4a_image_release(depth_image);
                    k4a_image_release(transformed_color_image);
                    k4a_image_release(color_image);
                    k4a_capture_release(capture);
                    return nullptr;
                }

                k4a_image_t pcimg;
                if (k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM, width, height, width * 3 * (int)sizeof(int16_t),&pcimg) != K4A_RESULT_SUCCEEDED){
                    std::cout << "A depth image could not be transformed into a point cloud." << std::endl;
                    k4a_image_release(depth_image);
                    k4a_image_release(transformed_color_image);
                    k4a_image_release(color_image);
                    k4a_capture_release(capture);
                    return nullptr;
                }

                if (k4a_transformation_depth_image_to_point_cloud(transformation_handle, depth_image, K4A_CALIBRATION_TYPE_DEPTH, pcimg) != K4A_RESULT_SUCCEEDED) {
                    std::cout << "A depth image could not be transformed into a point cloud." << std::endl;
                    k4a_image_release(depth_image);
                    k4a_image_release(transformed_color_image);
                    k4a_image_release(color_image);
                    k4a_capture_release(capture);
                    return nullptr;
                }

                pc->positions = new Vec4f[width * height];
                pc->colors = new Vec4b[width * height];
                pc->modelMatrix = transformation * depthToColorTransform;
                pc->lookupImageTo3D = DFToCS;
                pc->lookup3DToImage = lookupTable3DToImage;
                pc->lookup3DToImageSize = LOOKUP_TABLE_SIZE;
                pc->frameID = currentFrame;
                pc->width = width;
                pc->height = height;

                int16_t* pcdata = (int16_t*)(void*)k4a_image_get_buffer(pcimg);
                uint8_t* bgradata = k4a_image_get_buffer(transformed_color_image);
                for(int y = 0; y < height; ++y){
                    for(int x = 0; x < width; ++x){
                        int idx = y * width + x;
                        pc->positions[idx] = Vec4f(pcdata[3 * idx] / 1000.f, pcdata[3 * idx + 1] / 1000.f, pcdata[3 * idx + 2] / 1000.f, 1.f);
                    }
                }

                std::memcpy(pc->colors, bgradata, width * height * sizeof(int));

                k4a_image_release(depth_image);
                k4a_image_release(transformed_color_image);
                k4a_image_release(color_image);
                k4a_capture_release(capture);
                return pc;
            }
        }
        return nullptr;
    }
};
