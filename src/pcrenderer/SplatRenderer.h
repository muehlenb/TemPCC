// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)
#pragma once

#include "src/pcrenderer/Renderer.h"

#include "src/util/gl/Shader.h"
#include "src/util/gl/GLMesh.h"

#include <chrono>

using namespace std::chrono;

class SplatRenderer : public Renderer {
    std::vector<std::shared_ptr<OrganizedPointCloud>> currentPointClouds;

    // Splat shader:
    Shader splatShader = Shader(CMAKE_SOURCE_DIR "/shader/splats.vert", CMAKE_SOURCE_DIR "/shader/splats.frag");
    Shader singleColorShader = Shader(CMAKE_SOURCE_DIR "/shader/singleColorShader.vert", CMAKE_SOURCE_DIR "/shader/singleColorShader.frag");

    // Pointer to Vertex Array Object (on GPU):
    GLuint vao = 0;

    // Pointer to vertex buffer of points:
    GLuint vbo_positions = 0;

    unsigned int bufferWidth = -1;
    unsigned int bufferHeight = -1;

    GLuint texture_positions = 0;
    GLuint texture_colors = 0;

public:
    float uploadTime = 0;

    // Should black pixels be discarded:
    bool discardBlackPixels = true;

    // Point size of splats:
    float pointSize = 3.f;

    SplatRenderer(){
        // Generate buffers for locations, colors:
        glGenBuffers(1, &vbo_positions);
        glGenVertexArrays(1, &vao);

        glGenTextures(1, &texture_positions);
        glGenTextures(1, &texture_colors);
    }

    ~SplatRenderer(){
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo_positions);

        glDeleteTextures(1, &texture_positions);
        glDeleteTextures(1, &texture_colors);
    }

    /**
     * Integrate new RGB XYZ images.
     */
    virtual void integratePointClouds(std::vector<std::shared_ptr<OrganizedPointCloud>> pointClouds) override {
        // This renderer accesses the point cloud in CPU-memory, so we need to ensure that it is there:
        for(std::shared_ptr<OrganizedPointCloud> pc : pointClouds){
            pc->toCPU();
        }

        currentPointClouds = pointClouds;

    };

    /**
     * Renders the point cloud
     */
    virtual void render(Mat4f projection, Mat4f view) override {
        for(std::shared_ptr<OrganizedPointCloud> pc : currentPointClouds){
            if(pc == nullptr || pc->width == 0 || pc->height == 0)
                continue;

            glPointSize(pointSize);
            int pointNum = pc->width * pc->height;

            // Update the buffer if the size changes:
            if(pc->width != bufferWidth || pc->height != bufferHeight){
                glBindVertexArray(vao);
                glBindBuffer(GL_ARRAY_BUFFER, vbo_positions);
                glBufferData(GL_ARRAY_BUFFER, pointNum * sizeof(Vec4f), &pc->positions[0], GL_DYNAMIC_DRAW);

                glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Vec4f), 0);
                glEnableVertexAttribArray(0);

                glBindBuffer(GL_ARRAY_BUFFER, 0);
                glBindVertexArray(0);

                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, texture_positions);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, pc->width, pc->height, 0, GL_RGBA, GL_FLOAT, &pc->positions[0]);

                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, texture_colors);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pc->width, pc->height, 0, GL_RGBA,  GL_UNSIGNED_BYTE, &pc->colors[0]);

                bufferWidth = pc->width;
                bufferHeight = pc->height;
            }
            splatShader.bind();
            splatShader.setUniform("projection", projection);
            splatShader.setUniform("view", view);
            splatShader.setUniform("model", pc->modelMatrix);
            splatShader.setUniform("discardBlackPixels", discardBlackPixels);
            splatShader.setUniform("positionTexture", 0);
            splatShader.setUniform("colorTexture", 1);

            auto time = high_resolution_clock::now();

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture_positions);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pc->width, pc->height, GL_RGBA, GL_FLOAT, &pc->positions[0]);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texture_colors);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pc->width, pc->height, GL_RGBA, GL_UNSIGNED_BYTE, &pc->colors[0]);

            glFlush();
            auto time2 = high_resolution_clock::now();
            uploadTime = duration_cast<microseconds>(time2 - time).count() / 1000.f;


            // Draw Buffer:
            glBindVertexArray(vao);
            glDrawArrays(GL_POINTS, 0, pointNum);
            glBindVertexArray(0);
        }
    };
};
