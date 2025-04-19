// © 2024, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#pragma once

#include "src/util/gl/GLMesh.h"
#include "src/util/gl/Shader.h"

/**
 * Represents a renderable mesh.
 */
class GLCoordinateSystem
{
private:
    GLMesh xAxis = GLMesh(CMAKE_SOURCE_DIR "/data/model/coordinate_system_x_axis.obj");
    GLMesh yAxis = GLMesh(CMAKE_SOURCE_DIR "/data/model/coordinate_system_y_axis.obj");
    GLMesh zAxis = GLMesh(CMAKE_SOURCE_DIR "/data/model/coordinate_system_z_axis.obj");
    Shader shader =  Shader(CMAKE_SOURCE_DIR "/shader/singleColorShader.vert", CMAKE_SOURCE_DIR "/shader/singleColorShader.frag");

public:
    void render(Mat4f projection, Mat4f view, Mat4f model){
        shader.bind();
        shader.setUniform("projection", projection);
        shader.setUniform("view", view);
        shader.setUniform("model", model);

        shader.setUniform("color", Vec4f(1.f, 0.f, 0.f));
        xAxis.render();

        shader.setUniform("color", Vec4f(0.f, 1.f, 0.f));
        yAxis.render();

        shader.setUniform("color", Vec4f(0.f, 0.f, 1.f));
        zAxis.render();
    }
};
