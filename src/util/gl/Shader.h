// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

// Include this file only once when compiling:
#pragma once

// Include utility to open files during runtime:
#include <fstream>
#include <iostream>

#include <unordered_map>

// Include string:
#include <string>

// Include Mat4f (and Vec4f):
#include <Mat4.h>

/**
 * This is a wrapper class for shader programs in OpenGL.
 */

class Shader {
public:
    /** Stores the ID of the shader program on the GPU */
    unsigned int shaderProgram;

    /** Is set to true in the constructor if initialization worked: */
    bool initialized = false;

    // Copy counter (This is needed for correct creation and deletion
    // of shaderProgram on GPU, since C++ copies objects on reassignment,
    // see copy constructor. In general, I would recommend holding Mesh,
    // Shader, SpinningTop and World objects by a shared_ptr instead of
    // doing this, because that ensures that exactly one explicitly created
    // mesh object exists, but we don't want to introduce 'Smart Pointers'
    // in Computergrafik 1 - just as a hint if you want to create your
    // own engine):
    int* numOfCopies;

    /** Stores the location of uniform names */
    std::unordered_map<std::string, int>* uniformLocationMap = nullptr;

public:
    /**
     * Loads the source code of the given file paths, compiles it on the
     * GPU and stores the reference (=id, =name) to the program in the
     * attribute 'shaderProgram'.
     *
     * NOTE:
     * OpenGL shaders are usually not compiled when the C++ program
     * is compiled, but only when the program is started. I.e. we have
     * to read in from the source files at this point and then pass it
     * to OpenGL to be compiled on the graphics card.
     *
     * In recent OpenGL versions, it is also possible to compile shaders
     * before hand (so they don't have to be compiled every time you
     * start a game, but that's another topic ;-)).
     */
    Shader(std::string vertexShaderPath, std::string fragmentShaderPath, std::string geometryShader = "");

    /**
     * Explicit copy constructor for reference counting (for correct
     * shaderProgram deletion).
     */
    Shader(const Shader& shader);

    /**
     * Deletes the shader and ensures that the memory on the GPU
     * is cleaned up.
     */
    ~Shader();

    /**
     * Binds the shader so that it will be used in future draw calls.
     */
    void bind();

    /**
     * Sets a uniform variable of type Mat4f.
     */
    void setUniform(std::string name, Mat4f value);

    /**
     * Sets a uniform variable of type Vec4f.
     */
    void setUniform(std::string name, Vec4f value, int num = 4);

    /**
     * Sets a uniform variable of type float.
     */
    void setUniform(std::string name, float value);

    /**
     * Sets a uniform variable of type int.
     */
    void setUniform(std::string name, int value);
};
