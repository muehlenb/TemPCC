// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

// Include this file only once when compiling:
#pragma once

// Include the std::vector class, which is comparable to the
// ArrayList in Java (this is not a mathematical vector!):
#include <vector>

// Include OpenGL3.3 Core functions:
#include <glad/glad.h>

// Include our Triangle primitive:
#include "src/util/gl/primitive/Triangle.h"

// Include renderable abstract class:
#include "GLRenderable.h"

/**
 * Represents a renderable mesh.
 */
class GLMesh : GLRenderable
{
    // Pointer to Vertex Array Object (on GPU):
    GLuint vao = 0;

    // Pointer to vertex buffer (interleaved):
    GLuint vbo = 0;

    // Number of vertices:
    unsigned int verticesNum = 0;

    // Store the triangles:
    std::vector<Triangle> triangles;

    // Copy counter (This is needed for correct creation and deletion
    // of VAO/VBOs, since C++ copies objects on reassignment, see copy
    // constructor. In general, I would recommend holding Mesh, Shader,
    // SpinningTop and World objects by a shared_ptr instead of doing
    // this, because that ensures that exactly one explicitly created
    // mesh object exists, but we don't want to introduce 'Smart Pointers'
    // in Computergrafik 1 - just as a hint if you want to create your
    // own engine):
    int* numOfCopies;

public:
    /**
     * Constructs a mesh which consists of the given vertices
     * and prepare for rendering (e.g. upload the data to the GPU).
     */
    GLMesh(std::vector<Triangle> triangles, bool initGL = false);

    /**
     * Loads the geometry data of an .obj file into this mesh object.
     */
    GLMesh(std::string filepath);

    /**
     * Explicit copy constructor for reference counting (for correct
     * VBO/VAO deletion).
     */
    GLMesh(const GLMesh& mesh);

    /**
     * Deletes the mesh and ensures that the memory on the GPU
     * is cleaned up when no copy exists anymore.
     */
    ~GLMesh();

    /**
     * Renders the mesh.
     */
    virtual void render();

    /**
     * Returns a list of all triangles:
     */
    std::vector<Triangle>& getTriangles();

private:
    /**
     * Creates buffers (VAO,VBO,etc.) at the GPU and uploads triangle
     * data. In previous frameworks, this was done directly in the
     * constructor, but it was splitted to be use with different
     * constructors.
     */
    void createGLBuffers();
};
