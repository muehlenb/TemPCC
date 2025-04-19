#include "GLMesh.h"

#include <iostream>

// We use the tiny_obj_loader library to load data of .obj-files into this mesh:
#define TINYOBJLOADER_IMPLEMENTATION

// We want to use triangulation when loading non-triangulated polygons:
#define TINYOBJLOADER_USE_MAPBOX_EARCUT

// Include tiny_obj_loader:
#include "tiny_obj_loader.h"


GLMesh::GLMesh(std::vector<Triangle> triangles, bool initGL)
	: triangles(triangles)
	, numOfCopies(new int(1))
{
	if(initGL){
		createGLBuffers();
	}
}

void GLMesh::createGLBuffers(){
    glGenBuffers(1, &vbo);

    verticesNum = (unsigned int)triangles.size() * 3;
	
	glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, triangles.size() * sizeof(Triangle), &triangles[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)sizeof(Vec4f));
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2*sizeof(Vec4f)));
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3*sizeof(Vec4f)));
	
	for(int i=0;i<4;++i)
		glEnableVertexAttribArray(i);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

/* This contructor loads an obj file into the VAO and VBO (see header) */
GLMesh::GLMesh(std::string filepath)
    : numOfCopies(new int(1))
{
    // Define buffers to load the data:
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    // Try to load the obj file into
    if(!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str())){
        std::cerr << "Obj-file " <<filepath << " could not be loaded. Check if file is available." << std::endl;
    }

    //if (!warn.empty()) {
    //    std::cout << "Warning while loaded obj-file " << filepath << ":" << std::endl;
    //    std::cout << warn << std::endl;
    //}

    if (!err.empty()) {
        std::cout << "Error while loaded obj-file " << filepath << ":" << std::endl;
        std::cerr << err << std::endl;
    }

    // Loop over shapes of the obj:
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces of that shapes:
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];

            // Create a triangle for our mesh:
            Triangle triangle;

            // Loop over vertices in the face.
            for (int v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                // Copy vertex position into triangle:
                std::memcpy(&triangle[v].position, &attrib.vertices[3*size_t(idx.vertex_index)+0], sizeof(float) * 3);

                // Copy vertex texcoord into triangle if available:
                if(idx.texcoord_index >= 0)
                    std::memcpy(&triangle[v].uv, &attrib.texcoords[2*size_t(idx.texcoord_index)+0], sizeof(float) * 2);

                // Copy vertex color into triangle:
                std::memcpy(&triangle[v].color, &attrib.colors[3*size_t(idx.vertex_index)+0], sizeof(float) * 3);
            }

            // Generate normals:
            {
                Vec4f normal = (triangle[1].position - triangle[0].position).cross(triangle[2].position - triangle[0].position).normalized();
                for(int v = 0; v < fv; ++v){
                    triangle[v].normal = normal;
                }
            }

            index_offset += fv;

            triangles.push_back(triangle);
        }
    }

    // This method creates VAO and VBOs like you have it done
    // in the previous exercise:
    createGLBuffers();
}

/**
 * This is an explicit copy constructor which is automatically
 * called every time an existing Mesh is assigned to a variable
 * via the = Operator. Not important for CG1 and C++ specific,
 * but it is needed here to avoid problems because of early
 * deletion of the shaderProgram when the destructor is called.
 */
GLMesh::GLMesh(const GLMesh& mesh){
    vao = mesh.vao;
    vbo = mesh.vbo;
    verticesNum = mesh.verticesNum;
    triangles = mesh.triangles;
    numOfCopies = mesh.numOfCopies;
    ++(*numOfCopies);
}

GLMesh::~GLMesh(){
    // We only want the VAO and VBO to be destroyed
    // when the last copy of the mesh is deleted:
    if(--(*numOfCopies) > 0)
        return;

    // If there is a vao, delete it from the GPU:
    if(vao != 0)
        glDeleteVertexArrays(1, &vao);

    // Same for the vbo:
    if(vbo != 0)
        glDeleteBuffers(1, &vbo);

    delete numOfCopies;
}

void GLMesh::render(){
    // Draws the geometry data that is defined by the vao:
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glDrawArrays(GL_TRIANGLES, 0, verticesNum);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

std::vector<Triangle>& GLMesh::getTriangles(){
    return triangles;
}
