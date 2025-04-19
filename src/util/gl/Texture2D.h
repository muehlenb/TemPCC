// © 2022, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

// Include this file only once when compiling:
#pragma once

#include <string>

/**
 * This struct defines the type of a texture, e.g. RGB Texture with byte precision,
 * without defining it's resolution.
 */

struct TextureType {
    int internalFormat;
    unsigned int format;
    unsigned int type;

    TextureType(int internalFormat, unsigned int format, unsigned int type)
        : internalFormat(internalFormat)
        , format(format)
        , type(type){}
};

/**
 * This class represents a texture which is stored on the GPU.
 */

class Texture2D {
public:
    /** Stores the ID of the texture on the GPU */
    unsigned int texture = 0;

    /** Is set to true in the constructor if texture was loaded */
    bool isLoaded = false;

    // Copy counter (This is needed for correct creation and deletion
    // of shaderProgram on GPU, since C++ copies objects on reassignment,
    // see copy constructor. In general, I would recommend holding Mesh,
    // Shader, Texture2D, etc. objects by a shared_ptr instead of
    // doing this, because that ensures that exactly one explicitly created
    // mesh object exists, but we don't want to introduce 'Smart Pointers'
    // in Computergrafik 1 - just as a hint if you want to create your
    // own engine):
    int* numOfCopies;

public:
    /**
     * Creates a dummy texture. When binding, the texture name 0 will be
     * bound which will deactivate the texture.
     */
    Texture2D();

    /**
     * Creates an empty texture of the given type and dimensions.
     */
    Texture2D(TextureType type, unsigned int width, unsigned int height);

    /**
     * Explicit copy constructor for reference counting (for correct
     * texture deletion on GPU).
     */
    Texture2D(const Texture2D& texture);

    /**
     * Loads a 2D image from the given file path (1) into the RAM
     * and (2) uploads it to the GPU (Video-RAM) as an image texture.
     *
     * Here, an GL_RGBA texture is used with unsigned byte precision.
     */
    Texture2D(std::string imageTexture);

    /**
     * Deletes the texture and ensures that the memory on the GPU
     * is cleaned up.
     */
    ~Texture2D();

    /**
     * Binds the texture so that it can be used in the shader.
     *
     * @textureSlot     Defines the slot where the texture should
     *                  be bound to.
     */
    void bind(int textureSlot = 0);

    /**
     * Returns true if no real texture data / texture name exists.
     */
    bool isDummy(){return texture == 0;}
};
