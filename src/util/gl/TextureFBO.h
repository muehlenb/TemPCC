// © 2022, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

// Include this file only once when compiling:
#pragma once

#include "Texture2D.h"

#include <vector>
#include <string>

/**
 * This class holds a framebuffer object with an arbitrary number
 * of textures where is rendered to.
 */

class TextureFBO {
private:
    /** The width of all textures (should be the screen width) */
    unsigned int width = 0;

    /** The height of all textures (should be the screen height) */
    unsigned int height = 0;

    /** Stores the ID of the FBO on the GPU */
    unsigned int fbo = 0;

    /** Contains description of the texture types we need */
    std::vector<TextureType> textureTypes;

    /** Contains the names/ids of actual texture objects (instanciated according to textureTypes) */
    std::vector<Texture2D> textures;

    /** Is set to true in the constructor if texture was loaded */
    bool isLoaded = false;

    /** Init all the textures and the fbo */
    void init(unsigned int texturesWidth, unsigned int texturesHeight);


public:
    /**
     * Creates a framebuffer object with the given types of textures.
     *
     * Note: The actual fbo and the textures are lazy initialized, i.e.
     * they are created when the first bind-function call is performed.
     */
    TextureFBO(std::vector<TextureType> textureTypes);

    /**
     * Deletes the fbo and ensures that the memory on the GPU is cleaned up.
     */
    ~TextureFBO();

    /**
     * Binds the fbo so that it is used for rendering
     */
    void bind(unsigned int texturesWidth, unsigned int texturesHeight);

    /**
     * Returns the actual texture at ID i.
     */
    Texture2D& getTexture2D(int i){
        return textures[i];
    }

    /**
     * Return the name of the fbo.
     */
    int getName(){
        return fbo;
    }
};
