#include "Texture2D.h"

// Include 'print to console' functionality:
#include <iostream>

// Include OpenGL3.3 Core functions:
#include <glad/glad.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Texture2D::Texture2D():numOfCopies(new int(1)){}

Texture2D::Texture2D(TextureType type, unsigned int width, unsigned int height)
    :numOfCopies(new int(1)){
    // Generate a texture name:
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Define how the texture should behave when exceeding coordinates of [0,1]:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Define how the texture should be interpolated when it is rendered smaller
    // or bigger that the original image size on the screen:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Allocate the actual space in the GPU where the pixels of the texture are stored
    // according to the given type and size (the last "0" means "empty"):
    glTexImage2D(GL_TEXTURE_2D, 0, type.internalFormat, width, height, 0, type.format, type.type, 0);

    // Deactive the texture:
    glBindTexture(GL_TEXTURE_2D, 0);
}

Texture2D::Texture2D(std::string imagePath)
    :numOfCopies(new int(1)){
    int width;
    int height;
    int channels;

    // Use the stb_image library to load the image:
    unsigned char *raw_image_data = stbi_load(imagePath.c_str(), &width, &height, &channels, 4);

    // Generate a texture and bind it:
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Define how the texture should behave when exceeding coordinates of [0,1]:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // Define how the texture should be interpolated when it is rendered smaller
    // or bigger that the original image size on the screen:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // If the image was successfully loaded to the RAM, then...
    if (raw_image_data){
        // ... upload it to the GPU:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw_image_data);

        // Internally generate different versions of the texture:
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        std::cerr << "Error loading texture from: " << imagePath << std::endl;
    }

    // Delete image from the RAM because we don't need it anymore
    // in the RAM when we have it at the GPU:
    stbi_image_free(raw_image_data);

    // Deactive the texture:
    glBindTexture(GL_TEXTURE_2D, 0);
}

/**
 * This is an explicit copy constructor which is automatically
 * called every time an existing Texture is assigned to a variable
 * via the = Operator. Not important for CG1 and C++ specific,
 * but it is needed here to avoid problems because of early
 * deletion of the texture (on GPU) when the destructor is called.
 */
Texture2D::Texture2D(const Texture2D& t){
    texture = t.texture;
    numOfCopies = t.numOfCopies;
    isLoaded = t.isLoaded;
    ++(*numOfCopies);
}

Texture2D::~Texture2D(){
    // We only want the texture to be destroyed
    // when the last copy of the texture is deleted:
    if(--(*numOfCopies) > 0)
        return;

    // If there is a compiled & linked shaderProgram, delete it from the GPU:
    if(texture != 0)
        glDeleteTextures(1, &texture);

    // To avoid reusage after deletion:
    texture = 0;

    // Delete also the numOfCopies variable:
    delete numOfCopies;
}

void Texture2D::bind(int textureSlot){
    glActiveTexture(GL_TEXTURE0 + textureSlot);

    // Activates the texture for the following rendering
    glBindTexture(GL_TEXTURE_2D, texture);
}
