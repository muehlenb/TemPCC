#include "TextureFBO.h"

#include <iostream>

// Include OpenGL3.3 Core functions:
#include <glad/glad.h>

TextureFBO::TextureFBO(std::vector<TextureType> textureTypes)
    : textureTypes(textureTypes){}

void TextureFBO::init(unsigned int texturesWidth, unsigned int texturesHeight){
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Clear fbo if it exists:
    if(fbo != 0)
        glDeleteFramebuffers(1, &fbo);

    // Delete previous textures if they exists:
    textures.clear();

    // Start with color attachment 0 and count upwards:
    unsigned int colorAttachment = GL_COLOR_ATTACHMENT0;

    // Just stores the constants of _color_ attachments we want to use, eg. GL_COLOR_ATTACHMENT0, ...:
    std::vector<unsigned int> usedColorAttachements;

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // (Re-)Create textures:
    for(unsigned int i=0; i < textureTypes.size(); ++i){
        // Creates the actual texture directly in the std::vector (at the end), instead
        // of constructor call and push_back:
        textures.emplace_back(textureTypes[i], texturesWidth, texturesHeight);

        // Attach the texture to the current framebuffer
        // (we have to distinguish between the depth component and color components):
        if(textureTypes[i].format != GL_DEPTH_COMPONENT){
            usedColorAttachements.push_back(colorAttachment);
            glFramebufferTexture(GL_FRAMEBUFFER, colorAttachment++, textures[i].texture, 0);
        } else {
            glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, textures[i].texture, 0);
        }
    }

    // Define how many color attachments should be active (This defines the number
    // of out-variables in the fragment shader for the first pass):
    glDrawBuffers(GLsizei(usedColorAttachements.size()), &usedColorAttachements[0]);

    // Set new height and width:
    width = texturesWidth;
    height = texturesHeight;
}


TextureFBO::~TextureFBO(){
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if(fbo != 0)
        glDeleteFramebuffers(1, &fbo);
}

void TextureFBO::bind(unsigned int texturesWidth, unsigned int texturesHeight){
    // If dimensions don't fit, (re)initialize the fbo and textures:
    if(texturesWidth != width || texturesHeight != height){
        init(texturesWidth, texturesHeight);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
}
