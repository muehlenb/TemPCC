// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "src/pcstreamer/binary/BinaryStream.h"

#include <algorithm>

#define LOOKUP_TABLE_SIZE 1024

BinaryStream::BinaryStream(std::string pRgbFilePath, std::string pDepthFilePath, Mat4f bMatrix, double bSyncOffset){
    rgbFilePath = pRgbFilePath;
    depthFilePath = pDepthFilePath;
    loadedModelMatrix = bMatrix;
    syncOffset = bSyncOffset;

    rgbStream = std::ifstream(rgbFilePath, std::ios::binary);
    depthStream = std::ifstream(depthFilePath, std::ios::binary | std::ios::ate);

    if(!rgbStream.good()){
        std::cout << "RGB Stream could not be opened!" << std::endl;
        return;
    }

    if(!depthStream.good()){
        std::cout << "Depth Stream could not be opened!" << std::endl;
        return;
    }

    //depthStream.ignore( std::numeric_limits<std::streamsize>::max() );
    std::streamsize length = depthStream.tellg();
    //std::cout << "Stream Length: " << length << std::endl;
    depthStream.clear();   //  Since ignore will have set eof.
    depthStream.seekg( 0, std::ios_base::beg );

    depthStream.read(reinterpret_cast<char*>(&imageWidth), sizeof(imageWidth));
    depthStream.read(reinterpret_cast<char*>(&imageHeight), sizeof(imageHeight));
    depthStream.read(reinterpret_cast<char*>(&pixelSize), sizeof(pixelSize));
    depthStream.read(reinterpret_cast<char*>(&halfFovX), sizeof(halfFovX));
    depthStream.read(reinterpret_cast<char*>(&halfFovY), sizeof(halfFovY));

    totalFrameCount = (unsigned int)((length - 20) / (imageWidth * imageHeight * pixelSize + 4));

    std::cout << "Stream initialized:" << std::endl;
    std::cout << "    Width: " << imageWidth << " | Height: " << imageHeight << " | PixelSize: " << pixelSize << std::endl;
    std::cout << "    FoV X: " << halfFovX << " | FoV Y: " << halfFovY << std::endl;

    // Create buffer for depth image:
    currentDepthImage = new uint16_t[imageWidth * imageHeight];
    currentColorImage = new uint8_t[imageWidth * imageHeight * 4];

    for(uint32_t i = 0; i < imageWidth * imageHeight; ++i){
        currentColorImage[i * 4] = 0;
        currentColorImage[i * 4 + 1] = 255;
        currentColorImage[i * 4 + 2] = 255;
        currentColorImage[i * 4 + 3] = 255;
    }

    // Set this stream as ready:
    isReady = true;

    currentFrame = -1;//readImage(0);
}


BinaryStream::~BinaryStream(){
    delete[] currentColorImage;
    delete[] currentDepthImage;

    delete[] lookup2DTo3D;
    delete[] lookup3DTo2D;
}

Vec4f BinaryStream::Gen3DPointByDepth(int sX, int sY, float z) {
    Vec4f result;

    result.z = z;
    result.x = tan(halfFovX) * -(float(sX * 2) / imageWidth - 1) * z;
    result.y = tan(halfFovY) * -(float(sY * 2) / imageHeight - 1) * z;

    return result / tan(halfFovX);
}

Vec4f BinaryStream::InverseGen3DPointByDepth(const Vec4f& point) {
    const Vec4f sPoint = point * tan(halfFovX);

    float z = sPoint.z;
    float x = ((-sPoint.x / (tan(halfFovX) * z) + 1) * imageWidth / 2);
    float y = ((-sPoint.y / (tan(halfFovY) * z) + 1) * imageHeight / 2);

    return Vec4f(x, y, z);
}

bool BinaryStream::readImage(unsigned int frame){
    if(!isReady)
        return false;

    if(frame < 0 || frame >= totalFrameCount)
        return false;

    int rgbImagePosition = 20 + (4 + imageWidth * imageHeight * 4) * frame;
    rgbStream.seekg( rgbImagePosition, std::ios_base::beg );

    int depthImagePosition = 20 + (4 + imageWidth * imageHeight * 2) * frame + 4;
    depthStream.seekg( depthImagePosition, std::ios_base::beg );

    rgbStream.read(reinterpret_cast<char*>(&currentRGBTime), sizeof(currentRGBTime));
    rgbStream.read(reinterpret_cast<char*>(&currentColorImage[0]), imageWidth * imageHeight * 4);
    depthStream.read(reinterpret_cast<char*>(&currentDepthImage[0]), imageWidth * imageHeight * 2);

    //std::cout << "Read frame " << frame << std::endl;

    currentFrame = frame;

    return true;
}

void BinaryStream::generatePointCloud(){
    convertRawToOrganizedPointCloud();
}

void BinaryStream::createLookupTables(){
    lookup2DTo3D = new float[imageWidth * imageHeight * 2];
    lookup3DTo2D = new float[LOOKUP_TABLE_SIZE * LOOKUP_TABLE_SIZE * 2];

    unsigned int pcSize = imageWidth * imageHeight;
    for(unsigned int i=0; i < pcSize; ++i){
        Vec4f point = Gen3DPointByDepth(i%imageWidth, i/imageWidth, 1.f);

        lookup2DTo3D[i*2] = point.x;
        lookup2DTo3D[i*2+1] = point.y;
    }

    for(unsigned int y = 0; y < LOOKUP_TABLE_SIZE; ++y){
        for(unsigned int x = 0; x < LOOKUP_TABLE_SIZE; ++x){
            Vec4f p(
                ((x / float(LOOKUP_TABLE_SIZE)) * 2 - 1) * 1000,
                ((y / float(LOOKUP_TABLE_SIZE)) * 2 - 1) * 1000,
                1000
            );

            Vec4f imgCoords = InverseGen3DPointByDepth(p);

            int i = y * LOOKUP_TABLE_SIZE + x;
            float relImgX = imgCoords.x / float(imageWidth);
            float relImgY = imgCoords.y / float(imageHeight);

            if(relImgX >= 0 && relImgX <= 1 && relImgY >= 0 && relImgY <= 1){
                lookup3DTo2D[i*2] = 1 - relImgX;
                lookup3DTo2D[i*2+1] = relImgY;
            } else {
                lookup3DTo2D[i*2] = -1;
                lookup3DTo2D[i*2+1] = -1;
            }
        }
    }

    lookupTablesInitialized = true;
}

void BinaryStream::convertRawToOrganizedPointCloud(){
    if(!lookupTablesInitialized){
        createLookupTables();
    }

    // Initialize new organized point cloud:
    std::shared_ptr<OrganizedPointCloud> pc = std::make_shared<OrganizedPointCloud>(imageWidth, imageHeight);
    pc->positions = new Vec4f[imageWidth * imageHeight];
    pc->colors = new Vec4b[imageWidth * imageHeight];
    pc->width = imageWidth;
    pc->height = imageHeight;
    pc->modelMatrix = loadedModelMatrix;
    pc->lookup3DToImage = lookup3DTo2D;
    pc->lookupImageTo3D = lookup2DTo3D;
    pc->lookup3DToImageSize = LOOKUP_TABLE_SIZE;
    pc->timestamp = currentRGBTime;
    pc->frameID = currentFrame;

    unsigned int pcSize = imageWidth * imageHeight;
    for(unsigned int i=0; i < pcSize; ++i){
        pc->positions[i] = Gen3DPointByDepth(imageWidth - i%imageWidth - 1, i/imageWidth, currentDepthImage[i] / 1000.f);
    }

    std::memcpy(&pc->colors[0], &currentColorImage[0], pcSize * 4);

    currentPointCloud = pc;
}
