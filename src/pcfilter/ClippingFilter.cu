// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "pcfilter/ClippingFilter.h"
#include <device_launch_parameters.h>

// Register this filter globally in the application:
ClippingFilterFactory globalClippingFilterFactory;


namespace cuda_cliping_filter {
    // Constant memory:
    __constant__ float cameraMatrix[16];

    // Faces:
    __constant__ float4 faces[6][4];

    __host__ __device__
    float4 MatMultVec4(float mat[16], float4 vec){
        float4 result;
        result.x = mat[0] * vec.x + mat[4] * vec.y + mat[8] * vec.z + mat[12] * vec.w;
        result.y = mat[1] * vec.x + mat[5] * vec.y + mat[9] * vec.z + mat[13] * vec.w;
        result.z = mat[2] * vec.x + mat[6] * vec.y + mat[10] * vec.z + mat[14] * vec.w;
        result.w = mat[3] * vec.x + mat[7] * vec.y + mat[11] * vec.z + mat[15] * vec.w;
        return result;
    }

    __host__ __device__
    inline float crossZOnly(const float4& a , const float4& b){
        return a.x * b.y - a.y * b.x;
    }

    __host__ __device__
        inline float4 subtractXY(const float4& a, const float4& b) {
        return {a.x - b.x, a.y - b.y, 0, a.w - b.w};
    }

    __host__ __device__
    inline bool pInTriangleXY(float4 p, float4 a, float4 b, float4 c) {
        float detT = (b.y - c.y)*(a.x - c.x) + (c.x - b.x)*(a.y - c.y);
        if (abs(detT) < 0.00001f) return false;  // Dreieck ist degeneriert
        float alpha = ((b.y - c.y)*(p.x - c.x) + (c.x - b.x)*(p.y - c.y)) / detT;
        float beta = ((c.y - a.y)*(p.x - c.x) + (a.x - c.x)*(p.y - c.y)) / detT;
        float gamma = 1 - alpha - beta;

        return (alpha >= 0) && (beta >= 0) && (gamma >= 0);
    }

    __host__ __device__
    inline bool pInFaceXY(float4 p, float4 a, float4 b, float4 c, float4 d){
        return pInTriangleXY(p, a, b, c) || pInTriangleXY(p, c, d, a);
    }

    __host__ __device__
    inline bool isFrontFaceInImage(float4 a, float4 b, float4 c){
        a.x /= a.z;
        a.y /= a.z;
        b.x /= b.z;
        b.y /= b.z;
        c.x /= c.z;
        c.y /= c.z;

        return crossZOnly(subtractXY(b, a), subtractXY(c, a)) > 0;
    }

    __global__ void kernel(float4* currentPositions, float3 min, float3 max, int width, int height) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;

        if (idx < width && idy < height) {
            int id = idy * width + idx;

            float4& pos = currentPositions[id];
            float4 worldPos = MatMultVec4(cameraMatrix, pos);

            // Assume simplified a pinhole camera model for -Inf / Inf
            // estimation (test, whether a clipped point lies in the
            // cube when viewed from the camera in image plane):
            float4 projPos = pos;
            projPos.x /= projPos.z;
            projPos.y /= projPos.z;

            // Check if point exceeds AABB:
            if(worldPos.x < min.x || worldPos.y < min.y || worldPos.z < min.z || worldPos.x > max.x || worldPos.y > max.y || worldPos.z > max.z){
                // Old implementation, only return NAN:
                // pos = float4{NAN, NAN, NAN, 1};

                // But we need to determine, whether the point is (from the view of the camera)
                // in front of the AABB, behind the AABB or completely out of the AABB (even when
                // the AABB is transformed into camera space and looked at from the camera in 2D).
                for(int fId = 0; fId < 6; ++fId){
                    // On host side, we mark back faces as invalid via nan in [0].x, since we
                    // don't need to check them:
                    if(!isnan(faces[fId][0].x)){
                        // Check if pos is in one of the faces in camera space (only xy):
                        if(pInFaceXY(projPos, faces[fId][0], faces[fId][1], faces[fId][2], faces[fId][3])){
                            bool inFrontOf = (fId == 0 && worldPos.z < min.z)
                                || (fId == 1 && worldPos.x > max.x)
                                || (fId == 2 && worldPos.z > max.z)
                                || (fId == 3 && worldPos.x < min.x)
                                || (fId == 4 && worldPos.y > max.y)
                                || (fId == 5 && worldPos.y < min.y);

                            if(inFrontOf){
                                pos = float4{-INFINITY, -INFINITY, -INFINITY, 1};
                            } else {
                                pos = float4{INFINITY, INFINITY, INFINITY, 1};
                            }
                            return;
                        }
                    }
                }
                pos = float4{NAN, NAN, NAN, 1};
            }
        }
    }
}

/**
 * Applies this noise filter.
 */
void ClippingFilter::applyFilter(std::vector<std::shared_ptr<OrganizedPointCloud>>& pointClouds) {
    for(unsigned int i = 0; i < pointClouds.size(); ++i){
        std::shared_ptr<OrganizedPointCloud>& pointCloud = pointClouds[i];

        pointCloud->toGPU();

        int height = int(pointCloud->height);
        int width = int(pointCloud->width);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        float4* pos = pointCloud->gpuPositions;

        // Copy device view matrix:
        cudaMemcpyToSymbol(cuda_cliping_filter::cameraMatrix, pointCloud->modelMatrix.data, sizeof(float) * 16);

        float4 faces[6][4] = {
            // Front:
            {
             {min.x, min.y, min.z, 1.f},
             {max.x, min.y, min.z, 1.f},
             {max.x, max.y, min.z, 1.f},
             {min.x, max.y, min.z, 1.f},
            },

            // Right:
            {
             {max.x, min.y, min.z, 1.f},
             {max.x, min.y, max.z, 1.f},
             {max.x, max.y, max.z, 1.f},
             {max.x, max.y, min.z, 1.f},
            },

            // Back:
            {
             {min.x, min.y, max.z, 1.f},
             {min.x, max.y, max.z, 1.f},
             {max.x, max.y, max.z, 1.f},
             {max.x, min.y, max.z, 1.f},
            },

            // Left:
            {
             {min.x, min.y, min.z, 1.f},
             {min.x, max.y, min.z, 1.f},
             {min.x, max.y, max.z, 1.f},
             {min.x, min.y, max.z, 1.f},
            },

            // Up:
            {
             {min.x, max.y, min.z, 1.f},
             {max.x, max.y, min.z, 1.f},
             {max.x, max.y, max.z, 1.f},
             {min.x, max.y, max.z, 1.f},
            },

            // Down:
            {
             {min.x, min.y, min.z, 1.f},
             {min.x, min.y, max.z, 1.f},
             {max.x, min.y, max.z, 1.f},
             {max.x, min.y, min.z, 1.f},
            }
        };

        // Transform that faces into camera space:
        for(int i=0; i < 6; ++i){
            for(int h=0; h < 4; ++h){
                faces[i][h] = cuda_cliping_filter::MatMultVec4(pointCloud->modelMatrix.inverse().data, faces[i][h]);

                // Transform to image space (pinhole model approximation):
                faces[i][h].x /= faces[i][h].z;
                faces[i][h].y /= faces[i][h].z;
            }

            // We don't need to check back faces, so we mark them as invalid:
            if(!cuda_cliping_filter::isFrontFaceInImage(faces[i][0], faces[i][1],faces[i][2])){
                faces[i][0].x = NAN;
            }
        }

        cudaMemcpyToSymbol(cuda_cliping_filter::faces, faces, sizeof(float4) * 6 * 4);

        cuda_cliping_filter::kernel<<<numBlocks, threadsPerBlock>>>(pos, float3{min.x, min.y, min.z}, float3{max.x, max.y, max.z}, width, height);
        cudaDeviceSynchronize();
    }
};
