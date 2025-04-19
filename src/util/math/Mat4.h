// © 2022, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

// Include this file only once when compiling:
#pragma once

/* Includes the some math functions like std::abs(...), std::max(...), etc.: */
#include <cmath>

/* Includes the std::vector class, which is comparable to an ArrayList in Java.
 * Note that this vector class is NOT meant to be a mathematical vector! */
#include <vector>

#include <cstring>
#include <iomanip>

/* Include our Vec4f which is meant to be a vector in the mathematical sense */
#include "Vec4.h"


/**
 * Implementation of a 4x4 Matrix. This matrix class is intended specifically
 * for the transformation of points and vectors. This will be explained in the
 * lecture later on.
 *
 * Note that we use a column-major order to store the values, like OpenGL does.
 * This means, that the first column is stored into data[0], data[1], data[2] and
 * data[3] and the first row is stored into data[0], data[4], data[8] and data[12].
 */

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class Mat4
{
public:
    /**
     * In this array, all the values of this 4x4 matrix are stored
     * (in column-major order).
     */
    T data[16];

    /**
     * Constructs an identity matrix.
     *
     * The cells on the diagonal of the identity matrix have the value 1, other
     * cells have the value 0.
     */
    Mat4(){
        // Set all values to zero, except for the diagonal of the matrix:
        for(int i=0; i < 16; ++i){
            data[i] = T(i%5 == 0 ? 1 : 0);
        }
    }

    /**
     * Constructs a matrix with the given data (in column-major order).
     */
    Mat4(const T pData[16]){
        // Just copy all values of the given array into the data-array:
        memcpy(data, pData, sizeof(T) * 16);
    }

    /**
     * Constructs a matrix with the given data (in column-major order).
     *
     * WARNING: This method expects the std::vector to have at least 16 entries,
     * because the first 16 entries will be copied unchecked.
     */
    Mat4(const std::vector<T> pData){
        memcpy(data, &pData[0], sizeof(T) * 16);
    }



    /**
     * Constructs a transformation matrix which rotates and scales the unit axes
     * to the given axis1, axis2 and axis3 and applies the given translation.
     */
    Mat4(const Vec4<T> axis1, const Vec4<T> axis2, const Vec4<T> axis3, const Vec4<T> translation){
        // Copy the first axis to the first four T values of the data-array:
        memcpy(&data[0], &axis1.x, sizeof(T) * 4);

        // Copy the second axis to the next four T values of the data-array:
        memcpy(&data[4], &axis2.x, sizeof(T) * 4);

        // Copy the third axis to the next four T values of the data-array:
        memcpy(&data[8], &axis3.x, sizeof(T) * 4);

        // Copy the translation to the next four T values of the data-array:
        memcpy(&data[12], &translation.x, sizeof(T) * 4);
    }

    /**
     * Transposes this matrix
     */
    Mat4<T> transpose() {
        Mat4<T> result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result.data[j * 4 + i] = data[i * 4 + j];
            }
        }

        return result;
    }


    /**
     * Multiplies this matrix by the given matrix and returns the result as a new
     * matrix.
     *
     * It is a normal multiplication of two 4x4 matrices as you know it from school
     * (using all 4 columns and rows).
     */
    Mat4<T> operator*(const Mat4<T> m) const {
        T result[16];

        for(int x = 0; x < 4; ++x){
            for(int y = 0; y < 4; ++y){
                result[x*4+y] = 0;

                for(int i = 0; i < 4; ++i){
                    result[x*4+y] += data[y+i*4] * m.data[x*4+i];
                }
            }
        }

        return Mat4<T>(result);
    }

    /**
     * Multiplies this matrix by the given vector and returns the result as a
     * new vector.
     *
     * It is a normal multiplication of a 4x4 matrix with a 4D-vector as you know it
     * from school (using all 4 columns and rows of the matrix and x,y,z AND w of the
     * 4D vector!).
     */
    Vec4<T> operator*(const Vec4<T> v) const {
        T dest[4];

        for(int i = 0; i < 4; ++i){
            dest[i] = data[i] * v.x;
            dest[i] += data[i+1*4] * v.y;
            dest[i] += data[i+2*4] * v.z;
            dest[i] += data[i+3*4] * v.w;
        }

        return Vec4<T>(dest);
    }

    /**
     * Returns a transformation matrix which performs a translation by x, y and z.
     */
    static Mat4<T> translation(const T x, const T y, const T z){
        Mat4<T> mat;
        mat.data[12] = x;
        mat.data[13] = y;
        mat.data[14] = z;
        return mat;
    }

    /**
     * Returns a transformation matrix which performs a rotation around the x axis
     * by the given angle (in radians).
     */
    static Mat4<T> rotationX(const T angle){
        Mat4<T> matrix;
        matrix.data[5] = cosf(angle);
        matrix.data[6] = sinf(angle);
        matrix.data[9] = -sinf(angle);
        matrix.data[10] = cosf(angle);
        return matrix;
    }

    /**
     * Returns a transformation matrix which performs a rotation around the y axis
     * by the given angle (in radians).
     */
    static Mat4<T> rotationY(const T angle){
        Mat4<T> matrix;
        matrix.data[0] = cosf(angle);
        matrix.data[2] = -sinf(angle);
        matrix.data[8] = sinf(angle);
        matrix.data[10] = cosf(angle);
        return matrix;
    }

    /**
     * Returns a transformation matrix which performs a rotation around the z axis
     * by the given angle (in radians).
     */
    static Mat4<T> rotationZ(const T angle){
        Mat4<T> matrix;

        matrix.data[0] = cosf(angle);
        matrix.data[1] = sin(angle);
        matrix.data[4] = -sinf(angle);
        matrix.data[5] = cosf(angle);

        return matrix;
    }

    /**
     * Returns a transformation matrix which performs a uniform scale.
     */
    static Mat4<T> scale(const T factor){
        Mat4<T> mat;
        mat.data[0] = factor;
        mat.data[5] = factor;
        mat.data[10] = factor;
        return mat;
    }

    /**
     * Returns a transformation matrix which performs a uniform scale.
     */
    static Mat4<T> scale(const T xScale, const T yScale, const T zScale){
        Mat4<T> mat;
        mat.data[0] = xScale;
        mat.data[5] = yScale;
        mat.data[10] = zScale;
        return mat;
    }

    /**
     * Returns the position where a point (0,0,0) would be transformed to after applying
     * this transformation matrix.
     *
     * In homogeneous coordinates, this is always the point which is stored in the fourth
     * column of the matrix.
     */
    Vec4<T> getPosition() const{
        return Vec4<T>(data[12], data[13], data[14], 1);
    }

    /**
     * Returns an inverse matrix to this transformation matrix, which consequently
     * reverses the transformation of this transformation matrix.
     *
     * In case no inversion is possible, the identity matrix is returned.
     */
    Mat4<T> inverse(){
        T result[16];

        // Because of the length, we just write m:
        T* m = data;

        result[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
        result[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
        result[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
        result[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
        result[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
        result[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
        result[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
        result[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
        result[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
        result[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
        result[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
        result[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
        result[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
        result[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
        result[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
        result[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

        T det = m[0] * result[0] + m[1] * result[4] + m[2] * result[8] + m[3] * result[12];

        // In case that the inversion is not possible, return the identity (Mat4f()):
        if(det < 0.0001f && det > -0.0001f)
            return Mat4<T>();

        det = 1.0f / det;

        for (int i = 0; i < 16; ++i)
            result[i] *= det;

        // Finally, return the inverse matrix:
        return Mat4<T>(result);
    }

    /**
     * Creates a perspective transformation matrix with the given parameters
     *
     * A perspective projection matrix creates a 3D impression which is similar
     * to photos, videos and the human eye.
     *
     * Note: This transformation is minimally adjusted so that you can switch
     * between "no perspective" and "perspective" without difficulty.
     *
     * @param aspectRatio       Aspect ratio of the window.
     * @param fov               The opening angle of the 'eye' or camera in *degree*.
     * @param nearClipDistance  If objects are below this distance, they are hidden.
     * @param farClipDistance   If objects are behind this distance, they are hidden.
     */
    static Mat4<T> perspectiveTransformation(T aspectRatio = 1.f, T fov = 75.f, T near = 0.01f, T far = 1000.f){
        Mat4<T> result;

        fov *= (M_PI / 180);

        result.data[0] = -cosf(fov / 2) / sinf(fov / 2) / aspectRatio;
        result.data[5] = cosf(fov / 2) / sinf(fov / 2);
        result.data[10] = (far + near) / (far - near);
        result.data[11] = 1;
        result.data[14] = (-2*far*near) / (far-near);
        result.data[15] = 0;
        return result;
    }

    /**
     * Prints the given Mat4f to an output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, Mat4<T> mat){
        os << std::fixed;
        for(int y=0; y < 4; ++y){
            for(int x=0; x < 4; ++x){
                os << std::left << std::setw(8) << std::setprecision(4) << mat.data[x*4+y] << std::setfill(' ');
            }
            os << std::endl;
        }
        os.unsetf(std::ios_base::fixed);
        return os;
    }

    /**
     * Creates a rotation matrix so that the UNIT vector start is rotated
     * so that it will be at UNIT vector end.
     *
     * Note: Both vectors are assumed to be unit vectors!
     */
    static Mat4<T> getRotationMatrix(Vec4<T> start, Vec4<T> end){
        Mat4<T> rotStart;

        if(start.length() < 0.01f || end.length() < 0.01f)
            return Mat4<T>();

        // Search for a sufficiently perpendicular axis:
        Vec4<T> perpendicular(0,1,0);
        if(abs(start.dot(perpendicular)) > 0.9 || abs(end.dot(perpendicular)) > 0.9){
            perpendicular = Vec4<T>(1,0,0);
        }

        if(abs(start.dot(perpendicular)) > 0.9 || abs(end.dot(perpendicular)) > 0.9)
            perpendicular = Vec4<T>(0,0,1);

        if(!start.equals(perpendicular)){
            rotStart.data[0] = start.x;
            rotStart.data[1] = start.y;
            rotStart.data[2] = start.z;

            Vec4<T> v1 = start.cross(perpendicular).normalized();
            rotStart.data[4] = v1.x;
            rotStart.data[5] = v1.y;
            rotStart.data[6] = v1.z;

            Vec4<T> v2 = start.cross(v1).normalized();
            rotStart.data[8] = v2.x;
            rotStart.data[9] = v2.y;
            rotStart.data[10] = v2.z;
        }

        Mat4<T> rotEnd;
        if(!end.equals(perpendicular)){
            rotEnd.data[0] = end.x;
            rotEnd.data[1] = end.y;
            rotEnd.data[2] = end.z;

            Vec4<T> w1 = end.cross(perpendicular).normalized();
            rotEnd.data[4] = w1.x;
            rotEnd.data[5] = w1.y;
            rotEnd.data[6] = w1.z;

            Vec4<T> w2 = end.cross(w1).normalized();
            rotEnd.data[8] = w2.x;
            rotEnd.data[9] = w2.y;
            rotEnd.data[10] = w2.z;
        }

        return rotEnd * rotStart.inverse();
    }

    /**
     * Creates a transformation matrix so that the vector `start` is rotated
     * and scaled so that it will be transformed to the vector `end`.
     *
     * Note: Both vectors are assumed to be unit vectors!
     */
    static Mat4<T> getRotationAndScaleMatrix(Vec4<T> start, Vec4<T> end){
        T endLength = end.length();
        T startLength = start.length();

        if(endLength < 0.01f || startLength < 0.01f)
            return Mat4<T>();

        T lengthRelative = endLength / startLength;

        Vec4<T> startNormed = start.normalized();
        Vec4<T> endNormed = end.normalized();

        Mat4<T> rotationMatrix = getRotationMatrix(startNormed, endNormed);


        return rotationMatrix * Mat4::scale(lengthRelative);
    }

    static Mat4 switchAxesMatrix(int a, int b, int c){
        Mat4 switchAxesMat;
        switchAxesMat.data[0] = abs(a) == 1 ? (a<0?-1:1) : 0;
        switchAxesMat.data[1] = abs(a) == 2 ? (a<0?-1:1) : 0;
        switchAxesMat.data[2] = abs(a) == 3 ? (a<0?-1:1) : 0;
        switchAxesMat.data[4] = abs(b) == 1 ? (b<0?-1:1) : 0;
        switchAxesMat.data[5] = abs(b) == 2 ? (b<0?-1:1) : 0;
        switchAxesMat.data[6] = abs(b) == 3 ? (b<0?-1:1) : 0;
        switchAxesMat.data[8] = abs(c) == 1 ? (c<0?-1:1) : 0;
        switchAxesMat.data[9] = abs(c) == 2 ? (c<0?-1:1) : 0;
        switchAxesMat.data[10]= abs(c) == 3 ? (c<0?-1:1) : 0;
        return switchAxesMat;
    }

    Mat4 switchAxes(int a, int b, int c){
        Mat4 switchAxesMat = switchAxesMatrix(a,b,c);
        return switchAxesMat * *this;
    }

    Mat4 scaleTranslation(float scale){
        Mat4 result = *this;
        result.data[12] *= scale;
        result.data[13] *= scale;
        result.data[14] *= scale;
        return result;
    }

    Mat4<float> toMat4f(){
        return Mat4<float>({
            float(data[0]), float(data[1]), float(data[2]), float(data[3]),
            float(data[4]), float(data[5]), float(data[6]), float(data[7]),
            float(data[8]), float(data[9]), float(data[10]), float(data[11]),
            float(data[12]), float(data[13]), float(data[14]), float(data[15])
        });
    }

    Mat4<double> toMat4d(){
        return Mat4<double>({
            double(data[0]), double(data[1]), double(data[2]), double(data[3]),
            double(data[4]), double(data[5]), double(data[6]), double(data[7]),
            double(data[8]), double(data[9]), double(data[10]), double(data[11]),
            double(data[12]), double(data[13]), double(data[14]), double(data[15])
        });
    }
};

typedef Mat4<float> Mat4f;
typedef Mat4<double> Mat4d;
