// © 2023, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

// Include this file only once when compiling:
#pragma once

/* Defines PI with float precision if PI is not already defined: */
#ifndef M_PI
#define M_PI          3.14159265f
#endif

/* Defines the threshold to which the difference between two components should be considered as equal: */
#define COMPARE_DELTA 0.0001f

/* Includes the some math functions like std::abs(...), std::max(...), etc.: */
#include <cmath>

/* Includes some floating point math function like std::isnan(...): */
#include <cfloat>

/* Include print functionalities (e.g. streams) */
#include <iostream>

/**
 * This class defines a 3D vector or point, depending on the
 * forth attribute 'w':
 *
 *  - If w == 0, then it is a vector.
 *  - If w == 1, then it is a point.
 */

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class Vec4
{
public:
    /** The x component of the vector */
    T x;

    /** The y component of the vector */
    T y;

    /** The z component of the vector */
    T z;

    /** The w component of the vector (w=0 for vectors, w=1 for points) */
    T w;

    /**
     * Constructor which creates a *point* at (0,0,0).
     */
    Vec4() : x(0), y(0), z(0), w(1){};

    /**
     * Constructor which creates a *point* with the given parameters (x, y, z).
     */
    Vec4(T x, T y, T z) : x(x), y(y), z(z), w(1){};

    /**
     * Constructor which creates a Vec4 with (x, y, z, w), which can be a point
     * or a vector depending on the chosen w value.
     */
    Vec4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w){};

    /**
     * Constructor which creates a Vec4 with (data[0], data[1], data[2], data[3]).
     */
    Vec4(T data[4]): x(data[0]), y(data[1]), z(data[2]), w(data[3]){}

    /**
     * Returns the length of this vector.
     *
     * Note: Ignores the w coordinate and only use the x,y and z coordinate!
     */
    T length() const {
        return T(sqrt(pow (x, 2) + pow (y, 2) + pow (z, 2)));
    }

    /**
     * Returns the squared length of this vector.
     *
     * Note: Ignores the w coordinate and only use the x,y and z coordinate!
     */
    T squaredLength() const {
        return x*x + y*y + z*z;
    }

    /**
     * Returns the Euclidean distance from this point to the given point.
     *
     * NOTE: Since we use this function only with points, only the (x,y,z)
     * coordinates are used to calculate the distance, NOT the w-coordinate!
     */
    T distanceTo(const Vec4 v) const{
        return T(sqrt(pow (x - v.x, 2) + pow (y - v.y, 2) + pow (z - v.z, 2)));
    }

    /**
     * Returns the dot product of this vector with the given vector
     *
     * NOTE: Only x,y and z coordinates are used, NOT the w-coordinate!
     */
    T dot(const Vec4 v) const{
        return x * v.x + y * v.y + z * v.z;
    }

    /**
     * Returns the cross product of this vector with the given vector
     *
     * NOTE: Only x,y and z coordinates are used, not the w-coordinate
     * and return a vector with w = 0!
     */
    Vec4 cross(const Vec4 v) const{
        return Vec4(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x,
            0
            );
    }

    /**
     * Returns a normalized Vector of this vector, so that the length
     * is 1. (Note that normalization only makes sense for vectors, not
     * for points, so you can assume that w=0).
     */

    Vec4 normalized() const {
        float length = sqrtf(x * x + y * y + z * z);
        return Vec4(x / length, y / length, z / length, 0);
    }

    /**
     * Checks that all components are similar enough for both vectors to be
     * considered equal.
     *
     * NOTE: Use 'COMPARE_DELTA' to check whether the difference is small
     * enough.
     */

    bool operator==(const Vec4 v) const {
        return abs(x-v.x) < COMPARE_DELTA && abs(y - v.y) < COMPARE_DELTA && abs(z - v.z) < COMPARE_DELTA;
    }

    /**
     * Checks if at least one component has a higher difference than
     * 'COMPARE_DELTA' when comparing this vector to the given one.
     */

    bool operator!=(const Vec4 v) const{
        return !(*this == v);
    }

    Vec4 onlyXY() const {
        return Vec4(x,y,0,0);
    }

    /**
     * Returns this vector negated.
     */
    Vec4 operator-() const{
        return Vec4(-x, -y, -z, w);
    }

    /**
     * Returns a vector where each component of this vector is added by the
     * respective component of the given vector (also the w component!).
     */
    Vec4 operator+(const Vec4 v) const {
        return Vec4(x + v.x, y + v.y, z + v.z, w + v.w);
    }

    /**
     * Returns a new vector where each component of this vector is subtracted
     * by the respective component of the given vector (also the w component!).
     */
    Vec4 operator-(const Vec4 v) const{
        return Vec4(x - v.x, y - v.y, z - v.z, w - v.w);
    };

    /**
     * Returns a new vector where the x, y and z components of this vector are
     * multiplicated by the given scalar (note that w should NOT be scaled!)
     */
    Vec4 operator*(const T scalar) const{
        return Vec4(x * scalar, y * scalar, z * scalar, w);
    }

    /**
     * Returns a new vector where the x, y and z components of this vector are
     * divided by the given scalar (note that w should NOT be scaled!).
     */
    Vec4 operator/(const T scalar) const{
        return Vec4(x / scalar, y / scalar, z / scalar, w);
    }

    /**
     * Allows access to individual components via indices as with arrays,
     * where x is at idx 0, y at idx 1, and so on.
     *
     * The returned value is a reference so that it can be written to it.
     */
    T& operator[](int i){
        if (i >= 0 && i < 4)
            throw std::invalid_argument( "The index i has to be between 0 and 3!" );

        return *(&x + i);
    }

    /**
     * Allows access to individual components via indices as with arrays,
     * where x is at idx 0, y at idx 1, and so on.
     *
     * This is the const implementation to be able to access the values
     * via [] on a const vector too.
     */
    T operator[](int i) const{
        if (i >= 0 && i < 4)
            throw std::invalid_argument( "The index i has to be between 0 and 3!" );

        return *(&x + i);
    }

    /**
     * Returns whether this vector is valid, which means that every
     * component is valid (and not NAN).
     */
    bool valid() const{
        return !isnan(x) && !isnan(y) && !isnan(z) && !isnan(w);
    }

    bool equals(Vec4<T> v, T tolerance = 0.0001f){
        return abs(x-v.x) < tolerance && abs(y-v.y) < tolerance && abs(z-v.z) < tolerance && abs(w-v.w) < tolerance;
    }

    /**
     * Adds the given Vec4 to an output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, Vec4 v){
        os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
        return os;
    }

    Vec4<float> toVec4f(){
        return Vec4<float>(float(x), float(y), float(z), float(w));
    }

    Vec4<double> toVec4d(){
        return Vec4<double>(double(x), double(y), double(z), double(w));
    }
};

typedef Vec4<float> Vec4f;
typedef Vec4<double> Vec4d;
typedef Vec4<uint8_t> Vec4b;
typedef Vec4<int> Vec4i;
