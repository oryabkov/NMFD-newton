#ifndef __PERLIN_NOISE_H__
#define __PERLIN_NOISE_H__

#include <cmath>

/**************************************/
// Perlin noise implementation
/**************************************/

// Simple hash function for gradient vectors
inline int hash3d(int x, int y, int z, unsigned int seed = 0)
{
    int h = x * 73856093 ^ y * 19349663 ^ z * 83492791;
    h = h ^ static_cast<int>(seed);
    h = h ^ (h >> 16);
    h = h ^ (h >> 8);
    return h & 0x7FFFFFFF;
}

// Fade function for smooth interpolation
template<typename Scalar>
inline Scalar fade(Scalar t)
{
    return t * t * t * (t * (t * 6 - 15) + 10);
}

// Linear interpolation
template<typename Scalar>
inline Scalar lerp(Scalar a, Scalar b, Scalar t)
{
    return a + t * (b - a);
}

// Dot product of gradient vector and distance vector
template<typename Scalar>
inline Scalar grad_dot(int hash, Scalar x, Scalar y, Scalar z)
{
    // Convert hash to one of 12 gradient directions
    int h = hash % 12;
    Scalar u, v;

    if (h < 4) {
        u = x;
        v = (h == 0) ? y : z;
    } else if (h == 4 || h == 8) {
        u = y;
        v = (h == 4) ? x : z;
    } else {
        u = z;
        v = (h == 5) ? x : y;
    }

    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// 3D Perlin noise function
template<typename Scalar>
Scalar perlin_noise_3d(Scalar x, Scalar y, Scalar z, unsigned int seed = 0)
{
    // Find unit cube that contains the point
    int X = static_cast<int>(std::floor(x)) & 255;
    int Y = static_cast<int>(std::floor(y)) & 255;
    int Z = static_cast<int>(std::floor(z)) & 255;

    // Find relative x, y, z of point in cube
    x -= std::floor(x);
    y -= std::floor(y);
    z -= std::floor(z);

    // Compute fade curves for each of x, y, z
    Scalar u = fade(x);
    Scalar v = fade(y);
    Scalar w = fade(z);

    // Hash coordinates of the 8 cube corners
    int A = hash3d(X, Y, Z, seed);
    int B = hash3d(X + 1, Y, Z, seed);
    int AA = hash3d(X, Y + 1, Z, seed);
    int BA = hash3d(X + 1, Y + 1, Z, seed);
    int AB = hash3d(X, Y, Z + 1, seed);
    int BB = hash3d(X + 1, Y, Z + 1, seed);
    int ABA = hash3d(X, Y + 1, Z + 1, seed);
    int BBA = hash3d(X + 1, Y + 1, Z + 1, seed);

    // And add blended results from 8 corners of the cube
    Scalar res = lerp(
        lerp(
            lerp(grad_dot(A, x, y, z), grad_dot(B, x - 1, y, z), u),
            lerp(grad_dot(AA, x, y - 1, z), grad_dot(BA, x - 1, y - 1, z), u),
            v),
        lerp(
            lerp(grad_dot(AB, x, y, z - 1), grad_dot(BB, x - 1, y, z - 1), u),
            lerp(grad_dot(ABA, x, y - 1, z - 1), grad_dot(BBA, x - 1, y - 1, z - 1), u),
            v),
        w);

    return res;
}

#endif // __PERLIN_NOISE_H__
