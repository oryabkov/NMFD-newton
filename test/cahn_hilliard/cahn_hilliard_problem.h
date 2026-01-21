#ifndef __CAHN_HILLIARD_PROBLEM_H__
#define __CAHN_HILLIARD_PROBLEM_H__

#include <cmath>
#include <scfd/utils/device_tag.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define PI M_PI

namespace tests
{

template <class Scalar, class TensorType, int m, int n, int k>
class trig_rhs
{
public:
    __DEVICE_TAG__ TensorType operator()( Scalar x, Scalar y, Scalar z ) const
    {

        return TensorType{
            -std::pow( PI, 4 ) * std::pow( k, 4 ) * std::sin( PI * k * z ) * std::sin( PI * m * x ) *
                    std::sin( PI * n * y ) -
                2 * std::pow( PI, 4 ) * std::pow( k, 2 ) * std::pow( m, 2 ) * std::sin( PI * k * z ) *
                    std::sin( PI * m * x ) * std::sin( PI * n * y ) -
                2 * std::pow( PI, 4 ) * std::pow( k, 2 ) * std::pow( n, 2 ) * std::sin( PI * k * z ) *
                    std::sin( PI * m * x ) * std::sin( PI * n * y ) -
                3 * std::pow( PI, 2 ) * std::pow( k, 2 ) * std::pow( std::sin( PI * k * z ), 3 ) *
                    std::pow( std::sin( PI * m * x ), 3 ) * std::pow( std::sin( PI * n * y ), 3 ) +
                6 * std::pow( PI, 2 ) * std::pow( k, 2 ) * std::sin( PI * k * z ) *
                    std::pow( std::sin( PI * m * x ), 3 ) * std::pow( std::sin( PI * n * y ), 3 ) *
                    std::pow( std::cos( PI * k * z ), 2 ) +
                std::pow( PI, 2 ) * std::pow( k, 2 ) * std::sin( PI * k * z ) * std::sin( PI * m * x ) *
                    std::sin( PI * n * y ) -
                std::pow( PI, 4 ) * std::pow( m, 4 ) * std::sin( PI * k * z ) * std::sin( PI * m * x ) *
                    std::sin( PI * n * y ) -
                2 * std::pow( PI, 4 ) * std::pow( m, 2 ) * std::pow( n, 2 ) * std::sin( PI * k * z ) *
                    std::sin( PI * m * x ) * std::sin( PI * n * y ) -
                3 * std::pow( PI, 2 ) * std::pow( m, 2 ) * std::pow( std::sin( PI * k * z ), 3 ) *
                    std::pow( std::sin( PI * m * x ), 3 ) * std::pow( std::sin( PI * n * y ), 3 ) +
                6 * std::pow( PI, 2 ) * std::pow( m, 2 ) * std::pow( std::sin( PI * k * z ), 3 ) *
                    std::sin( PI * m * x ) * std::pow( std::sin( PI * n * y ), 3 ) *
                    std::pow( std::cos( PI * m * x ), 2 ) +
                std::pow( PI, 2 ) * std::pow( m, 2 ) * std::sin( PI * k * z ) * std::sin( PI * m * x ) *
                    std::sin( PI * n * y ) -
                std::pow( PI, 4 ) * std::pow( n, 4 ) * std::sin( PI * k * z ) * std::sin( PI * m * x ) *
                    std::sin( PI * n * y ) -
                3 * std::pow( PI, 2 ) * std::pow( n, 2 ) * std::pow( std::sin( PI * k * z ), 3 ) *
                    std::pow( std::sin( PI * m * x ), 3 ) * std::pow( std::sin( PI * n * y ), 3 ) +
                6 * std::pow( PI, 2 ) * std::pow( n, 2 ) * std::pow( std::sin( PI * k * z ), 3 ) *
                    std::pow( std::sin( PI * m * x ), 3 ) * std::sin( PI * n * y ) *
                    std::pow( std::cos( PI * n * y ), 2 ) +
                std::pow( PI, 2 ) * std::pow( n, 2 ) * std::sin( PI * k * z ) * std::sin( PI * m * x ) *
                    std::sin( PI * n * y ),
            0.0
        };
    }

    __DEVICE_TAG__ TensorType get_exact_solution( Scalar x, Scalar y, Scalar z ) const
    {

        return TensorType{
            std::pow( PI, 2 ) * std::pow( k, 2 ) * std::sin( PI * k * z ) * std::sin( PI * m * x ) *
                    std::sin( PI * n * y ) +
                std::pow( PI, 2 ) * std::pow( m, 2 ) * std::sin( PI * k * z ) * std::sin( PI * m * x ) *
                    std::sin( PI * n * y ) +
                std::pow( PI, 2 ) * std::pow( n, 2 ) * std::sin( PI * k * z ) * std::sin( PI * m * x ) *
                    std::sin( PI * n * y ) +
                std::pow( std::sin( PI * k * z ), 3 ) * std::pow( std::sin( PI * m * x ), 3 ) *
                    std::pow( std::sin( PI * n * y ), 3 ) -
                std::sin( PI * k * z ) * std::sin( PI * m * x ) * std::sin( PI * n * y ),
            std::sin( PI * k * z ) * std::sin( PI * m * x ) * std::sin( PI * n * y )
        };
    }
};

template <class Scalar, class TensorType>
class poli_rhs
{
public:
    __DEVICE_TAG__ TensorType operator()( Scalar x, Scalar y, Scalar z ) const
    {
        return TensorType{
            24 * x * y + 24 * x * z + 24 * y * z + 828 * x * y * z - 48 * pow( x, 3 ) * y - 48 * pow( x, 3 ) * z -
                564 * pow( x, 2 ) * y * z - 48 * x * pow( y, 3 ) - 564 * x * pow( y, 2 ) * z -
                564 * x * y * pow( z, 2 ) - 48 * x * pow( z, 3 ) - 48 * pow( y, 3 ) * z - 48 * y * pow( z, 3 ) +
                24 * pow( x, 4 ) * y + 24 * pow( x, 4 ) * z - 528 * pow( x, 3 ) * y * z +
                288 * pow( x, 2 ) * pow( y, 2 ) * z + 288 * pow( x, 2 ) * y * pow( z, 2 ) + 24 * x * pow( y, 4 ) -
                528 * x * pow( y, 3 ) * z + 288 * x * pow( y, 2 ) * pow( z, 2 ) - 528 * x * y * pow( z, 3 ) +
                24 * x * pow( z, 4 ) + 24 * pow( y, 4 ) * z + 24 * y * pow( z, 4 ) + 264 * pow( x, 4 ) * y * z +
                96 * pow( x, 3 ) * pow( y, 3 ) + 552 * pow( x, 3 ) * pow( y, 2 ) * z +
                552 * pow( x, 3 ) * y * pow( z, 2 ) + 96 * pow( x, 3 ) * pow( z, 3 ) +
                552 * pow( x, 2 ) * pow( y, 3 ) * z + 552 * pow( x, 2 ) * y * pow( z, 3 ) + 264 * x * pow( y, 4 ) * z +
                552 * x * pow( y, 3 ) * pow( z, 2 ) + 552 * x * pow( y, 2 ) * pow( z, 3 ) + 264 * x * y * pow( z, 4 ) +
                96 * pow( y, 3 ) * pow( z, 3 ) - 48 * pow( x, 4 ) * pow( y, 3 ) - 276 * pow( x, 4 ) * pow( y, 2 ) * z -
                276 * pow( x, 4 ) * y * pow( z, 2 ) - 48 * pow( x, 4 ) * pow( z, 3 ) - 48 * pow( x, 3 ) * pow( y, 4 ) -
                54 * pow( x, 3 ) * pow( y, 3 ) * z - 576 * pow( x, 3 ) * pow( y, 2 ) * pow( z, 2 ) -
                54 * pow( x, 3 ) * y * pow( z, 3 ) - 48 * pow( x, 3 ) * pow( z, 4 ) -
                276 * pow( x, 2 ) * pow( y, 4 ) * z - 576 * pow( x, 2 ) * pow( y, 3 ) * pow( z, 2 ) -
                576 * pow( x, 2 ) * pow( y, 2 ) * pow( z, 3 ) - 276 * pow( x, 2 ) * y * pow( z, 4 ) -
                276 * x * pow( y, 4 ) * pow( z, 2 ) - 54 * x * pow( y, 3 ) * pow( z, 3 ) -
                276 * x * pow( y, 2 ) * pow( z, 4 ) - 48 * pow( y, 4 ) * pow( z, 3 ) - 48 * pow( y, 3 ) * pow( z, 4 ) +
                24 * pow( x, 4 ) * pow( y, 4 ) + 24 * pow( x, 4 ) * pow( y, 3 ) * z +
                288 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 2 ) + 24 * pow( x, 4 ) * y * pow( z, 3 ) +
                24 * pow( x, 4 ) * pow( z, 4 ) + 24 * pow( x, 3 ) * pow( y, 4 ) * z +
                48 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 2 ) + 48 * pow( x, 3 ) * pow( y, 2 ) * pow( z, 3 ) +
                24 * pow( x, 3 ) * y * pow( z, 4 ) + 288 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 2 ) +
                48 * pow( x, 2 ) * pow( y, 3 ) * pow( z, 3 ) + 288 * pow( x, 2 ) * pow( y, 2 ) * pow( z, 4 ) +
                24 * x * pow( y, 4 ) * pow( z, 3 ) + 24 * x * pow( y, 3 ) * pow( z, 4 ) +
                24 * pow( y, 4 ) * pow( z, 4 ) + 36 * pow( x, 5 ) * pow( y, 3 ) * z +
                36 * pow( x, 5 ) * y * pow( z, 3 ) - 12 * pow( x, 4 ) * pow( y, 4 ) * z -
                24 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 2 ) - 24 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 3 ) -
                12 * pow( x, 4 ) * y * pow( z, 4 ) + 36 * pow( x, 3 ) * pow( y, 5 ) * z -
                24 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 2 ) + 360 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 3 ) -
                24 * pow( x, 3 ) * pow( y, 2 ) * pow( z, 4 ) + 36 * pow( x, 3 ) * y * pow( z, 5 ) -
                24 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 3 ) - 24 * pow( x, 2 ) * pow( y, 3 ) * pow( z, 4 ) +
                36 * x * pow( y, 5 ) * pow( z, 3 ) - 12 * x * pow( y, 4 ) * pow( z, 4 ) +
                36 * x * pow( y, 3 ) * pow( z, 5 ) - 18 * pow( x, 6 ) * pow( y, 3 ) * z -
                18 * pow( x, 6 ) * y * pow( z, 3 ) + 12 * pow( x, 4 ) * pow( y, 4 ) * pow( z, 2 ) -
                90 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 3 ) + 12 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 4 ) -
                18 * pow( x, 3 ) * pow( y, 6 ) * z - 90 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 3 ) -
                90 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 4 ) - 18 * pow( x, 3 ) * y * pow( z, 6 ) +
                12 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 4 ) - 18 * x * pow( y, 6 ) * pow( z, 3 ) -
                18 * x * pow( y, 3 ) * pow( z, 6 ) - 72 * pow( x, 7 ) * pow( y, 3 ) * z -
                72 * pow( x, 7 ) * y * pow( z, 3 ) - 216 * pow( x, 5 ) * pow( y, 5 ) * z -
                1944 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 3 ) - 216 * pow( x, 5 ) * y * pow( z, 5 ) -
                72 * pow( x, 3 ) * pow( y, 7 ) * z - 1944 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 3 ) -
                1944 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 5 ) - 72 * pow( x, 3 ) * y * pow( z, 7 ) -
                72 * x * pow( y, 7 ) * pow( z, 3 ) - 216 * x * pow( y, 5 ) * pow( z, 5 ) -
                72 * x * pow( y, 3 ) * pow( z, 7 ) + 72 * pow( x, 8 ) * pow( y, 3 ) * z +
                72 * pow( x, 8 ) * y * pow( z, 3 ) + 108 * pow( x, 6 ) * pow( y, 5 ) * z +
                1392 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 3 ) + 108 * pow( x, 6 ) * y * pow( z, 5 ) +
                108 * pow( x, 5 ) * pow( y, 6 ) * z + 540 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 3 ) +
                540 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 4 ) + 108 * pow( x, 5 ) * y * pow( z, 6 ) +
                540 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 3 ) + 540 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 5 ) +
                72 * pow( x, 3 ) * pow( y, 8 ) * z + 1392 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 3 ) +
                540 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 4 ) + 540 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 5 ) +
                1392 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 6 ) + 72 * pow( x, 3 ) * y * pow( z, 8 ) +
                72 * x * pow( y, 8 ) * pow( z, 3 ) + 108 * x * pow( y, 6 ) * pow( z, 5 ) +
                108 * x * pow( y, 5 ) * pow( z, 6 ) + 72 * x * pow( y, 3 ) * pow( z, 8 ) +
                30 * pow( x, 9 ) * pow( y, 3 ) * z + 30 * pow( x, 9 ) * y * pow( z, 3 ) +
                432 * pow( x, 7 ) * pow( y, 5 ) * z + 3240 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 3 ) +
                432 * pow( x, 7 ) * y * pow( z, 5 ) - 54 * pow( x, 6 ) * pow( y, 6 ) * z -
                270 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 3 ) - 270 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 4 ) -
                54 * pow( x, 6 ) * y * pow( z, 6 ) + 432 * pow( x, 5 ) * pow( y, 7 ) * z +
                10368 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 3 ) + 10368 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 5 ) +
                432 * pow( x, 5 ) * y * pow( z, 7 ) - 270 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 3 ) -
                270 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 6 ) + 30 * pow( x, 3 ) * pow( y, 9 ) * z +
                3240 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 3 ) - 270 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 4 ) +
                10368 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 5 ) - 270 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 6 ) +
                3240 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 7 ) + 30 * pow( x, 3 ) * y * pow( z, 9 ) +
                30 * x * pow( y, 9 ) * pow( z, 3 ) + 432 * x * pow( y, 7 ) * pow( z, 5 ) -
                54 * x * pow( y, 6 ) * pow( z, 6 ) + 432 * x * pow( y, 5 ) * pow( z, 7 ) +
                30 * x * pow( y, 3 ) * pow( z, 9 ) - 72 * pow( x, 10 ) * pow( y, 3 ) * z -
                72 * pow( x, 10 ) * y * pow( z, 3 ) - 432 * pow( x, 8 ) * pow( y, 5 ) * z -
                3960 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 3 ) - 432 * pow( x, 8 ) * y * pow( z, 5 ) -
                216 * pow( x, 7 ) * pow( y, 6 ) * z - 1080 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 3 ) -
                1080 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 4 ) - 216 * pow( x, 7 ) * y * pow( z, 6 ) -
                216 * pow( x, 6 ) * pow( y, 7 ) * z - 7704 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 3 ) -
                7704 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 5 ) - 216 * pow( x, 6 ) * y * pow( z, 7 ) -
                432 * pow( x, 5 ) * pow( y, 8 ) * z - 7704 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 3 ) -
                3240 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 4 ) - 3240 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 5 ) -
                7704 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 6 ) - 432 * pow( x, 5 ) * y * pow( z, 8 ) -
                1080 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 3 ) - 3240 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 5 ) -
                1080 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 7 ) - 72 * pow( x, 3 ) * pow( y, 10 ) * z -
                3960 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 3 ) - 1080 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 4 ) -
                7704 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 5 ) - 7704 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 6 ) -
                1080 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 7 ) - 3960 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 8 ) -
                72 * pow( x, 3 ) * y * pow( z, 10 ) - 72 * x * pow( y, 10 ) * pow( z, 3 ) -
                432 * x * pow( y, 8 ) * pow( z, 5 ) - 216 * x * pow( y, 7 ) * pow( z, 6 ) -
                216 * x * pow( y, 6 ) * pow( z, 7 ) - 432 * x * pow( y, 5 ) * pow( z, 8 ) -
                72 * x * pow( y, 3 ) * pow( z, 10 ) + 36 * pow( x, 11 ) * pow( y, 3 ) * z +
                36 * pow( x, 11 ) * y * pow( z, 3 ) - 180 * pow( x, 9 ) * pow( y, 5 ) * z -
                540 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 3 ) - 180 * pow( x, 9 ) * y * pow( z, 5 ) +
                216 * pow( x, 8 ) * pow( y, 6 ) * z + 1080 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 3 ) +
                1080 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 4 ) + 216 * pow( x, 8 ) * y * pow( z, 6 ) -
                864 * pow( x, 7 ) * pow( y, 7 ) * z - 16848 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 3 ) -
                16848 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 5 ) - 864 * pow( x, 7 ) * y * pow( z, 7 ) +
                216 * pow( x, 6 ) * pow( y, 8 ) * z + 5112 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 3 ) +
                1620 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 4 ) + 1620 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 5 ) +
                5112 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 6 ) + 216 * pow( x, 6 ) * y * pow( z, 8 ) -
                180 * pow( x, 5 ) * pow( y, 9 ) * z - 16848 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 3 ) +
                1620 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 4 ) - 54432 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 5 ) +
                1620 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 6 ) - 16848 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 7 ) -
                180 * pow( x, 5 ) * y * pow( z, 9 ) + 1080 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 3 ) +
                1620 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 5 ) + 1620 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 6 ) +
                1080 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 8 ) + 36 * pow( x, 3 ) * pow( y, 11 ) * z -
                540 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 3 ) + 1080 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 4 ) -
                16848 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 5 ) + 5112 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 6 ) -
                16848 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 7 ) + 1080 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 8 ) -
                540 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 9 ) + 36 * pow( x, 3 ) * y * pow( z, 11 ) +
                36 * x * pow( y, 11 ) * pow( z, 3 ) - 180 * x * pow( y, 9 ) * pow( z, 5 ) +
                216 * x * pow( y, 8 ) * pow( z, 6 ) - 864 * x * pow( y, 7 ) * pow( z, 7 ) +
                216 * x * pow( y, 6 ) * pow( z, 8 ) - 180 * x * pow( y, 5 ) * pow( z, 9 ) +
                36 * x * pow( y, 3 ) * pow( z, 11 ) - 6 * pow( x, 12 ) * pow( y, 3 ) * z -
                6 * pow( x, 12 ) * y * pow( z, 3 ) + 432 * pow( x, 10 ) * pow( y, 5 ) * z +
                2748 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 3 ) + 432 * pow( x, 10 ) * y * pow( z, 5 ) +
                90 * pow( x, 9 ) * pow( y, 6 ) * z + 450 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 3 ) +
                450 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 4 ) + 90 * pow( x, 9 ) * y * pow( z, 6 ) +
                864 * pow( x, 8 ) * pow( y, 7 ) * z + 21168 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 3 ) +
                21168 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 5 ) + 864 * pow( x, 8 ) * y * pow( z, 7 ) +
                864 * pow( x, 7 ) * pow( y, 8 ) * z + 13464 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 3 ) +
                6480 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 4 ) + 6480 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 5 ) +
                13464 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 6 ) + 864 * pow( x, 7 ) * y * pow( z, 8 ) +
                90 * pow( x, 6 ) * pow( y, 9 ) * z + 13464 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 3 ) -
                810 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 4 ) + 42336 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 5 ) -
                810 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 6 ) + 13464 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 7 ) +
                90 * pow( x, 6 ) * y * pow( z, 9 ) + 432 * pow( x, 5 ) * pow( y, 10 ) * z +
                21168 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 3 ) + 6480 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 4 ) +
                42336 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 5 ) + 42336 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 6 ) +
                6480 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 7 ) + 21168 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 8 ) +
                432 * pow( x, 5 ) * y * pow( z, 10 ) + 450 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 3 ) +
                6480 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 5 ) - 810 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 6 ) +
                6480 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 7 ) + 450 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 9 ) -
                6 * pow( x, 3 ) * pow( y, 12 ) * z + 2748 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 3 ) +
                450 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 4 ) + 21168 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 5 ) +
                13464 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 6 ) + 13464 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 7 ) +
                21168 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 8 ) + 450 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 9 ) +
                2748 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 10 ) - 6 * pow( x, 3 ) * y * pow( z, 12 ) -
                6 * x * pow( y, 12 ) * pow( z, 3 ) + 432 * x * pow( y, 10 ) * pow( z, 5 ) +
                90 * x * pow( y, 9 ) * pow( z, 6 ) + 864 * x * pow( y, 8 ) * pow( z, 7 ) +
                864 * x * pow( y, 7 ) * pow( z, 8 ) + 90 * x * pow( y, 6 ) * pow( z, 9 ) +
                432 * x * pow( y, 5 ) * pow( z, 10 ) - 6 * x * pow( y, 3 ) * pow( z, 12 ) -
                216 * pow( x, 11 ) * pow( y, 5 ) * z - 1440 * pow( x, 11 ) * pow( y, 3 ) * pow( z, 3 ) -
                216 * pow( x, 11 ) * y * pow( z, 5 ) - 216 * pow( x, 10 ) * pow( y, 6 ) * z -
                1080 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 3 ) - 1080 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 4 ) -
                216 * pow( x, 10 ) * y * pow( z, 6 ) + 360 * pow( x, 9 ) * pow( y, 7 ) * z +
                2160 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 3 ) + 2160 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 5 ) +
                360 * pow( x, 9 ) * y * pow( z, 7 ) - 864 * pow( x, 8 ) * pow( y, 8 ) * z -
                15624 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 3 ) - 6480 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 4 ) -
                6480 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 5 ) - 15624 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 6 ) -
                864 * pow( x, 8 ) * y * pow( z, 8 ) + 360 * pow( x, 7 ) * pow( y, 9 ) * z +
                25920 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 3 ) - 3240 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 4 ) +
                85536 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 5 ) - 3240 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 6 ) +
                25920 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 7 ) + 360 * pow( x, 7 ) * y * pow( z, 9 ) -
                216 * pow( x, 6 ) * pow( y, 10 ) * z - 15624 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 3 ) -
                3240 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 4 ) - 28728 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 5 ) -
                28728 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 6 ) - 3240 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 7 ) -
                15624 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 8 ) - 216 * pow( x, 6 ) * y * pow( z, 10 ) -
                216 * pow( x, 5 ) * pow( y, 11 ) * z + 2160 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 3 ) -
                6480 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 4 ) + 85536 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 5 ) -
                28728 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 6 ) + 85536 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 7 ) -
                6480 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 8 ) + 2160 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 9 ) -
                216 * pow( x, 5 ) * y * pow( z, 11 ) - 1080 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 3 ) -
                6480 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 5 ) - 3240 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 6 ) -
                3240 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 7 ) - 6480 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 8 ) -
                1080 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 10 ) - 1440 * pow( x, 3 ) * pow( y, 11 ) * pow( z, 3 ) -
                1080 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 4 ) + 2160 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 5 ) -
                15624 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 6 ) + 25920 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 7 ) -
                15624 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 8 ) + 2160 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 9 ) -
                1080 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 10 ) - 1440 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 11 ) -
                216 * x * pow( y, 11 ) * pow( z, 5 ) - 216 * x * pow( y, 10 ) * pow( z, 6 ) +
                360 * x * pow( y, 9 ) * pow( z, 7 ) - 864 * x * pow( y, 8 ) * pow( z, 8 ) +
                360 * x * pow( y, 7 ) * pow( z, 9 ) - 216 * x * pow( y, 6 ) * pow( z, 10 ) -
                216 * x * pow( y, 5 ) * pow( z, 11 ) + 36 * pow( x, 12 ) * pow( y, 5 ) * z +
                240 * pow( x, 12 ) * pow( y, 3 ) * pow( z, 3 ) + 36 * pow( x, 12 ) * y * pow( z, 5 ) +
                108 * pow( x, 11 ) * pow( y, 6 ) * z + 540 * pow( x, 11 ) * pow( y, 4 ) * pow( z, 3 ) +
                540 * pow( x, 11 ) * pow( y, 3 ) * pow( z, 4 ) + 108 * pow( x, 11 ) * y * pow( z, 6 ) -
                864 * pow( x, 10 ) * pow( y, 7 ) * z - 13896 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 3 ) -
                13896 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 5 ) - 864 * pow( x, 10 ) * y * pow( z, 7 ) -
                360 * pow( x, 9 ) * pow( y, 8 ) * z - 3180 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 3 ) -
                2700 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 4 ) - 2700 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 5 ) -
                3180 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 6 ) - 360 * pow( x, 9 ) * y * pow( z, 8 ) -
                360 * pow( x, 8 ) * pow( y, 9 ) * z - 34560 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 3 ) +
                3240 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 4 ) - 111456 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 5 ) +
                3240 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 6 ) - 34560 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 7 ) -
                360 * pow( x, 8 ) * y * pow( z, 9 ) - 864 * pow( x, 7 ) * pow( y, 10 ) * z -
                34560 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 3 ) - 12960 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 4 ) -
                73008 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 5 ) - 73008 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 6 ) -
                12960 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 7 ) - 34560 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 8 ) -
                864 * pow( x, 7 ) * y * pow( z, 10 ) + 108 * pow( x, 6 ) * pow( y, 11 ) * z -
                3180 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 3 ) + 3240 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 4 ) -
                73008 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 5 ) + 18144 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 6 ) -
                73008 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 7 ) + 3240 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 8 ) -
                3180 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 9 ) + 108 * pow( x, 6 ) * y * pow( z, 11 ) +
                36 * pow( x, 5 ) * pow( y, 12 ) * z - 13896 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 3 ) -
                2700 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 4 ) - 111456 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 5 ) -
                73008 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 6 ) - 73008 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 7 ) -
                111456 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 8 ) - 2700 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 9 ) -
                13896 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 10 ) + 36 * pow( x, 5 ) * y * pow( z, 12 ) +
                540 * pow( x, 4 ) * pow( y, 11 ) * pow( z, 3 ) - 2700 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 5 ) +
                3240 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 6 ) - 12960 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 7 ) +
                3240 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 8 ) - 2700 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 9 ) +
                540 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 11 ) + 240 * pow( x, 3 ) * pow( y, 12 ) * pow( z, 3 ) +
                540 * pow( x, 3 ) * pow( y, 11 ) * pow( z, 4 ) - 13896 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 5 ) -
                3180 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 6 ) - 34560 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 7 ) -
                34560 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 8 ) - 3180 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 9 ) -
                13896 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 10 ) + 540 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 11 ) +
                240 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 12 ) + 36 * x * pow( y, 12 ) * pow( z, 5 ) +
                108 * x * pow( y, 11 ) * pow( z, 6 ) - 864 * x * pow( y, 10 ) * pow( z, 7 ) -
                360 * x * pow( y, 9 ) * pow( z, 8 ) - 360 * x * pow( y, 8 ) * pow( z, 9 ) -
                864 * x * pow( y, 7 ) * pow( z, 10 ) + 108 * x * pow( y, 6 ) * pow( z, 11 ) +
                36 * x * pow( y, 5 ) * pow( z, 12 ) - 18 * pow( x, 12 ) * pow( y, 6 ) * z -
                90 * pow( x, 12 ) * pow( y, 4 ) * pow( z, 3 ) - 90 * pow( x, 12 ) * pow( y, 3 ) * pow( z, 4 ) -
                18 * pow( x, 12 ) * y * pow( z, 6 ) + 432 * pow( x, 11 ) * pow( y, 7 ) * z +
                7344 * pow( x, 11 ) * pow( y, 5 ) * pow( z, 3 ) + 7344 * pow( x, 11 ) * pow( y, 3 ) * pow( z, 5 ) +
                432 * pow( x, 11 ) * y * pow( z, 7 ) + 864 * pow( x, 10 ) * pow( y, 8 ) * z +
                11988 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 3 ) + 6480 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 4 ) +
                6480 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 5 ) + 11988 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 6 ) +
                864 * pow( x, 10 ) * y * pow( z, 8 ) - 150 * pow( x, 9 ) * pow( y, 9 ) * z -
                1080 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 3 ) + 1350 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 4 ) -
                6480 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 5 ) + 1350 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 6 ) -
                1080 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 7 ) - 150 * pow( x, 9 ) * y * pow( z, 9 ) +
                864 * pow( x, 8 ) * pow( y, 10 ) * z + 43200 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 3 ) +
                12960 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 4 ) + 85968 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 5 ) +
                85968 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 6 ) + 12960 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 7 ) +
                43200 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 8 ) + 864 * pow( x, 8 ) * y * pow( z, 10 ) +
                432 * pow( x, 7 ) * pow( y, 11 ) * z - 1080 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 3 ) +
                12960 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 4 ) - 124416 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 5 ) +
                51624 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 6 ) - 124416 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 7 ) +
                12960 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 8 ) - 1080 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 9 ) +
                432 * pow( x, 7 ) * y * pow( z, 11 ) - 18 * pow( x, 6 ) * pow( y, 12 ) * z +
                11988 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 3 ) + 1350 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 4 ) +
                85968 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 5 ) + 51624 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 6 ) +
                51624 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 7 ) + 85968 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 8 ) +
                1350 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 9 ) + 11988 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 10 ) -
                18 * pow( x, 6 ) * y * pow( z, 12 ) + 7344 * pow( x, 5 ) * pow( y, 11 ) * pow( z, 3 ) +
                6480 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 4 ) - 6480 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 5 ) +
                85968 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 6 ) - 124416 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 7 ) +
                85968 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 8 ) - 6480 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 9 ) +
                6480 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 10 ) + 7344 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 11 ) -
                90 * pow( x, 4 ) * pow( y, 12 ) * pow( z, 3 ) + 6480 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 5 ) +
                1350 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 6 ) + 12960 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 7 ) +
                12960 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 8 ) + 1350 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 9 ) +
                6480 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 10 ) - 90 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 12 ) -
                90 * pow( x, 3 ) * pow( y, 12 ) * pow( z, 4 ) + 7344 * pow( x, 3 ) * pow( y, 11 ) * pow( z, 5 ) +
                11988 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 6 ) - 1080 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 7 ) +
                43200 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 8 ) - 1080 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 9 ) +
                11988 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 10 ) + 7344 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 11 ) -
                90 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 12 ) - 18 * x * pow( y, 12 ) * pow( z, 6 ) +
                432 * x * pow( y, 11 ) * pow( z, 7 ) + 864 * x * pow( y, 10 ) * pow( z, 8 ) -
                150 * x * pow( y, 9 ) * pow( z, 9 ) + 864 * x * pow( y, 8 ) * pow( z, 10 ) +
                432 * x * pow( y, 7 ) * pow( z, 11 ) - 18 * x * pow( y, 6 ) * pow( z, 12 ) -
                72 * pow( x, 12 ) * pow( y, 7 ) * z - 1224 * pow( x, 12 ) * pow( y, 5 ) * pow( z, 3 ) -
                1224 * pow( x, 12 ) * pow( y, 3 ) * pow( z, 5 ) - 72 * pow( x, 12 ) * y * pow( z, 7 ) -
                432 * pow( x, 11 ) * pow( y, 8 ) * z - 6192 * pow( x, 11 ) * pow( y, 6 ) * pow( z, 3 ) -
                3240 * pow( x, 11 ) * pow( y, 5 ) * pow( z, 4 ) - 3240 * pow( x, 11 ) * pow( y, 4 ) * pow( z, 5 ) -
                6192 * pow( x, 11 ) * pow( y, 3 ) * pow( z, 6 ) - 432 * pow( x, 11 ) * y * pow( z, 8 ) +
                360 * pow( x, 10 ) * pow( y, 9 ) * z + 20016 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 3 ) -
                3240 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 4 ) + 67824 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 5 ) -
                3240 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 6 ) + 20016 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 7 ) +
                360 * pow( x, 10 ) * y * pow( z, 9 ) + 360 * pow( x, 9 ) * pow( y, 10 ) * z +
                4680 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 3 ) + 5400 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 4 ) +
                15840 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 5 ) + 15840 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 6 ) +
                5400 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 7 ) + 4680 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 8 ) +
                360 * pow( x, 9 ) * y * pow( z, 10 ) - 432 * pow( x, 8 ) * pow( y, 11 ) * z +
                4680 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 3 ) - 12960 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 4 ) +
                176256 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 5 ) - 58104 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 6 ) +
                176256 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 7 ) - 12960 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 8 ) +
                4680 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 9 ) - 432 * pow( x, 8 ) * y * pow( z, 11 ) -
                72 * pow( x, 7 ) * pow( y, 12 ) * z + 20016 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 3 ) +
                5400 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 4 ) + 176256 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 5 ) +
                122688 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 6 ) + 122688 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 7 ) +
                176256 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 8 ) + 5400 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 9 ) +
                20016 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 10 ) - 72 * pow( x, 7 ) * y * pow( z, 12 ) -
                6192 * pow( x, 6 ) * pow( y, 11 ) * pow( z, 3 ) - 3240 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 4 ) +
                15840 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 5 ) - 58104 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 6 ) +
                122688 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 7 ) - 58104 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 8 ) +
                15840 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 9 ) - 3240 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 10 ) -
                6192 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 11 ) - 1224 * pow( x, 5 ) * pow( y, 12 ) * pow( z, 3 ) -
                3240 * pow( x, 5 ) * pow( y, 11 ) * pow( z, 4 ) + 67824 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 5 ) +
                15840 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 6 ) + 176256 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 7 ) +
                176256 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 8 ) + 15840 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 9 ) +
                67824 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 10 ) - 3240 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 11 ) -
                1224 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 12 ) - 3240 * pow( x, 4 ) * pow( y, 11 ) * pow( z, 5 ) -
                3240 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 6 ) + 5400 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 7 ) -
                12960 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 8 ) + 5400 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 9 ) -
                3240 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 10 ) - 3240 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 11 ) -
                1224 * pow( x, 3 ) * pow( y, 12 ) * pow( z, 5 ) - 6192 * pow( x, 3 ) * pow( y, 11 ) * pow( z, 6 ) +
                20016 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 7 ) + 4680 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 8 ) +
                4680 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 9 ) + 20016 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 10 ) -
                6192 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 11 ) - 1224 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 12 ) -
                72 * x * pow( y, 12 ) * pow( z, 7 ) - 432 * x * pow( y, 11 ) * pow( z, 8 ) +
                360 * x * pow( y, 10 ) * pow( z, 9 ) + 360 * x * pow( y, 9 ) * pow( z, 10 ) -
                432 * x * pow( y, 8 ) * pow( z, 11 ) - 72 * x * pow( y, 7 ) * pow( z, 12 ) +
                72 * pow( x, 12 ) * pow( y, 8 ) * z + 1032 * pow( x, 12 ) * pow( y, 6 ) * pow( z, 3 ) +
                540 * pow( x, 12 ) * pow( y, 5 ) * pow( z, 4 ) + 540 * pow( x, 12 ) * pow( y, 4 ) * pow( z, 5 ) +
                1032 * pow( x, 12 ) * pow( y, 3 ) * pow( z, 6 ) + 72 * pow( x, 12 ) * y * pow( z, 8 ) -
                180 * pow( x, 11 ) * pow( y, 9 ) * z - 10800 * pow( x, 11 ) * pow( y, 7 ) * pow( z, 3 ) +
                1620 * pow( x, 11 ) * pow( y, 6 ) * pow( z, 4 ) - 36288 * pow( x, 11 ) * pow( y, 5 ) * pow( z, 5 ) +
                1620 * pow( x, 11 ) * pow( y, 4 ) * pow( z, 6 ) - 10800 * pow( x, 11 ) * pow( y, 3 ) * pow( z, 7 ) -
                180 * pow( x, 11 ) * y * pow( z, 9 ) - 864 * pow( x, 10 ) * pow( y, 10 ) * z -
                28656 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 3 ) - 12960 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 4 ) -
                64152 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 5 ) - 64152 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 6 ) -
                12960 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 7 ) - 28656 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 8 ) -
                864 * pow( x, 10 ) * y * pow( z, 10 ) - 180 * pow( x, 9 ) * pow( y, 11 ) * z -
                3600 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 3 ) - 5400 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 4 ) -
                6480 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 5 ) - 14220 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 6 ) -
                6480 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 7 ) - 5400 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 8 ) -
                3600 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 9 ) - 180 * pow( x, 9 ) * y * pow( z, 11 ) +
                72 * pow( x, 8 ) * pow( y, 12 ) * z - 28656 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 3 ) -
                5400 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 4 ) - 228096 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 5 ) -
                148608 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 6 ) - 148608 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 7 ) -
                228096 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 8 ) - 5400 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 9 ) -
                28656 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 10 ) + 72 * pow( x, 8 ) * y * pow( z, 12 ) -
                10800 * pow( x, 7 ) * pow( y, 11 ) * pow( z, 3 ) - 12960 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 4 ) -
                6480 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 5 ) - 148608 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 6 ) +
                155520 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 7 ) - 148608 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 8 ) -
                6480 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 9 ) - 12960 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 10 ) -
                10800 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 11 ) + 1032 * pow( x, 6 ) * pow( y, 12 ) * pow( z, 3 ) +
                1620 * pow( x, 6 ) * pow( y, 11 ) * pow( z, 4 ) - 64152 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 5 ) -
                14220 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 6 ) - 148608 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 7 ) -
                148608 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 8 ) - 14220 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 9 ) -
                64152 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 10 ) + 1620 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 11 ) +
                1032 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 12 ) + 540 * pow( x, 5 ) * pow( y, 12 ) * pow( z, 4 ) -
                36288 * pow( x, 5 ) * pow( y, 11 ) * pow( z, 5 ) - 64152 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 6 ) -
                6480 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 7 ) - 228096 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 8 ) -
                6480 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 9 ) - 64152 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 10 ) -
                36288 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 11 ) + 540 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 12 ) +
                540 * pow( x, 4 ) * pow( y, 12 ) * pow( z, 5 ) + 1620 * pow( x, 4 ) * pow( y, 11 ) * pow( z, 6 ) -
                12960 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 7 ) - 5400 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 8 ) -
                5400 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 9 ) - 12960 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 10 ) +
                1620 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 11 ) + 540 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 12 ) +
                1032 * pow( x, 3 ) * pow( y, 12 ) * pow( z, 6 ) - 10800 * pow( x, 3 ) * pow( y, 11 ) * pow( z, 7 ) -
                28656 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 8 ) - 3600 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 9 ) -
                28656 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 10 ) - 10800 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 11 ) +
                1032 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 12 ) + 72 * x * pow( y, 12 ) * pow( z, 8 ) -
                180 * x * pow( y, 11 ) * pow( z, 9 ) - 864 * x * pow( y, 10 ) * pow( z, 10 ) -
                180 * x * pow( y, 9 ) * pow( z, 11 ) + 72 * x * pow( y, 8 ) * pow( z, 12 ) +
                30 * pow( x, 12 ) * pow( y, 9 ) * z + 1800 * pow( x, 12 ) * pow( y, 7 ) * pow( z, 3 ) -
                270 * pow( x, 12 ) * pow( y, 6 ) * pow( z, 4 ) + 6048 * pow( x, 12 ) * pow( y, 5 ) * pow( z, 5 ) -
                270 * pow( x, 12 ) * pow( y, 4 ) * pow( z, 6 ) + 1800 * pow( x, 12 ) * pow( y, 3 ) * pow( z, 7 ) +
                30 * pow( x, 12 ) * y * pow( z, 9 ) + 432 * pow( x, 11 ) * pow( y, 10 ) * z +
                15120 * pow( x, 11 ) * pow( y, 8 ) * pow( z, 3 ) + 6480 * pow( x, 11 ) * pow( y, 7 ) * pow( z, 4 ) +
                33264 * pow( x, 11 ) * pow( y, 6 ) * pow( z, 5 ) + 33264 * pow( x, 11 ) * pow( y, 5 ) * pow( z, 6 ) +
                6480 * pow( x, 11 ) * pow( y, 4 ) * pow( z, 7 ) + 15120 * pow( x, 11 ) * pow( y, 3 ) * pow( z, 8 ) +
                432 * pow( x, 11 ) * y * pow( z, 10 ) + 432 * pow( x, 10 ) * pow( y, 11 ) * z +
                1380 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 3 ) + 12960 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 4 ) -
                88992 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 5 ) + 47196 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 6 ) -
                88992 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 7 ) + 12960 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 8 ) +
                1380 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 9 ) + 432 * pow( x, 10 ) * y * pow( z, 11 ) +
                30 * pow( x, 9 ) * pow( y, 12 ) * z + 1380 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 3 ) -
                2250 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 4 ) - 15120 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 5 ) -
                21960 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 6 ) - 21960 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 7 ) -
                15120 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 8 ) - 2250 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 9 ) +
                1380 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 10 ) + 30 * pow( x, 9 ) * y * pow( z, 12 ) +
                15120 * pow( x, 8 ) * pow( y, 11 ) * pow( z, 3 ) + 12960 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 4 ) -
                15120 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 5 ) + 174528 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 6 ) -
                259200 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 7 ) + 174528 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 8 ) -
                15120 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 9 ) + 12960 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 10 ) +
                15120 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 11 ) + 1800 * pow( x, 7 ) * pow( y, 12 ) * pow( z, 3 ) +
                6480 * pow( x, 7 ) * pow( y, 11 ) * pow( z, 4 ) - 88992 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 5 ) -
                21960 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 6 ) - 259200 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 7 ) -
                259200 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 8 ) - 21960 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 9 ) -
                88992 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 10 ) + 6480 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 11 ) +
                1800 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 12 ) - 270 * pow( x, 6 ) * pow( y, 12 ) * pow( z, 4 ) +
                33264 * pow( x, 6 ) * pow( y, 11 ) * pow( z, 5 ) + 47196 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 6 ) -
                21960 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 7 ) + 174528 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 8 ) -
                21960 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 9 ) + 47196 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 10 ) +
                33264 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 11 ) - 270 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 12 ) +
                6048 * pow( x, 5 ) * pow( y, 12 ) * pow( z, 5 ) + 33264 * pow( x, 5 ) * pow( y, 11 ) * pow( z, 6 ) -
                88992 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 7 ) - 15120 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 8 ) -
                15120 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 9 ) - 88992 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 10 ) +
                33264 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 11 ) + 6048 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 12 ) -
                270 * pow( x, 4 ) * pow( y, 12 ) * pow( z, 6 ) + 6480 * pow( x, 4 ) * pow( y, 11 ) * pow( z, 7 ) +
                12960 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 8 ) - 2250 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 9 ) +
                12960 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 10 ) + 6480 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 11 ) -
                270 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 12 ) + 1800 * pow( x, 3 ) * pow( y, 12 ) * pow( z, 7 ) +
                15120 * pow( x, 3 ) * pow( y, 11 ) * pow( z, 8 ) + 1380 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 9 ) +
                1380 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 10 ) + 15120 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 11 ) +
                1800 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 12 ) + 30 * x * pow( y, 12 ) * pow( z, 9 ) +
                432 * x * pow( y, 11 ) * pow( z, 10 ) + 432 * x * pow( y, 10 ) * pow( z, 11 ) +
                30 * x * pow( y, 9 ) * pow( z, 12 ) - 72 * pow( x, 12 ) * pow( y, 10 ) * z -
                2520 * pow( x, 12 ) * pow( y, 8 ) * pow( z, 3 ) - 1080 * pow( x, 12 ) * pow( y, 7 ) * pow( z, 4 ) -
                5544 * pow( x, 12 ) * pow( y, 6 ) * pow( z, 5 ) - 5544 * pow( x, 12 ) * pow( y, 5 ) * pow( z, 6 ) -
                1080 * pow( x, 12 ) * pow( y, 4 ) * pow( z, 7 ) - 2520 * pow( x, 12 ) * pow( y, 3 ) * pow( z, 8 ) -
                72 * pow( x, 12 ) * y * pow( z, 10 ) - 216 * pow( x, 11 ) * pow( y, 11 ) * z -
                360 * pow( x, 11 ) * pow( y, 9 ) * pow( z, 3 ) - 6480 * pow( x, 11 ) * pow( y, 8 ) * pow( z, 4 ) +
                49248 * pow( x, 11 ) * pow( y, 7 ) * pow( z, 5 ) - 24192 * pow( x, 11 ) * pow( y, 6 ) * pow( z, 6 ) +
                49248 * pow( x, 11 ) * pow( y, 5 ) * pow( z, 7 ) - 6480 * pow( x, 11 ) * pow( y, 4 ) * pow( z, 8 ) -
                360 * pow( x, 11 ) * pow( y, 3 ) * pow( z, 9 ) - 216 * pow( x, 11 ) * y * pow( z, 11 ) -
                72 * pow( x, 10 ) * pow( y, 12 ) * z + 14112 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 3 ) +
                5400 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 4 ) + 140832 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 5 ) +
                104976 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 6 ) + 104976 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 7 ) +
                140832 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 8 ) + 5400 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 9 ) +
                14112 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 10 ) - 72 * pow( x, 10 ) * y * pow( z, 12 ) -
                360 * pow( x, 9 ) * pow( y, 11 ) * pow( z, 3 ) + 5400 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 4 ) +
                27000 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 5 ) + 32760 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 6 ) +
                51840 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 7 ) + 32760 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 8 ) +
                27000 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 9 ) + 5400 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 10 ) -
                360 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 11 ) - 2520 * pow( x, 8 ) * pow( y, 12 ) * pow( z, 3 ) -
                6480 * pow( x, 8 ) * pow( y, 11 ) * pow( z, 4 ) + 140832 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 5 ) +
                32760 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 6 ) + 362880 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 7 ) +
                362880 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 8 ) + 32760 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 9 ) +
                140832 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 10 ) - 6480 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 11 ) -
                2520 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 12 ) - 1080 * pow( x, 7 ) * pow( y, 12 ) * pow( z, 4 ) +
                49248 * pow( x, 7 ) * pow( y, 11 ) * pow( z, 5 ) + 104976 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 6 ) +
                51840 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 7 ) + 362880 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 8 ) +
                51840 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 9 ) + 104976 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 10 ) +
                49248 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 11 ) - 1080 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 12 ) -
                5544 * pow( x, 6 ) * pow( y, 12 ) * pow( z, 5 ) - 24192 * pow( x, 6 ) * pow( y, 11 ) * pow( z, 6 ) +
                104976 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 7 ) + 32760 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 8 ) +
                32760 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 9 ) + 104976 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 10 ) -
                24192 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 11 ) - 5544 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 12 ) -
                5544 * pow( x, 5 ) * pow( y, 12 ) * pow( z, 6 ) + 49248 * pow( x, 5 ) * pow( y, 11 ) * pow( z, 7 ) +
                140832 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 8 ) + 27000 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 9 ) +
                140832 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 10 ) + 49248 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 11 ) -
                5544 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 12 ) - 1080 * pow( x, 4 ) * pow( y, 12 ) * pow( z, 7 ) -
                6480 * pow( x, 4 ) * pow( y, 11 ) * pow( z, 8 ) + 5400 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 9 ) +
                5400 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 10 ) - 6480 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 11 ) -
                1080 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 12 ) - 2520 * pow( x, 3 ) * pow( y, 12 ) * pow( z, 8 ) -
                360 * pow( x, 3 ) * pow( y, 11 ) * pow( z, 9 ) + 14112 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 10 ) -
                360 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 11 ) - 2520 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 12 ) -
                72 * x * pow( y, 12 ) * pow( z, 10 ) - 216 * x * pow( y, 11 ) * pow( z, 11 ) -
                72 * x * pow( y, 10 ) * pow( z, 12 ) + 36 * pow( x, 12 ) * pow( y, 11 ) * z +
                60 * pow( x, 12 ) * pow( y, 9 ) * pow( z, 3 ) + 1080 * pow( x, 12 ) * pow( y, 8 ) * pow( z, 4 ) -
                8208 * pow( x, 12 ) * pow( y, 7 ) * pow( z, 5 ) + 4032 * pow( x, 12 ) * pow( y, 6 ) * pow( z, 6 ) -
                8208 * pow( x, 12 ) * pow( y, 5 ) * pow( z, 7 ) + 1080 * pow( x, 12 ) * pow( y, 4 ) * pow( z, 8 ) +
                60 * pow( x, 12 ) * pow( y, 3 ) * pow( z, 9 ) + 36 * pow( x, 12 ) * y * pow( z, 11 ) +
                36 * pow( x, 11 ) * pow( y, 12 ) * z - 7848 * pow( x, 11 ) * pow( y, 10 ) * pow( z, 3 ) -
                2700 * pow( x, 11 ) * pow( y, 9 ) * pow( z, 4 ) - 75168 * pow( x, 11 ) * pow( y, 8 ) * pow( z, 5 ) -
                54864 * pow( x, 11 ) * pow( y, 7 ) * pow( z, 6 ) - 54864 * pow( x, 11 ) * pow( y, 6 ) * pow( z, 7 ) -
                75168 * pow( x, 11 ) * pow( y, 5 ) * pow( z, 8 ) - 2700 * pow( x, 11 ) * pow( y, 4 ) * pow( z, 9 ) -
                7848 * pow( x, 11 ) * pow( y, 3 ) * pow( z, 10 ) + 36 * pow( x, 11 ) * y * pow( z, 12 ) -
                7848 * pow( x, 10 ) * pow( y, 11 ) * pow( z, 3 ) - 12960 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 4 ) -
                21240 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 5 ) - 130896 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 6 ) +
                84672 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 7 ) - 130896 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 8 ) -
                21240 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 9 ) - 12960 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 10 ) -
                7848 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 11 ) + 60 * pow( x, 9 ) * pow( y, 12 ) * pow( z, 3 ) -
                2700 * pow( x, 9 ) * pow( y, 11 ) * pow( z, 4 ) - 21240 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 5 ) -
                3000 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 6 ) - 8640 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 7 ) -
                8640 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 8 ) - 3000 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 9 ) -
                21240 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 10 ) - 2700 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 11 ) +
                60 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 12 ) + 1080 * pow( x, 8 ) * pow( y, 12 ) * pow( z, 4 ) -
                75168 * pow( x, 8 ) * pow( y, 11 ) * pow( z, 5 ) - 130896 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 6 ) -
                8640 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 7 ) - 466560 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 8 ) -
                8640 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 9 ) - 130896 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 10 ) -
                75168 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 11 ) + 1080 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 12 ) -
                8208 * pow( x, 7 ) * pow( y, 12 ) * pow( z, 5 ) - 54864 * pow( x, 7 ) * pow( y, 11 ) * pow( z, 6 ) +
                84672 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 7 ) - 8640 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 8 ) -
                8640 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 9 ) + 84672 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 10 ) -
                54864 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 11 ) - 8208 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 12 ) +
                4032 * pow( x, 6 ) * pow( y, 12 ) * pow( z, 6 ) - 54864 * pow( x, 6 ) * pow( y, 11 ) * pow( z, 7 ) -
                130896 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 8 ) - 3000 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 9 ) -
                130896 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 10 ) - 54864 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 11 ) +
                4032 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 12 ) - 8208 * pow( x, 5 ) * pow( y, 12 ) * pow( z, 7 ) -
                75168 * pow( x, 5 ) * pow( y, 11 ) * pow( z, 8 ) - 21240 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 9 ) -
                21240 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 10 ) - 75168 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 11 ) -
                8208 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 12 ) + 1080 * pow( x, 4 ) * pow( y, 12 ) * pow( z, 8 ) -
                2700 * pow( x, 4 ) * pow( y, 11 ) * pow( z, 9 ) - 12960 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 10 ) -
                2700 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 11 ) + 1080 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 12 ) +
                60 * pow( x, 3 ) * pow( y, 12 ) * pow( z, 9 ) - 7848 * pow( x, 3 ) * pow( y, 11 ) * pow( z, 10 ) -
                7848 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 11 ) + 60 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 12 ) +
                36 * x * pow( y, 12 ) * pow( z, 11 ) + 36 * x * pow( y, 11 ) * pow( z, 12 ) -
                6 * pow( x, 12 ) * pow( y, 12 ) * z + 1308 * pow( x, 12 ) * pow( y, 10 ) * pow( z, 3 ) +
                450 * pow( x, 12 ) * pow( y, 9 ) * pow( z, 4 ) + 12528 * pow( x, 12 ) * pow( y, 8 ) * pow( z, 5 ) +
                9144 * pow( x, 12 ) * pow( y, 7 ) * pow( z, 6 ) + 9144 * pow( x, 12 ) * pow( y, 6 ) * pow( z, 7 ) +
                12528 * pow( x, 12 ) * pow( y, 5 ) * pow( z, 8 ) + 450 * pow( x, 12 ) * pow( y, 4 ) * pow( z, 9 ) +
                1308 * pow( x, 12 ) * pow( y, 3 ) * pow( z, 10 ) - 6 * pow( x, 12 ) * y * pow( z, 12 ) +
                4320 * pow( x, 11 ) * pow( y, 11 ) * pow( z, 3 ) + 6480 * pow( x, 11 ) * pow( y, 10 ) * pow( z, 4 ) +
                8640 * pow( x, 11 ) * pow( y, 9 ) * pow( z, 5 ) + 67824 * pow( x, 11 ) * pow( y, 8 ) * pow( z, 6 ) -
                51840 * pow( x, 11 ) * pow( y, 7 ) * pow( z, 7 ) + 67824 * pow( x, 11 ) * pow( y, 6 ) * pow( z, 8 ) +
                8640 * pow( x, 11 ) * pow( y, 5 ) * pow( z, 9 ) + 6480 * pow( x, 11 ) * pow( y, 4 ) * pow( z, 10 ) +
                4320 * pow( x, 11 ) * pow( y, 3 ) * pow( z, 11 ) + 1308 * pow( x, 10 ) * pow( y, 12 ) * pow( z, 3 ) +
                6480 * pow( x, 10 ) * pow( y, 11 ) * pow( z, 4 ) - 53568 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 5 ) -
                14580 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 6 ) - 188352 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 7 ) -
                188352 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 8 ) - 14580 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 9 ) -
                53568 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 10 ) + 6480 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 11 ) +
                1308 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 12 ) + 450 * pow( x, 9 ) * pow( y, 12 ) * pow( z, 4 ) +
                8640 * pow( x, 9 ) * pow( y, 11 ) * pow( z, 5 ) - 14580 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 6 ) -
                70200 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 7 ) - 34560 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 8 ) -
                70200 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 9 ) - 14580 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 10 ) +
                8640 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 11 ) + 450 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 12 ) +
                12528 * pow( x, 8 ) * pow( y, 12 ) * pow( z, 5 ) + 67824 * pow( x, 8 ) * pow( y, 11 ) * pow( z, 6 ) -
                188352 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 7 ) - 34560 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 8 ) -
                34560 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 9 ) - 188352 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 10 ) +
                67824 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 11 ) + 12528 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 12 ) +
                9144 * pow( x, 7 ) * pow( y, 12 ) * pow( z, 6 ) - 51840 * pow( x, 7 ) * pow( y, 11 ) * pow( z, 7 ) -
                188352 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 8 ) - 70200 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 9 ) -
                188352 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 10 ) - 51840 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 11 ) +
                9144 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 12 ) + 9144 * pow( x, 6 ) * pow( y, 12 ) * pow( z, 7 ) +
                67824 * pow( x, 6 ) * pow( y, 11 ) * pow( z, 8 ) - 14580 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 9 ) -
                14580 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 10 ) + 67824 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 11 ) +
                9144 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 12 ) + 12528 * pow( x, 5 ) * pow( y, 12 ) * pow( z, 8 ) +
                8640 * pow( x, 5 ) * pow( y, 11 ) * pow( z, 9 ) - 53568 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 10 ) +
                8640 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 11 ) + 12528 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 12 ) +
                450 * pow( x, 4 ) * pow( y, 12 ) * pow( z, 9 ) + 6480 * pow( x, 4 ) * pow( y, 11 ) * pow( z, 10 ) +
                6480 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 11 ) + 450 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 12 ) +
                1308 * pow( x, 3 ) * pow( y, 12 ) * pow( z, 10 ) + 4320 * pow( x, 3 ) * pow( y, 11 ) * pow( z, 11 ) +
                1308 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 12 ) - 6 * x * pow( y, 12 ) * pow( z, 12 ) -
                720 * pow( x, 12 ) * pow( y, 11 ) * pow( z, 3 ) - 1080 * pow( x, 12 ) * pow( y, 10 ) * pow( z, 4 ) -
                1440 * pow( x, 12 ) * pow( y, 9 ) * pow( z, 5 ) - 11304 * pow( x, 12 ) * pow( y, 8 ) * pow( z, 6 ) +
                8640 * pow( x, 12 ) * pow( y, 7 ) * pow( z, 7 ) - 11304 * pow( x, 12 ) * pow( y, 6 ) * pow( z, 8 ) -
                1440 * pow( x, 12 ) * pow( y, 5 ) * pow( z, 9 ) - 1080 * pow( x, 12 ) * pow( y, 4 ) * pow( z, 10 ) -
                720 * pow( x, 12 ) * pow( y, 3 ) * pow( z, 11 ) - 720 * pow( x, 11 ) * pow( y, 12 ) * pow( z, 3 ) -
                3240 * pow( x, 11 ) * pow( y, 11 ) * pow( z, 4 ) + 31536 * pow( x, 11 ) * pow( y, 10 ) * pow( z, 5 ) +
                8280 * pow( x, 11 ) * pow( y, 9 ) * pow( z, 6 ) + 103680 * pow( x, 11 ) * pow( y, 8 ) * pow( z, 7 ) +
                103680 * pow( x, 11 ) * pow( y, 7 ) * pow( z, 8 ) + 8280 * pow( x, 11 ) * pow( y, 6 ) * pow( z, 9 ) +
                31536 * pow( x, 11 ) * pow( y, 5 ) * pow( z, 10 ) - 3240 * pow( x, 11 ) * pow( y, 4 ) * pow( z, 11 ) -
                720 * pow( x, 11 ) * pow( y, 3 ) * pow( z, 12 ) - 1080 * pow( x, 10 ) * pow( y, 12 ) * pow( z, 4 ) +
                31536 * pow( x, 10 ) * pow( y, 11 ) * pow( z, 5 ) + 87264 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 6 ) +
                81360 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 7 ) + 292032 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 8 ) +
                81360 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 9 ) + 87264 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 10 ) +
                31536 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 11 ) - 1080 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 12 ) -
                1440 * pow( x, 9 ) * pow( y, 12 ) * pow( z, 5 ) + 8280 * pow( x, 9 ) * pow( y, 11 ) * pow( z, 6 ) +
                81360 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 7 ) + 52200 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 8 ) +
                52200 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 9 ) + 81360 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 10 ) +
                8280 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 11 ) - 1440 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 12 ) -
                11304 * pow( x, 8 ) * pow( y, 12 ) * pow( z, 6 ) + 103680 * pow( x, 8 ) * pow( y, 11 ) * pow( z, 7 ) +
                292032 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 8 ) + 52200 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 9 ) +
                292032 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 10 ) + 103680 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 11 ) -
                11304 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 12 ) + 8640 * pow( x, 7 ) * pow( y, 12 ) * pow( z, 7 ) +
                103680 * pow( x, 7 ) * pow( y, 11 ) * pow( z, 8 ) + 81360 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 9 ) +
                81360 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 10 ) + 103680 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 11 ) +
                8640 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 12 ) - 11304 * pow( x, 6 ) * pow( y, 12 ) * pow( z, 8 ) +
                8280 * pow( x, 6 ) * pow( y, 11 ) * pow( z, 9 ) + 87264 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 10 ) +
                8280 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 11 ) - 11304 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 12 ) -
                1440 * pow( x, 5 ) * pow( y, 12 ) * pow( z, 9 ) + 31536 * pow( x, 5 ) * pow( y, 11 ) * pow( z, 10 ) +
                31536 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 11 ) - 1440 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 12 ) -
                1080 * pow( x, 4 ) * pow( y, 12 ) * pow( z, 10 ) - 3240 * pow( x, 4 ) * pow( y, 11 ) * pow( z, 11 ) -
                1080 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 12 ) - 720 * pow( x, 3 ) * pow( y, 12 ) * pow( z, 11 ) -
                720 * pow( x, 3 ) * pow( y, 11 ) * pow( z, 12 ) + 120 * pow( x, 12 ) * pow( y, 12 ) * pow( z, 3 ) +
                540 * pow( x, 12 ) * pow( y, 11 ) * pow( z, 4 ) - 5256 * pow( x, 12 ) * pow( y, 10 ) * pow( z, 5 ) -
                1380 * pow( x, 12 ) * pow( y, 9 ) * pow( z, 6 ) - 17280 * pow( x, 12 ) * pow( y, 8 ) * pow( z, 7 ) -
                17280 * pow( x, 12 ) * pow( y, 7 ) * pow( z, 8 ) - 1380 * pow( x, 12 ) * pow( y, 6 ) * pow( z, 9 ) -
                5256 * pow( x, 12 ) * pow( y, 5 ) * pow( z, 10 ) + 540 * pow( x, 12 ) * pow( y, 4 ) * pow( z, 11 ) +
                120 * pow( x, 12 ) * pow( y, 3 ) * pow( z, 12 ) + 540 * pow( x, 11 ) * pow( y, 12 ) * pow( z, 4 ) -
                18144 * pow( x, 11 ) * pow( y, 11 ) * pow( z, 5 ) - 46008 * pow( x, 11 ) * pow( y, 10 ) * pow( z, 6 ) -
                36720 * pow( x, 11 ) * pow( y, 9 ) * pow( z, 7 ) - 155520 * pow( x, 11 ) * pow( y, 8 ) * pow( z, 8 ) -
                36720 * pow( x, 11 ) * pow( y, 7 ) * pow( z, 9 ) - 46008 * pow( x, 11 ) * pow( y, 6 ) * pow( z, 10 ) -
                18144 * pow( x, 11 ) * pow( y, 5 ) * pow( z, 11 ) + 540 * pow( x, 11 ) * pow( y, 4 ) * pow( z, 12 ) -
                5256 * pow( x, 10 ) * pow( y, 12 ) * pow( z, 5 ) - 46008 * pow( x, 10 ) * pow( y, 11 ) * pow( z, 6 ) +
                13824 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 7 ) - 38160 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 8 ) -
                38160 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 9 ) + 13824 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 10 ) -
                46008 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 11 ) - 5256 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 12 ) -
                1380 * pow( x, 9 ) * pow( y, 12 ) * pow( z, 6 ) - 36720 * pow( x, 9 ) * pow( y, 11 ) * pow( z, 7 ) -
                38160 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 8 ) + 49500 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 9 ) -
                38160 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 10 ) - 36720 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 11 ) -
                1380 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 12 ) - 17280 * pow( x, 8 ) * pow( y, 12 ) * pow( z, 7 ) -
                155520 * pow( x, 8 ) * pow( y, 11 ) * pow( z, 8 ) - 38160 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 9 ) -
                38160 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 10 ) - 155520 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 11 ) -
                17280 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 12 ) - 17280 * pow( x, 7 ) * pow( y, 12 ) * pow( z, 8 ) -
                36720 * pow( x, 7 ) * pow( y, 11 ) * pow( z, 9 ) + 13824 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 10 ) -
                36720 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 11 ) - 17280 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 12 ) -
                1380 * pow( x, 6 ) * pow( y, 12 ) * pow( z, 9 ) - 46008 * pow( x, 6 ) * pow( y, 11 ) * pow( z, 10 ) -
                46008 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 11 ) - 1380 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 12 ) -
                5256 * pow( x, 5 ) * pow( y, 12 ) * pow( z, 10 ) - 18144 * pow( x, 5 ) * pow( y, 11 ) * pow( z, 11 ) -
                5256 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 12 ) + 540 * pow( x, 4 ) * pow( y, 12 ) * pow( z, 11 ) +
                540 * pow( x, 4 ) * pow( y, 11 ) * pow( z, 12 ) + 120 * pow( x, 3 ) * pow( y, 12 ) * pow( z, 12 ) -
                90 * pow( x, 12 ) * pow( y, 12 ) * pow( z, 4 ) + 3024 * pow( x, 12 ) * pow( y, 11 ) * pow( z, 5 ) +
                7668 * pow( x, 12 ) * pow( y, 10 ) * pow( z, 6 ) + 6120 * pow( x, 12 ) * pow( y, 9 ) * pow( z, 7 ) +
                25920 * pow( x, 12 ) * pow( y, 8 ) * pow( z, 8 ) + 6120 * pow( x, 12 ) * pow( y, 7 ) * pow( z, 9 ) +
                7668 * pow( x, 12 ) * pow( y, 6 ) * pow( z, 10 ) + 3024 * pow( x, 12 ) * pow( y, 5 ) * pow( z, 11 ) -
                90 * pow( x, 12 ) * pow( y, 4 ) * pow( z, 12 ) + 3024 * pow( x, 11 ) * pow( y, 12 ) * pow( z, 5 ) +
                24192 * pow( x, 11 ) * pow( y, 11 ) * pow( z, 6 ) - 16416 * pow( x, 11 ) * pow( y, 10 ) * pow( z, 7 ) +
                15120 * pow( x, 11 ) * pow( y, 9 ) * pow( z, 8 ) + 15120 * pow( x, 11 ) * pow( y, 8 ) * pow( z, 9 ) -
                16416 * pow( x, 11 ) * pow( y, 7 ) * pow( z, 10 ) + 24192 * pow( x, 11 ) * pow( y, 6 ) * pow( z, 11 ) +
                3024 * pow( x, 11 ) * pow( y, 5 ) * pow( z, 12 ) + 7668 * pow( x, 10 ) * pow( y, 12 ) * pow( z, 6 ) -
                16416 * pow( x, 10 ) * pow( y, 11 ) * pow( z, 7 ) - 117504 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 8 ) -
                82500 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 9 ) - 117504 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 10 ) -
                16416 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 11 ) + 7668 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 12 ) +
                6120 * pow( x, 9 ) * pow( y, 12 ) * pow( z, 7 ) + 15120 * pow( x, 9 ) * pow( y, 11 ) * pow( z, 8 ) -
                82500 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 9 ) - 82500 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 10 ) +
                15120 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 11 ) + 6120 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 12 ) +
                25920 * pow( x, 8 ) * pow( y, 12 ) * pow( z, 8 ) + 15120 * pow( x, 8 ) * pow( y, 11 ) * pow( z, 9 ) -
                117504 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 10 ) + 15120 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 11 ) +
                25920 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 12 ) + 6120 * pow( x, 7 ) * pow( y, 12 ) * pow( z, 9 ) -
                16416 * pow( x, 7 ) * pow( y, 11 ) * pow( z, 10 ) - 16416 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 11 ) +
                6120 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 12 ) + 7668 * pow( x, 6 ) * pow( y, 12 ) * pow( z, 10 ) +
                24192 * pow( x, 6 ) * pow( y, 11 ) * pow( z, 11 ) + 7668 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 12 ) +
                3024 * pow( x, 5 ) * pow( y, 12 ) * pow( z, 11 ) + 3024 * pow( x, 5 ) * pow( y, 11 ) * pow( z, 12 ) -
                90 * pow( x, 4 ) * pow( y, 12 ) * pow( z, 12 ) - 504 * pow( x, 12 ) * pow( y, 12 ) * pow( z, 5 ) -
                4032 * pow( x, 12 ) * pow( y, 11 ) * pow( z, 6 ) + 2736 * pow( x, 12 ) * pow( y, 10 ) * pow( z, 7 ) -
                2520 * pow( x, 12 ) * pow( y, 9 ) * pow( z, 8 ) - 2520 * pow( x, 12 ) * pow( y, 8 ) * pow( z, 9 ) +
                2736 * pow( x, 12 ) * pow( y, 7 ) * pow( z, 10 ) - 4032 * pow( x, 12 ) * pow( y, 6 ) * pow( z, 11 ) -
                504 * pow( x, 12 ) * pow( y, 5 ) * pow( z, 12 ) - 4032 * pow( x, 11 ) * pow( y, 12 ) * pow( z, 6 ) +
                12960 * pow( x, 11 ) * pow( y, 11 ) * pow( z, 7 ) + 68256 * pow( x, 11 ) * pow( y, 10 ) * pow( z, 8 ) +
                39600 * pow( x, 11 ) * pow( y, 9 ) * pow( z, 9 ) + 68256 * pow( x, 11 ) * pow( y, 8 ) * pow( z, 10 ) +
                12960 * pow( x, 11 ) * pow( y, 7 ) * pow( z, 11 ) - 4032 * pow( x, 11 ) * pow( y, 6 ) * pow( z, 12 ) +
                2736 * pow( x, 10 ) * pow( y, 12 ) * pow( z, 7 ) + 68256 * pow( x, 10 ) * pow( y, 11 ) * pow( z, 8 ) +
                110880 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 9 ) +
                110880 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 10 ) + 68256 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 11 ) +
                2736 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 12 ) - 2520 * pow( x, 9 ) * pow( y, 12 ) * pow( z, 8 ) +
                39600 * pow( x, 9 ) * pow( y, 11 ) * pow( z, 9 ) + 110880 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 10 ) +
                39600 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 11 ) - 2520 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 12 ) -
                2520 * pow( x, 8 ) * pow( y, 12 ) * pow( z, 9 ) + 68256 * pow( x, 8 ) * pow( y, 11 ) * pow( z, 10 ) +
                68256 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 11 ) - 2520 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 12 ) +
                2736 * pow( x, 7 ) * pow( y, 12 ) * pow( z, 10 ) + 12960 * pow( x, 7 ) * pow( y, 11 ) * pow( z, 11 ) +
                2736 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 12 ) - 4032 * pow( x, 6 ) * pow( y, 12 ) * pow( z, 11 ) -
                4032 * pow( x, 6 ) * pow( y, 11 ) * pow( z, 12 ) - 504 * pow( x, 5 ) * pow( y, 12 ) * pow( z, 12 ) +
                672 * pow( x, 12 ) * pow( y, 12 ) * pow( z, 6 ) - 2160 * pow( x, 12 ) * pow( y, 11 ) * pow( z, 7 ) -
                11376 * pow( x, 12 ) * pow( y, 10 ) * pow( z, 8 ) - 6600 * pow( x, 12 ) * pow( y, 9 ) * pow( z, 9 ) -
                11376 * pow( x, 12 ) * pow( y, 8 ) * pow( z, 10 ) - 2160 * pow( x, 12 ) * pow( y, 7 ) * pow( z, 11 ) +
                672 * pow( x, 12 ) * pow( y, 6 ) * pow( z, 12 ) - 2160 * pow( x, 11 ) * pow( y, 12 ) * pow( z, 7 ) -
                38880 * pow( x, 11 ) * pow( y, 11 ) * pow( z, 8 ) - 51480 * pow( x, 11 ) * pow( y, 10 ) * pow( z, 9 ) -
                51480 * pow( x, 11 ) * pow( y, 9 ) * pow( z, 10 ) - 38880 * pow( x, 11 ) * pow( y, 8 ) * pow( z, 11 ) -
                2160 * pow( x, 11 ) * pow( y, 7 ) * pow( z, 12 ) - 11376 * pow( x, 10 ) * pow( y, 12 ) * pow( z, 8 ) -
                51480 * pow( x, 10 ) * pow( y, 11 ) * pow( z, 9 ) - 57024 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 10 ) -
                51480 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 11 ) - 11376 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 12 ) -
                6600 * pow( x, 9 ) * pow( y, 12 ) * pow( z, 9 ) - 51480 * pow( x, 9 ) * pow( y, 11 ) * pow( z, 10 ) -
                51480 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 11 ) - 6600 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 12 ) -
                11376 * pow( x, 8 ) * pow( y, 12 ) * pow( z, 10 ) - 38880 * pow( x, 8 ) * pow( y, 11 ) * pow( z, 11 ) -
                11376 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 12 ) - 2160 * pow( x, 7 ) * pow( y, 12 ) * pow( z, 11 ) -
                2160 * pow( x, 7 ) * pow( y, 11 ) * pow( z, 12 ) + 672 * pow( x, 6 ) * pow( y, 12 ) * pow( z, 12 ) +
                360 * pow( x, 12 ) * pow( y, 12 ) * pow( z, 7 ) + 6480 * pow( x, 12 ) * pow( y, 11 ) * pow( z, 8 ) +
                8580 * pow( x, 12 ) * pow( y, 10 ) * pow( z, 9 ) + 8580 * pow( x, 12 ) * pow( y, 9 ) * pow( z, 10 ) +
                6480 * pow( x, 12 ) * pow( y, 8 ) * pow( z, 11 ) + 360 * pow( x, 12 ) * pow( y, 7 ) * pow( z, 12 ) +
                6480 * pow( x, 11 ) * pow( y, 12 ) * pow( z, 8 ) + 23760 * pow( x, 11 ) * pow( y, 11 ) * pow( z, 9 ) +
                19008 * pow( x, 11 ) * pow( y, 10 ) * pow( z, 10 ) + 23760 * pow( x, 11 ) * pow( y, 9 ) * pow( z, 11 ) +
                6480 * pow( x, 11 ) * pow( y, 8 ) * pow( z, 12 ) + 8580 * pow( x, 10 ) * pow( y, 12 ) * pow( z, 9 ) +
                19008 * pow( x, 10 ) * pow( y, 11 ) * pow( z, 10 ) +
                19008 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 11 ) + 8580 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 12 ) +
                8580 * pow( x, 9 ) * pow( y, 12 ) * pow( z, 10 ) + 23760 * pow( x, 9 ) * pow( y, 11 ) * pow( z, 11 ) +
                8580 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 12 ) + 6480 * pow( x, 8 ) * pow( y, 12 ) * pow( z, 11 ) +
                6480 * pow( x, 8 ) * pow( y, 11 ) * pow( z, 12 ) + 360 * pow( x, 7 ) * pow( y, 12 ) * pow( z, 12 ) -
                1080 * pow( x, 12 ) * pow( y, 12 ) * pow( z, 8 ) - 3960 * pow( x, 12 ) * pow( y, 11 ) * pow( z, 9 ) -
                3168 * pow( x, 12 ) * pow( y, 10 ) * pow( z, 10 ) - 3960 * pow( x, 12 ) * pow( y, 9 ) * pow( z, 11 ) -
                1080 * pow( x, 12 ) * pow( y, 8 ) * pow( z, 12 ) - 3960 * pow( x, 11 ) * pow( y, 12 ) * pow( z, 9 ) -
                4752 * pow( x, 11 ) * pow( y, 11 ) * pow( z, 10 ) - 4752 * pow( x, 11 ) * pow( y, 10 ) * pow( z, 11 ) -
                3960 * pow( x, 11 ) * pow( y, 9 ) * pow( z, 12 ) - 3168 * pow( x, 10 ) * pow( y, 12 ) * pow( z, 10 ) -
                4752 * pow( x, 10 ) * pow( y, 11 ) * pow( z, 11 ) - 3168 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 12 ) -
                3960 * pow( x, 9 ) * pow( y, 12 ) * pow( z, 11 ) - 3960 * pow( x, 9 ) * pow( y, 11 ) * pow( z, 12 ) -
                1080 * pow( x, 8 ) * pow( y, 12 ) * pow( z, 12 ) + 660 * pow( x, 12 ) * pow( y, 12 ) * pow( z, 9 ) +
                792 * pow( x, 12 ) * pow( y, 11 ) * pow( z, 10 ) + 792 * pow( x, 12 ) * pow( y, 10 ) * pow( z, 11 ) +
                660 * pow( x, 12 ) * pow( y, 9 ) * pow( z, 12 ) + 792 * pow( x, 11 ) * pow( y, 12 ) * pow( z, 10 ) +
                792 * pow( x, 11 ) * pow( y, 10 ) * pow( z, 12 ) + 792 * pow( x, 10 ) * pow( y, 12 ) * pow( z, 11 ) +
                792 * pow( x, 10 ) * pow( y, 11 ) * pow( z, 12 ) + 660 * pow( x, 9 ) * pow( y, 12 ) * pow( z, 12 ) -
                132 * pow( x, 12 ) * pow( y, 12 ) * pow( z, 10 ) - 132 * pow( x, 12 ) * pow( y, 10 ) * pow( z, 12 ) -
                132 * pow( x, 10 ) * pow( y, 12 ) * pow( z, 12 ),
            0.0
        };
    }

    __DEVICE_TAG__ TensorType get_exact_solution( Scalar x, Scalar y, Scalar z ) const
    {
        Scalar g = x * ( 1 - x ) * y * ( 1 - y ) * z * ( 1 - z );

        return TensorType{
            g * ( -35 - 23 * x - 23 * y - 23 * z + 23 * pow( x, 2 ) - 11 * x * y - 11 * x * z + 23 * pow( y, 2 ) -
                  11 * y * z + 23 * pow( z, 2 ) + 11 * pow( x, 2 ) * y + 11 * pow( x, 2 ) * z + 11 * x * pow( y, 2 ) +
                  x * y * z + 11 * x * pow( z, 2 ) + 11 * pow( y, 2 ) * z + 11 * y * pow( z, 2 ) -
                  11 * pow( x, 2 ) * pow( y, 2 ) - pow( x, 2 ) * y * z - 11 * pow( x, 2 ) * pow( z, 2 ) -
                  x * pow( y, 2 ) * z - x * y * pow( z, 2 ) - 11 * pow( y, 2 ) * pow( z, 2 ) +
                  pow( x, 2 ) * pow( y, 2 ) * z + pow( x, 2 ) * y * pow( z, 2 ) + x * pow( y, 2 ) * pow( z, 2 ) -
                  2 * pow( x, 2 ) * pow( y, 2 ) * pow( z, 2 ) - pow( x, 3 ) * pow( y, 2 ) * pow( z, 2 ) -
                  pow( x, 2 ) * pow( y, 3 ) * pow( z, 2 ) - pow( x, 2 ) * pow( y, 2 ) * pow( z, 3 ) +
                  5 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 2 ) - pow( x, 3 ) * pow( y, 3 ) * pow( z, 2 ) -
                  pow( x, 3 ) * pow( y, 2 ) * pow( z, 3 ) + 5 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 2 ) -
                  pow( x, 2 ) * pow( y, 3 ) * pow( z, 3 ) + 5 * pow( x, 2 ) * pow( y, 2 ) * pow( z, 4 ) +
                  2 * pow( x, 5 ) * pow( y, 2 ) * pow( z, 2 ) + 5 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 2 ) +
                  5 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 3 ) + 5 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 2 ) -
                  pow( x, 3 ) * pow( y, 3 ) * pow( z, 3 ) + 5 * pow( x, 3 ) * pow( y, 2 ) * pow( z, 4 ) +
                  2 * pow( x, 2 ) * pow( y, 5 ) * pow( z, 2 ) + 5 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 3 ) +
                  5 * pow( x, 2 ) * pow( y, 3 ) * pow( z, 4 ) + 2 * pow( x, 2 ) * pow( y, 2 ) * pow( z, 5 ) -
                  10 * pow( x, 6 ) * pow( y, 2 ) * pow( z, 2 ) + 2 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 2 ) +
                  2 * pow( x, 5 ) * pow( y, 2 ) * pow( z, 3 ) - 25 * pow( x, 4 ) * pow( y, 4 ) * pow( z, 2 ) +
                  5 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 3 ) - 25 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 4 ) +
                  2 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 2 ) + 5 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 3 ) +
                  5 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 4 ) + 2 * pow( x, 3 ) * pow( y, 2 ) * pow( z, 5 ) -
                  10 * pow( x, 2 ) * pow( y, 6 ) * pow( z, 2 ) + 2 * pow( x, 2 ) * pow( y, 5 ) * pow( z, 3 ) -
                  25 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 4 ) + 2 * pow( x, 2 ) * pow( y, 3 ) * pow( z, 5 ) -
                  10 * pow( x, 2 ) * pow( y, 2 ) * pow( z, 6 ) + 2 * pow( x, 7 ) * pow( y, 2 ) * pow( z, 2 ) -
                  10 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 2 ) - 10 * pow( x, 6 ) * pow( y, 2 ) * pow( z, 3 ) -
                  10 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 2 ) + 2 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 3 ) -
                  10 * pow( x, 5 ) * pow( y, 2 ) * pow( z, 4 ) - 10 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 2 ) -
                  25 * pow( x, 4 ) * pow( y, 4 ) * pow( z, 3 ) - 25 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 4 ) -
                  10 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 5 ) - 10 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 2 ) +
                  2 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 3 ) - 25 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 4 ) +
                  2 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 5 ) - 10 * pow( x, 3 ) * pow( y, 2 ) * pow( z, 6 ) +
                  2 * pow( x, 2 ) * pow( y, 7 ) * pow( z, 2 ) - 10 * pow( x, 2 ) * pow( y, 6 ) * pow( z, 3 ) -
                  10 * pow( x, 2 ) * pow( y, 5 ) * pow( z, 4 ) - 10 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 5 ) -
                  10 * pow( x, 2 ) * pow( y, 3 ) * pow( z, 6 ) + 2 * pow( x, 2 ) * pow( y, 2 ) * pow( z, 7 ) +
                  7 * pow( x, 8 ) * pow( y, 2 ) * pow( z, 2 ) + 2 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 2 ) +
                  2 * pow( x, 7 ) * pow( y, 2 ) * pow( z, 3 ) + 50 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 2 ) -
                  10 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 3 ) + 50 * pow( x, 6 ) * pow( y, 2 ) * pow( z, 4 ) -
                  4 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 2 ) - 10 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 3 ) -
                  10 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 4 ) - 4 * pow( x, 5 ) * pow( y, 2 ) * pow( z, 5 ) +
                  50 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 2 ) - 10 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 3 ) +
                  125 * pow( x, 4 ) * pow( y, 4 ) * pow( z, 4 ) - 10 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 5 ) +
                  50 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 6 ) + 2 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 2 ) -
                  10 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 3 ) - 10 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 4 ) -
                  10 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 5 ) - 10 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 6 ) +
                  2 * pow( x, 3 ) * pow( y, 2 ) * pow( z, 7 ) + 7 * pow( x, 2 ) * pow( y, 8 ) * pow( z, 2 ) +
                  2 * pow( x, 2 ) * pow( y, 7 ) * pow( z, 3 ) + 50 * pow( x, 2 ) * pow( y, 6 ) * pow( z, 4 ) -
                  4 * pow( x, 2 ) * pow( y, 5 ) * pow( z, 5 ) + 50 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 6 ) +
                  2 * pow( x, 2 ) * pow( y, 3 ) * pow( z, 7 ) + 7 * pow( x, 2 ) * pow( y, 2 ) * pow( z, 8 ) -
                  5 * pow( x, 9 ) * pow( y, 2 ) * pow( z, 2 ) + 7 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 2 ) +
                  7 * pow( x, 8 ) * pow( y, 2 ) * pow( z, 3 ) - 10 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 2 ) +
                  2 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 3 ) - 10 * pow( x, 7 ) * pow( y, 2 ) * pow( z, 4 ) +
                  20 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 2 ) + 50 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 3 ) +
                  50 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 4 ) + 20 * pow( x, 6 ) * pow( y, 2 ) * pow( z, 5 ) +
                  20 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 2 ) - 4 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 3 ) +
                  50 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 4 ) - 4 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 5 ) +
                  20 * pow( x, 5 ) * pow( y, 2 ) * pow( z, 6 ) - 10 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 2 ) +
                  50 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 3 ) + 50 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 4 ) +
                  50 * pow( x, 4 ) * pow( y, 4 ) * pow( z, 5 ) + 50 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 6 ) -
                  10 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 7 ) + 7 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 2 ) +
                  2 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 3 ) + 50 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 4 ) -
                  4 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 5 ) + 50 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 6 ) +
                  2 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 7 ) + 7 * pow( x, 3 ) * pow( y, 2 ) * pow( z, 8 ) -
                  5 * pow( x, 2 ) * pow( y, 9 ) * pow( z, 2 ) + 7 * pow( x, 2 ) * pow( y, 8 ) * pow( z, 3 ) -
                  10 * pow( x, 2 ) * pow( y, 7 ) * pow( z, 4 ) + 20 * pow( x, 2 ) * pow( y, 6 ) * pow( z, 5 ) +
                  20 * pow( x, 2 ) * pow( y, 5 ) * pow( z, 6 ) - 10 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 7 ) +
                  7 * pow( x, 2 ) * pow( y, 3 ) * pow( z, 8 ) - 5 * pow( x, 2 ) * pow( y, 2 ) * pow( z, 9 ) +
                  pow( x, 10 ) * pow( y, 2 ) * pow( z, 2 ) - 5 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 2 ) -
                  5 * pow( x, 9 ) * pow( y, 2 ) * pow( z, 3 ) - 35 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 2 ) +
                  7 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 3 ) - 35 * pow( x, 8 ) * pow( y, 2 ) * pow( z, 4 ) -
                  4 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 2 ) - 10 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 3 ) -
                  10 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 4 ) - 4 * pow( x, 7 ) * pow( y, 2 ) * pow( z, 5 ) -
                  100 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 2 ) + 20 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 3 ) -
                  250 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 4 ) + 20 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 5 ) -
                  100 * pow( x, 6 ) * pow( y, 2 ) * pow( z, 6 ) - 4 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 2 ) +
                  20 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 3 ) + 20 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 4 ) +
                  20 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 5 ) + 20 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 6 ) -
                  4 * pow( x, 5 ) * pow( y, 2 ) * pow( z, 7 ) - 35 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 2 ) -
                  10 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 3 ) - 250 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 4 ) +
                  20 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 5 ) - 250 * pow( x, 4 ) * pow( y, 4 ) * pow( z, 6 ) -
                  10 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 7 ) - 35 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 8 ) -
                  5 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 2 ) + 7 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 3 ) -
                  10 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 4 ) + 20 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 5 ) +
                  20 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 6 ) - 10 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 7 ) +
                  7 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 8 ) - 5 * pow( x, 3 ) * pow( y, 2 ) * pow( z, 9 ) +
                  pow( x, 2 ) * pow( y, 10 ) * pow( z, 2 ) - 5 * pow( x, 2 ) * pow( y, 9 ) * pow( z, 3 ) -
                  35 * pow( x, 2 ) * pow( y, 8 ) * pow( z, 4 ) - 4 * pow( x, 2 ) * pow( y, 7 ) * pow( z, 5 ) -
                  100 * pow( x, 2 ) * pow( y, 6 ) * pow( z, 6 ) - 4 * pow( x, 2 ) * pow( y, 5 ) * pow( z, 7 ) -
                  35 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 8 ) - 5 * pow( x, 2 ) * pow( y, 3 ) * pow( z, 9 ) +
                  pow( x, 2 ) * pow( y, 2 ) * pow( z, 10 ) + pow( x, 10 ) * pow( y, 3 ) * pow( z, 2 ) +
                  pow( x, 10 ) * pow( y, 2 ) * pow( z, 3 ) + 25 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 2 ) -
                  5 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 3 ) + 25 * pow( x, 9 ) * pow( y, 2 ) * pow( z, 4 ) -
                  14 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 2 ) - 35 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 3 ) -
                  35 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 4 ) - 14 * pow( x, 8 ) * pow( y, 2 ) * pow( z, 5 ) +
                  20 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 2 ) - 4 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 3 ) +
                  50 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 4 ) - 4 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 5 ) +
                  20 * pow( x, 7 ) * pow( y, 2 ) * pow( z, 6 ) + 20 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 2 ) -
                  100 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 3 ) - 100 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 4 ) -
                  100 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 5 ) - 100 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 6 ) +
                  20 * pow( x, 6 ) * pow( y, 2 ) * pow( z, 7 ) - 14 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 2 ) -
                  4 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 3 ) - 100 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 4 ) +
                  8 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 5 ) - 100 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 6 ) -
                  4 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 7 ) - 14 * pow( x, 5 ) * pow( y, 2 ) * pow( z, 8 ) +
                  25 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 2 ) - 35 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 3 ) +
                  50 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 4 ) - 100 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 5 ) -
                  100 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 6 ) + 50 * pow( x, 4 ) * pow( y, 4 ) * pow( z, 7 ) -
                  35 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 8 ) + 25 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 9 ) +
                  pow( x, 3 ) * pow( y, 10 ) * pow( z, 2 ) - 5 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 3 ) -
                  35 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 4 ) - 4 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 5 ) -
                  100 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 6 ) - 4 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 7 ) -
                  35 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 8 ) - 5 * pow( x, 3 ) * pow( y, 3 ) * pow( z, 9 ) +
                  pow( x, 3 ) * pow( y, 2 ) * pow( z, 10 ) + pow( x, 2 ) * pow( y, 10 ) * pow( z, 3 ) +
                  25 * pow( x, 2 ) * pow( y, 9 ) * pow( z, 4 ) - 14 * pow( x, 2 ) * pow( y, 8 ) * pow( z, 5 ) +
                  20 * pow( x, 2 ) * pow( y, 7 ) * pow( z, 6 ) + 20 * pow( x, 2 ) * pow( y, 6 ) * pow( z, 7 ) -
                  14 * pow( x, 2 ) * pow( y, 5 ) * pow( z, 8 ) + 25 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 9 ) +
                  pow( x, 2 ) * pow( y, 3 ) * pow( z, 10 ) - 5 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 2 ) +
                  pow( x, 10 ) * pow( y, 3 ) * pow( z, 3 ) - 5 * pow( x, 10 ) * pow( y, 2 ) * pow( z, 4 ) +
                  10 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 2 ) + 25 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 3 ) +
                  25 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 4 ) + 10 * pow( x, 9 ) * pow( y, 2 ) * pow( z, 5 ) +
                  70 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 2 ) - 14 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 3 ) +
                  175 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 4 ) - 14 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 5 ) +
                  70 * pow( x, 8 ) * pow( y, 2 ) * pow( z, 6 ) - 4 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 2 ) +
                  20 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 3 ) + 20 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 4 ) +
                  20 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 5 ) + 20 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 6 ) -
                  4 * pow( x, 7 ) * pow( y, 2 ) * pow( z, 7 ) + 70 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 2 ) +
                  20 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 3 ) + 500 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 4 ) -
                  40 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 5 ) + 500 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 6 ) +
                  20 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 7 ) + 70 * pow( x, 6 ) * pow( y, 2 ) * pow( z, 8 ) +
                  10 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 2 ) - 14 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 3 ) +
                  20 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 4 ) - 40 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 5 ) -
                  40 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 6 ) + 20 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 7 ) -
                  14 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 8 ) + 10 * pow( x, 5 ) * pow( y, 2 ) * pow( z, 9 ) -
                  5 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 2 ) + 25 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 3 ) +
                  175 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 4 ) + 20 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 5 ) +
                  500 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 6 ) + 20 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 7 ) +
                  175 * pow( x, 4 ) * pow( y, 4 ) * pow( z, 8 ) + 25 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 9 ) -
                  5 * pow( x, 4 ) * pow( y, 2 ) * pow( z, 10 ) + pow( x, 3 ) * pow( y, 10 ) * pow( z, 3 ) +
                  25 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 4 ) - 14 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 5 ) +
                  20 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 6 ) + 20 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 7 ) -
                  14 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 8 ) + 25 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 9 ) +
                  pow( x, 3 ) * pow( y, 3 ) * pow( z, 10 ) - 5 * pow( x, 2 ) * pow( y, 10 ) * pow( z, 4 ) +
                  10 * pow( x, 2 ) * pow( y, 9 ) * pow( z, 5 ) + 70 * pow( x, 2 ) * pow( y, 8 ) * pow( z, 6 ) -
                  4 * pow( x, 2 ) * pow( y, 7 ) * pow( z, 7 ) + 70 * pow( x, 2 ) * pow( y, 6 ) * pow( z, 8 ) +
                  10 * pow( x, 2 ) * pow( y, 5 ) * pow( z, 9 ) - 5 * pow( x, 2 ) * pow( y, 4 ) * pow( z, 10 ) -
                  2 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 2 ) - 5 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 3 ) -
                  5 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 4 ) - 2 * pow( x, 10 ) * pow( y, 2 ) * pow( z, 5 ) -
                  50 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 2 ) + 10 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 3 ) -
                  125 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 4 ) + 10 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 5 ) -
                  50 * pow( x, 9 ) * pow( y, 2 ) * pow( z, 6 ) - 14 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 2 ) +
                  70 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 3 ) + 70 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 4 ) +
                  70 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 5 ) + 70 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 6 ) -
                  14 * pow( x, 8 ) * pow( y, 2 ) * pow( z, 7 ) - 14 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 2 ) -
                  4 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 3 ) - 100 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 4 ) +
                  8 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 5 ) - 100 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 6 ) -
                  4 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 7 ) - 14 * pow( x, 7 ) * pow( y, 2 ) * pow( z, 8 ) -
                  50 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 2 ) + 70 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 3 ) -
                  100 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 4 ) + 200 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 5 ) +
                  200 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 6 ) - 100 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 7 ) +
                  70 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 8 ) - 50 * pow( x, 6 ) * pow( y, 2 ) * pow( z, 9 ) -
                  2 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 2 ) + 10 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 3 ) +
                  70 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 4 ) + 8 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 5 ) +
                  200 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 6 ) + 8 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 7 ) +
                  70 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 8 ) + 10 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 9 ) -
                  2 * pow( x, 5 ) * pow( y, 2 ) * pow( z, 10 ) - 5 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 3 ) -
                  125 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 4 ) + 70 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 5 ) -
                  100 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 6 ) - 100 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 7 ) +
                  70 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 8 ) - 125 * pow( x, 4 ) * pow( y, 4 ) * pow( z, 9 ) -
                  5 * pow( x, 4 ) * pow( y, 3 ) * pow( z, 10 ) - 5 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 4 ) +
                  10 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 5 ) + 70 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 6 ) -
                  4 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 7 ) + 70 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 8 ) +
                  10 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 9 ) - 5 * pow( x, 3 ) * pow( y, 4 ) * pow( z, 10 ) -
                  2 * pow( x, 2 ) * pow( y, 10 ) * pow( z, 5 ) - 50 * pow( x, 2 ) * pow( y, 9 ) * pow( z, 6 ) -
                  14 * pow( x, 2 ) * pow( y, 8 ) * pow( z, 7 ) - 14 * pow( x, 2 ) * pow( y, 7 ) * pow( z, 8 ) -
                  50 * pow( x, 2 ) * pow( y, 6 ) * pow( z, 9 ) - 2 * pow( x, 2 ) * pow( y, 5 ) * pow( z, 10 ) +
                  10 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 2 ) - 2 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 3 ) +
                  25 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 4 ) - 2 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 5 ) +
                  10 * pow( x, 10 ) * pow( y, 2 ) * pow( z, 6 ) + 10 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 2 ) -
                  50 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 3 ) - 50 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 4 ) -
                  50 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 5 ) - 50 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 6 ) +
                  10 * pow( x, 9 ) * pow( y, 2 ) * pow( z, 7 ) - 49 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 2 ) -
                  14 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 3 ) - 350 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 4 ) +
                  28 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 5 ) - 350 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 6 ) -
                  14 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 7 ) - 49 * pow( x, 8 ) * pow( y, 2 ) * pow( z, 8 ) +
                  10 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 2 ) - 14 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 3 ) +
                  20 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 4 ) - 40 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 5 ) -
                  40 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 6 ) + 20 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 7 ) -
                  14 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 8 ) + 10 * pow( x, 7 ) * pow( y, 2 ) * pow( z, 9 ) +
                  10 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 2 ) - 50 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 3 ) -
                  350 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 4 ) - 40 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 5 ) -
                  1000 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 6 ) - 40 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 7 ) -
                  350 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 8 ) - 50 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 9 ) +
                  10 * pow( x, 6 ) * pow( y, 2 ) * pow( z, 10 ) - 2 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 3 ) -
                  50 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 4 ) + 28 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 5 ) -
                  40 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 6 ) - 40 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 7 ) +
                  28 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 8 ) - 50 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 9 ) -
                  2 * pow( x, 5 ) * pow( y, 3 ) * pow( z, 10 ) + 25 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 4 ) -
                  50 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 5 ) - 350 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 6 ) +
                  20 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 7 ) - 350 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 8 ) -
                  50 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 9 ) + 25 * pow( x, 4 ) * pow( y, 4 ) * pow( z, 10 ) -
                  2 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 5 ) - 50 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 6 ) -
                  14 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 7 ) - 14 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 8 ) -
                  50 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 9 ) - 2 * pow( x, 3 ) * pow( y, 5 ) * pow( z, 10 ) +
                  10 * pow( x, 2 ) * pow( y, 10 ) * pow( z, 6 ) + 10 * pow( x, 2 ) * pow( y, 9 ) * pow( z, 7 ) -
                  49 * pow( x, 2 ) * pow( y, 8 ) * pow( z, 8 ) + 10 * pow( x, 2 ) * pow( y, 7 ) * pow( z, 9 ) +
                  10 * pow( x, 2 ) * pow( y, 6 ) * pow( z, 10 ) - 2 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 2 ) +
                  10 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 3 ) + 10 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 4 ) +
                  10 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 5 ) + 10 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 6 ) -
                  2 * pow( x, 10 ) * pow( y, 2 ) * pow( z, 7 ) + 35 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 2 ) +
                  10 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 3 ) + 250 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 4 ) -
                  20 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 5 ) + 250 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 6 ) +
                  10 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 7 ) + 35 * pow( x, 9 ) * pow( y, 2 ) * pow( z, 8 ) +
                  35 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 2 ) - 49 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 3 ) +
                  70 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 4 ) - 140 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 5 ) -
                  140 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 6 ) + 70 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 7 ) -
                  49 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 8 ) + 35 * pow( x, 8 ) * pow( y, 2 ) * pow( z, 9 ) -
                  2 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 2 ) + 10 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 3 ) +
                  70 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 4 ) + 8 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 5 ) +
                  200 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 6 ) + 8 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 7 ) +
                  70 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 8 ) + 10 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 9 ) -
                  2 * pow( x, 7 ) * pow( y, 2 ) * pow( z, 10 ) + 10 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 3 ) +
                  250 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 4 ) - 140 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 5 ) +
                  200 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 6 ) + 200 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 7 ) -
                  140 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 8 ) + 250 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 9 ) +
                  10 * pow( x, 6 ) * pow( y, 3 ) * pow( z, 10 ) + 10 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 4 ) -
                  20 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 5 ) - 140 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 6 ) +
                  8 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 7 ) - 140 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 8 ) -
                  20 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 9 ) + 10 * pow( x, 5 ) * pow( y, 4 ) * pow( z, 10 ) +
                  10 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 5 ) + 250 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 6 ) +
                  70 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 7 ) + 70 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 8 ) +
                  250 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 9 ) + 10 * pow( x, 4 ) * pow( y, 5 ) * pow( z, 10 ) +
                  10 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 6 ) + 10 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 7 ) -
                  49 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 8 ) + 10 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 9 ) +
                  10 * pow( x, 3 ) * pow( y, 6 ) * pow( z, 10 ) - 2 * pow( x, 2 ) * pow( y, 10 ) * pow( z, 7 ) +
                  35 * pow( x, 2 ) * pow( y, 9 ) * pow( z, 8 ) + 35 * pow( x, 2 ) * pow( y, 8 ) * pow( z, 9 ) -
                  2 * pow( x, 2 ) * pow( y, 7 ) * pow( z, 10 ) - 7 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 2 ) -
                  2 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 3 ) - 50 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 4 ) +
                  4 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 5 ) - 50 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 6 ) -
                  2 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 7 ) - 7 * pow( x, 10 ) * pow( y, 2 ) * pow( z, 8 ) -
                  25 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 2 ) + 35 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 3 ) -
                  50 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 4 ) + 100 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 5 ) +
                  100 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 6 ) - 50 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 7 ) +
                  35 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 8 ) - 25 * pow( x, 9 ) * pow( y, 2 ) * pow( z, 9 ) -
                  7 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 2 ) + 35 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 3 ) +
                  245 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 4 ) + 28 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 5 ) +
                  700 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 6 ) + 28 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 7 ) +
                  245 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 8 ) + 35 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 9 ) -
                  7 * pow( x, 8 ) * pow( y, 2 ) * pow( z, 10 ) - 2 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 3 ) -
                  50 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 4 ) + 28 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 5 ) -
                  40 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 6 ) - 40 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 7 ) +
                  28 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 8 ) - 50 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 9 ) -
                  2 * pow( x, 7 ) * pow( y, 3 ) * pow( z, 10 ) - 50 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 4 ) +
                  100 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 5 ) + 700 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 6 ) -
                  40 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 7 ) + 700 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 8 ) +
                  100 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 9 ) - 50 * pow( x, 6 ) * pow( y, 4 ) * pow( z, 10 ) +
                  4 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 5 ) + 100 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 6 ) +
                  28 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 7 ) + 28 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 8 ) +
                  100 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 9 ) + 4 * pow( x, 5 ) * pow( y, 5 ) * pow( z, 10 ) -
                  50 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 6 ) - 50 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 7 ) +
                  245 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 8 ) - 50 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 9 ) -
                  50 * pow( x, 4 ) * pow( y, 6 ) * pow( z, 10 ) - 2 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 7 ) +
                  35 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 8 ) + 35 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 9 ) -
                  2 * pow( x, 3 ) * pow( y, 7 ) * pow( z, 10 ) - 7 * pow( x, 2 ) * pow( y, 10 ) * pow( z, 8 ) -
                  25 * pow( x, 2 ) * pow( y, 9 ) * pow( z, 9 ) - 7 * pow( x, 2 ) * pow( y, 8 ) * pow( z, 10 ) +
                  5 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 2 ) - 7 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 3 ) +
                  10 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 4 ) - 20 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 5 ) -
                  20 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 6 ) + 10 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 7 ) -
                  7 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 8 ) + 5 * pow( x, 10 ) * pow( y, 2 ) * pow( z, 9 ) +
                  5 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 2 ) - 25 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 3 ) -
                  175 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 4 ) - 20 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 5 ) -
                  500 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 6 ) - 20 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 7 ) -
                  175 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 8 ) - 25 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 9 ) +
                  5 * pow( x, 9 ) * pow( y, 2 ) * pow( z, 10 ) - 7 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 3 ) -
                  175 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 4 ) + 98 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 5 ) -
                  140 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 6 ) - 140 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 7 ) +
                  98 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 8 ) - 175 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 9 ) -
                  7 * pow( x, 8 ) * pow( y, 3 ) * pow( z, 10 ) + 10 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 4 ) -
                  20 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 5 ) - 140 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 6 ) +
                  8 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 7 ) - 140 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 8 ) -
                  20 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 9 ) + 10 * pow( x, 7 ) * pow( y, 4 ) * pow( z, 10 ) -
                  20 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 5 ) - 500 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 6 ) -
                  140 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 7 ) - 140 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 8 ) -
                  500 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 9 ) - 20 * pow( x, 6 ) * pow( y, 5 ) * pow( z, 10 ) -
                  20 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 6 ) - 20 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 7 ) +
                  98 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 8 ) - 20 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 9 ) -
                  20 * pow( x, 5 ) * pow( y, 6 ) * pow( z, 10 ) + 10 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 7 ) -
                  175 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 8 ) - 175 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 9 ) +
                  10 * pow( x, 4 ) * pow( y, 7 ) * pow( z, 10 ) - 7 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 8 ) -
                  25 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 9 ) - 7 * pow( x, 3 ) * pow( y, 8 ) * pow( z, 10 ) +
                  5 * pow( x, 2 ) * pow( y, 10 ) * pow( z, 9 ) + 5 * pow( x, 2 ) * pow( y, 9 ) * pow( z, 10 ) -
                  pow( x, 10 ) * pow( y, 10 ) * pow( z, 2 ) + 5 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 3 ) +
                  35 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 4 ) + 4 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 5 ) +
                  100 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 6 ) + 4 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 7 ) +
                  35 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 8 ) + 5 * pow( x, 10 ) * pow( y, 3 ) * pow( z, 9 ) -
                  pow( x, 10 ) * pow( y, 2 ) * pow( z, 10 ) + 5 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 3 ) +
                  125 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 4 ) - 70 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 5 ) +
                  100 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 6 ) + 100 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 7 ) -
                  70 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 8 ) + 125 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 9 ) +
                  5 * pow( x, 9 ) * pow( y, 3 ) * pow( z, 10 ) + 35 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 4 ) -
                  70 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 5 ) - 490 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 6 ) +
                  28 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 7 ) - 490 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 8 ) -
                  70 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 9 ) + 35 * pow( x, 8 ) * pow( y, 4 ) * pow( z, 10 ) +
                  4 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 5 ) + 100 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 6 ) +
                  28 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 7 ) + 28 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 8 ) +
                  100 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 9 ) + 4 * pow( x, 7 ) * pow( y, 5 ) * pow( z, 10 ) +
                  100 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 6 ) + 100 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 7 ) -
                  490 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 8 ) + 100 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 9 ) +
                  100 * pow( x, 6 ) * pow( y, 6 ) * pow( z, 10 ) + 4 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 7 ) -
                  70 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 8 ) - 70 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 9 ) +
                  4 * pow( x, 5 ) * pow( y, 7 ) * pow( z, 10 ) + 35 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 8 ) +
                  125 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 9 ) + 35 * pow( x, 4 ) * pow( y, 8 ) * pow( z, 10 ) +
                  5 * pow( x, 3 ) * pow( y, 10 ) * pow( z, 9 ) + 5 * pow( x, 3 ) * pow( y, 9 ) * pow( z, 10 ) -
                  pow( x, 2 ) * pow( y, 10 ) * pow( z, 10 ) - pow( x, 10 ) * pow( y, 10 ) * pow( z, 3 ) -
                  25 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 4 ) + 14 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 5 ) -
                  20 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 6 ) - 20 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 7 ) +
                  14 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 8 ) - 25 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 9 ) -
                  pow( x, 10 ) * pow( y, 3 ) * pow( z, 10 ) - 25 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 4 ) +
                  50 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 5 ) + 350 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 6 ) -
                  20 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 7 ) + 350 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 8 ) +
                  50 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 9 ) - 25 * pow( x, 9 ) * pow( y, 4 ) * pow( z, 10 ) +
                  14 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 5 ) + 350 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 6 ) +
                  98 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 7 ) + 98 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 8 ) +
                  350 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 9 ) + 14 * pow( x, 8 ) * pow( y, 5 ) * pow( z, 10 ) -
                  20 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 6 ) - 20 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 7 ) +
                  98 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 8 ) - 20 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 9 ) -
                  20 * pow( x, 7 ) * pow( y, 6 ) * pow( z, 10 ) - 20 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 7 ) +
                  350 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 8 ) + 350 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 9 ) -
                  20 * pow( x, 6 ) * pow( y, 7 ) * pow( z, 10 ) + 14 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 8 ) +
                  50 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 9 ) + 14 * pow( x, 5 ) * pow( y, 8 ) * pow( z, 10 ) -
                  25 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 9 ) - 25 * pow( x, 4 ) * pow( y, 9 ) * pow( z, 10 ) -
                  pow( x, 3 ) * pow( y, 10 ) * pow( z, 10 ) + 5 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 4 ) -
                  10 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 5 ) - 70 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 6 ) +
                  4 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 7 ) - 70 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 8 ) -
                  10 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 9 ) + 5 * pow( x, 10 ) * pow( y, 4 ) * pow( z, 10 ) -
                  10 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 5 ) - 250 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 6 ) -
                  70 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 7 ) - 70 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 8 ) -
                  250 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 9 ) - 10 * pow( x, 9 ) * pow( y, 5 ) * pow( z, 10 ) -
                  70 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 6 ) - 70 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 7 ) +
                  343 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 8 ) - 70 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 9 ) -
                  70 * pow( x, 8 ) * pow( y, 6 ) * pow( z, 10 ) + 4 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 7 ) -
                  70 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 8 ) - 70 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 9 ) +
                  4 * pow( x, 7 ) * pow( y, 7 ) * pow( z, 10 ) - 70 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 8 ) -
                  250 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 9 ) - 70 * pow( x, 6 ) * pow( y, 8 ) * pow( z, 10 ) -
                  10 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 9 ) - 10 * pow( x, 5 ) * pow( y, 9 ) * pow( z, 10 ) +
                  5 * pow( x, 4 ) * pow( y, 10 ) * pow( z, 10 ) + 2 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 5 ) +
                  50 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 6 ) + 14 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 7 ) +
                  14 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 8 ) + 50 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 9 ) +
                  2 * pow( x, 10 ) * pow( y, 5 ) * pow( z, 10 ) + 50 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 6 ) +
                  50 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 7 ) - 245 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 8 ) +
                  50 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 9 ) + 50 * pow( x, 9 ) * pow( y, 6 ) * pow( z, 10 ) +
                  14 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 7 ) - 245 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 8 ) -
                  245 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 9 ) + 14 * pow( x, 8 ) * pow( y, 7 ) * pow( z, 10 ) +
                  14 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 8 ) + 50 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 9 ) +
                  14 * pow( x, 7 ) * pow( y, 8 ) * pow( z, 10 ) + 50 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 9 ) +
                  50 * pow( x, 6 ) * pow( y, 9 ) * pow( z, 10 ) + 2 * pow( x, 5 ) * pow( y, 10 ) * pow( z, 10 ) -
                  10 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 6 ) - 10 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 7 ) +
                  49 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 8 ) - 10 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 9 ) -
                  10 * pow( x, 10 ) * pow( y, 6 ) * pow( z, 10 ) - 10 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 7 ) +
                  175 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 8 ) + 175 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 9 ) -
                  10 * pow( x, 9 ) * pow( y, 7 ) * pow( z, 10 ) + 49 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 8 ) +
                  175 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 9 ) + 49 * pow( x, 8 ) * pow( y, 8 ) * pow( z, 10 ) -
                  10 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 9 ) - 10 * pow( x, 7 ) * pow( y, 9 ) * pow( z, 10 ) -
                  10 * pow( x, 6 ) * pow( y, 10 ) * pow( z, 10 ) + 2 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 7 ) -
                  35 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 8 ) - 35 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 9 ) +
                  2 * pow( x, 10 ) * pow( y, 7 ) * pow( z, 10 ) - 35 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 8 ) -
                  125 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 9 ) - 35 * pow( x, 9 ) * pow( y, 8 ) * pow( z, 10 ) -
                  35 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 9 ) - 35 * pow( x, 8 ) * pow( y, 9 ) * pow( z, 10 ) +
                  2 * pow( x, 7 ) * pow( y, 10 ) * pow( z, 10 ) + 7 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 8 ) +
                  25 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 9 ) + 7 * pow( x, 10 ) * pow( y, 8 ) * pow( z, 10 ) +
                  25 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 9 ) + 25 * pow( x, 9 ) * pow( y, 9 ) * pow( z, 10 ) +
                  7 * pow( x, 8 ) * pow( y, 10 ) * pow( z, 10 ) - 5 * pow( x, 10 ) * pow( y, 10 ) * pow( z, 9 ) -
                  5 * pow( x, 10 ) * pow( y, 9 ) * pow( z, 10 ) - 5 * pow( x, 9 ) * pow( y, 10 ) * pow( z, 10 ) +
                  pow( x, 10 ) * pow( y, 10 ) * pow( z, 10 ) ),

            g * ( -1 - x - y - z + pow( x, 2 ) - x * y - x * z + pow( y, 2 ) - y * z + pow( z, 2 ) + pow( x, 2 ) * y +
                  pow( x, 2 ) * z + x * pow( y, 2 ) - x * y * z + x * pow( z, 2 ) + pow( y, 2 ) * z + y * pow( z, 2 ) -
                  pow( x, 2 ) * pow( y, 2 ) + pow( x, 2 ) * y * z - pow( x, 2 ) * pow( z, 2 ) + x * pow( y, 2 ) * z +
                  x * y * pow( z, 2 ) - pow( y, 2 ) * pow( z, 2 ) - pow( x, 2 ) * pow( y, 2 ) * z -
                  pow( x, 2 ) * y * pow( z, 2 ) - x * pow( y, 2 ) * pow( z, 2 ) +
                  pow( x, 2 ) * pow( y, 2 ) * pow( z, 2 ) )
        };
    }
};

} // namespace tests

#endif
