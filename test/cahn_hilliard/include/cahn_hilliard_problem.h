#ifndef __CAHN_HILLIARD_PROBLEM_H__
#define __CAHN_HILLIARD_PROBLEM_H__

#include <cmath>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>
// #ifndef M_st::pi()
// #define M_st::pi() 3.14159265358979323846
// #endif
// #define st::pi() M_st::pi()

namespace tests
{

template <class Scalar, class TensorType, int m, int n, int k>
class trig_rhs
{
    using st = scfd::utils::scalar_traits<Scalar>;

public:
    __DEVICE_TAG__ TensorType operator()( Scalar x, Scalar y, Scalar z ) const
    {

        return TensorType{
            -st::pow( st::pi(), 4 ) * st::pow( k, 4 ) * st::sin( st::pi() * k * z ) * st::sin( st::pi() * m * x ) *
                    st::sin( st::pi() * n * y ) -
                2 * st::pow( st::pi(), 4 ) * st::pow( k, 2 ) * st::pow( m, 2 ) * st::sin( st::pi() * k * z ) *
                    st::sin( st::pi() * m * x ) * st::sin( st::pi() * n * y ) -
                2 * st::pow( st::pi(), 4 ) * st::pow( k, 2 ) * st::pow( n, 2 ) * st::sin( st::pi() * k * z ) *
                    st::sin( st::pi() * m * x ) * st::sin( st::pi() * n * y ) -
                3 * st::pow( st::pi(), 2 ) * st::pow( k, 2 ) * st::pow( st::sin( st::pi() * k * z ), 3 ) *
                    st::pow( st::sin( st::pi() * m * x ), 3 ) * st::pow( st::sin( st::pi() * n * y ), 3 ) +
                6 * st::pow( st::pi(), 2 ) * st::pow( k, 2 ) * st::sin( st::pi() * k * z ) *
                    st::pow( st::sin( st::pi() * m * x ), 3 ) * st::pow( st::sin( st::pi() * n * y ), 3 ) *
                    st::pow( st::cos( st::pi() * k * z ), 2 ) +
                st::pow( st::pi(), 2 ) * st::pow( k, 2 ) * st::sin( st::pi() * k * z ) * st::sin( st::pi() * m * x ) *
                    st::sin( st::pi() * n * y ) -
                st::pow( st::pi(), 4 ) * st::pow( m, 4 ) * st::sin( st::pi() * k * z ) * st::sin( st::pi() * m * x ) *
                    st::sin( st::pi() * n * y ) -
                2 * st::pow( st::pi(), 4 ) * st::pow( m, 2 ) * st::pow( n, 2 ) * st::sin( st::pi() * k * z ) *
                    st::sin( st::pi() * m * x ) * st::sin( st::pi() * n * y ) -
                3 * st::pow( st::pi(), 2 ) * st::pow( m, 2 ) * st::pow( st::sin( st::pi() * k * z ), 3 ) *
                    st::pow( st::sin( st::pi() * m * x ), 3 ) * st::pow( st::sin( st::pi() * n * y ), 3 ) +
                6 * st::pow( st::pi(), 2 ) * st::pow( m, 2 ) * st::pow( st::sin( st::pi() * k * z ), 3 ) *
                    st::sin( st::pi() * m * x ) * st::pow( st::sin( st::pi() * n * y ), 3 ) *
                    st::pow( st::cos( st::pi() * m * x ), 2 ) +
                st::pow( st::pi(), 2 ) * st::pow( m, 2 ) * st::sin( st::pi() * k * z ) * st::sin( st::pi() * m * x ) *
                    st::sin( st::pi() * n * y ) -
                st::pow( st::pi(), 4 ) * st::pow( n, 4 ) * st::sin( st::pi() * k * z ) * st::sin( st::pi() * m * x ) *
                    st::sin( st::pi() * n * y ) -
                3 * st::pow( st::pi(), 2 ) * st::pow( n, 2 ) * st::pow( st::sin( st::pi() * k * z ), 3 ) *
                    st::pow( st::sin( st::pi() * m * x ), 3 ) * st::pow( st::sin( st::pi() * n * y ), 3 ) +
                6 * st::pow( st::pi(), 2 ) * st::pow( n, 2 ) * st::pow( st::sin( st::pi() * k * z ), 3 ) *
                    st::pow( st::sin( st::pi() * m * x ), 3 ) * st::sin( st::pi() * n * y ) *
                    st::pow( st::cos( st::pi() * n * y ), 2 ) +
                st::pow( st::pi(), 2 ) * st::pow( n, 2 ) * st::sin( st::pi() * k * z ) * st::sin( st::pi() * m * x ) *
                    st::sin( st::pi() * n * y ),
            0.0
        };
    }

    __DEVICE_TAG__ TensorType get_exact_solution( Scalar x, Scalar y, Scalar z ) const
    {

        return TensorType{
            st::pow( st::pi(), 2 ) * st::pow( k, 2 ) * st::sin( st::pi() * k * z ) * st::sin( st::pi() * m * x ) *
                    st::sin( st::pi() * n * y ) +
                st::pow( st::pi(), 2 ) * st::pow( m, 2 ) * st::sin( st::pi() * k * z ) * st::sin( st::pi() * m * x ) *
                    st::sin( st::pi() * n * y ) +
                st::pow( st::pi(), 2 ) * st::pow( n, 2 ) * st::sin( st::pi() * k * z ) * st::sin( st::pi() * m * x ) *
                    st::sin( st::pi() * n * y ) +
                st::pow( st::sin( st::pi() * k * z ), 3 ) * st::pow( st::sin( st::pi() * m * x ), 3 ) *
                    st::pow( st::sin( st::pi() * n * y ), 3 ) -
                st::sin( st::pi() * k * z ) * st::sin( st::pi() * m * x ) * st::sin( st::pi() * n * y ),
            st::sin( st::pi() * k * z ) * st::sin( st::pi() * m * x ) * st::sin( st::pi() * n * y )
        };
    }
};

template <class Scalar, class TensorType>
class poli_rhs
{
    using st = scfd::utils::scalar_traits<Scalar>;

public:
    __DEVICE_TAG__ TensorType operator()( Scalar x, Scalar y, Scalar z ) const
    {
        return TensorType{
            24 * x * y + 24 * x * z + 24 * y * z + 828 * x * y * z - 48 * st::pow( x, 3 ) * y - 48 * st::pow( x, 3 ) * z -
                564 * st::pow( x, 2 ) * y * z - 48 * x * st::pow( y, 3 ) - 564 * x * st::pow( y, 2 ) * z -
                564 * x * y * st::pow( z, 2 ) - 48 * x * st::pow( z, 3 ) - 48 * st::pow( y, 3 ) * z - 48 * y * st::pow( z, 3 ) +
                24 * st::pow( x, 4 ) * y + 24 * st::pow( x, 4 ) * z - 528 * st::pow( x, 3 ) * y * z +
                288 * st::pow( x, 2 ) * st::pow( y, 2 ) * z + 288 * st::pow( x, 2 ) * y * st::pow( z, 2 ) + 24 * x * st::pow( y, 4 ) -
                528 * x * st::pow( y, 3 ) * z + 288 * x * st::pow( y, 2 ) * st::pow( z, 2 ) - 528 * x * y * st::pow( z, 3 ) +
                24 * x * st::pow( z, 4 ) + 24 * st::pow( y, 4 ) * z + 24 * y * st::pow( z, 4 ) + 264 * st::pow( x, 4 ) * y * z +
                96 * st::pow( x, 3 ) * st::pow( y, 3 ) + 552 * st::pow( x, 3 ) * st::pow( y, 2 ) * z +
                552 * st::pow( x, 3 ) * y * st::pow( z, 2 ) + 96 * st::pow( x, 3 ) * st::pow( z, 3 ) +
                552 * st::pow( x, 2 ) * st::pow( y, 3 ) * z + 552 * st::pow( x, 2 ) * y * st::pow( z, 3 ) + 264 * x * st::pow( y, 4 ) * z +
                552 * x * st::pow( y, 3 ) * st::pow( z, 2 ) + 552 * x * st::pow( y, 2 ) * st::pow( z, 3 ) + 264 * x * y * st::pow( z, 4 ) +
                96 * st::pow( y, 3 ) * st::pow( z, 3 ) - 48 * st::pow( x, 4 ) * st::pow( y, 3 ) - 276 * st::pow( x, 4 ) * st::pow( y, 2 ) * z -
                276 * st::pow( x, 4 ) * y * st::pow( z, 2 ) - 48 * st::pow( x, 4 ) * st::pow( z, 3 ) - 48 * st::pow( x, 3 ) * st::pow( y, 4 ) -
                54 * st::pow( x, 3 ) * st::pow( y, 3 ) * z - 576 * st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 2 ) -
                54 * st::pow( x, 3 ) * y * st::pow( z, 3 ) - 48 * st::pow( x, 3 ) * st::pow( z, 4 ) -
                276 * st::pow( x, 2 ) * st::pow( y, 4 ) * z - 576 * st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 2 ) -
                576 * st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 3 ) - 276 * st::pow( x, 2 ) * y * st::pow( z, 4 ) -
                276 * x * st::pow( y, 4 ) * st::pow( z, 2 ) - 54 * x * st::pow( y, 3 ) * st::pow( z, 3 ) -
                276 * x * st::pow( y, 2 ) * st::pow( z, 4 ) - 48 * st::pow( y, 4 ) * st::pow( z, 3 ) - 48 * st::pow( y, 3 ) * st::pow( z, 4 ) +
                24 * st::pow( x, 4 ) * st::pow( y, 4 ) + 24 * st::pow( x, 4 ) * st::pow( y, 3 ) * z +
                288 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 2 ) + 24 * st::pow( x, 4 ) * y * st::pow( z, 3 ) +
                24 * st::pow( x, 4 ) * st::pow( z, 4 ) + 24 * st::pow( x, 3 ) * st::pow( y, 4 ) * z +
                48 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 2 ) + 48 * st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 3 ) +
                24 * st::pow( x, 3 ) * y * st::pow( z, 4 ) + 288 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 2 ) +
                48 * st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 3 ) + 288 * st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 4 ) +
                24 * x * st::pow( y, 4 ) * st::pow( z, 3 ) + 24 * x * st::pow( y, 3 ) * st::pow( z, 4 ) +
                24 * st::pow( y, 4 ) * st::pow( z, 4 ) + 36 * st::pow( x, 5 ) * st::pow( y, 3 ) * z +
                36 * st::pow( x, 5 ) * y * st::pow( z, 3 ) - 12 * st::pow( x, 4 ) * st::pow( y, 4 ) * z -
                24 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 2 ) - 24 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 3 ) -
                12 * st::pow( x, 4 ) * y * st::pow( z, 4 ) + 36 * st::pow( x, 3 ) * st::pow( y, 5 ) * z -
                24 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 2 ) + 360 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 3 ) -
                24 * st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 4 ) + 36 * st::pow( x, 3 ) * y * st::pow( z, 5 ) -
                24 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 3 ) - 24 * st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 4 ) +
                36 * x * st::pow( y, 5 ) * st::pow( z, 3 ) - 12 * x * st::pow( y, 4 ) * st::pow( z, 4 ) +
                36 * x * st::pow( y, 3 ) * st::pow( z, 5 ) - 18 * st::pow( x, 6 ) * st::pow( y, 3 ) * z -
                18 * st::pow( x, 6 ) * y * st::pow( z, 3 ) + 12 * st::pow( x, 4 ) * st::pow( y, 4 ) * st::pow( z, 2 ) -
                90 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 3 ) + 12 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 4 ) -
                18 * st::pow( x, 3 ) * st::pow( y, 6 ) * z - 90 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 3 ) -
                90 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 4 ) - 18 * st::pow( x, 3 ) * y * st::pow( z, 6 ) +
                12 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 4 ) - 18 * x * st::pow( y, 6 ) * st::pow( z, 3 ) -
                18 * x * st::pow( y, 3 ) * st::pow( z, 6 ) - 72 * st::pow( x, 7 ) * st::pow( y, 3 ) * z -
                72 * st::pow( x, 7 ) * y * st::pow( z, 3 ) - 216 * st::pow( x, 5 ) * st::pow( y, 5 ) * z -
                1944 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 3 ) - 216 * st::pow( x, 5 ) * y * st::pow( z, 5 ) -
                72 * st::pow( x, 3 ) * st::pow( y, 7 ) * z - 1944 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 3 ) -
                1944 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 5 ) - 72 * st::pow( x, 3 ) * y * st::pow( z, 7 ) -
                72 * x * st::pow( y, 7 ) * st::pow( z, 3 ) - 216 * x * st::pow( y, 5 ) * st::pow( z, 5 ) -
                72 * x * st::pow( y, 3 ) * st::pow( z, 7 ) + 72 * st::pow( x, 8 ) * st::pow( y, 3 ) * z +
                72 * st::pow( x, 8 ) * y * st::pow( z, 3 ) + 108 * st::pow( x, 6 ) * st::pow( y, 5 ) * z +
                1392 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 3 ) + 108 * st::pow( x, 6 ) * y * st::pow( z, 5 ) +
                108 * st::pow( x, 5 ) * st::pow( y, 6 ) * z + 540 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 3 ) +
                540 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 4 ) + 108 * st::pow( x, 5 ) * y * st::pow( z, 6 ) +
                540 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 3 ) + 540 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 5 ) +
                72 * st::pow( x, 3 ) * st::pow( y, 8 ) * z + 1392 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 3 ) +
                540 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 4 ) + 540 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 5 ) +
                1392 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 6 ) + 72 * st::pow( x, 3 ) * y * st::pow( z, 8 ) +
                72 * x * st::pow( y, 8 ) * st::pow( z, 3 ) + 108 * x * st::pow( y, 6 ) * st::pow( z, 5 ) +
                108 * x * st::pow( y, 5 ) * st::pow( z, 6 ) + 72 * x * st::pow( y, 3 ) * st::pow( z, 8 ) +
                30 * st::pow( x, 9 ) * st::pow( y, 3 ) * z + 30 * st::pow( x, 9 ) * y * st::pow( z, 3 ) +
                432 * st::pow( x, 7 ) * st::pow( y, 5 ) * z + 3240 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 3 ) +
                432 * st::pow( x, 7 ) * y * st::pow( z, 5 ) - 54 * st::pow( x, 6 ) * st::pow( y, 6 ) * z -
                270 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 3 ) - 270 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 4 ) -
                54 * st::pow( x, 6 ) * y * st::pow( z, 6 ) + 432 * st::pow( x, 5 ) * st::pow( y, 7 ) * z +
                10368 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 3 ) + 10368 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 5 ) +
                432 * st::pow( x, 5 ) * y * st::pow( z, 7 ) - 270 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 3 ) -
                270 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 6 ) + 30 * st::pow( x, 3 ) * st::pow( y, 9 ) * z +
                3240 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 3 ) - 270 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 4 ) +
                10368 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 5 ) - 270 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 6 ) +
                3240 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 7 ) + 30 * st::pow( x, 3 ) * y * st::pow( z, 9 ) +
                30 * x * st::pow( y, 9 ) * st::pow( z, 3 ) + 432 * x * st::pow( y, 7 ) * st::pow( z, 5 ) -
                54 * x * st::pow( y, 6 ) * st::pow( z, 6 ) + 432 * x * st::pow( y, 5 ) * st::pow( z, 7 ) +
                30 * x * st::pow( y, 3 ) * st::pow( z, 9 ) - 72 * st::pow( x, 10 ) * st::pow( y, 3 ) * z -
                72 * st::pow( x, 10 ) * y * st::pow( z, 3 ) - 432 * st::pow( x, 8 ) * st::pow( y, 5 ) * z -
                3960 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 3 ) - 432 * st::pow( x, 8 ) * y * st::pow( z, 5 ) -
                216 * st::pow( x, 7 ) * st::pow( y, 6 ) * z - 1080 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 3 ) -
                1080 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 4 ) - 216 * st::pow( x, 7 ) * y * st::pow( z, 6 ) -
                216 * st::pow( x, 6 ) * st::pow( y, 7 ) * z - 7704 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 3 ) -
                7704 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 5 ) - 216 * st::pow( x, 6 ) * y * st::pow( z, 7 ) -
                432 * st::pow( x, 5 ) * st::pow( y, 8 ) * z - 7704 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 3 ) -
                3240 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 4 ) - 3240 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 5 ) -
                7704 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 6 ) - 432 * st::pow( x, 5 ) * y * st::pow( z, 8 ) -
                1080 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 3 ) - 3240 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 5 ) -
                1080 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 7 ) - 72 * st::pow( x, 3 ) * st::pow( y, 10 ) * z -
                3960 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 3 ) - 1080 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 4 ) -
                7704 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 5 ) - 7704 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 6 ) -
                1080 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 7 ) - 3960 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 8 ) -
                72 * st::pow( x, 3 ) * y * st::pow( z, 10 ) - 72 * x * st::pow( y, 10 ) * st::pow( z, 3 ) -
                432 * x * st::pow( y, 8 ) * st::pow( z, 5 ) - 216 * x * st::pow( y, 7 ) * st::pow( z, 6 ) -
                216 * x * st::pow( y, 6 ) * st::pow( z, 7 ) - 432 * x * st::pow( y, 5 ) * st::pow( z, 8 ) -
                72 * x * st::pow( y, 3 ) * st::pow( z, 10 ) + 36 * st::pow( x, 11 ) * st::pow( y, 3 ) * z +
                36 * st::pow( x, 11 ) * y * st::pow( z, 3 ) - 180 * st::pow( x, 9 ) * st::pow( y, 5 ) * z -
                540 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 3 ) - 180 * st::pow( x, 9 ) * y * st::pow( z, 5 ) +
                216 * st::pow( x, 8 ) * st::pow( y, 6 ) * z + 1080 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 3 ) +
                1080 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 4 ) + 216 * st::pow( x, 8 ) * y * st::pow( z, 6 ) -
                864 * st::pow( x, 7 ) * st::pow( y, 7 ) * z - 16848 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 3 ) -
                16848 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 5 ) - 864 * st::pow( x, 7 ) * y * st::pow( z, 7 ) +
                216 * st::pow( x, 6 ) * st::pow( y, 8 ) * z + 5112 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 3 ) +
                1620 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 4 ) + 1620 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 5 ) +
                5112 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 6 ) + 216 * st::pow( x, 6 ) * y * st::pow( z, 8 ) -
                180 * st::pow( x, 5 ) * st::pow( y, 9 ) * z - 16848 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 3 ) +
                1620 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 4 ) - 54432 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 5 ) +
                1620 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 6 ) - 16848 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 7 ) -
                180 * st::pow( x, 5 ) * y * st::pow( z, 9 ) + 1080 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 3 ) +
                1620 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 5 ) + 1620 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 6 ) +
                1080 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 8 ) + 36 * st::pow( x, 3 ) * st::pow( y, 11 ) * z -
                540 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 3 ) + 1080 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 4 ) -
                16848 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 5 ) + 5112 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 6 ) -
                16848 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 7 ) + 1080 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 8 ) -
                540 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 9 ) + 36 * st::pow( x, 3 ) * y * st::pow( z, 11 ) +
                36 * x * st::pow( y, 11 ) * st::pow( z, 3 ) - 180 * x * st::pow( y, 9 ) * st::pow( z, 5 ) +
                216 * x * st::pow( y, 8 ) * st::pow( z, 6 ) - 864 * x * st::pow( y, 7 ) * st::pow( z, 7 ) +
                216 * x * st::pow( y, 6 ) * st::pow( z, 8 ) - 180 * x * st::pow( y, 5 ) * st::pow( z, 9 ) +
                36 * x * st::pow( y, 3 ) * st::pow( z, 11 ) - 6 * st::pow( x, 12 ) * st::pow( y, 3 ) * z -
                6 * st::pow( x, 12 ) * y * st::pow( z, 3 ) + 432 * st::pow( x, 10 ) * st::pow( y, 5 ) * z +
                2748 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 3 ) + 432 * st::pow( x, 10 ) * y * st::pow( z, 5 ) +
                90 * st::pow( x, 9 ) * st::pow( y, 6 ) * z + 450 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 3 ) +
                450 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 4 ) + 90 * st::pow( x, 9 ) * y * st::pow( z, 6 ) +
                864 * st::pow( x, 8 ) * st::pow( y, 7 ) * z + 21168 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 3 ) +
                21168 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 5 ) + 864 * st::pow( x, 8 ) * y * st::pow( z, 7 ) +
                864 * st::pow( x, 7 ) * st::pow( y, 8 ) * z + 13464 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 3 ) +
                6480 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 4 ) + 6480 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 5 ) +
                13464 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 6 ) + 864 * st::pow( x, 7 ) * y * st::pow( z, 8 ) +
                90 * st::pow( x, 6 ) * st::pow( y, 9 ) * z + 13464 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 3 ) -
                810 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 4 ) + 42336 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 5 ) -
                810 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 6 ) + 13464 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 7 ) +
                90 * st::pow( x, 6 ) * y * st::pow( z, 9 ) + 432 * st::pow( x, 5 ) * st::pow( y, 10 ) * z +
                21168 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 3 ) + 6480 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 4 ) +
                42336 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 5 ) + 42336 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 6 ) +
                6480 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 7 ) + 21168 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 8 ) +
                432 * st::pow( x, 5 ) * y * st::pow( z, 10 ) + 450 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 3 ) +
                6480 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 5 ) - 810 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 6 ) +
                6480 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 7 ) + 450 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 9 ) -
                6 * st::pow( x, 3 ) * st::pow( y, 12 ) * z + 2748 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 3 ) +
                450 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 4 ) + 21168 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 5 ) +
                13464 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 6 ) + 13464 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 7 ) +
                21168 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 8 ) + 450 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 9 ) +
                2748 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 10 ) - 6 * st::pow( x, 3 ) * y * st::pow( z, 12 ) -
                6 * x * st::pow( y, 12 ) * st::pow( z, 3 ) + 432 * x * st::pow( y, 10 ) * st::pow( z, 5 ) +
                90 * x * st::pow( y, 9 ) * st::pow( z, 6 ) + 864 * x * st::pow( y, 8 ) * st::pow( z, 7 ) +
                864 * x * st::pow( y, 7 ) * st::pow( z, 8 ) + 90 * x * st::pow( y, 6 ) * st::pow( z, 9 ) +
                432 * x * st::pow( y, 5 ) * st::pow( z, 10 ) - 6 * x * st::pow( y, 3 ) * st::pow( z, 12 ) -
                216 * st::pow( x, 11 ) * st::pow( y, 5 ) * z - 1440 * st::pow( x, 11 ) * st::pow( y, 3 ) * st::pow( z, 3 ) -
                216 * st::pow( x, 11 ) * y * st::pow( z, 5 ) - 216 * st::pow( x, 10 ) * st::pow( y, 6 ) * z -
                1080 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 3 ) - 1080 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 4 ) -
                216 * st::pow( x, 10 ) * y * st::pow( z, 6 ) + 360 * st::pow( x, 9 ) * st::pow( y, 7 ) * z +
                2160 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 3 ) + 2160 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 5 ) +
                360 * st::pow( x, 9 ) * y * st::pow( z, 7 ) - 864 * st::pow( x, 8 ) * st::pow( y, 8 ) * z -
                15624 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 3 ) - 6480 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 4 ) -
                6480 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 5 ) - 15624 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 6 ) -
                864 * st::pow( x, 8 ) * y * st::pow( z, 8 ) + 360 * st::pow( x, 7 ) * st::pow( y, 9 ) * z +
                25920 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 3 ) - 3240 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 4 ) +
                85536 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 5 ) - 3240 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 6 ) +
                25920 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 7 ) + 360 * st::pow( x, 7 ) * y * st::pow( z, 9 ) -
                216 * st::pow( x, 6 ) * st::pow( y, 10 ) * z - 15624 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 3 ) -
                3240 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 4 ) - 28728 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 5 ) -
                28728 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 6 ) - 3240 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 7 ) -
                15624 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 8 ) - 216 * st::pow( x, 6 ) * y * st::pow( z, 10 ) -
                216 * st::pow( x, 5 ) * st::pow( y, 11 ) * z + 2160 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 3 ) -
                6480 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 4 ) + 85536 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 5 ) -
                28728 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 6 ) + 85536 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 7 ) -
                6480 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 8 ) + 2160 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 9 ) -
                216 * st::pow( x, 5 ) * y * st::pow( z, 11 ) - 1080 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 3 ) -
                6480 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 5 ) - 3240 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 6 ) -
                3240 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 7 ) - 6480 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 8 ) -
                1080 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 10 ) - 1440 * st::pow( x, 3 ) * st::pow( y, 11 ) * st::pow( z, 3 ) -
                1080 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 4 ) + 2160 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 5 ) -
                15624 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 6 ) + 25920 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 7 ) -
                15624 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 8 ) + 2160 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 9 ) -
                1080 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 10 ) - 1440 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 11 ) -
                216 * x * st::pow( y, 11 ) * st::pow( z, 5 ) - 216 * x * st::pow( y, 10 ) * st::pow( z, 6 ) +
                360 * x * st::pow( y, 9 ) * st::pow( z, 7 ) - 864 * x * st::pow( y, 8 ) * st::pow( z, 8 ) +
                360 * x * st::pow( y, 7 ) * st::pow( z, 9 ) - 216 * x * st::pow( y, 6 ) * st::pow( z, 10 ) -
                216 * x * st::pow( y, 5 ) * st::pow( z, 11 ) + 36 * st::pow( x, 12 ) * st::pow( y, 5 ) * z +
                240 * st::pow( x, 12 ) * st::pow( y, 3 ) * st::pow( z, 3 ) + 36 * st::pow( x, 12 ) * y * st::pow( z, 5 ) +
                108 * st::pow( x, 11 ) * st::pow( y, 6 ) * z + 540 * st::pow( x, 11 ) * st::pow( y, 4 ) * st::pow( z, 3 ) +
                540 * st::pow( x, 11 ) * st::pow( y, 3 ) * st::pow( z, 4 ) + 108 * st::pow( x, 11 ) * y * st::pow( z, 6 ) -
                864 * st::pow( x, 10 ) * st::pow( y, 7 ) * z - 13896 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 3 ) -
                13896 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 5 ) - 864 * st::pow( x, 10 ) * y * st::pow( z, 7 ) -
                360 * st::pow( x, 9 ) * st::pow( y, 8 ) * z - 3180 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 3 ) -
                2700 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 4 ) - 2700 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 5 ) -
                3180 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 6 ) - 360 * st::pow( x, 9 ) * y * st::pow( z, 8 ) -
                360 * st::pow( x, 8 ) * st::pow( y, 9 ) * z - 34560 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 3 ) +
                3240 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 4 ) - 111456 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 5 ) +
                3240 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 6 ) - 34560 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 7 ) -
                360 * st::pow( x, 8 ) * y * st::pow( z, 9 ) - 864 * st::pow( x, 7 ) * st::pow( y, 10 ) * z -
                34560 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 3 ) - 12960 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 4 ) -
                73008 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 5 ) - 73008 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 6 ) -
                12960 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 7 ) - 34560 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 8 ) -
                864 * st::pow( x, 7 ) * y * st::pow( z, 10 ) + 108 * st::pow( x, 6 ) * st::pow( y, 11 ) * z -
                3180 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 3 ) + 3240 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 4 ) -
                73008 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 5 ) + 18144 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 6 ) -
                73008 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 7 ) + 3240 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 8 ) -
                3180 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 9 ) + 108 * st::pow( x, 6 ) * y * st::pow( z, 11 ) +
                36 * st::pow( x, 5 ) * st::pow( y, 12 ) * z - 13896 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 3 ) -
                2700 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 4 ) - 111456 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 5 ) -
                73008 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 6 ) - 73008 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 7 ) -
                111456 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 8 ) - 2700 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 9 ) -
                13896 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 10 ) + 36 * st::pow( x, 5 ) * y * st::pow( z, 12 ) +
                540 * st::pow( x, 4 ) * st::pow( y, 11 ) * st::pow( z, 3 ) - 2700 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 5 ) +
                3240 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 6 ) - 12960 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 7 ) +
                3240 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 8 ) - 2700 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 9 ) +
                540 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 11 ) + 240 * st::pow( x, 3 ) * st::pow( y, 12 ) * st::pow( z, 3 ) +
                540 * st::pow( x, 3 ) * st::pow( y, 11 ) * st::pow( z, 4 ) - 13896 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 5 ) -
                3180 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 6 ) - 34560 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 7 ) -
                34560 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 8 ) - 3180 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 9 ) -
                13896 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 10 ) + 540 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 11 ) +
                240 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 12 ) + 36 * x * st::pow( y, 12 ) * st::pow( z, 5 ) +
                108 * x * st::pow( y, 11 ) * st::pow( z, 6 ) - 864 * x * st::pow( y, 10 ) * st::pow( z, 7 ) -
                360 * x * st::pow( y, 9 ) * st::pow( z, 8 ) - 360 * x * st::pow( y, 8 ) * st::pow( z, 9 ) -
                864 * x * st::pow( y, 7 ) * st::pow( z, 10 ) + 108 * x * st::pow( y, 6 ) * st::pow( z, 11 ) +
                36 * x * st::pow( y, 5 ) * st::pow( z, 12 ) - 18 * st::pow( x, 12 ) * st::pow( y, 6 ) * z -
                90 * st::pow( x, 12 ) * st::pow( y, 4 ) * st::pow( z, 3 ) - 90 * st::pow( x, 12 ) * st::pow( y, 3 ) * st::pow( z, 4 ) -
                18 * st::pow( x, 12 ) * y * st::pow( z, 6 ) + 432 * st::pow( x, 11 ) * st::pow( y, 7 ) * z +
                7344 * st::pow( x, 11 ) * st::pow( y, 5 ) * st::pow( z, 3 ) + 7344 * st::pow( x, 11 ) * st::pow( y, 3 ) * st::pow( z, 5 ) +
                432 * st::pow( x, 11 ) * y * st::pow( z, 7 ) + 864 * st::pow( x, 10 ) * st::pow( y, 8 ) * z +
                11988 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 3 ) + 6480 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 4 ) +
                6480 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 5 ) + 11988 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 6 ) +
                864 * st::pow( x, 10 ) * y * st::pow( z, 8 ) - 150 * st::pow( x, 9 ) * st::pow( y, 9 ) * z -
                1080 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 3 ) + 1350 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 4 ) -
                6480 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 5 ) + 1350 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 6 ) -
                1080 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 7 ) - 150 * st::pow( x, 9 ) * y * st::pow( z, 9 ) +
                864 * st::pow( x, 8 ) * st::pow( y, 10 ) * z + 43200 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 3 ) +
                12960 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 4 ) + 85968 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 5 ) +
                85968 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 6 ) + 12960 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 7 ) +
                43200 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 8 ) + 864 * st::pow( x, 8 ) * y * st::pow( z, 10 ) +
                432 * st::pow( x, 7 ) * st::pow( y, 11 ) * z - 1080 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 3 ) +
                12960 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 4 ) - 124416 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 5 ) +
                51624 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 6 ) - 124416 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 7 ) +
                12960 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 8 ) - 1080 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 9 ) +
                432 * st::pow( x, 7 ) * y * st::pow( z, 11 ) - 18 * st::pow( x, 6 ) * st::pow( y, 12 ) * z +
                11988 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 3 ) + 1350 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 4 ) +
                85968 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 5 ) + 51624 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 6 ) +
                51624 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 7 ) + 85968 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 8 ) +
                1350 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 9 ) + 11988 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 10 ) -
                18 * st::pow( x, 6 ) * y * st::pow( z, 12 ) + 7344 * st::pow( x, 5 ) * st::pow( y, 11 ) * st::pow( z, 3 ) +
                6480 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 4 ) - 6480 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 5 ) +
                85968 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 6 ) - 124416 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 7 ) +
                85968 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 8 ) - 6480 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 9 ) +
                6480 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 10 ) + 7344 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 11 ) -
                90 * st::pow( x, 4 ) * st::pow( y, 12 ) * st::pow( z, 3 ) + 6480 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 5 ) +
                1350 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 6 ) + 12960 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 7 ) +
                12960 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 8 ) + 1350 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 9 ) +
                6480 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 10 ) - 90 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 12 ) -
                90 * st::pow( x, 3 ) * st::pow( y, 12 ) * st::pow( z, 4 ) + 7344 * st::pow( x, 3 ) * st::pow( y, 11 ) * st::pow( z, 5 ) +
                11988 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 6 ) - 1080 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 7 ) +
                43200 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 8 ) - 1080 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 9 ) +
                11988 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 10 ) + 7344 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 11 ) -
                90 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 12 ) - 18 * x * st::pow( y, 12 ) * st::pow( z, 6 ) +
                432 * x * st::pow( y, 11 ) * st::pow( z, 7 ) + 864 * x * st::pow( y, 10 ) * st::pow( z, 8 ) -
                150 * x * st::pow( y, 9 ) * st::pow( z, 9 ) + 864 * x * st::pow( y, 8 ) * st::pow( z, 10 ) +
                432 * x * st::pow( y, 7 ) * st::pow( z, 11 ) - 18 * x * st::pow( y, 6 ) * st::pow( z, 12 ) -
                72 * st::pow( x, 12 ) * st::pow( y, 7 ) * z - 1224 * st::pow( x, 12 ) * st::pow( y, 5 ) * st::pow( z, 3 ) -
                1224 * st::pow( x, 12 ) * st::pow( y, 3 ) * st::pow( z, 5 ) - 72 * st::pow( x, 12 ) * y * st::pow( z, 7 ) -
                432 * st::pow( x, 11 ) * st::pow( y, 8 ) * z - 6192 * st::pow( x, 11 ) * st::pow( y, 6 ) * st::pow( z, 3 ) -
                3240 * st::pow( x, 11 ) * st::pow( y, 5 ) * st::pow( z, 4 ) - 3240 * st::pow( x, 11 ) * st::pow( y, 4 ) * st::pow( z, 5 ) -
                6192 * st::pow( x, 11 ) * st::pow( y, 3 ) * st::pow( z, 6 ) - 432 * st::pow( x, 11 ) * y * st::pow( z, 8 ) +
                360 * st::pow( x, 10 ) * st::pow( y, 9 ) * z + 20016 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 3 ) -
                3240 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 4 ) + 67824 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 5 ) -
                3240 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 6 ) + 20016 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 7 ) +
                360 * st::pow( x, 10 ) * y * st::pow( z, 9 ) + 360 * st::pow( x, 9 ) * st::pow( y, 10 ) * z +
                4680 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 3 ) + 5400 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 4 ) +
                15840 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 5 ) + 15840 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 6 ) +
                5400 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 7 ) + 4680 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 8 ) +
                360 * st::pow( x, 9 ) * y * st::pow( z, 10 ) - 432 * st::pow( x, 8 ) * st::pow( y, 11 ) * z +
                4680 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 3 ) - 12960 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 4 ) +
                176256 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 5 ) - 58104 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 6 ) +
                176256 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 7 ) - 12960 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 8 ) +
                4680 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 9 ) - 432 * st::pow( x, 8 ) * y * st::pow( z, 11 ) -
                72 * st::pow( x, 7 ) * st::pow( y, 12 ) * z + 20016 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 3 ) +
                5400 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 4 ) + 176256 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 5 ) +
                122688 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 6 ) + 122688 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 7 ) +
                176256 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 8 ) + 5400 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 9 ) +
                20016 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 10 ) - 72 * st::pow( x, 7 ) * y * st::pow( z, 12 ) -
                6192 * st::pow( x, 6 ) * st::pow( y, 11 ) * st::pow( z, 3 ) - 3240 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 4 ) +
                15840 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 5 ) - 58104 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 6 ) +
                122688 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 7 ) - 58104 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 8 ) +
                15840 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 9 ) - 3240 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 10 ) -
                6192 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 11 ) - 1224 * st::pow( x, 5 ) * st::pow( y, 12 ) * st::pow( z, 3 ) -
                3240 * st::pow( x, 5 ) * st::pow( y, 11 ) * st::pow( z, 4 ) + 67824 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 5 ) +
                15840 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 6 ) + 176256 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 7 ) +
                176256 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 8 ) + 15840 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 9 ) +
                67824 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 10 ) - 3240 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 11 ) -
                1224 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 12 ) - 3240 * st::pow( x, 4 ) * st::pow( y, 11 ) * st::pow( z, 5 ) -
                3240 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 6 ) + 5400 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 7 ) -
                12960 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 8 ) + 5400 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 9 ) -
                3240 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 10 ) - 3240 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 11 ) -
                1224 * st::pow( x, 3 ) * st::pow( y, 12 ) * st::pow( z, 5 ) - 6192 * st::pow( x, 3 ) * st::pow( y, 11 ) * st::pow( z, 6 ) +
                20016 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 7 ) + 4680 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 8 ) +
                4680 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 9 ) + 20016 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 10 ) -
                6192 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 11 ) - 1224 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 12 ) -
                72 * x * st::pow( y, 12 ) * st::pow( z, 7 ) - 432 * x * st::pow( y, 11 ) * st::pow( z, 8 ) +
                360 * x * st::pow( y, 10 ) * st::pow( z, 9 ) + 360 * x * st::pow( y, 9 ) * st::pow( z, 10 ) -
                432 * x * st::pow( y, 8 ) * st::pow( z, 11 ) - 72 * x * st::pow( y, 7 ) * st::pow( z, 12 ) +
                72 * st::pow( x, 12 ) * st::pow( y, 8 ) * z + 1032 * st::pow( x, 12 ) * st::pow( y, 6 ) * st::pow( z, 3 ) +
                540 * st::pow( x, 12 ) * st::pow( y, 5 ) * st::pow( z, 4 ) + 540 * st::pow( x, 12 ) * st::pow( y, 4 ) * st::pow( z, 5 ) +
                1032 * st::pow( x, 12 ) * st::pow( y, 3 ) * st::pow( z, 6 ) + 72 * st::pow( x, 12 ) * y * st::pow( z, 8 ) -
                180 * st::pow( x, 11 ) * st::pow( y, 9 ) * z - 10800 * st::pow( x, 11 ) * st::pow( y, 7 ) * st::pow( z, 3 ) +
                1620 * st::pow( x, 11 ) * st::pow( y, 6 ) * st::pow( z, 4 ) - 36288 * st::pow( x, 11 ) * st::pow( y, 5 ) * st::pow( z, 5 ) +
                1620 * st::pow( x, 11 ) * st::pow( y, 4 ) * st::pow( z, 6 ) - 10800 * st::pow( x, 11 ) * st::pow( y, 3 ) * st::pow( z, 7 ) -
                180 * st::pow( x, 11 ) * y * st::pow( z, 9 ) - 864 * st::pow( x, 10 ) * st::pow( y, 10 ) * z -
                28656 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 3 ) - 12960 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 4 ) -
                64152 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 5 ) - 64152 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 6 ) -
                12960 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 7 ) - 28656 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 8 ) -
                864 * st::pow( x, 10 ) * y * st::pow( z, 10 ) - 180 * st::pow( x, 9 ) * st::pow( y, 11 ) * z -
                3600 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 3 ) - 5400 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 4 ) -
                6480 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 5 ) - 14220 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 6 ) -
                6480 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 7 ) - 5400 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 8 ) -
                3600 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 9 ) - 180 * st::pow( x, 9 ) * y * st::pow( z, 11 ) +
                72 * st::pow( x, 8 ) * st::pow( y, 12 ) * z - 28656 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 3 ) -
                5400 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 4 ) - 228096 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 5 ) -
                148608 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 6 ) - 148608 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 7 ) -
                228096 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 8 ) - 5400 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 9 ) -
                28656 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 10 ) + 72 * st::pow( x, 8 ) * y * st::pow( z, 12 ) -
                10800 * st::pow( x, 7 ) * st::pow( y, 11 ) * st::pow( z, 3 ) - 12960 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 4 ) -
                6480 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 5 ) - 148608 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 6 ) +
                155520 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 7 ) - 148608 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 8 ) -
                6480 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 9 ) - 12960 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 10 ) -
                10800 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 11 ) + 1032 * st::pow( x, 6 ) * st::pow( y, 12 ) * st::pow( z, 3 ) +
                1620 * st::pow( x, 6 ) * st::pow( y, 11 ) * st::pow( z, 4 ) - 64152 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 5 ) -
                14220 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 6 ) - 148608 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 7 ) -
                148608 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 8 ) - 14220 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 9 ) -
                64152 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 10 ) + 1620 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 11 ) +
                1032 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 12 ) + 540 * st::pow( x, 5 ) * st::pow( y, 12 ) * st::pow( z, 4 ) -
                36288 * st::pow( x, 5 ) * st::pow( y, 11 ) * st::pow( z, 5 ) - 64152 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 6 ) -
                6480 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 7 ) - 228096 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 8 ) -
                6480 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 9 ) - 64152 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 10 ) -
                36288 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 11 ) + 540 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 12 ) +
                540 * st::pow( x, 4 ) * st::pow( y, 12 ) * st::pow( z, 5 ) + 1620 * st::pow( x, 4 ) * st::pow( y, 11 ) * st::pow( z, 6 ) -
                12960 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 7 ) - 5400 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 8 ) -
                5400 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 9 ) - 12960 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 10 ) +
                1620 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 11 ) + 540 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 12 ) +
                1032 * st::pow( x, 3 ) * st::pow( y, 12 ) * st::pow( z, 6 ) - 10800 * st::pow( x, 3 ) * st::pow( y, 11 ) * st::pow( z, 7 ) -
                28656 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 8 ) - 3600 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 9 ) -
                28656 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 10 ) - 10800 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 11 ) +
                1032 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 12 ) + 72 * x * st::pow( y, 12 ) * st::pow( z, 8 ) -
                180 * x * st::pow( y, 11 ) * st::pow( z, 9 ) - 864 * x * st::pow( y, 10 ) * st::pow( z, 10 ) -
                180 * x * st::pow( y, 9 ) * st::pow( z, 11 ) + 72 * x * st::pow( y, 8 ) * st::pow( z, 12 ) +
                30 * st::pow( x, 12 ) * st::pow( y, 9 ) * z + 1800 * st::pow( x, 12 ) * st::pow( y, 7 ) * st::pow( z, 3 ) -
                270 * st::pow( x, 12 ) * st::pow( y, 6 ) * st::pow( z, 4 ) + 6048 * st::pow( x, 12 ) * st::pow( y, 5 ) * st::pow( z, 5 ) -
                270 * st::pow( x, 12 ) * st::pow( y, 4 ) * st::pow( z, 6 ) + 1800 * st::pow( x, 12 ) * st::pow( y, 3 ) * st::pow( z, 7 ) +
                30 * st::pow( x, 12 ) * y * st::pow( z, 9 ) + 432 * st::pow( x, 11 ) * st::pow( y, 10 ) * z +
                15120 * st::pow( x, 11 ) * st::pow( y, 8 ) * st::pow( z, 3 ) + 6480 * st::pow( x, 11 ) * st::pow( y, 7 ) * st::pow( z, 4 ) +
                33264 * st::pow( x, 11 ) * st::pow( y, 6 ) * st::pow( z, 5 ) + 33264 * st::pow( x, 11 ) * st::pow( y, 5 ) * st::pow( z, 6 ) +
                6480 * st::pow( x, 11 ) * st::pow( y, 4 ) * st::pow( z, 7 ) + 15120 * st::pow( x, 11 ) * st::pow( y, 3 ) * st::pow( z, 8 ) +
                432 * st::pow( x, 11 ) * y * st::pow( z, 10 ) + 432 * st::pow( x, 10 ) * st::pow( y, 11 ) * z +
                1380 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 3 ) + 12960 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 4 ) -
                88992 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 5 ) + 47196 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 6 ) -
                88992 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 7 ) + 12960 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 8 ) +
                1380 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 9 ) + 432 * st::pow( x, 10 ) * y * st::pow( z, 11 ) +
                30 * st::pow( x, 9 ) * st::pow( y, 12 ) * z + 1380 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 3 ) -
                2250 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 4 ) - 15120 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 5 ) -
                21960 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 6 ) - 21960 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 7 ) -
                15120 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 8 ) - 2250 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 9 ) +
                1380 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 10 ) + 30 * st::pow( x, 9 ) * y * st::pow( z, 12 ) +
                15120 * st::pow( x, 8 ) * st::pow( y, 11 ) * st::pow( z, 3 ) + 12960 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 4 ) -
                15120 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 5 ) + 174528 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 6 ) -
                259200 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 7 ) + 174528 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 8 ) -
                15120 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 9 ) + 12960 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 10 ) +
                15120 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 11 ) + 1800 * st::pow( x, 7 ) * st::pow( y, 12 ) * st::pow( z, 3 ) +
                6480 * st::pow( x, 7 ) * st::pow( y, 11 ) * st::pow( z, 4 ) - 88992 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 5 ) -
                21960 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 6 ) - 259200 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 7 ) -
                259200 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 8 ) - 21960 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 9 ) -
                88992 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 10 ) + 6480 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 11 ) +
                1800 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 12 ) - 270 * st::pow( x, 6 ) * st::pow( y, 12 ) * st::pow( z, 4 ) +
                33264 * st::pow( x, 6 ) * st::pow( y, 11 ) * st::pow( z, 5 ) + 47196 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 6 ) -
                21960 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 7 ) + 174528 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 8 ) -
                21960 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 9 ) + 47196 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 10 ) +
                33264 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 11 ) - 270 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 12 ) +
                6048 * st::pow( x, 5 ) * st::pow( y, 12 ) * st::pow( z, 5 ) + 33264 * st::pow( x, 5 ) * st::pow( y, 11 ) * st::pow( z, 6 ) -
                88992 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 7 ) - 15120 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 8 ) -
                15120 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 9 ) - 88992 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 10 ) +
                33264 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 11 ) + 6048 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 12 ) -
                270 * st::pow( x, 4 ) * st::pow( y, 12 ) * st::pow( z, 6 ) + 6480 * st::pow( x, 4 ) * st::pow( y, 11 ) * st::pow( z, 7 ) +
                12960 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 8 ) - 2250 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 9 ) +
                12960 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 10 ) + 6480 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 11 ) -
                270 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 12 ) + 1800 * st::pow( x, 3 ) * st::pow( y, 12 ) * st::pow( z, 7 ) +
                15120 * st::pow( x, 3 ) * st::pow( y, 11 ) * st::pow( z, 8 ) + 1380 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 9 ) +
                1380 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 10 ) + 15120 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 11 ) +
                1800 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 12 ) + 30 * x * st::pow( y, 12 ) * st::pow( z, 9 ) +
                432 * x * st::pow( y, 11 ) * st::pow( z, 10 ) + 432 * x * st::pow( y, 10 ) * st::pow( z, 11 ) +
                30 * x * st::pow( y, 9 ) * st::pow( z, 12 ) - 72 * st::pow( x, 12 ) * st::pow( y, 10 ) * z -
                2520 * st::pow( x, 12 ) * st::pow( y, 8 ) * st::pow( z, 3 ) - 1080 * st::pow( x, 12 ) * st::pow( y, 7 ) * st::pow( z, 4 ) -
                5544 * st::pow( x, 12 ) * st::pow( y, 6 ) * st::pow( z, 5 ) - 5544 * st::pow( x, 12 ) * st::pow( y, 5 ) * st::pow( z, 6 ) -
                1080 * st::pow( x, 12 ) * st::pow( y, 4 ) * st::pow( z, 7 ) - 2520 * st::pow( x, 12 ) * st::pow( y, 3 ) * st::pow( z, 8 ) -
                72 * st::pow( x, 12 ) * y * st::pow( z, 10 ) - 216 * st::pow( x, 11 ) * st::pow( y, 11 ) * z -
                360 * st::pow( x, 11 ) * st::pow( y, 9 ) * st::pow( z, 3 ) - 6480 * st::pow( x, 11 ) * st::pow( y, 8 ) * st::pow( z, 4 ) +
                49248 * st::pow( x, 11 ) * st::pow( y, 7 ) * st::pow( z, 5 ) - 24192 * st::pow( x, 11 ) * st::pow( y, 6 ) * st::pow( z, 6 ) +
                49248 * st::pow( x, 11 ) * st::pow( y, 5 ) * st::pow( z, 7 ) - 6480 * st::pow( x, 11 ) * st::pow( y, 4 ) * st::pow( z, 8 ) -
                360 * st::pow( x, 11 ) * st::pow( y, 3 ) * st::pow( z, 9 ) - 216 * st::pow( x, 11 ) * y * st::pow( z, 11 ) -
                72 * st::pow( x, 10 ) * st::pow( y, 12 ) * z + 14112 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 3 ) +
                5400 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 4 ) + 140832 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 5 ) +
                104976 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 6 ) + 104976 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 7 ) +
                140832 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 8 ) + 5400 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 9 ) +
                14112 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 10 ) - 72 * st::pow( x, 10 ) * y * st::pow( z, 12 ) -
                360 * st::pow( x, 9 ) * st::pow( y, 11 ) * st::pow( z, 3 ) + 5400 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 4 ) +
                27000 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 5 ) + 32760 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 6 ) +
                51840 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 7 ) + 32760 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 8 ) +
                27000 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 9 ) + 5400 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 10 ) -
                360 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 11 ) - 2520 * st::pow( x, 8 ) * st::pow( y, 12 ) * st::pow( z, 3 ) -
                6480 * st::pow( x, 8 ) * st::pow( y, 11 ) * st::pow( z, 4 ) + 140832 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 5 ) +
                32760 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 6 ) + 362880 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 7 ) +
                362880 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 8 ) + 32760 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 9 ) +
                140832 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 10 ) - 6480 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 11 ) -
                2520 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 12 ) - 1080 * st::pow( x, 7 ) * st::pow( y, 12 ) * st::pow( z, 4 ) +
                49248 * st::pow( x, 7 ) * st::pow( y, 11 ) * st::pow( z, 5 ) + 104976 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 6 ) +
                51840 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 7 ) + 362880 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 8 ) +
                51840 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 9 ) + 104976 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 10 ) +
                49248 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 11 ) - 1080 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 12 ) -
                5544 * st::pow( x, 6 ) * st::pow( y, 12 ) * st::pow( z, 5 ) - 24192 * st::pow( x, 6 ) * st::pow( y, 11 ) * st::pow( z, 6 ) +
                104976 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 7 ) + 32760 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 8 ) +
                32760 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 9 ) + 104976 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 10 ) -
                24192 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 11 ) - 5544 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 12 ) -
                5544 * st::pow( x, 5 ) * st::pow( y, 12 ) * st::pow( z, 6 ) + 49248 * st::pow( x, 5 ) * st::pow( y, 11 ) * st::pow( z, 7 ) +
                140832 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 8 ) + 27000 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 9 ) +
                140832 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 10 ) + 49248 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 11 ) -
                5544 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 12 ) - 1080 * st::pow( x, 4 ) * st::pow( y, 12 ) * st::pow( z, 7 ) -
                6480 * st::pow( x, 4 ) * st::pow( y, 11 ) * st::pow( z, 8 ) + 5400 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 9 ) +
                5400 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 10 ) - 6480 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 11 ) -
                1080 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 12 ) - 2520 * st::pow( x, 3 ) * st::pow( y, 12 ) * st::pow( z, 8 ) -
                360 * st::pow( x, 3 ) * st::pow( y, 11 ) * st::pow( z, 9 ) + 14112 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 10 ) -
                360 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 11 ) - 2520 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 12 ) -
                72 * x * st::pow( y, 12 ) * st::pow( z, 10 ) - 216 * x * st::pow( y, 11 ) * st::pow( z, 11 ) -
                72 * x * st::pow( y, 10 ) * st::pow( z, 12 ) + 36 * st::pow( x, 12 ) * st::pow( y, 11 ) * z +
                60 * st::pow( x, 12 ) * st::pow( y, 9 ) * st::pow( z, 3 ) + 1080 * st::pow( x, 12 ) * st::pow( y, 8 ) * st::pow( z, 4 ) -
                8208 * st::pow( x, 12 ) * st::pow( y, 7 ) * st::pow( z, 5 ) + 4032 * st::pow( x, 12 ) * st::pow( y, 6 ) * st::pow( z, 6 ) -
                8208 * st::pow( x, 12 ) * st::pow( y, 5 ) * st::pow( z, 7 ) + 1080 * st::pow( x, 12 ) * st::pow( y, 4 ) * st::pow( z, 8 ) +
                60 * st::pow( x, 12 ) * st::pow( y, 3 ) * st::pow( z, 9 ) + 36 * st::pow( x, 12 ) * y * st::pow( z, 11 ) +
                36 * st::pow( x, 11 ) * st::pow( y, 12 ) * z - 7848 * st::pow( x, 11 ) * st::pow( y, 10 ) * st::pow( z, 3 ) -
                2700 * st::pow( x, 11 ) * st::pow( y, 9 ) * st::pow( z, 4 ) - 75168 * st::pow( x, 11 ) * st::pow( y, 8 ) * st::pow( z, 5 ) -
                54864 * st::pow( x, 11 ) * st::pow( y, 7 ) * st::pow( z, 6 ) - 54864 * st::pow( x, 11 ) * st::pow( y, 6 ) * st::pow( z, 7 ) -
                75168 * st::pow( x, 11 ) * st::pow( y, 5 ) * st::pow( z, 8 ) - 2700 * st::pow( x, 11 ) * st::pow( y, 4 ) * st::pow( z, 9 ) -
                7848 * st::pow( x, 11 ) * st::pow( y, 3 ) * st::pow( z, 10 ) + 36 * st::pow( x, 11 ) * y * st::pow( z, 12 ) -
                7848 * st::pow( x, 10 ) * st::pow( y, 11 ) * st::pow( z, 3 ) - 12960 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 4 ) -
                21240 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 5 ) - 130896 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 6 ) +
                84672 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 7 ) - 130896 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 8 ) -
                21240 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 9 ) - 12960 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 10 ) -
                7848 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 11 ) + 60 * st::pow( x, 9 ) * st::pow( y, 12 ) * st::pow( z, 3 ) -
                2700 * st::pow( x, 9 ) * st::pow( y, 11 ) * st::pow( z, 4 ) - 21240 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 5 ) -
                3000 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 6 ) - 8640 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 7 ) -
                8640 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 8 ) - 3000 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 9 ) -
                21240 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 10 ) - 2700 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 11 ) +
                60 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 12 ) + 1080 * st::pow( x, 8 ) * st::pow( y, 12 ) * st::pow( z, 4 ) -
                75168 * st::pow( x, 8 ) * st::pow( y, 11 ) * st::pow( z, 5 ) - 130896 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 6 ) -
                8640 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 7 ) - 466560 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 8 ) -
                8640 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 9 ) - 130896 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 10 ) -
                75168 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 11 ) + 1080 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 12 ) -
                8208 * st::pow( x, 7 ) * st::pow( y, 12 ) * st::pow( z, 5 ) - 54864 * st::pow( x, 7 ) * st::pow( y, 11 ) * st::pow( z, 6 ) +
                84672 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 7 ) - 8640 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 8 ) -
                8640 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 9 ) + 84672 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 10 ) -
                54864 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 11 ) - 8208 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 12 ) +
                4032 * st::pow( x, 6 ) * st::pow( y, 12 ) * st::pow( z, 6 ) - 54864 * st::pow( x, 6 ) * st::pow( y, 11 ) * st::pow( z, 7 ) -
                130896 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 8 ) - 3000 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 9 ) -
                130896 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 10 ) - 54864 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 11 ) +
                4032 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 12 ) - 8208 * st::pow( x, 5 ) * st::pow( y, 12 ) * st::pow( z, 7 ) -
                75168 * st::pow( x, 5 ) * st::pow( y, 11 ) * st::pow( z, 8 ) - 21240 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 9 ) -
                21240 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 10 ) - 75168 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 11 ) -
                8208 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 12 ) + 1080 * st::pow( x, 4 ) * st::pow( y, 12 ) * st::pow( z, 8 ) -
                2700 * st::pow( x, 4 ) * st::pow( y, 11 ) * st::pow( z, 9 ) - 12960 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 10 ) -
                2700 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 11 ) + 1080 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 12 ) +
                60 * st::pow( x, 3 ) * st::pow( y, 12 ) * st::pow( z, 9 ) - 7848 * st::pow( x, 3 ) * st::pow( y, 11 ) * st::pow( z, 10 ) -
                7848 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 11 ) + 60 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 12 ) +
                36 * x * st::pow( y, 12 ) * st::pow( z, 11 ) + 36 * x * st::pow( y, 11 ) * st::pow( z, 12 ) -
                6 * st::pow( x, 12 ) * st::pow( y, 12 ) * z + 1308 * st::pow( x, 12 ) * st::pow( y, 10 ) * st::pow( z, 3 ) +
                450 * st::pow( x, 12 ) * st::pow( y, 9 ) * st::pow( z, 4 ) + 12528 * st::pow( x, 12 ) * st::pow( y, 8 ) * st::pow( z, 5 ) +
                9144 * st::pow( x, 12 ) * st::pow( y, 7 ) * st::pow( z, 6 ) + 9144 * st::pow( x, 12 ) * st::pow( y, 6 ) * st::pow( z, 7 ) +
                12528 * st::pow( x, 12 ) * st::pow( y, 5 ) * st::pow( z, 8 ) + 450 * st::pow( x, 12 ) * st::pow( y, 4 ) * st::pow( z, 9 ) +
                1308 * st::pow( x, 12 ) * st::pow( y, 3 ) * st::pow( z, 10 ) - 6 * st::pow( x, 12 ) * y * st::pow( z, 12 ) +
                4320 * st::pow( x, 11 ) * st::pow( y, 11 ) * st::pow( z, 3 ) + 6480 * st::pow( x, 11 ) * st::pow( y, 10 ) * st::pow( z, 4 ) +
                8640 * st::pow( x, 11 ) * st::pow( y, 9 ) * st::pow( z, 5 ) + 67824 * st::pow( x, 11 ) * st::pow( y, 8 ) * st::pow( z, 6 ) -
                51840 * st::pow( x, 11 ) * st::pow( y, 7 ) * st::pow( z, 7 ) + 67824 * st::pow( x, 11 ) * st::pow( y, 6 ) * st::pow( z, 8 ) +
                8640 * st::pow( x, 11 ) * st::pow( y, 5 ) * st::pow( z, 9 ) + 6480 * st::pow( x, 11 ) * st::pow( y, 4 ) * st::pow( z, 10 ) +
                4320 * st::pow( x, 11 ) * st::pow( y, 3 ) * st::pow( z, 11 ) + 1308 * st::pow( x, 10 ) * st::pow( y, 12 ) * st::pow( z, 3 ) +
                6480 * st::pow( x, 10 ) * st::pow( y, 11 ) * st::pow( z, 4 ) - 53568 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 5 ) -
                14580 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 6 ) - 188352 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 7 ) -
                188352 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 8 ) - 14580 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 9 ) -
                53568 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 10 ) + 6480 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 11 ) +
                1308 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 12 ) + 450 * st::pow( x, 9 ) * st::pow( y, 12 ) * st::pow( z, 4 ) +
                8640 * st::pow( x, 9 ) * st::pow( y, 11 ) * st::pow( z, 5 ) - 14580 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 6 ) -
                70200 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 7 ) - 34560 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 8 ) -
                70200 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 9 ) - 14580 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 10 ) +
                8640 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 11 ) + 450 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 12 ) +
                12528 * st::pow( x, 8 ) * st::pow( y, 12 ) * st::pow( z, 5 ) + 67824 * st::pow( x, 8 ) * st::pow( y, 11 ) * st::pow( z, 6 ) -
                188352 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 7 ) - 34560 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 8 ) -
                34560 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 9 ) - 188352 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 10 ) +
                67824 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 11 ) + 12528 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 12 ) +
                9144 * st::pow( x, 7 ) * st::pow( y, 12 ) * st::pow( z, 6 ) - 51840 * st::pow( x, 7 ) * st::pow( y, 11 ) * st::pow( z, 7 ) -
                188352 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 8 ) - 70200 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 9 ) -
                188352 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 10 ) - 51840 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 11 ) +
                9144 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 12 ) + 9144 * st::pow( x, 6 ) * st::pow( y, 12 ) * st::pow( z, 7 ) +
                67824 * st::pow( x, 6 ) * st::pow( y, 11 ) * st::pow( z, 8 ) - 14580 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 9 ) -
                14580 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 10 ) + 67824 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 11 ) +
                9144 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 12 ) + 12528 * st::pow( x, 5 ) * st::pow( y, 12 ) * st::pow( z, 8 ) +
                8640 * st::pow( x, 5 ) * st::pow( y, 11 ) * st::pow( z, 9 ) - 53568 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 10 ) +
                8640 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 11 ) + 12528 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 12 ) +
                450 * st::pow( x, 4 ) * st::pow( y, 12 ) * st::pow( z, 9 ) + 6480 * st::pow( x, 4 ) * st::pow( y, 11 ) * st::pow( z, 10 ) +
                6480 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 11 ) + 450 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 12 ) +
                1308 * st::pow( x, 3 ) * st::pow( y, 12 ) * st::pow( z, 10 ) + 4320 * st::pow( x, 3 ) * st::pow( y, 11 ) * st::pow( z, 11 ) +
                1308 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 12 ) - 6 * x * st::pow( y, 12 ) * st::pow( z, 12 ) -
                720 * st::pow( x, 12 ) * st::pow( y, 11 ) * st::pow( z, 3 ) - 1080 * st::pow( x, 12 ) * st::pow( y, 10 ) * st::pow( z, 4 ) -
                1440 * st::pow( x, 12 ) * st::pow( y, 9 ) * st::pow( z, 5 ) - 11304 * st::pow( x, 12 ) * st::pow( y, 8 ) * st::pow( z, 6 ) +
                8640 * st::pow( x, 12 ) * st::pow( y, 7 ) * st::pow( z, 7 ) - 11304 * st::pow( x, 12 ) * st::pow( y, 6 ) * st::pow( z, 8 ) -
                1440 * st::pow( x, 12 ) * st::pow( y, 5 ) * st::pow( z, 9 ) - 1080 * st::pow( x, 12 ) * st::pow( y, 4 ) * st::pow( z, 10 ) -
                720 * st::pow( x, 12 ) * st::pow( y, 3 ) * st::pow( z, 11 ) - 720 * st::pow( x, 11 ) * st::pow( y, 12 ) * st::pow( z, 3 ) -
                3240 * st::pow( x, 11 ) * st::pow( y, 11 ) * st::pow( z, 4 ) + 31536 * st::pow( x, 11 ) * st::pow( y, 10 ) * st::pow( z, 5 ) +
                8280 * st::pow( x, 11 ) * st::pow( y, 9 ) * st::pow( z, 6 ) + 103680 * st::pow( x, 11 ) * st::pow( y, 8 ) * st::pow( z, 7 ) +
                103680 * st::pow( x, 11 ) * st::pow( y, 7 ) * st::pow( z, 8 ) + 8280 * st::pow( x, 11 ) * st::pow( y, 6 ) * st::pow( z, 9 ) +
                31536 * st::pow( x, 11 ) * st::pow( y, 5 ) * st::pow( z, 10 ) - 3240 * st::pow( x, 11 ) * st::pow( y, 4 ) * st::pow( z, 11 ) -
                720 * st::pow( x, 11 ) * st::pow( y, 3 ) * st::pow( z, 12 ) - 1080 * st::pow( x, 10 ) * st::pow( y, 12 ) * st::pow( z, 4 ) +
                31536 * st::pow( x, 10 ) * st::pow( y, 11 ) * st::pow( z, 5 ) + 87264 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 6 ) +
                81360 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 7 ) + 292032 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 8 ) +
                81360 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 9 ) + 87264 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 10 ) +
                31536 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 11 ) - 1080 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 12 ) -
                1440 * st::pow( x, 9 ) * st::pow( y, 12 ) * st::pow( z, 5 ) + 8280 * st::pow( x, 9 ) * st::pow( y, 11 ) * st::pow( z, 6 ) +
                81360 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 7 ) + 52200 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 8 ) +
                52200 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 9 ) + 81360 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 10 ) +
                8280 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 11 ) - 1440 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 12 ) -
                11304 * st::pow( x, 8 ) * st::pow( y, 12 ) * st::pow( z, 6 ) + 103680 * st::pow( x, 8 ) * st::pow( y, 11 ) * st::pow( z, 7 ) +
                292032 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 8 ) + 52200 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 9 ) +
                292032 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 10 ) + 103680 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 11 ) -
                11304 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 12 ) + 8640 * st::pow( x, 7 ) * st::pow( y, 12 ) * st::pow( z, 7 ) +
                103680 * st::pow( x, 7 ) * st::pow( y, 11 ) * st::pow( z, 8 ) + 81360 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 9 ) +
                81360 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 10 ) + 103680 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 11 ) +
                8640 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 12 ) - 11304 * st::pow( x, 6 ) * st::pow( y, 12 ) * st::pow( z, 8 ) +
                8280 * st::pow( x, 6 ) * st::pow( y, 11 ) * st::pow( z, 9 ) + 87264 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 10 ) +
                8280 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 11 ) - 11304 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 12 ) -
                1440 * st::pow( x, 5 ) * st::pow( y, 12 ) * st::pow( z, 9 ) + 31536 * st::pow( x, 5 ) * st::pow( y, 11 ) * st::pow( z, 10 ) +
                31536 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 11 ) - 1440 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 12 ) -
                1080 * st::pow( x, 4 ) * st::pow( y, 12 ) * st::pow( z, 10 ) - 3240 * st::pow( x, 4 ) * st::pow( y, 11 ) * st::pow( z, 11 ) -
                1080 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 12 ) - 720 * st::pow( x, 3 ) * st::pow( y, 12 ) * st::pow( z, 11 ) -
                720 * st::pow( x, 3 ) * st::pow( y, 11 ) * st::pow( z, 12 ) + 120 * st::pow( x, 12 ) * st::pow( y, 12 ) * st::pow( z, 3 ) +
                540 * st::pow( x, 12 ) * st::pow( y, 11 ) * st::pow( z, 4 ) - 5256 * st::pow( x, 12 ) * st::pow( y, 10 ) * st::pow( z, 5 ) -
                1380 * st::pow( x, 12 ) * st::pow( y, 9 ) * st::pow( z, 6 ) - 17280 * st::pow( x, 12 ) * st::pow( y, 8 ) * st::pow( z, 7 ) -
                17280 * st::pow( x, 12 ) * st::pow( y, 7 ) * st::pow( z, 8 ) - 1380 * st::pow( x, 12 ) * st::pow( y, 6 ) * st::pow( z, 9 ) -
                5256 * st::pow( x, 12 ) * st::pow( y, 5 ) * st::pow( z, 10 ) + 540 * st::pow( x, 12 ) * st::pow( y, 4 ) * st::pow( z, 11 ) +
                120 * st::pow( x, 12 ) * st::pow( y, 3 ) * st::pow( z, 12 ) + 540 * st::pow( x, 11 ) * st::pow( y, 12 ) * st::pow( z, 4 ) -
                18144 * st::pow( x, 11 ) * st::pow( y, 11 ) * st::pow( z, 5 ) - 46008 * st::pow( x, 11 ) * st::pow( y, 10 ) * st::pow( z, 6 ) -
                36720 * st::pow( x, 11 ) * st::pow( y, 9 ) * st::pow( z, 7 ) - 155520 * st::pow( x, 11 ) * st::pow( y, 8 ) * st::pow( z, 8 ) -
                36720 * st::pow( x, 11 ) * st::pow( y, 7 ) * st::pow( z, 9 ) - 46008 * st::pow( x, 11 ) * st::pow( y, 6 ) * st::pow( z, 10 ) -
                18144 * st::pow( x, 11 ) * st::pow( y, 5 ) * st::pow( z, 11 ) + 540 * st::pow( x, 11 ) * st::pow( y, 4 ) * st::pow( z, 12 ) -
                5256 * st::pow( x, 10 ) * st::pow( y, 12 ) * st::pow( z, 5 ) - 46008 * st::pow( x, 10 ) * st::pow( y, 11 ) * st::pow( z, 6 ) +
                13824 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 7 ) - 38160 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 8 ) -
                38160 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 9 ) + 13824 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 10 ) -
                46008 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 11 ) - 5256 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 12 ) -
                1380 * st::pow( x, 9 ) * st::pow( y, 12 ) * st::pow( z, 6 ) - 36720 * st::pow( x, 9 ) * st::pow( y, 11 ) * st::pow( z, 7 ) -
                38160 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 8 ) + 49500 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 9 ) -
                38160 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 10 ) - 36720 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 11 ) -
                1380 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 12 ) - 17280 * st::pow( x, 8 ) * st::pow( y, 12 ) * st::pow( z, 7 ) -
                155520 * st::pow( x, 8 ) * st::pow( y, 11 ) * st::pow( z, 8 ) - 38160 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 9 ) -
                38160 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 10 ) - 155520 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 11 ) -
                17280 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 12 ) - 17280 * st::pow( x, 7 ) * st::pow( y, 12 ) * st::pow( z, 8 ) -
                36720 * st::pow( x, 7 ) * st::pow( y, 11 ) * st::pow( z, 9 ) + 13824 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 10 ) -
                36720 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 11 ) - 17280 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 12 ) -
                1380 * st::pow( x, 6 ) * st::pow( y, 12 ) * st::pow( z, 9 ) - 46008 * st::pow( x, 6 ) * st::pow( y, 11 ) * st::pow( z, 10 ) -
                46008 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 11 ) - 1380 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 12 ) -
                5256 * st::pow( x, 5 ) * st::pow( y, 12 ) * st::pow( z, 10 ) - 18144 * st::pow( x, 5 ) * st::pow( y, 11 ) * st::pow( z, 11 ) -
                5256 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 12 ) + 540 * st::pow( x, 4 ) * st::pow( y, 12 ) * st::pow( z, 11 ) +
                540 * st::pow( x, 4 ) * st::pow( y, 11 ) * st::pow( z, 12 ) + 120 * st::pow( x, 3 ) * st::pow( y, 12 ) * st::pow( z, 12 ) -
                90 * st::pow( x, 12 ) * st::pow( y, 12 ) * st::pow( z, 4 ) + 3024 * st::pow( x, 12 ) * st::pow( y, 11 ) * st::pow( z, 5 ) +
                7668 * st::pow( x, 12 ) * st::pow( y, 10 ) * st::pow( z, 6 ) + 6120 * st::pow( x, 12 ) * st::pow( y, 9 ) * st::pow( z, 7 ) +
                25920 * st::pow( x, 12 ) * st::pow( y, 8 ) * st::pow( z, 8 ) + 6120 * st::pow( x, 12 ) * st::pow( y, 7 ) * st::pow( z, 9 ) +
                7668 * st::pow( x, 12 ) * st::pow( y, 6 ) * st::pow( z, 10 ) + 3024 * st::pow( x, 12 ) * st::pow( y, 5 ) * st::pow( z, 11 ) -
                90 * st::pow( x, 12 ) * st::pow( y, 4 ) * st::pow( z, 12 ) + 3024 * st::pow( x, 11 ) * st::pow( y, 12 ) * st::pow( z, 5 ) +
                24192 * st::pow( x, 11 ) * st::pow( y, 11 ) * st::pow( z, 6 ) - 16416 * st::pow( x, 11 ) * st::pow( y, 10 ) * st::pow( z, 7 ) +
                15120 * st::pow( x, 11 ) * st::pow( y, 9 ) * st::pow( z, 8 ) + 15120 * st::pow( x, 11 ) * st::pow( y, 8 ) * st::pow( z, 9 ) -
                16416 * st::pow( x, 11 ) * st::pow( y, 7 ) * st::pow( z, 10 ) + 24192 * st::pow( x, 11 ) * st::pow( y, 6 ) * st::pow( z, 11 ) +
                3024 * st::pow( x, 11 ) * st::pow( y, 5 ) * st::pow( z, 12 ) + 7668 * st::pow( x, 10 ) * st::pow( y, 12 ) * st::pow( z, 6 ) -
                16416 * st::pow( x, 10 ) * st::pow( y, 11 ) * st::pow( z, 7 ) - 117504 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 8 ) -
                82500 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 9 ) - 117504 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 10 ) -
                16416 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 11 ) + 7668 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 12 ) +
                6120 * st::pow( x, 9 ) * st::pow( y, 12 ) * st::pow( z, 7 ) + 15120 * st::pow( x, 9 ) * st::pow( y, 11 ) * st::pow( z, 8 ) -
                82500 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 9 ) - 82500 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 10 ) +
                15120 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 11 ) + 6120 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 12 ) +
                25920 * st::pow( x, 8 ) * st::pow( y, 12 ) * st::pow( z, 8 ) + 15120 * st::pow( x, 8 ) * st::pow( y, 11 ) * st::pow( z, 9 ) -
                117504 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 10 ) + 15120 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 11 ) +
                25920 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 12 ) + 6120 * st::pow( x, 7 ) * st::pow( y, 12 ) * st::pow( z, 9 ) -
                16416 * st::pow( x, 7 ) * st::pow( y, 11 ) * st::pow( z, 10 ) - 16416 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 11 ) +
                6120 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 12 ) + 7668 * st::pow( x, 6 ) * st::pow( y, 12 ) * st::pow( z, 10 ) +
                24192 * st::pow( x, 6 ) * st::pow( y, 11 ) * st::pow( z, 11 ) + 7668 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 12 ) +
                3024 * st::pow( x, 5 ) * st::pow( y, 12 ) * st::pow( z, 11 ) + 3024 * st::pow( x, 5 ) * st::pow( y, 11 ) * st::pow( z, 12 ) -
                90 * st::pow( x, 4 ) * st::pow( y, 12 ) * st::pow( z, 12 ) - 504 * st::pow( x, 12 ) * st::pow( y, 12 ) * st::pow( z, 5 ) -
                4032 * st::pow( x, 12 ) * st::pow( y, 11 ) * st::pow( z, 6 ) + 2736 * st::pow( x, 12 ) * st::pow( y, 10 ) * st::pow( z, 7 ) -
                2520 * st::pow( x, 12 ) * st::pow( y, 9 ) * st::pow( z, 8 ) - 2520 * st::pow( x, 12 ) * st::pow( y, 8 ) * st::pow( z, 9 ) +
                2736 * st::pow( x, 12 ) * st::pow( y, 7 ) * st::pow( z, 10 ) - 4032 * st::pow( x, 12 ) * st::pow( y, 6 ) * st::pow( z, 11 ) -
                504 * st::pow( x, 12 ) * st::pow( y, 5 ) * st::pow( z, 12 ) - 4032 * st::pow( x, 11 ) * st::pow( y, 12 ) * st::pow( z, 6 ) +
                12960 * st::pow( x, 11 ) * st::pow( y, 11 ) * st::pow( z, 7 ) + 68256 * st::pow( x, 11 ) * st::pow( y, 10 ) * st::pow( z, 8 ) +
                39600 * st::pow( x, 11 ) * st::pow( y, 9 ) * st::pow( z, 9 ) + 68256 * st::pow( x, 11 ) * st::pow( y, 8 ) * st::pow( z, 10 ) +
                12960 * st::pow( x, 11 ) * st::pow( y, 7 ) * st::pow( z, 11 ) - 4032 * st::pow( x, 11 ) * st::pow( y, 6 ) * st::pow( z, 12 ) +
                2736 * st::pow( x, 10 ) * st::pow( y, 12 ) * st::pow( z, 7 ) + 68256 * st::pow( x, 10 ) * st::pow( y, 11 ) * st::pow( z, 8 ) +
                110880 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 9 ) +
                110880 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 10 ) + 68256 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 11 ) +
                2736 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 12 ) - 2520 * st::pow( x, 9 ) * st::pow( y, 12 ) * st::pow( z, 8 ) +
                39600 * st::pow( x, 9 ) * st::pow( y, 11 ) * st::pow( z, 9 ) + 110880 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 10 ) +
                39600 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 11 ) - 2520 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 12 ) -
                2520 * st::pow( x, 8 ) * st::pow( y, 12 ) * st::pow( z, 9 ) + 68256 * st::pow( x, 8 ) * st::pow( y, 11 ) * st::pow( z, 10 ) +
                68256 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 11 ) - 2520 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 12 ) +
                2736 * st::pow( x, 7 ) * st::pow( y, 12 ) * st::pow( z, 10 ) + 12960 * st::pow( x, 7 ) * st::pow( y, 11 ) * st::pow( z, 11 ) +
                2736 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 12 ) - 4032 * st::pow( x, 6 ) * st::pow( y, 12 ) * st::pow( z, 11 ) -
                4032 * st::pow( x, 6 ) * st::pow( y, 11 ) * st::pow( z, 12 ) - 504 * st::pow( x, 5 ) * st::pow( y, 12 ) * st::pow( z, 12 ) +
                672 * st::pow( x, 12 ) * st::pow( y, 12 ) * st::pow( z, 6 ) - 2160 * st::pow( x, 12 ) * st::pow( y, 11 ) * st::pow( z, 7 ) -
                11376 * st::pow( x, 12 ) * st::pow( y, 10 ) * st::pow( z, 8 ) - 6600 * st::pow( x, 12 ) * st::pow( y, 9 ) * st::pow( z, 9 ) -
                11376 * st::pow( x, 12 ) * st::pow( y, 8 ) * st::pow( z, 10 ) - 2160 * st::pow( x, 12 ) * st::pow( y, 7 ) * st::pow( z, 11 ) +
                672 * st::pow( x, 12 ) * st::pow( y, 6 ) * st::pow( z, 12 ) - 2160 * st::pow( x, 11 ) * st::pow( y, 12 ) * st::pow( z, 7 ) -
                38880 * st::pow( x, 11 ) * st::pow( y, 11 ) * st::pow( z, 8 ) - 51480 * st::pow( x, 11 ) * st::pow( y, 10 ) * st::pow( z, 9 ) -
                51480 * st::pow( x, 11 ) * st::pow( y, 9 ) * st::pow( z, 10 ) - 38880 * st::pow( x, 11 ) * st::pow( y, 8 ) * st::pow( z, 11 ) -
                2160 * st::pow( x, 11 ) * st::pow( y, 7 ) * st::pow( z, 12 ) - 11376 * st::pow( x, 10 ) * st::pow( y, 12 ) * st::pow( z, 8 ) -
                51480 * st::pow( x, 10 ) * st::pow( y, 11 ) * st::pow( z, 9 ) - 57024 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 10 ) -
                51480 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 11 ) - 11376 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 12 ) -
                6600 * st::pow( x, 9 ) * st::pow( y, 12 ) * st::pow( z, 9 ) - 51480 * st::pow( x, 9 ) * st::pow( y, 11 ) * st::pow( z, 10 ) -
                51480 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 11 ) - 6600 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 12 ) -
                11376 * st::pow( x, 8 ) * st::pow( y, 12 ) * st::pow( z, 10 ) - 38880 * st::pow( x, 8 ) * st::pow( y, 11 ) * st::pow( z, 11 ) -
                11376 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 12 ) - 2160 * st::pow( x, 7 ) * st::pow( y, 12 ) * st::pow( z, 11 ) -
                2160 * st::pow( x, 7 ) * st::pow( y, 11 ) * st::pow( z, 12 ) + 672 * st::pow( x, 6 ) * st::pow( y, 12 ) * st::pow( z, 12 ) +
                360 * st::pow( x, 12 ) * st::pow( y, 12 ) * st::pow( z, 7 ) + 6480 * st::pow( x, 12 ) * st::pow( y, 11 ) * st::pow( z, 8 ) +
                8580 * st::pow( x, 12 ) * st::pow( y, 10 ) * st::pow( z, 9 ) + 8580 * st::pow( x, 12 ) * st::pow( y, 9 ) * st::pow( z, 10 ) +
                6480 * st::pow( x, 12 ) * st::pow( y, 8 ) * st::pow( z, 11 ) + 360 * st::pow( x, 12 ) * st::pow( y, 7 ) * st::pow( z, 12 ) +
                6480 * st::pow( x, 11 ) * st::pow( y, 12 ) * st::pow( z, 8 ) + 23760 * st::pow( x, 11 ) * st::pow( y, 11 ) * st::pow( z, 9 ) +
                19008 * st::pow( x, 11 ) * st::pow( y, 10 ) * st::pow( z, 10 ) + 23760 * st::pow( x, 11 ) * st::pow( y, 9 ) * st::pow( z, 11 ) +
                6480 * st::pow( x, 11 ) * st::pow( y, 8 ) * st::pow( z, 12 ) + 8580 * st::pow( x, 10 ) * st::pow( y, 12 ) * st::pow( z, 9 ) +
                19008 * st::pow( x, 10 ) * st::pow( y, 11 ) * st::pow( z, 10 ) +
                19008 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 11 ) + 8580 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 12 ) +
                8580 * st::pow( x, 9 ) * st::pow( y, 12 ) * st::pow( z, 10 ) + 23760 * st::pow( x, 9 ) * st::pow( y, 11 ) * st::pow( z, 11 ) +
                8580 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 12 ) + 6480 * st::pow( x, 8 ) * st::pow( y, 12 ) * st::pow( z, 11 ) +
                6480 * st::pow( x, 8 ) * st::pow( y, 11 ) * st::pow( z, 12 ) + 360 * st::pow( x, 7 ) * st::pow( y, 12 ) * st::pow( z, 12 ) -
                1080 * st::pow( x, 12 ) * st::pow( y, 12 ) * st::pow( z, 8 ) - 3960 * st::pow( x, 12 ) * st::pow( y, 11 ) * st::pow( z, 9 ) -
                3168 * st::pow( x, 12 ) * st::pow( y, 10 ) * st::pow( z, 10 ) - 3960 * st::pow( x, 12 ) * st::pow( y, 9 ) * st::pow( z, 11 ) -
                1080 * st::pow( x, 12 ) * st::pow( y, 8 ) * st::pow( z, 12 ) - 3960 * st::pow( x, 11 ) * st::pow( y, 12 ) * st::pow( z, 9 ) -
                4752 * st::pow( x, 11 ) * st::pow( y, 11 ) * st::pow( z, 10 ) - 4752 * st::pow( x, 11 ) * st::pow( y, 10 ) * st::pow( z, 11 ) -
                3960 * st::pow( x, 11 ) * st::pow( y, 9 ) * st::pow( z, 12 ) - 3168 * st::pow( x, 10 ) * st::pow( y, 12 ) * st::pow( z, 10 ) -
                4752 * st::pow( x, 10 ) * st::pow( y, 11 ) * st::pow( z, 11 ) - 3168 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 12 ) -
                3960 * st::pow( x, 9 ) * st::pow( y, 12 ) * st::pow( z, 11 ) - 3960 * st::pow( x, 9 ) * st::pow( y, 11 ) * st::pow( z, 12 ) -
                1080 * st::pow( x, 8 ) * st::pow( y, 12 ) * st::pow( z, 12 ) + 660 * st::pow( x, 12 ) * st::pow( y, 12 ) * st::pow( z, 9 ) +
                792 * st::pow( x, 12 ) * st::pow( y, 11 ) * st::pow( z, 10 ) + 792 * st::pow( x, 12 ) * st::pow( y, 10 ) * st::pow( z, 11 ) +
                660 * st::pow( x, 12 ) * st::pow( y, 9 ) * st::pow( z, 12 ) + 792 * st::pow( x, 11 ) * st::pow( y, 12 ) * st::pow( z, 10 ) +
                792 * st::pow( x, 11 ) * st::pow( y, 10 ) * st::pow( z, 12 ) + 792 * st::pow( x, 10 ) * st::pow( y, 12 ) * st::pow( z, 11 ) +
                792 * st::pow( x, 10 ) * st::pow( y, 11 ) * st::pow( z, 12 ) + 660 * st::pow( x, 9 ) * st::pow( y, 12 ) * st::pow( z, 12 ) -
                132 * st::pow( x, 12 ) * st::pow( y, 12 ) * st::pow( z, 10 ) - 132 * st::pow( x, 12 ) * st::pow( y, 10 ) * st::pow( z, 12 ) -
                132 * st::pow( x, 10 ) * st::pow( y, 12 ) * st::pow( z, 12 ),
            0.0
        };
    }

    __DEVICE_TAG__ TensorType get_exact_solution( Scalar x, Scalar y, Scalar z ) const
    {
        Scalar g = x * ( 1 - x ) * y * ( 1 - y ) * z * ( 1 - z );

        return TensorType{
            g * ( -35 - 23 * x - 23 * y - 23 * z + 23 * st::pow( x, 2 ) - 11 * x * y - 11 * x * z + 23 * st::pow( y, 2 ) -
                  11 * y * z + 23 * st::pow( z, 2 ) + 11 * st::pow( x, 2 ) * y + 11 * st::pow( x, 2 ) * z + 11 * x * st::pow( y, 2 ) +
                  x * y * z + 11 * x * st::pow( z, 2 ) + 11 * st::pow( y, 2 ) * z + 11 * y * st::pow( z, 2 ) -
                  11 * st::pow( x, 2 ) * st::pow( y, 2 ) - st::pow( x, 2 ) * y * z - 11 * st::pow( x, 2 ) * st::pow( z, 2 ) -
                  x * st::pow( y, 2 ) * z - x * y * st::pow( z, 2 ) - 11 * st::pow( y, 2 ) * st::pow( z, 2 ) +
                  st::pow( x, 2 ) * st::pow( y, 2 ) * z + st::pow( x, 2 ) * y * st::pow( z, 2 ) + x * st::pow( y, 2 ) * st::pow( z, 2 ) -
                  2 * st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 2 ) - st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 2 ) -
                  st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 2 ) - st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 3 ) +
                  5 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 2 ) - st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 2 ) -
                  st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 3 ) + 5 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 2 ) -
                  st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 3 ) + 5 * st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 4 ) +
                  2 * st::pow( x, 5 ) * st::pow( y, 2 ) * st::pow( z, 2 ) + 5 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 2 ) +
                  5 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 3 ) + 5 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 2 ) -
                  st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 3 ) + 5 * st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 4 ) +
                  2 * st::pow( x, 2 ) * st::pow( y, 5 ) * st::pow( z, 2 ) + 5 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 3 ) +
                  5 * st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 4 ) + 2 * st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 5 ) -
                  10 * st::pow( x, 6 ) * st::pow( y, 2 ) * st::pow( z, 2 ) + 2 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 2 ) +
                  2 * st::pow( x, 5 ) * st::pow( y, 2 ) * st::pow( z, 3 ) - 25 * st::pow( x, 4 ) * st::pow( y, 4 ) * st::pow( z, 2 ) +
                  5 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 3 ) - 25 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 4 ) +
                  2 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 2 ) + 5 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 3 ) +
                  5 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 4 ) + 2 * st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 5 ) -
                  10 * st::pow( x, 2 ) * st::pow( y, 6 ) * st::pow( z, 2 ) + 2 * st::pow( x, 2 ) * st::pow( y, 5 ) * st::pow( z, 3 ) -
                  25 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 4 ) + 2 * st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 5 ) -
                  10 * st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 6 ) + 2 * st::pow( x, 7 ) * st::pow( y, 2 ) * st::pow( z, 2 ) -
                  10 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 2 ) - 10 * st::pow( x, 6 ) * st::pow( y, 2 ) * st::pow( z, 3 ) -
                  10 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 2 ) + 2 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 3 ) -
                  10 * st::pow( x, 5 ) * st::pow( y, 2 ) * st::pow( z, 4 ) - 10 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 2 ) -
                  25 * st::pow( x, 4 ) * st::pow( y, 4 ) * st::pow( z, 3 ) - 25 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 4 ) -
                  10 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 5 ) - 10 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 2 ) +
                  2 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 3 ) - 25 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 4 ) +
                  2 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 5 ) - 10 * st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 6 ) +
                  2 * st::pow( x, 2 ) * st::pow( y, 7 ) * st::pow( z, 2 ) - 10 * st::pow( x, 2 ) * st::pow( y, 6 ) * st::pow( z, 3 ) -
                  10 * st::pow( x, 2 ) * st::pow( y, 5 ) * st::pow( z, 4 ) - 10 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 5 ) -
                  10 * st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 6 ) + 2 * st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 7 ) +
                  7 * st::pow( x, 8 ) * st::pow( y, 2 ) * st::pow( z, 2 ) + 2 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 2 ) +
                  2 * st::pow( x, 7 ) * st::pow( y, 2 ) * st::pow( z, 3 ) + 50 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 2 ) -
                  10 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 3 ) + 50 * st::pow( x, 6 ) * st::pow( y, 2 ) * st::pow( z, 4 ) -
                  4 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 2 ) - 10 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 3 ) -
                  10 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 4 ) - 4 * st::pow( x, 5 ) * st::pow( y, 2 ) * st::pow( z, 5 ) +
                  50 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 2 ) - 10 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 3 ) +
                  125 * st::pow( x, 4 ) * st::pow( y, 4 ) * st::pow( z, 4 ) - 10 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 5 ) +
                  50 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 6 ) + 2 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 2 ) -
                  10 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 3 ) - 10 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 4 ) -
                  10 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 5 ) - 10 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 6 ) +
                  2 * st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 7 ) + 7 * st::pow( x, 2 ) * st::pow( y, 8 ) * st::pow( z, 2 ) +
                  2 * st::pow( x, 2 ) * st::pow( y, 7 ) * st::pow( z, 3 ) + 50 * st::pow( x, 2 ) * st::pow( y, 6 ) * st::pow( z, 4 ) -
                  4 * st::pow( x, 2 ) * st::pow( y, 5 ) * st::pow( z, 5 ) + 50 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 6 ) +
                  2 * st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 7 ) + 7 * st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 8 ) -
                  5 * st::pow( x, 9 ) * st::pow( y, 2 ) * st::pow( z, 2 ) + 7 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 2 ) +
                  7 * st::pow( x, 8 ) * st::pow( y, 2 ) * st::pow( z, 3 ) - 10 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 2 ) +
                  2 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 3 ) - 10 * st::pow( x, 7 ) * st::pow( y, 2 ) * st::pow( z, 4 ) +
                  20 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 2 ) + 50 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 3 ) +
                  50 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 4 ) + 20 * st::pow( x, 6 ) * st::pow( y, 2 ) * st::pow( z, 5 ) +
                  20 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 2 ) - 4 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 3 ) +
                  50 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 4 ) - 4 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 5 ) +
                  20 * st::pow( x, 5 ) * st::pow( y, 2 ) * st::pow( z, 6 ) - 10 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 2 ) +
                  50 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 3 ) + 50 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 4 ) +
                  50 * st::pow( x, 4 ) * st::pow( y, 4 ) * st::pow( z, 5 ) + 50 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 6 ) -
                  10 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 7 ) + 7 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 2 ) +
                  2 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 3 ) + 50 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 4 ) -
                  4 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 5 ) + 50 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 6 ) +
                  2 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 7 ) + 7 * st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 8 ) -
                  5 * st::pow( x, 2 ) * st::pow( y, 9 ) * st::pow( z, 2 ) + 7 * st::pow( x, 2 ) * st::pow( y, 8 ) * st::pow( z, 3 ) -
                  10 * st::pow( x, 2 ) * st::pow( y, 7 ) * st::pow( z, 4 ) + 20 * st::pow( x, 2 ) * st::pow( y, 6 ) * st::pow( z, 5 ) +
                  20 * st::pow( x, 2 ) * st::pow( y, 5 ) * st::pow( z, 6 ) - 10 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 7 ) +
                  7 * st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 8 ) - 5 * st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 9 ) +
                  st::pow( x, 10 ) * st::pow( y, 2 ) * st::pow( z, 2 ) - 5 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 2 ) -
                  5 * st::pow( x, 9 ) * st::pow( y, 2 ) * st::pow( z, 3 ) - 35 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 2 ) +
                  7 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 3 ) - 35 * st::pow( x, 8 ) * st::pow( y, 2 ) * st::pow( z, 4 ) -
                  4 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 2 ) - 10 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 3 ) -
                  10 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 4 ) - 4 * st::pow( x, 7 ) * st::pow( y, 2 ) * st::pow( z, 5 ) -
                  100 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 2 ) + 20 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 3 ) -
                  250 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 4 ) + 20 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 5 ) -
                  100 * st::pow( x, 6 ) * st::pow( y, 2 ) * st::pow( z, 6 ) - 4 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 2 ) +
                  20 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 3 ) + 20 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 4 ) +
                  20 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 5 ) + 20 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 6 ) -
                  4 * st::pow( x, 5 ) * st::pow( y, 2 ) * st::pow( z, 7 ) - 35 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 2 ) -
                  10 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 3 ) - 250 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 4 ) +
                  20 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 5 ) - 250 * st::pow( x, 4 ) * st::pow( y, 4 ) * st::pow( z, 6 ) -
                  10 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 7 ) - 35 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 8 ) -
                  5 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 2 ) + 7 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 3 ) -
                  10 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 4 ) + 20 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 5 ) +
                  20 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 6 ) - 10 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 7 ) +
                  7 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 8 ) - 5 * st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 9 ) +
                  st::pow( x, 2 ) * st::pow( y, 10 ) * st::pow( z, 2 ) - 5 * st::pow( x, 2 ) * st::pow( y, 9 ) * st::pow( z, 3 ) -
                  35 * st::pow( x, 2 ) * st::pow( y, 8 ) * st::pow( z, 4 ) - 4 * st::pow( x, 2 ) * st::pow( y, 7 ) * st::pow( z, 5 ) -
                  100 * st::pow( x, 2 ) * st::pow( y, 6 ) * st::pow( z, 6 ) - 4 * st::pow( x, 2 ) * st::pow( y, 5 ) * st::pow( z, 7 ) -
                  35 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 8 ) - 5 * st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 9 ) +
                  st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 10 ) + st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 2 ) +
                  st::pow( x, 10 ) * st::pow( y, 2 ) * st::pow( z, 3 ) + 25 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 2 ) -
                  5 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 3 ) + 25 * st::pow( x, 9 ) * st::pow( y, 2 ) * st::pow( z, 4 ) -
                  14 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 2 ) - 35 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 3 ) -
                  35 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 4 ) - 14 * st::pow( x, 8 ) * st::pow( y, 2 ) * st::pow( z, 5 ) +
                  20 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 2 ) - 4 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 3 ) +
                  50 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 4 ) - 4 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 5 ) +
                  20 * st::pow( x, 7 ) * st::pow( y, 2 ) * st::pow( z, 6 ) + 20 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 2 ) -
                  100 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 3 ) - 100 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 4 ) -
                  100 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 5 ) - 100 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 6 ) +
                  20 * st::pow( x, 6 ) * st::pow( y, 2 ) * st::pow( z, 7 ) - 14 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 2 ) -
                  4 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 3 ) - 100 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 4 ) +
                  8 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 5 ) - 100 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 6 ) -
                  4 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 7 ) - 14 * st::pow( x, 5 ) * st::pow( y, 2 ) * st::pow( z, 8 ) +
                  25 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 2 ) - 35 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 3 ) +
                  50 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 4 ) - 100 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 5 ) -
                  100 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 6 ) + 50 * st::pow( x, 4 ) * st::pow( y, 4 ) * st::pow( z, 7 ) -
                  35 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 8 ) + 25 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 9 ) +
                  st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 2 ) - 5 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 3 ) -
                  35 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 4 ) - 4 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 5 ) -
                  100 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 6 ) - 4 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 7 ) -
                  35 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 8 ) - 5 * st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 9 ) +
                  st::pow( x, 3 ) * st::pow( y, 2 ) * st::pow( z, 10 ) + st::pow( x, 2 ) * st::pow( y, 10 ) * st::pow( z, 3 ) +
                  25 * st::pow( x, 2 ) * st::pow( y, 9 ) * st::pow( z, 4 ) - 14 * st::pow( x, 2 ) * st::pow( y, 8 ) * st::pow( z, 5 ) +
                  20 * st::pow( x, 2 ) * st::pow( y, 7 ) * st::pow( z, 6 ) + 20 * st::pow( x, 2 ) * st::pow( y, 6 ) * st::pow( z, 7 ) -
                  14 * st::pow( x, 2 ) * st::pow( y, 5 ) * st::pow( z, 8 ) + 25 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 9 ) +
                  st::pow( x, 2 ) * st::pow( y, 3 ) * st::pow( z, 10 ) - 5 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 2 ) +
                  st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 3 ) - 5 * st::pow( x, 10 ) * st::pow( y, 2 ) * st::pow( z, 4 ) +
                  10 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 2 ) + 25 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 3 ) +
                  25 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 4 ) + 10 * st::pow( x, 9 ) * st::pow( y, 2 ) * st::pow( z, 5 ) +
                  70 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 2 ) - 14 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 3 ) +
                  175 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 4 ) - 14 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 5 ) +
                  70 * st::pow( x, 8 ) * st::pow( y, 2 ) * st::pow( z, 6 ) - 4 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 2 ) +
                  20 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 3 ) + 20 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 4 ) +
                  20 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 5 ) + 20 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 6 ) -
                  4 * st::pow( x, 7 ) * st::pow( y, 2 ) * st::pow( z, 7 ) + 70 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 2 ) +
                  20 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 3 ) + 500 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 4 ) -
                  40 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 5 ) + 500 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 6 ) +
                  20 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 7 ) + 70 * st::pow( x, 6 ) * st::pow( y, 2 ) * st::pow( z, 8 ) +
                  10 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 2 ) - 14 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 3 ) +
                  20 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 4 ) - 40 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 5 ) -
                  40 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 6 ) + 20 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 7 ) -
                  14 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 8 ) + 10 * st::pow( x, 5 ) * st::pow( y, 2 ) * st::pow( z, 9 ) -
                  5 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 2 ) + 25 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 3 ) +
                  175 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 4 ) + 20 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 5 ) +
                  500 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 6 ) + 20 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 7 ) +
                  175 * st::pow( x, 4 ) * st::pow( y, 4 ) * st::pow( z, 8 ) + 25 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 9 ) -
                  5 * st::pow( x, 4 ) * st::pow( y, 2 ) * st::pow( z, 10 ) + st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 3 ) +
                  25 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 4 ) - 14 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 5 ) +
                  20 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 6 ) + 20 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 7 ) -
                  14 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 8 ) + 25 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 9 ) +
                  st::pow( x, 3 ) * st::pow( y, 3 ) * st::pow( z, 10 ) - 5 * st::pow( x, 2 ) * st::pow( y, 10 ) * st::pow( z, 4 ) +
                  10 * st::pow( x, 2 ) * st::pow( y, 9 ) * st::pow( z, 5 ) + 70 * st::pow( x, 2 ) * st::pow( y, 8 ) * st::pow( z, 6 ) -
                  4 * st::pow( x, 2 ) * st::pow( y, 7 ) * st::pow( z, 7 ) + 70 * st::pow( x, 2 ) * st::pow( y, 6 ) * st::pow( z, 8 ) +
                  10 * st::pow( x, 2 ) * st::pow( y, 5 ) * st::pow( z, 9 ) - 5 * st::pow( x, 2 ) * st::pow( y, 4 ) * st::pow( z, 10 ) -
                  2 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 2 ) - 5 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 3 ) -
                  5 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 4 ) - 2 * st::pow( x, 10 ) * st::pow( y, 2 ) * st::pow( z, 5 ) -
                  50 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 2 ) + 10 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 3 ) -
                  125 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 4 ) + 10 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 5 ) -
                  50 * st::pow( x, 9 ) * st::pow( y, 2 ) * st::pow( z, 6 ) - 14 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 2 ) +
                  70 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 3 ) + 70 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 4 ) +
                  70 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 5 ) + 70 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 6 ) -
                  14 * st::pow( x, 8 ) * st::pow( y, 2 ) * st::pow( z, 7 ) - 14 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 2 ) -
                  4 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 3 ) - 100 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 4 ) +
                  8 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 5 ) - 100 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 6 ) -
                  4 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 7 ) - 14 * st::pow( x, 7 ) * st::pow( y, 2 ) * st::pow( z, 8 ) -
                  50 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 2 ) + 70 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 3 ) -
                  100 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 4 ) + 200 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 5 ) +
                  200 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 6 ) - 100 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 7 ) +
                  70 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 8 ) - 50 * st::pow( x, 6 ) * st::pow( y, 2 ) * st::pow( z, 9 ) -
                  2 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 2 ) + 10 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 3 ) +
                  70 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 4 ) + 8 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 5 ) +
                  200 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 6 ) + 8 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 7 ) +
                  70 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 8 ) + 10 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 9 ) -
                  2 * st::pow( x, 5 ) * st::pow( y, 2 ) * st::pow( z, 10 ) - 5 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 3 ) -
                  125 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 4 ) + 70 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 5 ) -
                  100 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 6 ) - 100 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 7 ) +
                  70 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 8 ) - 125 * st::pow( x, 4 ) * st::pow( y, 4 ) * st::pow( z, 9 ) -
                  5 * st::pow( x, 4 ) * st::pow( y, 3 ) * st::pow( z, 10 ) - 5 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 4 ) +
                  10 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 5 ) + 70 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 6 ) -
                  4 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 7 ) + 70 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 8 ) +
                  10 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 9 ) - 5 * st::pow( x, 3 ) * st::pow( y, 4 ) * st::pow( z, 10 ) -
                  2 * st::pow( x, 2 ) * st::pow( y, 10 ) * st::pow( z, 5 ) - 50 * st::pow( x, 2 ) * st::pow( y, 9 ) * st::pow( z, 6 ) -
                  14 * st::pow( x, 2 ) * st::pow( y, 8 ) * st::pow( z, 7 ) - 14 * st::pow( x, 2 ) * st::pow( y, 7 ) * st::pow( z, 8 ) -
                  50 * st::pow( x, 2 ) * st::pow( y, 6 ) * st::pow( z, 9 ) - 2 * st::pow( x, 2 ) * st::pow( y, 5 ) * st::pow( z, 10 ) +
                  10 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 2 ) - 2 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 3 ) +
                  25 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 4 ) - 2 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 5 ) +
                  10 * st::pow( x, 10 ) * st::pow( y, 2 ) * st::pow( z, 6 ) + 10 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 2 ) -
                  50 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 3 ) - 50 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 4 ) -
                  50 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 5 ) - 50 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 6 ) +
                  10 * st::pow( x, 9 ) * st::pow( y, 2 ) * st::pow( z, 7 ) - 49 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 2 ) -
                  14 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 3 ) - 350 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 4 ) +
                  28 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 5 ) - 350 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 6 ) -
                  14 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 7 ) - 49 * st::pow( x, 8 ) * st::pow( y, 2 ) * st::pow( z, 8 ) +
                  10 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 2 ) - 14 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 3 ) +
                  20 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 4 ) - 40 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 5 ) -
                  40 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 6 ) + 20 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 7 ) -
                  14 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 8 ) + 10 * st::pow( x, 7 ) * st::pow( y, 2 ) * st::pow( z, 9 ) +
                  10 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 2 ) - 50 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 3 ) -
                  350 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 4 ) - 40 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 5 ) -
                  1000 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 6 ) - 40 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 7 ) -
                  350 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 8 ) - 50 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 9 ) +
                  10 * st::pow( x, 6 ) * st::pow( y, 2 ) * st::pow( z, 10 ) - 2 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 3 ) -
                  50 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 4 ) + 28 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 5 ) -
                  40 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 6 ) - 40 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 7 ) +
                  28 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 8 ) - 50 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 9 ) -
                  2 * st::pow( x, 5 ) * st::pow( y, 3 ) * st::pow( z, 10 ) + 25 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 4 ) -
                  50 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 5 ) - 350 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 6 ) +
                  20 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 7 ) - 350 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 8 ) -
                  50 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 9 ) + 25 * st::pow( x, 4 ) * st::pow( y, 4 ) * st::pow( z, 10 ) -
                  2 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 5 ) - 50 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 6 ) -
                  14 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 7 ) - 14 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 8 ) -
                  50 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 9 ) - 2 * st::pow( x, 3 ) * st::pow( y, 5 ) * st::pow( z, 10 ) +
                  10 * st::pow( x, 2 ) * st::pow( y, 10 ) * st::pow( z, 6 ) + 10 * st::pow( x, 2 ) * st::pow( y, 9 ) * st::pow( z, 7 ) -
                  49 * st::pow( x, 2 ) * st::pow( y, 8 ) * st::pow( z, 8 ) + 10 * st::pow( x, 2 ) * st::pow( y, 7 ) * st::pow( z, 9 ) +
                  10 * st::pow( x, 2 ) * st::pow( y, 6 ) * st::pow( z, 10 ) - 2 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 2 ) +
                  10 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 3 ) + 10 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 4 ) +
                  10 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 5 ) + 10 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 6 ) -
                  2 * st::pow( x, 10 ) * st::pow( y, 2 ) * st::pow( z, 7 ) + 35 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 2 ) +
                  10 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 3 ) + 250 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 4 ) -
                  20 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 5 ) + 250 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 6 ) +
                  10 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 7 ) + 35 * st::pow( x, 9 ) * st::pow( y, 2 ) * st::pow( z, 8 ) +
                  35 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 2 ) - 49 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 3 ) +
                  70 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 4 ) - 140 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 5 ) -
                  140 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 6 ) + 70 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 7 ) -
                  49 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 8 ) + 35 * st::pow( x, 8 ) * st::pow( y, 2 ) * st::pow( z, 9 ) -
                  2 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 2 ) + 10 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 3 ) +
                  70 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 4 ) + 8 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 5 ) +
                  200 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 6 ) + 8 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 7 ) +
                  70 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 8 ) + 10 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 9 ) -
                  2 * st::pow( x, 7 ) * st::pow( y, 2 ) * st::pow( z, 10 ) + 10 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 3 ) +
                  250 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 4 ) - 140 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 5 ) +
                  200 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 6 ) + 200 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 7 ) -
                  140 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 8 ) + 250 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 9 ) +
                  10 * st::pow( x, 6 ) * st::pow( y, 3 ) * st::pow( z, 10 ) + 10 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 4 ) -
                  20 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 5 ) - 140 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 6 ) +
                  8 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 7 ) - 140 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 8 ) -
                  20 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 9 ) + 10 * st::pow( x, 5 ) * st::pow( y, 4 ) * st::pow( z, 10 ) +
                  10 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 5 ) + 250 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 6 ) +
                  70 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 7 ) + 70 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 8 ) +
                  250 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 9 ) + 10 * st::pow( x, 4 ) * st::pow( y, 5 ) * st::pow( z, 10 ) +
                  10 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 6 ) + 10 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 7 ) -
                  49 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 8 ) + 10 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 9 ) +
                  10 * st::pow( x, 3 ) * st::pow( y, 6 ) * st::pow( z, 10 ) - 2 * st::pow( x, 2 ) * st::pow( y, 10 ) * st::pow( z, 7 ) +
                  35 * st::pow( x, 2 ) * st::pow( y, 9 ) * st::pow( z, 8 ) + 35 * st::pow( x, 2 ) * st::pow( y, 8 ) * st::pow( z, 9 ) -
                  2 * st::pow( x, 2 ) * st::pow( y, 7 ) * st::pow( z, 10 ) - 7 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 2 ) -
                  2 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 3 ) - 50 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 4 ) +
                  4 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 5 ) - 50 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 6 ) -
                  2 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 7 ) - 7 * st::pow( x, 10 ) * st::pow( y, 2 ) * st::pow( z, 8 ) -
                  25 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 2 ) + 35 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 3 ) -
                  50 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 4 ) + 100 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 5 ) +
                  100 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 6 ) - 50 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 7 ) +
                  35 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 8 ) - 25 * st::pow( x, 9 ) * st::pow( y, 2 ) * st::pow( z, 9 ) -
                  7 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 2 ) + 35 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 3 ) +
                  245 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 4 ) + 28 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 5 ) +
                  700 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 6 ) + 28 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 7 ) +
                  245 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 8 ) + 35 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 9 ) -
                  7 * st::pow( x, 8 ) * st::pow( y, 2 ) * st::pow( z, 10 ) - 2 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 3 ) -
                  50 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 4 ) + 28 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 5 ) -
                  40 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 6 ) - 40 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 7 ) +
                  28 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 8 ) - 50 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 9 ) -
                  2 * st::pow( x, 7 ) * st::pow( y, 3 ) * st::pow( z, 10 ) - 50 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 4 ) +
                  100 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 5 ) + 700 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 6 ) -
                  40 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 7 ) + 700 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 8 ) +
                  100 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 9 ) - 50 * st::pow( x, 6 ) * st::pow( y, 4 ) * st::pow( z, 10 ) +
                  4 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 5 ) + 100 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 6 ) +
                  28 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 7 ) + 28 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 8 ) +
                  100 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 9 ) + 4 * st::pow( x, 5 ) * st::pow( y, 5 ) * st::pow( z, 10 ) -
                  50 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 6 ) - 50 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 7 ) +
                  245 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 8 ) - 50 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 9 ) -
                  50 * st::pow( x, 4 ) * st::pow( y, 6 ) * st::pow( z, 10 ) - 2 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 7 ) +
                  35 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 8 ) + 35 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 9 ) -
                  2 * st::pow( x, 3 ) * st::pow( y, 7 ) * st::pow( z, 10 ) - 7 * st::pow( x, 2 ) * st::pow( y, 10 ) * st::pow( z, 8 ) -
                  25 * st::pow( x, 2 ) * st::pow( y, 9 ) * st::pow( z, 9 ) - 7 * st::pow( x, 2 ) * st::pow( y, 8 ) * st::pow( z, 10 ) +
                  5 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 2 ) - 7 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 3 ) +
                  10 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 4 ) - 20 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 5 ) -
                  20 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 6 ) + 10 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 7 ) -
                  7 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 8 ) + 5 * st::pow( x, 10 ) * st::pow( y, 2 ) * st::pow( z, 9 ) +
                  5 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 2 ) - 25 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 3 ) -
                  175 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 4 ) - 20 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 5 ) -
                  500 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 6 ) - 20 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 7 ) -
                  175 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 8 ) - 25 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 9 ) +
                  5 * st::pow( x, 9 ) * st::pow( y, 2 ) * st::pow( z, 10 ) - 7 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 3 ) -
                  175 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 4 ) + 98 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 5 ) -
                  140 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 6 ) - 140 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 7 ) +
                  98 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 8 ) - 175 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 9 ) -
                  7 * st::pow( x, 8 ) * st::pow( y, 3 ) * st::pow( z, 10 ) + 10 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 4 ) -
                  20 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 5 ) - 140 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 6 ) +
                  8 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 7 ) - 140 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 8 ) -
                  20 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 9 ) + 10 * st::pow( x, 7 ) * st::pow( y, 4 ) * st::pow( z, 10 ) -
                  20 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 5 ) - 500 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 6 ) -
                  140 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 7 ) - 140 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 8 ) -
                  500 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 9 ) - 20 * st::pow( x, 6 ) * st::pow( y, 5 ) * st::pow( z, 10 ) -
                  20 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 6 ) - 20 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 7 ) +
                  98 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 8 ) - 20 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 9 ) -
                  20 * st::pow( x, 5 ) * st::pow( y, 6 ) * st::pow( z, 10 ) + 10 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 7 ) -
                  175 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 8 ) - 175 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 9 ) +
                  10 * st::pow( x, 4 ) * st::pow( y, 7 ) * st::pow( z, 10 ) - 7 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 8 ) -
                  25 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 9 ) - 7 * st::pow( x, 3 ) * st::pow( y, 8 ) * st::pow( z, 10 ) +
                  5 * st::pow( x, 2 ) * st::pow( y, 10 ) * st::pow( z, 9 ) + 5 * st::pow( x, 2 ) * st::pow( y, 9 ) * st::pow( z, 10 ) -
                  st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 2 ) + 5 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 3 ) +
                  35 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 4 ) + 4 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 5 ) +
                  100 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 6 ) + 4 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 7 ) +
                  35 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 8 ) + 5 * st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 9 ) -
                  st::pow( x, 10 ) * st::pow( y, 2 ) * st::pow( z, 10 ) + 5 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 3 ) +
                  125 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 4 ) - 70 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 5 ) +
                  100 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 6 ) + 100 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 7 ) -
                  70 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 8 ) + 125 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 9 ) +
                  5 * st::pow( x, 9 ) * st::pow( y, 3 ) * st::pow( z, 10 ) + 35 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 4 ) -
                  70 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 5 ) - 490 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 6 ) +
                  28 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 7 ) - 490 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 8 ) -
                  70 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 9 ) + 35 * st::pow( x, 8 ) * st::pow( y, 4 ) * st::pow( z, 10 ) +
                  4 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 5 ) + 100 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 6 ) +
                  28 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 7 ) + 28 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 8 ) +
                  100 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 9 ) + 4 * st::pow( x, 7 ) * st::pow( y, 5 ) * st::pow( z, 10 ) +
                  100 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 6 ) + 100 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 7 ) -
                  490 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 8 ) + 100 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 9 ) +
                  100 * st::pow( x, 6 ) * st::pow( y, 6 ) * st::pow( z, 10 ) + 4 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 7 ) -
                  70 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 8 ) - 70 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 9 ) +
                  4 * st::pow( x, 5 ) * st::pow( y, 7 ) * st::pow( z, 10 ) + 35 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 8 ) +
                  125 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 9 ) + 35 * st::pow( x, 4 ) * st::pow( y, 8 ) * st::pow( z, 10 ) +
                  5 * st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 9 ) + 5 * st::pow( x, 3 ) * st::pow( y, 9 ) * st::pow( z, 10 ) -
                  st::pow( x, 2 ) * st::pow( y, 10 ) * st::pow( z, 10 ) - st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 3 ) -
                  25 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 4 ) + 14 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 5 ) -
                  20 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 6 ) - 20 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 7 ) +
                  14 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 8 ) - 25 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 9 ) -
                  st::pow( x, 10 ) * st::pow( y, 3 ) * st::pow( z, 10 ) - 25 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 4 ) +
                  50 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 5 ) + 350 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 6 ) -
                  20 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 7 ) + 350 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 8 ) +
                  50 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 9 ) - 25 * st::pow( x, 9 ) * st::pow( y, 4 ) * st::pow( z, 10 ) +
                  14 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 5 ) + 350 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 6 ) +
                  98 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 7 ) + 98 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 8 ) +
                  350 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 9 ) + 14 * st::pow( x, 8 ) * st::pow( y, 5 ) * st::pow( z, 10 ) -
                  20 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 6 ) - 20 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 7 ) +
                  98 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 8 ) - 20 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 9 ) -
                  20 * st::pow( x, 7 ) * st::pow( y, 6 ) * st::pow( z, 10 ) - 20 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 7 ) +
                  350 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 8 ) + 350 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 9 ) -
                  20 * st::pow( x, 6 ) * st::pow( y, 7 ) * st::pow( z, 10 ) + 14 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 8 ) +
                  50 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 9 ) + 14 * st::pow( x, 5 ) * st::pow( y, 8 ) * st::pow( z, 10 ) -
                  25 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 9 ) - 25 * st::pow( x, 4 ) * st::pow( y, 9 ) * st::pow( z, 10 ) -
                  st::pow( x, 3 ) * st::pow( y, 10 ) * st::pow( z, 10 ) + 5 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 4 ) -
                  10 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 5 ) - 70 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 6 ) +
                  4 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 7 ) - 70 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 8 ) -
                  10 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 9 ) + 5 * st::pow( x, 10 ) * st::pow( y, 4 ) * st::pow( z, 10 ) -
                  10 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 5 ) - 250 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 6 ) -
                  70 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 7 ) - 70 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 8 ) -
                  250 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 9 ) - 10 * st::pow( x, 9 ) * st::pow( y, 5 ) * st::pow( z, 10 ) -
                  70 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 6 ) - 70 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 7 ) +
                  343 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 8 ) - 70 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 9 ) -
                  70 * st::pow( x, 8 ) * st::pow( y, 6 ) * st::pow( z, 10 ) + 4 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 7 ) -
                  70 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 8 ) - 70 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 9 ) +
                  4 * st::pow( x, 7 ) * st::pow( y, 7 ) * st::pow( z, 10 ) - 70 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 8 ) -
                  250 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 9 ) - 70 * st::pow( x, 6 ) * st::pow( y, 8 ) * st::pow( z, 10 ) -
                  10 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 9 ) - 10 * st::pow( x, 5 ) * st::pow( y, 9 ) * st::pow( z, 10 ) +
                  5 * st::pow( x, 4 ) * st::pow( y, 10 ) * st::pow( z, 10 ) + 2 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 5 ) +
                  50 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 6 ) + 14 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 7 ) +
                  14 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 8 ) + 50 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 9 ) +
                  2 * st::pow( x, 10 ) * st::pow( y, 5 ) * st::pow( z, 10 ) + 50 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 6 ) +
                  50 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 7 ) - 245 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 8 ) +
                  50 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 9 ) + 50 * st::pow( x, 9 ) * st::pow( y, 6 ) * st::pow( z, 10 ) +
                  14 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 7 ) - 245 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 8 ) -
                  245 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 9 ) + 14 * st::pow( x, 8 ) * st::pow( y, 7 ) * st::pow( z, 10 ) +
                  14 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 8 ) + 50 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 9 ) +
                  14 * st::pow( x, 7 ) * st::pow( y, 8 ) * st::pow( z, 10 ) + 50 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 9 ) +
                  50 * st::pow( x, 6 ) * st::pow( y, 9 ) * st::pow( z, 10 ) + 2 * st::pow( x, 5 ) * st::pow( y, 10 ) * st::pow( z, 10 ) -
                  10 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 6 ) - 10 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 7 ) +
                  49 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 8 ) - 10 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 9 ) -
                  10 * st::pow( x, 10 ) * st::pow( y, 6 ) * st::pow( z, 10 ) - 10 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 7 ) +
                  175 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 8 ) + 175 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 9 ) -
                  10 * st::pow( x, 9 ) * st::pow( y, 7 ) * st::pow( z, 10 ) + 49 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 8 ) +
                  175 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 9 ) + 49 * st::pow( x, 8 ) * st::pow( y, 8 ) * st::pow( z, 10 ) -
                  10 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 9 ) - 10 * st::pow( x, 7 ) * st::pow( y, 9 ) * st::pow( z, 10 ) -
                  10 * st::pow( x, 6 ) * st::pow( y, 10 ) * st::pow( z, 10 ) + 2 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 7 ) -
                  35 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 8 ) - 35 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 9 ) +
                  2 * st::pow( x, 10 ) * st::pow( y, 7 ) * st::pow( z, 10 ) - 35 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 8 ) -
                  125 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 9 ) - 35 * st::pow( x, 9 ) * st::pow( y, 8 ) * st::pow( z, 10 ) -
                  35 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 9 ) - 35 * st::pow( x, 8 ) * st::pow( y, 9 ) * st::pow( z, 10 ) +
                  2 * st::pow( x, 7 ) * st::pow( y, 10 ) * st::pow( z, 10 ) + 7 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 8 ) +
                  25 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 9 ) + 7 * st::pow( x, 10 ) * st::pow( y, 8 ) * st::pow( z, 10 ) +
                  25 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 9 ) + 25 * st::pow( x, 9 ) * st::pow( y, 9 ) * st::pow( z, 10 ) +
                  7 * st::pow( x, 8 ) * st::pow( y, 10 ) * st::pow( z, 10 ) - 5 * st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 9 ) -
                  5 * st::pow( x, 10 ) * st::pow( y, 9 ) * st::pow( z, 10 ) - 5 * st::pow( x, 9 ) * st::pow( y, 10 ) * st::pow( z, 10 ) +
                  st::pow( x, 10 ) * st::pow( y, 10 ) * st::pow( z, 10 ) ),

            g * ( -1 - x - y - z + st::pow( x, 2 ) - x * y - x * z + st::pow( y, 2 ) - y * z + st::pow( z, 2 ) + st::pow( x, 2 ) * y +
                  st::pow( x, 2 ) * z + x * st::pow( y, 2 ) - x * y * z + x * st::pow( z, 2 ) + st::pow( y, 2 ) * z + y * st::pow( z, 2 ) -
                  st::pow( x, 2 ) * st::pow( y, 2 ) + st::pow( x, 2 ) * y * z - st::pow( x, 2 ) * st::pow( z, 2 ) + x * st::pow( y, 2 ) * z +
                  x * y * st::pow( z, 2 ) - st::pow( y, 2 ) * st::pow( z, 2 ) - st::pow( x, 2 ) * st::pow( y, 2 ) * z -
                  st::pow( x, 2 ) * y * st::pow( z, 2 ) - x * st::pow( y, 2 ) * st::pow( z, 2 ) +
                  st::pow( x, 2 ) * st::pow( y, 2 ) * st::pow( z, 2 ) )
        };
    }
};

} // namespace tests

#endif
