#ifndef __BIHARMONIC_PROBLEM_H__
#define __BIHARMONIC_PROBLEM_H__

#include <cmath>
#include <scfd/utils/device_tag.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define PI M_PI

namespace tests
{

template <class Scalar, class TensorType>
class zero_rhs
{
public:
    __DEVICE_TAG__ TensorType operator()( Scalar x, Scalar y, Scalar z ) const
    {
        return TensorType{ 0.0, 0.0 };
    }
};

template <class Scalar, class TensorType>
class trig_rhs
{
public:
    __DEVICE_TAG__ TensorType operator()( Scalar x, Scalar y, Scalar z ) const
    {
        Scalar g = std::sin( PI * x ) * std::sin( PI * y ) * std::sin( PI * z );

        return TensorType{
            3 * PI * PI * g,
            g,
        };
    }

    __DEVICE_TAG__ TensorType get_exact_solution( Scalar x, Scalar y, Scalar z ) const
    {
        Scalar g = std::sin( PI * x ) * std::sin( PI * y ) * std::sin( PI * z );

        return TensorType{ -9 * std::pow( PI, 4 ) * g, 0.0 };
    }
};

} // namespace tests

#endif
