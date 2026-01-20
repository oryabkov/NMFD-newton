#ifndef __BIHARMONIC_PROBLEM_H__
#define __BIHARMONIC_PROBLEM_H__

#include <cmath>

#define PI 2 * std::acos( 0.0 )

namespace tests
{

template <class Scalar, class TensorType>
class zero_rhs
{
public:
    TensorType operator()( Scalar x, Scalar y, Scalar z ) const
    {
        return TensorType{ 0.0, 0.0 };
    }
};

template <class Scalar, class TensorType>
class trig_rhs
{
public:
    TensorType operator()( Scalar x, Scalar y, Scalar z ) const
    {
        Scalar g = std::sin( PI * x ) * std::sin( PI * y ) * std::sin( PI * z );

        return TensorType{
            3 * PI * PI * g,
            g,
        };
    }

    TensorType get_exact_solution( Scalar x, Scalar y, Scalar z ) const
    {
        Scalar g = std::sin( PI * x ) * std::sin( PI * y ) * std::sin( PI * z );

        return TensorType{ -9 * std::pow( PI, 4 ) * g, 0.0 };
    }
};

} // namespace tests

#endif
