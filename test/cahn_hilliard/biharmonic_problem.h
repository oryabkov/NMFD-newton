#ifndef __BIHARMONIC_PROBLEM_H__
#define __BIHARMONIC_PROBLEM_H__

#include <cmath>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>

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
    using st = scfd::utils::scalar_traits<Scalar>;

public:
    __DEVICE_TAG__ TensorType operator()( Scalar x, Scalar y, Scalar z ) const
    {
        Scalar k = 2 * st::pi(); // periodic on the unit cube (g = 0 and g' matched at walls)
        Scalar g = st::sin( k * x ) * st::sin( k * y ) * st::sin( k * z );

        return TensorType{
            3 * k * k * g,
            g,
        };
    }

    __DEVICE_TAG__ TensorType get_exact_solution( Scalar x, Scalar y, Scalar z ) const
    {
        Scalar k = 2 * st::pi();
        Scalar g = st::sin( k * x ) * st::sin( k * y ) * st::sin( k * z );

        return TensorType{ -9 * st::pow( k, 4 ) * g, 0.0 };
    }
};

} // namespace tests

#endif
