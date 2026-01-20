#ifndef __PHOBIC_ENERGY_RHS_H__
#define __PHOBIC_ENERGY_RHS_H__

#include <cmath>

namespace tests
{

template <class Scalar>
class double_well_potential
{
public:
    Scalar operator()( Scalar phi ) const
    {
        return ( phi * phi - 1 ) * phi;
    }

    Scalar get_derivative( Scalar phi ) const
    {
        return 3 * phi * phi - 1;
    }
};

template <class Scalar>
class zero_potential
{
public:
    Scalar operator()( Scalar phi ) const
    {
        return Scalar( 0.0 );
    }

    Scalar get_derivative( Scalar phi ) const
    {
        return Scalar( 0.0 );
    }
};

} // namespace tests

#endif
