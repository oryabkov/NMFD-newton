#ifndef __PHOBIC_ENERGY_RHS_H__
#define __PHOBIC_ENERGY_RHS_H__

#include <cmath>
#include <scfd/utils/device_tag.h>


namespace tests
{

template <class Scalar>
class double_well_potential
{
public:
    __DEVICE_TAG__ Scalar operator()( Scalar phi ) const
    {
        return ( phi * phi - 1 ) * phi;
    }

    __DEVICE_TAG__ Scalar get_derivative( Scalar phi ) const
    {
        return 3 * phi * phi - 1;
    }
};

template <class Scalar>
class zero_potential
{
public:
    __DEVICE_TAG__ Scalar operator()( Scalar phi ) const
    {
        return Scalar( 0.0 );
    }

    __DEVICE_TAG__ Scalar get_derivative( Scalar phi ) const
    {
        return Scalar( 0.0 );
    }
};

} // namespace tests

#endif
