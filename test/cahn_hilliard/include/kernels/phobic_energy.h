#ifndef __PHOBIC_ENERGY_RHS_H__
#define __PHOBIC_ENERGY_RHS_H__

#include <cmath>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>


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
class logarithmic_potential
{
    using st = scfd::utils::scalar_traits<Scalar>;

public:
    logarithmic_potential(Scalar omega = 3.0): omega_(omega)
    {
    }

    __DEVICE_TAG__ Scalar operator()( Scalar phi ) const
    {
        return std::log((Scalar( 1.0 ) + phi) / (Scalar( 1.0 ) - phi)) - omega_ * phi;
    }

    __DEVICE_TAG__ Scalar get_derivative( Scalar phi ) const
    {
        return Scalar( 2.0 ) / (Scalar( 1.0 ) - phi * phi) - omega_;
    }

private:
    Scalar omega_;
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
