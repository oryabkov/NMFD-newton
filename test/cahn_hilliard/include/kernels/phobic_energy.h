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
    __DEVICE_TAG__ Scalar operator()( Scalar phi_impl, Scalar phi_expl ) const
    {
        return phi_impl * phi_impl * phi_impl - phi_expl;
    }

    __DEVICE_TAG__ Scalar get_derivative( Scalar phi ) const
    {
        return 3 * phi * phi - 1;
    }

    __DEVICE_TAG__ Scalar get_energy( Scalar phi ) const
    {
        return Scalar( 0.25 ) * ( phi * phi - 1 ) * ( phi * phi - 1 );
    }

    Scalar get_phi_eq() const
    {
        return Scalar( 1 );
    }

    Scalar get_curvature() const
    {
        return Scalar( 2 );
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

    __DEVICE_TAG__ Scalar operator()( Scalar phi_impl, Scalar phi_expl ) const
    {
        return std::log((Scalar( 1.0 ) + phi_impl) / (Scalar( 1.0 ) - phi_impl)) - omega_ * phi_expl;
    }

    __DEVICE_TAG__ Scalar get_derivative( Scalar phi ) const
    {
        return Scalar( 2.0 ) / (Scalar( 1.0 ) - phi * phi) - omega_;
    }

    __DEVICE_TAG__ Scalar get_energy( Scalar phi ) const
    {
        return ( Scalar( 1.0 ) + phi ) * std::log( Scalar( 1.0 ) + phi )
             + ( Scalar( 1.0 ) - phi ) * std::log( Scalar( 1.0 ) - phi )
             - ( omega_ / Scalar( 2.0 ) ) * phi * phi;
    }

    Scalar get_phi_eq() const
    {
        if ( omega_ <= Scalar( 2.0 ) )
        {
            return Scalar( 0.0 );
        }
        Scalar lo = Scalar( 1e-12 ), hi = Scalar( 1.0 ) - Scalar( 1e-15 );
        for ( int iter = 0; iter < 200; ++iter )
        {
            Scalar mid = Scalar( 0.5 ) * ( lo + hi );
            if ( std::log( ( Scalar( 1.0 ) + mid ) / ( Scalar( 1.0 ) - mid ) ) - omega_ * mid < Scalar( 0.0 ) )
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        return Scalar( 0.5 ) * ( lo + hi );
    }

    Scalar get_curvature() const
    {
        return get_derivative( get_phi_eq() );
    }

    Scalar get_omega() const
    {
        return omega_;
    }

private:
    Scalar omega_;
};

template <class Scalar>
class zero_potential
{
public:
    __DEVICE_TAG__ Scalar operator()( Scalar phi_impl, Scalar phi_expl ) const
    {
        return Scalar( 0.0 );
    }

    __DEVICE_TAG__ Scalar get_derivative( Scalar phi ) const
    {
        return Scalar( 0.0 );
    }

    __DEVICE_TAG__ Scalar get_energy( Scalar phi ) const
    {
        return Scalar( 0.0 );
    }

    Scalar get_phi_eq() const
    {
        return Scalar( 1 );
    }

    Scalar get_curvature() const
    {
        return Scalar( 0 );
    }
};

} // namespace tests

#endif
