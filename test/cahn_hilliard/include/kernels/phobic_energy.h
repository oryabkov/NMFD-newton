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

    Scalar get_profile_scale( Scalar gamma ) const
    {
        return Scalar( 2 ) * std::sqrt( gamma / get_curvature() );
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

    Scalar get_profile_scale( Scalar gamma ) const
    {
        return Scalar( 2 ) * std::sqrt( gamma / get_curvature() );
    }

    Scalar get_omega() const
    {
        return omega_;
    }

private:
    Scalar omega_;
};

template <class Scalar>
class smoothed_obstacle_potential
{
    using st = scfd::utils::scalar_traits<Scalar>;

public:
    smoothed_obstacle_potential(Scalar eta = 0.15)
        : eta_(eta),
          a_( Scalar( 0.25 ) / ( std::sqrt( Scalar( 1.0 ) + eta * eta * eta * eta ) - eta * eta ) )
    {
    }

    __DEVICE_TAG__ Scalar operator()( Scalar phi_impl, Scalar phi_expl ) const
    {
        Scalar e4 = eta_ * eta_ * eta_ * eta_;
        Scalar s  = Scalar( 1.0 ) - phi_impl * phi_impl;
        return phi_impl * ( Scalar( 1.0 ) - Scalar( 2.0 ) * a_ * s / st::sqrt( s * s + e4 ) ) - phi_expl;
    }

    __DEVICE_TAG__ Scalar get_derivative( Scalar phi ) const
    {
        Scalar e4 = eta_ * eta_ * eta_ * eta_;
        Scalar s  = Scalar( 1.0 ) - phi * phi;
        Scalar d  = st::sqrt( s * s + e4 );
        return a_ * ( Scalar( -2.0 ) * s / d + Scalar( 4.0 ) * phi * phi * e4 / ( d * d * d ) );
    }

    __DEVICE_TAG__ Scalar get_energy( Scalar phi ) const
    {
        Scalar e2 = eta_ * eta_;
        Scalar s  = Scalar( 1.0 ) - phi * phi;
        return a_ * ( st::sqrt( s * s + e2 * e2 ) - e2 );
    }

    Scalar get_phi_eq() const
    {
        return Scalar( 1 );
    }

    Scalar get_curvature() const
    {
        return get_derivative( get_phi_eq() );
    }

    // The tail length sqrt(gamma/f''(phi_eq)) collapses as eta -> 0 and no longer describes the
    // profile, so the initial tanh is matched to the slope at phi = 0 instead. The prefactor a_
    // pins the barrier at 1/4 for every eta, so this returns sqrt(2 gamma) -- the same width the
    // double well gets at the same gamma, which is what makes --gamma comparable across potentials.
    Scalar get_profile_scale( Scalar gamma ) const
    {
        return std::sqrt( gamma / ( Scalar( 2 ) * ( get_energy( Scalar( 0 ) ) - get_energy( get_phi_eq() ) ) ) );
    }

    Scalar get_eta() const
    {
        return eta_;
    }

private:
    Scalar eta_;
    Scalar a_;
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

    Scalar get_profile_scale( Scalar gamma ) const
    {
        return std::sqrt( Scalar( 2 ) * gamma );
    }
};

} // namespace tests

#endif
