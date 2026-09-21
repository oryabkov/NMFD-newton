#ifndef __MOBILITY_H__
#define __MOBILITY_H__

#include <cmath>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>


namespace tests
{


enum class face_avg
{
    midpoint,
    arithmetic,
    geometric,
    harmonic
};

template <class Scalar>
class constant_mobility
{
    using st = scfd::utils::scalar_traits<Scalar>;

public:
    constant_mobility(Scalar D = 1.0, face_avg rule = face_avg::midpoint): D_(D), rule_(rule)
    {
    }

    __DEVICE_TAG__ Scalar operator()( Scalar phi ) const
    {
        return D_;
    }

    __DEVICE_TAG__ Scalar get_derivative( Scalar phi ) const
    {
        return Scalar( 0.0 );
    }

    __DEVICE_TAG__ Scalar face( Scalar a, Scalar b ) const
    {
        switch ( rule_ )
        {
            case face_avg::midpoint:
            {
                const Scalar m = ( a + b ) / 2;
                return (*this)( m );
            }
            case face_avg::arithmetic:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return 0.5 * ( Ma + Mb );
            }
            case face_avg::geometric:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return st::sqrt( Ma * Mb );
            }
            case face_avg::harmonic:
            default:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return 2 * Ma * Mb / ( Ma + Mb );
            }
        }
    }

    __DEVICE_TAG__ Scalar face_diff_a( Scalar a, Scalar b ) const
    {
        switch ( rule_ )
        {
            case face_avg::midpoint:
            {
                const Scalar m = ( a + b ) / 2;
                return 0.5 * get_derivative( m );
            }
            case face_avg::arithmetic:
            {
                return 0.5 * get_derivative( a );
            }
            case face_avg::geometric:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return Scalar( 0.5 ) * st::sqrt( Mb / Ma ) * get_derivative( a );
            }
            case face_avg::harmonic:
            default:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return 2 * Mb * Mb / ( ( Ma + Mb ) * ( Ma + Mb ) ) * get_derivative( a );
            }
        }
    }

    __DEVICE_TAG__ Scalar face_diff_b( Scalar a, Scalar b ) const
    {
        switch ( rule_ )
        {
            case face_avg::midpoint:
            {
                const Scalar m = ( a + b ) / 2;
                return 0.5 * get_derivative( m );
            }
            case face_avg::arithmetic:
            {
                return 0.5 * get_derivative( b );
            }
            case face_avg::geometric:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return Scalar( 0.5 ) * st::sqrt( Ma / Mb ) * get_derivative( b );
            }
            case face_avg::harmonic:
            default:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return 2 * Ma * Ma / ( ( Ma + Mb ) * ( Ma + Mb ) ) * get_derivative( b );
            }
        }
    }

    Scalar get_D() const
    {
        return D_;
    }

    face_avg get_face_avg() const
    {
        return rule_;
    }

private:
    Scalar D_;
    face_avg rule_;
};

template <class Scalar>
class parabolic_mobility
{
    using st = scfd::utils::scalar_traits<Scalar>;

public:
    parabolic_mobility(Scalar D = 1.0, Scalar floor_val = 1e-5, Scalar phi_eq = 1.0, face_avg rule = face_avg::midpoint): D_(D), floor_(floor_val), phi_eq_(phi_eq), rule_(rule)
    {
        A_ = D_*D_ - floor_*floor_;
    }

    __DEVICE_TAG__ Scalar operator()( Scalar phi ) const
    {
        if (st::abs(phi) <= phi_eq_)
        {
            const Scalar u = Scalar( 1 ) - phi * phi / ( phi_eq_ * phi_eq_ );
            return st::sqrt(A_ * u * u + floor_ * floor_);
        }
        else
        {
            return floor_;
        }
    }

    __DEVICE_TAG__ Scalar get_derivative( Scalar phi ) const
    {
        if (st::abs(phi) <= phi_eq_)
        {
            const Scalar u = Scalar( 1 ) - phi * phi / ( phi_eq_ * phi_eq_ );
            return -2 * A_ * u * phi / ( phi_eq_ * phi_eq_ * st::sqrt(A_ * u * u + floor_ * floor_) );
        }
        else
        {
            return Scalar( 0.0 );
        }
    }

    __DEVICE_TAG__ Scalar face( Scalar a, Scalar b ) const
    {
        switch ( rule_ )
        {
            case face_avg::midpoint:
            {
                const Scalar m = ( a + b ) / 2;
                return (*this)( m );
            }
            case face_avg::arithmetic:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return 0.5 * ( Ma + Mb );
            }
            case face_avg::geometric:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return st::sqrt( Ma * Mb );
            }
            case face_avg::harmonic:
            default:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return 2 * Ma * Mb / ( Ma + Mb );
            }
        }
    }

    __DEVICE_TAG__ Scalar face_diff_a( Scalar a, Scalar b ) const
    {
        switch ( rule_ )
        {
            case face_avg::midpoint:
            {
                const Scalar m = ( a + b ) / 2;
                return 0.5 * get_derivative( m );
            }
            case face_avg::arithmetic:
            {
                return 0.5 * get_derivative( a );
            }
            case face_avg::geometric:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return Scalar( 0.5 ) * st::sqrt( Mb / Ma ) * get_derivative( a );
            }
            case face_avg::harmonic:
            default:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return 2 * Mb * Mb / ( ( Ma + Mb ) * ( Ma + Mb ) ) * get_derivative( a );
            }
        }
    }

    __DEVICE_TAG__ Scalar face_diff_b( Scalar a, Scalar b ) const
    {
        switch ( rule_ )
        {
            case face_avg::midpoint:
            {
                const Scalar m = ( a + b ) / 2;
                return 0.5 * get_derivative( m );
            }
            case face_avg::arithmetic:
            {
                return 0.5 * get_derivative( b );
            }
            case face_avg::geometric:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return Scalar( 0.5 ) * st::sqrt( Ma / Mb ) * get_derivative( b );
            }
            case face_avg::harmonic:
            default:
            {
                const Scalar Ma = (*this)( a );
                const Scalar Mb = (*this)( b );
                return 2 * Ma * Ma / ( ( Ma + Mb ) * ( Ma + Mb ) ) * get_derivative( b );
            }
        }
    }

    Scalar get_D() const
    {
        return D_;
    }

    Scalar get_floor() const
    {
        return floor_;
    }

    Scalar get_phi_eq() const
    {
        return phi_eq_;
    }

    face_avg get_face_avg() const
    {
        return rule_;
    }

private:
    Scalar D_, floor_, phi_eq_;
    Scalar A_; // Helpful constants
    face_avg rule_;
};

} // namespace tests

#endif
