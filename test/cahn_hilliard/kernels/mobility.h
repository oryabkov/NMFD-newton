#ifndef __MOBILITY_H__
#define __MOBILITY_H__

#include <cmath>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>


namespace tests
{


template <class Scalar>
class constant_mobility
{
public:
    constant_mobility(Scalar D = 1.0): D_(D)
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


private:
    Scalar D_;
};

template <class Scalar>
class parabolic_mobility
{
    using st = scfd::utils::scalar_traits<Scalar>;

public:
    parabolic_mobility(Scalar D = 1.0, Scalar offset=0.5): D_(1), offset_(0.8)
    {
        A_ = D_*D_ - offset_*offset_;
        // B_ = offset_*offset_ / A_;
    }

    __DEVICE_TAG__ Scalar operator()( Scalar phi ) const
    {
        if (st::abs(phi) <= 1)
        {
            // return D_ * st::sqrt((1 + phi) * (1 + phi) * (1 - phi) * (1 - phi) + offset_ * offset_);
            return st::sqrt(A_ * (1 + phi) * (1 + phi) * (1 - phi) * (1 - phi) + offset_ * offset_);
        }
        else
        {
            // return D_ * offset_;
            return offset_;
        }
    }

    __DEVICE_TAG__ Scalar get_derivative( Scalar phi ) const
    {
        if (st::abs(phi) <= 1)
        {
            // return 2 * D_ * phi * (phi * phi - 1) / st::sqrt((1 + phi) * (1 + phi) * (1 - phi) * (1 - phi) + offset_ * offset_);
            return 2 * A_ * phi * (phi * phi - 1) / st::sqrt(A_ * (1 + phi) * (1 + phi) * (1 - phi) * (1 - phi) + offset_ * offset_);
        }
        else
        {
            return Scalar( 0.0 );
        }
    }

private:
    Scalar D_, offset_;
    Scalar A_; // Helpful constants
};

} // namespace tests

#endif
