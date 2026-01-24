#ifndef __RESTRICTOR_KERNEL_H__
#define __RESTRICTOR_KERNEL_H__

#include <scfd/static_vec/rect.h>
#include <scfd/utils/device_tag.h>

namespace kernels
{

template <class IdxND, class Ord, class VectorType, int TensorDim>
struct restrictor_kernel
{
    VectorType dom, img;

#if 0

    __DEVICE_TAG__ void operator()( const IdxND idx ) const // traversing image space
    {
        using Rect   = typename scfd::static_vec::rect<Ord, IdxND::dim>;
        using Scalar = typename VectorType::value_type;

        const Ord num_cells = Ord{ 1 } << IdxND::dim;

        const IdxND begin = Ord{ 2 } * ( idx + IdxND::make_zero() );
        const IdxND end   = Ord{ 2 } * ( idx + IdxND::make_ones() );

        Rect r{ begin, end };
        for ( int i = 0; i < TensorDim; i++ )
        {
            Scalar sum{ 0 };

#    pragma unroll
            for ( IdxND j = r._bypass_start(); r.is_own( j ); r._bypass_step( j ) )
            {
                sum += dom( j, i );
            }

            img( idx, i ) = sum / num_cells;
        }
    }

#else

    //static const int stencil_1d_half_sz = 2;
    //const typename VectorType::value_type coeffs_1d[stencil_1d_half_sz*2] = {0.0,0.5,0.5,0.0};

    //static const int                      stencil_1d_half_sz                = 2;
    //const typename VectorType::value_type coeffs_1d[stencil_1d_half_sz * 2] = {
    //    1.3020833333335308e-01, 3.6458333333330939e-01, 3.6458333333330939e-01, 1.3020833333335308e-01 };

    static const int                      stencil_1d_half_sz                = 3;
    const typename VectorType::value_type coeffs_1d[stencil_1d_half_sz * 2] = {
        5.2083333333793015e-03,
        1.3020833333335308e-01,
        3.6458333333330939e-01,
        3.6458333333330939e-01,
        1.3020833333335308e-01,
        5.2083333333793015e-03 };

    __DEVICE_TAG__ void operator()( const IdxND idx ) const // traversing image space
    {
        using Rect   = typename scfd::static_vec::rect<Ord, IdxND::dim>;
        using Scalar = typename VectorType::value_type;

        const IdxND begin0 = Ord{ 2 } * ( idx + IdxND::make_zero() );
        const IdxND end0   = Ord{ 2 } * ( idx + IdxND::make_ones() );

        Rect r{ begin0, end0 };
        r          = r.enlarged( stencil_1d_half_sz - 1 );
        Rect dom_r = dom.rect_nd();
        for ( int i = 0; i < TensorDim; i++ )
        {
            Scalar sum{ 0 };

#    pragma unroll
            for ( IdxND j = r._bypass_start(); r.is_own( j ); r._bypass_step( j ) )
            {
                if ( !dom_r.is_own( j ) )
                    continue;

                IdxND  rel_j = j - r.i1;
                Scalar mul( 1 );
#    pragma unroll
                for ( Ord k = 0; k < IdxND::dim; ++k )
                    mul *= coeffs_1d[rel_j[k]];
                sum += dom( j, i ) * mul;
            }

            img( idx, i ) = sum;
        }
    }

#endif
};

} // namespace kernels

#endif
