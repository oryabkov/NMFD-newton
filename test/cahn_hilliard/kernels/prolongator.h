#ifndef __PROLONGATOR_KERNEL_H__
#define __PROLONGATOR_KERNEL_H__

#include <scfd/utils/device_tag.h>

namespace kernels
{

template <class IdxND, class Ord, class VectorType, int TensorDim, class BoundaryCond>
struct prolongator_kernel
{
    using Rect = typename scfd::static_vec::rect<Ord, IdxND::dim>;

    VectorType   dom, img; // dom can be [-1, 0, 1, ..., N-1, N]
    BoundaryCond cond;
    Rect dom_r; // Real rect [0, 1, ..., N-1]

#if 0

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        const auto &curr = idx;
        const auto  half = idx / Ord{ 2 };

        for ( int i = 0; i < TensorDim; i++ )
        {
            img( curr, i ) = dom( half, i );
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
        5.2083333333793015e-03, 1.3020833333335308e-01, 3.6458333333330939e-01,
        3.6458333333330939e-01, 1.3020833333335308e-01, 5.2083333333793015e-03 };

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        // using Rect   = typename scfd::static_vec::rect<Ord, IdxND::dim>;
        using Scalar = typename VectorType::value_type;
        using Tensor = typename scfd::static_vec::vec<Scalar, TensorDim>;

        const IdxND begin0 = ( idx - IdxND::make_ones() * ( stencil_1d_half_sz - 1 ) ) / Ord{ 2 };
        const IdxND end0   = ( idx + IdxND::make_ones() * ( stencil_1d_half_sz - 1 ) ) / Ord{ 2 } + IdxND::make_ones();

        Rect r{ begin0, end0 };
        //TODO use dom_r as external parameter (important for mgpu)
        // Rect dom_r = dom.rect_nd();
        for ( int i = 0; i < TensorDim; i++ )
        {
            Scalar sum{ 0 }, mul_sum{ 0 };

#    pragma unroll
            for ( IdxND j = r._bypass_start(); r.is_own( j ); r._bypass_step( j ) )
            {
                IdxND  rel_j = idx - j * Ord{ 2 } + IdxND::make_ones() * ( stencil_1d_half_sz - 1 );
                Scalar mul( 1 );
#    pragma unroll
                for ( Ord k = 0; k < IdxND::dim; ++k )
                    mul *= coeffs_1d[rel_j[k]] * Scalar( 2 );
                if ( !dom_r.is_own( j ) )
                {
                    Tensor ghost;
                    //TODO 1st argument should be linearization point - pass it as external param - important for nonlinear bc
                    cond.get_ghost_tensor_linearized( dom, dom, dom_r.i2, j, ghost );
                    sum += ghost[i] * mul;
                }
                else
                {
                    sum += dom( j, i ) * mul;
                }
                mul_sum += mul;
            }

            img( idx, i ) = sum;
            //img( idx, i ) = sum / mul_sum;
        }
    }

#endif
};

} // namespace kernels

#endif
