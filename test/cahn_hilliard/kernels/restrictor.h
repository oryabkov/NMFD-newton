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

#pragma unroll
            for ( IdxND j = r._bypass_start(); r.is_own( j ); r._bypass_step( j ) )
            {
                sum += dom( j, i );
            }

            img( idx, i ) = sum / num_cells;
        }
    }
};

} // namespace kernels

#endif
