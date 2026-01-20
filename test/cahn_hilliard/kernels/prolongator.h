#ifndef __PROLONGATOR_KERNEL_H__
#define __PROLONGATOR_KERNEL_H__

#include <scfd/utils/device_tag.h>

namespace kernels
{

template <class IdxND, class Ord, class VectorType, int TensorDim>
struct prolongator_kernel
{
    VectorType dom, img;

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        const auto &curr = idx;
        const auto  half = idx / Ord{ 2 };

        for ( int i = 0; i < TensorDim; i++ )
        {
            img( curr, i ) = dom( half, i );
        }
    }
};

} // namespace kernels

#endif
