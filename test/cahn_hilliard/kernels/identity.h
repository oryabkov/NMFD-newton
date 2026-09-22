#ifndef __IDENTITY_KERNEL_H__
#define __IDENTITY_KERNEL_H__

#include <scfd/utils/device_tag.h>

namespace kernels
{

template <class IdxND, class VectorType, int TensorDim>
struct identity_kernel
{
    VectorType dom, img;

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        for ( int i = 0; i < TensorDim; i++ )
        {
            img( idx, i ) = dom( idx, i );
        }
    }
};

} // namespace kernels

#endif
