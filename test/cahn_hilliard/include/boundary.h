#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__

#include <scfd/utils/device_tag.h>

template <int Dim, int TensorDim>
struct boundary_cond
{
    // -1 for dirichlet
    // +1 for neumann
    using conditions = int;

    conditions left[Dim][TensorDim];
    conditions right[Dim][TensorDim];

    //TODO add account for periodic (0)

    template <class Vector, class IdxND, class Tensor>
    __DEVICE_TAG__ void
    get_ghost_tensor( const Vector &vector, const IdxND &dom_sz, const IdxND &ghost_idx, Tensor &res ) const
    {
        IdxND  internal_idx = ghost_idx;
        Tensor mul          = Tensor::make_ones();
#pragma unroll
        for ( int j = 0; j < Dim; ++j )
        {
            if ( ghost_idx[j] < 0 )
            {
                internal_idx[j] = -ghost_idx[j] - 1;
#pragma unroll
                for ( int jj = 0; jj < TensorDim; ++jj )
                {
                    mul[jj] *= left[j][jj];
                }
            }
            else if ( ghost_idx[j] >= dom_sz[j] )
            {
                internal_idx[j] = 2 * dom_sz[j] - ghost_idx[j] - 1;
#pragma unroll
                for ( int jj = 0; jj < TensorDim; ++jj )
                {
                    mul[jj] *= right[j][jj];
                }
            }
        }
        vector.get_vec( res, internal_idx );
#pragma unroll
        for ( int jj = 0; jj < TensorDim; ++jj )
        {
            res[jj] *= mul[jj];
        }
    }
    template <class Vector, class IdxND, class Tensor>
    __DEVICE_TAG__ void get_ghost_tensor_linearized(
        const Vector &vector, const Vector &v, const IdxND &dom_sz, const IdxND &ghost_idx, Tensor &res
    ) const
    {
        get_ghost_tensor( v, dom_sz, ghost_idx, res );
    }
};

#endif
