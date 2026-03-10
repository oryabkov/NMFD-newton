#ifndef __JACOBI_OP_KERNEL_H__
#define __JACOBI_OP_KERNEL_H__

#include <scfd/utils/device_tag.h>

namespace kernels
{

template <
    class IdxND,
    class Scalar,
    class TensorType,
    class VectorType,
    class GridStep,
    class BoundaryCond,
    class PhobicEnergy>
struct jacobi_op_kernel
{
    VectorType   in, out, vector;
    IdxND        range;
    GridStep     step;
    BoundaryCond cond;
    PhobicEnergy phobic_en;
    Scalar       dt_inf;

    Scalar D;
    Scalar gamma;

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        TensorType state{ Scalar(0), Scalar(0) };
        TensorType ghost{ Scalar(0), Scalar(0) };

        auto curr = in.get_vec( idx );

        // First equation Jacobian: D * laplace(d_psi) - d(d_phi)/dt
        #pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ ) // iterate over x, y, z,... dimension
        {
            auto N = range[j];

            auto ej = IdxND::make_unit( j );
            auto hj = step[j];

            Scalar prev_val;
            if ( idx[j] == 0 )
            {
                if ( cond.left[j][0] == 0 ) // periodic
                {
                    IdxND periodic_idx = idx;
                    periodic_idx[j] = N - 1;
                    prev_val = in.get_vec( periodic_idx )[0];
                }
                else
                {
                    // prev_val = cond.left[j][0] * curr[0];
                    cond.get_ghost_tensor_linearized( in, in, range, idx - ej, ghost );
                    prev_val = ghost[0];
                }
            }
            else
            {
                prev_val = in.get_vec( idx - ej )[0];
            }

            Scalar next_val;
            if ( idx[j] == N - 1 )
            {
                if ( cond.right[j][0] == 0 ) // periodic
                {
                    IdxND periodic_idx = idx;
                    periodic_idx[j] = 0;
                    next_val = in.get_vec( periodic_idx )[0];
                }
                else
                {
                    // next_val = cond.right[j][0] * curr[0];
                    cond.get_ghost_tensor_linearized( in, in, range, idx + ej, ghost );
                    next_val = ghost[0];
                }
            }
            else
            {
                next_val = in.get_vec( idx + ej )[0];
            }

            state[0] += D * ( next_val + prev_val - Scalar(2) * curr[0] ) / Scalar(hj * hj);
        }
        state[0] -= curr[1] * dt_inf;

        // Second equation Jacobian: d_psi + gamma * laplace(d_phi) - f'(phi) * d_phi
        Scalar phi = vector.get_vec( idx )[1];
        state[1]   = curr[0] - phobic_en.get_derivative( phi ) * curr[1];
        #pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ )
        {
            auto N = range[j];

            auto ej = IdxND::make_unit( j );
            auto hj = step[j];

            Scalar prev_val;
            if ( idx[j] == 0u )
            {
                if ( cond.left[j][1] == 0 ) // periodic
                {
                    IdxND periodic_idx = idx;
                    periodic_idx[j] = N - 1;
                    prev_val = in.get_vec( periodic_idx )[1];
                }
                else
                {
                    // prev_val = cond.left[j][1] * curr[1];
                    cond.get_ghost_tensor_linearized( in, in, range, idx - ej, ghost );
                    prev_val = ghost[1];
                }
            }
            else
            {
                prev_val = in.get_vec( idx - ej )[1];
            }

            Scalar next_val;
            if ( idx[j] == N - 1u )
            {
                if ( cond.right[j][1] == 0 ) // periodic
                {
                    IdxND periodic_idx = idx;
                    periodic_idx[j] = 0;
                    next_val = in.get_vec( periodic_idx )[1];
                }
                else
                {
                    // next_val = cond.right[j][1] * curr[1];
                    cond.get_ghost_tensor_linearized( in, in, range, idx + ej, ghost );
                    prev_val = ghost[1];
                }
            }
            else
            {
                next_val = in.get_vec( idx + ej )[1];
            }

            state[1] += gamma * ( next_val + prev_val - Scalar(2) * curr[1] ) / Scalar(hj * hj);
        }

        out.set_vec( state, idx );
    }
};

} // namespace kernels

#endif
