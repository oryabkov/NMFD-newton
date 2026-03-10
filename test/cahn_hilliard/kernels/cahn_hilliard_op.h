#ifndef __CAHN_HILLIARD_OP_KERNEL_H__
#define __CAHN_HILLIARD_OP_KERNEL_H__

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
    class PhobicEnergy,
    class Rhs>
struct cahn_hilliard_op_kernel
{
    VectorType     in, out;
    IdxND          range;
    GridStep       step;
    BoundaryCond   cond;
    PhobicEnergy   phobic_en;
    Rhs            rhs;
    VectorType     previous_state;
    Scalar         dt_inf;

    Scalar D;     // diffusion coefficient
    Scalar gamma; // squared length of the transition regions between the domains

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        // TensorType state{ 0, 0 };
        Scalar x = step[0] * ( 0.5 + idx[0] );
        Scalar y = step[1] * ( 0.5 + idx[1] );
        Scalar z = step[2] * ( 0.5 + idx[2] );

        // Apply rhs
        TensorType state = -rhs( x, y, z );

        auto curr = in.get_vec( idx );
        auto prev = previous_state.get_vec( idx );

        // Apply time derivative
        state[0] -= (curr[1] - prev[1]) * dt_inf;

        TensorType ghost{ Scalar(0), Scalar(0) };;

        // First equation: D * laplace(psi)
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
                    cond.get_ghost_tensor_linearized( in, in, range, idx + ej, ghost );
                    next_val = ghost[0];
                }
            }
            else
            {
                next_val = in.get_vec( idx + ej )[0];
            }

            state[0] += D * ( next_val + prev_val - 2 * curr[0] ) / ( hj * hj );
        }

        // Second equation: psi + gamma * laplace(phi) - f(phi) = 0
        state[1] += curr[0] - phobic_en( curr[1] );
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
                    cond.get_ghost_tensor_linearized( in, in, range, idx + ej, ghost );
                    next_val = ghost[1];
                }
            }
            else
            {
                next_val = in.get_vec( idx + ej )[1];
            }

            state[1] += gamma * ( next_val + prev_val - 2 * curr[1] ) / ( hj * hj );
        }

        out.set_vec( state, idx );
    }
};

} // namespace kernels

#endif
