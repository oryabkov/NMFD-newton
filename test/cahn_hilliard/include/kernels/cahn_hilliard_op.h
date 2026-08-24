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
    class Mobility>
struct cahn_hilliard_op_kernel
{
    VectorType     in, out;
    IdxND          range;
    GridStep       step;
    BoundaryCond   cond;
    PhobicEnergy   phobic_en;
    VectorType     rhs;
    Mobility       mobility;
    VectorType     previous_state;
    Scalar         dt_inf;
    Scalar gamma; // squared length of the transition regions between the domains

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        // TensorType state{ 0, 0 };
        // Apply rhs
        TensorType state = -rhs.get_vec( idx );

        auto curr = in.get_vec( idx );
        auto prev = previous_state.get_vec( idx );

        // First equation: div(M(phi) grad(psi))
        // Second equation: psi + gamma * laplace(phi) - F(phi) = 0
        state[0] -= (curr[1] - prev[1]) * dt_inf; // Apply time derivative
        state[1] += curr[0] - phobic_en( curr[1], prev[1] );
        #pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ ) // iterate over x, y, z,... dimension
        {
            auto N = range[j];

            auto ej = IdxND::make_unit( j );
            auto hj = step[j];

            TensorType prev_vec;
            if ( idx[j] == 0 )
            {
                cond.get_lin_neighbor( in, range, idx - ej, j, /*is_left*/ true, step, prev_vec );
            }
            else
            {
                prev_vec = in.get_vec( idx - ej );
            }

            TensorType next_vec;
            if ( idx[j] == N - 1 )
            {
                cond.get_lin_neighbor( in, range, idx + ej, j, /*is_left*/ false, step, next_vec );
            }
            else
            {
                next_vec = in.get_vec( idx + ej );
            }

            const Scalar mobility_plus_half  = mobility( ( next_vec[1] + curr[1] ) / Scalar( 2 ) ); // i+1/2
            const Scalar mobility_minus_half = mobility( ( prev_vec[1] + curr[1] ) / Scalar( 2 ) ); // i-1/2

            state[0] += (
                mobility_plus_half * next_vec[0] +
                mobility_minus_half * prev_vec[0] -
                ( mobility_plus_half + mobility_minus_half ) * curr[0]
            ) / ( hj * hj );
            state[1] += gamma * ( next_vec[1] + prev_vec[1] - Scalar( 2 ) * curr[1] ) / ( hj * hj );
        }

        out.set_vec( state, idx );
    }
};

} // namespace kernels

#endif
