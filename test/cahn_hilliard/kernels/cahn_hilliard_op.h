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
    class Rhs,
    class Mobility>
struct cahn_hilliard_op_kernel
{
    VectorType     in, out;
    IdxND          range;
    GridStep       step;
    BoundaryCond   cond;
    PhobicEnergy   phobic_en;
    Rhs            rhs;
    Mobility       mobility;
    VectorType     previous_state;
    Scalar         dt_inf;
    Scalar gamma; // squared length of the transition regions between the domains

    // using periodic_bc_vector = tests::periodic_bc_vector<IdxND, Scalar, TensorType, VectorType>;

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

        TensorType ghost{ Scalar(0), Scalar(0) };

        // First equation: div(M(phi) grad(psi))
        // Second equation: psi + gamma * laplace(phi) - F(phi) = 0
        state[0] -= (curr[1] - prev[1]) * dt_inf; // Apply time derivative
        state[1] += curr[0] - phobic_en( curr[1] );
        #pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ ) // iterate over x, y, z,... dimension
        {
            auto N = range[j];

            auto ej = IdxND::make_unit( j );
            auto hj = step[j];

            TensorType prev_vec;
            if ( idx[j] == 0 )
            {
                cond.get_ghost_tensor( in, range, idx - ej, step, ghost ); // Calculate bc vector
                const auto periodic_vec = tests::periodic_bc_vector<IdxND, Scalar, TensorType, VectorType>( in, idx, j, N, true ); // Calculate periodic bc vector
                #pragma unroll
                for ( int c = 0; c < TensorType::dim; ++c )
                {
                    prev_vec[c] = ( cond.left[j][c] == 0 ) ? periodic_vec[c] : ghost[c];
                }
            }
            else
            {
                prev_vec = in.get_vec( idx - ej );
            }

            TensorType next_vec;
            if ( idx[j] == N - 1 )
            {
                cond.get_ghost_tensor( in, range, idx + ej, step, ghost ); // Calculate bc vector
                const auto periodic_vec = tests::periodic_bc_vector<IdxND, Scalar, TensorType, VectorType>( in, idx, j, N, false ); // Calculate periodic bc vector
                #pragma unroll
                for ( int c = 0; c < TensorType::dim; ++c )
                {
                    next_vec[c] = ( cond.right[j][c] == 0 ) ? periodic_vec[c] : ghost[c];
                }
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
