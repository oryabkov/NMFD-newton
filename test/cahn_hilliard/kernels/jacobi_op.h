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
    class PhobicEnergy,
    class Mobility>
struct jacobi_op_kernel
{
    VectorType   in, out, lin_vector;
    IdxND        range;
    GridStep     step;
    BoundaryCond cond;
    PhobicEnergy phobic_en;
    Mobility     mobility;
    Scalar       dt_inf;
    Scalar gamma;

    // using periodic_bc_vector = tests::periodic_bc_vector<IdxND, Scalar, TensorType, VectorType>;

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        TensorType state{ Scalar(0), Scalar(0) };
        TensorType ghost{ Scalar(0), Scalar(0) };
        TensorType lin_ghost{ Scalar(0), Scalar(0) };

        auto curr = in.get_vec( idx ); // [d_psi, d_phi]
        auto lin_curr = lin_vector.get_vec( idx ); // [psi_lin, phi_lin]

        // First equation Jacobian: div(M(phi_lin) grad(d_psi)) + div(M'(phi_lin) grad(psi)) * d_phi - d(d_phi)/dt
        // Second equation Jacobian: d_psi + gamma * laplace(d_phi) - f'(phi_lin) * d_phi
        state[0] = -curr[1] * dt_inf;
        state[1] = curr[0] - phobic_en.get_derivative( lin_curr[1] ) * curr[1];
        #pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ ) // iterate over x, y, z,... dimension
        {
            auto N = range[j];

            auto ej = IdxND::make_unit( j );
            auto hj = step[j];

            TensorType prev_vec;
            TensorType prev_lin_vec;
            if ( idx[j] == 0 )
            {
                // BC for linearized problem
                cond.get_ghost_tensor_linearized( lin_vector, in, range, idx - ej, step, ghost );
                const auto periodic_vec = tests::periodic_bc_vector<IdxND, Scalar, TensorType, VectorType>( in, idx, j, N, true );

                // BC for CH problem
                cond.get_ghost_tensor( lin_vector, range, idx - ej, step, lin_ghost );
                const auto periodic_lin_vec = tests::periodic_bc_vector<IdxND, Scalar, TensorType, VectorType>( lin_vector, idx, j, N, true );

                #pragma unroll
                for ( int c = 0; c < TensorType::dim; ++c )
                {
                    prev_vec[c]     = ( cond.left[j][c] == 0 ) ? periodic_vec[c]     : ghost[c];
                    prev_lin_vec[c] = ( cond.left[j][c] == 0 ) ? periodic_lin_vec[c] : lin_ghost[c];
                }
            }
            else
            {
                prev_vec = in.get_vec( idx - ej );
                prev_lin_vec = lin_vector.get_vec( idx - ej );
            }

            TensorType next_vec;
            TensorType next_lin_vec;
            if ( idx[j] == N - 1 )
            {
                // BC for linearized problem
                cond.get_ghost_tensor_linearized( lin_vector, in, range, idx + ej, step, ghost );
                const auto periodic_vec = tests::periodic_bc_vector<IdxND, Scalar, TensorType, VectorType>( in, idx, j, N, false );

                // BC for CH problem
                cond.get_ghost_tensor( lin_vector, range, idx + ej, step, lin_ghost );
                const auto periodic_lin_vec = tests::periodic_bc_vector<IdxND, Scalar, TensorType, VectorType>( lin_vector, idx, j, N, false );

                #pragma unroll
                for ( int c = 0; c < TensorType::dim; ++c )
                {
                    next_vec[c]     = ( cond.right[j][c] == 0 ) ? periodic_vec[c]     : ghost[c];
                    next_lin_vec[c] = ( cond.right[j][c] == 0 ) ? periodic_lin_vec[c] : lin_ghost[c];
                }
            }
            else
            {
                next_vec = in.get_vec( idx + ej );
                next_lin_vec = lin_vector.get_vec( idx + ej );
            }

            const Scalar mobility_plus_half  = mobility( ( next_lin_vec[1] + lin_curr[1] ) / Scalar( 2 ) );
            const Scalar mobility_minus_half = mobility( ( prev_lin_vec[1] + lin_curr[1] ) / Scalar( 2 ) );

            const Scalar mobility_deriv_plus_half  = mobility.get_derivative( ( next_lin_vec[1] + lin_curr[1] ) / Scalar( 2 ) );
            const Scalar mobility_deriv_minus_half = mobility.get_derivative( ( prev_lin_vec[1] + lin_curr[1] ) / Scalar( 2 ) );

            // [eq.1] div(M(phi_lin) grad(d_psi))
            state[0] += (
                mobility_plus_half * next_vec[0] +
                mobility_minus_half * prev_vec[0] -
                ( mobility_plus_half + mobility_minus_half ) * curr[0]
            ) / Scalar( hj * hj );

            // [eq.1] div(M'(phi_lin) grad(psi)) * d_phi
            state[0] += (
                mobility_deriv_plus_half * next_lin_vec[0] +
                mobility_deriv_minus_half * prev_lin_vec[0] -
                ( mobility_deriv_plus_half + mobility_deriv_minus_half ) * lin_curr[0]
            ) / Scalar( hj * hj ) * curr[1];

            // [eq.2] gamma * laplace(d_phi)
            state[1] += gamma * ( next_vec[1] + prev_vec[1] - Scalar(2) * curr[1] ) / Scalar(hj * hj);
        }

        out.set_vec( state, idx );
    }
};

} // namespace kernels

#endif
