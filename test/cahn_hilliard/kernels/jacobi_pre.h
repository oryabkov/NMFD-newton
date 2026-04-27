#ifndef __JACOBI_PRE_KERNEL_H__
#define __JACOBI_PRE_KERNEL_H__

#include <scfd/static_mat/mat.h>
#include <scfd/utils/device_tag.h>

namespace kernels
{

template <
    class IdxND,
    class Scalar,
    class TensorType,
    class VectorType,
    class MatType,
    class GridStep,
    class BoundaryCond,
    class PhobicEnergy,
    class Mobility>
struct jacobi_pre_kernel
{
    VectorType   vector, lin_vector;
    IdxND        range;
    GridStep     step;
    BoundaryCond cond;
    PhobicEnergy phobic_en;
    Mobility     mobility;
    Scalar       dt_inf;
    Scalar       alpha;
    Scalar gamma;

    // using periodic_bc_vector = tests::periodic_bc_vector<IdxND, Scalar, TensorType, VectorType>;

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        MatType mat{ Scalar(0), Scalar(0), Scalar(0), Scalar(0) };

        auto vec = vector.get_vec( idx );
        auto lin_curr = lin_vector.get_vec( idx ); // [psi_lin, phi_lin]

        TensorType diag_ghost{ Scalar(0), Scalar(0) };
        TensorType lin_ghost{ Scalar(0), Scalar(0) };

#pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ )
        {
            const auto N  = range[j];
            const auto hj = step[j];
            const auto ej = IdxND::make_unit( j );

            TensorType diag_j{ Scalar(-2), Scalar(-2) };

            TensorType prev_lin_vec;
            if ( idx[j] == 0 )
            {
                cond.get_ghost_coef_linearized( lin_vector, range, idx - ej, step, diag_ghost );
                diag_j += diag_ghost;

                // BC for CH problem
                cond.get_ghost_tensor( lin_vector, range, idx - ej, step, lin_ghost );
                const auto periodic_lin_vec = tests::periodic_bc_vector<IdxND, Scalar, TensorType, VectorType>( lin_vector, idx, j, N, true );

                #pragma unroll
                for ( int c = 0; c < TensorType::dim; ++c )
                {
                    prev_lin_vec[c] = ( cond.left[j][c] == 0 ) ? periodic_lin_vec[c] : lin_ghost[c];
                }
            }
            else
            {
                prev_lin_vec = lin_vector.get_vec( idx - ej );
            }

            TensorType next_lin_vec;
            if ( idx[j] == N - 1 )
            {
                cond.get_ghost_coef_linearized( lin_vector, range, idx + ej, step, diag_ghost );
                diag_j += diag_ghost;

                // BC for CH problem
                cond.get_ghost_tensor( lin_vector, range, idx + ej, step, lin_ghost );
                const auto periodic_lin_vec = tests::periodic_bc_vector<IdxND, Scalar, TensorType, VectorType>( lin_vector, idx, j, N, false );

                #pragma unroll
                for ( int c = 0; c < TensorType::dim; ++c )
                {
                    next_lin_vec[c] = ( cond.right[j][c] == 0 ) ? periodic_lin_vec[c] : lin_ghost[c];
                }
            }
            else
            {
                next_lin_vec = lin_vector.get_vec( idx + ej );
            }

            const Scalar mobility_deriv_plus_half  = mobility.get_derivative( ( next_lin_vec[1] + lin_curr[1] ) / Scalar( 2 ) );
            const Scalar mobility_deriv_minus_half = mobility.get_derivative( ( prev_lin_vec[1] + lin_curr[1] ) / Scalar( 2 ) );

            mat( 0, 0 ) += mobility(lin_curr[1]) * diag_j[0] / Scalar(hj * hj);
            mat( 0, 1 ) += (
                mobility_deriv_plus_half * next_lin_vec[0] +
                mobility_deriv_minus_half * prev_lin_vec[0] -
                ( mobility_deriv_plus_half + mobility_deriv_minus_half ) * lin_curr[0]
            ) / Scalar( hj * hj );
            mat( 1, 1 ) += gamma * diag_j[1] / Scalar(hj * hj);
        }
        mat( 0, 1 ) -= dt_inf;
        mat( 1, 1 ) -= phobic_en.get_derivative( lin_curr[1] );
        mat( 1, 0 ) = Scalar(1);

        auto result = alpha * inv( mat ) * vec;
        vector.set_vec( result, idx );
    }
};

} // namespace kernels

#endif
