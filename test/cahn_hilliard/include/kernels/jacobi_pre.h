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
    Scalar gamma;
    Scalar       alpha_min;
    Scalar       alpha_max;

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        MatType mat{ Scalar(0), Scalar(0), Scalar(0), Scalar(0) };

        auto vec = vector.get_vec( idx );
        auto lin_curr = lin_vector.get_vec( idx ); // [psi_lin, phi_lin]

        TensorType diag_ghost{ Scalar(0), Scalar(0) };

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
                cond.get_diag_ghost_coef_linearized( lin_vector, range, idx - ej, j, /*is_left*/ true, step, diag_ghost );
                diag_j += diag_ghost;
                cond.get_lin_neighbor( lin_vector, range, idx - ej, j, /*is_left*/ true, step, prev_lin_vec );
            }
            else
            {
                prev_lin_vec = lin_vector.get_vec( idx - ej );
            }

            TensorType next_lin_vec;
            if ( idx[j] == N - 1 )
            {
                cond.get_diag_ghost_coef_linearized( lin_vector, range, idx + ej, j, /*is_left*/ false, step, diag_ghost );
                diag_j += diag_ghost;
                cond.get_lin_neighbor( lin_vector, range, idx + ej, j, /*is_left*/ false, step, next_lin_vec );
            }
            else
            {
                next_lin_vec = lin_vector.get_vec( idx + ej );
            }

            const Scalar mobility_deriv_plus_half  = mobility.get_derivative( ( next_lin_vec[1] + lin_curr[1] ) / Scalar( 2 ) );
            const Scalar mobility_deriv_minus_half = mobility.get_derivative( ( prev_lin_vec[1] + lin_curr[1] ) / Scalar( 2 ) );

            mat( 0, 0 ) += mobility(lin_curr[1]) * diag_j[0] / Scalar(hj * hj);
            // ============ VARIANT 1: continuous linearization ============
            // Слагаемое M' * (grad(d_phi) . grad(psi)) при центральных разностях
            // не содержит d_phi_i, поэтому вклада в диагональ нет.
            // mat( 0, 1 ) += (
            //     mobility_deriv_plus_half  * next_lin_vec[0] +
            //     mobility_deriv_minus_half * prev_lin_vec[0] -
            //     ( mobility_deriv_plus_half + mobility_deriv_minus_half ) * lin_curr[0]
            // ) / Scalar( hj * hj );

            // ============ VARIANT 2: discrete linearization ============
            // // Только диагональный коэффициент при d_phi_i, с множителем 1/2
            mat( 0, 1 ) += Scalar( 0.5 ) * (
                mobility_deriv_plus_half  * next_lin_vec[0] +
                mobility_deriv_minus_half * prev_lin_vec[0] -
                ( mobility_deriv_plus_half + mobility_deriv_minus_half ) * lin_curr[0]
            ) / Scalar( hj * hj );

            mat( 1, 1 ) += gamma * diag_j[1] / Scalar(hj * hj);
        }
        mat( 0, 1 ) -= dt_inf;
        const Scalar phobic_deriv = phobic_en.get_derivative( lin_curr[1] );
        mat( 1, 1 ) -= phobic_deriv;
        mat( 1, 0 ) = Scalar(1);

        auto Dinv = inv( mat );

        // Local-Fourier-analysis estimate: at the Nyquist frequency the
        // centered-difference Laplacian symbol doubles the plain diagonal,
        // so N(pi) = -diag(mat(0,0), mat(1,1)+phobic_deriv); alpha* minimizes
        // the worst-case weighted-Jacobi amplification for that mode.
        const Scalar lap_psi = mat( 0, 0 );
        const Scalar lap_phi = mat( 1, 1 ) + phobic_deriv;
        const Scalar trace_Dinv_N = -( Dinv( 0, 0 ) * lap_psi + Dinv( 1, 1 ) * lap_phi );
        Scalar alpha_eff = Scalar(2) / ( Scalar(2) - trace_Dinv_N );
        alpha_eff = alpha_eff < alpha_min ? alpha_min : ( alpha_eff > alpha_max ? alpha_max : alpha_eff );

        auto result = alpha_eff * Dinv * vec;
        vector.set_vec( result, idx );
    }
};

} // namespace kernels

#endif
