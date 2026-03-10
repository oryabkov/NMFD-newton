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
    class PhobicEnergy>
struct jacobi_pre_kernel
{
    VectorType   v, vector;
    IdxND        range;
    GridStep     step;
    BoundaryCond cond;
    PhobicEnergy phobic_en;
    Scalar       dt_inf;

    Scalar D;
    Scalar gamma;

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        MatType mat{ Scalar(0), Scalar(0), Scalar(0), Scalar(0) };

        auto vec = v.get_vec( idx );

#pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ )
        {
            const auto N  = range[j];
            const auto hj = step[j];
            const auto ej = IdxND::make_unit( j );

            TensorType diag_j{ Scalar(-2), Scalar(-2) };
            TensorType diag_ghost{ Scalar(0), Scalar(0) };

            if ( idx[j] == 0 )
            {
                cond.get_ghost_coef_linearized( vector, range, idx - ej, diag_ghost );
                diag_j += diag_ghost;
            }

            if ( idx[j] == N - 1 )
            {
                cond.get_ghost_coef_linearized( vector, range, idx + ej, diag_ghost );
                diag_j += diag_ghost;
            }

            mat( 0, 0 ) += D * diag_j[0] / Scalar(hj * hj);
            mat( 1, 1 ) += gamma * diag_j[1] / Scalar(hj * hj);
        }
        mat( 0, 1 ) = -dt_inf;
        Scalar phi = vector.get_vec( idx )[1];
        mat( 1, 1 ) -= phobic_en.get_derivative( phi );
        mat( 1, 0 ) = Scalar(1);

        auto result = inv( mat ) * vec;
        v.set_vec( result, idx );
    }
};

} // namespace kernels

#endif
