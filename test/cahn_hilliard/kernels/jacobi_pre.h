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
        MatType mat{ 0., 0., 0., 0. };

        auto vec = v.get_vec( idx );

#pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ )
        {
            const auto N  = range[j];
            const auto hj = step[j];

            // Scalar diag_j_0{ -2. };
            // Scalar diag_j_1{ -2. };
            TensorType diag_j{ -2.0, -2.0 };

            if ( idx[j] == 0 )
            {
                // diag_j_0 += cond.left[j][0];
                // diag_j_1 += cond.left[j][1];
                diag_j += cond.left[j];
            }

            if ( idx[j] == N - 1 )
            {
                // diag_j_0 += cond.right[j][0];
                // diag_j_1 += cond.right[j][1];
                diag_j += cond.right[j];
            }

            mat( 0, 0 ) += D * diag_j[0] / ( hj * hj );
            mat( 1, 1 ) += gamma * diag_j[1] / ( hj * hj );
        }
        mat( 0, 0 ) -= dt_inf;
        Scalar phi = vector.get_vec( idx )[1];
        mat( 1, 1 ) -= phobic_en.get_derivative( phi );
        mat( 1, 0 ) = 1.;

        auto result = inv( mat ) * vec;
        v.set_vec( result, idx );
    }
};

} // namespace kernels

#endif
