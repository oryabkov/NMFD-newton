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
        TensorType state{ 0, 0 };

        auto curr = in.get_vec( idx );

// D * laplace(d_psi) - d_psi / dt
#pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ ) // iterate over x, y, z,... dimension
        {
            auto N = range[j];

            auto ej = IdxND::make_unit( j );
            auto hj = step[j];

            auto prev = idx[j] == 0 ? cond.left[j][0] * curr[0] : in.get_vec( idx - ej )[0];
            auto next = idx[j] == N - 1 ? cond.right[j][0] * curr[0] : in.get_vec( idx + ej )[0];

            state[0] += D * ( next + prev - 2 * curr[0] ) / ( hj * hj );
        }
        state[0] -= curr[0] * dt_inf;

        // d_psi + gamma * laplace(d_phi) - (3 * phi^2 - 1) * d_phi
        Scalar phi = vector.get_vec( idx )[1];
        state[1]   = curr[0] - phobic_en.get_derivative( phi ) * curr[1];
#pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ )
        {
            auto N = range[j];

            auto ej = IdxND::make_unit( j );
            auto hj = step[j];

            auto prev = idx[j] == 0u ? cond.left[j][1] * curr[1] : in.get_vec( idx - ej )[1];
            auto next = idx[j] == N - 1u ? cond.right[j][1] * curr[1] : in.get_vec( idx + ej )[1];

            state[1] += gamma * ( next + prev - 2 * curr[1] ) / ( hj * hj );
        }

        out.set_vec( state, idx );
    }
};

} // namespace kernels

#endif
