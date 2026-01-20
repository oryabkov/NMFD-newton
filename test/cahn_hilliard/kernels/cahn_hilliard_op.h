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
    VectorType   in, out;
    IdxND        range;
    GridStep     step;
    BoundaryCond cond;
    PhobicEnergy phobic_en;
    Rhs          rhs;

    Scalar D;     // diffusion coefficient
    Scalar gamma; // squared length of the transition regions between the domains

    __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        // TensorType state{ 0, 0 };
        Scalar x = step[0] * ( 0.5 + idx[0] );
        Scalar y = step[1] * ( 0.5 + idx[1] );
        Scalar z = step[2] * ( 0.5 + idx[2] );

        TensorType state = -rhs( x, y, z );

        auto curr = in.get_vec( idx );

// D * laplace(psi)
#pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ ) // iterate over x, y, z,... dimension
        {
            auto N = range[j];

            auto ej = IdxND::make_unit( j );
            auto hj = step[j];

            auto prev = idx[j] == 0 ? cond.left[j] * curr[0] : in.get_vec( idx - ej )[0];
            auto next = idx[j] == N - 1 ? cond.right[j] * curr[0] : in.get_vec( idx + ej )[0];

            state[0] += D * ( next + prev - 2 * curr[0] ) / ( hj * hj );
        }

        // psi + gamma * laplace(phi) - (phi^2 - 1) * phi
        state[1] += curr[0] - phobic_en( curr[1] );
#pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ )
        {
            auto N = range[j];

            auto ej = IdxND::make_unit( j );
            auto hj = step[j];

            auto prev = idx[j] == 0u ? cond.left[j] * curr[1] : in.get_vec( idx - ej )[1];
            auto next = idx[j] == N - 1u ? cond.right[j] * curr[1] : in.get_vec( idx + ej )[1];

            state[1] += gamma * ( next + prev - 2 * curr[1] ) / ( hj * hj );
        }

        out.set_vec( state, idx );
    }
};

} // namespace kernels

#endif
