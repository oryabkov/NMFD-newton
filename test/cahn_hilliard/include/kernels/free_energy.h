#ifndef __FREE_ENERGY_KERNEL_H__
#define __FREE_ENERGY_KERNEL_H__

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
struct free_energy_kernel
{
    VectorType   in, out;
    IdxND        range;
    GridStep     step;
    BoundaryCond cond;
    PhobicEnergy phobic_en;
    Scalar       gamma;
    Scalar       cell_volume;

    __DEVICE_TAG__ __DEVICE_TAG__ void operator()( const IdxND idx ) const
    {
        auto curr = in.get_vec( idx );

        Scalar grad_sq = Scalar( 0 );
        TensorType neighbor{};
        #pragma unroll
        for ( int j = 0; j < IdxND::dim; j++ ) // iterate over x, y, z,... dimension
        {
            auto N = range[j];

            auto ej = IdxND::make_unit( j );
            auto hj = step[j];

            Scalar prev_phi;
            if ( idx[j] == 0 )
            {
                cond.get_lin_neighbor( in, range, idx - ej, j, /*is_left*/ true, step, neighbor );
                prev_phi = neighbor[1];
            }
            else
            {
                prev_phi = in.get_vec( idx - ej )[1];
            }

            Scalar next_phi;
            if ( idx[j] == N - 1 )
            {
                cond.get_lin_neighbor( in, range, idx + ej, j, /*is_left*/ false, step, neighbor );
                next_phi = neighbor[1];
            }
            else
            {
                next_phi = in.get_vec( idx + ej )[1];
            }

            const Scalar grad_j = ( next_phi - prev_phi ) / Scalar( 2 * hj );
            grad_sq += grad_j * grad_j;
        }

        // Reuse the (psi,phi)-shaped tensor slot to carry (phobic, philic) energy densities.
        TensorType state;
        state[0] = phobic_en.get_energy( curr[1] ) * cell_volume;
        state[1] = Scalar( 0.5 ) * gamma * grad_sq * cell_volume;

        out.set_vec( state, idx );
    }
};

} // namespace kernels

#endif
