#ifndef __PROLONGATOR_KERNEL_H__
#define __PROLONGATOR_KERNEL_H__

#include <scfd/utils/device_tag.h>

namespace kernels
{

/**
 * Trilinear interpolation prolongator kernel for cell-centered multigrid.
 *
 * For cell-centered grids, the interpolation weights are:
 * - Fine cell at 2*ic:   interpolates from coarse(ic-1) with w=1/4 and coarse(ic) with w=3/4
 * - Fine cell at 2*ic+1: interpolates from coarse(ic) with w=3/4 and coarse(ic+1) with w=1/4
 *
 * This produces a smooth transition that is appropriate for Laplace-based operators
 * and significantly improves multigrid convergence compared to piecewise constant.
 */
template <class IdxND, class Ord, class VectorType, int TensorDim, class ScalarType>
struct prolongator_kernel
{
    static constexpr int dim = IdxND::dim;

    VectorType  dom, img;
    IdxND       coarse_range;

    __DEVICE_TAG__ void operator()( const IdxND fine_idx ) const
    {
        // Arrays to hold the two contributing coarse indices and weights per dimension
        Ord        c_idx[dim][2];
        ScalarType w[dim][2];

        for ( int d = 0; d < dim; d++ )
        {
            Ord ic = fine_idx[d] / Ord{ 2 };
            Ord di = fine_idx[d] % Ord{ 2 };

            if ( di == Ord{ 0 } )
            {
                // Fine point at 2*ic: interpolate from ic-1 (w=0.25) and ic (w=0.75)
                Ord ic_low = ic - Ord{ 1 };
                if ( ic_low < Ord{ 0 } )
                {
                    // Left boundary: use only ic with full weight
                    c_idx[d][0] = ic;
                    c_idx[d][1] = ic;
                    w[d][0]     = ScalarType{ 1 };
                    w[d][1]     = ScalarType{ 0 };
                }
                else
                {
                    c_idx[d][0] = ic_low;
                    c_idx[d][1] = ic;
                    w[d][0]     = ScalarType{ 0.25 };
                    w[d][1]     = ScalarType{ 0.75 };
                }
            }
            else
            {
                // Fine point at 2*ic+1: interpolate from ic (w=0.75) and ic+1 (w=0.25)
                Ord ic_high = ic + Ord{ 1 };
                if ( ic_high >= coarse_range[d] )
                {
                    // Right boundary: use only ic with full weight
                    c_idx[d][0] = ic;
                    c_idx[d][1] = ic;
                    w[d][0]     = ScalarType{ 1 };
                    w[d][1]     = ScalarType{ 0 };
                }
                else
                {
                    c_idx[d][0] = ic;
                    c_idx[d][1] = ic_high;
                    w[d][0]     = ScalarType{ 0.75 };
                    w[d][1]     = ScalarType{ 0.25 };
                }
            }
        }

        // Trilinear interpolation: sum over all 2^dim corners of the interpolation stencil
        for ( int t = 0; t < TensorDim; t++ )
        {
            ScalarType val = ScalarType{ 0 };

            // Iterate over all 2^dim combinations (8 in 3D)
            for ( int corner = 0; corner < ( 1 << dim ); corner++ )
            {
                IdxND      coarse_idx;
                ScalarType weight = ScalarType{ 1 };

                for ( int d = 0; d < dim; d++ )
                {
                    int sel        = ( corner >> d ) & 1;
                    coarse_idx[d]  = c_idx[d][sel];
                    weight        *= w[d][sel];
                }

                if ( weight > ScalarType{ 0 } )
                {
                    val += weight * dom( coarse_idx, t );
                }
            }

            img( fine_idx, t ) = val;
        }
    }
};

} // namespace kernels

#endif
