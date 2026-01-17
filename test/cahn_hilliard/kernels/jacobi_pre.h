#ifndef __JACOBI_PRE_H__
#define __JACOBI_PRE_H__

#include <scfd/static_mat/mat.h>

namespace kernels
{

template <class IdxND, class Scalar, class VectorType, class MatType, class GridStep, class BoundaryCond>
struct jacobi_preconditioner
{

    VectorType       v;
    IdxND        range;
    GridStep      step;
    BoundaryCond  cond;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        MatType mat{
            0., 0.,
            0., 0.
        };

        #pragma unroll
        for (int j = 0; j < IdxND::dim; j++)
        {
            const auto N     = range[j];
            const auto hj    = step[j];

            Scalar diag_j {-2.};

            if (idx[j] == 0) {
                diag_j += cond.left[j];
            }

            if (idx[j] == N - 1) {
                diag_j += cond.right[j];
            }

            mat(0, 0) += diag_j / (hj * hj);
            mat(1, 1) += diag_j / (hj * hj);
        }
        mat(1, 0) = 1.;

        auto vec = v.get_vec(idx);
        auto result = inv(mat) * vec;
        v.set_vec(result, idx);
    }
};

}// namespace kernels

#endif
