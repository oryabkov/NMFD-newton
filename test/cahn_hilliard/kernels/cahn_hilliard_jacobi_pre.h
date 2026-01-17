#ifndef __CAHN_HILLIARD_JACOBI_PRECONDITIONER_H__
#define __CAHN_HILLIARD_JACOBI_PRECONDITIONER_H__

#include <scfd/static_mat/mat.h>

namespace kernels
{

template <class IdxND, class Scalar, class VectorType, class MatType, class GridStep, class BoundaryCond>
struct cahh_hilliard_jacobi_preconditioner
{
    VectorType    v, vector;
    IdxND             range;
    GridStep           step;
    BoundaryCond       cond;
    Scalar                D;
    Scalar            gamma;


    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        MatType mat{
            0., 0.,
            0., 0.
        };

        auto vec = v.get_vec(idx);

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

            mat(0, 0) += D * diag_j / (hj * hj);
            mat(1, 1) += gamma * diag_j / (hj * hj);
        }
        Scalar phi  = vector.get_vec(idx)[1];
        mat(1, 1) -= 3 * phi * phi - 1;
        mat(1, 0) = 1.;

        auto result = inv(mat) * vec;
        v.set_vec(result, idx);
    }
};

}// namespace kernels

#endif
