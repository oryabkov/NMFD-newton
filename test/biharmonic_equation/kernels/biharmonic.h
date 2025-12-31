#ifndef __BIHARMONIC_H__
#define __BIHARMONIC_H__

namespace kernels
{

template <class IdxND, class Scalar, class TensorType, class VectorType, class GridStep, class BoundaryCond>
struct biharmonic_op
{

    VectorType in, out;
    IdxND        range;
    GridStep      step;
    BoundaryCond  cond;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        TensorType biharmonic{0, 0};

        auto curr = in.get_vec(idx);

        // laplace * psi
        #pragma unroll
        for (int j = 0; j < IdxND::dim; j++)
        {
            auto N     = range[j];

            auto ej    = IdxND::make_unit(j);
            auto hj    = step[j];

            auto prev  = idx[j] ==     0 ? cond. left[j] * curr[0] : in.get_vec(idx - ej)[0];
            auto next  = idx[j] == N - 1 ? cond.right[j] * curr[0] : in.get_vec(idx + ej)[0];

            biharmonic[0] += (next + prev - 2 * curr[0]) / (hj * hj);
        }

        // psi + laplace * phi
        biharmonic[1] = curr[0];
        #pragma unroll
        for (int j = 0; j < IdxND::dim; j++)
        {
            auto N     = range[j];

            auto ej    = IdxND::make_unit(j);
            auto hj    = step[j];

            auto prev  = idx[j] ==     0u ? cond. left[j] * curr[1] : in.get_vec(idx - ej)[1];
            auto next  = idx[j] == N - 1u ? cond.right[j] * curr[1] : in.get_vec(idx + ej)[1];

            biharmonic[1] += (next + prev - 2 * curr[1]) / (hj * hj);
        }

        out.set_vec(biharmonic, idx);
    }
};

}// namespace kernels

#endif
