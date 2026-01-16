#ifndef __VECTOR_SPACE_H__
#define __VECTOR_SPACE_H__

#include <cstddef>
#include <utility>

#include <scfd/utils/device_tag.h>

/************************************************************
 * Backend independent kernels implementing vector operations
 * to run on heterogenous devices.
 ************************************************************/

namespace kernels
{

template <class Scalar, class ValueType>
struct assign_scalar
{
    Scalar s;
    ValueType* x;

    template <class Idx>
    __DEVICE_TAG__ void operator()(const Idx idx)
    {
        x[idx] = s;
    }
};

} // namespace kernels

#endif
