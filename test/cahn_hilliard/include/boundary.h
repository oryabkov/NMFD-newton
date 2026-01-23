#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__

template <int Dim, int TensorDim>
struct boundary_cond
{
    // -1 for dirichlet
    // +1 for neumann
    using conditions = int;

    conditions left [Dim][TensorDim];
    conditions right[Dim][TensorDim];
};

#endif
