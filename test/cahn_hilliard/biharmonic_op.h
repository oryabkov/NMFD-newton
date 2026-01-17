#ifndef __BIHARMONIC_OP_H__
#define __BIHARMONIC_OP_H__

#include <memory>

#include "include/boundary.h" // for boundary conditions
#include "kernels/biharmonic.h"

namespace tests
{

template
<
    class VectorSpace, class Log,
    /**********************************************/
    class Backend=typename VectorSpace::backend_type
>
class biharmonic_op
{
public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;

    using scalar_type           = typename VectorSpace::scalar_type;
    using tensor_type           = scfd::static_vec::vec<scalar_type, tensor_dim>;
    using vector_type           = typename VectorSpace::vector_type;
    using vector_space_type     = VectorSpace;
    using grid_step_type        = scfd::static_vec::vec<scalar_type, dim>;
    using boundary_cond_type    = boundary_cond<dim>;
    using ordinal_type          = typename VectorSpace::ordinal_type;

    using Ord = ordinal_type;

    using vector_space_ptr   = std::shared_ptr<VectorSpace>;
    using idx_nd_type        = typename VectorSpace::idx_nd_type;

    using for_each_nd_type   = typename Backend::template for_each_nd_type<dim>;

public: // Especially for SYCL
    using biharmonic_kernel = kernels::biharmonic
    <
        idx_nd_type, scalar_type, tensor_type, vector_type,
        grid_step_type, boundary_cond_type
    >;
private:
    vector_space_ptr     vspace;
    idx_nd_type          range;
    grid_step_type       step;
    boundary_cond_type   b_cond;
public:

    biharmonic_op(
        idx_nd_type r,
        grid_step_type grid_step,
        boundary_cond_type cond
    ):
        vspace(std::make_shared<vector_space_type>(r)),
        range(r), step(grid_step), b_cond(cond)
    {}

    biharmonic_op(
        const vector_space_type& vec_space,
        grid_step_type grid_step,
        boundary_cond_type cond
    ): biharmonic_op(vec_space.get_size(), grid_step, cond) {}

    vector_space_ptr        get_space()  const
    {
        return std::make_shared<vector_space_type>(range);
    }

    idx_nd_type             get_size()   const noexcept { return range;  }
    grid_step_type          get_h()      const noexcept { return step;   }
    boundary_cond_type      get_b_cond() const noexcept { return b_cond; }

    vector_space_ptr get_dom_space() const { return get_space(); }
    vector_space_ptr get_im_space()  const { return get_space(); }

    void apply(const vector_type &in, vector_type &out) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst(biharmonic_kernel{in, out, range, step, b_cond}, range);
    };
};

}// namespace tests

#endif
