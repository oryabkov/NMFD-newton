#ifndef __DEVICE_JACOBI_PRE_H__
#define __DEVICE_JACOBI_PRE_H__

#include <memory>
#include <nmfd/preconditioners/preconditioner_interface.h>
#include <scfd/static_mat/mat.h>

#include "biharmonic_op.h"
#include "kernels/jacobi_pre.h"

namespace tests
{

template
<
    class VectorSpace, class Log,
    /**********************************************/
    class LinOp=biharmonic_op<VectorSpace, Log>,
    class Backend=typename VectorSpace::backend_type
>
class jacobi_pre : public nmfd::preconditioners::preconditioner_interface<VectorSpace,LinOp>
{
    using lin_op_t              = LinOp;
public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;
    using vector_space_type     = VectorSpace; // defines Vector Space working in
    using vector_space_ptr      = std::shared_ptr<VectorSpace>;

    using scalar_type           = typename VectorSpace::scalar_type;
    using ordinal_type          = typename VectorSpace::ordinal_type;

    using vector_type           = typename VectorSpace::vector_type;
    using idx_nd_type           = typename VectorSpace::idx_nd_type;

    using for_each_nd_type      = typename Backend::template for_each_nd_type<dim>;

    using grid_step_type        = scfd::static_vec::vec<scalar_type, dim>;
    using boundary_cond_type    = boundary_cond<dim>;
    using mat_type              = scfd::static_mat::mat<scalar_type, tensor_dim, tensor_dim>;

public: // Especially for SYCL
    using preconditioner_kernel = kernels::jacobi_preconditioner
    <
        idx_nd_type, scalar_type, vector_type, mat_type, grid_step_type, boundary_cond_type
    >;

private:
    vector_space_ptr     vspace;
    idx_nd_type          range;
    grid_step_type       step;
    boundary_cond_type   b_cond;

public:
    struct params
    {
        params(const std::string &log_prefix = "",
               const std::string &log_name   = "smoother_elliptic::") {}
    };
    using params_hierarchy = params;
    struct utils {};
    using utils_hierarchy = utils;

    jacobi_pre(const utils_hierarchy &u, const params_hierarchy &p) {}

    jacobi_pre(
        vector_space_ptr   vec_space,
        grid_step_type     grid_step,
        boundary_cond_type cond
    ):
        vspace(vec_space),
        range(vspace->get_range()),
        step(grid_step), b_cond(cond) {}

    jacobi_pre(std::shared_ptr<const lin_op_t> op){ set_operator(op); }

    void set_operator(std::shared_ptr<const lin_op_t> op)
    {
        vspace = op->get_space();
        range  = op->get_size();
        step   = op->get_h();
        b_cond = op->get_b_cond();
    }

public:
    vector_space_ptr        get_space()  const
    {
        return std::make_shared<vector_space_type>(range);
    }

    idx_nd_type             get_size()   const noexcept { return range;  }
    grid_step_type          get_h()      const noexcept { return step;   }
    boundary_cond_type      get_b_cond() const noexcept { return b_cond; }

    vector_space_ptr get_dom_space() const { return get_space(); }
    vector_space_ptr get_im_space()  const { return get_space(); }

    void apply(vector_type &v) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst(preconditioner_kernel{v, range, step, b_cond}, range);
    };
    void apply(const vector_type &x,vector_type &y) const
    {
        vspace->assign(x,y);
        apply(y);
    };
};

}// namespace tests

#endif
