#ifndef __CAHN_HILLIARD_OP_H__
#define __CAHN_HILLIARD_OP_H__

#include "include/boundary.h" // for boundary conditions
#include "kernels/cahn_hilliard_op.h"
#include "kernels/jacobi_op.h"
#include "kernels/phobic_energy.h"
#include "time_derivative.h"

#include <memory>
#include <scfd/static_vec/vec.h>

namespace tests
{

template <
    class VectorSpace,
    class JacobiOperator,
    class Log,
    class PhobicEnergy,
    class Rhs,
    class TimeDerivative,
    /**********************************************/
    class Backend = typename VectorSpace::backend_type>
class cahn_hilliard_op
{
public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;

    using scalar_type        = typename VectorSpace::scalar_type;
    using tensor_type        = scfd::static_vec::vec<scalar_type, tensor_dim>;
    using vector_type        = typename VectorSpace::vector_type;
    using vector_space_type  = VectorSpace;
    using grid_step_type     = scfd::static_vec::vec<scalar_type, dim>;
    using boundary_cond_type = boundary_cond<dim>;
    using ordinal_type       = typename VectorSpace::ordinal_type;

    using Ord = ordinal_type;

    using vector_space_ptr = std::shared_ptr<VectorSpace>;
    using idx_nd_type      = typename VectorSpace::idx_nd_type;

    using for_each_nd_type = typename Backend::template for_each_nd_type<dim>;

    using jacobi_operator_type = JacobiOperator;
    using jacobi_operator_ptr  = std::shared_ptr<JacobiOperator>;

    using time_derivative_ptr = std::shared_ptr<TimeDerivative>;

public: // Especially for SYCL
    using cahn_hilliard_kernel = kernels::cahn_hilliard_op_kernel<
        idx_nd_type,
        scalar_type,
        tensor_type,
        vector_type,
        grid_step_type,
        boundary_cond_type,
        PhobicEnergy,
        Rhs>;

public:
    cahn_hilliard_op( idx_nd_type range, grid_step_type step, boundary_cond_type b_cond, jacobi_operator_ptr jacobi_op )
        : vspace_( std::make_shared<vector_space_type>( range ) ), range_( range ), step_( step ), b_cond_( b_cond ),
          jacobi_op_( jacobi_op ), phobic_en_(), rhs_(), time_derivative_( std::make_shared<TimeDerivative>( range ) )
    {
    }

    cahn_hilliard_op(
        const vector_space_type &vspace, grid_step_type step, boundary_cond_type b_cond, jacobi_operator_ptr jacobi_op
    )
        : cahn_hilliard_op( vspace.get_size(), step, b_cond, jacobi_op )
    {
    }

    cahn_hilliard_op(
        idx_nd_type         range,
        grid_step_type      step,
        boundary_cond_type  b_cond,
        jacobi_operator_ptr jacobi_op,
        time_derivative_ptr time_derivative
    )
        : vspace_( std::make_shared<vector_space_type>( range ) ), range_( range ), step_( step ), b_cond_( b_cond ),
          jacobi_op_( jacobi_op ), phobic_en_(), rhs_(), time_derivative_( time_derivative )
    {
    }

    cahn_hilliard_op(
        const vector_space_type &vspace,
        grid_step_type           step,
        boundary_cond_type       b_cond,
        jacobi_operator_ptr      jacobi_op,
        time_derivative_ptr      time_derivative
    )
        : cahn_hilliard_op( vspace.get_size(), step, b_cond, jacobi_op, time_derivative )
    {
    }

    vector_space_ptr get_space() const
    {
        return std::make_shared<vector_space_type>( range_ );
    }

    idx_nd_type get_size() const noexcept
    {
        return range_;
    }
    grid_step_type get_h() const noexcept
    {
        return step_;
    }
    boundary_cond_type get_b_cond() const noexcept
    {
        return b_cond_;
    }

    vector_space_ptr get_dom_space() const
    {
        return get_space();
    }
    vector_space_ptr get_im_space() const
    {
        return get_space();
    }

    void apply( const vector_type &in, vector_type &out ) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst(
            cahn_hilliard_kernel{
                in,
                out,
                range_,
                step_,
                b_cond_,
                phobic_en_,
                rhs_,
                time_derivative_->get_previous_state(),
                time_derivative_->get_dt_inf(),
                D_,
                gamma_
            },
            range_
        );
    };

    void set_linearization_point( const vector_type &p )
    {
        jacobi_op_->set_vector( p );
    }

    const jacobi_operator_ptr &get_jacobi_operator() const
    {
        return jacobi_op_;
    }

private:
    vector_space_ptr   vspace_;
    idx_nd_type        range_;
    grid_step_type     step_;
    boundary_cond_type b_cond_;

    jacobi_operator_ptr jacobi_op_;
    PhobicEnergy        phobic_en_;
    Rhs                 rhs_;
    time_derivative_ptr time_derivative_;

    scalar_type D_     = 1.0;
    scalar_type gamma_ = 1.0;
};

} // namespace tests

#endif
