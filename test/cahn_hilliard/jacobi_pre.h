#ifndef __JACOBI_PRE_H__
#define __JACOBI_PRE_H__

#include "jacobi_op.h"
#include "kernels/jacobi_pre.h"

#include <memory>
#include <nmfd/preconditioners/preconditioner_interface.h>
#include <scfd/static_mat/mat.h>
#include <nmfd/detail/vector_wrap.h>

namespace tests
{

template <
    class VectorSpace,
    class Log,
    class PhobicEnergy,
    class TimeDerivative,
    /**********************************************/
    class LinOp   = jacobi_op<VectorSpace, Log, PhobicEnergy, TimeDerivative>,
    class Backend = typename VectorSpace::backend_type>
class jacobi_pre : public nmfd::preconditioners::preconditioner_interface<VectorSpace, LinOp>
{
    using lin_op_t = LinOp;

public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;
    using vector_space_type     = VectorSpace; // defines Vector Space working in
    using vector_space_ptr      = std::shared_ptr<VectorSpace>;

    using scalar_type  = typename VectorSpace::scalar_type;
    using ordinal_type = typename VectorSpace::ordinal_type;

    using vector_type = typename VectorSpace::vector_type;
    using idx_nd_type = typename VectorSpace::idx_nd_type;

    using for_each_nd_type = typename Backend::template for_each_nd_type<dim>;

    using grid_step_type     = scfd::static_vec::vec<scalar_type, dim>;
    using boundary_cond_type = boundary_cond<dim>;
    using mat_type           = scfd::static_mat::mat<scalar_type, tensor_dim, tensor_dim>;

    using time_derivative_ptr = std::shared_ptr<TimeDerivative>;

public: // Especially for SYCL
    using preconditioner_kernel = kernels::jacobi_pre_kernel<
        idx_nd_type,
        scalar_type,
        vector_type,
        mat_type,
        grid_step_type,
        boundary_cond_type,
        PhobicEnergy>;

public:
    struct params
    {
        params( const std::string &log_prefix = "", const std::string &log_name = "smoother_elliptic::" )
        {
        }
    };
    using params_hierarchy = params;
    struct utils
    {
    };
    using utils_hierarchy = utils;

    jacobi_pre( const utils_hierarchy &u, const params_hierarchy &p )
    {
    }

    jacobi_pre( vector_space_ptr vspace, grid_step_type step, boundary_cond_type b_cond)
        : vspace_( vspace ), range_( vspace_->get_range() ), step_( step ), b_cond_( b_cond ),
          vector_wrap_( std::make_unique<vector_wrap_t>( *vspace ) ), phobic_en_(), time_derivative_( std::make_shared<TimeDerivative>(vspace) )
    {
        vspace_->assign_scalar( 0.0, *vector_wrap_ );
    }

    jacobi_pre(
        vector_space_ptr vspace, grid_step_type step, boundary_cond_type b_cond, time_derivative_ptr time_derivative
    )
        : vspace_( vspace ), range_( vspace_->get_range() ), step_( step ), b_cond_( b_cond ),
          vector_wrap_( std::make_unique<vector_wrap_t>( *vspace ) ), phobic_en_(), time_derivative_( time_derivative )
    {
        vspace_->assign_scalar( 0.0, *vector_wrap_ );
    }

    jacobi_pre( std::shared_ptr<const lin_op_t> op )
    {
        set_operator( op );
    }

    void set_operator( std::shared_ptr<const lin_op_t> op )
    {
        vspace_ = op->get_space();
        range_  = op->get_size();
        step_   = op->get_h();
        b_cond_ = op->get_b_cond();
        D_      = op->get_D();
        gamma_  = op->get_gamma();

        // Always recreate vector_wrap_ with the current VectorSpace
        vector_wrap_ = std::make_unique<vector_wrap_t>( *vspace_ );
        vspace_->assign( op->get_vector(), **vector_wrap_ );

        time_derivative_ = op->get_time_derivative();
    }

public:
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

    void apply( vector_type &v ) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst(
            preconditioner_kernel{
                v, **vector_wrap_, range_, step_, b_cond_, phobic_en_, time_derivative_->get_dt_inf(), D_, gamma_
            },
            range_
        );
    };
    void apply( const vector_type &x, vector_type &y ) const
    {
        vspace_->assign( x, y );
        apply( y );
    };

private:
    vector_space_ptr   vspace_;
    idx_nd_type        range_;
    grid_step_type     step_;
    boundary_cond_type b_cond_;

    using vector_wrap_t = nmfd::detail::vector_wrap<VectorSpace, true, true>;
    std::unique_ptr<vector_wrap_t> vector_wrap_;
    PhobicEnergy                   phobic_en_;
    time_derivative_ptr            time_derivative_;

    scalar_type D_     = scalar_type( 1 );
    scalar_type gamma_ = scalar_type( 1 );
};

} // namespace tests

#endif
