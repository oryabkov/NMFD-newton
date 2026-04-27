#ifndef __JACOBI_PRE_H__
#define __JACOBI_PRE_H__

#include "jacobi_op.h"
#include "kernels/jacobi_pre.h"
#include "kernels/mobility.h"

#include <memory>
#include <scfd/static_vec/vec.h>
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
    class Mobility,
    /**********************************************/
    class LinOp   = jacobi_op<VectorSpace, Log, PhobicEnergy, TimeDerivative, Mobility>,
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
    using tensor_type  = scfd::static_vec::vec<scalar_type, tensor_dim>;
    using vector_type = typename VectorSpace::vector_type;
    using idx_nd_type = typename VectorSpace::idx_nd_type;

    using for_each_nd_type = typename Backend::template for_each_nd_type<dim>;

    using grid_step_type     = scfd::static_vec::vec<scalar_type, dim>;
    using boundary_cond_type = boundary_cond<VectorSpace>;
    using mat_type           = scfd::static_mat::mat<scalar_type, tensor_dim, tensor_dim>;

    using time_derivative_ptr = std::shared_ptr<TimeDerivative>;

public: // Especially for SYCL
    using preconditioner_kernel = kernels::jacobi_pre_kernel<
        idx_nd_type,
        scalar_type,
        tensor_type,
        vector_type,
        mat_type,
        grid_step_type,
        boundary_cond_type,
        PhobicEnergy,
        Mobility>;

public:
    struct params
    {
        scalar_type alpha;

        params( const std::string &log_prefix = "", const std::string &log_name = "smoother_elliptic::" ) :
            alpha(0.71)
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
        : vspace_( std::move(vspace) ), range_( vspace_->get_size() ), step_( step ), b_cond_( std::make_unique<boundary_cond_type>( b_cond ) ),
          lin_vector_wrap_( std::make_unique<vector_wrap_t>( *vspace_ ) ), phobic_en_(), time_derivative_( std::make_shared<TimeDerivative>(vspace_) )
    {
        vspace_->assign_scalar( 0.0, *lin_vector_wrap_ );
    }

    jacobi_pre(
        vector_space_ptr vspace, grid_step_type step, boundary_cond_type b_cond, time_derivative_ptr time_derivative
    )
        : vspace_( std::move(vspace) ), range_( vspace_->get_size() ), step_( step ), b_cond_( std::make_unique<boundary_cond_type>( b_cond ) ),
          lin_vector_wrap_( std::make_unique<vector_wrap_t>( *vspace_ ) ), phobic_en_(), time_derivative_( std::move(time_derivative) )
    {
        vspace_->assign_scalar( 0.0, *lin_vector_wrap_ );
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
        b_cond_ = std::make_unique<boundary_cond_type>( op->get_b_cond() );
        gamma_  = op->get_gamma();
        mobility_ = op->get_mobility();

        // Always recreate lin_vector_wrap_ with the current VectorSpace
        lin_vector_wrap_ = std::make_unique<vector_wrap_t>( *vspace_ );
        vspace_->assign( op->get_lin_vector(), **lin_vector_wrap_ );

        time_derivative_ = op->get_time_derivative();
    }

public:
    const vector_space_ptr &get_space() const noexcept
    {
        return vspace_;
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
        return *b_cond_;
    }

    scalar_type get_gamma() const noexcept
    {
        return gamma_;
    }
    void set_mobility( const Mobility &mobility )
    {
        mobility_ = mobility;
    }
    void set_gamma( scalar_type gamma )
    {
        gamma_ = gamma;
    }

    const vector_space_ptr &get_dom_space() const noexcept
    {
        return get_space();
    }
    const vector_space_ptr &get_im_space() const noexcept
    {
        return get_space();
    }

    void apply( vector_type &vector ) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst(
            preconditioner_kernel{
                vector,
                **lin_vector_wrap_,
                range_,
                step_,
                *b_cond_,
                phobic_en_,
                mobility_,
                time_derivative_->get_dt_inf(),
                params_.alpha,
                gamma_
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
    std::unique_ptr<boundary_cond_type> b_cond_;

    using vector_wrap_t = nmfd::detail::vector_wrap<VectorSpace, true, true>;
    std::unique_ptr<vector_wrap_t> lin_vector_wrap_;
    PhobicEnergy                   phobic_en_;
    Mobility                       mobility_;
    time_derivative_ptr            time_derivative_;

    scalar_type gamma_ = scalar_type( 1 );

    params_hierarchy params_;
};

} // namespace tests

#endif
