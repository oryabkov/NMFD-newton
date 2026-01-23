#ifndef __JACOBI_OP_H__
#define __JACOBI_OP_H__

#include "include/boundary.h" // for boundary conditions
#include "kernels/jacobi_op.h"

#include <memory>
#include <scfd/static_vec/vec.h>
#include <nmfd/detail/vector_wrap.h>

namespace tests
{

template <
    class VectorSpace,
    class Log,
    class PhobicEnergy,
    class TimeDerivative,
    /**********************************************/
    class Backend = typename VectorSpace::backend_type>
class jacobi_op
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

    using time_derivative_ptr = std::shared_ptr<TimeDerivative>;

public: // Especially for SYCL
    using jacobi_op_kernel = kernels::jacobi_op_kernel<
        idx_nd_type,
        scalar_type,
        tensor_type,
        vector_type,
        grid_step_type,
        boundary_cond_type,
        PhobicEnergy>;

public:
    jacobi_op( idx_nd_type range, grid_step_type step, boundary_cond_type b_cond )
        : vspace_( std::make_shared<vector_space_type>( range ) ), range_( range ), step_( step ), b_cond_( b_cond ),
          vector_wrap_( *vspace_ ), phobic_en_(), time_derivative_( std::make_shared<TimeDerivative>(range) )
    {
        vspace_->assign_scalar( 0.0, *vector_wrap_ );
    }

    jacobi_op( const vector_space_type &vspace, grid_step_type step, boundary_cond_type b_cond, vector_type vector_)
        : jacobi_op( vspace.get_size(), step, b_cond, vector_ )
    {
    }

    jacobi_op( idx_nd_type range, grid_step_type step, boundary_cond_type b_cond, time_derivative_ptr time_derivative )
        : vspace_( std::make_shared<vector_space_type>( range ) ), range_( range ), step_( step ), b_cond_( b_cond ),
          vector_wrap_( *vspace_ ), phobic_en_(), time_derivative_( time_derivative )
    {
        vspace_->assign_scalar( 0.0, *vector_wrap_ );
    }

    jacobi_op( const vector_space_type &vspace, grid_step_type step, boundary_cond_type b_cond, time_derivative_ptr time_derivative)
        : jacobi_op( vspace.get_size(), step, b_cond, time_derivative )
    {
    }

    vector_space_ptr get_space() const
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
        return b_cond_;
    }
    scalar_type get_D() const noexcept
    {
        return D_;
    }
    scalar_type get_gamma() const noexcept
    {
        return gamma_;
    }

    void set_D( scalar_type D )
    {
        D_ = D;
    }
    void set_gamma( scalar_type gamma )
    {
        gamma_ = gamma;
    }

    vector_space_ptr get_dom_space() const
    {
        return get_space();
    }
    vector_space_ptr get_im_space() const
    {
        return get_space();
    }

    vector_type get_vector() const
    {
        return *vector_wrap_;
    }
    void set_vector( const vector_type &vector )
    {
        vspace_->assign( vector, *vector_wrap_ );
    }

    time_derivative_ptr get_time_derivative() const
    {
        return  time_derivative_;
    }

    void apply( const vector_type &in, vector_type &out ) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst(
            jacobi_op_kernel{ in, out, *vector_wrap_, range_, step_, b_cond_, phobic_en_, time_derivative_->get_dt_inf(), D_, gamma_ }, range_
        );
    };

private:
    vector_space_ptr   vspace_;
    idx_nd_type        range_;
    grid_step_type     step_;
    boundary_cond_type b_cond_;

    using vector_wrap_t = nmfd::detail::vector_wrap<VectorSpace, true, true>;
    vector_wrap_t       vector_wrap_;
    PhobicEnergy        phobic_en_;
    time_derivative_ptr time_derivative_;

    scalar_type D_     = scalar_type( 1 );
    scalar_type gamma_ = scalar_type( 1 );
};

} // namespace tests

#endif
