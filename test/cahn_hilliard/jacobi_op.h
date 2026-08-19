#ifndef __JACOBI_OP_H__
#define __JACOBI_OP_H__

#include "include/boundary.h" // for boundary conditions
#include "kernels/jacobi_op.h"
#include "kernels/mobility.h"

#include <memory>
#include <scfd/static_vec/vec.h>
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/utils/profiling.h>

namespace tests
{

template <
    class VectorSpace,
    class Log,
    class PhobicEnergy,
    class TimeDerivative,
    class Mobility,
    class Distributor,
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
    using boundary_cond_type = boundary_cond<VectorSpace>;
    using ordinal_type       = typename VectorSpace::ordinal_type;
    using dist_type          = Distributor;
    using dist_ptr           = std::shared_ptr<const dist_type>;

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
        PhobicEnergy,
        Mobility>;

public:
    jacobi_op( vector_space_ptr vspace, grid_step_type step, boundary_cond_type b_cond, dist_ptr dist, time_derivative_ptr time_derivative )
        : vspace_( std::move( vspace ) ), range_( vspace_->get_size() ), step_( step ), b_cond_( b_cond ),
          lin_vector_wrap_( *vspace_ ), phobic_en_(), dist_( std::move(dist) ), time_derivative_( std::move( time_derivative ) )
    {
        vspace_->assign_scalar( 0.0, *lin_vector_wrap_ );
    }

    jacobi_op( idx_nd_type range, grid_step_type step, boundary_cond_type b_cond, dist_ptr dist )
        : jacobi_op(
              std::make_shared<vector_space_type>( range ), step, b_cond, dist, std::make_shared<TimeDerivative>( range )
          )
    {
    }

    jacobi_op( idx_nd_type range, grid_step_type step, boundary_cond_type b_cond, dist_ptr dist, time_derivative_ptr time_derivative )
        : jacobi_op(
              std::make_shared<vector_space_type>( range ), step, b_cond, std::move( dist ), std::move( time_derivative )
          )
    {
    }

    jacobi_op( vector_space_ptr vspace, grid_step_type step, boundary_cond_type b_cond, dist_ptr dist )
        : jacobi_op( vspace, step, b_cond, std::move( dist ), std::make_shared<TimeDerivative>( vspace ) )
    {
    }

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
        return b_cond_;
    }
    scalar_type get_gamma() const noexcept
    {
        return gamma_;
    }
    Mobility get_mobility() const noexcept
    {
        return mobility_;
    }

    void set_mobility( const Mobility &mobility )
    {
        mobility_ = mobility;
    }
    void set_gamma( scalar_type gamma )
    {
        gamma_ = gamma;
    }

    void set_distributor( dist_ptr dist )
    {
        dist_ = std::move( dist );
    }

    const dist_ptr &get_distributor() const noexcept
    {
        return dist_;
    }

    const vector_space_ptr &get_dom_space() const noexcept
    {
        return get_space();
    }
    const vector_space_ptr &get_im_space() const noexcept
    {
        return get_space();
    }

    void set_linearization_point( const vector_type &p )
    {
        vspace_->assign( p, *lin_vector_wrap_ );
    }

    vector_type get_lin_vector() const
    {
        return *lin_vector_wrap_;
    }

    const time_derivative_ptr &get_time_derivative() const noexcept
    {
        return time_derivative_;
    }

    void apply( const vector_type &in, vector_type &out ) const
    {
        // Synchronized all data between processes between calling foreach
        {
            SCFD_PLATFORM_SCOPED_TIC( "Comm::sync" );
            dist_->sync( in );
            dist_->sync( *lin_vector_wrap_ );
        }

        {
            SCFD_PLATFORM_SCOPED_TIC( "Operator::apply" );
            for_each_nd_type for_each_nd_inst;
            for_each_nd_inst(
                jacobi_op_kernel{
                    in,
                    out,
                    *lin_vector_wrap_,
                    range_,
                    step_,
                    b_cond_,
                    phobic_en_,
                    mobility_,
                    time_derivative_->get_dt_inf(),
                    gamma_
                },
                range_
            );
        }
    };

private:
    vector_space_ptr   vspace_;
    idx_nd_type        range_;
    grid_step_type     step_;
    boundary_cond_type b_cond_;

    using vector_wrap_t = nmfd::detail::vector_wrap<VectorSpace, true, true>;
    vector_wrap_t       lin_vector_wrap_;
    PhobicEnergy        phobic_en_;
    Mobility            mobility_;
    time_derivative_ptr time_derivative_;

    scalar_type gamma_ = scalar_type( 1 );

    dist_ptr dist_;
};

} // namespace tests

#endif
