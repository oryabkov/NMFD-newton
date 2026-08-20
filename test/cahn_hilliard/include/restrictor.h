#ifndef __RESTRICTOR_H__
#define __RESTRICTOR_H__

#include <memory>

#include "kernels/restrictor.h"
#include "boundary.h"
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/utils/profiling.h>
#include <scfd/static_vec/vec.h>

namespace tests
{

template <
    class VectorSpace, class Log, class Distributor,
    /**********************************************/
    class Comm    = typename VectorSpace::comm_type,
    class Backend = typename VectorSpace::backend_type>
class restrictor
{
public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;
    using scalar_type           = typename VectorSpace::scalar_type;
    using vector_type           = typename VectorSpace::vector_type;
    using vector_space_type     = VectorSpace;
    using comm_type             = Comm;
    using dist_type             = Distributor;
    using dist_ptr              = std::shared_ptr<const dist_type>;
    using ordinal_type          = typename VectorSpace::ordinal_type;
    using grid_step_type        = scfd::static_vec::vec<scalar_type, dim>;

    using Ord = ordinal_type;

    using vector_space_ptr = std::shared_ptr<VectorSpace>;
    using idx_nd_type      = typename VectorSpace::idx_nd_type;

    using boundary_cond_type = boundary_cond<VectorSpace>;

    using for_each_nd_type = typename Backend::template for_each_nd_type<dim>;

public: // Especially for SYCL
    using restrictor_kernel =
        kernels::restrictor_kernel<idx_nd_type, ordinal_type, vector_type, tensor_dim, boundary_cond_type, grid_step_type>;
    using rect_type = typename restrictor_kernel::Rect;

public:
    restrictor(
        idx_nd_type range, grid_step_type step, boundary_cond_type b_cond, const comm_type &comm, dist_ptr dist,
        ordinal_type stencil, int max_stencil_order )
        : range_( range ), step_( step ), b_cond_( b_cond ), dist_( std::move( dist ) ),
          vspace_( std::make_shared<vector_space_type>( range, comm, false, stencil, max_stencil_order ) ),
          lin_vector_wrap_( *vspace_ )
    {
        for ( int i = 0; i < idx_nd_type::dim; ++i )
        {
            if ( range[i] % 2u != 0 )
                throw std::logic_error(
                    "nmfd::restrictor: encountered odd value in vector_space range_! not supported case"
                );
        }
        vspace_->assign_scalar( 0.0, *lin_vector_wrap_ );
    }

    idx_nd_type get_size() const noexcept
    {
        return range_;
    }

    grid_step_type get_h() const noexcept
    {
        return step_;
    }

    vector_space_ptr get_dom_space() const
    {
        return vector_space_ptr( range_ / Ord{ 2u } );
    }

    vector_space_ptr get_im_space() const
    {
        return vector_space_ptr( range_ );
    }

    void set_linearization_point( const vector_type &p )
    {
        vspace_->assign( p, *lin_vector_wrap_ );
        dist_->sync( *lin_vector_wrap_ );
    }

    vector_type get_lin_vector() const
    {
        return *lin_vector_wrap_;
    }

    void set_b_cond( const boundary_cond_type &b_cond )
    {
        b_cond_ = b_cond;
    }

    // domain -> (restrict) -> image
    void apply( vector_type &from, vector_type &to, bool use_linearized_ghost = true ) const
    {
        // The stencil reaches two cells beyond the block, so the neighbours' values have to be in
        // the halo before the sweep starts.
        {
            SCFD_PLATFORM_SCOPED_TIC( "Comm::sync" );
            dist_->sync( from );
        }

        // The computed region, not the allocation: everything outside it goes through the boundary
        // condition, which is what tells a halo cell apart from a physical boundary.
        rect_type        dom_r{ idx_nd_type::make_zero(), range_ };
        auto             half_r = range_ / Ord{ 2u };
        {
            SCFD_PLATFORM_SCOPED_TIC( "Restrictor::apply" );
            for_each_nd_type for_each_nd_inst;
            for_each_nd_inst( restrictor_kernel{ from, to, *lin_vector_wrap_, b_cond_, step_, dom_r, use_linearized_ghost }, half_r );
        }
    };

private:
    idx_nd_type        range_; // in dom space
    grid_step_type     step_;
    boundary_cond_type b_cond_;
    dist_ptr           dist_;

    vector_space_ptr vspace_;
    using vector_wrap_t = nmfd::detail::vector_wrap<VectorSpace, true, true>;
    vector_wrap_t    lin_vector_wrap_;
};

} // namespace tests

#endif
