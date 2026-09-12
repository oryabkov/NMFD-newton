#ifndef __FREE_ENERGY_H__
#define __FREE_ENERGY_H__

#include "boundary.h"
#include "kernels/free_energy.h"

#include <memory>
#include <nmfd/detail/vector_wrap.h>
#include <scfd/utils/profiling.h>
#include <scfd/static_vec/vec.h>

namespace tests
{

template <
    class VectorSpace,
    class PhobicEnergy,
    class Distributor,
    /**********************************************/
    class Backend = typename VectorSpace::backend_type>
class free_energy
{
public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;

    using scalar_type        = typename VectorSpace::scalar_type;
    using tensor_type        = scfd::static_vec::vec<scalar_type, tensor_dim>;
    using vector_type        = typename VectorSpace::vector_type;
    using vector_space_type  = VectorSpace;
    using grid_step_type     = scfd::static_vec::vec<scalar_type, dim>;
    using boundary_cond_type = boundary_cond<vector_space_type>;
    using idx_nd_type        = typename VectorSpace::idx_nd_type;
    using view_type          = typename vector_type::view_type;
    using dist_type          = Distributor;
    using dist_ptr           = std::shared_ptr<const dist_type>;

    using vector_space_ptr = std::shared_ptr<VectorSpace>;

    using for_each_nd_type = typename Backend::template for_each_nd_type<dim>;

public: // Especially for SYCL
    using free_energy_kernel = kernels::free_energy_kernel<
        idx_nd_type, scalar_type, tensor_type, vector_type, grid_step_type, boundary_cond_type, PhobicEnergy>;

    struct energies
    {
        scalar_type phobic;
        scalar_type philic;
    };

public:
    free_energy( vector_space_ptr vspace, grid_step_type step, boundary_cond_type b_cond, dist_ptr dist, PhobicEnergy phobic_en, scalar_type gamma )
        : vspace_( std::move( vspace ) ), range_( vspace_->get_size() ), step_( step ), b_cond_( b_cond ),
          dist_( std::move( dist ) ), phobic_en_( std::move( phobic_en ) ), gamma_( gamma ), density_( *vspace_ ), e0_( *vspace_ )
    {
        view_type e0_view( *e0_, false );
        for ( int i = 0; i < range_[0]; i++ )
        {
            for ( int j = 0; j < range_[1]; j++ )
            {
                for ( int k = 0; k < range_[2]; k++ )
                {
                    e0_view( i, j, k, 0 ) = scalar_type( 1 );
                    e0_view( i, j, k, 1 ) = scalar_type( 0 );
                }
            }
        }
        e0_view.release();
    }

    energies compute( const vector_type &state ) const
    {
        {
            SCFD_PROFILING_SCOPED_TIC( "Comm::sync" );
            dist_->sync( state );
        }

        scalar_type cell_volume = step_.components_prod();
        {
            SCFD_PROFILING_SCOPED_TIC( "FreeEnergy::apply" );
            for_each_nd_type for_each_nd_inst;
            for_each_nd_inst(
                free_energy_kernel{ state, *density_, range_, step_, b_cond_, phobic_en_, gamma_, cell_volume }, range_
            );
        }

        scalar_type phobic = vspace_->scalar_prod( *density_, *e0_ );
        scalar_type total  = vspace_->sum( *density_ );
        return energies{ phobic, total - phobic };
    }

private:
    vector_space_ptr   vspace_;
    idx_nd_type        range_;
    grid_step_type     step_;
    boundary_cond_type b_cond_;
    dist_ptr           dist_;
    PhobicEnergy       phobic_en_;
    scalar_type        gamma_;

    using vector_wrap_t = nmfd::detail::vector_wrap<VectorSpace, true, true>;
    mutable vector_wrap_t density_;
    vector_wrap_t         e0_;
};

} // namespace tests

#endif
