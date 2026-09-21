#ifndef __DROPLET_STATS_H__
#define __DROPLET_STATS_H__

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <scfd/static_vec/vec.h>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

namespace tests
{

template <
    class VectorSpace,
    class Distributor>
class droplet_stats
{
public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;

    using scalar_type       = typename VectorSpace::scalar_type;
    using vector_type       = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;
    using grid_step_type    = scfd::static_vec::vec<scalar_type, dim>;
    using idx_nd_type       = typename VectorSpace::idx_nd_type;
    using view_type         = typename vector_type::view_type;
    using dist_type         = Distributor;
    using dist_ptr          = std::shared_ptr<const dist_type>;
    using comm_type         = typename VectorSpace::comm_type;

    using vector_space_ptr = std::shared_ptr<VectorSpace>;

    struct stats
    {
        scalar_type mass;
        scalar_type drop_volume;
        scalar_type r_eff;
        scalar_type phi_max;
        scalar_type phi_min;
        scalar_type phi_centre;
    };

public:
    droplet_stats( vector_space_ptr vspace, grid_step_type step, dist_ptr dist, comm_type comm )
        : vspace_( std::move( vspace ) ), range_( vspace_->get_size() ), step_( step ), dist_( std::move( dist ) ), comm_( comm )
    {
    }

    stats compute( const vector_type &state ) const
    {
        view_type view( state, true );

        scalar_type dV = step_.components_prod();

        scalar_type mass_loc        = scalar_type( 0 );
        scalar_type drop_volume_loc = scalar_type( 0 );
        scalar_type phi_max         = -std::numeric_limits<scalar_type>::max();
        scalar_type phi_min         = std::numeric_limits<scalar_type>::max();
        scalar_type phi_centre      = scalar_type( 0 );
        scalar_type best_dist2      = std::numeric_limits<scalar_type>::max();

        for ( int i = 0; i < range_[0]; i++ )
        {
            for ( int j = 0; j < range_[1]; j++ )
            {
                for ( int k = 0; k < range_[2]; k++ )
                {
                    scalar_type phi = view( i, j, k, 1 );

                    mass_loc += phi * dV;
                    if ( phi > scalar_type( 0 ) )
                    {
                        drop_volume_loc += dV;
                    }
                    phi_max = std::max( phi_max, phi );
                    phi_min = std::min( phi_min, phi );

                    scalar_type x     = step_[0] * ( scalar_type( 0.5 ) + i );
                    scalar_type y     = step_[1] * ( scalar_type( 0.5 ) + j );
                    scalar_type z     = step_[2] * ( scalar_type( 0.5 ) + k );
                    scalar_type dist2 = ( x - scalar_type( 0.5 ) ) * ( x - scalar_type( 0.5 ) ) +
                                        ( y - scalar_type( 0.5 ) ) * ( y - scalar_type( 0.5 ) ) +
                                        ( z - scalar_type( 0.5 ) ) * ( z - scalar_type( 0.5 ) );
                    if ( dist2 < best_dist2 )
                    {
                        best_dist2 = dist2;
                        phi_centre = phi;
                    }
                }
            }
        }

        stats s;
        s.mass        = comm_.all_reduce_sum( mass_loc );
        s.drop_volume = comm_.all_reduce_sum( drop_volume_loc );
        s.r_eff       = std::cbrt( scalar_type( 3 ) * s.drop_volume / ( scalar_type( 4 ) * scalar_type( M_PI ) ) );
        s.phi_max     = phi_max;
        s.phi_min     = phi_min;
        s.phi_centre  = phi_centre;
        return s;
    }

private:
    vector_space_ptr vspace_;
    idx_nd_type      range_;
    grid_step_type   step_;
    dist_ptr         dist_;
    comm_type        comm_;
};

} // namespace tests

#endif
