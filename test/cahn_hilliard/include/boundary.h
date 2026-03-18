#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__

#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>
#include <scfd/static_vec/vec.h>

namespace tests
{


template <class VectorSpace>
// template <class VectorSpace, class scalar_type, int dim, int tensor_dim>
class boundary_cond
{
public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;

    using vector_space_type  = VectorSpace;
    using vector_space_ptr = std::shared_ptr<vector_space_type>;
    using scalar_type  = typename VectorSpace::scalar_type;
    using vector_type  = typename VectorSpace::vector_type;
    using tensor_type  = scfd::static_vec::vec<scalar_type, tensor_dim>;

    using idx_nd_type      = typename VectorSpace::idx_nd_type;
    using grid_step_type   = scfd::static_vec::vec<scalar_type, dim>;

    using conditions   = int;
    using st = scfd::utils::scalar_traits<scalar_type>;

public:
    // -1 for dirichlet
    // +1 for neumann
    //  0 for periodic
    //  2 for nonlinear
    conditions left[dim][tensor_dim];   // left  boundary condition
    conditions right[dim][tensor_dim];  // right boundary condition

public:
    boundary_cond( conditions left[dim][tensor_dim], conditions right[dim][tensor_dim] )
        : sigma_( 0 ), cos_theta_( 0 ), alpha_( 0 )
    {
        for ( int j = 0; j < dim; ++j )
        {
            for ( int jj = 0; jj < tensor_dim; ++jj )
            {
                this->left[j][jj] = left[j][jj];
                this->right[j][jj] = right[j][jj];
            }
        }
    }

    boundary_cond( conditions left[dim][tensor_dim], conditions right[dim][tensor_dim], scalar_type sigma, scalar_type cos_theta, scalar_type alpha)
        : boundary_cond(left, right)
    {
        sigma_ = sigma;
        cos_theta_ = cos_theta;
        alpha_ = alpha;
    }

    // A = (3/16) * h_j / alpha * sigma * cos_theta * delta_val
    __DEVICE_TAG__ scalar_type compute_A( scalar_type h_j, scalar_type delta_val ) const
    {
        return scalar_type( 3 ) / scalar_type( 16 ) * h_j / alpha_ * sigma_ * cos_theta_ * delta_val;
    }


    __DEVICE_TAG__ scalar_type nonlinear_ghost( scalar_type C0, scalar_type A ) const
    {
        if ( st::abs(cos_theta_) < scalar_type( 1e-7 ) )
        {
            return C0;
        }

        scalar_type D = scalar_type( 8 ) * A * C0 + scalar_type( 16 ) * A * A + scalar_type( 1 );
        if ( D <= scalar_type( 0 ) )
        {
            static_assert( true, "D <= 0 in nonlinear_ghost" );
        }

        return ( -( scalar_type( 2 ) * A * C0 + scalar_type( 1 ) ) + st::sqrt( D ) ) / ( scalar_type( 2 ) * A );
    }


    __DEVICE_TAG__ scalar_type nonlinear_ghost_coef_linearized( scalar_type C0_lin, scalar_type A ) const
    {
        if ( st::abs(cos_theta_) < scalar_type( 1e-7 ) )
        {
            return scalar_type( 1 );
        }

        scalar_type D = scalar_type( 8 ) * A * C0_lin + scalar_type( 16 ) * A * A + scalar_type( 1 );
        if ( D <= scalar_type( 0 ) )
        {
            static_assert( true, "D <= 0 in nonlinear_ghost_coef" );
        }

        return scalar_type( 2 ) / st::sqrt( D ) - scalar_type( 1 );
    }


    // Used only in cahn_hilliard_op. Support only 1 stencil size!!!
    __DEVICE_TAG__ void get_ghost_tensor(
        const vector_type &vector, const idx_nd_type &dom_sz, const idx_nd_type &ghost_idx,
        const grid_step_type &step, tensor_type &res
    ) const
    {
        idx_nd_type internal_idx = ghost_idx;
        grid_step_type scaled_step = step;
        get_internal_idx( ghost_idx, dom_sz, step, internal_idx, scaled_step );

        vector.get_vec( res, internal_idx );

        // TODO: Think how to index values in delta
        // auto delta_ = delta_wrap_.get_vec( internal_idx );
        auto delta_ = 1;

        #pragma unroll
        for ( int j = 0; j < dim; ++j )
        {
            if ( ghost_idx[j] < 0 )
            {
                #pragma unroll
                for ( int jj = 0; jj < tensor_dim; ++jj )
                {
                    if ( left[j][jj] == 2 )
                    {
                        res[jj] = nonlinear_ghost( res[jj], compute_A( scaled_step[j], delta_ ) );
                    }
                    else
                    {
                        res[jj] *= left[j][jj];
                    }
                }
            }
            else if ( ghost_idx[j] >= dom_sz[j] )
            {
                #pragma unroll
                for ( int jj = 0; jj < tensor_dim; ++jj )
                {
                    if ( right[j][jj] == 2 )
                    {
                        res[jj] = nonlinear_ghost( res[jj], compute_A( scaled_step[j], delta_ ) );
                    }
                    else
                    {
                        res[jj] *= right[j][jj];
                    }
                }
            }
        }
    }


    // Used for linear operators like jacobi_op, jacobi_pre, prolongator and restrictor
    // Support any stencil size!!!
    __DEVICE_TAG__ void get_ghost_tensor_linearized(
        const vector_type &lin_vector, const vector_type &vector, const idx_nd_type &dom_sz,
        const idx_nd_type &ghost_idx, const grid_step_type &step, tensor_type &res
    ) const
    {
        idx_nd_type internal_idx = ghost_idx;
        grid_step_type scaled_step = step;
        get_internal_idx( ghost_idx, dom_sz, step, internal_idx, scaled_step );

        vector.get_vec( res, internal_idx );

        // vector.get_vec( res, internal_idx );
        // auto lin_res = lin_vector.get_vec( internal_idx );

        // // auto delta_  = delta_wrap_.get_vec( internal_idx );
        // auto delta_ = 1;

        // #pragma unroll
        // for ( int j = 0; j < dim; ++j )
        // {
        //     if ( ghost_idx[j] < 0 )
        //     {
        //         #pragma unroll
        //         for ( int jj = 0; jj < tensor_dim; ++jj )
        //         {
        //             if ( left[j][jj] == 2 )
        //                 res[jj] = nonlinear_ghost_coef( lin_res[jj], compute_A( scaled_step[j], delta_ ) ) * res[jj];
        //             else
        //                 res[jj] *= left[j][jj];
        //         }
        //     }
        //     else if ( ghost_idx[j] >= dom_sz[j] )
        //     {
        //         #pragma unroll
        //         for ( int jj = 0; jj < tensor_dim; ++jj )
        //         {
        //             if ( right[j][jj] == 2 )
        //                 res[jj] = nonlinear_ghost_coef( lin_res[jj], compute_A( scaled_step[j], delta_ ) ) * res[jj];
        //             else
        //                 res[jj] *= right[j][jj];
        //         }
        //     }
        // }

        tensor_type mul = tensor_type::make_ones();
        get_ghost_coef_linearized( lin_vector, dom_sz, ghost_idx, step, mul );

        #pragma unroll
        for ( int j = 0; j < tensor_dim; ++j )
        {
            res[j] *= mul[j];
        }

    }


    __DEVICE_TAG__ void get_ghost_coef_linearized(
        const vector_type &lin_vector, const idx_nd_type &dom_sz, const idx_nd_type &ghost_idx,
        const grid_step_type &step, tensor_type &mul
    ) const
    {
        idx_nd_type internal_idx = ghost_idx;
        grid_step_type scaled_step = step;
        get_internal_idx( ghost_idx, dom_sz, step, internal_idx, scaled_step );

        auto lin_res = lin_vector.get_vec( internal_idx );
        // auto delta_  = delta_wrap_.get_vec( internal_idx );
        auto delta_ = 1;

        #pragma unroll
        for ( int j = 0; j < tensor_dim; ++j )
        {
            mul[j] = 1;
        }

        #pragma unroll
        for ( int j = 0; j < dim; ++j )
        {
            if ( ghost_idx[j] < 0 )
            {
                #pragma unroll
                for ( int jj = 0; jj < tensor_dim; ++jj )
                {
                    if ( left[j][jj] == 2 )
                    {
                        mul[jj] *= nonlinear_ghost_coef_linearized( lin_res[jj], compute_A( scaled_step[j], delta_ ) );
                    }
                    else
                    {
                        mul[jj] *= left[j][jj];
                    }
                }
            }

            if ( ghost_idx[j] >= dom_sz[j] )
            {
                #pragma unroll
                for ( int jj = 0; jj < tensor_dim; ++jj )
                {
                    if ( right[j][jj] == 2 )
                    {
                        mul[jj] *= nonlinear_ghost_coef_linearized( lin_res[jj], compute_A( scaled_step[j], delta_ ) );
                    }
                    else
                    {
                        mul[jj] *= right[j][jj];
                    }
                }
            }
        }
    }

    __DEVICE_TAG__ void get_internal_idx( const idx_nd_type &ghost_idx, const idx_nd_type &dom_sz, const grid_step_type &step, idx_nd_type &internal_idx, grid_step_type &scaled_step) const
    {
        #pragma unroll
        for ( int j = 0; j < dim; ++j )
        {
            if ( ghost_idx[j] < 0 )
            {
                internal_idx[j] = -ghost_idx[j] - 1;
                scaled_step[j] = (internal_idx[j] - ghost_idx[j]) * step[j];
            }
            else if ( ghost_idx[j] >= dom_sz[j] )
            {
                internal_idx[j] = 2 * dom_sz[j] - ghost_idx[j] - 1;
                scaled_step[j] = (ghost_idx[j] - internal_idx[j]) * step[j];
            }
        }
    }

private:
    // Parameters for nonlinear boundary condition
    scalar_type     sigma_;
    scalar_type     cos_theta_;
    scalar_type     alpha_;
};

} // namespace tests

#endif
