#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__

#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>
#include <scfd/static_vec/vec.h>

namespace tests
{

template <class IdxND, class Scalar, class TensorType, class VectorType>
__DEVICE_TAG__ TensorType periodic_bc_vector(
    const VectorType       &vector,
    const IdxND            &idx,
    int                     axis,
    int                     N,
    bool                    is_left
)
{
    TensorType periodic_vals;
    #pragma unroll
    for ( int c = 0; c < TensorType::dim; ++c )
    {
        IdxND periodic_idx = idx;
        if ( idx[axis] == 0 )
        {
            periodic_idx[axis] = is_left ? N - 1 : 1;
        }

        if ( idx[axis] == N - 1 )
        {
            periodic_idx[axis] = is_left ? N - 2 : 0;
        }

        periodic_vals[c] = vector.get_vec( periodic_idx )[c];
    }

    return periodic_vals;
}

template <class VectorSpace>
class boundary_cond
{
public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;

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
        : gamma_( 0 ), cos_theta_( 0 )
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

    boundary_cond( conditions left[dim][tensor_dim], conditions right[dim][tensor_dim], scalar_type gamma, scalar_type cos_theta)
        : boundary_cond(left, right)
    {
        gamma_ = gamma;
        cos_theta_ = cos_theta;
    }

    __DEVICE_TAG__ scalar_type compute_A( scalar_type h_j, scalar_type delta_val ) const
    {
        return delta_val * h_j * cos_theta_ / st::sqrt(2 * gamma_);
    }

    __DEVICE_TAG__ scalar_type compute_D( scalar_type C0, scalar_type A ) const
    {
        // return 1 + A * C0 + A * A;
        return A * A + scalar_type( 2 ) * A * C0 + scalar_type( 1 );
    }


    __DEVICE_TAG__ scalar_type nonlinear_ghost( scalar_type C0, scalar_type A ) const
    {
        if ( st::abs(cos_theta_) < scalar_type( 1e-7 ) )
        {
            return C0;
        }

        scalar_type Cq = st::max( scalar_type( -1 ), st::min( C0, scalar_type( 1 ) ) );
        // scalar_type Cq = C0;

        scalar_type D = compute_D( Cq, A );
        if ( D <= scalar_type( 0 ) )
        {
            #if !defined(PLATFORM_CUDA)
                std::cout << "D <= 0 in nonlinear_ghost" << " C0: " << C0 << " A: " << A << " D: " << D << std::endl;
            #endif
        }

        // return ( -( scalar_type( 2 ) + A * C0 ) + 2 * st::sqrt( D ) ) / A;
        return ( scalar_type( 2 ) * st::sqrt( D ) - A * Cq - scalar_type( 2 ) ) / A;
    }


    __DEVICE_TAG__ scalar_type nonlinear_ghost_coef_linearized( scalar_type C0_lin, scalar_type A ) const
    {
        if ( st::abs(cos_theta_) < scalar_type( 1e-7 ) )
        {
            return scalar_type( 1 );
        }

        scalar_type Cq = st::max( scalar_type( -1 ), st::min( C0_lin, scalar_type( 1 ) ) );
        // scalar_type Cq = C0_lin;

        scalar_type D = compute_D( Cq, A );
        if ( D <= scalar_type( 0 ) )
        {
            #if !defined(PLATFORM_CUDA)
                std::cout << "D <= 0 in nonlinear_ghost_coef_linearized" << " C0: " << C0_lin << " A: " << A << " D: " << D << std::endl;
                // assert();
                // printf();
            #endif
        }

        // D = st::max( D, scalar_type( 1e-14 ) );
        // return scalar_type( 1 ) / st::sqrt( D ) - scalar_type( 1 );

        scalar_type dxdCq = scalar_type( 2 ) / st::sqrt( D ) - scalar_type( 1 );
        if ( C0_lin < scalar_type( -1 ) || C0_lin > scalar_type( 1 ) )
        {
            return scalar_type( 0 );
        }

        return dxdCq;
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

    __DEVICE_TAG__ void set_gamma( scalar_type gamma )
    {
        gamma_ = gamma;
    }

private:
    // Parameters for nonlinear boundary condition
    scalar_type     gamma_;
    scalar_type     cos_theta_;
};

} // namespace tests

#endif
