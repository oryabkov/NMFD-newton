#include <memory>
#include <cmath>

#include <scfd/utils/device_tag.h>

#define PLATFORM_SERIAL_CPU

#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/backend/backend.h>
#include <nmfd/operations/matrix_operator.h>

#ifndef USE_DOUBLE_PRECISION
using scalar                = float;
inline constexpr scalar eps = 1e-5f;
#else
using scalar                = double;
inline constexpr scalar eps = 1e-10;
#endif

namespace
{

template <class Scalar>
struct residual_kernel
{
    const Scalar *u;
    Scalar       *f;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        f[idx] = ( idx == Idx( 0 ) ) ? ( u[0] * u[0] + u[1] * u[1] - Scalar{ 2 } ) : ( u[0] * u[1] - Scalar{ 1 } );
    }
};

template <class Scalar, class MatrixType>
struct jacobi_2d_kernel
{
    const Scalar *u;
    MatrixType    J;

    template <class IdxND>
    __DEVICE_TAG__ void operator()( const IdxND &idx )
    {
        const auto i = idx[0];
        const auto j = idx[1];
        J( i, j )    = ( i == 1 && j == 1 ) ? u[0]
                       : ( i == 1 )         ? u[1]
                       : ( j == 1 )         ? Scalar{ 2 } * u[1]
                                            : Scalar{ 2 } * u[0];
    }
};

template <class DenseOpsBase>
struct newton_dense_ops : DenseOpsBase
{
    using DenseOpsBase::DenseOpsBase;
    using DenseOpsBase::for_each_inst_;
    using DenseOpsBase::for_each_nd_inst_;
    using DenseOpsBase::vt_;
    using typename DenseOpsBase::matrix_type;
    using typename DenseOpsBase::scalar_type;
    using typename DenseOpsBase::vector_type;

    void apply_residual( const vector_type &u, vector_type &f ) const
    {
        const scalar_type *pu = vt_.get_raw_ptr( u );
        scalar_type       *pf = vt_.get_raw_ptr( f );
        for_each_inst_( residual_kernel<scalar_type>{ pu, pf }, 2 );
    }

    void fill_jacobian( matrix_type &J, const vector_type &u ) const
    {
        const scalar_type *pu = vt_.get_raw_ptr( u );
        for_each_nd_inst_( jacobi_2d_kernel<scalar_type, matrix_type>{ pu, J }, J.size_nd() );
    }
};

} // namespace

template <class T, class VectorType>
T eq_residual( const VectorType &x )
{
    T f0 = x( 0 ) * x( 0 ) + x( 1 ) * x( 1 ) - 2;
    T f1 = x( 0 ) * x( 1 ) - 1;
    return std::sqrt( f0 * f0 + f1 * f1 );
}

int main( int argc, char const *args[] )
{
    using T            = scalar;
    using backend_type = nmfd::backend::current<T>;
    using log_t        = typename backend_type::log_type;

    using dense_ops_base_t  = typename backend_type::dense_operations_type;
    using dense_ops_t       = newton_dense_ops<dense_ops_base_t>;
    using vec_space_t       = typename backend_type::vector_space_type;
    using vector_type       = typename dense_ops_t::vector_type;
    using matrix_type       = typename dense_ops_t::matrix_type;
    using matrix_operator_t = nmfd::operations::matrix_operator<dense_ops_t>;

    struct direct_solver_t
    {
        explicit direct_solver_t( std::shared_ptr<dense_ops_t> ops ) : ops_( std::move( ops ) )
        {
        }

        void set_operator( std::shared_ptr<const matrix_operator_t> A )
        {
            A_ = std::move( A );
        }

        void solve( const vector_type &b, vector_type &x ) const
        {
            ops_->solve( *A_, b, x );
        }

        std::shared_ptr<dense_ops_t>             ops_;
        std::shared_ptr<const matrix_operator_t> A_;
    };

    struct system_op_t
    {
        using scalar_type          = T;
        using jacobi_operator_type = matrix_operator_t;

        explicit system_op_t( std::shared_ptr<dense_ops_t> ops ) : ops_( std::move( ops ) )
        {
            matrix_type M;
            ops_->init_matrix( 2, 2, M );
            ops_->assign_zero_matrix( M );
            jacobi_ = std::make_shared<matrix_operator_t>( ops_, std::move( M ) );
        }

        void apply( const vector_type &u, vector_type &f ) const
        {
            ops_->apply_residual( u, f );
        }

        void set_linearization_point( const vector_type &u )
        {
            ops_->fill_jacobian( *jacobi_, u );
        }

        [[nodiscard]] std::shared_ptr<const matrix_operator_t> get_jacobi_operator() const
        {
            return jacobi_;
        }

        std::shared_ptr<dense_ops_t>       ops_;
        std::shared_ptr<matrix_operator_t> jacobi_;
    };

    using newton_iteration_t = nmfd::solvers::newton_iteration<vec_space_t, system_op_t, direct_solver_t>;
    using newton_solver_t    = nmfd::solvers::nonlinear_solver<vec_space_t, log_t, system_op_t, newton_iteration_t>;

    int          error = 0;
    backend_type backend;
    log_t       &log = backend.log();

    auto ops    = std::make_shared<dense_ops_t>( 2 );
    auto vec_sp = std::make_shared<vec_space_t>( 2 );

    {
        log.info( "test newton with direct solver" );

        auto direct_solver = std::make_shared<direct_solver_t>( ops );
        auto newton_iter   = std::make_shared<newton_iteration_t>( vec_sp, direct_solver );
        auto newton_solver = std::make_shared<newton_solver_t>( vec_sp, &log, newton_iter );
        newton_solver->convergence_strategy()->set_tolerance( eps );

        system_op_t system_op( ops );

        vector_type x;
        vec_sp->init_vector( x );
        vec_sp->start_use_vector( x );
        x( 0 ) = T{ 10 };
        x( 1 ) = T{ 2 };

        newton_solver->solve( &system_op, nullptr, nullptr, x );
        log.info_f( "result vector x: %0.15f %0.15f", x( 0 ), x( 1 ) );
        T resid = eq_residual<T>( x );
        log.info_f( "residual norm: %0.15e", resid );
        if ( resid > eps )
        {
            log.error( "Failed to converge!!" );
            error++;
        }

        vec_sp->stop_use_vector( x );
        vec_sp->free_vector( x );
    }

    if ( error > 0 )
    {
        log.error_f( "Got error = %d.", error );
    }
    else
    {
        log.info( "No errors." );
    }

    return error;
}
