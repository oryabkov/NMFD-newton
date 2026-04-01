#include <memory>
#include <cmath>

#include <scfd/utils/log.h>

#define PLATFORM_SERIAL_CPU

#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/backend/backend.h>
#include <nmfd/operations/matrix_operator.h>

template <class T, class VectorType>
T eq_residual( const VectorType &x )
{
    T f0 = x( 0 ) * x( 0 ) + x( 1 ) * x( 1 ) - 2;
    T f1 = x( 0 ) * x( 1 ) - 1;
    return std::sqrt( f0 * f0 + f1 * f1 );
}

int main( int argc, char const *args[] )
{
    using log_t        = scfd::utils::log_std;
    using T            = double;
    using backend_type = nmfd::backend::current<T, log_t>;

    using dense_ops_t       = backend_type::dense_operations_type;
    using vec_space_t       = backend_type::vector_space_type;
    using vector_type       = typename dense_ops_t::vector_type;
    using matrix_type       = typename dense_ops_t::matrix_type;
    using matrix_operator_t = nmfd::operations::matrix_operator<dense_ops_t>;
    using monitor_t         = nmfd::solvers::monitor_krylov<vec_space_t, log_t>;
    using gmres_t           = nmfd::solvers::gmres<vec_space_t, monitor_t, log_t, matrix_operator_t>;

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

        void apply( const vector_type &x, vector_type &f ) const
        {
            f( 0 ) = x( 0 ) * x( 0 ) + x( 1 ) * x( 1 ) - 2.;
            f( 1 ) = x( 0 ) * x( 1 ) - 1.;
        }

        void set_linearization_point( const vector_type &x )
        {
            auto v    = jacobi_->create_view( false );
            v( 0, 0 ) = 2 * x( 0 );
            v( 0, 1 ) = 2 * x( 1 );
            v( 1, 0 ) = x( 1 );
            v( 1, 1 ) = x( 0 );
            v.release( true );
        }

        [[nodiscard]] std::shared_ptr<const matrix_operator_t> get_jacobi_operator() const
        {
            return jacobi_;
        }

        std::shared_ptr<dense_ops_t>       ops_;
        std::shared_ptr<matrix_operator_t> jacobi_;
    };

    using newton_iteration_t = nmfd::solvers::newton_iteration<vec_space_t, system_op_t, gmres_t>;
    using newton_solver_t    = nmfd::solvers::nonlinear_solver<vec_space_t, log_t, system_op_t, newton_iteration_t>;

    const T eps = 1e-10;

    int          error = 0;
    backend_type backend;
    log_t       &log = backend.log();

    auto ops    = std::make_shared<dense_ops_t>( 2 );
    auto vec_sp = std::make_shared<vec_space_t>( 2 );

    {
        log.info( "test newton with backend construction" );
        std::shared_ptr<newton_solver_t> newton_solver =
            std::make_shared<newton_solver_t>( newton_solver_t::utils_hierarchy( backend, vec_sp ) );
        newton_solver->convergence_strategy()->set_tolerance( eps );
        system_op_t system_op( ops );

        vector_type x;
        vec_sp->init_vector( x );
        vec_sp->start_use_vector( x );
        x( 0 ) = 10.;
        x( 1 ) = 2.;

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
