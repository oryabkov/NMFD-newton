#include <memory>
#include <cmath>

#include <scfd/utils/log.h>

#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/backend/backend.h>
#include <nmfd/operations/matrix_operator.h>

#define M_PIl 3.141592653589793238462643383279502884L

template <typename MatrixType, typename MatrixOperator, typename DenseOperations>
auto get_matrix_operator( const int N )
{
    auto       ops = std::make_shared<DenseOperations>( N );
    MatrixType A_mat;
    ops->init_matrix( N, N, A_mat );
    ops->assign_zero_matrix( A_mat );
    {
        auto           v    = A_mat.create_view( false );
        constexpr auto diag = 4;
        constexpr auto off  = -1;
        for ( std::size_t i = 0; i < N; ++i )
        {
            v( i, i ) = diag;
            if ( i > 0 )
            {
                v( i, i - 1 ) = off;
            }
            if ( i + 1 < N )
            {
                v( i, i + 1 ) = off;
            }
        }
        v.release( true );
    }
    return std::make_shared<MatrixOperator>( ops, std::move( A_mat ) );
}


int main( int argc, char const *args[] )
{
    using log_t        = scfd::utils::log_std;
    using T            = double;
    using backend_type = nmfd::backend::current<T, log_t>;

    using memory_type       = backend_type::memory_type;
    using dense_ops_t       = backend_type::dense_operations_type;
    using vec_ops_t         = backend_type::vector_space_type;
    using vector_type       = typename dense_ops_t::vector_type;
    using matrix_type       = typename dense_ops_t::matrix_type;
    using matrix_operator_t = nmfd::operations::matrix_operator<dense_ops_t>;
    using monitor_t         = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
    using gmres_t           = nmfd::solvers::gmres<vec_ops_t, monitor_t, log_t, matrix_operator_t>;


    int   error = 0;
    log_t log;
    log.info( "test gmres" );


    std::size_t N = 50;

    auto vec_ops = std::make_shared<vec_ops_t>( N );

    auto A_op = get_matrix_operator<matrix_type, matrix_operator_t, dense_ops_t>( N );

    auto get_residual = [&log, &vec_ops]( auto &A, auto &x, auto &y ) {
        vector_type resid;
        vec_ops->init_vector( resid );
        vec_ops->start_use_vector( resid );
        A.apply( x, resid );
        vec_ops->add_lin_comb( 1, y, -1, resid );
        log.info_f( "||Lx-y|| = %e", vec_ops->norm( resid ) );
        vec_ops->stop_use_vector( resid );
        vec_ops->free_vector( resid );
    };

    {
        vector_type x, y;
        vec_ops->init_vector( x );
        vec_ops->init_vector( y );
        vec_ops->start_use_vector( x );
        vec_ops->start_use_vector( y );

        log.info_f( "=> dense GMRES, size %zu", vec_ops->size() );

        for ( int j = 0; j < N; j++ )
        {
            y( j ) = std::sin( 1.0 * j / ( N - 1 ) * M_PIl );
            // x[j] = 0.1*std::sin(1.0*j/(N-1)*M_PIl);
        }
        y( 0 ) = y( N - 1 ) = 0;
        typename gmres_t::params prm;
        prm.basis_size            = 30;
        prm.monitor.rel_tol       = 1.0e-10;
        prm.monitor.max_iters_num = 300;

        gmres_t gmres( A_op, vec_ops, &log, prm );

        vec_ops->assign_scalar( 0.0, x );
        log.info( "no preconditioner" );
        bool res = gmres.solve( y, x );
        error += ( !res );

        log.info_f( "gmres res: %s", res ? "true" : "false" );
        log.info( "reusing the solution..." );
        vec_ops->add_mul_scalar( 0.0, 0.9999999, x );
        res = gmres.solve( y, x );
        error += ( !res );
        log.info_f( "gmres res with x0: %s", res ? "true" : "false" );
        get_residual( *A_op, x, y );

        vec_ops->stop_use_vector( x );
        vec_ops->stop_use_vector( y );
        vec_ops->free_vector( x );
        vec_ops->free_vector( y );
    }


    if ( error > 0 )
    {
        log.error_f( "Got error = %e.", error );
    }
    else
    {
        log.info( "No errors." );
    }

    return error;
}
