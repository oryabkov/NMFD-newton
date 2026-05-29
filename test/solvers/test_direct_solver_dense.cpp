#include <memory>
#include <cmath>
#include <random>

#include <nmfd/backend/backend.h>

#ifndef USE_DOUBLE_PRECISION
using scalar                      = float;
inline constexpr scalar eps       = 1e-5f;
inline constexpr scalar rel_tol   = 1e-3f;
#else
using scalar                      = double;
inline constexpr scalar eps       = 1e-10;
inline constexpr scalar rel_tol   = 1e-10;
#endif

int main( int argc, char const *args[] )
{
    using T            = scalar;
    using backend_type = nmfd::backend::current<T>;
    using log_t        = typename backend_type::log_type;

    using dense_ops_t = typename backend_type::dense_operations_type;
    using vec_space_t = typename backend_type::vector_space_type;
    using vector_type = typename dense_ops_t::vector_type;
    using matrix_type = typename dense_ops_t::matrix_type;

    int          error = 0;
    backend_type backend;
    log_t       &log = backend.log();

    {
        log.info( "test direct solver: 5x5 known system" );

        constexpr int N      = 5;
        auto          ops    = std::make_shared<dense_ops_t>( N );
        auto          vec_sp = std::make_shared<vec_space_t>( N );

        matrix_type A = {
            { 1, 0, 2, 0, 0 },   //
            { 0, 2, 0, 0, 1 },   //
            { 1, 0, 3, 0, 1 },   //
            { 1, 0, 0, -3, 1 },  //
            { 10, 0, 0, 0, -5 }, //
        };

        vector_type b = { 7, 9, 15, -6, -15 };
        vector_type x( N );

        ops->solve( A, b, x );

        const T ref[N] = { 1, 2, 3, 4, 5 };
        for ( int i = 0; i < N; ++i )
        {
            const T xi   = vec_sp->get_value_at_point( i, x );
            const T diff = std::abs( xi - ref[i] );
            if ( diff > eps )
            {
                log.error_f( "x[%d] = %.15f, expected %.1f, diff = %.2e", i, xi, ref[i], diff );
                error++;
            }
        }
        log.info_f(
            "x: %.6f %.6f %.6f %.6f %.6f", vec_sp->get_value_at_point( 0, x ),
            vec_sp->get_value_at_point( 1, x ), vec_sp->get_value_at_point( 2, x ),
            vec_sp->get_value_at_point( 3, x ), vec_sp->get_value_at_point( 4, x )
        );
    }

    {
        constexpr int M = 50;
        log.info_f( "test direct solver: %dx%d random system", M, M );

        auto ops    = std::make_shared<dense_ops_t>( M );
        auto vec_sp = std::make_shared<vec_space_t>( M );

        matrix_type A;
        ops->init_matrix( M, M, A );

        vector_type b, x, resid;
        vec_sp->init_vector( b );
        vec_sp->init_vector( x );
        vec_sp->init_vector( resid );
        vec_sp->start_use_vector( b );
        vec_sp->start_use_vector( x );
        vec_sp->start_use_vector( resid );

        std::mt19937                      gen( 42 );
        std::uniform_real_distribution<T> dis( -10., 10. );
        {
            auto av = A.create_view( false );
            auto bv = b.create_view( false );
            for ( int i = 0; i < M; ++i )
            {
                for ( int j = 0; j < M; ++j )
                    av( i, j ) = dis( gen );
                bv( i ) = dis( gen );
            }
            bv.release( true );
            av.release( true );
        }

        ops->solve( A, b, x );

        ops->add_matrix_vector_prod( scalar( 1 ), A, x, scalar( 0 ), resid );
        vec_sp->add_lin_comb( scalar( -1 ), b, resid );

        const T rhs_norm = vec_sp->norm( b );
        const T rel      = ( rhs_norm > scalar( 0 ) ) ? vec_sp->norm( resid ) / rhs_norm : vec_sp->norm( resid );
        log.info_f( "relative residual ||Ax-b||/||b|| = %.2e", rel );

        if ( rel > rel_tol )
        {
            log.error( "residual too large" );
            error++;
        }

        vec_sp->stop_use_vector( b );
        vec_sp->stop_use_vector( x );
        vec_sp->stop_use_vector( resid );
        vec_sp->free_vector( b );
        vec_sp->free_vector( x );
        vec_sp->free_vector( resid );
    }

    if ( error > 0 )
        log.error_f( "Got %d errors.", error );
    else
        log.info( "No errors." );

    return error;
}
