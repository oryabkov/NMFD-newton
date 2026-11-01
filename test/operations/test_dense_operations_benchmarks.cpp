
#include <cmath>
#include <limits>
#include <string>
#include <memory>


#include <scfd/arrays/array.h>
#include <scfd/backend/backend.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/utils/log.h>

#include <nmfd/operations/dense_operations_base.h>
#include <nmfd/platform/dense_solvers_singleton_impl.h>

#ifndef USE_DOUBLE_PRECISION
using scalar                = float;
inline constexpr scalar eps = 1e-5f;
#else
using scalar                = double;
inline constexpr scalar eps = 1e-10;
#endif

template <typename MatrixType>
MatrixType create_nilpotent_matrix( size_t rows )
{
    MatrixType result;
    result.init( rows, rows );

    auto view = result.create_view( false );

    for ( size_t i = 0; i < rows; ++i )
    {
        for ( size_t j = 0; j < rows; ++j )
        {
            if ( j == i + 1 )
            {
                view( i, j ) = 1;
            }
            else
            {
                view( i, j ) = 0;
            }
        }
    }
    view.release();
    return result;
}

template <typename VectorType>
VectorType create_test_vector( size_t size )
{
    VectorType result;
    result.init( size );
    auto view = result.create_view( false );
    for ( size_t i = 0; i < size; ++i )
    {
        view( i ) = i + 1;
    }
    view.release();
    return result;
}

bool parse_matrix_size( int argc, const char *args[], size_t &rows, scfd::utils::log_std &log )
{
    try
    {
        if ( argc != 2 || args[1][0] == '-' )
            throw std::invalid_argument( "Expected a positive matrix size" );
        size_t     parsed = 0;
        const auto value  = std::stoull( args[1], &parsed );
        if ( args[1][parsed] != '\0' || value == 0 ||
             value > static_cast<unsigned long long>( std::numeric_limits<int>::max() ) )
            throw std::invalid_argument( "Invalid matrix size" );
        rows = static_cast<size_t>( value );
    }
    catch ( const std::exception & )
    {
        log.error( "Usage: test_dense_operations_cublas <positive matrix size>" );
        return false;
    }

    return true;
}

int main( int argc, char const *args[] )
{

    using log_t            = scfd::utils::log_std;
    using T                = scalar;
    using backend_type     = scfd::backend::current;
    using memory_type      = backend_type::memory_type;
    using dense_ops_t      = nmfd::operations::dense_operations<T, backend_type>;
    using vector_type      = typename dense_ops_t::vector_type;
    using matrix_type      = typename dense_ops_t::matrix_type;
    using timer_event_type = typename backend_type::timer_event_type;

    log_t  log;
    size_t rows = 0;
    if ( !parse_matrix_size( argc, args, rows, log ) )
        return 1;

    backend_type::init_device( log );
#ifdef PLATFORM_CUDA
    backend_type::blas_wrap_type   blas( true );
    backend_type::solver_wrap_type solver( &blas, true );
#endif

    size_t failed_counter = 0;
    size_t passed_counter = 0;
    auto   ops            = std::make_shared<dense_ops_t>();

    log.info( "Benchmarks for dense_operations" );
    log.info( std::string( "Backend: " ) + backend_type::name() );

    {
        log.info( "Matrix product: square of a nilpotent matrix has ones on the second superdiagonal" );
        timer_event_type start;
        timer_event_type finish;
        auto             matrix = create_nilpotent_matrix<matrix_type>( rows );

        start.record();
        auto result = ops->matrix_matrix_prod( matrix, matrix );
        finish.record();
        log.info( "Elapsed time (ms): " + std::to_string( finish.elapsed_time( start ) ) );

        auto view   = result->create_view( true );
        bool passed = true;
        for ( size_t i = 0; i < rows; ++i )
        {
            for ( size_t j = 0; j < rows; ++j )
            {
                const T expected = j == i + 2 ? T{ 1 } : T{ 0 };
                if ( !std::isfinite( view( i, j ) ) || std::abs( view( i, j ) - expected ) > eps )
                    passed = false;
            }
        }
        view.release( false );
        if ( !passed )
        {
            log.error( "Matrix product failed" );
            ++failed_counter;
        }
        else
        {
            ++passed_counter;
        }
    }
    {
        log.info( "Matrix-vector product: shift components by one index and set the last to zero" );
        timer_event_type start;
        timer_event_type finish;
        auto             matrix = create_nilpotent_matrix<matrix_type>( rows );
        auto             vector = create_test_vector<vector_type>( rows );
        vector_type      result;
        result.init( rows );

        start.record();
        ops->add_matrix_vector_prod( T{ 1 }, matrix, vector, T{ 0 }, result );
        finish.record();
        log.info( "Elapsed time (ms): " + std::to_string( finish.elapsed_time( start ) ) );

        auto view   = result.create_view( true );
        bool passed = true;
        for ( size_t i = 0; i < rows; ++i )
        {
            const T expected = i + 1 < rows ? static_cast<T>( i + 2 ) : T{ 0 };
            if ( !std::isfinite( view( i ) ) || std::abs( view( i ) - expected ) > eps )
                passed = false;
        }
        view.release( false );
        if ( !passed )
        {
            log.error( "Matrix-vector product failed" );
            ++failed_counter;
        }
        else
        {
            ++passed_counter;
        }
    }
    {
        log.info( "Solve: diagonal A[i,i] = i + 1, expected solution x[i] = 1 + i / N" );
        timer_event_type start;
        timer_event_type finish;

        matrix_type matrix;
        matrix.init( rows, rows );
        vector_type rhs, result;
        rhs.init( rows );
        result.init( rows );

        auto matrix_view = matrix.create_view( false );
        auto rhs_view    = rhs.create_view( false );
        for ( size_t i = 0; i < rows; ++i )
        {
            const T diagonal = static_cast<T>( i + 1 );
            const T expected = T{ 1 } + static_cast<T>( i ) / static_cast<T>( rows );
            for ( size_t j = 0; j < rows; ++j )
                matrix_view( i, j ) = i == j ? diagonal : T{ 0 };
            rhs_view( i ) = diagonal * expected;
        }
        matrix_view.release();
        rhs_view.release();

        start.record();
        ops->solve( matrix, rhs, result );
        finish.record();
        log.info( "Elapsed time (ms): " + std::to_string( finish.elapsed_time( start ) ) );

        auto view   = result.create_view( true );
        bool passed = true;
        for ( size_t i = 0; i < rows; ++i )
        {
            const T expected = T{ 1 } + static_cast<T>( i ) / static_cast<T>( rows );
            if ( !std::isfinite( view( i ) ) || std::abs( view( i ) - expected ) > eps * expected )
                passed = false;
        }
        view.release( false );
        if ( !passed )
        {
            log.error( "Solve failed" );
            ++failed_counter;
        }
        else
        {
            ++passed_counter;
        }
    }

    log.info( "Passed: " + std::to_string( 3 - failed_counter ) + ", failed: " + std::to_string( failed_counter ) );
    return failed_counter == 0 ? 0 : 1;
}
