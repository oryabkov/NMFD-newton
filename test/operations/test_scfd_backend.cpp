#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <nmfd/backend/backend.h>
#include <nmfd/detail/vector_wrap.h>

unsigned int checks = 0;

void require( bool passed, const char *message )
{
    ++checks;
    if ( !passed )
    {
        throw std::runtime_error( message );
    }
}

template <class T, class Ordinal>
void test()
{
    using backend_type    = nmfd::backend::current<T, scfd::utils::log_std, Ordinal>;
    using space_type      = typename backend_type::vector_space_type;
    using operations_type = typename backend_type::dense_operations_type;
    using matrix_type     = typename operations_type::matrix_type;
    static_assert( std::is_same_v<typename backend_type::ordinal_type, Ordinal> );
    static_assert( std::is_same_v<typename space_type::ordinal_type, Ordinal> );
    static_assert( std::is_same_v<typename operations_type::ordinal_type, Ordinal> );
    static_assert( std::is_same_v<typename space_type::for_each_type, typename backend_type::for_each_type> );

    backend_type                                        backend;
    auto                                                space = std::make_shared<space_type>( 3 );
    operations_type                                     operations( 3 );
    nmfd::detail::vector_wrap<space_type, false, false> x( *space ), y( *space ), result( *space );
    x.init();
    x.start_use();
    y.init();
    y.start_use();
    result.init();
    result.start_use();

    space->assign_scalar( T( 2 ), *x );
    space->assign_scalar( T( 3 ), *y );
    space->assign_lin_comb( T( 2 ), *x, T( -1 ), *y, *result );
    require(
        space->size() == 3 && space->scalar_prod( *result, *result ) == T( 3 ),
        "backend vector mapping or reduction failed"
    );
    require( space->is_valid_number( *result ), "backend rejected finite entries" );
    for ( T invalid : { std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN() } )
    {
        space->set_value_at_point( invalid, 1, *result );
        require( !space->is_valid_number( *result ), "backend failed to detect a nonfinite entry" );
    }

    typename space_type::multivector_type basis;
    space->init_multivector( basis, 2 );
    space->start_use_multivector( basis, 2 );
    space->assign( *x, basis, 2, 1 );
    space->assign( basis, 2, 1, *y );
    require( space->scalar_prod( basis, 2, 1, *y ) == T( 12 ), "multivector assignment or reduction failed" );
    space->add_lin_comb( T( -1 ), basis, 2, 1, T( 1 ), *y );
    require( space->norm_l2( *y ) == T( 0 ), "multivector linear combination failed" );
    space->stop_use_multivector( basis, 2 );
    space->free_multivector( basis, 2 );
    require( basis.empty(), "multivector storage was not released" );

    matrix_type matrix, copy;
    operations.init_matrix( 3, 3, matrix );
    operations.init_matrix( 3, 3, copy );
    operations.assign_zero_matrix( matrix );
    require( operations.matrix_norm_fro( matrix ) == T( 0 ), "matrix zero assignment failed" );
    {
        auto view    = matrix.create_view( true );
        view( 0, 0 ) = T( 2 );
        view( 0, 1 ) = view( 1, 0 ) = T( 1 );
        view( 1, 1 )                = T( 3 );
        view( 2, 2 )                = T( 4 );
        view.release( true );
    }
    operations.assign_matrix( matrix, copy );
    auto difference = operations.matrix_matrix_sum( T( 1 ), matrix, T( -1 ), copy );
    require( operations.matrix_norm_fro( *difference ) == T( 0 ), "matrix copy assignment failed" );
    space->assign_scalar( T( 1 ), *x );
    space->assign_scalar( T( 0 ), *y );
    operations.add_matrix_vector_prod( T( 1 ), copy, *x, T( 0 ), *y );
    require(
        space->get_value_at_point( 0, *y ) == T( 3 ) && space->get_value_at_point( 1, *y ) == T( 4 ) &&
            space->get_value_at_point( 2, *y ) == T( 4 ),
        "matrix-vector application failed"
    );
    operations.solve( copy, *y, *result );
    space->add_lin_comb( T( -1 ), *x, *result );
    const T error = space->norm_l2( *result );
    require(
        std::isfinite( error ) && error < T( 100 ) * std::numeric_limits<T>::epsilon(),
        "dense solve failed to recover the analytical solution"
    );
}

int main()
{
    try
    {
        test<float, int>();
        test<double, int>();
        test<float, std::ptrdiff_t>();
        test<double, std::ptrdiff_t>();
        std::printf( "%u SCFD backend checks, 0 failures\n", checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "SCFD backend compatibility failed: %s\n", error.what() );
        return 1;
    }
}
