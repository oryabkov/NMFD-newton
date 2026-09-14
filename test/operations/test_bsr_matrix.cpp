#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

#include <scfd/arrays/array.h>
#include <scfd/memory/host.h>
#include <scfd/backend/backend.h>

#include <nmfd/operations/bsr_matrix.h>
#include <nmfd/operations/bsr_operations.h>


using T      = double;
using Memory = scfd::backend::memory;
using Ord    = std::ptrdiff_t;

using bsr_t    = nmfd::operations::bsr_matrix<T, Ord, Memory>;
using vector_t = scfd::arrays::array<T, Memory>;

using host_memory_t = scfd::memory::host;
using host_vector_t = scfd::arrays::array<T, host_memory_t>;


// ============================================================================
// BSR matrix initialization
// ============================================================================

void fill_bsr_matrix_1x1( bsr_t &A )
{
    Ord *row_ptr = A.row_ptrs_data();
    Ord *col_ind = A.col_inds_data();
    T   *vals    = A.vals_data();
    scfd::backend::for_each<Ord>()( [=] __DEVICE_TAG__( Ord i ) { row_ptr[i] = i; }, 3 );

    scfd::backend::for_each<Ord>()(
        [=] __DEVICE_TAG__( Ord i ) {
            col_ind[i] = i;

            vals[i] = static_cast<T>( i + 2 );
        },
        2
    );

    scfd::backend::for_each<Ord>().wait();
}


void fill_bsr_matrix_2x2( bsr_t &A )
{
    /*
        A = [ 1  2 | 0  0 ]
            [ 3  4 | 0  0 ]
            -------------
            [ 0  0 | 5  6 ]
            [ 0  0 | 7  8 ]
    */
    Ord *row_ptr = A.row_ptrs_data();
    Ord *col_ind = A.col_inds_data();
    T   *vals    = A.vals_data();

    // row_ptr = [0, 1, 2]
    scfd::backend::for_each<Ord>()( [=] __DEVICE_TAG__( Ord i ) { row_ptr[i] = i; }, 3 );
    scfd::backend::for_each<Ord>()(
        [=] __DEVICE_TAG__( Ord k ) {
            col_ind[k] = k;

            if ( k == 0 )
            {
                vals[0] = static_cast<T>( 1 );
                vals[1] = static_cast<T>( 2 );
                vals[2] = static_cast<T>( 3 );
                vals[3] = static_cast<T>( 4 );
            }
            else
            {
                vals[4] = static_cast<T>( 5 );
                vals[5] = static_cast<T>( 6 );
                vals[6] = static_cast<T>( 7 );
                vals[7] = static_cast<T>( 8 );
            }
        },
        2
    );

    scfd::backend::for_each<Ord>().wait();
}


// ============================================================================
// Vector initialization
// ============================================================================

void fill_vector_1x1( vector_t &x )
{
    scfd::backend::for_each<Ord>()( [=] __DEVICE_TAG__( Ord i ) { x( i ) = static_cast<T>( i + 1 ); }, 2 );

    scfd::backend::for_each<Ord>().wait();
}


void fill_vector_2x2( vector_t &x )
{
    scfd::backend::for_each<Ord>()( [=] __DEVICE_TAG__( Ord i ) { x( i ) = 1.0; }, 4 );
    scfd::backend::for_each<Ord>().wait();
}


// ============================================================================
// Copy vector to host
// ============================================================================

#if PLATFORM_SERIAL_CPU or PLATFORM_OMP

host_vector_t copy_vector_to_host( const vector_t &x )
{
    return x;
}

#else

host_vector_t copy_vector_to_host( const vector_t &x )
{
    host_vector_t result;

    result.init( x.size() );

    Memory::copy_to_host( sizeof( T ) * x.size(), x.raw_ptr(), result.raw_ptr() );

    return result;
}

#endif


// ============================================================================
// Vector check
// ============================================================================

void check_vector( const host_vector_t &actual, const T *expected, Ord n, const char *name )
{
    const T eps = 1e-12;

    for ( Ord i = 0; i < n; ++i )
    {
        if ( std::abs( actual( i ) - expected[i] ) > eps )
        {
            std::cerr << name << ": mismatch at " << i << ": expected " << expected[i] << ", got " << actual( i )
                      << std::endl;

            throw std::runtime_error( "vector check failed" );
        }
    }

    std::cout << name << ": OK" << std::endl;
}


// ============================================================================
// BSR matrix check
// ============================================================================


void check_bsr_matrix(
    const bsr_t &A, const Ord *expected_row_ptr, const Ord *expected_col_ind, const T *expected_vals,
    Ord expected_nrows, Ord expected_ncols, Ord expected_nnzb, Ord block_sz_r, Ord block_sz_c, const char *name
)
{
    const T eps = 1e-12;

    if ( A.nrows() != expected_nrows )
        throw std::runtime_error( "wrong number of block rows" );

    if ( A.ncols() != expected_ncols )
        throw std::runtime_error( "wrong number of block columns" );

    if ( A.nnzb() != expected_nnzb )
        throw std::runtime_error( "wrong number of nonzero blocks" );

    if ( A.block_sz_r() != block_sz_r )
        throw std::runtime_error( "wrong block row size" );

    if ( A.block_sz_c() != block_sz_c )
        throw std::runtime_error( "wrong block column size" );

    for ( Ord i = 0; i < expected_nrows + 1; ++i )
    {
        if ( A.row_ptr( i ) != expected_row_ptr[i] )
        {
            std::cerr << name << ": row_ptr[" << i << "] = " << A.row_ptr( i ) << ", expected " << expected_row_ptr[i]
                      << std::endl;

            throw std::runtime_error( "BSR row_ptr check failed" );
        }
    }

    for ( Ord i = 0; i < expected_nnzb; ++i )
    {
        if ( A.col_ind( i ) != expected_col_ind[i] )
        {
            std::cerr << name << ": col_ind[" << i << "] = " << A.col_ind( i ) << ", expected " << expected_col_ind[i]
                      << std::endl;

            throw std::runtime_error( "BSR col_ind check failed" );
        }
    }

    for ( Ord k = 0; k < expected_nnzb; ++k )
    {
        for ( Ord r = 0; r < block_sz_r; ++r )
        {
            for ( Ord c = 0; c < block_sz_c; ++c )
            {
                const Ord index = k * block_sz_r * block_sz_c + r * block_sz_c + c;

                const T actual   = A.vals( k, r, c );
                const T expected = expected_vals[index];

                if ( std::abs( actual - expected ) > eps )
                {
                    std::cerr << name << ": vals(" << k << ", " << r << ", " << c << ") = " << actual << ", expected "
                              << expected << std::endl;

                    throw std::runtime_error( "BSR values check failed" );
                }
            }
        }
    }

    std::cout << name << ": OK" << std::endl;
}



// ============================================================================
// Test 1: BSR mat-vec, 1x1 blocks
// ============================================================================

void test_bsr_mat_vec_1x1()
{
    std::cout << "Test 1: BSR mat-vec product (1x1 blocks)" << std::endl;

    bsr_t A;

    A.init( 2, 2, 2, 1 );

    fill_bsr_matrix_1x1( A );

    vector_t x;
    vector_t y;

    x.init( 2 );
    y.init( 2 );

    fill_vector_1x1( x );

    nmfd::operations::bsr_mat_vec_prod<T, Ord, Memory>( A, x, y );

    host_vector_t y_host = copy_vector_to_host( y );

    const T expected[] = { 2.0, 6.0 };

    check_vector( y_host, expected, 2, "Test 1" );
}


// ============================================================================
// Test 2: BSR mat-vec, 2x2 blocks
// ============================================================================

void test_bsr_mat_vec_2x2()
{
    std::cout << "Test 2: BSR mat-vec product (2x2 blocks)" << std::endl;

    bsr_t A;

    A.init( 2, 2, 2, 2 );

    fill_bsr_matrix_2x2( A );

    vector_t x;
    vector_t y;

    x.init( 4 );
    y.init( 4 );

    fill_vector_2x2( x );

    nmfd::operations::bsr_mat_vec_prod<T, Ord, Memory>( A, x, y );

    host_vector_t y_host = copy_vector_to_host( y );

    const T expected[] = { 3.0, 7.0, 11.0, 15.0 };

    check_vector( y_host, expected, 4, "Test 2" );
}


// ============================================================================
// Test 3: BSR mat-mat, 1x1 blocks
// ============================================================================

void test_bsr_mat_mat_1x1()
{
    std::cout << "Test 3: BSR mat-mat product (1x1 blocks)" << std::endl;

    bsr_t A, B;

    /*
        A = [ 2  0 ]
            [ 0  3 ]

        B = [ 4  0 ]
            [ 0  5 ]

        C = A * B

          = [ 8   0 ]
            [ 0  15 ]
    */

    A.init( 2, 2, 2, 1 );

    A.row_ptr( 0 ) = 0;
    A.row_ptr( 1 ) = 1;
    A.row_ptr( 2 ) = 2;

    A.col_ind( 0 ) = 0;
    A.col_ind( 1 ) = 1;

    A.vals( 0, 0, 0 ) = 2.0;
    A.vals( 1, 0, 0 ) = 3.0;


    B.init( 2, 2, 2, 1 );

    B.row_ptr( 0 ) = 0;
    B.row_ptr( 1 ) = 1;
    B.row_ptr( 2 ) = 2;

    B.col_ind( 0 ) = 0;
    B.col_ind( 1 ) = 1;

    B.vals( 0, 0, 0 ) = 4.0;
    B.vals( 1, 0, 0 ) = 5.0;
    bsr_t C;
    nmfd::operations::bsr_mat_mat_prod_skeleton<T, Ord, Memory>( A, B, C );
    nmfd::operations::bsr_mat_mat_prod<T, Ord, Memory>( A, B, C );

    const Ord expected_row_ptr[] = { 0, 1, 2 };

    const Ord expected_col_ind[] = { 0, 1 };

    const T expected_vals[] = { 8.0, 15.0 };

    check_bsr_matrix( C, expected_row_ptr, expected_col_ind, expected_vals, 2, 2, 2, 1, 1, "Test 3" );
}


// ============================================================================
// Test 4: BSR mat-mat, 2x2 blocks
// ============================================================================

void test_bsr_mat_mat_2x2()
{
    std::cout << "Test 4: BSR mat-mat product (2x2 blocks)" << std::endl;

    bsr_t A, B;

    /*
        A = [ A0  0 ]
            [ 0   A1 ]

        A0 = [1 2]
             [3 4]

        A1 = [5 6]
             [7 8]


        B = [ B0  0 ]
            [ 0   B1 ]

        B0 = [1 0]
             [0 1]

        B1 = [2 0]
             [0 2]


        C = A * B

        C0 = A0

        C1 = 2 * A1
    */

    A.init( 2, 2, 2, 2 );

    A.row_ptr( 0 ) = 0;
    A.row_ptr( 1 ) = 1;
    A.row_ptr( 2 ) = 2;

    A.col_ind( 0 ) = 0;
    A.col_ind( 1 ) = 1;

    A.vals( 0, 0, 0 ) = 1.0;
    A.vals( 0, 0, 1 ) = 2.0;
    A.vals( 0, 1, 0 ) = 3.0;
    A.vals( 0, 1, 1 ) = 4.0;

    A.vals( 1, 0, 0 ) = 5.0;
    A.vals( 1, 0, 1 ) = 6.0;
    A.vals( 1, 1, 0 ) = 7.0;
    A.vals( 1, 1, 1 ) = 8.0;


    B.init( 2, 2, 2, 2 );

    B.row_ptr( 0 ) = 0;
    B.row_ptr( 1 ) = 1;
    B.row_ptr( 2 ) = 2;

    B.col_ind( 0 ) = 0;
    B.col_ind( 1 ) = 1;

    B.vals( 0, 0, 0 ) = 1.0;
    B.vals( 0, 0, 1 ) = 0.0;
    B.vals( 0, 1, 0 ) = 0.0;
    B.vals( 0, 1, 1 ) = 1.0;

    B.vals( 1, 0, 0 ) = 2.0;
    B.vals( 1, 0, 1 ) = 0.0;
    B.vals( 1, 1, 0 ) = 0.0;
    B.vals( 1, 1, 1 ) = 2.0;


    bsr_t C;
    nmfd::operations::bsr_mat_mat_prod_skeleton<T, Ord, Memory>( A, B, C );
    nmfd::operations::bsr_mat_mat_prod<T, Ord, Memory>( A, B, C );

    const Ord expected_row_ptr[] = { 0, 1, 2 };

    const Ord expected_col_ind[] = { 0, 1 };

    const T expected_vals[] = { 1.0,  2.0,  3.0,  4.0,

                                10.0, 12.0, 14.0, 16.0 };

    check_bsr_matrix( C, expected_row_ptr, expected_col_ind, expected_vals, 2, 2, 2, 2, 2, "Test 4" );
}


// ============================================================================
// Test 5: BSR mat-mat with multiple blocks
// ============================================================================

void test_bsr_mat_mat_multiple_blocks()
{
    std::cout << "Test 5: BSR mat-mat product (multiple blocks)" << std::endl;

    bsr_t A;
    bsr_t B;
    /*
        A = [ 1  1 ]
            [ 0  1 ]

        B = [ 1  0 ]
            [ 1  1 ]

        C = A * B

          = [ 2  1 ]
            [ 1  1 ]
    */

    A.init( 2, 2, 3, 1 );

    A.row_ptr( 0 ) = 0;
    A.row_ptr( 1 ) = 2;
    A.row_ptr( 2 ) = 3;

    A.col_ind( 0 ) = 0;
    A.col_ind( 1 ) = 1;
    A.col_ind( 2 ) = 1;

    A.vals( 0, 0, 0 ) = 1.0;
    A.vals( 1, 0, 0 ) = 1.0;
    A.vals( 2, 0, 0 ) = 1.0;


    B.init( 2, 2, 3, 1 );

    B.row_ptr( 0 ) = 0;
    B.row_ptr( 1 ) = 1;
    B.row_ptr( 2 ) = 3;

    B.col_ind( 0 ) = 0;
    B.col_ind( 1 ) = 0;
    B.col_ind( 2 ) = 1;

    B.vals( 0, 0, 0 ) = 1.0;
    B.vals( 1, 0, 0 ) = 1.0;
    B.vals( 2, 0, 0 ) = 1.0;


    bsr_t C;
    nmfd::operations::bsr_mat_mat_prod_skeleton<T, Ord, Memory>( A, B, C );
    nmfd::operations::bsr_mat_mat_prod<T, Ord, Memory>( A, B, C );

    const Ord expected_row_ptr[] = { 0, 2, 4 };

    const Ord expected_col_ind[] = { 0, 1, 0, 1 };

    const T expected_vals[] = { 2.0, 1.0, 1.0, 1.0 };

    check_bsr_matrix( C, expected_row_ptr, expected_col_ind, expected_vals, 2, 2, 4, 1, 1, "Test 5" );
}




// ============================================================================
// Main
// ============================================================================

int main()
{
    try
    {
        test_bsr_mat_vec_1x1();
        test_bsr_mat_vec_2x2();

#if PLATFORM_SERIAL_CPU or PLATFORM_OMP

        test_bsr_mat_mat_1x1();
        test_bsr_mat_mat_2x2();
        test_bsr_mat_mat_multiple_blocks();

#endif

        std::cout << "All tests passed." << std::endl;

        return 0;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "Test failed: " << e.what() << std::endl;

        return 1;
    }
}