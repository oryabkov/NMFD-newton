#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

#include <scfd/arrays/array.h>
#include <scfd/backend/backend.h>

#include <nmfd/operations/bsr_matrix.h>


#if PLATFORM_CUDA
#    include <nmfd/operations/bsr_operations_cuda.h>
#else
#    include <nmfd/operations/bsr_operations.h>
#endif


using T      = double;
using Memory = scfd::backend::memory;
using Ord    = std::ptrdiff_t;

using bsr_t    = nmfd::operations::bsr_matrix<T, Ord, Memory>;
using vector_t = scfd::arrays::array<T, Memory>;


// ============================================================================
// BSR matrix initialization
// ============================================================================

void fill_bsr_matrix_1x1( bsr_t &A )
{
    auto row_ptr = A.create_row_ptrs_view( false );
    auto col_ind = A.create_col_inds_view( false );
    auto vals    = A.create_vals_view( false );

    for ( Ord i = 0; i < 3; ++i )
    {
        row_ptr( i ) = i;
    }

    for ( Ord i = 0; i < 2; ++i )
    {
        col_ind( i ) = i;

        vals( i, 0, 0 ) = static_cast<T>( i + 2 );
    }

    row_ptr.release( true );
    col_ind.release( true );
    vals.release( true );
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
    auto row_ptr = A.create_row_ptrs_view( false );
    auto col_ind = A.create_col_inds_view( false );
    auto vals    = A.create_vals_view( false );

    // row_ptr = [0, 1, 2]
    for ( Ord i = 0; i < 3; ++i )
    {
        row_ptr( i ) = i;
    }

    const T block_vals[2][4] = { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 } };

    for ( Ord k = 0; k < 2; ++k )
    {
        col_ind( k ) = k;

        for ( Ord r = 0; r < 2; ++r )
        {
            for ( Ord c = 0; c < 2; ++c )
            {
                vals( k, r, c ) = block_vals[k][r * 2 + c];
            }
        }
    }

    row_ptr.release( true );
    col_ind.release( true );
    vals.release( true );
}


void fill_bsr_matrix_2x3( bsr_t &A )
{
    /*
        A = [ 1  2  3 ]   (block 0, block row 0)
            [ 4  5  6 ]
            [ 7  8  9 ]   (block 1, block row 1)
            [ 10 11 12 ]
    */
    auto row_ptr = A.create_row_ptrs_view( false );
    auto col_ind = A.create_col_inds_view( false );
    auto vals    = A.create_vals_view( false );

    // row_ptr = [0, 1, 2]
    for ( Ord i = 0; i < 3; ++i )
    {
        row_ptr( i ) = i;
    }

    const T block_vals[2][6] = { { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 } };

    for ( Ord k = 0; k < 2; ++k )
    {
        col_ind( k ) = 0;

        for ( Ord r = 0; r < 2; ++r )
        {
            for ( Ord c = 0; c < 3; ++c )
            {
                vals( k, r, c ) = block_vals[k][r * 3 + c];
            }
        }
    }

    row_ptr.release( true );
    col_ind.release( true );
    vals.release( true );
}


void fill_bsr_matrix_diag_1x1( bsr_t &A, T d0, T d1 )
{
    auto row_ptr = A.create_row_ptrs_view( false );
    auto col_ind = A.create_col_inds_view( false );
    auto vals    = A.create_vals_view( false );

    row_ptr( 0 ) = 0;
    row_ptr( 1 ) = 1;
    row_ptr( 2 ) = 2;

    col_ind( 0 ) = 0;
    col_ind( 1 ) = 1;

    vals( 0, 0, 0 ) = d0;
    vals( 1, 0, 0 ) = d1;

    row_ptr.release( true );
    col_ind.release( true );
    vals.release( true );
}


void fill_bsr_matrix_diag_2x2( bsr_t &A, const T block0[4], const T block1[4] )
{
    auto row_ptr = A.create_row_ptrs_view( false );
    auto col_ind = A.create_col_inds_view( false );
    auto vals    = A.create_vals_view( false );

    row_ptr( 0 ) = 0;
    row_ptr( 1 ) = 1;
    row_ptr( 2 ) = 2;

    col_ind( 0 ) = 0;
    col_ind( 1 ) = 1;

    for ( Ord r = 0; r < 2; ++r )
    {
        for ( Ord c = 0; c < 2; ++c )
        {
            vals( 0, r, c ) = block0[r * 2 + c];
            vals( 1, r, c ) = block1[r * 2 + c];
        }
    }

    row_ptr.release( true );
    col_ind.release( true );
    vals.release( true );
}


void fill_bsr_matrix_test5_A( bsr_t &A )
{
    auto row_ptr = A.create_row_ptrs_view( false );
    auto col_ind = A.create_col_inds_view( false );
    auto vals    = A.create_vals_view( false );

    row_ptr( 0 ) = 0;
    row_ptr( 1 ) = 2;
    row_ptr( 2 ) = 3;

    col_ind( 0 ) = 0;
    col_ind( 1 ) = 1;
    col_ind( 2 ) = 1;

    vals( 0, 0, 0 ) = 1.0;
    vals( 1, 0, 0 ) = 1.0;
    vals( 2, 0, 0 ) = 1.0;

    row_ptr.release( true );
    col_ind.release( true );
    vals.release( true );
}


void fill_bsr_matrix_test5_B( bsr_t &B )
{
    auto row_ptr = B.create_row_ptrs_view( false );
    auto col_ind = B.create_col_inds_view( false );
    auto vals    = B.create_vals_view( false );

    row_ptr( 0 ) = 0;
    row_ptr( 1 ) = 1;
    row_ptr( 2 ) = 3;

    col_ind( 0 ) = 0;
    col_ind( 1 ) = 0;
    col_ind( 2 ) = 1;

    vals( 0, 0, 0 ) = 1.0;
    vals( 1, 0, 0 ) = 1.0;
    vals( 2, 0, 0 ) = 1.0;

    row_ptr.release( true );
    col_ind.release( true );
    vals.release( true );
}


// ============================================================================
// Vector initialization
// ============================================================================

struct fill_vector_iota
{
    vector_t x;

    __DEVICE_TAG__ void operator()( Ord i ) const
    {
        x( i ) = static_cast<T>( i + 1 );
    }
};


struct fill_vector_const
{
    vector_t x;
    T        value;

    __DEVICE_TAG__ void operator()( Ord i ) const
    {
        x( i ) = value;
    }
};


void fill_vector_1x1( vector_t &x )
{
    scfd::backend::for_each<Ord>()( fill_vector_iota{ x }, 2 );

    scfd::backend::for_each<Ord>().wait();
}


void fill_vector_2x2( vector_t &x )
{
    scfd::backend::for_each<Ord>()( fill_vector_const{ x, 1.0 }, 4 );

    scfd::backend::for_each<Ord>().wait();
}


void fill_vector_3( vector_t &x )
{
    scfd::backend::for_each<Ord>()( fill_vector_const{ x, 1.0 }, 3 );

    scfd::backend::for_each<Ord>().wait();
}


// ============================================================================
// Vector check
// ============================================================================

template <class VectorView>
void check_vector( const VectorView &actual, const T *expected, Ord n, const char *name )
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

    auto row_ptrs = A.create_row_ptrs_view( true );
    auto col_inds = A.create_col_inds_view( true );
    auto vals     = A.create_vals_view( true );

    for ( Ord i = 0; i < expected_nrows + 1; ++i )
    {
        if ( row_ptrs( i ) != expected_row_ptr[i] )
        {
            std::cerr << name << ": row_ptr[" << i << "] = " << row_ptrs( i ) << ", expected " << expected_row_ptr[i]
                      << std::endl;

            throw std::runtime_error( "BSR row_ptr check failed" );
        }
    }

    for ( Ord i = 0; i < expected_nnzb; ++i )
    {
        if ( col_inds( i ) != expected_col_ind[i] )
        {
            std::cerr << name << ": col_ind[" << i << "] = " << col_inds( i ) << ", expected " << expected_col_ind[i]
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

                const T actual   = vals( k, r, c );
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
// Platform dispatch
// ============================================================================

void bsr_mat_vec_prod_dispatch( const bsr_t &A, const vector_t &x, vector_t &y )
{
#if PLATFORM_CUDA
    nmfd::operations::bsr_mat_vec_prod_cuda<T, Ord, Memory>( A, x, y );
#else
    nmfd::operations::bsr_mat_vec_prod<T, Ord, Memory>( A, x, y );
#endif
}


void bsr_mat_mat_prod_dispatch( const bsr_t &A, const bsr_t &B, bsr_t &C )
{
#if PLATFORM_CUDA
    nmfd::operations::bsr_mat_mat_prod_skeleton_cuda<T, Ord>( A, B, C );
    nmfd::operations::bsr_mat_mat_prod_cuda<T, Ord>( A, B, C );
#else
    nmfd::operations::bsr_mat_mat_prod_skeleton<T, Ord, Memory>( A, B, C );
    nmfd::operations::bsr_mat_mat_prod<T, Ord, Memory>( A, B, C );
#endif
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

    bsr_mat_vec_prod_dispatch( A, x, y );

    auto y_host = y.create_view( true );

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

    bsr_mat_vec_prod_dispatch( A, x, y );

    auto y_host = y.create_view( true );

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
    fill_bsr_matrix_diag_1x1( A, 2.0, 3.0 );

    B.init( 2, 2, 2, 1 );
    fill_bsr_matrix_diag_1x1( B, 4.0, 5.0 );

    bsr_t C;
    bsr_mat_mat_prod_dispatch( A, B, C );

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

    const T a0[4] = { 1.0, 2.0, 3.0, 4.0 };
    const T a1[4] = { 5.0, 6.0, 7.0, 8.0 };

    fill_bsr_matrix_diag_2x2( A, a0, a1 );

    B.init( 2, 2, 2, 2 );

    const T b0[4] = { 1.0, 0.0, 0.0, 1.0 };
    const T b1[4] = { 2.0, 0.0, 0.0, 2.0 };

    fill_bsr_matrix_diag_2x2( B, b0, b1 );

    bsr_t C;
    bsr_mat_mat_prod_dispatch( A, B, C );

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
    fill_bsr_matrix_test5_A( A );

    B.init( 2, 2, 3, 1 );
    fill_bsr_matrix_test5_B( B );

    bsr_t C;
    bsr_mat_mat_prod_dispatch( A, B, C );

    const Ord expected_row_ptr[] = { 0, 2, 4 };

    const Ord expected_col_ind[] = { 0, 1, 0, 1 };

    const T expected_vals[] = { 2.0, 1.0, 1.0, 1.0 };

    check_bsr_matrix( C, expected_row_ptr, expected_col_ind, expected_vals, 2, 2, 4, 1, 1, "Test 5" );
}


// ============================================================================
// Test 6: BSR mat-vec, rectangular 2x3 blocks
// ============================================================================

void test_bsr_mat_vec_rect()
{
    std::cout << "Test 6: BSR mat-vec product (2x3 blocks)" << std::endl;

    bsr_t A;

    A.init( 2, 1, 2, 2, 3 );

    fill_bsr_matrix_2x3( A );

    vector_t x;
    vector_t y;

    x.init( 3 );
    y.init( 4 );

    fill_vector_3( x );

    bsr_mat_vec_prod_dispatch( A, x, y );

    auto y_host = y.create_view( true );

    const T expected[] = { 6.0, 15.0, 24.0, 33.0 };

    check_vector( y_host, expected, 4, "Test 6" );
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
        test_bsr_mat_mat_1x1();
        test_bsr_mat_mat_2x2();
        test_bsr_mat_mat_multiple_blocks();
        test_bsr_mat_vec_rect();

        std::cout << "All tests passed." << std::endl;

        return 0;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "Test failed: " << e.what() << std::endl;

        return 1;
    }
}