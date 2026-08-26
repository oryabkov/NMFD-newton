#include <cmath>
#include <iostream>
#include <cstddef>

#include <scfd/arrays/array.h>
#include <scfd/memory/host.h>

#include <nmfd/operations/bsr_matrix.h>
#include <nmfd/operations/bsr_operations.h>

using T        = double;
using Memory   = scfd::memory::host;
using bsr_t    = nmfd::operations::bsr_matrix<T, Memory>;
using vector_t = scfd::arrays::array<T, Memory>;

void print_bsr( const bsr_t &A, const char *name )
{
    std::cout << "=== " << name << " ===" << std::endl;

    std::cout << "  nrows=" << A.nrows() << " ncols=" << A.ncols() << " nnzb=" << A.nnzb()
              << " block_sz_r=" << A.block_sz_r() << " block_sz_c=" << A.block_sz_c() << std::endl;

    for ( std::ptrdiff_t i = 0; i < A.nrows(); ++i )
    {
        std::cout << "  row " << i << ":";

        for ( std::ptrdiff_t k = A.row_ptr( i ); k < A.row_ptr( i + 1 ); ++k )
        {
            const std::ptrdiff_t col = A.col_ind( k );

            std::cout << " [col=" << col << " block=(";

            for ( std::ptrdiff_t r = 0; r < A.block_sz_r(); ++r )
            {
                for ( std::ptrdiff_t c = 0; c < A.block_sz_c(); ++c )
                {
                    if ( r > 0 || c > 0 )
                    {
                        std::cout << ",";
                    }

                    std::cout << A.block_val( k, r, c );
                }
            }

            std::cout << ")]";
        }

        std::cout << std::endl;
    }
}

bool check_vector( const vector_t &actual, const T *expected, std::ptrdiff_t size )
{
    bool ok = true;

    if ( actual.size() != size )
    {
        std::cerr << "  FAIL: vector size = " << actual.size() << ", expected " << size << std::endl;

        return false;
    }

    for ( std::ptrdiff_t i = 0; i < size; ++i )
    {
        if ( std::abs( actual( i ) - expected[i] ) > 1e-12 )
        {
            std::cerr << "  FAIL: y[" << i << "] = " << actual( i ) << ", expected " << expected[i] << std::endl;

            ok = false;
        }
    }

    return ok;
}

bool check_identity_block( const bsr_t &A, std::ptrdiff_t k, T diagonal_value )
{
    bool ok = true;

    for ( std::ptrdiff_t r = 0; r < A.block_sz_r(); ++r )
    {
        for ( std::ptrdiff_t c = 0; c < A.block_sz_c(); ++c )
        {
            const T expected = ( r == c ) ? diagonal_value : T( 0 );

            const T actual = A.block_val( k, r, c );

            if ( std::abs( actual - expected ) > 1e-12 )
            {
                std::cerr << "  FAIL: block k=" << k << " [" << r << "," << c << "] = " << actual << ", expected "
                          << expected << std::endl;

                ok = false;
            }
        }
    }

    return ok;
}

int main()
{
    int errors = 0;

    // ====================================================================
    // Test 1: BSR matrix-vector product (1x1 blocks, diagonal)
    // ====================================================================

    std::cout << "Test 1: BSR mat-vec product (1x1 blocks)" << std::endl;

    // A = [ 2  0 ]
    //     [ 0  3 ]

    bsr_t A1;

    A1.init( 2, 2, 2, 1 );

    A1.row_ptr( 0 ) = 0;
    A1.row_ptr( 1 ) = 1;
    A1.row_ptr( 2 ) = 2;

    A1.col_ind( 0 ) = 0;
    A1.col_ind( 1 ) = 1;

    A1.block_val( 0, 0, 0 ) = 2.0;
    A1.block_val( 1, 0, 0 ) = 3.0;

    vector_t x1;
    vector_t y1;

    x1.init( 2 );
    y1.init( 2 );

    x1( 0 ) = 1.0;
    x1( 1 ) = 2.0;

    nmfd::operations::bsr_mat_vec_prod( A1, x1, y1 );

    // Expected:
    //
    // y = [2, 6]

    const T expected1[2] = { 2.0, 6.0 };

    if ( check_vector( y1, expected1, 2 ) )
    {
        std::cout << "  PASS" << std::endl;
    }
    else
    {
        ++errors;
    }


    // ====================================================================
    // Test 2: BSR matrix-vector product (2x2 block)
    // ====================================================================

    std::cout << "Test 2: BSR mat-vec product (2x2 blocks)" << std::endl;

    // A = [ 1 2 ]
    //     [ 3 4 ]

    bsr_t A2;

    A2.init( 1, 1, 1, 2 );

    A2.row_ptr( 0 ) = 0;
    A2.row_ptr( 1 ) = 1;

    A2.col_ind( 0 ) = 0;

    A2.block_val( 0, 0, 0 ) = 1.0;
    A2.block_val( 0, 0, 1 ) = 2.0;
    A2.block_val( 0, 1, 0 ) = 3.0;
    A2.block_val( 0, 1, 1 ) = 4.0;

    vector_t x2;
    vector_t y2;

    x2.init( 2 );
    y2.init( 2 );

    x2( 0 ) = 1.0;
    x2( 1 ) = 2.0;

    nmfd::operations::bsr_mat_vec_prod( A2, x2, y2 );

    // Expected:
    //
    // [1 2] [1]   [ 5]
    // [3 4] [2] = [11]

    const T expected2[2] = { 5.0, 11.0 };

    if ( check_vector( y2, expected2, 2 ) )
    {
        std::cout << "  PASS" << std::endl;
    }
    else
    {
        ++errors;
    }


    // ====================================================================
    // Test 3: BSR matrix-matrix product
    // ====================================================================

    std::cout << "Test 3: BSR mat-mat product" << std::endl;

    // A = [ I2  I2 ]
    //     [ I2   0 ]
    //
    // B = [ I2  0  ]
    //     [ 0   I2 ]
    //
    // C = A * B
    //
    //   = [ I2  I2 ]
    //     [ I2   0 ]

    bsr_t Amat;

    Amat.init( 2, 2, 3, 2 );

    Amat.row_ptr( 0 ) = 0;
    Amat.row_ptr( 1 ) = 2;
    Amat.row_ptr( 2 ) = 3;

    Amat.col_ind( 0 ) = 0;
    Amat.col_ind( 1 ) = 1;
    Amat.col_ind( 2 ) = 0;

    // A[0,0] = I2

    Amat.block_val( 0, 0, 0 ) = 1.0;
    Amat.block_val( 0, 0, 1 ) = 0.0;
    Amat.block_val( 0, 1, 0 ) = 0.0;
    Amat.block_val( 0, 1, 1 ) = 1.0;

    // A[0,1] = I2

    Amat.block_val( 1, 0, 0 ) = 1.0;
    Amat.block_val( 1, 0, 1 ) = 0.0;
    Amat.block_val( 1, 1, 0 ) = 0.0;
    Amat.block_val( 1, 1, 1 ) = 1.0;

    // A[1,0] = I2

    Amat.block_val( 2, 0, 0 ) = 1.0;
    Amat.block_val( 2, 0, 1 ) = 0.0;
    Amat.block_val( 2, 1, 0 ) = 0.0;
    Amat.block_val( 2, 1, 1 ) = 1.0;


    bsr_t Bmat;

    Bmat.init( 2, 2, 2, 2 );

    Bmat.row_ptr( 0 ) = 0;
    Bmat.row_ptr( 1 ) = 1;
    Bmat.row_ptr( 2 ) = 2;

    Bmat.col_ind( 0 ) = 0;
    Bmat.col_ind( 1 ) = 1;

    // B[0,0] = I2

    Bmat.block_val( 0, 0, 0 ) = 1.0;
    Bmat.block_val( 0, 0, 1 ) = 0.0;
    Bmat.block_val( 0, 1, 0 ) = 0.0;
    Bmat.block_val( 0, 1, 1 ) = 1.0;

    // B[1,1] = I2

    Bmat.block_val( 1, 0, 0 ) = 1.0;
    Bmat.block_val( 1, 0, 1 ) = 0.0;
    Bmat.block_val( 1, 1, 0 ) = 0.0;
    Bmat.block_val( 1, 1, 1 ) = 1.0;


    bsr_t Cmat;

    nmfd::operations::bsr_mat_mat_prod_skeleton( Amat, Bmat, Cmat );

    bool pattern_ok = true;

    // Expected C pattern:
    //
    // row 0: columns 0, 1
    // row 1: column 0
    //
    // row_ptr = [0, 2, 3]
    // col_ind = [0, 1, 0]

    const std::ptrdiff_t expected_row_ptr[3] = { 0, 2, 3 };

    const std::ptrdiff_t expected_col_ind[3] = { 0, 1, 0 };

    if ( Cmat.nnzb() != 3 )
    {
        std::cerr << "  FAIL: C.nnzb() = " << Cmat.nnzb() << ", expected 3" << std::endl;

        pattern_ok = false;
        ++errors;
    }

    for ( std::ptrdiff_t i = 0; i <= 2; ++i )
    {
        if ( Cmat.row_ptr( i ) != expected_row_ptr[i] )
        {
            std::cerr << "  FAIL: C.row_ptr(" << i << ") = " << Cmat.row_ptr( i ) << ", expected "
                      << expected_row_ptr[i] << std::endl;

            pattern_ok = false;
            ++errors;
        }
    }

    for ( std::ptrdiff_t k = 0; k < 3; ++k )
    {
        if ( Cmat.col_ind( k ) != expected_col_ind[k] )
        {
            std::cerr << "  FAIL: C.col_ind(" << k << ") = " << Cmat.col_ind( k ) << ", expected "
                      << expected_col_ind[k] << std::endl;

            pattern_ok = false;
            ++errors;
        }
    }

    if ( pattern_ok )
    {
        nmfd::operations::bsr_mat_mat_prod( Amat, Bmat, Cmat );

        bool values_ok = true;

        // Check all three blocks.
        //
        // C[0,0] = I
        // C[0,1] = I
        // C[1,0] = I

        for ( std::ptrdiff_t i = 0; i < Cmat.nrows(); ++i )
        {
            for ( std::ptrdiff_t k = Cmat.row_ptr( i ); k < Cmat.row_ptr( i + 1 ); ++k )
            {
                if ( !check_identity_block( Cmat, k, 1.0 ) )
                {
                    values_ok = false;
                    ++errors;
                }
            }
        }

        if ( values_ok )
        {
            std::cout << "  PASS" << std::endl;
        }
    }


    // ====================================================================
    // Test 4: BSR matrix-vector product
    //         4x4 tridiagonal, 1x1 blocks
    // ====================================================================

    std::cout << "Test 4: BSR mat-vec (4x4 tridiagonal)" << std::endl;

    // A =
    //
    // [ 2 -1  0  0 ]
    // [-1  2 -1  0 ]
    // [ 0 -1  2 -1 ]
    // [ 0  0 -1  2 ]

    bsr_t A4;

    A4.init( 4, 4, 10, 1 );

    A4.row_ptr( 0 ) = 0;
    A4.row_ptr( 1 ) = 2;
    A4.row_ptr( 2 ) = 5;
    A4.row_ptr( 3 ) = 8;
    A4.row_ptr( 4 ) = 10;

    A4.col_ind( 0 )         = 0;
    A4.block_val( 0, 0, 0 ) = 2.0;

    A4.col_ind( 1 )         = 1;
    A4.block_val( 1, 0, 0 ) = -1.0;

    A4.col_ind( 2 )         = 0;
    A4.block_val( 2, 0, 0 ) = -1.0;

    A4.col_ind( 3 )         = 1;
    A4.block_val( 3, 0, 0 ) = 2.0;

    A4.col_ind( 4 )         = 2;
    A4.block_val( 4, 0, 0 ) = -1.0;

    A4.col_ind( 5 )         = 1;
    A4.block_val( 5, 0, 0 ) = -1.0;

    A4.col_ind( 6 )         = 2;
    A4.block_val( 6, 0, 0 ) = 2.0;

    A4.col_ind( 7 )         = 3;
    A4.block_val( 7, 0, 0 ) = -1.0;

    A4.col_ind( 8 )         = 2;
    A4.block_val( 8, 0, 0 ) = -1.0;

    A4.col_ind( 9 )         = 3;
    A4.block_val( 9, 0, 0 ) = 2.0;

    vector_t x4;
    vector_t y4;

    x4.init( 4 );
    y4.init( 4 );

    x4( 0 ) = 1.0;
    x4( 1 ) = 2.0;
    x4( 2 ) = 3.0;
    x4( 3 ) = 4.0;

    nmfd::operations::bsr_mat_vec_prod( A4, x4, y4 );

    // Expected:
    //
    // [ 2 -1  0  0 ] [1]   [0]
    // [-1  2 -1  0 ] [2]   [0]
    // [ 0 -1  2 -1 ] [3] = [0]
    // [ 0  0 -1  2 ] [4]   [5]

    const T expected4[4] = { 0.0, 0.0, 0.0, 5.0 };

    if ( check_vector( y4, expected4, 4 ) )
    {
        std::cout << "  PASS" << std::endl;
    }
    else
    {
        ++errors;
    }


    // ====================================================================
    // Test 5: BSR matrix-matrix product
    //         accumulation into the same block
    // ====================================================================

    std::cout << "Test 5: BSR mat-mat product with accumulation" << std::endl;

    // A = [ I  I ]
    //
    // B = [ I ]
    //     [ I ]
    //
    // C = A * B
    //
    //   = I*I + I*I
    //
    //   = 2I

    bsr_t A5;

    A5.init( 1, 2, 2, 2 );

    A5.row_ptr( 0 ) = 0;
    A5.row_ptr( 1 ) = 2;

    A5.col_ind( 0 ) = 0;
    A5.col_ind( 1 ) = 1;

    // A[0,0] = I2

    A5.block_val( 0, 0, 0 ) = 1.0;
    A5.block_val( 0, 0, 1 ) = 0.0;
    A5.block_val( 0, 1, 0 ) = 0.0;
    A5.block_val( 0, 1, 1 ) = 1.0;

    // A[0,1] = I2

    A5.block_val( 1, 0, 0 ) = 1.0;
    A5.block_val( 1, 0, 1 ) = 0.0;
    A5.block_val( 1, 1, 0 ) = 0.0;
    A5.block_val( 1, 1, 1 ) = 1.0;


    bsr_t B5;

    B5.init( 2, 1, 2, 2 );

    B5.row_ptr( 0 ) = 0;
    B5.row_ptr( 1 ) = 1;
    B5.row_ptr( 2 ) = 2;

    B5.col_ind( 0 ) = 0;
    B5.col_ind( 1 ) = 0;

    // B[0,0] = I2

    B5.block_val( 0, 0, 0 ) = 1.0;
    B5.block_val( 0, 0, 1 ) = 0.0;
    B5.block_val( 0, 1, 0 ) = 0.0;
    B5.block_val( 0, 1, 1 ) = 1.0;

    // B[1,0] = I2

    B5.block_val( 1, 0, 0 ) = 1.0;
    B5.block_val( 1, 0, 1 ) = 0.0;
    B5.block_val( 1, 1, 0 ) = 0.0;
    B5.block_val( 1, 1, 1 ) = 1.0;


    bsr_t C5;

    nmfd::operations::bsr_mat_mat_prod_skeleton( A5, B5, C5 );

    bool pattern5_ok = true;

    if ( C5.nnzb() != 1 )
    {
        std::cerr << "  FAIL: C.nnzb() = " << C5.nnzb() << ", expected 1" << std::endl;

        pattern5_ok = false;
        ++errors;
    }

    if ( C5.row_ptr( 0 ) != 0 )
    {
        std::cerr << "  FAIL: C.row_ptr(0) = " << C5.row_ptr( 0 ) << ", expected 0" << std::endl;

        pattern5_ok = false;
        ++errors;
    }

    if ( C5.row_ptr( 1 ) != 1 )
    {
        std::cerr << "  FAIL: C.row_ptr(1) = " << C5.row_ptr( 1 ) << ", expected 1" << std::endl;

        pattern5_ok = false;
        ++errors;
    }

    if ( C5.col_ind( 0 ) != 0 )
    {
        std::cerr << "  FAIL: C.col_ind(0) = " << C5.col_ind( 0 ) << ", expected 0" << std::endl;

        pattern5_ok = false;
        ++errors;
    }

    if ( pattern5_ok )
    {
        nmfd::operations::bsr_mat_mat_prod( A5, B5, C5 );

        bool values5_ok = true;

        // Expected:
        //
        // [2 0]
        // [0 2]

        for ( std::ptrdiff_t r = 0; r < 2; ++r )
        {
            for ( std::ptrdiff_t c = 0; c < 2; ++c )
            {
                const T expected = ( r == c ) ? 2.0 : 0.0;

                const T actual = C5.block_val( 0, r, c );

                if ( std::abs( actual - expected ) > 1e-12 )
                {
                    std::cerr << "  FAIL: C[0,0][" << r << "," << c << "] = " << actual << ", expected " << expected
                              << std::endl;

                    values5_ok = false;
                    ++errors;
                }
            }
        }

        if ( values5_ok )
        {
            std::cout << "  PASS" << std::endl;
        }
    }


    // ====================================================================
    // Result
    // ====================================================================

    std::cout << std::endl;

    if ( errors == 0 )
    {
        std::cout << "All tests passed!" << std::endl;
    }
    else
    {
        std::cerr << errors << " test(s) failed!" << std::endl;
    }

    return errors;
}