#include <scfd/utils/log.h>

#include <scfd/backend/backend.h>

#include <nmfd/operations/dense_operations_base.h>

#ifndef USE_DOUBLE_PRECISION
using scalar                = float;
inline constexpr scalar eps = 1e-5f;
#else
using scalar                = double;
inline constexpr scalar eps = 1e-10;
#endif


int main( int argc, char const *args[] )
{
    using log_t        = scfd::utils::log_std;
    using T            = double;
    using backend_type = scfd::backend::current;
    using memory_type  = backend_type::memory_type;
    using dense_ops_t  = nmfd::operations::dense_operations<T, backend_type>;
    using vector_type  = typename dense_ops_t::vector_type;
    using matrix_type  = typename dense_ops_t::matrix_type;

    log_t log;
    log.info( "Testing dense_operations" );
    size_t passed_counter = 0;
    size_t failed_counter = 0;

    auto ops = std::make_shared<dense_ops_t>();

    // ====================================================================
    // GROUP 1: Reduction operations on vectors of different sizes
    // ====================================================================
    log.info( "=== Reduction: scalar_prod on size 3 ===" );
    {
        vector_type x  = { 1, 2, 3 };
        vector_type y  = { 4, 5, 6 };
        T           sp = ops->scalar_prod( x, y );
        if ( std::abs( sp - 32 ) < eps )
        {
            log.info( "PASS: scalar_prod({1,2,3}, {4,5,6}) = 32" );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: scalar_prod expected 32, got " + std::to_string( sp ) );
            failed_counter++;
        }
    }

    log.info( "=== Reduction: norm on size 5 (helper must grow) ===" );
    {
        vector_type x        = { 1, 2, 3, 4, 5 };
        T           n        = ops->norm( x );
        T           expected = std::sqrt( 1 + 4 + 9 + 16 + 25 );
        if ( std::abs( n - expected ) < eps )
        {
            log.info( "PASS: norm({1,2,3,4,5}) = " + std::to_string( expected ) );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: norm expected " + std::to_string( expected ) + ", got " + std::to_string( n ) );
            failed_counter++;
        }
    }

    log.info( "=== Reduction: sum on size 2 (helper should not shrink) ===" );
    {
        vector_type x = { 10, 20 };
        T           s = ops->sum( x );
        if ( std::abs( s - 30 ) < eps )
        {
            log.info( "PASS: sum({10, 20}) = 30" );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: sum expected 30, got " + std::to_string( s ) );
            failed_counter++;
        }
    }

    log.info( "=== Reduction: asum on size 4 ===" );
    {
        vector_type x = { 1, -2, 3, -4 };
        T           a = ops->asum( x );
        if ( std::abs( a - 10 ) < eps )
        {
            log.info( "PASS: asum({1,-2,3,-4}) = 10" );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: asum expected 10, got " + std::to_string( a ) );
            failed_counter++;
        }
    }

    log.info( "=== Reduction: norm on size 7 (helper grows again) ===" );
    {
        vector_type x        = { 1, 1, 1, 1, 1, 1, 1 };
        T           n        = ops->norm( x );
        T           expected = std::sqrt( 7.0 );
        if ( std::abs( n - expected ) < eps )
        {
            log.info( "PASS: norm(ones(7)) = " + std::to_string( expected ) );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: norm expected " + std::to_string( expected ) + ", got " + std::to_string( n ) );
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 2: Matrix-vector operations
    // ====================================================================
    log.info( "=== Test: y = mat * x (3x3) ===" );
    {
        matrix_type mat = {
            { 1, 2, 3 }, //
            { 4, 5, 6 }, //
            { 7, 8, 9 }
        };

        vector_type x = { 1, 2, 3 };
        vector_type y = { 0, 0, 0 };

        ops->add_matrix_vector_prod( 1.0, mat, x, 0.0, y );

        const auto yv = y.create_view( true );
        if ( std::abs( yv( 0 ) - 14 ) < eps && std::abs( yv( 1 ) - 32 ) < eps && std::abs( yv( 2 ) - 50 ) < eps )
        {
            log.info( "PASS: y = mat * x" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: y = mat * x. Expected {14, 32, 50} but got {" + std::to_string( yv( 0 ) ) + ", " +
                std::to_string( yv( 1 ) ) + ", " + std::to_string( yv( 2 ) ) + "}"
            );
            failed_counter++;
        }
        mat.free();
    }

    log.info( "=== Test: y = alpha*mat*x + beta*y (2x2) ===" );
    {
        matrix_type mat = {
            { 1, 0 }, //
            { 0, 1 }
        };

        vector_type x = { 3, 4 };
        vector_type y = { 1, 2 };

        ops->add_matrix_vector_prod( 2.0, mat, x, 3.0, y );

        const auto yv = y.create_view( true );
        if ( std::abs( yv( 0 ) - 9 ) < eps && std::abs( yv( 1 ) - 14 ) < eps )
        {
            log.info( "PASS: y = alpha*mat*x + beta*y" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: Expected {9, 14} but got {" + std::to_string( yv( 0 ) ) + ", " + std::to_string( yv( 1 ) ) + "}"
            );
            failed_counter++;
        }
        mat.free();
    }

    log.info( "=== Test: non-square matrix (2x3) ===" );
    {
        matrix_type mat = {
            { 1, 2, 3 }, //
            { 4, 5, 6 }
        };

        vector_type x = { 1, 2, 3 };
        vector_type y = { 0, 0 };

        ops->add_matrix_vector_prod( 1.0, mat, x, 0.0, y );

        const auto yv = y.create_view( true );
        if ( std::abs( yv( 0 ) - 14 ) < eps && std::abs( yv( 1 ) - 32 ) < eps )
        {
            log.info( "PASS: non-square (2x3) matrix" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: Expected {14, 32} but got {" + std::to_string( yv( 0 ) ) + ", " + std::to_string( yv( 1 ) ) + "}"
            );
            failed_counter++;
        }
        mat.free();
    }

    log.info( "=== Test: assign_matrix_vector_prod ===" );
    {
        matrix_type mat = {
            { 2, 0 }, //
            { 0, 2 }
        };

        vector_type x = { 3, 4 };
        vector_type y = { 10, 20 };
        vector_type z = { 0, 0 };

        ops->assign_matrix_vector_prod( 1.0, mat, x, 1.0, y, z );

        const auto zv = z.create_view( true );
        if ( std::abs( zv( 0 ) - 16 ) < eps && std::abs( zv( 1 ) - 28 ) < eps )
        {
            log.info( "PASS: assign_matrix_vector_prod" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: Expected {16, 28} but got {" + std::to_string( zv( 0 ) ) + ", " + std::to_string( zv( 1 ) ) + "}"
            );
            failed_counter++;
        }
        mat.free();
    }

    // ====================================================================
    // GROUP 3: Matrix-matrix operations
    // ====================================================================

    log.info( "=== Test: matrix_matrix_prod (2x3) * (3x2) ===" );
    {
        matrix_type A = {
            { 1, 2, 3 }, //
            { 4, 5, 6 }
        };
        matrix_type B = {
            { 1, 4 }, //
            { 2, 5 }, //
            { 3, 6 }
        };

        auto C  = ops->matrix_matrix_prod( A, B );
        auto cv = C->create_view( true );
        auto sz = C->size_nd();

        if ( sz[0] == 2 && sz[1] == 2 && std::abs( cv( 0, 0 ) - 14 ) < eps && std::abs( cv( 0, 1 ) - 32 ) < eps &&
             std::abs( cv( 1, 0 ) - 32 ) < eps && std::abs( cv( 1, 1 ) - 77 ) < eps )
        {
            log.info( "PASS: matrix_matrix_prod (2x3)*(3x2)" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: matrix_matrix_prod. Expected {{14,32},{32,77}} but got {{" + std::to_string( cv( 0, 0 ) ) + "," +
                std::to_string( cv( 0, 1 ) ) + "},{" + std::to_string( cv( 1, 0 ) ) + "," +
                std::to_string( cv( 1, 1 ) ) + "}}"
            );
            failed_counter++;
        }
        A.free();
        B.free();
        C->free();
    }

    log.info( "=== Test: matrix_matrix_prod square (2x2) * (2x2) ===" );
    {
        matrix_type A = {
            { 1, 2 }, //
            { 3, 4 }
        };
        matrix_type B = {
            { 5, 6 }, //
            { 7, 8 }
        };

        auto C  = ops->matrix_matrix_prod( A, B );
        auto cv = C->create_view( true );

        if ( std::abs( cv( 0, 0 ) - 19 ) < eps && std::abs( cv( 0, 1 ) - 22 ) < eps &&
             std::abs( cv( 1, 0 ) - 43 ) < eps && std::abs( cv( 1, 1 ) - 50 ) < eps )
        {
            log.info( "PASS: matrix_matrix_prod (2x2)*(2x2)" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: Expected {{19,22},{43,50}} but got {{" + std::to_string( cv( 0, 0 ) ) + "," +
                std::to_string( cv( 0, 1 ) ) + "},{" + std::to_string( cv( 1, 0 ) ) + "," +
                std::to_string( cv( 1, 1 ) ) + "}}"
            );
            failed_counter++;
        }
        A.free();
        B.free();
        C->free();
    }

    log.info( "=== Test: matrix_matrix_sum ===" );
    {
        matrix_type A = {
            { 1, 2 }, //
            { 3, 4 }
        };
        matrix_type B = {
            { 10, 20 }, //
            { 30, 40 }
        };

        auto C  = ops->matrix_matrix_sum( 2.0, A, 3.0, B );
        auto cv = C->create_view( true );

        if ( std::abs( cv( 0, 0 ) - 32 ) < eps && std::abs( cv( 0, 1 ) - 64 ) < eps &&
             std::abs( cv( 1, 0 ) - 96 ) < eps && std::abs( cv( 1, 1 ) - 128 ) < eps )
        {
            log.info( "PASS: matrix_matrix_sum 2*A + 3*B" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: Expected {{32,64},{96,128}} but got {{" + std::to_string( cv( 0, 0 ) ) + "," +
                std::to_string( cv( 0, 1 ) ) + "},{" + std::to_string( cv( 1, 0 ) ) + "," +
                std::to_string( cv( 1, 1 ) ) + "}}"
            );
            failed_counter++;
        }
        A.free();
        B.free();
        C->free();
    }

    log.info( "=== Test: matrix_transpose ===" );
    {
        matrix_type A = {
            { 1, 2, 3 }, //
            { 4, 5, 6 }
        };

        auto AT = ops->matrix_transpose( A );
        auto tv = AT->create_view( true );
        auto sz = AT->size_nd();

        if ( sz[0] == 3 && sz[1] == 2 && std::abs( tv( 0, 0 ) - 1 ) < eps && std::abs( tv( 0, 1 ) - 4 ) < eps &&
             std::abs( tv( 1, 0 ) - 2 ) < eps && std::abs( tv( 1, 1 ) - 5 ) < eps && std::abs( tv( 2, 0 ) - 3 ) < eps &&
             std::abs( tv( 2, 1 ) - 6 ) < eps )
        {
            log.info( "PASS: matrix_transpose (2x3) -> (3x2)" );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: matrix_transpose" );
            failed_counter++;
        }
        A.free();
        AT->free();
    }

    log.info( "=== Test: matrix_norm_fro 3x3 ===" );
    {
        matrix_type A = {
            { 5, 6, 7 },  //
            { 8, 9, 10 }, //
            { 11, 12, 13 }
        };

        T nf       = ops->matrix_norm_fro( A );
        T expected = std::sqrt( 789 );

        if ( std::abs( nf - expected ) < eps )
        {
            log.info( "PASS: matrix_norm_fro = " + std::to_string( expected ) );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: matrix_norm_fro expected " + std::to_string( expected ) + ", got " + std::to_string( nf )
            );
            failed_counter++;
        }
        A.free();
    }

    log.info( "=== Test: matrix_norm_fro 2x2 ===" );
    {
        matrix_type A = {
            { 1, 2 }, //
            { 3, 4 }
        };

        T nf       = ops->matrix_norm_fro( A );
        T expected = std::sqrt( 30.0 );

        if ( std::abs( nf - expected ) < eps )
        {
            log.info( "PASS: matrix_norm_fro = " + std::to_string( expected ) );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: matrix_norm_fro expected " + std::to_string( expected ) + ", got " + std::to_string( nf )
            );
            failed_counter++;
        }
        A.free();
    }

    log.info( "=== Test: diag_matrix_from_vector ===" );
    {
        vector_type x = { 2, 5, 7 };

        auto D  = ops->diag_matrix_from_vector( x );
        auto dv = D->create_view( true );
        auto sz = D->size_nd();

        if ( sz[0] == 3 && sz[1] == 3 && std::abs( dv( 0, 0 ) - 2 ) < eps && std::abs( dv( 1, 1 ) - 5 ) < eps &&
             std::abs( dv( 2, 2 ) - 7 ) < eps && std::abs( dv( 0, 1 ) ) < eps && std::abs( dv( 0, 2 ) ) < eps &&
             std::abs( dv( 1, 0 ) ) < eps && std::abs( dv( 1, 2 ) ) < eps && std::abs( dv( 2, 0 ) ) < eps &&
             std::abs( dv( 2, 1 ) ) < eps )
        {
            log.info( "PASS: diag_matrix_from_vector" );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: diag_matrix_from_vector" );
            failed_counter++;
        }
        D->free();
    }

    log.info( "=== Test: matrix_diag (to vector) ===" );
    {
        matrix_type A = {
            { 10, 1, 2 }, //
            { 3, 20, 4 }, //
            { 5, 6, 30 }
        };

        vector_type d = { 0, 0, 0 };
        ops->matrix_diag( A, d );
        auto dv = d.create_view( true );

        if ( std::abs( dv( 0 ) - 10 ) < eps && std::abs( dv( 1 ) - 20 ) < eps && std::abs( dv( 2 ) - 30 ) < eps )
        {
            log.info( "PASS: matrix_diag to vector" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: matrix_diag to vector. Expected {10,20,30} but got {" + std::to_string( dv( 0 ) ) + "," +
                std::to_string( dv( 1 ) ) + "," + std::to_string( dv( 2 ) ) + "}"
            );
            failed_counter++;
        }
        A.free();
    }

    log.info( "=== Test: matrix_diag (to vector, inverted) ===" );
    {
        matrix_type A = {
            { 2, 99 }, //
            { 99, 5 }
        };

        vector_type d = { 0, 0 };
        ops->matrix_diag( A, d, true );
        auto dv = d.create_view( true );

        if ( std::abs( dv( 0 ) - 0.5 ) < eps && std::abs( dv( 1 ) - 0.2 ) < eps )
        {
            log.info( "PASS: matrix_diag to vector (inverted)" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: Expected {0.5, 0.2} but got {" + std::to_string( dv( 0 ) ) + "," + std::to_string( dv( 1 ) ) +
                "}"
            );
            failed_counter++;
        }
        A.free();
    }

    log.info( "=== Test: matrix_diag (to diagonal matrix) ===" );
    {
        matrix_type A = {
            { 10, 1, 2 }, //
            { 3, 20, 4 }, //
            { 5, 6, 30 }
        };

        auto D  = ops->matrix_diag( A );
        auto dv = D->create_view( true );
        auto sz = D->size_nd();

        if ( sz[0] == 3 && sz[1] == 3 && std::abs( dv( 0, 0 ) - 10 ) < eps && std::abs( dv( 1, 1 ) - 20 ) < eps &&
             std::abs( dv( 2, 2 ) - 30 ) < eps && std::abs( dv( 0, 1 ) ) < eps && std::abs( dv( 0, 2 ) ) < eps &&
             std::abs( dv( 1, 0 ) ) < eps && std::abs( dv( 1, 2 ) ) < eps && std::abs( dv( 2, 0 ) ) < eps &&
             std::abs( dv( 2, 1 ) ) < eps )
        {
            log.info( "PASS: matrix_diag to diagonal matrix" );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: matrix_diag to diagonal matrix" );
            failed_counter++;
        }
        A.free();
        D->free();
    }

    log.info( "=== Test: scalar_matrix ===" );
    {
        vector_type x  = { 0, 0, 0 };
        auto        I  = ops->scalar_matrix( x, 3.0 );
        auto        iv = I->create_view( true );
        auto        sz = I->size_nd();

        if ( sz[0] == 3 && sz[1] == 3 && std::abs( iv( 0, 0 ) - 3 ) < eps && std::abs( iv( 1, 1 ) - 3 ) < eps &&
             std::abs( iv( 2, 2 ) - 3 ) < eps && std::abs( iv( 0, 1 ) ) < eps && std::abs( iv( 0, 2 ) ) < eps &&
             std::abs( iv( 1, 0 ) ) < eps && std::abs( iv( 1, 2 ) ) < eps && std::abs( iv( 2, 0 ) ) < eps &&
             std::abs( iv( 2, 1 ) ) < eps )
        {
            log.info( "PASS: scalar_matrix 3*I" );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: scalar_matrix" );
            failed_counter++;
        }
        I->free();
    }

    log.info( "=== Test: matrix_matrix_prod identity ===" );
    {
        matrix_type A = {
            { 1, 2 }, //
            { 3, 4 }
        };
        matrix_type I = {
            { 1, 0 }, //
            { 0, 1 }
        };

        auto C  = ops->matrix_matrix_prod( A, I );
        auto cv = C->create_view( true );

        if ( std::abs( cv( 0, 0 ) - 1 ) < eps && std::abs( cv( 0, 1 ) - 2 ) < eps && std::abs( cv( 1, 0 ) - 3 ) < eps &&
             std::abs( cv( 1, 1 ) - 4 ) < eps )
        {
            log.info( "PASS: A * I = A" );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: A * I != A" );
            failed_counter++;
        }
        A.free();
        I.free();
        C->free();
    }

    log.info( "=== Test: (A^T)^T = A ===" );
    {
        matrix_type A = {
            { 1, 2, 3 }, //
            { 4, 5, 6 }
        };

        auto AT  = ops->matrix_transpose( A );
        auto ATT = ops->matrix_transpose( *AT );
        auto sz  = ATT->size_nd();
        auto v   = ATT->create_view( true );

        if ( sz[0] == 2 && sz[1] == 3 && std::abs( v( 0, 0 ) - 1 ) < eps && std::abs( v( 0, 1 ) - 2 ) < eps &&
             std::abs( v( 0, 2 ) - 3 ) < eps && std::abs( v( 1, 0 ) - 4 ) < eps && std::abs( v( 1, 1 ) - 5 ) < eps &&
             std::abs( v( 1, 2 ) - 6 ) < eps )
        {
            log.info( "PASS: (A^T)^T = A" );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: (A^T)^T != A" );
            failed_counter++;
        }
        A.free();
        AT->free();
        ATT->free();
    }

    log.info( "=== Test: norm_fro of identity ===" );
    {
        matrix_type I = {
            { 1, 0, 0 }, //
            { 0, 1, 0 }, //
            { 0, 0, 1 }
        };

        T nf       = ops->matrix_norm_fro( I );
        T expected = std::sqrt( 3.0 );

        if ( std::abs( nf - expected ) < eps )
        {
            log.info( "PASS: ||I_3||_F = sqrt(3)" );
            passed_counter++;
        }
        else
        {
            log.error( "FAIL: expected " + std::to_string( expected ) + ", got " + std::to_string( nf ) );
            failed_counter++;
        }
        I.free();
    }

    // ====================================================================
    // FINAL SUMMARY
    // ====================================================================
    log.info( "================================================" );
    log.info( "=== TEST SUMMARY ===" );
    log.info( "Passed: " + std::to_string( passed_counter ) );
    log.info( "Failed: " + std::to_string( failed_counter ) );
    log.info( "Total:  " + std::to_string( passed_counter + failed_counter ) );
    if ( failed_counter == 0 )
    {
        log.info( "All tests passed!" );
    }
    else
    {
        log.info( "Some tests FAILED." );
    }
    log.info( "================================================" );

    return ( failed_counter == 0 ) ? 0 : 1;
}
