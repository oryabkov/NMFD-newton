#include "scfd_array_traits.h"

#include <cmath>
#include <memory>
#include <nmfd/operations/dense_operations_base.h>
#include <scfd/arrays/array.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/utils/log.h>

#include <scfd/backend/backend.h>

const double eps = 1e-10;

int main( int argc, char const *args[] )
{
    using log_t         = scfd::utils::log_std;
    using T             = double;
    using backend_type  = scfd::backend::current;
    using memory_type   = backend_type::memory_type;
    using vector_type   = scfd::arrays::array<T, memory_type>;
    using vector_traits = scfd_array_traits<T, memory_type>;
    using dense_ops_t   = nmfd::operations::dense_operations<T, vector_traits, backend_type>;
    using matrix_type   = typename dense_ops_t::matrix_type;

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
        matrix_type mat;
        mat.init( 3, 3 );
        {
            auto mv    = mat.create_view( false );
            mv( 0, 0 ) = 1;
            mv( 0, 1 ) = 2;
            mv( 0, 2 ) = 3;
            mv( 1, 0 ) = 4;
            mv( 1, 1 ) = 5;
            mv( 1, 2 ) = 6;
            mv( 2, 0 ) = 7;
            mv( 2, 1 ) = 8;
            mv( 2, 2 ) = 9;
            mv.release( true );
        }

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
        matrix_type mat;
        mat.init( 2, 2 );
        {
            auto mv    = mat.create_view( false );
            mv( 0, 0 ) = 1;
            mv( 0, 1 ) = 0;
            mv( 1, 0 ) = 0;
            mv( 1, 1 ) = 1;
            mv.release( true );
        }

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
        matrix_type mat;
        mat.init( 2, 3 );
        {
            auto mv    = mat.create_view( false );
            mv( 0, 0 ) = 1;
            mv( 0, 1 ) = 2;
            mv( 0, 2 ) = 3;
            mv( 1, 0 ) = 4;
            mv( 1, 1 ) = 5;
            mv( 1, 2 ) = 6;
            mv.release( true );
        }

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
        matrix_type mat;
        mat.init( 2, 2 );
        {
            auto mv    = mat.create_view( false );
            mv( 0, 0 ) = 2;
            mv( 0, 1 ) = 0;
            mv( 1, 0 ) = 0;
            mv( 1, 1 ) = 2;
            mv.release( true );
        }

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
