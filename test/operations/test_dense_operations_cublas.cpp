#include <cmath>
#include <memory>
#include <nmfd/operations/dense_operations_cuda.h>
#include <scfd/arrays/array.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/external_libraries/cublas_wrap_singleton_impl.h>
#include <scfd/utils/log.h>

const double eps = 1e-10;

#ifndef USE_DOUBLE_PRECISION
using scalar = float;
#else
using scalar = double;
#endif

int main( int argc, char const *args[] )
{
    using log_t       = scfd::utils::log_std;
    using T           = double;
    using dense_ops_t = nmfd::operations::dense_operations_cuda<T>;
    using vector_type = typename dense_ops_t::vector_type;
    using matrix_type = typename dense_ops_t::matrix_type;

    scfd::cublas_wrap cublas( true );

    log_t  log;
    size_t passed_counter = 0;
    size_t failed_counter = 0;

    log.info( "Testing dense_operations_cuda (cuBLAS)" );

    auto ops = std::make_shared<dense_ops_t>();

    // ====================================================================
    // Test: matrix_matrix_prod via cuBLAS gemm
    // ====================================================================

    log.info( "=== Test: gemm (2x3) * (3x2) ===" );
    {
        matrix_type A = { { 1, 2, 3 }, { 4, 5, 6 } };
        matrix_type B = { { 1, 4 }, { 2, 5 }, { 3, 6 } };

        auto C  = ops->matrix_matrix_prod( A, B );
        auto cv = C->create_view( true );
        auto sz = C->size_nd();

        if ( sz[0] == 2 && sz[1] == 2 && std::abs( cv( 0, 0 ) - 14 ) < eps && std::abs( cv( 0, 1 ) - 32 ) < eps &&
             std::abs( cv( 1, 0 ) - 32 ) < eps && std::abs( cv( 1, 1 ) - 77 ) < eps )
        {
            log.info( "PASS: gemm (2x3)*(3x2)" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: gemm. Expected {{14,32},{32,77}} but got {{" + std::to_string( cv( 0, 0 ) ) + "," +
                std::to_string( cv( 0, 1 ) ) + "},{" + std::to_string( cv( 1, 0 ) ) + "," +
                std::to_string( cv( 1, 1 ) ) + "}}"
            );
            failed_counter++;
        }
        A.free();
        B.free();
        C->free();
    }

    log.info( "=== Test: gemm square (2x2) * (2x2) ===" );
    {
        matrix_type A = { { 1, 2 }, { 3, 4 } };
        matrix_type B = { { 5, 6 }, { 7, 8 } };

        auto C  = ops->matrix_matrix_prod( A, B );
        auto cv = C->create_view( true );

        if ( std::abs( cv( 0, 0 ) - 19 ) < eps && std::abs( cv( 0, 1 ) - 22 ) < eps &&
             std::abs( cv( 1, 0 ) - 43 ) < eps && std::abs( cv( 1, 1 ) - 50 ) < eps )
        {
            log.info( "PASS: gemm (2x2)*(2x2)" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: gemm. Expected {{19,22},{43,50}} but got {{" + std::to_string( cv( 0, 0 ) ) + "," +
                std::to_string( cv( 0, 1 ) ) + "},{" + std::to_string( cv( 1, 0 ) ) + "," +
                std::to_string( cv( 1, 1 ) ) + "}}"
            );
            failed_counter++;
        }
        A.free();
        B.free();
        C->free();
    }

    log.info( "=== Test: gemm A * I = A ===" );
    {
        matrix_type A = { { 1, 2 }, { 3, 4 } };
        matrix_type I = { { 1, 0 }, { 0, 1 } };

        auto C  = ops->matrix_matrix_prod( A, I );
        auto cv = C->create_view( true );

        if ( std::abs( cv( 0, 0 ) - 1 ) < eps && std::abs( cv( 0, 1 ) - 2 ) < eps && std::abs( cv( 1, 0 ) - 3 ) < eps &&
             std::abs( cv( 1, 1 ) - 4 ) < eps )
        {
            log.info( "PASS: gemm A * I = A" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: A*I. Expected {{1,2},{3,4}} but got {{" + std::to_string( cv( 0, 0 ) ) + "," +
                std::to_string( cv( 0, 1 ) ) + "},{" + std::to_string( cv( 1, 0 ) ) + "," +
                std::to_string( cv( 1, 1 ) ) + "}}"
            );
            failed_counter++;
        }
        A.free();
        I.free();
        C->free();
    }

    // ====================================================================
    // Test: add_matrix_vector_prod via cuBLAS gemv
    // ====================================================================

    log.info( "=== Test: gemv y = mat * x (3x3) ===" );
    {
        matrix_type mat = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
        vector_type x   = { 1, 2, 3 };
        vector_type y   = { 0, 0, 0 };

        ops->add_matrix_vector_prod( 1.0, mat, x, 0.0, y );

        const auto yv = y.create_view( true );
        if ( std::abs( yv( 0 ) - 14 ) < eps && std::abs( yv( 1 ) - 32 ) < eps && std::abs( yv( 2 ) - 50 ) < eps )
        {
            log.info( "PASS: gemv y = mat * x" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: gemv. Expected {14, 32, 50} but got {" + std::to_string( yv( 0 ) ) + ", " +
                std::to_string( yv( 1 ) ) + ", " + std::to_string( yv( 2 ) ) + "}"
            );
            failed_counter++;
        }
        mat.free();
    }

    log.info( "=== Test: gemv y = alpha*mat*x + beta*y ===" );
    {
        matrix_type mat = { { 1, 0 }, { 0, 1 } };
        vector_type x   = { 3, 4 };
        vector_type y   = { 1, 2 };

        ops->add_matrix_vector_prod( 2.0, mat, x, 3.0, y );

        const auto yv = y.create_view( true );
        if ( std::abs( yv( 0 ) - 9 ) < eps && std::abs( yv( 1 ) - 14 ) < eps )
        {
            log.info( "PASS: gemv y = alpha*mat*x + beta*y" );
            passed_counter++;
        }
        else
        {
            log.error(
                "FAIL: gemv. Expected {9, 14} but got {" + std::to_string( yv( 0 ) ) + ", " +
                std::to_string( yv( 1 ) ) + "}"
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
