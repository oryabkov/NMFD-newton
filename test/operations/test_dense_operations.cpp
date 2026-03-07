#include <cmath>
#include <memory>
#include <nmfd/operations/dense_operations_base.h>
#include <scfd/arrays/array.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/utils/log.h>

#include "scfd_array_traits.h"

#include <scfd/backend/serial_cpu.h>

const double eps = 1e-10;

int main( int argc, char const *args[] )
{
    using log_t         = scfd::utils::log_std;
    using T             = double;
    using memory_type   = scfd::backend::serial_cpu::memory_type;
    using vector_type   = scfd::arrays::array<T, memory_type>;
    using vector_traits = scfd_array_traits<T, memory_type>;
    using dense_ops_t   = nmfd::operations::dense_operations<T, vector_traits, scfd::backend::serial_cpu>;
    using matrix_type   = typename dense_ops_t::matrix_type;

    log_t log;
    log.info( "Testing dense_operations (matrix-vector)" );
    size_t passed_counter = 0;
    size_t failed_counter = 0;

    const int rows    = 3;
    const int cols    = 3;
    const int vec_dim = cols;

    auto ops = std::make_shared<dense_ops_t>( vec_dim );

    // ====================================================================
    // Test 1: y := 1.0 * mat * x + 0.0 * y  (simple mat*x)
    // ====================================================================
    log.info( "=== Test 1: y = mat * x ===" );
    {
        //     mat = | 1  2  3 |    x = | 1 |    mat*x = | 1*1+2*2+3*3 |   | 14 |
        //           | 4  5  6 |        | 2 |            | 4*1+5*2+6*3 | = | 32 |
        //           | 7  8  9 |        | 3 |            | 7*1+8*2+9*3 |   | 50 |

        matrix_type mat;
        mat.init( rows, cols );
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

    // ====================================================================
    // Test 2: y := alpha * mat * x + beta * y
    // ====================================================================
    log.info( "=== Test 2: y = alpha*mat*x + beta*y ===" );
    {
        //     mat = | 1  0 |    x = | 3 |    alpha=2, beta=3
        //           | 0  1 |        | 4 |
        //     mat*x = | 3 |    alpha*mat*x = | 6 |    y_init = | 1 |
        //             | 4 |                  | 8 |              | 2 |
        //     result = 6 + 3*1 = 9,  8 + 3*2 = 14

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
                "FAIL: y = alpha*mat*x + beta*y. Expected {9, 14} but got {" + std::to_string( yv( 0 ) ) + ", " +
                std::to_string( yv( 1 ) ) + "}"
            );
            failed_counter++;
        }
        mat.free();
    }

    // ====================================================================
    // Test 3: non-square matrix  (2x3) * (3x1) -> (2x1)
    // ====================================================================
    log.info( "=== Test 3: non-square matrix ===" );
    {
        //     mat = | 1  2  3 |    x = | 1 |    mat*x = | 1+4+9  | = | 14 |
        //           | 4  5  6 |        | 2 |            | 4+10+18|   | 32 |
        //                              | 3 |

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
                "FAIL: non-square (2x3) matrix. Expected {14, 32} but got {" + std::to_string( yv( 0 ) ) + ", " +
                std::to_string( yv( 1 ) ) + "}"
            );
            failed_counter++;
        }
        mat.free();
    }

    // ====================================================================
    // Test 4: assign_matrix_vector_prod  z := alpha*mat*x + beta*y
    // ====================================================================
    log.info( "=== Test 4: assign_matrix_vector_prod ===" );
    {
        //     mat = | 2  0 |    x = | 3 |    alpha=1, beta=1
        //           | 0  2 |        | 4 |
        //     mat*x = | 6 |    y = | 10 |
        //             | 8 |        | 20 |
        //     z = 1 * mat*x + 1 * y = {6+10, 8+20} = {16, 28}

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
                "FAIL: assign_matrix_vector_prod. Expected {16, 28} but got {" + std::to_string( zv( 0 ) ) + ", " +
                std::to_string( zv( 1 ) ) + "}"
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
