#include <cmath>
#include <array>
#include <scfd/utils/log.h>
#include <nmfd/operations/static_vector_space.h>

const double eps = 1e-10;

int main(int argc, char const *args[])
{
    using log_t = scfd::utils::log_std;
    using T = double;
    static const int Dim = 3;
    using vector_type = std::array<T, Dim>;
    using static_vector_space_t = nmfd::operations::static_vector_space<T, Dim, vector_type>;

    log_t log;
    log.info("Testing static vector space implementation");
    size_t passed_counter = 0;
    size_t failed_counter = 0;

    // Initialize vector space and test vectors
    static_vector_space_t vec_space;
    vector_type x = {1, 2, 3};
    vector_type y = {4, 5, 6};

    // ====================================================================
    // GROUP 1: Basic Vector Space Properties and Validation
    // ====================================================================
    log.info("=== Testing Basic Vector Space Properties ===");

    // Test vector space dimensions
    {
        int dim = vec_space.size();
        if (dim == Dim)
        {
            log.info("✓ `size()` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `size()` method test failed. Expected " + std::to_string(Dim) + " but returned " + std::to_string(dim));
            failed_counter++;
        }
    }

    {
        int vec_size = vec_space.get_size(x);
        if (vec_size == Dim)
        {
            log.info("✓ `get_size(x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `get_size(x)` method test failed. Expected " + std::to_string(Dim) + " but returned " + std::to_string(vec_size));
            failed_counter++;
        }
    }

    // Test number validity checking
    {
        if (vec_space.check_is_valid_number(x))
        {
            log.info("✓ `check_is_valid_number(x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `check_is_valid_number(x)` method test failed");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 2: Norms and Scalar Products
    // ====================================================================
    log.info("=== Testing Norms and Scalar Products ===");

    // Test scalar product (dot product)
    {
        T scalar_prod = vec_space.scalar_prod(x, y);
        if (std::abs(scalar_prod - 32) < eps)  // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        {
            log.info("✓ `scalar_prod(x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `scalar_prod(x, y)` method test failed. Expected 32 but returned " + std::to_string(scalar_prod));
            failed_counter++;
        }
    }

    // Test squared L2 norm
    {
        T norm = vec_space.norm_sq(x);
        if (std::abs(norm - 14) < eps)  // 1² + 2² + 3² = 1 + 4 + 9 = 14
        {
            log.info("✓ `norm_sq(x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `norm_sq(x)` method test failed. Expected 14 but returned " + std::to_string(norm));
            failed_counter++;
        }
    }

    // Test L2 norm
    {
        T norm = vec_space.norm(x);
        if (std::abs(norm - std::sqrt(14)) < eps)
        {
            log.info("✓ `norm(x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `norm(x)` method test failed. Expected " + std::to_string(std::sqrt(14)) + " but returned " + std::to_string(norm));
            failed_counter++;
        }
    }

    // Test infinity norm (max absolute value)
    {
        T norm = vec_space.norm_inf(x);
        if (std::abs(norm - 3) < eps)  // max(|1|, |2|, |3|) = 3
        {
            log.info("✓ `norm_inf(x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `norm_inf(x)` method test failed. Expected 3 but returned " + std::to_string(norm));
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 3: Element Access and Point-wise Operations
    // ====================================================================
    log.info("=== Testing Element Access and Point-wise Operations ===");

    // Test setting value at specific index
    {
        vector_type tmp_x = {1, 2, 3};
        vec_space.set_value_at_point(10, 0, tmp_x);
        if ((tmp_x[0] - 10) < eps && (tmp_x[1] - 2) < eps && (tmp_x[2] - 3) < eps)
        {
            log.info("✓ `set_value_at_point(val_x, at, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `set_value_at_point(val_x, at, x)` method test failed. Expected {10, 2, 3} but got {" + std::to_string(tmp_x[0]) + ", " + std::to_string(tmp_x[1]) + ", " + std::to_string(tmp_x[2]) + "}");
            failed_counter++;
        }
    }

    // Test getting value at specific index
    {
        vector_type tmp_x = {1, 2, 3};
        T value = vec_space.get_value_at_point(0, tmp_x);
        if ((value - 1) < eps && (tmp_x[1] - 2) < eps && (tmp_x[2] - 3) < eps)
        {
            log.info("✓ `get_value_at_point(at, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `get_value_at_point(at, x)` method test failed. Expected 1 but returned " + std::to_string(value));
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 4: Scalar-Vector Operations
    // ====================================================================
    log.info("=== Testing Scalar-Vector Operations ===");

    // Test assigning scalar to all elements
    {
        vector_type tmp_x = {1, 2, 3};
        vec_space.assign_scalar(10, tmp_x);
        if ((tmp_x[0] - 10) < eps && (tmp_x[1] - 10) < eps && (tmp_x[2] - 10) < eps)
        {
            log.info("✓ `assign_scalar(scalar, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign_scalar(scalar, x)` method test failed. Expected {10, 10, 10} but got {" + std::to_string(tmp_x[0]) + ", " + std::to_string(tmp_x[1]) + ", " + std::to_string(tmp_x[2]) + "}");
            failed_counter++;
        }
    }

    // Test scalar multiplication and addition: x = mul_x * x + scalar
    {
        vector_type tmp_x = {1, 2, 3};
        vec_space.add_mul_scalar(10, 2, tmp_x);  // x = 2*x + 10 = {2*1+10, 2*2+10, 2*3+10} = {12, 14, 16}
        if ((tmp_x[0] - 12) < eps && (tmp_x[1] - 14) < eps && (tmp_x[2] - 16) < eps)
        {
            log.info("✓ `add_mul_scalar(scalar, mul_x, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul_scalar(scalar, mul_x, x)` method test failed. Expected {12, 14, 16} but got {" + std::to_string(tmp_x[0]) + ", " + std::to_string(tmp_x[1]) + ", " + std::to_string(tmp_x[2]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 5: Vector Assignment and Basic Operations
    // ====================================================================
    log.info("=== Testing Vector Assignment and Basic Operations ===");

    // Test vector assignment: y = x
    {
        vector_type tmp_x = {1, 2, 3};
        vector_type tmp_y = {0, 0, 0};
        vec_space.assign(tmp_x, tmp_y);
        if ((tmp_y[0] - 1) < eps && (tmp_y[1] - 2) < eps && (tmp_y[2] - 3) < eps)
        {
            log.info("✓ `assign(x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign(x, y)` method test failed. Expected {1, 2, 3} but got {" + std::to_string(tmp_y[0]) + ", " + std::to_string(tmp_y[1]) + ", " + std::to_string(tmp_y[2]) + "}");
            failed_counter++;
        }
    }

    // Test scalar multiplication assignment: y = mul_x * x
    {
        vector_type tmp_x = {1, 2, 3};
        vector_type tmp_y = {0, 0, 0};
        vec_space.assign_mul(2, tmp_x, tmp_y);  // y = 2 * {1, 2, 3} = {2, 4, 6}
        if ((tmp_y[0] - 2) < eps && (tmp_y[1] - 4) < eps && (tmp_y[2] - 6) < eps)
        {
            log.info("✓ `assign_mul(mul_x, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign_mul(mul_x, x, y)` method test failed. Expected {2, 4, 6} but got {" + std::to_string(tmp_y[0]) + ", " + std::to_string(tmp_y[1]) + ", " + std::to_string(tmp_y[2]) + "}");
            failed_counter++;
        }
    }

    // Test linear combination assignment: z = mul_x * x + mul_y * y
    {
        vector_type tmp_x = {1, 2, 3};
        vector_type tmp_y = {4, 5, 6};
        vector_type tmp_z = {0, 0, 0};
        vec_space.assign_mul(2, tmp_x, 3, tmp_y, tmp_z);  // z = 2*{1,2,3} + 3*{4,5,6} = {2,4,6} + {12,15,18} = {14,19,24}
        if ((tmp_z[0] - 14) < eps && (tmp_z[1] - 19) < eps && (tmp_z[2] - 24) < eps)
        {
            log.info("✓ `assign_mul(mul_x, x, mul_y, y, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign_mul(mul_x, x, mul_y, y, z)` method test failed. Expected {14, 19, 24} but got {" + std::to_string(tmp_z[0]) + ", " + std::to_string(tmp_z[1]) + ", " + std::to_string(tmp_z[2]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 6: Add-Multiply Operations (In-place Linear Combinations)
    // ====================================================================
    log.info("=== Testing Add-Multiply Operations ===");

    // Test add-multiply: y = y + mul_x * x
    {
        vector_type tmp_x = {1, 2, 3};
        vector_type tmp_y = {4, 5, 6};
        vec_space.add_mul(2, tmp_x, tmp_y);  // y = {4,5,6} + 2*{1,2,3} = {4,5,6} + {2,4,6} = {6,9,12}
        if ((tmp_y[0] - 6) < eps && (tmp_y[1] - 9) < eps && (tmp_y[2] - 12) < eps)
        {
            log.info("✓ `add_mul(mul_x, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul(mul_x, x, y)` method test failed. Expected {6, 9, 12} but got {" + std::to_string(tmp_y[0]) + ", " + std::to_string(tmp_y[1]) + ", " + std::to_string(tmp_y[2]) + "}");
            failed_counter++;
        }
    }

    // Test add-multiply: y = mul_x * x + mul_y * y
    {
        vector_type tmp_x = {1, 2, 3};
        vector_type tmp_y = {4, 5, 6};
        vec_space.add_mul(2, tmp_x, 3, tmp_y);  // y = 2*{1,2,3} + 3*{4,5,6} = {2,4,6} + {12,15,18} = {14,19,24}
        if ((tmp_y[0] - 14) < eps && (tmp_y[1] - 19) < eps && (tmp_y[2] - 24) < eps)
        {
            log.info("✓ `add_mul(mul_x, x, mul_y, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul(mul_x, x, mul_y, y)` method test failed. Expected {14, 19, 24} but got {" + std::to_string(tmp_y[0]) + ", " + std::to_string(tmp_y[1]) + ", " + std::to_string(tmp_y[2]) + "}");
            failed_counter++;
        }
    }

    // Test add-multiply: z = mul_x * x + mul_y * y + mul_z * z
    {
        vector_type tmp_x = {1, 2, 3};
        vector_type tmp_y = {4, 5, 6};
        vector_type tmp_z = {7, 8, 9};
        vec_space.add_mul(2, tmp_x, 3, tmp_y, 4, tmp_z);  // z = 2*{1,2,3} + 3*{4,5,6} + 4*{7,8,9} = {2,4,6} + {12,15,18} + {28,32,36} = {42,51,60}
        if ((tmp_z[0] - 42) < eps && (tmp_z[1] - 51) < eps && (tmp_z[2] - 60) < eps)
        {
            log.info("✓ `add_mul(mul_x, x, mul_y, y, mul_z, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul(mul_x, x, mul_y, y, mul_z, z)` method test failed. Expected {42, 51, 60} but got {" + std::to_string(tmp_z[0]) + ", " + std::to_string(tmp_z[1]) + ", " + std::to_string(tmp_z[2]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 7: Absolute Value Operations
    // ====================================================================
    log.info("=== Testing Absolute Value Operations ===");

    // Test absolute value copy: y = |x|
    {
        vector_type tmp_x = {1, -2, 3};
        vector_type tmp_y = {0, 0, 0};
        vec_space.make_abs_copy(tmp_x, tmp_y);
        if ((tmp_y[0] - 1) < eps && (tmp_y[1] - 2) < eps && (tmp_y[2] - 3) < eps)
        {
            log.info("✓ `make_abs_copy(x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `make_abs_copy(x, y)` method test failed. Expected {1, 2, 3} but got {" + std::to_string(tmp_y[0]) + ", " + std::to_string(tmp_y[1]) + ", " + std::to_string(tmp_y[2]) + "}");
            failed_counter++;
        }
    }

    // Test in-place absolute value: x = |x|
    {
        vector_type tmp_x = {1, -2, 3};
        vec_space.make_abs(tmp_x);
        if ((tmp_x[0] - 1) < eps && (tmp_x[1] - 2) < eps && (tmp_x[2] - 3) < eps)
        {
            log.info("✓ `make_abs(x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `make_abs(x)` method test failed. Expected {1, 2, 3} but got {" + std::to_string(tmp_x[0]) + ", " + std::to_string(tmp_x[1]) + ", " + std::to_string(tmp_x[2]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 8: Min/Max Point-wise Operations
    // ====================================================================
    log.info("=== Testing Min/Max Point-wise Operations ===");

    // Test max point-wise: y = max(sc, x, y)
    {
        vector_type tmp_x = {1, -2, 3};
        vector_type tmp_y = {4, 5, -6};
        vec_space.max_pointwise(2, tmp_x, tmp_y);  // y = max(2, {1,-2,3}, {4,5,-6}) = max(2, max({1,-2,3}, {4,5,-6})) = max(2, {4,5,3}) = {4,5,3}
        if ((tmp_y[0] - 4) < eps && (tmp_y[1] - 5) < eps && (tmp_y[2] - 3) < eps)
        {
            log.info("✓ `max_pointwise(sc, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `max_pointwise(sc, x, y)` method test failed. Expected {4, 5, 3} but got {" + std::to_string(tmp_y[0]) + ", " + std::to_string(tmp_y[1]) + ", " + std::to_string(tmp_y[2]) + "}");
            failed_counter++;
        }
    }

    // Test max point-wise with scalar: x = max(sc, x)
    {
        vector_type tmp_x = {1, -2, 3};
        vec_space.max_pointwise(2, tmp_x);  // x = max(2, {1,-2,3}) = {2, 2, 3}
        if ((tmp_x[0] - 2) < eps && (tmp_x[1] - 2) < eps && (tmp_x[2] - 3) < eps)
        {
            log.info("✓ `max_pointwise(sc, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `max_pointwise(sc, x)` method test failed. Expected {2, 2, 3} but got {" + std::to_string(tmp_x[0]) + ", " + std::to_string(tmp_x[1]) + ", " + std::to_string(tmp_x[2]) + "}");
            failed_counter++;
        }
    }

    // Test min point-wise: y = min(sc, x, y)
    {
        vector_type tmp_x = {1, -2, 3};
        vector_type tmp_y = {4, 5, -6};
        vec_space.min_pointwise(2, tmp_x, tmp_y);  // y = min(2, {1,-2,3}, {4,5,-6}) = min(2, min({1,-2,3}, {4,5,-6})) = min(2, {1,-2,-6}) = {1,-2,-6}
        if ((tmp_y[0] - 1) < eps && (tmp_y[1] + 2) < eps && (tmp_y[2] + 6) < eps)
        {
            log.info("✓ `min_pointwise(sc, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `min_pointwise(sc, x, y)` method test failed. Expected {1, -2, -6} but got {" + std::to_string(tmp_y[0]) + ", " + std::to_string(tmp_y[1]) + ", " + std::to_string(tmp_y[2]) + "}");
            failed_counter++;
        }
    }

    // Test min point-wise with scalar: x = min(sc, x)
    {
        vector_type tmp_x = {1, -2, 3};
        vec_space.min_pointwise(2, tmp_x);  // x = min(2, {1,-2,3}) = {1, -2, 2}
        if ((tmp_x[0] - 1) < eps && (tmp_x[1] + 2) < eps && (tmp_x[2] - 2) < eps)
        {
            log.info("✓ `min_pointwise(sc, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `min_pointwise(sc, x)` method test failed. Expected {1, -2, 2} but got {" + std::to_string(tmp_x[0]) + ", " + std::to_string(tmp_x[1]) + ", " + std::to_string(tmp_x[2]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 9: Point-wise Multiplication and Division
    // ====================================================================
    log.info("=== Testing Point-wise Multiplication and Division ===");

    // Test point-wise multiplication: x = x * (mul_y * y)
    {
        vector_type tmp_x = {1, 2, 3};
        vector_type tmp_y = {4, 5, 6};
        vec_space.mul_pointwise(tmp_x, 2, tmp_y);  // x = {1,2,3} * (2 * {4,5,6}) = {1,2,3} * {8,10,12} = {8,20,36}
        if ((tmp_x[0] - 8) < eps && (tmp_x[1] - 20) < eps && (tmp_x[2] - 36) < eps)
        {
            log.info("✓ `mul_pointwise(x, mul_y, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `mul_pointwise(x, mul_y, y)` method test failed. Expected {8, 20, 36} but got {" + std::to_string(tmp_x[0]) + ", " + std::to_string(tmp_x[1]) + ", " + std::to_string(tmp_x[2]) + "}");
            failed_counter++;
        }
    }

    // Test point-wise multiplication assignment: z = (mul_x * x) * (mul_y * y)
    {
        vector_type tmp_x = {1, 2, 3};
        vector_type tmp_y = {4, 5, 6};
        vector_type tmp_z = {0, 0, 0};
        vec_space.mul_pointwise(2, tmp_x, 3, tmp_y, tmp_z);  // z = (2*{1,2,3}) * (3*{4,5,6}) = {2,4,6} * {12,15,18} = {24,60,108}
        if ((tmp_z[0] - 24) < eps && (tmp_z[1] - 60) < eps && (tmp_z[2] - 108) < eps)
        {
            log.info("✓ `mul_pointwise(mul_x, x, mul_y, y, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `mul_pointwise(mul_x, x, mul_y, y, z)` method test failed. Expected {24, 60, 108} but got {" + std::to_string(tmp_z[0]) + ", " + std::to_string(tmp_z[1]) + ", " + std::to_string(tmp_z[2]) + "}");
            failed_counter++;
        }
    }

    // Test point-wise division assignment: z = (mul_x * x) / (mul_y * y)
    {
        vector_type tmp_x = {1, 2, 3};
        vector_type tmp_y = {4, 5, 6};
        vector_type tmp_z = {0, 0, 0};
        vec_space.div_pointwise(2, tmp_x, 3, tmp_y, tmp_z);  // z = (2*{1,2,3}) / (3*{4,5,6}) = {2,4,6} / {12,15,18} = {2/12, 4/15, 6/18}
        if ((12 * tmp_z[0] - 2) < eps && (15 * tmp_z[1] - 4) < eps && (18 * tmp_z[2] - 6) < eps)
        {
            log.info("✓ `div_pointwise(mul_x, x, mul_y, y, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `div_pointwise(mul_x, x, mul_y, y, z)` method test failed. Expected {2/12, 4/15, 6/18} but got {" + std::to_string(tmp_z[0]) + ", " + std::to_string(tmp_z[1]) + ", " + std::to_string(tmp_z[2]) + "}");
            failed_counter++;
        }
    }

    // Test point-wise division: x = x / (mul_y * y)
    {
        vector_type tmp_x = {1, 2, 3};
        vector_type tmp_y = {4, 5, 6};
        vec_space.div_pointwise(tmp_x, 3, tmp_y);  // x = {1,2,3} / (3 * {4,5,6}) = {1,2,3} / {12,15,18} = {1/12, 2/15, 3/18}
        if ((12 * tmp_x[0] - 1) < eps && (15 * tmp_x[1] - 2) < eps && (18 * tmp_x[2] - 3) < eps)
        {
            log.info("✓ `div_pointwise(x, mul_y, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `div_pointwise(x, mul_y, y)` method test failed. Expected {1/12, 2/15, 3/18} but got {" + std::to_string(tmp_x[0]) + ", " + std::to_string(tmp_x[1]) + ", " + std::to_string(tmp_x[2]) + "}");
            failed_counter++;
        }
    }

    // Additional tests: Slice Operations
    log.info("=== Testing Slice Operations ===");
    {
        static_vector_space_t vec_space2;
        vector_type x2 = {1, 2, 3};
        vector_type y2 = {0, 0, 0};
        std::vector<std::pair<size_t,size_t>> slices = {{0,2},{2,3}}; // copy 0,1 then 2 -> all
        vec_space2.assign_slices(x2, slices, y2);
        if ((y2[0]-1)<eps && (y2[1]-2)<eps && (y2[2]-3)<eps) {
            log.info("✓ `assign_slices(x, slices, y)` method test passed");
            passed_counter++;
        } else {
            log.error("✗ `assign_slices(x, slices, y)` method failed.");
            failed_counter++;
        }
    }

    {
        static_vector_space_t vec_space3;
        vector_type x3 = {1, 2, 3};
        vector_type y3 = {0, 0, 0};
        std::vector<std::pair<size_t,size_t>> skip = {{1,2}}; // skip 1..2 -> keep 0
        vec_space3.assign_skip_slices(x3, skip, y3);
        if ((y3[0]-1)<eps) {
            log.info("✓ `assign_skip_slices(x, skip_slices, y)` method test passed");
            passed_counter++;
        } else {
            log.error("✗ `assign_skip_slices(x, skip_slices, y)` method failed.");
            failed_counter++;
        }
    }

    // ====================================================================
    // FINAL SUMMARY
    // ====================================================================
    log.info("================================================");
    log.info("=== TEST SUMMARY ===");
    log.info("✓ Passed: " + std::to_string(passed_counter));
    log.info("✗ Failed: " + std::to_string(failed_counter));
    log.info("Total tests: " + std::to_string(passed_counter + failed_counter));

    if (failed_counter == 0) {
        log.info("🎉 All tests passed successfully!");
    } else {
        log.info("⚠️  Some tests failed. Please review the output above.");
    }
    log.info("================================================");

    return (failed_counter == 0) ? 0 : 1;
}
