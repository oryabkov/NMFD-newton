#include <cmath>
#include <array>
#include <scfd/utils/log.h>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/operations/pair_vector_space.h>

const double eps = 1e-10;

int main(int argc, char const *args[])
{
    using log_t = scfd::utils::log_std;
    using T = double;
    static const int Dim1 = 3;
    static const int Dim2 = 2;
    using vector1_type = std::array<T, Dim1>;
    using vector2_type = std::array<T, Dim2>;
    using vector_type = std::pair<vector1_type, vector2_type>;
    using static_vector_space1_type = nmfd::operations::static_vector_space<T, Dim1, vector1_type>;
    using static_vector_space2_type = nmfd::operations::static_vector_space<T, Dim2, vector2_type>;
    using pair_vector_space_type = nmfd::operations::pair_vector_space<static_vector_space1_type, static_vector_space2_type>;

    log_t log;
    log.info("Testing static vector space implementation");
    size_t passed_counter = 0;
    size_t failed_counter = 0;

    // Initialize vector space and test vectors
    pair_vector_space_type pair_vec_space;

    // ====================================================================
    // GROUP 1: Basic Vector Space Properties and Validation
    // ====================================================================
    log.info("=== Testing Basic Vector Space Properties ===");

    // Test vector space dimensions
    {
        int dim = pair_vec_space.size();
        if (dim == Dim1 + Dim2)
        {
            log.info("✓ `size()` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `size()` method test failed. Expected " + std::to_string(Dim1 + Dim2) + " but returned " + std::to_string(dim));
            failed_counter++;
        }
    }

    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        int vec_size = pair_vec_space.get_size(x);
        if (vec_size == Dim1 + Dim2)
        {
            log.info("✓ `get_size(x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `get_size(x)` method test failed. Expected " + std::to_string(Dim1 + Dim2) + " but returned " + std::to_string(vec_size));
            failed_counter++;
        }
    }

    // Test number validity checking
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        if (pair_vec_space.check_is_valid_number(x))
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
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        T scalar_prod = pair_vec_space.scalar_prod(x, y);
        if (std::abs(scalar_prod - 130) < eps)  // 1*6 + 2*7 + 3*8 + 4*9 + 5*10 = 6 + 14 + 24 + 36 + 50 = 130
        {
            log.info("✓ `scalar_prod(x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `scalar_prod(x, y)` method test failed. Expected 130 but returned " + std::to_string(scalar_prod));
            failed_counter++;
        }
    }

    // Test squared L2 norm
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        T norm = pair_vec_space.norm_sq(x);
        if (std::abs(norm - 55) < eps)  // 1² + 2² + 3² + 4² + 5² = 1 + 4 + 9 + 16 + 25 = 55
        {
            log.info("✓ `norm_sq(x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `norm_sq(x)` method test failed. Expected 55 but returned " + std::to_string(norm));
            failed_counter++;
        }
    }

    // Test L2 norm
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        T norm = pair_vec_space.norm(x);
        if (std::abs(norm - std::sqrt(55)) < eps)
        {
            log.info("✓ `norm(x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `norm(x)` method test failed. Expected " + std::to_string(std::sqrt(55)) + " but returned " + std::to_string(norm));
            failed_counter++;
        }
    }

    // Test infinity norm (max absolute value)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        T norm = pair_vec_space.norm_inf(x);
        if (std::abs(norm - 5) < eps)  // max(|1|, |2|, |3|, |4|, |5|) = 5
        {
            log.info("✓ `norm_inf(x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `norm_inf(x)` method test failed. Expected 5 but returned " + std::to_string(norm));
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 3: New Norm/Inner-Product Aliases
    // ====================================================================
    log.info("=== Testing New Norm/Inner-Product Aliases ===");
    {
        vector_type xa = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type ya = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});

        // scalar_prod_l2 should match scalar_prod
        T sp = pair_vec_space.scalar_prod(xa, ya);
        T sp_l2 = pair_vec_space.scalar_prod_l2(xa, ya);
        if (std::abs(sp - 130) < eps && std::abs(sp_l2 - sp) < eps) {
            log.info("✓ `scalar_prod_l2(x, y)` matches `scalar_prod(x, y)`");
            passed_counter++;
        } else {
            log.error("✗ `scalar_prod_l2` mismatch: sp=" + std::to_string(sp) + ", sp_l2=" + std::to_string(sp_l2));
            failed_counter++;
        }

        // norm2/norm_l2 should match norm; *_sq should match norm_sq
        T n = pair_vec_space.norm(xa);
        T n2 = pair_vec_space.norm2(xa);
        T nl2 = pair_vec_space.norm_l2(xa);
        T nsq = pair_vec_space.norm_sq(xa);
        T n2sq = pair_vec_space.norm2_sq(xa);
        T nl2sq = pair_vec_space.norm_l2_sq(xa);
        if (std::abs(n - std::sqrt(55)) < eps && std::abs(n2 - n) < eps && std::abs(nl2 - n) < eps &&
            std::abs(nsq - 55) < eps && std::abs(n2sq - nsq) < eps && std::abs(nl2sq - nsq) < eps) {
            log.info("✓ `norm2`, `norm_l2` and their *_sq variants match base L2");
            passed_counter++;
        } else {
            log.error("✗ L2 alias norms mismatch");
            failed_counter++;
        }

        // norm1/norm_l1 should match asum
        vector_type xb = vector_type(vector1_type{1, -2, 3}, vector2_type{-4, 5});
        T a1 = pair_vec_space.asum(xb);
        T n1 = pair_vec_space.norm1(xb);
        T nl1 = pair_vec_space.norm_l1(xb);
        if (std::abs(a1 - 15) < eps && std::abs(n1 - a1) < eps && std::abs(nl1 - a1) < eps) {
            log.info("✓ `norm1` and `norm_l1` match `asum`");
            passed_counter++;
        } else {
            log.error("✗ L1 alias norms mismatch");
            failed_counter++;
        }

        // norm_l_inf should match norm_inf
        T ni = pair_vec_space.norm_inf(xb);
        T nli = pair_vec_space.norm_l_inf(xb);
        if (std::abs(ni - 5) < eps && std::abs(nli - ni) < eps) {
            log.info("✓ `norm_l_inf` matches `norm_inf`");
            passed_counter++;
        } else {
            log.error("✗ L_inf alias norm mismatch");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 4: Element Access and Point-wise Operations
    // ====================================================================
    log.info("=== Testing Element Access and Point-wise Operations ===");

    // Test setting value at specific global index
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        pair_vec_space.set_value_at_point(10, 1, x);
        pair_vec_space.set_value_at_point(11, 3, x);
        if ((x.first[0] - 1) < eps && (x.first[1] - 10) < eps && (x.first[2] - 3) < eps &&
            (x.second[0] - 11) < eps && (x.second[1] - 5) < eps)
        {
            log.info("✓ `set_value_at_point(val_x, at, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `set_value_at_point(val_x, at, x)` method test failed. Expected {{10, 2, 3}, {11, 5}} but got {" + std::to_string(x.first[1]) + ", " + std::to_string(x.second[0]) + "}");
            failed_counter++;
        }
    }

    // Test getting value at specific global index
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        T value1 = pair_vec_space.get_value_at_point(1, x);
        T value2 = pair_vec_space.get_value_at_point(3, x);
        if ((value1 - 2) < eps && (value2 - 4) < eps)
        {
            log.info("✓ `get_value_at_point(at, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `get_value_at_point(at, x)` method test failed. Expected 2 but returned " + std::to_string(value1) + " and 4 but returned " + std::to_string(value2));
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 5: Scalar-Vector Operations
    // ====================================================================
    log.info("=== Testing Scalar-Vector Operations ===");

    // Test assigning scalar to all elements (single scalar for both subvectors)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        pair_vec_space.assign_scalar(10, x);
        if ((x.first[0] - 10) < eps && (x.first[1] - 10) < eps && (x.first[2] - 10) < eps &&
            (x.second[0] - 10) < eps && (x.second[1] - 10) < eps)
        {
            log.info("✓ `assign_scalar(scalar, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign_scalar(scalar, x)` method test failed. Expected {{10, 10, 10}, {10, 10}} but got {" + std::to_string(x.first[0]) + ", " + std::to_string(x.first[1]) + ", " + std::to_string(x.first[2]) + ", " + std::to_string(x.second[0]) + ", " + std::to_string(x.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test scalar multiplication and addition: x = mul_x * x + scalar (single scalars)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        pair_vec_space.add_mul_scalar(-1, 2, x);  // x = 2*x - 1
        if ((x.first[0] - 1) < eps && (x.first[1] - 3) < eps && (x.first[2] - 5) < eps &&
            (x.second[0] - 7) < eps && (x.second[1] - 9) < eps)
        {
            log.info("✓ `add_mul_scalar(scalar, mul_x, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul_scalar(scalar, mul_x, x)` method failed. Expected {{1, 3, 5}, {7, 9}} but got {" + std::to_string(x.first[0]) + ", " + std::to_string(x.first[1]) + ", " + std::to_string(x.first[2]) + ", " + std::to_string(x.second[0]) + ", " + std::to_string(x.second[1]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 6: Vector Assignment and Basic Operations
    // ====================================================================
    log.info("=== Testing Vector Assignment and Basic Operations ===");

    // Test vector assignment: y = x
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{0, 0, 0}, vector2_type{0, 0});
        pair_vec_space.assign(x, y);
        if ((y.first[0] - 1) < eps && (y.first[1] - 2) < eps && (y.first[2] - 3) < eps &&
            (y.second[0] - 4) < eps && (y.second[1] - 5) < eps)
        {
            log.info("✓ `assign(x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign(x, y)` method test failed. Expected {{1, 2, 3}, {4, 5}} but got {" + std::to_string(y.first[0]) + ", " + std::to_string(y.first[1]) + ", " + std::to_string(y.first[2]) + ", " + std::to_string(y.second[0]) + ", " + std::to_string(y.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test scalar multiplication assignment: y = mul_x * x
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{0, 0, 0}, vector2_type{0, 0});
        pair_vec_space.assign_mul(2, x, y);  // y = 2*x (both subvectors)
        if ((y.first[0] - 2) < eps && (y.first[1] - 4) < eps && (y.first[2] - 6) < eps &&
            (y.second[0] - 8) < eps && (y.second[1] - 10) < eps)
        {
            log.info("✓ `assign_mul(mul_x, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign_mul(mul_x, x, y)` method test failed. Expected {{2, 4, 6}, {8, 10}} but got {" + std::to_string(y.first[0]) + ", " + std::to_string(y.first[1]) + ", " + std::to_string(y.first[2]) + ", " + std::to_string(y.second[0]) + ", " + std::to_string(y.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test linear combination assignment: z = mul_x * x + mul_y * y
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        vector_type z = vector_type(vector1_type{0, 0, 0}, vector2_type{0, 0});
        pair_vec_space.assign_mul(2, x, 3, y, z);
        if ((z.first[0] - 20) < eps && (z.first[1] - 25) < eps && (z.first[2] - 30) < eps &&
            (z.second[0] - 35) < eps && (z.second[1] - 40) < eps)
        {
            log.info("✓ `assign_mul(mul_x, x, mul_y, y, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign_mul(mul_x, x, mul_y, y, z)` method failed. Expected {{20, 25, 30}, {35, 40}} but got {" + std::to_string(z.first[0]) + ", " + std::to_string(z.first[1]) + ", " + std::to_string(z.first[2]) + ", " + std::to_string(z.second[0]) + ", " + std::to_string(z.second[1]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 7: Add-Multiply Operations (In-place Linear Combinations)
    // ====================================================================
    log.info("=== Testing Add-Multiply Operations ===");

    // Test add-multiply: y = y + mul_x * x
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        pair_vec_space.add_mul(2, x, y);  // y = y + 2*x
        if ((y.first[0] - 8) < eps && (y.first[1] - 11) < eps && (y.first[2] - 14) < eps &&
            (y.second[0] - 17) < eps && (y.second[1] - 20) < eps)
        {
            log.info("✓ `add_mul(mul_x, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul(mul_x, x, y)` method failed. Expected {{8, 11, 14}, {17, 20}} but got {" + std::to_string(y.first[0]) + ", " + std::to_string(y.first[1]) + ", " + std::to_string(y.first[2]) + ", " + std::to_string(y.second[0]) + ", " + std::to_string(y.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test add-multiply: y = mul_x * x + mul_y * y
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        pair_vec_space.add_mul(2, x, 3, y);
        if ((y.first[0] - 20) < eps && (y.first[1] - 25) < eps && (y.first[2] - 30) < eps &&
            (y.second[0] - 35) < eps && (y.second[1] - 40) < eps)
        {
            log.info("✓ `add_mul(mul_x, x, mul_y, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul(mul_x, x, mul_y, y)` method failed. Expected {{20, 25, 30}, {35, 40}} but got {" + std::to_string(y.first[0]) + ", " + std::to_string(y.first[1]) + ", " + std::to_string(y.first[2]) + ", " + std::to_string(y.second[0]) + ", " + std::to_string(y.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test add-multiply: z = mul_x * x + mul_y * y + mul_z * z
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        vector_type z = vector_type(vector1_type{11, 12, 13}, vector2_type{14, 15});
        pair_vec_space.add_mul(2, x, 3, y, 4, z);
        if ((z.first[0] - 64) < eps && (z.first[1] - 73) < eps && (z.first[2] - 82) < eps &&
            (z.second[0] - 91) < eps && (z.second[1] - 100) < eps)
        {
            log.info("✓ `add_mul(mul_x, x, mul_y, y, mul_z, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul(mul_x, x, mul_y, y, mul_z, z)` method failed. Expected {{64, 73, 82}, {91, 100}} but got {" + std::to_string(z.first[0]) + ", " + std::to_string(z.first[1]) + ", " + std::to_string(z.first[2]) + ", " + std::to_string(z.second[0]) + ", " + std::to_string(z.second[1]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 8: Absolute Value Operations
    // ====================================================================
    log.info("=== Testing Absolute Value Operations ===");

    // Test absolute value copy: y = |x|
    {
        vector_type x = vector_type(vector1_type{1, 2, -3}, vector2_type{-4, 5});
        vector_type y = vector_type(vector1_type{0, 0, 0}, vector2_type{0, 0});
        pair_vec_space.make_abs_copy(x, y);
        if ((y.first[0] - 1) < eps && (y.first[1] - 2) < eps && (y.first[2] - 3) < eps &&
            (y.second[0] - 4) < eps && (y.second[1] - 5) < eps)
        {
            log.info("✓ `make_abs_copy(x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `make_abs_copy(x, y)` method test failed. Expected {{1, 2, 3}, {4, 5}} but got {" + std::to_string(y.first[0]) + ", " + std::to_string(y.first[1]) + ", " + std::to_string(y.first[2]) + ", " + std::to_string(y.second[0]) + ", " + std::to_string(y.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test in-place absolute value: x = |x|
    {
        vector_type x = vector_type(vector1_type{1, 2, -3}, vector2_type{-4, 5});
        pair_vec_space.make_abs(x);
        if ((x.first[0] - 1) < eps && (x.first[1] - 2) < eps && (x.first[2] - 3) < eps &&
            (x.second[0] - 4) < eps && (x.second[1] - 5) < eps)
        {
            log.info("✓ `make_abs(x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `make_abs(x)` method test failed. Expected {{1, 2, 3}, {4, 5}} but got {" + std::to_string(x.first[0]) + ", " + std::to_string(x.first[1]) + ", " + std::to_string(x.first[2]) + ", " + std::to_string(x.second[0]) + ", " + std::to_string(x.second[1]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 9: Min/Max Point-wise Operations
    // ====================================================================
    log.info("=== Testing Min/Max Point-wise Operations ===");

    // Test max point-wise: y = max(sc, x, y)
    {
        vector_type x = vector_type(vector1_type{1, 2, -3}, vector2_type{-9, 10});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{4, 5});
        pair_vec_space.max_pointwise(7, x, y);  // y = max(7, x, y)
        if ((y.first[0] - 7) < eps && (y.first[1] - 7) < eps && (y.first[2] - 8) < eps &&
            (y.second[0] - 7) < eps && (y.second[1] - 10) < eps)
        {
            log.info("✓ `max_pointwise(sc, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `max_pointwise(sc, x, y)` method failed. Expected {{7, 7, 8}, {7, 10}} but got {" + std::to_string(y.first[0]) + ", " + std::to_string(y.first[1]) + ", " + std::to_string(y.first[2]) + ", " + std::to_string(y.second[0]) + ", " + std::to_string(y.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test max point-wise with scalar: x = max(sc, x)
    {
        vector_type x = vector_type(vector1_type{-1, 2, 3}, vector2_type{-9, 10});
        pair_vec_space.max_pointwise(2, x);  // x = max(2, x)
        if ((x.first[0] - 2) < eps && (x.first[1] - 2) < eps && (x.first[2] - 3) < eps &&
            (x.second[0] - 2) < eps && (x.second[1] - 10) < eps)
        {
            log.info("✓ `max_pointwise(sc, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `max_pointwise(sc, x)` method failed. Expected {{2, 2, 3}, {2, 10}} but got {" + std::to_string(x.first[0]) + ", " + std::to_string(x.first[1]) + ", " + std::to_string(x.first[2]) + ", " + std::to_string(x.second[0]) + ", " + std::to_string(x.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test min point-wise: y = min(sc, x, y)
    {
        vector_type x = vector_type(vector1_type{1, 2, -3}, vector2_type{-9, 10});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{4, 5});
        pair_vec_space.min_pointwise(7, x, y);
        if ((y.first[0] - 1) < eps && (y.first[1] - 2) < eps && (y.first[2] + 3) < eps &&
            (y.second[0] + 9) < eps && (y.second[1] - 5) < eps)
        {
            log.info("✓ `min_pointwise(sc, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `min_pointwise(sc, x, y)` method failed.");
            failed_counter++;
        }
    }

    // Test min point-wise with scalar: x = min(sc, x)
    {
        vector_type x = vector_type(vector1_type{-1, 2, 3}, vector2_type{-9, 10});
        pair_vec_space.min_pointwise(2, x);
        if ((x.first[0] + 1) < eps && (x.first[1] - 2) < eps && (x.first[2] - 2) < eps &&
            (x.second[0] + 9) < eps && (x.second[1] - 2) < eps)
        {
            log.info("✓ `min_pointwise(sc, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `min_pointwise(sc, x)` method failed.");
            failed_counter++;
        }
    }

    // // ====================================================================
    // // GROUP 10: Point-wise Multiplication and Division
    // // ====================================================================
    log.info("=== Testing Point-wise Multiplication and Division ===");

    // Test point-wise multiplication: x = x * (mul_y * y)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        pair_vec_space.mul_pointwise(x, 2, y);
        if ((x.first[0] - 12) < eps && (x.first[1] - 28) < eps && (x.first[2] - 48) < eps &&
            (x.second[0] - 72) < eps && (x.second[1] - 100) < eps)
        {
            log.info("✓ `mul_pointwise(x, mul_y, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `mul_pointwise(x, mul_y, y)` method failed.");
            failed_counter++;
        }
    }

    // Test point-wise multiplication assignment: z = (mul_x * x) * (mul_y * y)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        vector_type z = vector_type(vector1_type{0, 0, 0}, vector2_type{0, 0});
        pair_vec_space.mul_pointwise(2, x, 3, y, z);
        if ((z.first[0] - 36) < eps && (z.first[1] - 84) < eps && (z.first[2] - 144) < eps &&
            (z.second[0] - 216) < eps && (z.second[1] - 300) < eps)
        {
            log.info("✓ `mul_pointwise(mul_x, x, mul_y, y, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `mul_pointwise(mul_x, x, mul_y, y, z)` method failed.");
            failed_counter++;
        }
    }

    // Test point-wise division assignment: z = (mul_x * x) / (mul_y * y)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        vector_type z = vector_type(vector1_type{0, 0, 0}, vector2_type{0, 0});
        pair_vec_space.div_pointwise(2, x, 3, y, z);
        if ((9 * z.first[0] - 1) < eps && (21 * z.first[1] - 4) < eps && (4 * z.first[2] - 1) < eps &&
            (27 * z.second[0] - 8) < eps && (3 * z.second[1] - 1) < eps)
        {
            log.info("✓ `div_pointwise(mul_x, x, mul_y, y, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `div_pointwise(mul_x, x, mul_y, y, z)` method failed.");
            failed_counter++;
        }
    }

    // Test point-wise division: x = x / (mul_y * y)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        pair_vec_space.div_pointwise(x, 2, y);
        if ((12 * x.first[0] - 1) < eps && (7 * x.first[1] - 1) < eps && (16 * x.first[2] - 3) < eps &&
            (18 * x.second[0] - 4) < eps && (20 * x.second[1] - 5) < eps)
        {
            log.info("✓ `div_pointwise(x, mul_y, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `div_pointwise(x, mul_y, y)` method failed.");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 11: Slice Operations
    // ====================================================================
    log.info("=== Testing Slice Operations ===");

    // assign_slices should copy selected ranges into y sequentially (global indices)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{0, 0, 0}, vector2_type{0, 0});
        std::vector<std::pair<size_t,size_t>> slices = {{0, 2}, {2, 5}}; // [0,1] + [2,3,4] -> all elements
        pair_vec_space.assign_slices(x, slices, y);
        if ((y.first[0] - 1) < eps && (y.first[1] - 2) < eps && (y.first[2] - 3) < eps &&
            (y.second[0] - 4) < eps && (y.second[1] - 5) < eps)
        {
            log.info("✓ `assign_slices(x, slices, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign_slices(x, slices, y)` method failed.");
            failed_counter++;
        }
    }

    // assign_skip_slices with skip range [1,3] should copy 0 and 4
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{0, 0, 0}, vector2_type{0, 0});
        std::vector<std::pair<size_t,size_t>> skip_slices = {{1, 3}}; // skip indices 1..3
        pair_vec_space.assign_skip_slices(x, skip_slices, y);
        if ((y.first[0] - 1) < eps && (y.first[1] - 5) < eps)
        {
            log.info("✓ `assign_skip_slices(x, skip_slices, y)` method test passed");
            passed_counter++;
        }
        else
        {
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
