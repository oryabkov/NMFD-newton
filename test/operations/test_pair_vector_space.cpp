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
    using scalar_type = std::pair<T, T>;
    using at_type = std::pair<size_t, size_t>;
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
    // GROUP 3: Element Access and Point-wise Operations
    // ====================================================================
    log.info("=== Testing Element Access and Point-wise Operations ===");

    // Test setting value at specific index
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        at_type at1 = at_type(0, 1);
        at_type at2 = at_type(1, 0);
        pair_vec_space.set_value_at_point(10, at1, x);
        pair_vec_space.set_value_at_point(11, at2, x);
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

    // Test getting value at specific index
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        at_type at1 = at_type(0, 1);
        at_type at2 = at_type(1, 0);
        T value1 = pair_vec_space.get_value_at_point(at1, x);
        T value2 = pair_vec_space.get_value_at_point(at2, x);
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
    // GROUP 4: Scalar-Vector Operations
    // ====================================================================
    log.info("=== Testing Scalar-Vector Operations ===");

    // Test assigning scalar to all elements
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        pair_vec_space.assign_scalar(scalar_type(10, 11), x);
        if (
            (x.first[0] - 10) < eps && (x.first[1] - 10) < eps && (x.first[2] - 10) < eps &&
            (x.second[0] - 11) < eps && (x.second[1] - 11) < eps)
        {
            log.info("✓ `assign_scalar(scalar, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign_scalar(scalar, x)` method test failed. Expected {{10, 10, 10}, {11, 11}} but got {" + std::to_string(x.first[0]) + ", " + std::to_string(x.first[1]) + ", " + std::to_string(x.first[2]) + ", " + std::to_string(x.second[0]) + ", " + std::to_string(x.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test scalar multiplication and addition: x = mul_x * x + scalar
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        scalar_type scalar = scalar_type(-1, -2);
        scalar_type mul_x = scalar_type(2, 2);
        pair_vec_space.add_mul_scalar(scalar, mul_x, x);  // x = 2*x + 10 = {2*1+10, 2*2+10, 2*3+10} = {12, 14, 16}
        if ((x.first[0] - 1) < eps && (x.first[1] - 3) < eps && (x.first[2] - 5) < eps &&
            (x.second[0] - 6) < eps && (x.second[1] - 8) < eps)
        {
            log.info("✓ `add_mul_scalar(scalar, mul_x, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul_scalar(scalar, mul_x, x)` method test failed. Expected {{12, 14, 16}, {6, 8}} but got {" + std::to_string(x.first[0]) + ", " + std::to_string(x.first[1]) + ", " + std::to_string(x.first[2]) + ", " + std::to_string(x.second[0]) + ", " + std::to_string(x.second[1]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 5: Vector Assignment and Basic Operations
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
        scalar_type mul_x = scalar_type(2, 3);
        pair_vec_space.assign_mul(mul_x, x, y);  // y = {2*{1,2,3}, 3*{4,5}} = {{2,4,6}, {12,15}}
        if ((y.first[0] - 2) < eps && (y.first[1] - 4) < eps && (y.first[2] - 6) < eps &&
            (y.second[0] - 12) < eps && (y.second[1] - 15) < eps)
        {
            log.info("✓ `assign_mul(mul_x, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign_mul(mul_x, x, y)` method test failed. Expected {{2, 4, 6}, {12, 15}} but got {" + std::to_string(y.first[0]) + ", " + std::to_string(y.first[1]) + ", " + std::to_string(y.first[2]) + ", " + std::to_string(y.second[0]) + ", " + std::to_string(y.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test linear combination assignment: z = mul_x * x + mul_y * y
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        vector_type z = vector_type(vector1_type{0, 0, 0}, vector2_type{0, 0});
        scalar_type mul_x = scalar_type(2, 3);
        scalar_type mul_y = scalar_type(3, 4);
        pair_vec_space.assign_mul(mul_x, x, mul_y, y, z);  // z = {2*{1,2,3}, 3*{4,5}} + {3*{6,7,8}, 4*{9,10}} = {{2,4,6}, {12,15}} + {{18,21,24}, {36,40}} = {{20,25,30}, {48,55}}
        if ((z.first[0] - 20) < eps && (z.first[1] - 25) < eps && (z.first[2] - 30) < eps &&
            (z.second[0] - 48) < eps && (z.second[1] - 55) < eps)
        {
            log.info("✓ `assign_mul(mul_x, x, mul_y, y, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `assign_mul(mul_x, x, mul_y, y, z)` method test failed. Expected {{20, 25, 30}, {48, 55}} but got {" + std::to_string(z.first[0]) + ", " + std::to_string(z.first[1]) + ", " + std::to_string(z.first[2]) + ", " + std::to_string(z.second[0]) + ", " + std::to_string(z.second[1]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 6: Add-Multiply Operations (In-place Linear Combinations)
    // ====================================================================
    log.info("=== Testing Add-Multiply Operations ===");

    // Test add-multiply: y = y + mul_x * x
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        scalar_type mul_x = scalar_type(2, 3);
        pair_vec_space.add_mul(mul_x, x, y);  // y = {{6,7,8}, {9,10}} + {2*{1,2,3}, 3*{4,5}} = {{8,11,14}, {21,25}}
        if ((y.first[0] - 8) < eps && (y.first[1] - 11) < eps && (y.first[2] - 14) < eps &&
            (y.second[0] - 21) < eps && (y.second[1] - 25) < eps)
        {
            log.info("✓ `add_mul(mul_x, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul(mul_x, x, y)` method test failed. Expected {{8, 11, 14}, {21, 25}} but got {" + std::to_string(y.first[0]) + ", " + std::to_string(y.first[1]) + ", " + std::to_string(y.first[2]) + ", " + std::to_string(y.second[0]) + ", " + std::to_string(y.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test add-multiply: y = mul_x * x + mul_y * y
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        scalar_type mul_x = scalar_type(2, 3);
        scalar_type mul_y = scalar_type(3, 4);
        pair_vec_space.add_mul(mul_x, x, mul_y, y);  // y = {2*{1,2,3}, 3*{4,5}} + {3*{6,7,8}, 4*{9,10}} = {{2,4,6}, {12,15}} + {{18,21,24}, {36,40}} = {{20,25,30}, {48,55}}
        if ((y.first[0] - 20) < eps && (y.first[1] - 25) < eps && (y.first[2] - 30) < eps &&
            (y.second[0] - 48) < eps && (y.second[1] - 55) < eps)
        {
            log.info("✓ `add_mul(mul_x, x, mul_y, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul(mul_x, x, mul_y, y)` method test failed. Expected {{20, 25, 30}, {48, 55}} but got {" + std::to_string(y.first[0]) + ", " + std::to_string(y.first[1]) + ", " + std::to_string(y.first[2]) + ", " + std::to_string(y.second[0]) + ", " + std::to_string(y.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test add-multiply: z = mul_x * x + mul_y * y + mul_z * z
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        vector_type z = vector_type(vector1_type{11, 12, 13}, vector2_type{14, 15});
        scalar_type mul_x = scalar_type(2, 3);
        scalar_type mul_y = scalar_type(3, 4);
        scalar_type mul_z = scalar_type(4, 5);
        pair_vec_space.add_mul(mul_x, x, mul_y, y, mul_z, z);  // z = {2*{1,2,3}, 3*{4,5}} + {3*{6,7,8}, 4*{9,10}} + {4*{11,12,13}, 5*{14,15}} = {{2,4,6}, {12,15}} + {{18,21,24}, {36,40}} + {{44,48,52}, {70,75}} = {{64,73,84}, {118,135}}
        if ((z.first[0] - 64) < eps && (z.first[1] - 73) < eps && (z.first[2] - 84) < eps &&
            (z.second[0] - 118) < eps && (z.second[1] - 135) < eps)
        {
            log.info("✓ `add_mul(mul_x, x, mul_y, y, mul_z, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `add_mul(mul_x, x, mul_y, y, mul_z, z)` method test failed. Expected {{42, 51, 60}, {126, 130}} but got {" + std::to_string(z.first[0]) + ", " + std::to_string(z.first[1]) + ", " + std::to_string(z.first[2]) + ", " + std::to_string(z.second[0]) + ", " + std::to_string(z.second[1]) + "}");
            failed_counter++;
        }
    }

    // ====================================================================
    // GROUP 7: Absolute Value Operations
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
    // GROUP 8: Min/Max Point-wise Operations
    // ====================================================================
    log.info("=== Testing Min/Max Point-wise Operations ===");

    // Test max point-wise: y = max(sc, x, y)
    {
        vector_type x = vector_type(vector1_type{1, 2, -3}, vector2_type{-9, 10});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{4, 5});
        scalar_type sc = scalar_type(7, 8);
        pair_vec_space.max_pointwise(sc, x, y);  // y = max({7, 8}, {{1,2,-3}, {-9,10}}, {{6,7,8}, {4,5}}) = {{7, 7, 8}, {8, 10}}
        if ((y.first[0] - 7) < eps && (y.first[1] - 7) < eps && (y.first[2] - 8) < eps &&
            (y.second[0] - 8) < eps && (y.second[1] - 10) < eps)
        {
            log.info("✓ `max_pointwise(sc, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `max_pointwise(sc, x, y)` method test failed. Expected {{7, 7, 8}, {8, 10}} but got {" + std::to_string(y.first[0]) + ", " + std::to_string(y.first[1]) + ", " + std::to_string(y.first[2]) + ", " + std::to_string(y.second[0]) + ", " + std::to_string(y.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test max point-wise with scalar: x = max(sc, x)
    {
        vector_type x = vector_type(vector1_type{-1, 2, 3}, vector2_type{-9, 10});
        scalar_type sc = scalar_type(2, 3);
        pair_vec_space.max_pointwise(sc, x);  // x = max({2,3}, {{1,-2,3}, {-9,10}}) = {{2, 2, 3}, {3, 10}}
        if ((x.first[0] - 2) < eps && (x.first[1] - 2) < eps && (x.first[2] - 3) < eps &&
            (x.second[0] - 3) < eps && (x.second[1] - 10) < eps)
        {
            log.info("✓ `max_pointwise(sc, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `max_pointwise(sc, x)` method test failed. Expected {{2, 2, 3}, {3, 10}} but got {" + std::to_string(x.first[0]) + ", " + std::to_string(x.first[1]) + ", " + std::to_string(x.first[2]) + ", " + std::to_string(x.second[0]) + ", " + std::to_string(x.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test min point-wise: y = min(sc, x, y)
    {
        vector_type x = vector_type(vector1_type{1, 2, -3}, vector2_type{-9, 10});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{4, 5});
        scalar_type sc = scalar_type(7, 8);
        pair_vec_space.min_pointwise(sc, x, y);  // y = min({7, 8}, {{1,2,-3}, {-9,10}}, {{6,7,8}, {4,5}}) = {{1, 2, -3}, {-9, 5}}
        if ((y.first[0] - 1) < eps && (y.first[1] - 2) < eps && (y.first[2] - 3) < eps &&
            (y.second[0] - 8) < eps && (y.second[1] - 10) < eps)
        {
            log.info("✓ `min_pointwise(sc, x, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `min_pointwise(sc, x, y)` method test failed. Expected {{1, 2, -3}, {-9, 5}} but got {" + std::to_string(y.first[0]) + ", " + std::to_string(y.first[1]) + ", " + std::to_string(y.first[2]) + ", " + std::to_string(y.second[0]) + ", " + std::to_string(y.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test min point-wise with scalar: x = min(sc, x)
    {
        vector_type x = vector_type(vector1_type{-1, 2, 3}, vector2_type{-9, 10});
        scalar_type sc = scalar_type(2, 3);
        pair_vec_space.min_pointwise(sc, x);  // x = min({2, 3}, {{-1,2,3}, {-9,10}}) = {{-1, 2, 2}, {-9, 3}}
        if ((x.first[0] + 1) < eps && (x.first[1] - 2) < eps && (x.first[2] - 2) < eps &&
            (x.second[0] + 9) < eps && (x.second[1] - 3) < eps)
        {
            log.info("✓ `min_pointwise(sc, x)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `min_pointwise(sc, x)` method test failed. Expected {{1, -2, 2}, {-9, 3}} but got {" + std::to_string(x.first[0]) + ", " + std::to_string(x.first[1]) + ", " + std::to_string(x.first[2]) + ", " + std::to_string(x.second[0]) + ", " + std::to_string(x.second[1]) + "}");
            failed_counter++;
        }
    }

    // // ====================================================================
    // // GROUP 9: Point-wise Multiplication and Division
    // // ====================================================================
    log.info("=== Testing Point-wise Multiplication and Division ===");

    // Test point-wise multiplication: x = x * (mul_y * y)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        scalar_type mul_y = scalar_type(2, 3);
        pair_vec_space.mul_pointwise(x, mul_y, y);  // x = {{1,2,3},{4,5}} * {2,3} * {{6,7,8},{9,10}} = {2*{1,2,3}*{6,7,8},3*{4,5}*{9,10}} = {{12,28,48},{108,150}}
        if ((x.first[0] - 12) < eps && (x.first[1] - 28) < eps && (x.first[2] - 48) < eps &&
            (x.second[0] - 108) < eps && (x.second[1] - 150) < eps)
        {
            log.info("✓ `mul_pointwise(x, mul_y, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `mul_pointwise(x, mul_y, y)` method test failed. Expected {{12, 28, 48}, {108, 150}} but got {" + std::to_string(x.first[0]) + ", " + std::to_string(x.first[1]) + ", " + std::to_string(x.first[2]) + ", " + std::to_string(x.second[0]) + ", " + std::to_string(x.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test point-wise multiplication assignment: z = (mul_x * x) * (mul_y * y)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        vector_type z = vector_type(vector1_type{0, 0, 0}, vector2_type{0, 0});
        scalar_type mul_x = scalar_type(2, 3);
        scalar_type mul_y = scalar_type(3, 4);
        pair_vec_space.mul_pointwise(mul_x, x, mul_y, y, z);  // z = {2,3}*{{1,2,3},{4,5}} * {3,4}*{{6,7,8},{9,10}} = {2*3*{1,2,3}*{6,7,8},3*4*{4,5}*{9,10}} = {{36,84,168},{432,600}}
        if ((z.first[0] - 36) < eps && (z.first[1] - 84) < eps && (z.first[2] - 168) < eps &&
            (z.second[0] - 432) < eps && (z.second[1] - 600) < eps)
        {
            log.info("✓ `mul_pointwise(mul_x, x, mul_y, y, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `mul_pointwise(mul_x, x, mul_y, y, z)` method test failed. Expected {{36, 84, 168}, {432, 600}} but got {" + std::to_string(z.first[0]) + ", " + std::to_string(z.first[1]) + ", " + std::to_string(z.first[2]) + ", " + std::to_string(z.second[0]) + ", " + std::to_string(z.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test point-wise division assignment: z = (mul_x * x) / (mul_y * y)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        vector_type z = vector_type(vector1_type{0, 0, 0}, vector2_type{0, 0});
        scalar_type mul_x = scalar_type(2, 3);
        scalar_type mul_y = scalar_type(3, 4);
        pair_vec_space.div_pointwise(mul_x, x, mul_y, y, z);  // z = {2,3}*{{1,2,3},{4,5}} / {3,4}*{{6,7,8},{9,10}} = {{2/18,4/21,6/24},{12/36,20/40}} = {{1/9,4/21,1/4}, {1/3,1/2}}
        if ((9 * z.first[0] - 1) < eps && (21 * z.first[1] - 4) < eps && (4 * z.first[2] - 1) < eps &&
            (3 * z.second[0] - 1) < eps && (2 * z.second[1] - 1) < eps)
        {
            log.info("✓ `div_pointwise(mul_x, x, mul_y, y, z)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `div_pointwise(mul_x, x, mul_y, y, z)` method test failed. Expected {{1/9,4/21,1/4}, {1/3,1/2}} but got {" + std::to_string(z.first[0]) + ", " + std::to_string(z.first[1]) + ", " + std::to_string(z.first[2]) + ", " + std::to_string(z.second[0]) + ", " + std::to_string(z.second[1]) + "}");
            failed_counter++;
        }
    }

    // Test point-wise division: x = x / (mul_y * y)
    {
        vector_type x = vector_type(vector1_type{1, 2, 3}, vector2_type{4, 5});
        vector_type y = vector_type(vector1_type{6, 7, 8}, vector2_type{9, 10});
        scalar_type mul_y = scalar_type(2, 3);
        pair_vec_space.div_pointwise(x, mul_y, y);  // x = {{1,2,3},{4,5}} / ({2,3} * {{6,7,8},{9,10}}) = {{1/12,2/14,3/16},{4/27,5/30}} = {{1/12,1/7,3/16},{4/27,1/6}}
        if ((12 * x.first[0] - 1) < eps && (7 * x.first[1] - 1) < eps && (16 * x.first[2] - 3) < eps &&
            (27 * x.second[0] - 4) < eps && (6 * x.second[1] - 1) < eps)
        {
            log.info("✓ `div_pointwise(x, mul_y, y)` method test passed");
            passed_counter++;
        }
        else
        {
            log.error("✗ `div_pointwise(x, mul_y, y)` method test failed. Expected {{1/12,1/7,1/6}, {4/27,1/6}} but got {" + std::to_string(x.first[0]) + ", " + std::to_string(x.first[1]) + ", " + std::to_string(x.first[2]) + ", " + std::to_string(x.second[0]) + ", " + std::to_string(x.second[1]) + "}");
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
