#include <cmath>
#include <array>
#include <tuple>
#include <scfd/utils/log.h>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/operations/pair_vector_space.h>
#include <nmfd/operations/dense1_extended_operator.h>

const double eps = 1e-10;

template<class VectorSpace>
class IdentityOperator
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;

    static_assert(std::tuple_size<vector_type>::value == 3, "VectorSpace must be 3D");

    IdentityOperator(std::shared_ptr<vector_space_type> vector_space) : vector_space(vector_space) {}

    void apply(const vector_type &x, vector_type &f)const
    {
        vector_space->assign(x, f);
    }

private:
    std::shared_ptr<vector_space_type> vector_space;
};

template<class VectorSpace>
class TestOperator
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;

    static_assert(std::tuple_size<vector_type>::value == 3, "VectorSpace must be 3D");

private:
    std::shared_ptr<vector_space_type> vector_space;

public:
    TestOperator(std::shared_ptr<vector_space_type> vector_space) : vector_space(vector_space) {}

    void apply(const vector_type &x, vector_type &f)const
    {
        // | 1 2 3 |
        // | 4 5 6 |
        // | 7 8 9 |
        f[0] = x[0]*1 + x[1]*2 + x[2]*3;
        f[1] = x[0]*4 + x[1]*5 + x[2]*6;
        f[2] = x[0]*7 + x[1]*8 + x[2]*9;
    }
};


int main(int argc, char const *args[])
{
    using log_t = scfd::utils::log_std;
    using T = double;
    static const int Dim = 3;

    using orig_vector_type = std::array<T, Dim>;
    using orig_vector_space_type = nmfd::operations::static_vector_space<T, Dim, orig_vector_type>;

    using scalar_vector_type = std::array<T, 1>;
    using scalar_vector_space_type = nmfd::operations::static_vector_space<T, 1, scalar_vector_type>;

    using vector_type = std::pair<orig_vector_type, scalar_vector_type>;
    using vector_space_type = nmfd::operations::pair_vector_space<orig_vector_space_type, scalar_vector_space_type>;

    log_t log;
    log.info("Testing dense1 extended operator implementation");
    size_t passed_counter = 0;
    size_t failed_counter = 0;

    // Initialize vector spaces
    auto orig_vector_space = std::make_shared<orig_vector_space_type>();
    auto scalar_vector_space = std::make_shared<scalar_vector_space_type>();
    auto vector_space = std::make_shared<vector_space_type>(orig_vector_space, scalar_vector_space);

    // Initialize extended values
    const orig_vector_type u = {1, 2, 3};
    const orig_vector_type v = {4, 5, 6};
    const scalar_vector_type w = {7};


    // ====================================================================
    // GROUP 1: Identity Operator
    // ====================================================================
    log.info("=== Testing Identity Operator ===");
    {
        using identity_operator_type = IdentityOperator<orig_vector_space_type>;
        auto identity_operator = std::make_shared<const identity_operator_type>(orig_vector_space);

        using dense1_extended_operator_type = nmfd::operations::dense1_extended_operator<identity_operator_type, orig_vector_space_type>;
        auto dense1_extended_operator = std::make_shared<dense1_extended_operator_type>(orig_vector_space, identity_operator, u, v, w);

        vector_type in = std::make_pair(orig_vector_type{1, 2, 3}, scalar_vector_type{4});
        vector_type out = std::make_pair(orig_vector_type{0, 0, 0}, scalar_vector_type{0});
        vector_type out_true = std::make_pair(orig_vector_type{5, 10, 15}, scalar_vector_type{60});
        dense1_extended_operator->apply(in, out);
        if (std::abs(out.first[0] - out_true.first[0]) < eps && std::abs(out.first[1] - out_true.first[1]) < eps && std::abs(out.first[2] - out_true.first[2]) < eps &&
            std::abs(out.second[0] - out_true.second[0]) < eps)
        {
            log.info("✓ Identity operator test passed");
            passed_counter++;
        }
        else {
            log.error("✗ Identity operator test failed. Expected {" + std::to_string(round(out_true.first[0])) + ", " + std::to_string(round(out_true.first[1])) + ", " + std::to_string(round(out_true.first[2])) + ", " + std::to_string(round(out_true.second[0])) + "} but got {" + std::to_string(round(out.first[0])) + ", " + std::to_string(round(out.first[1])) + ", " + std::to_string(round(out.first[2])) + ", " + std::to_string(round(out.second[0])) + "}");
            failed_counter++;
        }
    }


    // ====================================================================
    // GROUP 2: Test Operator (custom 3x3)
    // ====================================================================
    log.info("=== Testing Test Operator ===");
    {
        using test_operator_type = TestOperator<orig_vector_space_type>;
        auto test_operator = std::make_shared<const test_operator_type>(orig_vector_space);

        using dense1_extended_operator_type = nmfd::operations::dense1_extended_operator<test_operator_type, orig_vector_space_type>;
        auto dense1_extended_operator = std::make_shared<dense1_extended_operator_type>(orig_vector_space, test_operator, u, v, w);

        vector_type in = std::make_pair(orig_vector_type{1, 2, 3}, scalar_vector_type{4});
        vector_type out = std::make_pair(orig_vector_type{0, 0, 0}, scalar_vector_type{0});
        vector_type out_true = std::make_pair(orig_vector_type{18, 40, 62}, scalar_vector_type{60});
        dense1_extended_operator->apply(in, out);
        if (std::abs(out.first[0] - out_true.first[0]) < eps && std::abs(out.first[1] - out_true.first[1]) < eps && std::abs(out.first[2] - out_true.first[2]) < eps &&
            std::abs(out.second[0] - out_true.second[0]) < eps) {
            log.info("✓ Test operator test passed");
            passed_counter++;
        }
        else {
            log.error("✗ Test operator test failed. Expected {" + std::to_string(round(out_true.first[0])) + ", " + std::to_string(round(out_true.first[1])) + ", " + std::to_string(round(out_true.first[2])) + ", " + std::to_string(round(out_true.second[0])) + "} but got {" + std::to_string(round(out.first[0])) + ", " + std::to_string(round(out.first[1])) + ", " + std::to_string(round(out.first[2])) + ", " + std::to_string(round(out.second[0])) + "}");
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
