#include <memory>
#include <cmath>
#include <scfd/utils/log.h>
#include <scfd/static_vec/vec.h>
#include <scfd/static_mat/mat.h>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/operations/pair_vector_space.h>
#include <nmfd/solvers/dense1_extended_solver.h>

const double eps = 1e-9;

template<class Mat, class VectorSpace>
class TestOperator
{
public:
    using mat_type = Mat;
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;

    TestOperator(std::shared_ptr<vector_space_type> vector_space, const mat_type &mat) :
        vector_space(vector_space),
        mat_(mat)
    {}

    void apply(const vector_type &x, vector_type &f)const
    {
        f = mat_*x;
    }

    void set_matrix(const mat_type &mat)
    {
        mat_ = mat;
    }

    mat_type get_matrix() const { return mat_; }

private:
    std::shared_ptr<vector_space_type> vector_space;
    mat_type mat_;
};


template<class Operator, class VectorSpace>
class linsolver
{
public:
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;
    using operator_type = Operator;
    using mat_type = typename operator_type::mat_type;

    linsolver() = default;

    void set_operator(std::shared_ptr<const operator_type> a)
    {
        a_inv = inv(a->get_matrix());
    }

    bool solve(const vector_type &rhs, vector_type &x) const
    {
        x = a_inv*rhs;
        return true;
    }

private:
    mat_type a_inv;
};


int main(int argc, char const *args[])
{
    using log_t = scfd::utils::log_std;
    using T = double;
    static const int Dim = 3;
    using mat_type = scfd::static_mat::mat<T, Dim, Dim>;

    using orig_vector_type = scfd::static_vec::vec<T, Dim>;
    using orig_vector_space_type = nmfd::operations::static_vector_space<T, Dim, orig_vector_type>;

    // TODO: Why scfd::static_vec::vec does not work here?!?!
    // using scalar_vector_type = scfd::static_vec::vec<T, 1>;
    using scalar_vector_type = std::array<T, 1>;
    using scalar_vector_space_type = nmfd::operations::static_vector_space<T, 1, scalar_vector_type>;

    using vector_type = std::pair<orig_vector_type, scalar_vector_type>;
    using vector_space_type = nmfd::operations::pair_vector_space<orig_vector_space_type, scalar_vector_space_type>;

    using operator_type = TestOperator<mat_type, orig_vector_space_type>;
    using linsolver_type = linsolver<operator_type, orig_vector_space_type>;
    using dense1_extended_operator_type = nmfd::operations::dense1_extended_operator<operator_type, orig_vector_space_type>;
    using dense1_extended_solver_type = nmfd::solvers::dense1_extended_solver<linsolver_type, operator_type, orig_vector_space_type>;


    log_t log;
    log.info("Testing dense1 extended solver implementation");
    size_t passed_counter = 0;
    size_t failed_counter = 0;

    // Initialize vector spaces
    auto orig_vector_space = std::make_shared<orig_vector_space_type>();
    auto scalar_vector_space = std::make_shared<scalar_vector_space_type>();
    auto vector_space = std::make_shared<vector_space_type>(orig_vector_space, scalar_vector_space);

    // Initialize vector and matrixes
    const mat_type mat = {1, 2, 4, 3, 5, 7, 6, 8, 9};
    const orig_vector_type u = {1, 2, 3};
    const orig_vector_type v = {4, 5, 6};
    const scalar_vector_type w = {7};

    // Initialize operators
    auto orig_operator = std::make_shared<operator_type>(orig_vector_space, mat);
    orig_operator->set_matrix(mat);
    auto dense1_extended_operator = std::make_shared<dense1_extended_operator_type>(orig_vector_space, orig_operator, u, v, w);

    // Initialize solver
    auto linsolver = std::make_shared<linsolver_type>();
    auto dense1_extended_solver = std::make_shared<dense1_extended_solver_type>(orig_vector_space, linsolver, dense1_extended_operator);


    // ====================================================================
    // GROUP 1: Test Operator (custom 3x3)
    // ====================================================================
    log.info("=== Testing Test Operator ===");
    {
        vector_type b = std::make_pair(orig_vector_type{19, 43, 69}, scalar_vector_type{50});
        vector_type res = std::make_pair(orig_vector_type{0, 0, 0}, scalar_vector_type{0});
        bool flag = dense1_extended_solver->solve(b, res);
        if (std::abs(res.first[0] - 4.0) < eps && std::abs(res.first[1] - 3.0) < eps && std::abs(res.first[2] - 2.0) < eps &&
            std::abs(res.second[0] - 1.0) < eps) {
            log.info("✓ Test operator test passed");
            passed_counter++;
        }
        else {
            log.error("✗ Test operator test failed. Expected {4, 3, 2, 1} but got {" + std::to_string(std::lround(res.first[0])) + ", " + std::to_string(std::lround(res.first[1])) + ", " + std::to_string(std::lround(res.first[2])) + ", " + std::to_string(std::lround(res.second[0])) + "}");
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
