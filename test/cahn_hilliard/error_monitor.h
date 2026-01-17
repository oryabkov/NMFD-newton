#ifndef __TESTS_ERROR_MONITOR_H__
#define __TESTS_ERROR_MONITOR_H__

#include <memory>
#include <scfd/utils/logged_obj_base.h>

namespace tests
{

/**
 * A ProjectOperator that computes and logs the error between the current
 * numerical solution and the exact (analytical) solution at each Newton iteration.
 *
 * This uses the ProjectOperator hook which is called after each Newton update
 * in default_convergence_strategy::update_solution.
 *
 * The operator does NOT modify the solution - it only monitors the error.
 */
template<class VectorSpace, class Log>
class error_monitor : public scfd::utils::logged_obj_base<Log>
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using logged_obj_type = scfd::utils::logged_obj_base<Log>;

private:
    std::shared_ptr<VectorSpace> vec_space_;
    const vector_type& exact_solution_;
    mutable vector_type error_vec_;
    mutable bool error_vec_initialized_ = false;
    mutable int iteration_count_ = 0;

public:
    struct params : public logged_obj_type::params
    {
        params(const std::string &log_prefix = "", const std::string &log_name = "error_monitor::")
            : logged_obj_type::params(0, log_prefix + log_name)
        {}
    };

    error_monitor(
        std::shared_ptr<VectorSpace> vec_space,
        const vector_type& exact_solution,
        Log* log,
        const params& prm = params()
    ) :
        logged_obj_type(log, prm),
        vec_space_(vec_space),
        exact_solution_(exact_solution)
    {
    }

    ~error_monitor()
    {
        if (error_vec_initialized_)
        {
            vec_space_->free_vector(error_vec_);
        }
    }

    /**
     * Called by default_convergence_strategy after each Newton update.
     * Computes ||x - exact||_2 and ||x - exact||_2 / ||exact||_2
     * and logs the result.
     */
    void apply(vector_type& x) const
    {
        // Lazy initialization of error vector (to avoid needing range in constructor)
        if (!error_vec_initialized_)
        {
            vec_space_->init_vector(error_vec_);
            error_vec_initialized_ = true;
        }

        vec_space_->start_use_vector(error_vec_);

        // error_vec_ = x - exact_solution_
        vec_space_->assign_lin_comb(
            scalar_type(1.0), x,
            scalar_type(-1.0), exact_solution_,
            error_vec_
        );

        scalar_type error_norm = vec_space_->norm_l2(error_vec_);
        scalar_type exact_norm = vec_space_->norm_l2(exact_solution_);
        scalar_type relative_error = error_norm / exact_norm;

        ++iteration_count_;
        logged_obj_type::info_f(
            "iteration %d: abs_resid = %le, relative_resid = %le",
            iteration_count_,
            static_cast<double>(error_norm),
            static_cast<double>(relative_error)
        );

        vec_space_->stop_use_vector(error_vec_);
    }

    void reset_iteration_count()
    {
        iteration_count_ = 0;
    }

    int get_iteration_count() const
    {
        return iteration_count_;
    }
};

} // namespace tests

#endif // __TESTS_ERROR_MONITOR_H__
