#ifndef __TESTS_NEWTON_CONVERGENCE_MONITOR_H__
#define __TESTS_NEWTON_CONVERGENCE_MONITOR_H__

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <functional>
#include <memory>
#include <scfd/utils/logged_obj_base.h>
#include <string>
#include <type_traits>

namespace tests
{

/**
 * A ProjectOperator that tracks Newton iterations and saves:
 * - convergence_history_N.dat for each Newton iteration (from linear solver)
 * - times.dat with timing info for each Newton iteration
 * - nonlinear_history.dat with Newton residual at each step
 *
 * This uses the ProjectOperator hook which is called after each Newton update.
 *
 * The monitor uses a callback function to access the linear solver's monitor,
 * since the linear solver type (jacobi or gmres) may vary.
 */
template<class VectorSpace, class Log, class NonlinearOp, class MonitorType, class Scalar>
class newton_convergence_monitor : public scfd::utils::logged_obj_base<Log>
{
public:
    using scalar_type = Scalar;
    using vector_type = typename VectorSpace::vector_type;
    using logged_obj_type = scfd::utils::logged_obj_base<Log>;
    using monitor_accessor_t = std::function<const MonitorType*()>;

private:
    std::shared_ptr<VectorSpace> vec_space_;
    std::shared_ptr<NonlinearOp> nonlinear_op_;
    monitor_accessor_t get_linear_monitor_;
    std::string output_dir_;
    std::string solver_type_;
    std::string preconditioner_type_;
    int grid_size_;
    mutable int newton_iteration_count_ = 0;
    mutable std::ofstream times_file_;
    mutable std::ofstream nonlinear_history_file_;
    mutable bool files_initialized_ = false;
    mutable std::vector<double> linear_solve_times_;
    mutable vector_type F_x_;
    mutable bool F_x_initialized_ = false;

public:
    struct params : public logged_obj_type::params
    {
        params(const std::string &log_prefix = "", const std::string &log_name = "newton_convergence_monitor::")
            : logged_obj_type::params(0, log_prefix + log_name)
        {}
    };

    newton_convergence_monitor(
        std::shared_ptr<VectorSpace> vec_space,
        std::shared_ptr<NonlinearOp> nonlinear_op,
        monitor_accessor_t get_linear_monitor,
        const std::string& output_dir,
        const std::string& solver_type,
        const std::string& preconditioner_type,
        int grid_size,
        Log* log,
        const params& prm = params()
    ) :
        logged_obj_type(log, prm),
        vec_space_(vec_space),
        nonlinear_op_(nonlinear_op),
        get_linear_monitor_(get_linear_monitor),
        output_dir_(output_dir),
        solver_type_(solver_type),
        preconditioner_type_(preconditioner_type),
        grid_size_(grid_size)
    {
    }

    ~newton_convergence_monitor()
    {
        if (files_initialized_)
        {
            times_file_.close();
            nonlinear_history_file_.close();
        }
        if (F_x_initialized_)
        {
            vec_space_->free_vector(F_x_);
        }
    }

    /**
     * Called by default_convergence_strategy after each Newton update.
     * Saves convergence history, timing, and nonlinear residual.
     */
    void apply(vector_type& x) const
    {
        ++newton_iteration_count_;

        // Initialize files on first call
        if (!files_initialized_)
        {
            // Open times.dat with header
            std::string times_file_name = output_dir_ + "/times.dat";
            times_file_.open(times_file_name, std::ios::out | std::ios::trunc);
            times_file_ << "solver,prec,arch,float_type,size,time(ms),iters_n,reduction_rate" << std::endl;

            // Open nonlinear_history.dat with header
            std::string nonlinear_file_name = output_dir_ + "/nonlinear_history.dat";
            nonlinear_history_file_.open(nonlinear_file_name, std::ios::out | std::ios::trunc);

            files_initialized_ = true;
        }

        // Lazy initialization of F_x vector
        if (!F_x_initialized_)
        {
            vec_space_->init_vector(F_x_);
            F_x_initialized_ = true;
        }

        vec_space_->start_use_vector(F_x_);

        // Compute Newton residual: F(x)
        nonlinear_op_->apply(x, F_x_);
        scalar_type residual_norm = vec_space_->norm_l2(F_x_);

        // Save to nonlinear_history.dat
        nonlinear_history_file_ << newton_iteration_count_ << " "
                                << std::scientific << std::setprecision(15)
                                << static_cast<double>(residual_norm) << std::endl;
        nonlinear_history_file_.flush();

        // Get linear solver monitor and save convergence history
        const MonitorType* linear_monitor = get_linear_monitor_();
        if (linear_monitor)
        {
            // Save convergence history for this Newton iteration
            std::string conv_file_name = output_dir_ + "/convergence_history_" +
                                        std::to_string(newton_iteration_count_) + ".dat";
            std::ofstream conv_history(conv_file_name, std::ios::out | std::ios::trunc);

            auto res_by_it = linear_monitor->convergence_history();
            std::for_each(begin(res_by_it), end(res_by_it),
                         [&](const std::pair<int, Scalar>& pair) {
                             conv_history << pair.first << " " << pair.second << std::endl;
                         });
            conv_history.close();

            // Get timing info (we'll need to store this separately)
            // For now, we'll use a placeholder time - in practice, we'd measure it
            // The timing should be captured when the linear solve happens
            // We'll use 0.0 as placeholder and note that actual timing needs to be captured
            double linear_solve_time = 0.0;
            if (newton_iteration_count_ <= static_cast<int>(linear_solve_times_.size()))
            {
                linear_solve_time = linear_solve_times_[newton_iteration_count_ - 1];
            }

            // Determine architecture
            std::string arch;
            #ifdef PLATFORM_CUDA
            arch = "cuda";
            #elif defined(PLATFORM_OMP)
            arch = "omp";
            #elif defined(PLATFORM_SERIAL_CPU)
            arch = "cpu";
            #else
            arch = "unknown";
            #endif

            // Determine type (f or d)
            std::string type = std::is_same_v<float, Scalar> ? "f" : "d";

            // Append to times.dat
            auto res_by_it_copy = linear_monitor->convergence_history();
            if (!res_by_it_copy.empty())
            {
                auto [i_0, init_res] = res_by_it_copy.front();
                auto [i_n, final_res] = res_by_it_copy.back();

                auto conv_rate = std::pow(final_res / init_res, Scalar(1) / (i_n - i_0));

                times_file_ << std::fixed << std::setprecision(10);
                times_file_ << solver_type_ << "," << preconditioner_type_ << "," << arch << ","
                           << type << "," << grid_size_ << "," << linear_solve_time << ","
                           << i_n << "," << conv_rate << std::endl;
            }
            else
            {
                times_file_ << std::fixed << std::setprecision(10);
                times_file_ << solver_type_ << "," << preconditioner_type_ << "," << arch << ","
                           << type << "," << grid_size_ << "," << linear_solve_time << ","
                           << 0 << "," << 0.0 << std::endl;
            }
            times_file_.flush();
        }

        vec_space_->stop_use_vector(F_x_);
    }

    /**
     * Write the initial residual (iteration 0) before Newton iterations start.
     * This should be called before newton_solver->solve().
     */
    void write_initial_residual(const vector_type& x) const
    {
        // Initialize files if not already done
        if (!files_initialized_)
        {
            // Open times.dat with header
            std::string times_file_name = output_dir_ + "/times.dat";
            times_file_.open(times_file_name, std::ios::out | std::ios::trunc);
            times_file_ << "solver,prec,arch,float_type,size,time(ms),iters_n,reduction_rate" << std::endl;

            // Open nonlinear_history.dat with header
            std::string nonlinear_file_name = output_dir_ + "/nonlinear_history.dat";
            nonlinear_history_file_.open(nonlinear_file_name, std::ios::out | std::ios::trunc);
            nonlinear_history_file_ << "iteration residual" << std::endl;

            files_initialized_ = true;
        }

        // Lazy initialization of F_x vector
        if (!F_x_initialized_)
        {
            vec_space_->init_vector(F_x_);
            F_x_initialized_ = true;
        }

        vec_space_->start_use_vector(F_x_);

        // Compute Newton residual: F(x)
        nonlinear_op_->apply(x, F_x_);
        scalar_type residual_norm = vec_space_->norm_l2(F_x_);

        // Write iteration 0 to nonlinear_history.dat
        nonlinear_history_file_ << 0 << " "
                                << std::scientific << std::setprecision(15)
                                << static_cast<double>(residual_norm) << std::endl;
        nonlinear_history_file_.flush();

        vec_space_->stop_use_vector(F_x_);
    }

    // Method to record linear solve time (called externally after each linear solve)
    void record_linear_solve_time(double time_ms) const
    {
        linear_solve_times_.push_back(time_ms);
    }

    int get_newton_iteration_count() const
    {
        return newton_iteration_count_;
    }

    void reset_iteration_count()
    {
        newton_iteration_count_ = 0;
        linear_solve_times_.clear();
    }
};

} // namespace tests

#endif // __TESTS_NEWTON_CONVERGENCE_MONITOR_H__
