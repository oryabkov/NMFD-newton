#ifndef __SOLVER_LOGGER_H__
#define __SOLVER_LOGGER_H__

#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>

#include <nmfd/solvers/default_monitor.h>


/**************************************/
// Solver results structure
/**************************************/

template<typename Scalar>
struct SolverResults
{
    bool        converged;
    int         iterations;
    int         max_iterations;
    double      time_ms;
    Scalar      initial_residual;
    Scalar      final_residual;
    Scalar      tolerance;
    Scalar      convergence_rate;

    std::string solver_name;
    std::string preconditioner_type;
    std::string run_label;
    std::string scalar_label;
    int         grid_size;
    int         tensor_dim;

    std::string csv_file;
    std::string log_file;
    std::string summary_file;
};


/**************************************/
// Iteration logger - writes CSV metrics and log file at each iteration
/**************************************/

template<typename VectorSpace>
class IterationLogger : public nmfd::solvers::monitor_custom_funcs<typename VectorSpace::vector_type>
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;

    IterationLogger(const std::string& csv_file, const std::string& log_file,
                    std::shared_ptr<VectorSpace> vspace, scalar_type tolerance)
        : vspace_(vspace), tolerance_(tolerance), start_time_(std::chrono::steady_clock::now())
    {
        // Open CSV file and write header
        csv_stream_.open(csv_file, std::ios::out | std::ios::trunc);
        csv_stream_ << "iteration,residual_norm,elapsed_ms" << std::endl;

        // Open log file
        log_stream_.open(log_file, std::ios::out | std::ios::trunc);
        log_stream_ << "Solver Log" << std::endl;
        log_stream_ << "==========" << std::endl;
        log_stream_ << "Tolerance: " << std::scientific << tolerance << std::endl;
        log_stream_ << std::endl;
    }

    void check_finished(int iters_performed, const vector_type& x, const vector_type& r) override
    {
        scalar_type resid_norm = vspace_->norm(r);
        auto elapsed = std::chrono::steady_clock::now() - start_time_;
        double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();

        // Write to CSV
        csv_stream_ << iters_performed << ","
                    << std::scientific << std::setprecision(6) << resid_norm << ","
                    << std::fixed << std::setprecision(3) << elapsed_ms
                    << std::endl;

        // Write to log file
        log_stream_ << "[iter " << std::setw(5) << iters_performed << "] "
                    << "residual = " << std::scientific << std::setprecision(6) << resid_norm
                    << "  (tol = " << tolerance_ << ")"
                    << std::endl;
    }

    void close()
    {
        csv_stream_.close();
        log_stream_.close();
    }

private:
    std::shared_ptr<VectorSpace> vspace_;
    scalar_type tolerance_;
    std::chrono::steady_clock::time_point start_time_;
    std::ofstream csv_stream_;
    std::ofstream log_stream_;
};


/**************************************/
// Summary writer - writes CSV summary and prints to console
/**************************************/

template<typename Scalar>
void write_summary_csv(const SolverResults<Scalar>& results)
{
    bool file_exists = std::filesystem::exists(results.summary_file);
    std::ofstream summary_output(results.summary_file, std::ios::out | std::ios::app);

    // Write header if new file
    if (!file_exists) {
        summary_output << "solver,preconditioner,run_label,scalar_type,grid_size,"
                      << "converged,iterations,time_ms,initial_residual,final_residual,convergence_rate"
                      << std::endl;
    }

    summary_output << results.solver_name << ","
                   << results.preconditioner_type << ","
                   << results.run_label << ","
                   << results.scalar_label << ","
                   << results.grid_size << ","
                   << (results.converged ? "true" : "false") << ","
                   << results.iterations << ","
                   << std::fixed << std::setprecision(3) << results.time_ms << ","
                   << std::scientific << std::setprecision(6) << results.initial_residual << ","
                   << results.final_residual << ","
                   << std::fixed << std::setprecision(6) << results.convergence_rate
                   << std::endl;
}


template<typename Scalar>
void print_solver_summary(const SolverResults<Scalar>& r)
{
    const int grid_size = r.grid_size;
    const long long dofs = static_cast<long long>(grid_size) * grid_size * grid_size * r.tensor_dim;

    std::cout << std::endl;
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                        SOLVER SUMMARY                            ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Problem:        Biharmonic Equation (3D)                        ║" << std::endl;
    std::cout << "║  Grid size:      " << std::setw(4) << grid_size << " x " << std::setw(4) << grid_size << " x " << std::setw(4) << grid_size
              << "                              ║" << std::endl;
    std::cout << "║  DOFs:           " << std::setw(12) << dofs
              << "                                  ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Solver:         " << std::left << std::setw(20) << r.solver_name
              << "                            ║" << std::endl;
    std::cout << "║  Preconditioner: " << std::left << std::setw(20) << r.preconditioner_type
              << "                            ║" << std::endl;
    std::cout << "║  Scalar type:    " << std::left << std::setw(20) << r.scalar_label
              << "                            ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Status:         " << std::left << std::setw(20) << (r.converged ? "✓ CONVERGED" : "✗ NOT CONVERGED")
              << "                            ║" << std::endl;
    std::cout << "║  Iterations:     " << std::right << std::setw(8) << r.iterations << " / " << std::setw(8) << r.max_iterations
              << "                            ║" << std::endl;
    std::cout << "║  Time:           " << std::right << std::setw(12) << std::fixed << std::setprecision(2) << r.time_ms
              << " ms                            ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Initial resid:  " << std::scientific << std::setprecision(4) << r.initial_residual
              << "                                  ║" << std::endl;
    std::cout << "║  Final residual: " << std::scientific << std::setprecision(4) << r.final_residual
              << "                                  ║" << std::endl;
    std::cout << "║  Tolerance:      " << std::scientific << std::setprecision(4) << r.tolerance
              << "                                  ║" << std::endl;
    std::cout << "║  Conv. rate:     " << std::fixed << std::setprecision(6) << r.convergence_rate
              << "                                  ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Output files:                                                   ║" << std::endl;
    std::cout << "║    Metrics CSV:  " << std::left << std::setw(48) << r.csv_file.substr(0, 48) << " ║" << std::endl;
    std::cout << "║    Log file:     " << std::left << std::setw(48) << r.log_file.substr(0, 48) << " ║" << std::endl;
    std::cout << "║    Summary:      " << std::left << std::setw(48) << r.summary_file.substr(0, 48) << " ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
}


#endif // __SOLVER_LOGGER_H__

