#include <memory>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <type_traits>
#include <filesystem>

#include <scfd/utils/log.h>
#include <scfd/backend/serial_cpu.h>
#include <nmfd/operations/rect_vector_space.h>
#include <nmfd/preconditioners/mg.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/jacobi.h>
#include <nmfd/solvers/gmres.h>

#include "biharmonic_op.h"
#include "jacobi_pre.h"
#include "solver_logger.h"

#include "include/boundary.h"


/**************************************/


constexpr int dim = 3;
constexpr int tensor_dim = 2;

using scalar = double;
using grid_step_type    = scfd::static_vec::vec<scalar, dim>;
using idx_nd_type       = scfd::static_vec::vec<int   , dim>;

using log_t             = scfd::utils::log_std;

using vec_ops_t         = nmfd::rect_vector_space<scalar,/*dim=*/dim,/*tensor_dim=*/tensor_dim, scfd::backend::serial_cpu>;

// using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>;
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

// using prolongator_t = tests::device_prolongator<vec_ops_t, log_t>;
// using restrictor_t  = tests::device_restrictor <vec_ops_t, log_t>;
// using ident_op_t    = tests::device_identity_op<vec_ops_t, log_t>;
using lin_op_t      = tests::biharmonic_op<vec_ops_t, log_t>;
using smoother_t    = tests::jacobi_pre<vec_ops_t, log_t>;
// using coarsening_t  = tests::device_coarsening<lin_op_t, log_t>;

using precond_interface = nmfd::preconditioners::preconditioner_interface<vec_ops_t,lin_op_t>;

// using mg_t = nmfd::preconditioners::mg
// <
//     lin_op_t, restrictor_t, prolongator_t,
//     smoother_t, ident_op_t, coarsening_t,
//     log_t
// >;
// using mg_params_t = mg_t::params_hierarchy;
// using mg_utils_t  = mg_t::utils_hierarchy;


using jacobi_solver = jacobi<vec_ops_t, lin_op_t, precond_interface, default_monitor_t, log_t>;
// using gmres_solver     = nmfd::solvers::gmres <vec_ops_t, krylov_monitor_t, log_t, lin_op_t, precond_interface>;

using vector_t         = typename vec_ops_t::vector_type;
using tensor_t         = scfd::static_vec::vec<scalar, tensor_dim>;
using vector_view_t    = typename vector_t::view_type;


/**************************************/

/* Example 0

psi(x, y, z) = 2 * y*(1-y) * z*(1-z) +
               2 * x*(1-x) * z*(1-z) +
               2 * x*(1-x) * y*(1-y)

phi(x, y, z) = x*(1-x) * y*(1-y) * z*(1-z)

f  (x, y, z) = -4 * [x*(1-x) + y*(1-y) + z*(1-z)]
*/

tensor_t f(scalar x, scalar y, scalar z)
{
    tensor_t res {0., 0.};

    res[0] = -4 * (x*(1-x) + y*(1-y) + z*(1-z));
    res[1] = 0.;

    return res;
};

tensor_t u(scalar x, scalar y, scalar z)
{
    tensor_t res {0., 0.};

    // psi
    res[0] = 2 * y*(1-y) * z*(1-z) +
             2 * x*(1-x) * z*(1-z) +
             2 * x*(1-x) * y*(1-y);

    //phi
    res[1] = x*(1-x) * y*(1-y) * z*(1-z);

    return res;
};

/**************************************/


// /**************************************/

// /* Example 1

// psi(x, y, z) = 2 * (x + y) * z*(1-z) +
//                2 * (x + z) * y*(1-y) +
//                2 * (y + z) * x*(1-x)

// phi(x, y, z) = x*(1-x) * y*(1-y) * z*(1-z) * (x + y + z)

// f  (x, y, z) = -4 * (x + y + z)
// */

// tensor_t f(scalar x, scalar y, scalar z)
// {
//     tensor_t res {0., 0.};

//     res[0] = -4 * (x + y + z);
//     res[1] = 0.;

//     return res;
// };

// tensor_t u(scalar x, scalar y, scalar z)
// {
//     tensor_t res {0., 0.};

//     // psi
//     res[0] = 2 * (x + y) * z*(1-z) +
//              2 * (x + z) * y*(1-y) +
//              2 * (y + z) * x*(1-x);

//     //phi
//     res[1] = x*(1-x) * y*(1-y) * z*(1-z) * (x + y + z);

//     return res;
// };

// /**************************************/


int main(int argc, char const *argv[])
{
    if (argc < 5)
    {
        std::cout << "USAGE: " << argv[0] << " <preconditioner> <grid_size> <max_iterations> <run_label>" << std::endl;
        std::cout << std::endl;
        std::cout << "Required arguments:" << std::endl;
        std::cout << "    preconditioner       Preconditioner type: 'diag' (diagonal/Jacobi) or 'mg' (multigrid)" << std::endl;
        std::cout << "    grid_size            Number of grid points per dimension (e.g., 32 for 32x32x32 grid)" << std::endl;
        std::cout << "    max_iterations       Maximum number of solver iterations (e.g., 100)" << std::endl;
        std::cout << "    run_label            Label for this run, used in output filenames (e.g., 'cpu', 'gpu')" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    const std::string preconditioner_type = argv[1];
    const int         grid_size           = std::stoi(argv[2]);
    const int         max_iterations      = std::stoi(argv[3]);
    const std::string run_label           = argv[4];

    // Solver configuration
    const std::string solver_name  = "jacobi";
    const std::string scalar_label = std::is_same_v<float, scalar> ? "float" : "double";

    log_t log;

    auto range = idx_nd_type::make_ones() * grid_size;
    auto step  = grid_step_type::make_ones() / scalar(grid_size);
    auto cond  = boundary_cond<dim>
    {
        {-1, -1, -1}, // left
        {-1, -1, -1}  // right
    }; // -1 = dirichlet, +1 = neuman

    vector_t solution(range), rhs(range), exact_solution(range);

    auto vspace = std::make_shared<vec_ops_t>(range);
    {
        vspace->assign_scalar(0.f, solution);  // Initialize solution to zero
        vector_view_t rhs_view(rhs, false), exact_view(exact_solution, false);

        for(int i=0; i<range[0]; i++) {
            for(int j=0; j<range[1]; j++) {
                for(int k=0; k<range[2]; k++)
                {
                    const auto coord_x = step[0] * (0.5f + i);
                    const auto coord_y = step[1] * (0.5f + j);
                    const auto coord_z = step[2] * (0.5f + k);

                    auto rhs_val   = f(coord_x, coord_y, coord_z);
                    auto exact_val = u(coord_x, coord_y, coord_z);
                    for (int t = 0; t < tensor_dim; t++) {
                        rhs_view(i, j, k, t)   = rhs_val[t];
                        exact_view(i, j, k, t) = exact_val[t];
                    }
                }
            }
        }

        rhs_view.release();
        exact_view.release();
    }

    auto l_op = std::make_shared<lin_op_t>(range, step, cond);

    std::shared_ptr<precond_interface> precond;
    if (preconditioner_type == "diag")
    {
        precond = std::make_shared<smoother_t>(l_op);
    }
    // else if (preconditioner_type == "mg")
    // {
    //     mg_utils_t    mg_utils;
    //     mg_params_t   mg_params;

    //     mg_utils.log               = &log;
    //     mg_params.direct_coarse    = false;
    //     mg_params.num_sweeps_pre   = 3;
    //     mg_params.num_sweeps_post  = 3;

    //     precond = std::make_shared<mg_t>(mg_utils, mg_params);
    // }
    else
    {
        std::cout << "ERROR: Unknown preconditioner type '" << preconditioner_type << "'. Use 'diag' or 'mg'." << std::endl;
        return 1;
    }

    // Ensure data directory exists
    std::filesystem::create_directories("data");

    // Build output filenames
    const std::string grid_size_str = std::to_string(grid_size);
    const std::string base_name = solver_name + "_" + preconditioner_type + "_"
                                + run_label + "_" + grid_size_str + "_" + scalar_label;

    const std::string csv_file     = "data/" + base_name + "_metrics.csv";
    const std::string log_file     = "data/" + base_name + ".log";
    const std::string summary_file = "data/runs_summary.csv";

    // Configure solver
    const scalar tolerance = std::is_same_v<float, scalar> ? 5e-6f : 1e-10;

    jacobi_solver::params solver_params;
    solver_params.monitor.rel_tol = tolerance;
    solver_params.monitor.max_iters_num = max_iterations;
    solver_params.monitor.save_convergence_history = true;
    jacobi_solver solver{l_op, vspace, &log, solver_params, precond};  // nullptr log to disable console output

    // Set up custom iteration logger (writes CSV + log file at each step)
    auto iter_logger = std::make_shared<IterationLogger<vec_ops_t>>(csv_file, log_file, vspace, tolerance);
    solver.monitor().set_custom_funcs(iter_logger);

    // Solve the system and measure execution time
    std::chrono::duration<double, std::milli> solve_time_ms;
    bool converged;
    {
        auto start = std::chrono::steady_clock::now();
        converged = solver.solve(rhs, solution);
        auto end = std::chrono::steady_clock::now();
        solve_time_ms = (end - start);
    }

    // Close logger streams
    iter_logger->close();

    // Get convergence history for summary
    auto convergence_history = solver.monitor().convergence_history();
    auto [first_iter, initial_residual] = convergence_history.front();
    auto [last_iter,  final_residual]   = convergence_history.back();
    int total_iterations = last_iter;
    scalar convergence_rate = std::pow(final_residual / initial_residual, scalar(1) / std::max(1, last_iter - first_iter));

    // Prepare results structure
    SolverResults<scalar> results;
    results.converged          = converged;
    results.iterations         = total_iterations;
    results.max_iterations     = max_iterations;
    results.time_ms            = solve_time_ms.count();
    results.initial_residual   = initial_residual;
    results.final_residual     = final_residual;
    results.tolerance          = tolerance;
    results.convergence_rate   = convergence_rate;
    results.solver_name        = solver_name;
    results.preconditioner_type = preconditioner_type;
    results.run_label          = run_label;
    results.scalar_label       = scalar_label;
    results.grid_size          = grid_size;
    results.tensor_dim         = tensor_dim;
    results.csv_file           = csv_file;
    results.log_file           = log_file;
    results.summary_file       = summary_file;

    // Write summary CSV and print to console
    write_summary_csv(results);
    print_solver_summary(results);

    return converged ? 0 : 1;
}
