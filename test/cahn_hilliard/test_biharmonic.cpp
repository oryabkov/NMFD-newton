#include "biharmonic_problem.h"
#include "jacobi_op.h"
#include "jacobi_pre.h"
#include "cahn_hilliard_op.h"
#include "coarsening.h"
#include "error_monitor.h"
#include "identity_op.h"
#include "include/boundary.h"
#include "jacobi_pre.h"
#include "kernels/phobic_energy.h"
#include "prolongator.h"
#include "restrictor.h"
#include "solver_logger.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nmfd/operations/rect_vector_space.h>
#include <nmfd/preconditioners/mg.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/jacobi.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <scfd/backend/backend.h>
#include <scfd/utils/log.h>
#include <string>
#include <type_traits>

struct backend
{
    using memory_type = scfd::backend::memory;
    template <class Ordinal = int>
    using for_each_type = scfd::backend::template for_each<Ordinal>;
    template <int Dim, class Ordinal = int>
    using for_each_nd_type = scfd::backend::template for_each_nd<Dim, Ordinal>;
    using reduce_type      = scfd::backend::reduce;
};

/**************************************/

constexpr int dim        = 3;
constexpr int tensor_dim = 2;

using scalar         = double;
using grid_step_type = scfd::static_vec::vec<scalar, dim>;
using idx_nd_type    = scfd::static_vec::vec<int, dim>;

using log_t = scfd::utils::log_std;

using vec_ops_t     = nmfd::rect_vector_space<scalar, /*dim=*/dim, /*tensor_dim=*/tensor_dim, backend>;
using vector_t      = typename vec_ops_t::vector_type;
using tensor_t      = scfd::static_vec::vec<scalar, tensor_dim>;
using vector_view_t = typename vector_t::view_type;

using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>;
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

using phobic_energy_t = tests::zero_potential<scalar>;
using zero_rhs_t      = tests::zero_rhs<scalar, tensor_t>;
using rhs_t           = tests::trig_rhs<scalar, tensor_t>;

using prolongator_t = tests::prolongator<vec_ops_t, log_t>;
using restrictor_t  = tests::restrictor<vec_ops_t, log_t>;
using lin_op_t      = tests::jacobi_op<vec_ops_t, log_t, phobic_energy_t>;
// using lin_op_t                  = tests::
//     cahn_hilliard_op<vec_ops_t, jacobi_op_t, log_t, phobic_energy_t, zero_rhs_t>;
using ident_op_t   = tests::identity_op<lin_op_t, vec_ops_t, log_t>;
using smoother_t   = tests::jacobi_pre<vec_ops_t, log_t, phobic_energy_t>;
using coarsening_t = tests::coarsening<lin_op_t, log_t>;

using precond_interface = nmfd::preconditioners::preconditioner_interface<vec_ops_t, lin_op_t>;

using mg_t =
    nmfd::preconditioners::mg<lin_op_t, restrictor_t, prolongator_t, smoother_t, ident_op_t, coarsening_t, log_t>;
using mg_params_t = mg_t::params_hierarchy;
using mg_utils_t  = mg_t::utils_hierarchy;

using jacobi_solver = nmfd::solvers::jacobi<vec_ops_t, lin_op_t, precond_interface, default_monitor_t, log_t>;
using gmres_solver  = nmfd::solvers::gmres<vec_ops_t, krylov_monitor_t, log_t, lin_op_t, precond_interface>;


int main( int argc, char const *argv[] )
{
    if ( argc < 6 )
    {
        std::cout << "USAGE: " << argv[0] << " <solver> <preconditioner> <grid_size> <max_iterations> <run_label>"
                  << std::endl;
        std::cout << std::endl;
        std::cout << "Required arguments:" << std::endl;
        std::cout << "    solver               Solver type: 'jacobi' or 'gmres'" << std::endl;
        std::cout << "    preconditioner       Preconditioner type: 'diag' "
                     "(diagonal/Jacobi) or "
                     "'mg' (multigrid)"
                  << std::endl;
        std::cout << "    grid_size            Number of grid points per dimension "
                     "(e.g., 32 for "
                     "32x32x32 grid)"
                  << std::endl;
        std::cout << "    max_iterations       Maximum number of solver iterations "
                     "(e.g., 100)"
                  << std::endl;
        std::cout << "    run_label            Label for this run, used in output "
                     "filenames (e.g., "
                     "'cpu', 'gpu')"
                  << std::endl;
        return 1;
    }

    // Parse command-line arguments
    const std::string solver_type         = argv[1];
    const std::string preconditioner_type = argv[2];
    const int         grid_size           = std::stoi( argv[3] );
    const int         max_iterations      = std::stoi( argv[4] );
    const std::string run_label           = argv[5];

    // Solver configuration
    const std::string solver_name  = solver_type;
    const std::string scalar_label = std::is_same_v<float, scalar> ? "float" : "double";

    log_t log;

    auto range = idx_nd_type::make_ones() * grid_size;
    auto step  = grid_step_type::make_ones() / scalar( grid_size );
    auto cond  = boundary_cond<dim>{
        { -1, -1, -1 }, // left
        { -1, -1, -1 }  // right
    };                  // -1 = dirichlet, +1 = neuman

    vector_t solution( range ), rhs( range ), exact_solution( range );
    rhs_t    rhs_function;

    auto vspace = std::make_shared<vec_ops_t>( range );
    {
        vspace->assign_scalar( 0.f, solution ); // Initialize solution to zero
        vector_view_t rhs_view( rhs, false ), exact_view( exact_solution, false );

        for ( int i = 0; i < range[0]; i++ )
        {
            for ( int j = 0; j < range[1]; j++ )
            {
                for ( int k = 0; k < range[2]; k++ )
                {
                    const auto coord_x = step[0] * ( 0.5f + i );
                    const auto coord_y = step[1] * ( 0.5f + j );
                    const auto coord_z = step[2] * ( 0.5f + k );

                    auto rhs_val   = rhs_function.get_exact_solution( coord_x, coord_y, coord_z );
                    auto exact_val = rhs_function( coord_x, coord_y, coord_z );
                    for ( int t = 0; t < tensor_dim; t++ )
                    {
                        rhs_view( i, j, k, t )   = rhs_val[t];
                        exact_view( i, j, k, t ) = exact_val[t];
                    }
                }
            }
        }

        rhs_view.release();
        exact_view.release();
    }

    auto l_op = std::make_shared<lin_op_t>( range, step, cond );

    std::shared_ptr<precond_interface> precond;
    if ( preconditioner_type == "diag" )
    {
        precond = std::make_shared<smoother_t>( l_op );
    }
    else if ( preconditioner_type == "mg" )
    {
        mg_utils_t  mg_utils;
        mg_params_t mg_params;

        mg_utils.log              = &log;
        mg_params.direct_coarse   = false;
        mg_params.num_sweeps_pre  = 4;
        mg_params.num_sweeps_post = 4;

        precond = std::make_shared<mg_t>( mg_utils, mg_params );
    }
    else
    {
        std::cout << "ERROR: Unknown preconditioner type '" << preconditioner_type << "'. Use 'diag' or 'mg'."
                  << std::endl;
        return 1;
    }

    // Ensure data directory exists
    std::filesystem::create_directories( "data" );

    // Build output filenames
    const std::string grid_size_str = std::to_string( grid_size );
    const std::string base_name =
        solver_name + "_" + preconditioner_type + "_" + run_label + "_" + grid_size_str + "_" + scalar_label;

    const std::string csv_file     = "data/" + base_name + "_metrics.csv";
    const std::string log_file     = "data/" + base_name + ".log";
    const std::string summary_file = "data/runs_summary.csv";

    // Configure solver
    const scalar tolerance = std::is_same_v<float, scalar> ? 5e-6f : 1e-10;

    // Set up custom iteration logger (writes CSV + log file at each step)
    auto iter_logger = std::make_shared<IterationLogger<vec_ops_t>>( csv_file, log_file, vspace, tolerance );

    // Solve the system and measure execution time
    std::chrono::duration<double, std::milli> solve_time_ms;
    bool                                      converged;
    std::vector<std::pair<int, scalar>>       convergence_history;

    if ( solver_type == "jacobi" )
    {
        jacobi_solver::params solver_params;
        solver_params.monitor.rel_tol                  = tolerance;
        solver_params.monitor.max_iters_num            = max_iterations;
        solver_params.monitor.save_convergence_history = true;
        jacobi_solver solver{ l_op, vspace, &log, solver_params, precond };

        solver.monitor().set_custom_funcs( iter_logger );

        {
            auto start    = std::chrono::steady_clock::now();
            converged     = solver.solve( rhs, solution );
            auto end      = std::chrono::steady_clock::now();
            solve_time_ms = ( end - start );
        }

        convergence_history = solver.monitor().convergence_history();
    }
    else // gmres
    {
        gmres_solver::params params_gmres;
        params_gmres.monitor.rel_tol = std::is_same_v<float, scalar> ? 5e-6f : 1e-10;
        // params_gmres.monitor.rel_tol = 1.0e-6;
        params_gmres.monitor.max_iters_num                = max_iterations;
        params_gmres.monitor.save_convergence_history     = true;
        params_gmres.do_restart_on_false_ritz_convergence = true;
        params_gmres.basis_size                           = 25;
        // params_gmres.basis_size = basis_size;
        params_gmres.preconditioner_side = 'L';
        params_gmres.reorthogonalization = true;
        gmres_solver solver{ l_op, vspace, &log, params_gmres, precond };

        solver.monitor().set_custom_funcs( iter_logger );

        {
            auto start    = std::chrono::steady_clock::now();
            converged     = solver.solve( rhs, solution );
            auto end      = std::chrono::steady_clock::now();
            solve_time_ms = ( end - start );
        }

        convergence_history = solver.monitor().convergence_history();
    }

    // Close logger streams
    iter_logger->close();

    // Verify that L(exact_solution) - rhs is close to zero
    vector_t L_exact( range );
    l_op->apply( exact_solution, L_exact );
    vector_t residual_exact( range );
    vspace->assign_lin_comb( scalar( 1 ), L_exact, scalar( -1 ), rhs, residual_exact );
    scalar residual_exact_norm = vspace->norm_l2( residual_exact );
    log.info_f( "Verification: ||L(exact_solution) - rhs||_2 = %le", static_cast<double>( residual_exact_norm ) );

    // Compute error between numerical and exact solutions
    vector_t error( range );
    vspace->assign_lin_comb( scalar( 1 ), solution, scalar( -1 ), exact_solution, error );
    scalar error_norm = vspace->norm_l2( error );
    scalar exact_norm = vspace->norm_l2( exact_solution );
    log.info_f(
        "Final: ||solution - exact||_2 = %le, relative = %le",
        static_cast<double>( error_norm ),
        static_cast<double>( error_norm / exact_norm )
    );

    /* -------------------------------------------------- */
    // Save solution and exact solution to binary files for visualization
    std::string data_dir = "data/";
    std::filesystem::create_directories( data_dir );

    auto save_solution = [&]( const vector_t &vec, const std::string &filename ) {
        std::ofstream out( filename, std::ios::binary );
        // Write header: dimensions and grid size
        int32_t dims[3]      = { grid_size, grid_size, grid_size };
        int32_t n_components = tensor_dim;
        out.write( reinterpret_cast<const char *>( dims ), sizeof( dims ) );
        out.write( reinterpret_cast<const char *>( &n_components ), sizeof( n_components ) );
        // Write data (psi, phi) for each grid point
        for ( int k = 0; k < grid_size; ++k )
        {
            for ( int j = 0; j < grid_size; ++j )
            {
                for ( int i = 0; i < grid_size; ++i )
                {
                    idx_nd_type idx{ i, j, k };
                    auto        v       = vec.get_vec( idx );
                    double      vals[2] = { static_cast<double>( v[0] ), static_cast<double>( v[1] ) };
                    out.write( reinterpret_cast<const char *>( vals ), sizeof( vals ) );
                }
            }
        }
        out.close();
        log.info_f( "Saved solution to %s", filename.c_str() );
    };

    save_solution( solution, data_dir + run_label + "_numerical.bin" );
    save_solution( exact_solution, data_dir + run_label + "_exact.bin" );

    // Get convergence history for summary
    auto [first_iter, initial_residual] = convergence_history.front();
    auto [last_iter, final_residual]    = convergence_history.back();
    int    total_iterations             = last_iter;
    scalar convergence_rate =
        std::pow( final_residual / initial_residual, scalar( 1 ) / std::max( 1, last_iter - first_iter ) );

    // Prepare results structure
    SolverResults<scalar> results;
    results.converged           = converged;
    results.iterations          = total_iterations;
    results.max_iterations      = max_iterations;
    results.time_ms             = solve_time_ms.count();
    results.initial_residual    = initial_residual;
    results.final_residual      = final_residual;
    results.tolerance           = tolerance;
    results.convergence_rate    = convergence_rate;
    results.solver_name         = solver_name;
    results.preconditioner_type = preconditioner_type;
    results.run_label           = run_label;
    results.scalar_label        = scalar_label;
    results.grid_size           = grid_size;
    results.tensor_dim          = tensor_dim;
    results.csv_file            = csv_file;
    results.log_file            = log_file;
    results.summary_file        = summary_file;

    // Write summary CSV and print to console
    write_summary_csv( results );
    print_solver_summary( results );

    return converged ? 0 : 1;
}
