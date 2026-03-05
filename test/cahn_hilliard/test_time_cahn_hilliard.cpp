#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <nmfd/operations/rect_vector_space.h>
#include <nmfd/preconditioners/mg.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <scfd/backend/backend.h>
#include <scfd/utils/log.h>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "cahn_hilliard_op.h"
#include "jacobi_op.h"
#include "jacobi_pre.h"

#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/jacobi.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>

#include "cahn_hilliard_problem.h"
#include "coarsening.h"
#include "convergence_history_io.h"
#include "error_monitor.h"
#include "identity_op.h"
#include "include/boundary.h"
#include "kernels/phobic_energy.h"
#include "newton_convergence_monitor.h"
#include "prolongator.h"
#include "restrictor.h"
#include "solution_io.h"
#include "timers.h"


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

#ifndef USE_DOUBLE_PRECISION
using scalar      = float;
#else
using scalar      = double;
#endif
using grid_step_type = scfd::static_vec::vec<scalar, dim>;
using idx_nd_type    = scfd::static_vec::vec<int, dim>;

using log_t = scfd::utils::log_std;

using vec_ops_t     = nmfd::rect_vector_space<scalar, /*dim=*/dim, /*tensor_dim=*/tensor_dim, backend>;
using vector_t      = typename vec_ops_t::vector_type;
using tensor_t      = scfd::static_vec::vec<scalar, tensor_dim>;
using vector_view_t = typename vector_t::view_type;

// Monitors
using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>;
using error_monitor_t   = tests::error_monitor<vec_ops_t, log_t>;
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

// Problem
using phobic_energy     = tests::double_well_potential<scalar>;
using rhs_t             = tests::trig_rhs<scalar, tensor_t, 4, 3, 1>;
using time_derivative_t = tests::time_derivative<vec_ops_t, tensor_t>;

// MG
using prolongator_t = tests::prolongator<vec_ops_t, log_t>;
using restrictor_t  = tests::restrictor<vec_ops_t, log_t>;
using jacobi_op_t   = tests::jacobi_op<vec_ops_t, log_t, phobic_energy, time_derivative_t>;
using ident_op_t    = tests::identity_op<jacobi_op_t, vec_ops_t, log_t>;
using smoother_t    = tests::jacobi_pre<vec_ops_t, log_t, phobic_energy, time_derivative_t>;
using coarsening_t  = tests::coarsening<jacobi_op_t, log_t>;

using precond_interface = nmfd::preconditioners::preconditioner_interface<vec_ops_t, jacobi_op_t>;

using mg_t =
    nmfd::preconditioners::mg<jacobi_op_t, restrictor_t, prolongator_t, smoother_t, ident_op_t, coarsening_t, log_t>;
using mg_params_t = mg_t::params_hierarchy;
using mg_utils_t  = mg_t::utils_hierarchy;

using jacobi_solver = nmfd::solvers::jacobi<vec_ops_t, jacobi_op_t, precond_interface, default_monitor_t, log_t>;
using gmres_solver  = nmfd::solvers::gmres<vec_ops_t, krylov_monitor_t, log_t, jacobi_op_t, precond_interface>;

// Newton
using cahn_hilliard_op_t = tests::cahn_hilliard_op<vec_ops_t, jacobi_op_t, log_t, phobic_energy, rhs_t, time_derivative_t>;

/**************************************/
// Logging helpers
/**************************************/

std::string get_timestamp_string()
{
    auto               now = std::chrono::system_clock::now();
    std::time_t        t   = std::chrono::system_clock::to_time_t( now );
    std::tm            tm  = *std::localtime( &t );
    std::ostringstream oss;
    oss << std::put_time( &tm, "%Y%m%d_%H%M%S" );
    return oss.str();
}

class tee_streambuf : public std::streambuf
{
public:
    tee_streambuf( std::streambuf *sb1, std::streambuf *sb2 ) : sb1_( sb1 ), sb2_( sb2 )
    {
    }

protected:
    int overflow( int c ) override
    {
        if ( c != EOF )
        {
            if ( sb1_ )
                sb1_->sputc( c );
            if ( sb2_ )
                sb2_->sputc( c );
        }
        return c;
    }

    int sync() override
    {
        int r1 = sb1_ ? sb1_->pubsync() : 0;
        int r2 = sb2_ ? sb2_->pubsync() : 0;
        return ( r1 == 0 && r2 == 0 ) ? 0 : -1;
    }

private:
    std::streambuf *sb1_;
    std::streambuf *sb2_;
};

/**************************************/
// Default solver parameters
/**************************************/
constexpr int    DEFAULT_MAX_ITERATIONS = 100;
constexpr int    DEFAULT_GMRES_BASIS    = 25;
constexpr int    DEFAULT_MG_SWEEPS_PRE  = 4;
constexpr int    DEFAULT_MG_SWEEPS_POST = 4;
constexpr scalar DEFAULT_NEWTON_TOL     = std::is_same<float, scalar>::value ? 5e-6f : 1e-10;
constexpr scalar DEFAULT_TOLERANCE      = std::is_same<float, scalar>::value ? 5e-6f : 1e-10;
constexpr int    DEFAULT_MAX_TIME_STEPS = 1;
constexpr scalar DEFAULT_DT_INF         = 1.0;
constexpr scalar DEFAULT_TIME_TOL       = std::is_same<float, scalar>::value ? 5e-6f : 1e-10;


/**************************************/

int main( int argc, char const *argv[] )
{
    // Parse CLI arguments
    bool        save_coords = false;
    bool        verbose     = false;
    std::string prefix      = "run";
    int         grid_size   = 32;
    std::string solver_type;
    std::string preconditioner_type;

    // Solver parameters (initialized to defaults)
    int    max_iterations = DEFAULT_MAX_ITERATIONS;
    int    gmres_basis    = DEFAULT_GMRES_BASIS;
    int    mg_sweeps_pre  = DEFAULT_MG_SWEEPS_PRE;
    int    mg_sweeps_post = DEFAULT_MG_SWEEPS_POST;
    scalar newton_tol     = DEFAULT_NEWTON_TOL;
    scalar tolerance      = DEFAULT_TOLERANCE;
    int    max_time_steps = DEFAULT_MAX_TIME_STEPS;
    scalar dt_inf         = DEFAULT_DT_INF;
    scalar time_tol       = DEFAULT_TIME_TOL;

    if ( argc < 4 )
    {
        std::cout << "USAGE: " << argv[0] << " <solver> <preconditioner> <grid_size> [prefix] [options...]"
                  << std::endl;
        std::cout << std::endl;
        std::cout << "Required arguments:" << std::endl;
        std::cout << "    solver               Solver type: 'jacobi' or 'gmres'" << std::endl;
        std::cout << "    preconditioner       Preconditioner type: 'diag' (diagonal/Jacobi) or 'mg' (multigrid)"
                  << std::endl;
        std::cout << "    grid_size            Number of grid points per dimension (e.g., 32)" << std::endl;
        std::cout << std::endl;
        std::cout << "Optional arguments:" << std::endl;
        std::cout << "    prefix               Output prefix (default: 'run')" << std::endl;
        std::cout << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "    --save-coords        Save numerical and exact solutions to binary files" << std::endl;
        std::cout << "    --verbose            Save convergence history to conv_history.dat" << std::endl;
        std::cout << "    --max-iterations N   Maximum solver iterations (default: " << DEFAULT_MAX_ITERATIONS << ")"
                  << std::endl;
        std::cout << "    --gmres-basis N      GMRES basis size (default: " << DEFAULT_GMRES_BASIS << ")" << std::endl;
        std::cout << "    --mg-sweeps-pre N    Multigrid pre-sweeps (default: " << DEFAULT_MG_SWEEPS_PRE << ")"
                  << std::endl;
        std::cout << "    --mg-sweeps-post N   Multigrid post-sweeps (default: " << DEFAULT_MG_SWEEPS_POST << ")"
                  << std::endl;
        std::cout << "    --newton-tol T       Newton solver tolerance (default: " << std::scientific
                  << DEFAULT_NEWTON_TOL << std::defaultfloat << ")" << std::endl;
        std::cout << "    --tolerance T        Linear solver tolerance (default: " << std::scientific << tolerance
                  << std::defaultfloat << ")" << std::endl;
        std::cout << "    --max-time-steps N   Maximum number of time steps (default: " << DEFAULT_MAX_TIME_STEPS << ")" << std::endl;
        std::cout << "    --dt-inf T           dt_inf parameter (1/dt, default: " << DEFAULT_DT_INF << ")" << std::endl;
        std::cout << "    --time-tol T         Time convergence tolerance (default: " << std::scientific
                  << DEFAULT_TIME_TOL << std::defaultfloat << ")" << std::endl;
        return 1;
    }

    solver_type         = argv[1];
    preconditioner_type = argv[2];
    grid_size           = std::stoi( argv[3] );

    // Validate solver and preconditioner types
    if ( solver_type != "jacobi" && solver_type != "gmres" )
    {
        std::cerr << "ERROR: Unknown solver type '" << solver_type << "'. Use 'jacobi' or 'gmres'." << std::endl;
        return 1;
    }

    if ( preconditioner_type != "diag" && preconditioner_type != "mg" )
    {
        std::cerr << "ERROR: Unknown preconditioner type '" << preconditioner_type << "'. Use 'diag' or 'mg'."
                  << std::endl;
        return 1;
    }

    // Parse optional arguments
    for ( int i = 4; i < argc; ++i )
    {
        std::string arg = argv[i];
        if ( arg == "--save-coords" )
        {
            save_coords = true;
        }
        else if ( arg == "--verbose" )
        {
            verbose = true;
        }
        else if ( arg == "--max-iterations" && i + 1 < argc )
        {
            max_iterations = std::stoi( argv[++i] );
        }
        else if ( arg == "--gmres-basis" && i + 1 < argc )
        {
            gmres_basis = std::stoi( argv[++i] );
        }
        else if ( arg == "--mg-sweeps-pre" && i + 1 < argc )
        {
            mg_sweeps_pre = std::stoi( argv[++i] );
        }
        else if ( arg == "--mg-sweeps-post" && i + 1 < argc )
        {
            mg_sweeps_post = std::stoi( argv[++i] );
        }
        else if ( arg == "--newton-tol" && i + 1 < argc )
        {
            newton_tol = std::stod( argv[++i] );
        }
        else if ( arg == "--tolerance" && i + 1 < argc )
        {
            tolerance = std::stod( argv[++i] );
        }
        else if ( arg == "--max-time-steps" && i + 1 < argc )
        {
            max_time_steps = std::stoi( argv[++i] );
        }
        else if ( arg == "--dt-inf" && i + 1 < argc )
        {
            dt_inf = std::stod( argv[++i] );
        }
        else if ( arg == "--time-tol" && i + 1 < argc )
        {
            time_tol = std::stod( argv[++i] );
        }
        else if ( arg.find( "--" ) == 0 )
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
        else
        {
            prefix = arg;
        }
    }

    // Create output directory with timestamp
    std::string output_dir = "data/" + prefix + "_" + get_timestamp_string();
    std::filesystem::create_directories( output_dir );

    // Open log file and set up tee output
    std::ofstream log_file( output_dir + "/log.txt" );
    tee_streambuf tee_buf( std::cout.rdbuf(), log_file.rdbuf() );
    std::ostream  tee_out( &tee_buf );

    // Redirect std::cout to tee
    auto *old_cout_buf = std::cout.rdbuf( &tee_buf );

    // Solver configuration
    const std::string scalar_label = std::is_same<float, scalar>::value ? "float" : "double";

    log_t log;
    // Set log verbosity: 0 suppresses INFO messages, 1 allows them
    log.set_verbosity( verbose ? 1 : 0 );

    // Write configuration header to log
    std::cout << "========================================" << std::endl;
    std::cout << "Cahn-Hilliard Time-Dependent Solver Configuration" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Problem Settings:" << std::endl;
    std::cout << "  Grid size:     " << grid_size << " x " << grid_size << " x " << grid_size << std::endl;
    std::cout << "  Tensor dim:    " << tensor_dim << std::endl;
    std::cout << "  Scalar type:   " << scalar_label << std::endl;
    std::cout << "  DOFs:          " << static_cast<long long>( grid_size ) * grid_size * grid_size * tensor_dim
              << std::endl;
    std::cout << std::endl;
    std::cout << "Newton Solver:" << std::endl;
    std::cout << "  Tolerance:     " << std::scientific << newton_tol << std::endl;
    std::cout << std::endl;
    std::cout << "Linear Solver:" << std::endl;
    std::cout << "  Type:          " << solver_type << std::endl;
    std::cout << "  Tolerance:     " << std::scientific << tolerance << std::endl;
    std::cout << "  Max iters:     " << max_iterations << std::endl;
    if ( solver_type == "gmres" )
    {
    std::cout << "  Basis size:    " << gmres_basis << std::endl;
    std::cout << "  Precond side:  L" << std::endl;
    std::cout << "  Reorthogon.:   true" << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Preconditioner:" << std::endl;
    std::cout << "  Type:          " << preconditioner_type << std::endl;
    if ( preconditioner_type == "mg" )
    {
    std::cout << "  Pre-sweeps:    " << mg_sweeps_pre << std::endl;
    std::cout << "  Post-sweeps:   " << mg_sweeps_post << std::endl;
    std::cout << "  Direct coarse: false" << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Time Integration:" << std::endl;
    std::cout << "  Max time steps: " << max_time_steps << std::endl;
    std::cout << "  dt_inf:         " << dt_inf << std::endl;
    std::cout << "  Time tol:       " << std::scientific << time_tol << std::defaultfloat << std::endl;
    std::cout << std::endl;
    std::cout << "Output:" << std::endl;
    std::cout << "  Directory:     " << output_dir << std::endl;
    std::cout << "  Save coords:   " << ( save_coords ? "yes" : "no" ) << std::endl;
    std::cout << "  Verbose:       " << ( verbose ? "yes" : "no" ) << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    auto range = idx_nd_type::make_ones() * grid_size;
    auto step  = grid_step_type::make_ones() / scalar( grid_size );
    auto cond  = boundary_cond<dim, tensor_dim>{
        { { -1, -1 }, { -1, -1 }, { -1, -1 } }, // left: [x,y,z][psi,phi]
        { { -1, -1 }, { -1, -1 }, { -1, -1 } }  // right: [x,y,z][psi,phi]
    };
    // Boundary condition values:
    //   -1 = dirichlet (value = 0 at boundary)
    //   +1 = neumann (derivative = 0 at boundary)
    //    0 = periodic (left boundary uses value from N-1, right boundary uses value from 0)

    vector_t solution( range ), exact_solution( range );
    rhs_t    rhs;
    auto     vspace = std::make_shared<vec_ops_t>( range );
    {
        vspace->assign_scalar( 0.0, solution );
        vector_view_t exact_view( exact_solution, false );

        for ( int i = 0; i < range[0]; i++ )
        {
            for ( int j = 0; j < range[1]; j++ )
            {
                for ( int k = 0; k < range[2]; k++ )
                {
                    scalar x = step[0] * ( 0.5 + i );
                    scalar y = step[1] * ( 0.5 + j );
                    scalar z = step[2] * ( 0.5 + k );

                    auto exact_val = rhs.get_exact_solution( x, y, z );
                    for ( int t = 0; t < tensor_dim; t++ )
                    {
                        exact_view( i, j, k, t ) = exact_val[t];
                    }
                }
            }
        }

        exact_view.release();
    }

    auto time_derivative = std::make_shared<time_derivative_t>( range );
    time_derivative->set_dt_inf( dt_inf );

    // Time-dependent operators (used for solving the time-dependent equation)
    auto cahn_hilliard_jacobi_op = std::make_shared<jacobi_op_t>( range, step, cond, time_derivative );
    auto cahn_hilliard_op        = std::make_shared<cahn_hilliard_op_t>( range, step, cond, cahn_hilliard_jacobi_op, time_derivative );

    // Stationary operators (used for checking time convergence to stationary solution)
    auto cahn_hilliard_jacobi_op_stationary = std::make_shared<jacobi_op_t>( range, step, cond );
    auto cahn_hilliard_op_stationary        = std::make_shared<cahn_hilliard_op_t>( range, step, cond, cahn_hilliard_jacobi_op_stationary );

    std::shared_ptr<precond_interface> precond;
    if ( preconditioner_type == "diag" )
    {
        precond = std::make_shared<smoother_t>( cahn_hilliard_jacobi_op );
    }
    else if ( preconditioner_type == "mg" )
    {
    mg_utils_t  mg_utils;
    mg_params_t mg_params;

    mg_utils.log              = &log;
    mg_params.direct_coarse   = false;
    mg_params.num_sweeps_pre  = mg_sweeps_pre;
    mg_params.num_sweeps_post = mg_sweeps_post;

    precond = std::make_shared<mg_t>( mg_utils, mg_params );
    }

    // Verify that F_stationary(exact_solution) is close to zero
    vector_t F_exact( range );
    cahn_hilliard_op_stationary->apply( exact_solution, F_exact );
    scalar F_exact_norm = vspace->norm_l2( F_exact );
    log.info_f( "Verification: ||F_stationary(exact_solution)||_2 = %le", static_cast<double>( F_exact_norm ) );

    // Open time convergence history file
    std::ofstream time_conv_file( output_dir + "/time_converge_history.dat" );
    time_conv_file << "step F_stationary_norm" << std::endl;

    // Compute and write initial F(x) norm (step 0)
    vector_t F_x_init( range );
    cahn_hilliard_op_stationary->apply( solution, F_x_init );
    scalar F_x_init_norm = vspace->norm_l2( F_x_init );
    time_conv_file << 0 << " " << std::scientific << std::setprecision(15)
                   << static_cast<double>( F_x_init_norm ) << std::endl;
    time_conv_file.flush();

    // Save initial approximation (index 0) if requested
    if ( save_coords )
    {
        std::string numerical_file = output_dir + "/numerical_0.bin";
        tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
    }

    // Solve and measure time for each time step
    std::vector<double> iteration_times;
    double              total_time_ms = 0.0;

    // Create solver instances based on type (will be reused for each time step)
    if ( solver_type == "jacobi" )
    {
        using newton_iteration_jacobi_t = nmfd::solvers::newton_iteration<vec_ops_t, cahn_hilliard_op_t, jacobi_solver>;
        using newton_conv_monitor_jacobi_t = tests::newton_convergence_monitor<vec_ops_t, log_t, cahn_hilliard_op_t, default_monitor_t, scalar>;
        using newton_solver_jacobi_t = nmfd::solvers::nonlinear_solver<vec_ops_t, log_t, cahn_hilliard_op_t, newton_iteration_jacobi_t, newton_conv_monitor_jacobi_t>;

        jacobi_solver::params solver_params;
        solver_params.monitor.rel_tol                  = tolerance;
        solver_params.monitor.max_iters_num            = max_iterations;
        solver_params.monitor.save_convergence_history = true;
        auto jacobi_lin_solver = std::make_shared<jacobi_solver>( cahn_hilliard_jacobi_op, vspace, &log, solver_params, precond );

        auto newton_iteration = std::make_shared<newton_iteration_jacobi_t>( vspace, jacobi_lin_solver );

        auto get_monitor = [jacobi_lin_solver]() -> const default_monitor_t* {
            return &(jacobi_lin_solver->monitor());
        };

        auto newton_solver = std::make_shared<newton_solver_jacobi_t>( vspace, &log, newton_iteration );
        newton_solver->convergence_strategy()->set_tolerance( newton_tol );

        auto error_monitor = std::make_shared<error_monitor_t>( vspace, exact_solution, &log );

        for ( int step = 0; step < max_time_steps; step++ )
        {
            std::cout << std::endl;
            std::cout << "Time iteration #" << ( step + 1 ) << " has started" << std::endl;
            std::cout << std::endl;

            // Create step-specific output directory
            std::string step_dir = output_dir + "/step_" + std::to_string( step + 1 );
            std::filesystem::create_directories( step_dir );

            // Create convergence monitor for this time step
            auto conv_monitor = std::make_shared<newton_conv_monitor_jacobi_t>(
                vspace, cahn_hilliard_op, get_monitor, step_dir, solver_type, preconditioner_type, grid_size, &log );

            // Write initial residual (iteration 0) before Newton iterations start
            conv_monitor->write_initial_residual( solution );

            // Solve and measure time
            double step_time;
            {
                Timer timer("Solve", false);
                newton_solver->solve( cahn_hilliard_op.get(), conv_monitor.get(), nullptr, solution );
                step_time = timer.stop_and_get_ms();
            }
            iteration_times.push_back( step_time );
            total_time_ms += step_time;

            // Compute norm F(x) - stationary residual (to check time convergence)
            vector_t F_x( range );
            cahn_hilliard_op_stationary->apply( solution, F_x );
            scalar F_x_norm = vspace->norm_l2( F_x );
            log.info_f( "||F_stationary(solution)||_2 = %le", static_cast<double>( F_x_norm ) );

            // Write to time convergence history file
            time_conv_file << ( step + 1 ) << " " << std::scientific << std::setprecision(15)
                          << static_cast<double>( F_x_norm ) << std::endl;
            time_conv_file.flush();

            // Check for early termination based on F(x) norm
            if ( F_x_norm < time_tol )
            {
                log.info_f( "Early termination: ||F(solution)||_2 = %le < %le (tolerance)",
                           static_cast<double>( F_x_norm ), static_cast<double>( time_tol ) );
                // Update previous step before breaking
                time_derivative->set_previous_state( solution );
                break;
            }

            // Compute norm of difference between solution and previous state
            vector_t previous_state = time_derivative->get_previous_state();
            vector_t diff_prev( range );
            vspace->assign_lin_comb( scalar( 1 ), solution, scalar( -1 ), previous_state, diff_prev );
            scalar diff_prev_norm = vspace->norm_l2( diff_prev );
            log.info_f( "||solution - previous_state||_2 = %le", static_cast<double>( diff_prev_norm ) );

            // Compute norm of difference between solution and exact solution
            vector_t diff_exact( range );
            vspace->assign_lin_comb( scalar( 1 ), solution, scalar( -1 ), exact_solution, diff_exact );
            scalar diff_exact_norm = vspace->norm_l2( diff_exact );
            log.info_f( "||solution - exact||_2 = %le", static_cast<double>( diff_exact_norm ) );

            // Update previous step before the next step
            time_derivative->set_previous_state( solution );

            // Save numerical solution at each step if requested
            if ( save_coords )
            {
                std::string numerical_file = output_dir + "/numerical_" + std::to_string( step + 1 ) + ".bin";
                tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
            }

            // Separate iterations with empty line
            if ( step < max_time_steps - 1 )
            {
                std::cout << std::endl;
            }
        }
    }
    else // gmres
    {
        using newton_iteration_gmres_t = nmfd::solvers::newton_iteration<vec_ops_t, cahn_hilliard_op_t, gmres_solver>;
        using newton_conv_monitor_gmres_t = tests::newton_convergence_monitor<vec_ops_t, log_t, cahn_hilliard_op_t, krylov_monitor_t, scalar>;
        using newton_solver_gmres_t = nmfd::solvers::nonlinear_solver<vec_ops_t, log_t, cahn_hilliard_op_t, newton_iteration_gmres_t, newton_conv_monitor_gmres_t>;

        gmres_solver::params params_gmres;
        params_gmres.monitor.rel_tol                      = tolerance;
        params_gmres.monitor.max_iters_num                = max_iterations;
        params_gmres.monitor.save_convergence_history     = true;
        params_gmres.do_restart_on_false_ritz_convergence = true;
        params_gmres.basis_size                           = gmres_basis;
        params_gmres.preconditioner_side                  = 'L';
        params_gmres.reorthogonalization                  = true;
        auto gmres_lin_solver = std::make_shared<gmres_solver>( cahn_hilliard_jacobi_op, vspace, &log, params_gmres, precond );

        auto newton_iteration = std::make_shared<newton_iteration_gmres_t>( vspace, gmres_lin_solver );

        auto get_monitor = [gmres_lin_solver]() -> const krylov_monitor_t* {
            return &(gmres_lin_solver->monitor());
        };

        auto newton_solver = std::make_shared<newton_solver_gmres_t>( vspace, &log, newton_iteration );
        newton_solver->convergence_strategy()->set_tolerance( newton_tol );

        auto error_monitor = std::make_shared<error_monitor_t>( vspace, exact_solution, &log );

        for ( int step = 0; step < max_time_steps; step++ )
        {
            std::cout << std::endl;
            std::cout << "Time iteration #" << ( step + 1 ) << " has started" << std::endl;
            std::cout << std::endl;

            // Create step-specific output directory
            std::string step_dir = output_dir + "/step_" + std::to_string( step + 1 );
            std::filesystem::create_directories( step_dir );

            // Create convergence monitor for this time step
            auto conv_monitor = std::make_shared<newton_conv_monitor_gmres_t>(
                vspace, cahn_hilliard_op, get_monitor, step_dir, solver_type, preconditioner_type, grid_size, &log );

            // Write initial residual (iteration 0) before Newton iterations start
            conv_monitor->write_initial_residual( solution );

            // Solve and measure time
            double step_time;
            {
                Timer timer("Solve", false);
                newton_solver->solve( cahn_hilliard_op.get(), conv_monitor.get(), nullptr, solution );
                step_time = timer.stop_and_get_ms();
            }
            iteration_times.push_back( step_time );
            total_time_ms += step_time;

            // Compute norm F(x) - stationary residual (to check time convergence)
            vector_t F_x( range );
            cahn_hilliard_op_stationary->apply( solution, F_x );
            scalar F_x_norm = vspace->norm_l2( F_x );
            log.info_f( "||F_stationary(solution)||_2 = %le", static_cast<double>( F_x_norm ) );

            // Write to time convergence history file
            time_conv_file << ( step + 1 ) << " " << std::scientific << std::setprecision(15)
                          << static_cast<double>( F_x_norm ) << std::endl;
            time_conv_file.flush();

            // Check for early termination based on F(x) norm
            if ( F_x_norm < time_tol )
            {
                log.info_f( "Early termination: ||F(solution)||_2 = %le < %le (tolerance)",
                           static_cast<double>( F_x_norm ), static_cast<double>( time_tol ) );
                // Update previous step before breaking
                time_derivative->set_previous_state( solution );
                break;
            }

            // Compute norm of difference between solution and previous state
            vector_t previous_state = time_derivative->get_previous_state();
            vector_t diff_prev( range );
            vspace->assign_lin_comb( scalar( 1 ), solution, scalar( -1 ), previous_state, diff_prev );
            scalar diff_prev_norm = vspace->norm_l2( diff_prev );
            log.info_f( "||solution - previous_state||_2 = %le", static_cast<double>( diff_prev_norm ) );

            // Compute norm of difference between solution and exact solution
            vector_t diff_exact( range );
            vspace->assign_lin_comb( scalar( 1 ), solution, scalar( -1 ), exact_solution, diff_exact );
            scalar diff_exact_norm = vspace->norm_l2( diff_exact );
            log.info_f( "||solution - exact||_2 = %le", static_cast<double>( diff_exact_norm ) );

            // Update previous step before the next step
            time_derivative->set_previous_state( solution );

            // Save numerical solution at each step if requested
            if ( save_coords )
            {
                std::string numerical_file = output_dir + "/numerical_" + std::to_string( step + 1 ) + ".bin";
                tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
            }

            // Separate iterations with empty line
            if ( step < max_time_steps - 1 )
            {
                std::cout << std::endl;
            }
        }
    }

    // Save exact solution once at the end if requested
    if ( save_coords )
    {
        std::string exact_file = output_dir + "/exact.bin";
        tests::save_solution_binary<vector_t, idx_nd_type>( exact_solution, exact_file, grid_size, tensor_dim );
    }

    // Final comparison with exact solution
    vector_t error( range );
    vspace->assign_lin_comb( scalar( 1 ), solution, scalar( -1 ), exact_solution, error );
    scalar error_norm = vspace->norm_l2( error );
    scalar exact_norm = vspace->norm_l2( exact_solution );

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  ||solution - exact||_2:          " << std::scientific << error_norm << std::endl;
    std::cout << "  Relative error:                  " << std::scientific << ( error_norm / exact_norm ) << std::endl;
    std::cout << "  Average iteration time:          " << std::fixed << std::setprecision( 2 )
              << ( total_time_ms / max_time_steps ) << " ms" << std::endl;
    std::cout << "  Total solve time:                " << std::fixed << std::setprecision( 2 ) << total_time_ms
              << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    if ( save_coords )
    {
        std::cout << std::endl;
        std::cout << "Saved solutions:" << std::endl;
        std::cout << "  Numerical: " << output_dir << "/numerical_*.bin" << std::endl;
        std::cout << "  Exact:     " << output_dir << "/exact.bin" << std::endl;
    }

    // Close time convergence history file
    time_conv_file.close();

    // Restore cout
    std::cout.rdbuf( old_cout_buf );
    log_file.close();

    return 0;
}
