#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <nmfd/operations/rect_vector_space.h>
#include <nmfd/preconditioners/mg.h>
#include <nmfd/preconditioners/dummy.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <scfd/backend/backend.h>
#include <scfd/utils/log.h>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <cmath>

#include "cahn_hilliard_op.h"
#include "jacobi_op.h"
#include "jacobi_pre.h"

#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>

#include "biharmonic_problem.h"
#include "coarsening.h"
#include "include/boundary.h"
#include "kernels/phobic_energy.h"
#include "prolongator.h"
#include "restrictor.h"
#include "solution_io.h"
#include "perlin_noise.h"


using backend = scfd::backend::current;

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

// Monitors
using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>;
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

// Problem
using phobic_energy     = tests::double_well_potential<scalar>;
using rhs_t             = tests::zero_rhs<scalar, tensor_t>;
using time_derivative_t = tests::time_derivative<vec_ops_t, tensor_t>;

// MG
using prolongator_t = tests::prolongator<vec_ops_t, log_t>;
using restrictor_t  = tests::restrictor<vec_ops_t, log_t>;
using jacobi_op_t   = tests::jacobi_op<vec_ops_t, log_t, phobic_energy, time_derivative_t>;
using ident_op_t    = nmfd::preconditioners::dummy<vec_ops_t, jacobi_op_t>;
using smoother_t    = tests::jacobi_pre<vec_ops_t, log_t, phobic_energy, time_derivative_t>;
using coarsening_t  = tests::coarsening<jacobi_op_t, log_t>;

using precond_interface = nmfd::preconditioners::preconditioner_interface<vec_ops_t, jacobi_op_t>;

using mg_t =
    nmfd::preconditioners::mg<jacobi_op_t, restrictor_t, prolongator_t, smoother_t, ident_op_t, coarsening_t, log_t>;
using mg_params_t = mg_t::params_hierarchy;
using mg_utils_t  = mg_t::utils_hierarchy;

using gmres_solver = nmfd::solvers::gmres<vec_ops_t, krylov_monitor_t, log_t, jacobi_op_t, precond_interface>;

// Newton
using cahn_hilliard_op_t = tests::cahn_hilliard_op<vec_ops_t, jacobi_op_t, log_t, phobic_energy, rhs_t, time_derivative_t>;
using newton_iteration_t = nmfd::solvers::newton_iteration<vec_ops_t, cahn_hilliard_op_t, gmres_solver>;
using newton_solver_t =
    nmfd::solvers::nonlinear_solver<vec_ops_t, log_t, cahn_hilliard_op_t, newton_iteration_t>;

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
constexpr scalar DEFAULT_GMRES_TOL      = std::is_same<float, scalar>::value ? 5e-6f : 1e-10;
constexpr int    DEFAULT_MAX_TIME_STEPS = 1;
constexpr scalar DEFAULT_DT_INF         = 1.0;
constexpr scalar DEFAULT_NOISE_AMPLITUDE = 1.0;
constexpr scalar DEFAULT_NOISE_FREQUENCY = 10.0;

/**************************************/

int main( int argc, char const *argv[] )
{
    // Parse CLI arguments
    bool        save_coords = false;
    std::string prefix      = "run";
    int         grid_size   = 32;

    // Solver parameters (initialized to defaults)
    int         max_iterations = DEFAULT_MAX_ITERATIONS;
    int         gmres_basis    = DEFAULT_GMRES_BASIS;
    int         mg_sweeps_pre  = DEFAULT_MG_SWEEPS_PRE;
    int         mg_sweeps_post = DEFAULT_MG_SWEEPS_POST;
    scalar      newton_tol     = DEFAULT_NEWTON_TOL;
    scalar      gmres_tol      = DEFAULT_GMRES_TOL;
    int         max_time_steps = DEFAULT_MAX_TIME_STEPS;
    scalar      dt_inf         = DEFAULT_DT_INF;
    unsigned int noise_seed    = 0; // Default seed for Perlin noise
    scalar      D              = scalar( 1.0 ); // Diffusion coefficient
    scalar      gamma          = scalar( 1.0 ); // Squared length of transition regions
    scalar      noise_amplitude = DEFAULT_NOISE_AMPLITUDE; // Amplitude of Perlin noise
    scalar      noise_frequency = DEFAULT_NOISE_FREQUENCY; // Frequency scale factor for Perlin noise

    if ( argc < 2 )
    {
        std::cout << "USAGE: " << argv[0] << " <grid_size> [prefix] [options...]" << std::endl;
        std::cout << std::endl;
        std::cout << "Arguments:" << std::endl;
        std::cout << "    grid_size              Number of grid points per dimension (e.g., 32)" << std::endl;
        std::cout << "    prefix                 Output prefix (default: 'run')" << std::endl;
        std::cout << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "    --save-coords          Save numerical solutions to binary files" << std::endl;
        std::cout << "    --max-iterations N     Maximum GMRES iterations (default: " << DEFAULT_MAX_ITERATIONS << ")"
                  << std::endl;
        std::cout << "    --gmres-basis N        GMRES basis size (default: " << DEFAULT_GMRES_BASIS << ")"
                  << std::endl;
        std::cout << "    --mg-sweeps-pre N      Multigrid pre-sweeps (default: " << DEFAULT_MG_SWEEPS_PRE << ")"
                  << std::endl;
        std::cout << "    --mg-sweeps-post N     Multigrid post-sweeps (default: " << DEFAULT_MG_SWEEPS_POST << ")"
                  << std::endl;
        std::cout << "    --newton-tol T         Newton solver tolerance (default: " << std::scientific
                  << DEFAULT_NEWTON_TOL << std::defaultfloat << ")" << std::endl;
        std::cout << "    --gmres-tol T          GMRES solver tolerance (default: " << std::scientific
                  << DEFAULT_GMRES_TOL << std::defaultfloat << ")" << std::endl;
        std::cout << "    --max-time-steps N     Maximum number of time steps (default: " << DEFAULT_MAX_TIME_STEPS << ")" << std::endl;
        std::cout << "    --dt-inf T             dt_inf parameter (1/dt, default: " << DEFAULT_DT_INF << ")" << std::endl;
        std::cout << "    --noise-seed N         Random seed for Perlin noise (default: 0)" << std::endl;
        std::cout << "    --noise-amplitude T    Amplitude of Perlin noise (default: " << DEFAULT_NOISE_AMPLITUDE << ")" << std::endl;
        std::cout << "    --noise-frequency T    Frequency scale factor for Perlin noise (default: " << DEFAULT_NOISE_FREQUENCY << ")" << std::endl;
        std::cout << "    --D T                  Diffusion coefficient (default: 1.0)" << std::endl;
        std::cout << "    --gamma T              Squared length of transition regions (default: 1.0)" << std::endl;
        return 1;
    }

    grid_size = std::stoi( argv[1] );

    for ( int i = 2; i < argc; ++i )
    {
        std::string arg = argv[i];
        if ( arg == "--save-coords" )
        {
            save_coords = true;
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
        else if ( arg == "--gmres-tol" && i + 1 < argc )
        {
            gmres_tol = std::stod( argv[++i] );
        }
        else if ( arg == "--max-time-steps" && i + 1 < argc )
        {
            max_time_steps = std::stoi( argv[++i] );
        }
        else if ( arg == "--dt-inf" && i + 1 < argc )
        {
            dt_inf = std::stod( argv[++i] );
        }
        else if ( arg == "--noise-seed" && i + 1 < argc )
        {
            noise_seed = static_cast<unsigned int>(std::stoul( argv[++i] ));
        }
        else if ( arg == "--noise-amplitude" && i + 1 < argc )
        {
            noise_amplitude = std::stod( argv[++i] );
        }
        else if ( arg == "--noise-frequency" && i + 1 < argc )
        {
            noise_frequency = std::stod( argv[++i] );
        }
        else if ( arg == "--D" && i + 1 < argc )
        {
            D = std::stod( argv[++i] );
        }
        else if ( arg == "--gamma" && i + 1 < argc )
        {
            gamma = std::stod( argv[++i] );
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
    log.set_verbosity(1);

    // Write configuration header to log
    std::cout << "========================================" << std::endl;
    std::cout << "Cahn-Hilliard Solver Configuration" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Problem Settings:" << std::endl;
    std::cout << "  Grid size:     " << grid_size << " x " << grid_size << " x " << grid_size << std::endl;
    std::cout << "  Tensor dim:    " << tensor_dim << std::endl;
    std::cout << "  Scalar type:   " << scalar_label << std::endl;
    std::cout << "  DOFs:          " << static_cast<long long>( grid_size ) * grid_size * grid_size * tensor_dim
              << std::endl;
    std::cout << "  Initial cond:  3D Perlin noise (amplitude " << noise_amplitude << ", frequency " << noise_frequency << ", seed " << noise_seed << ")" << std::endl;
    std::cout << "  RHS:           Zero" << std::endl;
    std::cout << "  D:             " << std::scientific << D << std::defaultfloat << std::endl;
    std::cout << "  gamma:         " << std::scientific << gamma << std::defaultfloat << std::endl;
    std::cout << std::endl;
    std::cout << "Newton Solver:" << std::endl;
    std::cout << "  Tolerance:     " << std::scientific << newton_tol << std::endl;
    std::cout << std::endl;
    std::cout << "GMRES Solver:" << std::endl;
    std::cout << "  Tolerance:     " << std::scientific << gmres_tol << std::endl;
    std::cout << "  Max iters:     " << max_iterations << std::endl;
    std::cout << "  Basis size:    " << gmres_basis << std::endl;
    std::cout << "  Precond side:  L" << std::endl;
    std::cout << "  Reorthogon.:   true" << std::endl;
    std::cout << std::endl;
    std::cout << "Multigrid Preconditioner:" << std::endl;
    std::cout << "  Pre-sweeps:    " << mg_sweeps_pre << std::endl;
    std::cout << "  Post-sweeps:   " << mg_sweeps_post << std::endl;
    std::cout << "  Direct coarse: false" << std::endl;
    std::cout << std::endl;
    std::cout << "Time Integration:" << std::endl;
    std::cout << "  Max time steps: " << max_time_steps << std::endl;
    std::cout << "  dt_inf:         " << dt_inf << std::endl;
    std::cout << std::endl;
    std::cout << "Output:" << std::endl;
    std::cout << "  Directory:     " << output_dir << std::endl;
    std::cout << "  Save coords:   " << ( save_coords ? "yes" : "no" ) << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    auto range = idx_nd_type::make_ones() * grid_size;
    auto step  = grid_step_type::make_ones() / scalar( grid_size );

    int left_bc[3][2]  = { { -1, -1 }, { -1, -1 }, { -1, -1 } }; // left:  [x,y,z][psi=Neumann, phi=nonlinear]
    int right_bc[3][2] = { { -1, -1 }, { -1, -1 }, { -1, -1 } };  // right: [x,y,z][psi=Neumann, phi=nonlinear]

    auto cond  = tests::boundary_cond<vec_ops_t>(
        range, left_bc, right_bc
    );
    // Boundary condition values:
    //   -1 = dirichlet (value = 0 at boundary)
    //   +1 = neumann (derivative = 0 at boundary)
    //    0 = periodic (left boundary uses value from N-1, right boundary uses value from 0)

    vector_t solution( range );
    auto     vspace = std::make_shared<vec_ops_t>( range );

    // Initialize solution with 3D Perlin noise
    {
        vector_view_t solution_view( solution, false );

        int d = range[0] / 8;
        for ( int i = 0; i < range[0]; i++ )
        {
            for ( int j = 0; j < range[1]; j++ )
            {
                for ( int k = 0; k < range[2]; k++ )
                {
                    scalar x = step[0] * ( 0.5 + i );
                    scalar y = step[1] * ( 0.5 + j );
                    scalar z = step[2] * ( 0.5 + k );

                    /*
                    // Generate Perlin noise value for psi (component 0)
                    scalar noise_val_psi = perlin_noise_3d<scalar>( x * noise_frequency, y * noise_frequency, z * noise_frequency, noise_seed );
                    noise_val_psi *= noise_amplitude;
                    solution_view( i, j, k, 0 ) = noise_val_psi;

                    // Generate Perlin noise value for phi (component 1) using different seed
                    scalar noise_val_phi = perlin_noise_3d<scalar>( x * noise_frequency, y * noise_frequency, z * noise_frequency, noise_seed + 1000 );
                    noise_val_phi *= noise_amplitude;
                    solution_view( i, j, k, 1 ) = noise_val_phi;
                    */

                    solution_view( i, j, k, 0 ) = 0.0;

                    if ( i < d && i > range[0] - d && j < d && j > range[1] - d && k < d && k > range[2] - d )
                    {
                        solution_view( i, j, k, 1 ) = 1.0;
                    }
                    else
                    {
                        solution_view( i, j, k, 1 ) = -1.0;
                    }
                }
            }
        }

        solution_view.release();
    }

    auto time_derivative = std::make_shared<time_derivative_t>( range );
    time_derivative->set_dt_inf( dt_inf );
    time_derivative->set_previous_state( solution );

    auto cahn_hilliard_jacobi_op = std::make_shared<jacobi_op_t>( range, step, cond, time_derivative );
    auto cahn_hilliard_op        = std::make_shared<cahn_hilliard_op_t>( range, step, cond, cahn_hilliard_jacobi_op, time_derivative );

    // Set D and gamma parameters
    cahn_hilliard_op->set_D( D );
    cahn_hilliard_op->set_gamma( gamma );

    std::shared_ptr<precond_interface> precond;

    mg_utils_t  mg_utils;
    mg_params_t mg_params;

    mg_utils.log              = &log;
    mg_params.direct_coarse   = false;
    mg_params.num_sweeps_pre  = mg_sweeps_pre;
    mg_params.num_sweeps_post = mg_sweeps_post;

    precond = std::make_shared<mg_t>( mg_utils, mg_params );

    gmres_solver::params params_gmres;
    params_gmres.monitor.rel_tol                      = gmres_tol;
    params_gmres.monitor.max_iters_num                = max_iterations;
    params_gmres.monitor.save_convergence_history     = true;
    params_gmres.do_restart_on_false_ritz_convergence = true;
    params_gmres.basis_size                           = gmres_basis;
    params_gmres.preconditioner_side                  = 'L';
    params_gmres.reorthogonalization                  = true;
    auto lin_solver = std::make_shared<gmres_solver>( cahn_hilliard_jacobi_op, vspace, &log, params_gmres, precond );

    auto newton_iteration = std::make_shared<newton_iteration_t>( vspace, lin_solver );

    auto newton_solver = std::make_shared<newton_solver_t>( vspace, &log, newton_iteration );
    newton_solver->convergence_strategy()->set_tolerance( newton_tol );

    // Save initial approximation (index 0) if requested
    if ( save_coords )
    {
        std::string numerical_file = output_dir + "/numerical_0.bin";
        tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
    }

    // Solve and measure time for each iteration
    std::vector<double> iteration_times;
    double              total_time_ms = 0.0;

    for ( int step = 0; step < max_time_steps; step++ )
    {
        std::cout << std::endl;
        std::cout << "Time iteration #" << ( step + 1 ) << " has started" << std::endl;
        std::cout << std::endl;

        auto start = std::chrono::steady_clock::now();
        newton_solver->solve( cahn_hilliard_op.get(), nullptr, nullptr, solution );
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> solve_time_ms = ( end - start );
        double                                      iteration_time = solve_time_ms.count();
        iteration_times.push_back( iteration_time );
        total_time_ms += iteration_time;

        // Compute norm of difference between solution and previous state
        vector_t previous_state = time_derivative->get_previous_state();
        vector_t diff_prev( range );
        vspace->assign_lin_comb( scalar( 1 ), solution, scalar( -1 ), previous_state, diff_prev );
        scalar diff_prev_norm = vspace->norm_l2( diff_prev );
        log.info_f( "||solution - previous_state||_2 = %le", static_cast<double>( diff_prev_norm ) );

        // Compute solution norm
        scalar solution_norm = vspace->norm_l2( solution );
        log.info_f( "||solution||_2 = %le", static_cast<double>( solution_norm ) );

        // Update previous step before the next step
        time_derivative->set_previous_state( solution );

        // Save numerical solution at each step if requested
        // Save as numerical_1.bin, numerical_2.bin, ... (index 0 was initial state)
        if ( save_coords )
        {
            std::string numerical_file = output_dir + "/numerical_" + std::to_string( step + 1 ) + ".bin";
            tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
        }

        // Initialize solution for next iteration with current solution (as initial guess)
        // Don't reset to zero - use the current solution as initial guess for the next time step
        // The solution vector will be updated by the Newton solver in the next iteration

        // Separate iterations with empty line
        if ( step < max_time_steps - 1 )
        {
            std::cout << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << "========================================" << std::endl;
    scalar final_solution_norm = vspace->norm_l2( solution );
    std::cout << "  ||solution||_2:                    " << std::scientific << final_solution_norm << std::endl;
    std::cout << "  Average iteration time:            " << std::fixed << std::setprecision( 2 )
              << ( total_time_ms / max_time_steps ) << " ms" << std::endl;
    std::cout << "  Total solve time:                  " << std::fixed << std::setprecision( 2 ) << total_time_ms
              << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    if ( save_coords )
    {
        std::cout << std::endl;
        std::cout << "Saved solutions:" << std::endl;
        std::cout << "  Numerical: " << output_dir << "/numerical_*.bin" << std::endl;
    }

    // Restore cout
    std::cout.rdbuf( old_cout_buf );
    log_file.close();

    return 0;
}
