#include "biharmonic_problem.h"
#include "convergence_history_io.h"
#include "jacobi_op.h"
#include "jacobi_pre.h"
#include "coarsening.h"
#include "identity_op.h"
#include "include/boundary.h"
#include "kernels/phobic_energy.h"
#include "prolongator.h"
#include "restrictor.h"
#include "time_derivative.h"
#include "solution_io.h"
#include "timers.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <nmfd/operations/rect_vector_space.h>
#include <nmfd/preconditioners/mg.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/jacobi.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <scfd/backend/backend.h>
#include <scfd/utils/log.h>
#include <algorithm>
#include <sstream>
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

using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>;
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

using phobic_energy_t = tests::zero_potential<scalar>;
using zero_rhs_t      = tests::zero_rhs<scalar, tensor_t>;
using rhs_t           = tests::trig_rhs<scalar, tensor_t>;
using time_derivative_t = tests::time_derivative<vec_ops_t, tensor_t>;

using prolongator_t = tests::prolongator<vec_ops_t, log_t>;
using restrictor_t  = tests::restrictor<vec_ops_t, log_t>;
using lin_op_t      = tests::jacobi_op<vec_ops_t, log_t, phobic_energy_t, time_derivative_t>;
using ident_op_t    = tests::identity_op<lin_op_t, vec_ops_t, log_t>;
using smoother_t    = tests::jacobi_pre<vec_ops_t, log_t, phobic_energy_t, time_derivative_t>;
using coarsening_t  = tests::coarsening<lin_op_t, log_t>;

using precond_interface = nmfd::preconditioners::preconditioner_interface<vec_ops_t, lin_op_t>;

using mg_t =
    nmfd::preconditioners::mg<lin_op_t, restrictor_t, prolongator_t, smoother_t, ident_op_t, coarsening_t, log_t>;
using mg_params_t = mg_t::params_hierarchy;
using mg_utils_t  = mg_t::utils_hierarchy;

using jacobi_solver = nmfd::solvers::jacobi<vec_ops_t, lin_op_t, precond_interface, default_monitor_t, log_t>;
using gmres_solver  = nmfd::solvers::gmres<vec_ops_t, krylov_monitor_t, log_t, lin_op_t, precond_interface>;

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
constexpr scalar DEFAULT_TOLERANCE      = std::is_same_v<float, scalar> ? 5e-6f : 1e-10;

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
    scalar tolerance      = DEFAULT_TOLERANCE;

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
        std::cout << "    --tolerance T        Solver tolerance (default: " << std::scientific << DEFAULT_TOLERANCE
                  << std::defaultfloat << ")" << std::endl;
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
        else if ( arg == "--tolerance" && i + 1 < argc )
        {
            tolerance = std::stod( argv[++i] );
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
    const std::string scalar_label = std::is_same_v<float, scalar> ? "float" : "double";

    log_t log;
    // Set log verbosity: 0 suppresses INFO messages, 1 allows them
    log.set_verbosity( verbose ? 1 : 0 );

    // Write configuration header to log
    std::cout << "========================================" << std::endl;
    std::cout << "Biharmonic Solver Configuration" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Problem Settings:" << std::endl;
    std::cout << "  Grid size:     " << grid_size << " x " << grid_size << " x " << grid_size << std::endl;
    std::cout << "  Tensor dim:    " << tensor_dim << std::endl;
    std::cout << "  Scalar type:   " << scalar_label << std::endl;
    std::cout << "  DOFs:          " << static_cast<long long>( grid_size ) * grid_size * grid_size * tensor_dim
              << std::endl;
    std::cout << std::endl;
    std::cout << "Solver:" << std::endl;
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
    std::cout << "Output:" << std::endl;
    std::cout << "  Directory:     " << output_dir << std::endl;
    std::cout << "  Save coords:   " << ( save_coords ? "yes" : "no" ) << std::endl;
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

    vector_t solution( range ), rhs( range ), exact_solution( range );
    rhs_t    rhs_function;

    auto vspace = std::make_shared<vec_ops_t>( range );
    {
        vspace->assign_scalar( 0.0, solution ); // Initialize solution to zero
        vector_view_t rhs_view( rhs, false ), exact_view( exact_solution, false );

        for ( int i = 0; i < range[0]; i++ )
        {
            for ( int j = 0; j < range[1]; j++ )
            {
                for ( int k = 0; k < range[2]; k++ )
                {
                    scalar x = step[0] * ( 0.5 + i );
                    scalar y = step[1] * ( 0.5 + j );
                    scalar z = step[2] * ( 0.5 + k );

                    auto rhs_val   = rhs_function.get_exact_solution( x, y, z );
                    auto exact_val = rhs_function( x, y, z );
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
        mg_params.num_sweeps_pre  = mg_sweeps_pre;
        mg_params.num_sweeps_post = mg_sweeps_post;

        precond = std::make_shared<mg_t>( mg_utils, mg_params );
    }

    // Solve the system and measure execution time
    double solve_time_ms;
    bool   converged;

    if ( solver_type == "jacobi" )
    {
        jacobi_solver::params solver_params;
        solver_params.monitor.rel_tol                  = tolerance;
        solver_params.monitor.max_iters_num            = max_iterations;
        solver_params.monitor.save_convergence_history = true;
        jacobi_solver solver{ l_op, vspace, &log, solver_params, precond };

        {
            Timer timer("Solve", false); // Don't print automatically, we'll print in Results section
            converged     = solver.solve( rhs, solution );
            solve_time_ms = timer.stop_and_get_ms();
        }

        // Save times.dat and convergence history
        {
            std::chrono::duration<double, std::milli> solve_time_duration(solve_time_ms);
            save_times_dat<default_monitor_t, scalar>( solver.monitor(), solver_type, preconditioner_type,
                                                        grid_size, solve_time_duration, output_dir );
            save_convergence_history<default_monitor_t, scalar>( solver.monitor(), output_dir );
        }

    }
    else // gmres
    {
        gmres_solver::params params_gmres;
        params_gmres.monitor.rel_tol                      = tolerance;
        params_gmres.monitor.max_iters_num                = max_iterations;
        params_gmres.monitor.save_convergence_history     = true;
        params_gmres.do_restart_on_false_ritz_convergence = true;
        params_gmres.basis_size                           = gmres_basis;
        params_gmres.preconditioner_side                  = 'L';
        params_gmres.reorthogonalization                  = true;
        gmres_solver solver{ l_op, vspace, &log, params_gmres, precond };

        {
            Timer timer("Solve", false); // Don't print automatically, we'll print in Results section
            converged     = solver.solve( rhs, solution );
            solve_time_ms = timer.stop_and_get_ms();
        }

        // Save times.dat and convergence history
        {
            std::chrono::duration<double, std::milli> solve_time_duration(solve_time_ms);
            save_times_dat<krylov_monitor_t, scalar>( solver.monitor(), solver_type, preconditioner_type,
                                                        grid_size, solve_time_duration, output_dir );
            save_convergence_history<krylov_monitor_t, scalar>( solver.monitor(), output_dir );
        }
    }

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

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Converged:                  " << ( converged ? "yes" : "no" ) << std::endl;
    std::cout << "  ||solution - exact||_2:     " << std::scientific << error_norm << std::endl;
    std::cout << "  Relative error:             " << std::scientific << ( error_norm / exact_norm ) << std::endl;
    std::cout << "  Total solve time:           " << std::fixed << std::setprecision( 2 ) << solve_time_ms
              << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    // Save solutions if requested
    if ( save_coords )
    {
        std::string numerical_file = output_dir + "/numerical.bin";
        std::string exact_file     = output_dir + "/exact.bin";

        tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
        tests::save_solution_binary<vector_t, idx_nd_type>( exact_solution, exact_file, grid_size, tensor_dim );

        std::cout << std::endl;
        std::cout << "Saved solutions:" << std::endl;
        std::cout << "  Numerical: " << numerical_file << std::endl;
        std::cout << "  Exact:     " << exact_file << std::endl;
    }

    // Restore cout
    std::cout.rdbuf( old_cout_buf );
    log_file.close();

    // return converged ? 0 : 1;
	return 0;
}
