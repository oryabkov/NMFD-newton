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

#include "cahn_hilliard_op.h"
#include "jacobi_op.h"
#include "jacobi_pre.h"

#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>

#include "cahn_hilliard_problem.h"
#include "coarsening.h"
#include "error_monitor.h"
#include "identity_op.h"
#include "include/boundary.h"
#include "kernels/phobic_energy.h"
#include "prolongator.h"
#include "restrictor.h"
#include "solution_io.h"


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

// Monitors
using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>;
using error_monitor_t   = tests::error_monitor<vec_ops_t, log_t>;
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

// Problem
using phobic_energy = tests::double_well_potential<scalar>;
using rhs_t         = tests::trig_rhs<scalar, tensor_t, 3, 3, 1>;

// MG
using prolongator_t = tests::prolongator<vec_ops_t, log_t>;
using restrictor_t  = tests::restrictor<vec_ops_t, log_t>;
using jacobi_op_t   = tests::jacobi_op<vec_ops_t, log_t, phobic_energy>;
using ident_op_t    = tests::identity_op<jacobi_op_t, vec_ops_t, log_t>;
using smoother_t    = tests::jacobi_pre<vec_ops_t, log_t, phobic_energy>;
using coarsening_t  = tests::coarsening<jacobi_op_t, log_t>;

using precond_interface = nmfd::preconditioners::preconditioner_interface<vec_ops_t, jacobi_op_t>;

using mg_t =
    nmfd::preconditioners::mg<jacobi_op_t, restrictor_t, prolongator_t, smoother_t, ident_op_t, coarsening_t, log_t>;
using mg_params_t = mg_t::params_hierarchy;
using mg_utils_t  = mg_t::utils_hierarchy;

using gmres_solver = nmfd::solvers::gmres<vec_ops_t, krylov_monitor_t, log_t, jacobi_op_t, precond_interface>;

// Newton
using cahn_hilliard_op_t = tests::cahn_hilliard_op<vec_ops_t, jacobi_op_t, log_t, phobic_energy, rhs_t>;
using newton_iteration_t = nmfd::solvers::newton_iteration<vec_ops_t, cahn_hilliard_op_t, gmres_solver>;
using newton_solver_t =
    nmfd::solvers::nonlinear_solver<vec_ops_t, log_t, cahn_hilliard_op_t, newton_iteration_t, error_monitor_t>;

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
constexpr scalar DEFAULT_NEWTON_TOL     = std::is_same_v<float, scalar> ? 5e-6f : 1e-10;
constexpr scalar DEFAULT_GMRES_TOL      = std::is_same_v<float, scalar> ? 5e-6f : 1e-10;

/**************************************/

int main( int argc, char const *argv[] )
{
    // Parse CLI arguments
    bool        save_coords = false;
    std::string prefix      = "run";
    int         grid_size   = 32;

    // Solver parameters (initialized to defaults)
    int    max_iterations = DEFAULT_MAX_ITERATIONS;
    int    gmres_basis    = DEFAULT_GMRES_BASIS;
    int    mg_sweeps_pre  = DEFAULT_MG_SWEEPS_PRE;
    int    mg_sweeps_post = DEFAULT_MG_SWEEPS_POST;
    scalar newton_tol     = DEFAULT_NEWTON_TOL;
    scalar gmres_tol      = DEFAULT_GMRES_TOL;

    if ( argc < 2 )
    {
        std::cout << "USAGE: " << argv[0] << " <grid_size> [prefix] [options...]" << std::endl;
        std::cout << std::endl;
        std::cout << "Arguments:" << std::endl;
        std::cout << "    grid_size              Number of grid points per dimension (e.g., 32)" << std::endl;
        std::cout << "    prefix                 Output prefix (default: 'run')" << std::endl;
        std::cout << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "    --save-coords          Save numerical and exact solutions to binary files" << std::endl;
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
    std::cout << "Output:" << std::endl;
    std::cout << "  Directory:     " << output_dir << std::endl;
    std::cout << "  Save coords:   " << ( save_coords ? "yes" : "no" ) << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    auto range = idx_nd_type::make_ones() * grid_size;
    auto step  = grid_step_type::make_ones() / scalar( grid_size );
    auto cond  = boundary_cond<dim>{
        { -1, -1, -1 }, // left
        { -1, -1, -1 }  // right
    };                  // -1 = dirichlet, +1 = neumann

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

    auto cahn_hilliard_jacobi_op = std::make_shared<jacobi_op_t>( range, step, cond );
    auto cahn_hilliard_op        = std::make_shared<cahn_hilliard_op_t>( range, step, cond, cahn_hilliard_jacobi_op );

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

    auto error_monitor = std::make_shared<error_monitor_t>( vspace, exact_solution, &log );

    auto newton_solver = std::make_shared<newton_solver_t>( vspace, &log, newton_iteration );
    newton_solver->convergence_strategy()->set_tolerance( newton_tol );

    // Verify that F(exact_solution) is close to zero
    vector_t F_exact( range );
    cahn_hilliard_op->apply( exact_solution, F_exact );
    scalar F_exact_norm = vspace->norm_l2( F_exact );
    log.info_f( "Verification: ||F(exact_solution)||_2 = %le", static_cast<double>( F_exact_norm ) );

    // Solve and measure time
    std::cout << std::endl << "Starting solve..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    newton_solver->solve( cahn_hilliard_op.get(), error_monitor.get(), nullptr, solution );
    auto                                      end           = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> solve_time_ms = ( end - start );

    // Final comparison
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
    std::cout << "  Total solve time:                " << std::fixed << std::setprecision( 2 ) << solve_time_ms.count()
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

    return 0;
}
