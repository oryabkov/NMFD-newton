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

#include "cahn_hilliard_op.h"
#include "jacobi_op.h"
#include "jacobi_pre.h"

#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/jacobi.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>

// #include "cahn_hilliard_problem.h"
#include "biharmonic_problem.h"
#include "coarsening.h"
#include "convergence_history_io.h"
#include "include/boundary.h"
#include "kernels/phobic_energy.h"
#include "newton_convergence_monitor.h"
#include "prolongator.h"
#include "restrictor.h"
#include "solution_io.h"
#include "timers.h"
#include "perlin_noise.h"
#include "scheduler.h"


using backend = scfd::backend::current;

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

using vec_ops_t        = nmfd::rect_vector_space<scalar, /*dim=*/dim, /*tensor_dim=*/tensor_dim, backend>;
using vector_t         = typename vec_ops_t::vector_type;
using tensor_t         = scfd::static_vec::vec<scalar, tensor_dim>;
using vector_view_t    = typename vector_t::view_type;
// Monitors
using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>;
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

// Problem
using phobic_energy = tests::double_well_potential<scalar>;
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

using jacobi_solver = nmfd::solvers::jacobi<vec_ops_t, jacobi_op_t, precond_interface, default_monitor_t, log_t>;
using gmres_solver  = nmfd::solvers::gmres<vec_ops_t, krylov_monitor_t, log_t, jacobi_op_t, precond_interface>;

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

static tensor_t compute_component_sums( const vector_t &v, const idx_nd_type &range )
{
    tensor_t sums;
    for ( int c = 0; c < tensor_dim; ++c )
        sums[c] = scalar( 0 );
    // IMPORTANT: this view may allocate a host buffer for device memory.
    // We must sync FROM the array before reading, and we must NOT sync back on release.
    vector_view_t view( const_cast<vector_t &>( v ), /*sync_from_array*/ true );
    for ( int i = 0; i < range[0]; ++i )
    {
        for ( int j = 0; j < range[1]; ++j )
        {
            for ( int k = 0; k < range[2]; ++k )
            {
                for ( int c = 0; c < tensor_dim; ++c )
                {
                    sums[c] += view( i, j, k, c );
                }
            }
        }
    }
    view.release( /*sync_to_array*/ false );
    return sums;
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
constexpr scalar DEFAULT_D              = 1.0;
constexpr scalar DEFAULT_GAMMA          = 1e-4;
constexpr scalar DEFAULT_COS_THETA      = 0.5;
constexpr scalar DEFAULT_DT_INF         = 1.0;
constexpr int    DEFAULT_MAX_TIME_STEPS = 1;
constexpr int    DEFAULT_MAX_RETRIES    = 10;
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
    scalar D              = DEFAULT_D;
    scalar gamma          = DEFAULT_GAMMA;
    scalar cos_theta      = DEFAULT_COS_THETA;
    scalar dt_inf         = DEFAULT_DT_INF;
    int    max_time_steps = DEFAULT_MAX_TIME_STEPS;
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
        std::cout << "    --D T                Diffusion coefficient (default: " << DEFAULT_D << ")" << std::endl;
        std::cout << "    --gamma T            Squared length of transition regions (default: " << DEFAULT_GAMMA << ")" << std::endl;
        std::cout << "    --cos-theta T        Cos(equilibrium contact angle) for boundary condition (default: " << DEFAULT_COS_THETA << ")" << std::endl;
        std::cout << "    --dt-inf T           1/dt for implicit time stepping (default: " << DEFAULT_DT_INF << ")" << std::endl;
        std::cout << "    --max-time-steps N   Maximum number of time steps (default: " << DEFAULT_MAX_TIME_STEPS << ")" << std::endl;
        std::cout << "    --time-tol T         Time convergence tolerance (default: " << std::scientific << DEFAULT_TIME_TOL << std::defaultfloat << ")" << std::endl;
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
        else if ( arg == "--D" && i + 1 < argc )
        {
            D = std::stod( argv[++i] );
        }
        else if ( arg == "--gamma" && i + 1 < argc )
        {
            gamma = std::stod( argv[++i] );
        }
        else if ( arg == "--cos-theta" && i + 1 < argc )
        {
            cos_theta = std::stod( argv[++i] );
        }
        else if ( arg == "--dt-inf" && i + 1 < argc )
        {
            dt_inf = std::stod( argv[++i] );
        }
        else if ( arg == "--max-time-steps" && i + 1 < argc )
        {
            max_time_steps = std::stoi( argv[++i] );
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
    std::cout << "Cahn-Hilliard Solver Configuration" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Problem Settings:" << std::endl;
    std::cout << "  Grid size:     " << grid_size << " x " << grid_size << " x " << grid_size << std::endl;
    std::cout << "  Tensor dim:    " << tensor_dim << std::endl;
    std::cout << "  Scalar type:   " << scalar_label << std::endl;
    std::cout << "  DOFs:          " << static_cast<long long>( grid_size ) * grid_size * grid_size * tensor_dim
              << std::endl;
    std::cout << "  D:             " << std::scientific << D << std::defaultfloat << std::endl;
    std::cout << "  gamma:         " << std::scientific << gamma << std::defaultfloat << std::endl;
    std::cout << "  cos(theta):    " << std::scientific << cos_theta << std::defaultfloat << std::endl;
    std::cout << "  dt_inf (init): " << std::scientific << dt_inf << std::defaultfloat << std::endl;
    std::cout << "  max_time_steps:" << max_time_steps << std::endl;
    std::cout << "  time_tol:      " << std::scientific << time_tol << std::defaultfloat << std::endl;
    std::cout << std::endl;
    std::cout << "Newton Solver:" << std::endl;
    std::cout << "  Tolerance:     " << std::scientific << newton_tol << std::endl;
    std::cout << "  Max iters:     " << 10 << std::endl;
    std::cout << "  Adaptive dt:   yes (rollback + retry, dt_inf*=2 on fail, dt_inf/=2 after streak)" << std::endl;
    std::cout << "  Retry cap:     " << ( DEFAULT_MAX_RETRIES + 1 ) << " attempts per step" << std::endl;
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
    std::cout << "Boundary condition parameters:" << std::endl;
    std::cout << "  cos(theta):    " << std::scientific << cos_theta << std::defaultfloat << std::endl;
    std::cout << "  BC types:      (printed below after initialization)" << std::endl;
    std::cout << std::endl;
    std::cout << "Output:" << std::endl;
    std::cout << "  Directory:     " << output_dir << std::endl;
    std::cout << "  Save coords:   " << ( save_coords ? "yes" : "no" ) << std::endl;
    std::cout << "  Verbose:       " << ( verbose ? "yes" : "no" ) << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    auto range = idx_nd_type::make_ones() * grid_size;
    auto step  = grid_step_type::make_ones() / scalar( grid_size );

    // int left_bc[3][2]  = { { 0, 0 }, { 0, 0 }, { 0, 0 } }; // left:  [x,y,z][psi=Neumann, phi=nonlinear]
    // int right_bc[3][2] = { { 0, 0 }, { 0, 0 }, { 0, 0 } };  // right: [x,y,z][psi=Neumann, phi=nonlinear]

    // int left_bc[3][2]  = { { +1, +1 }, { +1, +1 }, { +1, +1 } }; // left:  [x,y,z][psi=Neumann, phi=nonlinear]
    // int right_bc[3][2] = { { +1, +1 }, { +1, +1 }, { +1, +1 } };  // right: [x,y,z][psi=Neumann, phi=nonlinear]

    int left_bc[3][2]  = { { 0, 0 }, { 0, 0 }, { +1, 2 } }; // left:  [x,y,z][psi=Neumann, phi=nonlinear]
    int right_bc[3][2] = { { 0, 0 }, { 0, 0 }, { +1, +1 } };  // right: [x,y,z][psi=Neumann, phi=nonlinear]

    auto cond  = tests::boundary_cond<vec_ops_t>(
        left_bc, right_bc, gamma, cos_theta
    );
    // Boundary condition values:
    //   -1 = dirichlet (value = 0 at boundary)
    //   +1 = neumann (derivative = 0 at boundary)
    //    0 = periodic (left boundary uses value from N-1, right boundary uses value from 0)
    //    2 = nonlinear contact-angle: alpha*nabla(C)*n = -sigma*cos(theta)*g'(C)

    std::cout << "Boundary conditions table (cell = (left,right)):" << std::endl;
    std::cout << std::setw( 10 ) << "" << std::setw( 14 ) << "x" << std::setw( 14 ) << "y" << std::setw( 14 ) << "z"
              << std::endl;
    for ( int c = 0; c < 2; ++c )
    {
        const char *row = ( c == 0 ) ? "psi" : "phi";
        std::cout << std::setw( 10 ) << row;
        for ( int a = 0; a < 3; ++a )
        {
            std::ostringstream cell;
            cell << "(" << left_bc[a][c] << "," << right_bc[a][c] << ")";
            std::cout << std::setw( 14 ) << cell.str();
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;


    vector_t solution( range );
    auto     vspace = std::make_shared<vec_ops_t>( range );

    // Initialize solution with 3D Perlin noise
    {
        vector_view_t solution_view( solution, false );

        scalar d = 1.0 / 4.0;
        for ( int i = 0; i < range[0]; i++ )
        {
            for ( int j = 0; j < range[1]; j++ )
            {
                for ( int k = 0; k < range[2]; k++ )
                {
                    scalar x = step[0] * ( 0.5 + i );
                    scalar y = step[1] * ( 0.5 + j );
                    scalar z = step[2] * ( 0.5 + k );


                    solution_view( i, j, k, 0 ) = 0.0;

                    if ( x > 0.5 - d && x < 0.5 + d && y > 0.5 - d && y < 0.5 + d && z < 2 * d)
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

    tests::scheduler<scalar> dt_scheduler( dt_inf, /*success_threshold=*/5 );

    auto cahn_hilliard_jacobi_op = std::make_shared<jacobi_op_t>( range, step, cond, time_derivative );
    auto cahn_hilliard_op        = std::make_shared<cahn_hilliard_op_t>( range, step, cond, cahn_hilliard_jacobi_op, time_derivative );

    // Set D and gamma parameters
    cahn_hilliard_op->set_D( D );
    cahn_hilliard_op->set_gamma( gamma );

    // Stationary operators (used for checking time convergence to stationary solution)
    auto cahn_hilliard_jacobi_op_stationary = std::make_shared<jacobi_op_t>( range, step, cond );
    auto cahn_hilliard_op_stationary        = std::make_shared<cahn_hilliard_op_t>( range, step, cond, cahn_hilliard_jacobi_op_stationary );
    cahn_hilliard_op_stationary->set_D( D );
    cahn_hilliard_op_stationary->set_gamma( gamma );

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

    // Verify that F_stationary(initial) norm
    vector_t F_init( range );
    cahn_hilliard_op_stationary->apply( solution, F_init );
    scalar F_init_norm = vspace->norm_l2( F_init );
    log.info_f( "||F_stationary(initial)||_2 = %le", static_cast<double>( F_init_norm ) );

    // Open time convergence history file
    std::ofstream time_conv_file( output_dir + "/time_converge_history.dat" );
    time_conv_file << "step F_stationary_norm" << std::endl;
    time_conv_file << 0 << " " << std::scientific << std::setprecision(15)
                   << static_cast<double>( F_init_norm ) << std::endl;
    time_conv_file.flush();

    // Record dt_inf that each accepted step converged with
    std::ofstream dt_history_file( output_dir + "/dt_history.dat" );
    dt_history_file << "step dt_inf\n";
    dt_history_file.flush();

    // Track total amount of each component after each step
    std::ofstream component_sums_csv( output_dir + "/component_sums.dat" );
    component_sums_csv << "step sum_psi sum_phi\n";
    component_sums_csv.flush();

    auto write_component_sums = [&]( int step_idx, const vector_t &x )
    {
        tensor_t sums = compute_component_sums( x, range );
        std::cout << "Component sums @ step " << step_idx << ": "
                  << std::scientific << std::setprecision(15)
                  << static_cast<double>( sums[0] ) << ", " << static_cast<double>( sums[1] )
                  << std::defaultfloat << std::endl;
        component_sums_csv << step_idx << " "
                           << std::scientific << std::setprecision(15)
                           << static_cast<double>( sums[0] ) << " " << static_cast<double>( sums[1] )
                           << std::defaultfloat << "\n";
        component_sums_csv.flush();
    };

    // Save initial approximation (index 0) if requested
    if ( save_coords )
    {
        std::string numerical_file = output_dir + "/numerical_0.bin";
        tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
    }

    // Solve and measure time for each time step
    double              total_time_ms = 0.0;

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
        newton_solver->convergence_strategy()->set_convergence_constants(
            /*tolerance_*/ newton_tol,
            /*maximum_iterations_*/ 10,
            /*relax_tolerance_factor_*/ scalar( 1 ),
            /*relax_tolerance_steps_*/ 0
        );

        int ts = 0;
        while ( ts < max_time_steps )
        {
            std::cout << std::endl;
            std::cout << "Time iteration #" << ( ts + 1 ) << " has started" << std::endl;
            std::cout << std::endl;

            // Create step-specific output directory
            std::string step_dir = output_dir + "/step_" + std::to_string( ts + 1 );
            std::filesystem::create_directories( step_dir );

            const vector_t backup_solution = solution;
            bool           step_accepted   = false;
            double         accepted_step_time = 0.0;
            scalar         accepted_dt_inf = scalar( 0 );
            int            attempt_idx     = 0;
            for ( attempt_idx = 1; attempt_idx <= DEFAULT_MAX_RETRIES + 1; ++attempt_idx )
            {
                const scalar current_dt_inf = dt_scheduler.get_dt_inf();
                time_derivative->set_dt_inf( current_dt_inf );

                std::cout << "  dt_inf attempt " << attempt_idx << ": "
                          << std::scientific << static_cast<double>( current_dt_inf ) << std::defaultfloat << std::endl;

                std::string attempt_dir = step_dir + "/attempt_" + std::to_string( attempt_idx );
                std::filesystem::create_directories( attempt_dir );

                // Create convergence monitor for this attempt
                auto conv_monitor = std::make_shared<newton_conv_monitor_jacobi_t>(
                    vspace, cahn_hilliard_op, get_monitor, attempt_dir, solver_type, preconditioner_type, grid_size, &log );

                // Write initial residual (iteration 0) before Newton iterations start
                conv_monitor->write_initial_residual( solution );

                Timer timer( "Solve", false );
                const bool converged = newton_solver->solve( cahn_hilliard_op.get(), conv_monitor.get(), nullptr, solution );
                const double attempt_time = timer.stop_and_get_ms();

                dt_scheduler.step( converged );

                if ( converged )
                {
                    accepted_step_time = attempt_time;
                    accepted_dt_inf    = current_dt_inf;
                    step_accepted      = true;
                    break;
                }

                // rollback state for the next attempt
                solution = backup_solution;
            }

            if ( !step_accepted )
            {
                std::cerr << "ERROR: Failed to take time step after " << ( DEFAULT_MAX_RETRIES + 1 )
                          << " attempts. Aborting." << std::endl;
                break;
            }

            total_time_ms += accepted_step_time;

            // Record dt_inf that this step converged with
            dt_history_file << ( ts + 1 ) << " " << std::scientific << std::setprecision(15)
                            << static_cast<double>( accepted_dt_inf ) << std::defaultfloat << "\n";
            dt_history_file.flush();

            // Compute stationary residual (to check time convergence)
            vector_t F_x( range );
            cahn_hilliard_op_stationary->apply( solution, F_x );
            scalar F_x_norm = vspace->norm_l2( F_x );
            log.info_f( "||F_stationary(solution)||_2 = %le", static_cast<double>( F_x_norm ) );

            // Track total amount of each component after this step
            write_component_sums( ts + 1, solution );

            // Write to time convergence history file
            time_conv_file << ( ts + 1 ) << " " << std::scientific << std::setprecision(15)
                          << static_cast<double>( F_x_norm ) << std::endl;
            time_conv_file.flush();

            // Check for early termination based on F(x) norm
            if ( F_x_norm < time_tol )
            {
                log.info_f( "Early termination: ||F_stationary(solution)||_2 = %le < %le (tolerance)",
                           static_cast<double>( F_x_norm ), static_cast<double>( time_tol ) );
                time_derivative->set_previous_state( solution );
                break;
            }

            // Compute norm of difference between solution and previous state
            vector_t previous_state = time_derivative->get_previous_state();
            vector_t diff_prev( range );
            vspace->assign_lin_comb( scalar( 1 ), solution, scalar( -1 ), previous_state, diff_prev );
            scalar diff_prev_norm = vspace->norm_l2( diff_prev );
            log.info_f( "||solution - previous_state||_2 = %le", static_cast<double>( diff_prev_norm ) );

            // Update previous state before the next step
            time_derivative->set_previous_state( solution );

            // Save numerical solution at each step if requested
            if ( save_coords )
            {
                std::string numerical_file = output_dir + "/numerical_" + std::to_string( ts + 1 ) + ".bin";
                tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
            }

            // Separate iterations with empty line
            if ( ts < max_time_steps - 1 )
            {
                std::cout << std::endl;
            }

            ++ts;
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
        newton_solver->convergence_strategy()->set_convergence_constants(
            /*tolerance_*/ newton_tol,
            /*maximum_iterations_*/ 10,
            /*relax_tolerance_factor_*/ scalar( 1 ),
            /*relax_tolerance_steps_*/ 0
        );

        int ts = 0;
        while ( ts < max_time_steps )
        {
            std::cout << std::endl;
            std::cout << "Time iteration #" << ( ts + 1 ) << " has started" << std::endl;
            std::cout << std::endl;

            // Create step-specific output directory
            std::string step_dir = output_dir + "/step_" + std::to_string( ts + 1 );
            std::filesystem::create_directories( step_dir );

            const vector_t backup_solution = solution;
            bool           step_accepted   = false;
            double         accepted_step_time = 0.0;
            scalar         accepted_dt_inf = scalar( 0 );
            int            attempt_idx     = 0;
            for ( attempt_idx = 1; attempt_idx <= DEFAULT_MAX_RETRIES + 1; ++attempt_idx )
            {
                const scalar current_dt_inf = dt_scheduler.get_dt_inf();
                time_derivative->set_dt_inf( current_dt_inf );

                std::cout << "  dt_inf attempt " << attempt_idx << ": "
                          << std::scientific << static_cast<double>( current_dt_inf ) << std::defaultfloat << std::endl;

                std::string attempt_dir = step_dir + "/attempt_" + std::to_string( attempt_idx );
                std::filesystem::create_directories( attempt_dir );

                // Create convergence monitor for this attempt
                auto conv_monitor = std::make_shared<newton_conv_monitor_gmres_t>(
                    vspace, cahn_hilliard_op, get_monitor, attempt_dir, solver_type, preconditioner_type, grid_size, &log );

                // Write initial residual (iteration 0) before Newton iterations start
                conv_monitor->write_initial_residual( solution );

                Timer timer( "Solve", false );
                const bool converged = newton_solver->solve( cahn_hilliard_op.get(), conv_monitor.get(), nullptr, solution );
                const double attempt_time = timer.stop_and_get_ms();

                dt_scheduler.step( converged );

                if ( converged )
                {
                    accepted_step_time = attempt_time;
                    accepted_dt_inf    = current_dt_inf;
                    step_accepted      = true;
                    break;
                }

                // rollback state for the next attempt
                solution = backup_solution;
            }

            if ( !step_accepted )
            {
                std::cerr << "ERROR: Failed to take time step after " << ( DEFAULT_MAX_RETRIES + 1 )
                          << " attempts. Aborting." << std::endl;
                break;
            }

            total_time_ms += accepted_step_time;

            // Record dt_inf that this step converged with
            dt_history_file << ( ts + 1 ) << " " << std::scientific << std::setprecision(15)
                            << static_cast<double>( accepted_dt_inf ) << std::defaultfloat << "\n";
            dt_history_file.flush();

            // Compute stationary residual (to check time convergence)
            vector_t F_x( range );
            cahn_hilliard_op_stationary->apply( solution, F_x );
            scalar F_x_norm = vspace->norm_l2( F_x );
            log.info_f( "||F_stationary(solution)||_2 = %le", static_cast<double>( F_x_norm ) );

            // Track total amount of each component after this step
            write_component_sums( ts + 1, solution );

            // Write to time convergence history file
            time_conv_file << ( ts + 1 ) << " " << std::scientific << std::setprecision(15)
                          << static_cast<double>( F_x_norm ) << std::endl;
            time_conv_file.flush();

            // Compute norm of difference between solution and previous state
            vector_t previous_state = time_derivative->get_previous_state();
            vector_t diff_prev( range );
            vspace->assign_lin_comb( scalar( 1 ), solution, scalar( -1 ), previous_state, diff_prev );
            scalar diff_prev_norm = vspace->norm_l2( diff_prev );
            log.info_f( "||solution - previous_state||_2 = %le", static_cast<double>( diff_prev_norm ) );

            // Update previous state before the next step
            time_derivative->set_previous_state( solution );

            // Save numerical solution at each step if requested
            if ( save_coords )
            {
                std::string numerical_file = output_dir + "/numerical_" + std::to_string( ts + 1 ) + ".bin";
                tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
            }

            // Separate iterations with empty line
            if ( ts < max_time_steps - 1 )
            {
                std::cout << std::endl;
            }

            ++ts;

            // Check for early termination based on F(x) norm
            if ( diff_prev_norm < time_tol )
            {
                log.info_f( "Early termination: ||solution - previous_state||_2 = %le < %le (tolerance)",
                           static_cast<double>( diff_prev_norm ), static_cast<double>( time_tol ) );
                time_derivative->set_previous_state( solution );
                break;
            }
        }
    }

    // Close time convergence history file
    time_conv_file.close();
    component_sums_csv.close();

    // Final results
    scalar final_solution_norm = vspace->norm_l2( solution );

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  ||solution||_2:              " << std::scientific << final_solution_norm << std::endl;
    std::cout << "  Average iteration time:      " << std::fixed << std::setprecision( 2 )
              << ( total_time_ms / max_time_steps ) << " ms" << std::endl;
    std::cout << "  Total solve time:            " << std::fixed << std::setprecision( 2 ) << total_time_ms
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
