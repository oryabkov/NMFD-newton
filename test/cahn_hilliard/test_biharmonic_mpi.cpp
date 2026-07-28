#include "balancer.h"
#include "biharmonic_problem.h"
#include "coarsening.h"
#include "convergence_history_io.h"
#include "jacobi_op.h"
#include "jacobi_pre.h"
#include "prolongator.h"
#include "restrictor.h"
#include "include/boundary.h"
#include "kernels/phobic_energy.h"
#include "kernels/mobility.h"
#include "time_derivative.h"
#include "solution_io.h"
#include "timers.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nmfd/operations/rect_vector_space.h>
#include <nmfd/preconditioners/dummy.h>
#include <nmfd/preconditioners/mg.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/iter_solver_base.h>
#include <nmfd/solvers/jacobi.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <scfd/backend/backend.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/mpi_rect_distributor.h>
#include <scfd/communication/rect_partitioner.h>
#include <scfd/utils/log.h>
#include <algorithm>
#include <sstream>
#include <string>
#include <type_traits>

using backend = scfd::backend::current;

/**************************************/

constexpr int dim               = 3;
constexpr int tensor_dim        = 2;
constexpr int stencil           = 1; // ghost width per side; must match the distributor stencil
constexpr int max_stencil_order = 1; // highest coupled stencil order for the halo exchange

#ifndef USE_DOUBLE_PRECISION
using scalar      = float;
#else
using scalar      = double;
#endif

using grid_step_type = scfd::static_vec::vec<scalar, dim>;
using idx_nd_type    = scfd::static_vec::vec<int, dim>;

using log_t = scfd::utils::log_std;

using ord_t           = int;
using big_ord_t       = long int;
using mem_t           = backend::memory_type;
using comm_platform_t = scfd::communication::mpi_wrap;
using comm_info_t     = scfd::communication::mpi_comm_info;
using part_t          = scfd::communication::rect_partitioner<dim, ord_t, big_ord_t, comm_info_t>;
using dist_for_each_t = backend::for_each_nd_type<dim, ord_t>;
using dist_t          = scfd::communication::mpi_rect_distributor<scalar, dim, mem_t, dist_for_each_t, ord_t, big_ord_t, comm_info_t>;
using big_idx_t       = scfd::static_vec::vec<big_ord_t, dim>;
using periodic_flags_t = scfd::static_vec::vec<bool, dim>;
using rect_t          = scfd::static_vec::rect<ord_t, dim>;
using big_rect_t      = scfd::static_vec::rect<big_ord_t, dim>;

using vec_ops_t     = nmfd::rect_vector_space<scalar, /*dim=*/dim, /*tensor_dim=*/tensor_dim, backend, comm_info_t>;
using vector_t      = typename vec_ops_t::vector_type;
using tensor_t      = scfd::static_vec::vec<scalar, tensor_dim>;
using vector_view_t = typename vector_t::view_type;

using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>;
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

using phobic_energy_t = tests::zero_potential<scalar>;
using mobility_t      = tests::constant_mobility<scalar>;
using zero_rhs_t      = tests::zero_rhs<scalar, tensor_t>;
using rhs_t           = tests::trig_rhs<scalar, tensor_t>;
using time_derivative_t = tests::time_derivative<vec_ops_t, tensor_t>;

using lin_op_t      = tests::jacobi_op<vec_ops_t, log_t, phobic_energy_t, time_derivative_t, mobility_t, dist_t>;
using smoother_t    = tests::jacobi_pre<vec_ops_t, log_t, phobic_energy_t, time_derivative_t, mobility_t, dist_t>;

using prolongator_t = tests::prolongator<vec_ops_t, log_t>;
using restrictor_t  = tests::restrictor<vec_ops_t, log_t>;
using ident_op_t    = nmfd::preconditioners::dummy<vec_ops_t, lin_op_t>;
using coarsening_t  = tests::coarsening<lin_op_t, log_t>;

using mg_t =
    nmfd::preconditioners::mg<lin_op_t, restrictor_t, prolongator_t, smoother_t, ident_op_t, coarsening_t, log_t>;
using mg_params_t = mg_t::params_hierarchy;
using mg_utils_t  = mg_t::utils_hierarchy;

using precond_interface = nmfd::preconditioners::preconditioner_interface<vec_ops_t, lin_op_t>;

using jacobi_solver = nmfd::solvers::jacobi<vec_ops_t, lin_op_t, precond_interface, krylov_monitor_t, log_t>;
using gmres_solver  = nmfd::solvers::gmres<vec_ops_t, krylov_monitor_t, log_t, lin_op_t, precond_interface>;
using linsolver_base_t = nmfd::solvers::iter_solver_base<vec_ops_t, krylov_monitor_t, log_t, lin_op_t, precond_interface>;

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
constexpr scalar DEFAULT_TOLERANCE      = std::is_same<float, scalar>::value ? 5e-6f : 1e-10;

/**************************************/

int main( int argc, char *argv[] )
{
    comm_platform_t comm( argc, argv );        // mpi_wrap: calls MPI_Init
    comm_info_t     comm_world = comm.comm_world();
    const bool      is_root = ( comm_world.myid == 0 );

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
        if ( is_root )
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
        }
        return 1;
    }

    solver_type         = argv[1];
    preconditioner_type = argv[2];
    grid_size           = std::stoi( argv[3] );

    // Validate solver and preconditioner types
    if ( solver_type != "jacobi" && solver_type != "gmres" )
    {
        if ( is_root )
            std::cerr << "ERROR: Unknown solver type '" << solver_type << "'. Use 'jacobi' or 'gmres'." << std::endl;
        return 1;
    }

    if ( preconditioner_type != "diag" && preconditioner_type != "mg" )
    {
        if ( is_root )
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
            if ( is_root )
                std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
        else
        {
            prefix = arg;
        }
    }

    // Variables hoisted so they remain in scope for the lifetime of the redirect
    std::string output_dir;
    std::ofstream log_file;
    std::streambuf* old_cout_buf = nullptr;
    // tee_buf must outlive the redirect of std::cout, so it is declared here
    // (unique_ptr so it can be conditionally constructed on root only)
    std::unique_ptr<tee_streambuf> tee_buf_ptr;

    if ( is_root )
    {
        // Create output directory with timestamp
        output_dir = "data/" + prefix + "_" + get_timestamp_string();
        std::filesystem::create_directories( output_dir );

        // Open log file and set up tee output
        log_file.open( output_dir + "/log.txt" );
        tee_buf_ptr.reset( new tee_streambuf( std::cout.rdbuf(), log_file.rdbuf() ) );

        // Redirect std::cout to tee
        old_cout_buf = std::cout.rdbuf( tee_buf_ptr.get() );
    }

    // Solver configuration
    const std::string scalar_label = std::is_same<float, scalar>::value ? "float" : "double";

    log_t log;
    // Set log verbosity: 0 suppresses INFO messages, 1 allows them
    log.set_verbosity( ( verbose && is_root ) ? 1 : 0 );

    if ( is_root )
    {
        // Write configuration header to log
        std::cout << "========================================" << std::endl;
        std::cout << "Biharmonic Solver Configuration (MPI)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::endl;
        std::cout << "Problem Settings:" << std::endl;
        std::cout << "  Grid size:     " << grid_size << " x " << grid_size << " x " << grid_size << std::endl;
        std::cout << "  Tensor dim:    " << tensor_dim << std::endl;
        std::cout << "  Scalar type:   " << scalar_label << std::endl;
        std::cout << "  DOFs:          " << static_cast<long long>( grid_size ) * grid_size * grid_size * tensor_dim
                  << std::endl;
        std::cout << "  MPI ranks:     " << comm_world.num_procs << std::endl;
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
        std::cout << std::endl;
        std::cout << "Output:" << std::endl;
        std::cout << "  Directory:     " << output_dir << std::endl;
        std::cout << "  Save coords:   " << ( save_coords ? "yes" : "no" ) << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::endl;
    }

    auto step  = grid_step_type::make_ones() / scalar( grid_size );

    // Automatic balanced decomposition for any power-of-two process count.
    if ( comm_world.num_procs < 1 || ( comm_world.num_procs & ( comm_world.num_procs - 1 ) ) != 0 )
    {
        if ( is_root )
            std::cerr << "ERROR: test_biharmonic_mpi requires a power-of-two MPI rank count." << std::endl;
        return 1;
    }

    // Boundary conditions of the WHOLE computational domain (same as the serial test):
    //   left  = dirichlet (-1) on every axis, right = periodic (0) on every axis  [psi, phi].
    //   -1 = dirichlet (value 0), +1 = neumann (derivative 0), 0 = periodic (reads opposite side).
    int global_left_bc[3][2]  = { { -1, -1 }, { -1, -1 }, { -1, -1 } };
    int global_right_bc[3][2] = { {  0,  0 }, {  0,  0 }, {  0,  0 } };

    big_idx_t dom_sz( grid_size, grid_size, grid_size );
    part_t    part( comm_world, dom_sz );

    // The balancer splits the global domain into congruent
    // power-of-two blocks and derives this rank's local BCs
    tests::balancer<dim, ord_t, big_ord_t, tensor_dim> bal;

    std::vector<big_rect_t> proc_rects;
    big_rect_t              my_own_glob_rect;
    int                     left_bc[3][2];
    int                     right_bc[3][2];
    periodic_flags_t        periodic_flags;
    try
    {
        bal.balance(
            dom_sz, comm_world.num_procs, comm_world.myid, global_left_bc, global_right_bc, proc_rects,
            my_own_glob_rect, left_bc, right_bc, periodic_flags );
    }
    catch ( const std::exception &e )
    {
        if ( is_root )
            std::cerr << "ERROR: domain decomposition failed: " << e.what() << std::endl;
        return 1;
    }
    part.proc_rects = proc_rects;

    rect_t my_own_loc_rect = rect_t( idx_nd_type::make_zero(), my_own_glob_rect.calc_size() );
    auto   range           = my_own_loc_rect.calc_size();

    auto cond = tests::boundary_cond<vec_ops_t>( left_bc, right_bc );

    // Distributor initialization: fills interior-interface halos and wraps the physical periodic
    // walls; halos at dirichlet walls are filled but ignored by the kernel.
    auto dist = std::make_shared<dist_t>();
    dist->init_for_tensors( tensor_dim, part, periodic_flags, stencil, max_stencil_order );

    rhs_t    rhs_function;

    auto vspace = std::make_shared<vec_ops_t>( range, comm_world, false, stencil, max_stencil_order );

    vector_t solution, rhs, exact_solution;
    vspace->init_vector( solution );
    vspace->init_vector( rhs );
    vspace->init_vector( exact_solution );
    {
        vspace->assign_scalar( 0.0, solution ); // Initialize solution to zero
        vector_view_t rhs_view( rhs, false ), exact_view( exact_solution, false );

        for ( int i = 0; i < range[0]; i++ )
        {
            for ( int j = 0; j < range[1]; j++ )
            {
                for ( int k = 0; k < range[2]; k++ )
                {
                    scalar x = step[0] * ( 0.5 + i + my_own_glob_rect.i1[0] );
                    scalar y = step[1] * ( 0.5 + j + my_own_glob_rect.i1[1] );
                    scalar z = step[2] * ( 0.5 + k + my_own_glob_rect.i1[2] );

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

    auto l_op = std::make_shared<lin_op_t>( vspace, step, cond, dist );

    std::shared_ptr<precond_interface> precond;
    if ( preconditioner_type == "diag" )
    {
        auto smoother = std::make_shared<smoother_t>( l_op, dist );
        precond = smoother;
    }
    else // mg
    {
        mg_utils_t  mg_utils;
        mg_params_t mg_params;

        mg_utils.log              = &log;
        mg_params.direct_coarse   = false;
        mg_params.num_sweeps_pre  = mg_sweeps_pre;
        mg_params.num_sweeps_post = mg_sweeps_post;

        // Coarse levels reuse this decomposition with every block halved
        mg_utils.coarsening.part              = part;
        mg_utils.coarsening.periodic_flags    = periodic_flags;
        mg_utils.coarsening.stencil           = stencil;
        mg_utils.coarsening.max_stencil_order = max_stencil_order;

        precond = std::make_shared<mg_t>( mg_utils, mg_params );
    }

    // Solve the system and measure execution time
    double solve_time_ms;
    bool   converged;

    std::shared_ptr<linsolver_base_t> solver;
    if ( solver_type == "jacobi" )
    {
        jacobi_solver::params solver_params;
        solver_params.monitor.rel_tol                  = tolerance;
        solver_params.monitor.max_iters_num            = max_iterations;
        solver_params.monitor.save_convergence_history = true;
        solver = std::make_shared<jacobi_solver>( l_op, vspace, &log, solver_params, precond );
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
        solver = std::make_shared<gmres_solver>( l_op, vspace, &log, params_gmres, precond );
    }

    {
        Timer timer("Solve", false); // Don't print automatically, we'll print in Results section
        converged     = solver->solve( rhs, solution );
        solve_time_ms = timer.stop_and_get_ms();
    }

    if ( is_root )
    {
        // Save times.dat and convergence history
        {
            std::chrono::duration<double, std::milli> solve_time_duration(solve_time_ms);
            save_times_dat<krylov_monitor_t, scalar>( solver->monitor(), solver_type, preconditioner_type,
                                                        grid_size, solve_time_duration, output_dir );
            save_convergence_history<krylov_monitor_t, scalar>( solver->monitor(), output_dir );
        }
    }

    // Verify that L(exact_solution) - rhs is close to zero
    vector_t L_exact;
    vspace->init_vector( L_exact );
    l_op->apply( exact_solution, L_exact );
    vector_t residual_exact;
    vspace->init_vector( residual_exact );
    vspace->assign_lin_comb( scalar( 1 ), L_exact, scalar( -1 ), rhs, residual_exact );
    scalar residual_exact_norm = vspace->norm_l2( residual_exact );
    log.info_f( "Verification: ||L(exact_solution) - rhs||_2 = %le", static_cast<double>( residual_exact_norm ) );

    // Compute error between numerical and exact solutions
    vector_t error;
    vspace->init_vector( error );
    vspace->assign_lin_comb( scalar( 1 ), solution, scalar( -1 ), exact_solution, error );
    scalar error_norm = vspace->norm_l2( error );
    scalar exact_norm = vspace->norm_l2( exact_solution );

    if ( is_root )
    {
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
    }

    // Save solutions if requested (only valid for a single rank owning the whole domain)
    if ( save_coords && comm_world.num_procs == 1 && is_root )
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

    // Restore cout (root only)
    if ( is_root )
    {
        std::cout.rdbuf( old_cout_buf );
        log_file.close();
    }

    return 0;
}
