#include "include/balancer.h"
#include "include/biharmonic_problem.h"
#include "include/coarsening.h"
#include "include/jacobi_op.h"
#include "include/jacobi_pre.h"
#include "include/prolongator.h"
#include "include/restrictor.h"
#include "include/boundary.h"
#include "include/kernels/phobic_energy.h"
#include "include/kernels/mobility.h"
#include "include/time_derivative.h"
#include "include/solution_io.h"
#include <nmfd/utils/logging.h>
#include <nmfd/utils/profiling.h>

#include <CLI/CLI.hpp>
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

#ifdef SCFD_BACKEND_ENABLE_MPI
#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/mpi_rect_distributor.h>
#else
#include <scfd/communication/trivial_platform.h>
#include <scfd/communication/trivial_comm.h>
#include <scfd/communication/rect_distributor.h>
#endif

#include <scfd/communication/rect_partitioner.h>
#include <scfd/utils/log.h>
#include <string>
#include <type_traits>

using backend = scfd::backend::current;

/**************************************/

constexpr int dim               = 3;
constexpr int tensor_dim        = 2;
// The multigrid restriction stencil reaches two cells past the block and spans the full diagonal,
// so the halo has to be two cells wide and exchanged with the corner neighbours as well.
constexpr int stencil           = 2;   // ghost width per side; must match the distributor stencil
constexpr int max_stencil_order = dim; // highest coupled stencil order for the halo exchange

#ifndef USE_DOUBLE_PRECISION
using scalar      = float;
#else
using scalar      = double;
#endif

using grid_step_type = scfd::static_vec::vec<scalar, dim>;
using idx_nd_type    = scfd::static_vec::vec<int, dim>;

using log_t = current_log;

using ord_t           = int;
using big_ord_t       = long int;
using mem_t           = backend::memory_type;
using dist_for_each_t = backend::for_each_nd_type<dim, ord_t>;

#ifdef SCFD_BACKEND_ENABLE_MPI
using comm_platform_t = scfd::communication::mpi_wrap;
using comm_info_t     = scfd::communication::mpi_comm_info;
using dist_t          = scfd::communication::mpi_rect_distributor<scalar, dim, mem_t, dist_for_each_t, ord_t, big_ord_t, comm_info_t>;
#else
using comm_platform_t = scfd::communication::trivial_platform<mem_t>;
using comm_info_t     = scfd::communication::trivial_comm<mem_t>;
using dist_t          = scfd::communication::rect_distributor<scalar, dim, mem_t, dist_for_each_t, ord_t, big_ord_t, comm_info_t>;
#endif

using part_t          = scfd::communication::rect_partitioner<dim, ord_t, big_ord_t, comm_info_t>;

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

using prolongator_t = tests::prolongator<vec_ops_t, log_t, dist_t>;
using restrictor_t  = tests::restrictor<vec_ops_t, log_t, dist_t>;
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
    comm_platform_t comm( argc, argv );        // mpi_wrap calls MPI_Init; trivial_platform is a single-rank stand-in
    comm_info_t     comm_world = comm.comm_world();

    auto prof = std::make_shared<current_prof>();
    current_prof::set_inst( prof.get() );

    // Parse CLI arguments
    CLI::App app{ "Biharmonic solver test" };
    app.get_formatter()->column_width( 42 );

    std::string solver_type;
    std::string preconditioner_type;
    int         grid_size   = 32;
    std::string output_dir  = ".";
    bool        save_coords = false;
    bool        verbose     = false;

    // Solver parameters (initialized to defaults)
    int    max_iterations = DEFAULT_MAX_ITERATIONS;
    int    gmres_basis    = DEFAULT_GMRES_BASIS;
    int    mg_sweeps_pre  = DEFAULT_MG_SWEEPS_PRE;
    int    mg_sweeps_post = DEFAULT_MG_SWEEPS_POST;
    scalar tolerance      = DEFAULT_TOLERANCE;

    app.add_option( "solver", solver_type, "Solver type" )
        ->required()
        ->check( CLI::IsMember( std::vector<std::string>{ "jacobi", "gmres" } ) );
    app.add_option( "preconditioner", preconditioner_type, "Preconditioner type (diagonal/Jacobi or multigrid)" )
        ->required()
        ->check( CLI::IsMember( std::vector<std::string>{ "diag", "mg" } ) );
    // Multigrid halves the grid down to two cells, so every extent must stay even all the way down.
    app.add_option( "grid_size", grid_size, "Number of grid points per dimension (e.g., 32)" )
        ->required()
        ->check( []( const std::string &str ) -> std::string {
            int val = std::stoi( str );
            if ( val < 2 || ( val & ( val - 1 ) ) != 0 )
                return "grid_size must be a power of two, got " + str + ".";
            return std::string();
        } );
    app.add_option( "output_dir", output_dir, "Output directory (must already exist; created by the caller, e.g. run.sh)" )
        ->capture_default_str();

    app.add_flag( "--save-coords", save_coords, "Save numerical and exact solutions to binary files" );
    app.add_flag( "--verbose", verbose, "Print per-iteration residuals and the profiler breakdown to the log" );
    app.add_option( "--max-iterations", max_iterations, "Maximum solver iterations" )->capture_default_str();
    app.add_option( "--gmres-basis", gmres_basis, "GMRES basis size" )->capture_default_str();
    app.add_option( "--mg-sweeps-pre", mg_sweeps_pre, "Multigrid pre-sweeps" )->capture_default_str();
    app.add_option( "--mg-sweeps-post", mg_sweeps_post, "Multigrid post-sweeps" )->capture_default_str();
    app.add_option( "--tolerance", tolerance, "Solver tolerance" )->capture_default_str();

    try
    {
        app.parse( argc, argv );
    }
    catch ( const CLI::ParseError &e )
    {
        int rc = ( comm_world.myid == 0 ) ? app.exit( e ) : e.get_exit_code();
        return rc;
    }

    // Solver configuration
    const std::string scalar_label = std::is_same<float, scalar>::value ? "float" : "double";

    log_t log;
    // Set log verbosity: 0 suppresses INFO messages, 1 allows them
    log.set_verbosity( verbose ? 1 : 0 );

    // Write configuration header to log
    log.info( "========================================" );
#ifdef SCFD_BACKEND_ENABLE_MPI
    log.info( "Biharmonic Solver Configuration (MPI)" );
#else
    log.info( "Biharmonic Solver Configuration" );
#endif
    log.info( "========================================" );
    log.info( "" );
    log.info( "Problem Settings:" );
    log.info_f( "  Grid size:     %d x %d x %d", grid_size, grid_size, grid_size );
    log.info_f( "  Tensor dim:    %d", tensor_dim );
    log.info_f( "  Scalar type:   %s", scalar_label.c_str() );
    log.info_f( "  DOFs:          %lld", static_cast<long long>( grid_size ) * grid_size * grid_size * tensor_dim );
    log.info_f( "  Processes:     %d", comm_world.num_procs );
    log.info( "" );
    log.info( "Solver:" );
    log.info_f( "  Type:          %s", solver_type.c_str() );
    log.info_f( "  Tolerance:     %e", static_cast<double>( tolerance ) );
    log.info_f( "  Max iters:     %d", max_iterations );
    if ( solver_type == "gmres" )
    {
        log.info_f( "  Basis size:    %d", gmres_basis );
        log.info( "  Precond side:  L" );
        log.info( "  Reorthogon.:   true" );
    }
    log.info( "" );
    log.info( "Preconditioner:" );
    log.info_f( "  Type:          %s", preconditioner_type.c_str() );
    log.info( "" );
    log.info( "Output:" );
    log.info_f( "  Directory:     %s", output_dir.c_str() );
    log.info_f( "  Save coords:   %s", save_coords ? "yes" : "no" );
    log.info( "========================================" );
    log.info( "" );

    auto step  = grid_step_type::make_ones() / scalar( grid_size );

    // Automatic balanced decomposition for any power-of-two process count.
    if ( comm_world.num_procs < 1 || ( comm_world.num_procs & ( comm_world.num_procs - 1 ) ) != 0 )
    {
        log.error( "process count must be a power of two." );
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
        log.error_f( "domain decomposition failed: %s", e.what() );
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
        solver_params.monitor.rel_tol       = tolerance;
        solver_params.monitor.max_iters_num = max_iterations;
        solver = std::make_shared<jacobi_solver>( l_op, vspace, &log, solver_params, precond );
    }
    else // gmres
    {
        gmres_solver::params params_gmres;
        params_gmres.monitor.rel_tol                      = tolerance;
        params_gmres.monitor.max_iters_num                = max_iterations;
        params_gmres.do_restart_on_false_ritz_convergence = true;
        params_gmres.basis_size                           = gmres_basis;
        params_gmres.preconditioner_side                  = 'L';
        params_gmres.reorthogonalization                  = true;
        solver = std::make_shared<gmres_solver>( l_op, vspace, &log, params_gmres, precond );
    }

    SCFD_PLATFORM_TIC( "Solve" );
    converged     = solver->solve( rhs, solution );
    solve_time_ms = current_prof::inst().toc( "Solve" );

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

    log.info( "" );
    log.info( "========================================" );
    log.info( "Results" );
    log.info( "========================================" );
    log.info_f( "  Converged:                  %s", converged ? "yes" : "no" );
    log.info_f( "  ||solution - exact||_2:     %e", static_cast<double>( error_norm ) );
    log.info_f( "  Relative error:             %e", static_cast<double>( error_norm / exact_norm ) );
    log.info_f( "  Total solve time:           %.2f ms", solve_time_ms );
    log.info( "========================================" );

    // Save solutions if requested (only valid for a single rank owning the whole domain)
    if ( save_coords && comm_world.num_procs == 1 )
    {
        std::string numerical_file = output_dir + "/numerical.bin";
        std::string exact_file     = output_dir + "/exact.bin";

        tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
        tests::save_solution_binary<vector_t, idx_nd_type>( exact_solution, exact_file, grid_size, tensor_dim );

        log.info( "" );
        log.info( "Saved solutions:" );
        log.info_f( "  Numerical: %s", numerical_file.c_str() );
        log.info_f( "  Exact:     %s", exact_file.c_str() );
    }

#ifdef SCFD_ENABLE_PROFILING
    if ( verbose )
    {
        current_prof::inst().log_print( log );
    }
    log.set_verbosity( 1 );
    current_prof::inst().log_print_totals( log );
#endif

    return 0;
}
