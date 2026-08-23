#include "include/balancer.h"
#include "include/coarsening.h"
#include "include/free_energy.h"
#include "include/cahn_hilliard_op.h"
#include "include/jacobi_op.h"
#include "include/jacobi_pre.h"
#include "include/prolongator.h"
#include "include/restrictor.h"
#include "include/boundary.h"
#include "include/kernels/phobic_energy.h"
#include "include/kernels/mobility.h"
#include "include/solution_io.h"
#include "include/perlin_noise.h"
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
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
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
#include <vector>

using backend = scfd::backend::current;

/**************************************/

constexpr int dim               = 3;
constexpr int tensor_dim        = 2;
constexpr int stencil           = 2;
constexpr int max_stencil_order = dim;

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

// Monitors
using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>;
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

// Problem
using phobic_energy     = tests::double_well_potential<scalar>;
using mobility_t        = tests::constant_mobility<scalar>;
using time_derivative_t = tests::time_derivative<vec_ops_t, tensor_t>;

using free_energy_t = tests::free_energy<vec_ops_t, phobic_energy, dist_t>;

// MG
using prolongator_t = tests::prolongator<vec_ops_t, log_t, dist_t>;
using restrictor_t  = tests::restrictor<vec_ops_t, log_t, dist_t>;
using jacobi_op_t   = tests::jacobi_op<vec_ops_t, log_t, phobic_energy, time_derivative_t, mobility_t, dist_t>;
using ident_op_t    = nmfd::preconditioners::dummy<vec_ops_t, jacobi_op_t>;
using smoother_t    = tests::jacobi_pre<vec_ops_t, log_t, phobic_energy, time_derivative_t, mobility_t, dist_t>;
using coarsening_t  = tests::coarsening<jacobi_op_t, log_t>;

using precond_interface = nmfd::preconditioners::preconditioner_interface<vec_ops_t, jacobi_op_t>;

using mg_t =
    nmfd::preconditioners::mg<jacobi_op_t, restrictor_t, prolongator_t, smoother_t, ident_op_t, coarsening_t, log_t>;
using mg_params_t = mg_t::params_hierarchy;
using mg_utils_t  = mg_t::utils_hierarchy;

using jacobi_solver = nmfd::solvers::jacobi<vec_ops_t, jacobi_op_t, precond_interface, krylov_monitor_t, log_t>;
using gmres_solver  = nmfd::solvers::gmres<vec_ops_t, krylov_monitor_t, log_t, jacobi_op_t, precond_interface>;
using linsolver_base_t = nmfd::solvers::iter_solver_base<vec_ops_t, krylov_monitor_t, log_t, jacobi_op_t, precond_interface>;

// Newton
using cahn_hilliard_op_t = tests::cahn_hilliard_op<vec_ops_t, jacobi_op_t, log_t, phobic_energy, time_derivative_t, mobility_t, dist_t>;
using newton_iteration_t = nmfd::solvers::newton_iteration<vec_ops_t, cahn_hilliard_op_t, linsolver_base_t>;
using newton_solver_t = nmfd::solvers::nonlinear_solver<vec_ops_t, log_t, cahn_hilliard_op_t, newton_iteration_t>;

/**************************************/
// Default solver parameters
/**************************************/
constexpr int    DEFAULT_MAX_ITERATIONS  = 100;
constexpr int    DEFAULT_GMRES_BASIS     = 25;
constexpr int    DEFAULT_MG_SWEEPS_PRE   = 4;
constexpr int    DEFAULT_MG_SWEEPS_POST  = 4;
constexpr scalar DEFAULT_NEWTON_TOL      = std::is_same<float, scalar>::value ? 5e-6f : 1e-10;
constexpr scalar DEFAULT_TOLERANCE       = std::is_same<float, scalar>::value ? 5e-6f : 1e-10;
constexpr int    DEFAULT_MAX_TIME_STEPS  = 10;
constexpr scalar DEFAULT_DT_INF          = 1.0;
constexpr scalar DEFAULT_NOISE_AMPLITUDE = 1.0;
constexpr scalar DEFAULT_NOISE_FREQUENCY = 10.0;

/**************************************/

int main( int argc, char *argv[] )
{
    comm_platform_t comm( argc, argv );
    comm_info_t     comm_world = comm.comm_world();

    auto prof = std::make_shared<current_prof>();
    current_prof::set_inst( prof.get() );

    // Parse CLI arguments
    CLI::App app{ "Cahn-Hilliard random-init solver test" };
    app.get_formatter()->column_width( 42 );

    std::string solver_type;
    std::string preconditioner_type;
    int         grid_size   = 32;
    std::string output_dir  = ".";
    bool        save_coords = false;
    bool        verbose     = false;

    // Solver parameters (initialized to defaults)
    int          max_iterations  = DEFAULT_MAX_ITERATIONS;
    int          gmres_basis     = DEFAULT_GMRES_BASIS;
    int          mg_sweeps_pre   = DEFAULT_MG_SWEEPS_PRE;
    int          mg_sweeps_post  = DEFAULT_MG_SWEEPS_POST;
    scalar       newton_tol      = DEFAULT_NEWTON_TOL;
    scalar       tolerance       = DEFAULT_TOLERANCE;
    int          max_time_steps  = DEFAULT_MAX_TIME_STEPS;
    scalar       dt_inf          = DEFAULT_DT_INF;
    unsigned int noise_seed      = 0;
    scalar       D               = scalar( 1.0 );
    scalar       gamma           = scalar( 1.0 );
    scalar       noise_amplitude = DEFAULT_NOISE_AMPLITUDE;
    scalar       noise_frequency = DEFAULT_NOISE_FREQUENCY;

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

    app.add_flag( "--save-coords", save_coords, "Save numerical solutions to binary files" );
    app.add_flag( "--verbose", verbose, "Print per-iteration residuals and the profiler breakdown to the log" );
    app.add_option( "--max-iterations", max_iterations, "Maximum solver iterations" )->capture_default_str();
    app.add_option( "--gmres-basis", gmres_basis, "GMRES basis size" )->capture_default_str();
    app.add_option( "--mg-sweeps-pre", mg_sweeps_pre, "Multigrid pre-sweeps" )->capture_default_str();
    app.add_option( "--mg-sweeps-post", mg_sweeps_post, "Multigrid post-sweeps" )->capture_default_str();
    app.add_option( "--tolerance", tolerance, "Linear solver tolerance" )->capture_default_str();
    app.add_option( "--newton-tol", newton_tol, "Newton solver tolerance" )->capture_default_str();
    app.add_option( "--max-time-steps", max_time_steps, "Maximum number of time steps" )->capture_default_str();
    app.add_option( "--dt-inf", dt_inf, "dt_inf parameter (1/dt)" )->capture_default_str();
    app.add_option( "--noise-seed", noise_seed, "Random seed for Perlin noise" )->capture_default_str();
    app.add_option( "--noise-amplitude", noise_amplitude, "Amplitude of Perlin noise" )->capture_default_str();
    app.add_option( "--noise-frequency", noise_frequency, "Frequency scale factor for Perlin noise" )->capture_default_str();
    app.add_option( "--D", D, "Diffusion coefficient" )->capture_default_str();
    app.add_option( "--gamma", gamma, "Squared length of transition regions" )->capture_default_str();

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
    log.info( "Cahn-Hilliard Random-Init Solver Configuration (MPI)" );
#else
    log.info( "Cahn-Hilliard Random-Init Solver Configuration" );
#endif
    log.info( "========================================" );
    log.info( "" );
    log.info( "Problem Settings:" );
    log.info_f( "  Grid size:     %d x %d x %d", grid_size, grid_size, grid_size );
    log.info_f( "  Tensor dim:    %d", tensor_dim );
    log.info_f( "  Scalar type:   %s", scalar_label.c_str() );
    log.info_f( "  DOFs:          %lld", static_cast<long long>( grid_size ) * grid_size * grid_size * tensor_dim );
    log.info_f( "  Processes:     %d", comm_world.num_procs );
    log.info_f( "  Initial cond:  3D Perlin noise (amplitude %e, frequency %e, seed %u)",
               static_cast<double>( noise_amplitude ), static_cast<double>( noise_frequency ), noise_seed );
    log.info( "  RHS:           Zero" );
    log.info_f( "  D:             %e", static_cast<double>( D ) );
    log.info_f( "  gamma:         %e", static_cast<double>( gamma ) );
    log.info( "" );
    log.info( "Newton Solver:" );
    log.info_f( "  Tolerance:     %e", static_cast<double>( newton_tol ) );
    log.info( "" );
    log.info( "Linear Solver:" );
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
    if ( preconditioner_type == "mg" )
    {
        log.info_f( "  Pre-sweeps:    %d", mg_sweeps_pre );
        log.info_f( "  Post-sweeps:   %d", mg_sweeps_post );
        log.info( "  Direct coarse: false" );
    }
    log.info( "" );
    log.info( "Time Integration:" );
    log.info_f( "  Max time steps: %d", max_time_steps );
    log.info_f( "  dt_inf:         %e", static_cast<double>( dt_inf ) );
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

    // Boundary conditions of the WHOLE computational domain:
    //   left  = dirichlet (-1) on every axis, right = dirichlet (-1) on every axis  [psi, phi].
    //   -1 = dirichlet (value 0), +1 = neumann (derivative 0), 0 = periodic (reads opposite side).
    int global_left_bc[3][2]  = { { -1, -1 }, { -1, -1 }, { -1, -1 } };
    int global_right_bc[3][2] = { { -1, -1 }, { -1, -1 }, { -1, -1 } };

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

    auto vspace = std::make_shared<vec_ops_t>( range, comm_world, false, stencil, max_stencil_order );

    vector_t solution, rhs;
    vspace->init_vector( solution );
    vspace->init_vector( rhs );
    vspace->assign_scalar( 0.0, rhs );

    // Initialize solution with 3D Perlin noise
    {
        vector_view_t solution_view( solution, false );

        for ( int i = 0; i < range[0]; i++ )
        {
            for ( int j = 0; j < range[1]; j++ )
            {
                for ( int k = 0; k < range[2]; k++ )
                {
                    scalar x = step[0] * ( 0.5 + i + my_own_glob_rect.i1[0] );
                    scalar y = step[1] * ( 0.5 + j + my_own_glob_rect.i1[1] );
                    scalar z = step[2] * ( 0.5 + k + my_own_glob_rect.i1[2] );

                    scalar noise_val_psi = perlin_noise_3d<scalar>(
                        x * noise_frequency, y * noise_frequency, z * noise_frequency, noise_seed );
                    noise_val_psi *= noise_amplitude;
                    solution_view( i, j, k, 0 ) = noise_val_psi;

                    scalar noise_val_phi = perlin_noise_3d<scalar>(
                        x * noise_frequency, y * noise_frequency, z * noise_frequency, noise_seed + 1000 );
                    noise_val_phi *= noise_amplitude;
                    solution_view( i, j, k, 1 ) = noise_val_phi;
                }
            }
        }

        solution_view.release();
    }

    auto time_derivative = std::make_shared<time_derivative_t>( vspace );
    time_derivative->set_dt_inf( dt_inf );
    time_derivative->set_previous_state( solution );

    auto cahn_hilliard_jacobi_op = std::make_shared<jacobi_op_t>( vspace, step, cond, dist, time_derivative );
    auto cahn_hilliard_op        = std::make_shared<cahn_hilliard_op_t>(
        vspace, step, cond, dist, rhs, cahn_hilliard_jacobi_op, time_derivative );

    cahn_hilliard_op->set_mobility( mobility_t( D ) );
    cahn_hilliard_op->set_gamma( gamma );

    free_energy_t free_energy_calc( vspace, step, cond, dist, phobic_energy{}, cahn_hilliard_jacobi_op->get_gamma() );

    std::shared_ptr<precond_interface> precond;
    if ( preconditioner_type == "diag" )
    {
        precond = std::make_shared<smoother_t>( cahn_hilliard_jacobi_op, dist );
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

    // Log initial free energy (step 0)
    auto energies_init = free_energy_calc.compute( solution );
    log.info_f( "Step 0:" );
    log.info_f( "  Phobic energy: %e", static_cast<double>( energies_init.phobic ) );
    log.info_f( "  Philic energy: %e", static_cast<double>( energies_init.philic ) );

    // Save initial approximation (index 0) if requested (only valid for a single rank owning the whole domain)
    if ( save_coords && comm_world.num_procs == 1 )
    {
        std::string numerical_file = output_dir + "/numerical_0.bin";
        tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
    }

    // Solve and measure time for each time step
    std::vector<double> iteration_times;
    double              total_time_ms = 0.0;

    // Create solver instance based on type (will be reused for each time step)
    std::shared_ptr<linsolver_base_t> lin_solver;
    if ( solver_type == "jacobi" )
    {
        jacobi_solver::params solver_params;
        solver_params.monitor.rel_tol                  = tolerance;
        solver_params.monitor.max_iters_num            = max_iterations;
        lin_solver = std::make_shared<jacobi_solver>( cahn_hilliard_jacobi_op, vspace, &log, solver_params, precond );
    }
    else // gmres
    {
        gmres_solver::params params_gmres;
        params_gmres.monitor.rel_tol                      = tolerance;
        params_gmres.monitor.max_iters_num                = max_iterations;
        params_gmres.do_restart_on_false_ritz_convergence = true;
        params_gmres.basis_size                           = gmres_basis;
        params_gmres.batch_size                           = 1;
        params_gmres.preconditioner_side                  = 'L';
        params_gmres.reorthogonalization                  = true;
        lin_solver = std::make_shared<gmres_solver>( cahn_hilliard_jacobi_op, vspace, &log, params_gmres, precond );
    }

    auto newton_iteration = std::make_shared<newton_iteration_t>( vspace, lin_solver );

    auto newton_solver = std::make_shared<newton_solver_t>( vspace, &log, newton_iteration );
    newton_solver->convergence_strategy()->set_tolerance( newton_tol );

    for ( int step_idx = 0; step_idx < max_time_steps; step_idx++ )
    {
        log.info( "" );
        log.info_f( "Time iteration #%d has started", step_idx + 1 );
        log.info( "" );

        // Solve and measure time
        SCFD_PLATFORM_TIC( "Solve" );
        newton_solver->solve( cahn_hilliard_op.get(), nullptr, nullptr, solution );
        double step_time = current_prof::inst().toc( "Solve" );
        iteration_times.push_back( step_time );
        total_time_ms += step_time;

        // Compute norm of difference between solution and previous state
        vector_t previous_state;
        vspace->init_vector( previous_state );
        vspace->assign( time_derivative->get_previous_state(), previous_state );
        vector_t diff_prev;
        vspace->init_vector( diff_prev );
        vspace->assign_lin_comb( scalar( 1 ), solution, scalar( -1 ), previous_state, diff_prev );
        scalar diff_prev_norm = vspace->norm_l2( diff_prev );
        log.info_f( "||solution - previous_state||_2 = %le", static_cast<double>( diff_prev_norm ) );

        // Compute solution norm
        scalar solution_norm = vspace->norm_l2( solution );
        log.info_f( "||solution||_2 = %le", static_cast<double>( solution_norm ) );

        // Compute and log the free energy for this step
        auto energies = free_energy_calc.compute( solution );
        log.info_f( "  Phobic energy: %e", static_cast<double>( energies.phobic ) );
        log.info_f( "  Philic energy: %e", static_cast<double>( energies.philic ) );

        // Update previous step before the next step
        time_derivative->set_previous_state( solution );

        // Save numerical solution at each step if requested (only valid for a single rank owning the whole domain)
        if ( save_coords && comm_world.num_procs == 1 )
        {
            std::string numerical_file = output_dir + "/numerical_" + std::to_string( step_idx + 1 ) + ".bin";
            tests::save_solution_binary<vector_t, idx_nd_type>( solution, numerical_file, grid_size, tensor_dim );
        }

        // Separate iterations with empty line
        if ( step_idx < max_time_steps - 1 )
        {
            log.info( "" );
        }
    }

    log.info( "" );
    log.info( "========================================" );
    log.info( "Results" );
    log.info( "========================================" );
    scalar final_solution_norm = vspace->norm_l2( solution );
    log.info_f( "  ||solution||_2:             %e", static_cast<double>( final_solution_norm ) );
    log.info_f( "  Average iteration time:     %.2f ms", total_time_ms / max_time_steps );
    log.info_f( "  Total solve time:           %.2f ms", total_time_ms );
    log.info( "========================================" );

    if ( save_coords && comm_world.num_procs == 1 )
    {
        log.info( "" );
        log.info( "Saved solutions:" );
        log.info_f( "  Numerical: %s/numerical_*.bin", output_dir.c_str() );
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
