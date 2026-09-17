#ifndef __CAHN_HILLIARD_COMMON_H__
#define __CAHN_HILLIARD_COMMON_H__

#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <CLI/CLI.hpp>

#include <scfd/backend/backend.h>

#ifdef SCFD_BACKEND_ENABLE_MPI
#include <scfd/communication/mpi_binary_file.h>
#include <scfd/communication/mpi_rect_distributor.h>
#include <scfd/communication/mpi_wrap.h>
#else
#include <scfd/communication/rect_distributor.h>
#include <scfd/communication/trivial_binary_file.h>
#include <scfd/communication/trivial_comm.h>
#include <scfd/communication/trivial_platform.h>
#endif

#include <scfd/communication/rect_partitioner.h>
#include <scfd/utils/log.h>
#include <scfd/utils/profiling.h>

#if defined(PLATFORM_CUDA)
#include <scfd/utils/cuda_timer_event.h>
SCFD_GLOBAL_PROFILING(scfd::utils::cuda_timer_event)
#elif defined(SCFD_BACKEND_ENABLE_MPI)
#include <scfd/utils/mpi_timer_event.h>
SCFD_GLOBAL_PROFILING(scfd::utils::mpi_timer_event)
#else
#include <scfd/utils/system_timer_event.h>
SCFD_GLOBAL_PROFILING(scfd::utils::system_timer_event)
#endif

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
#include <nmfd/utils/logging.h>

#include "include/balancer.h"
#include "include/biharmonic_problem.h"
#include "include/boundary.h"
#include "include/cahn_hilliard_op.h"
#include "include/cahn_hilliard_problem.h"
#include "include/coarsening.h"
#include "include/error_monitor.h"
#include "include/free_energy.h"
#include "include/jacobi_op.h"
#include "include/jacobi_pre.h"
#include "include/kernels/mobility.h"
#include "include/kernels/phobic_energy.h"
#include "include/perlin_noise.h"
#include "include/prolongator.h"
#include "include/restrictor.h"
#include "include/scheduler.h"
#include "include/solution_io.h"
#include "include/time_derivative.h"

using backend = scfd::backend::current;

constexpr int dim               = 3;
constexpr int tensor_dim        = 2;
// The multigrid restriction stencil reaches two cells past the block and spans the full diagonal,
// so the halo has to be two cells wide and exchanged with the corner neighbours as well.
constexpr int stencil           = 2;   // ghost width per side; must match the distributor stencil
constexpr int max_stencil_order = dim; // highest coupled stencil order for the halo exchange

#ifndef USE_DOUBLE_PRECISION
using scalar = float;
#else
using scalar = double;
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
template <class T> using binary_file_t = scfd::communication::mpi_binary_file<T>;
#else
using comm_platform_t = scfd::communication::trivial_platform<mem_t>;
using comm_info_t     = scfd::communication::trivial_comm<mem_t>;
using dist_t          = scfd::communication::rect_distributor<scalar, dim, mem_t, dist_for_each_t, ord_t, big_ord_t, comm_info_t>;
template <class T> using binary_file_t = scfd::communication::trivial_binary_file<T>;
#endif

using part_t = scfd::communication::rect_partitioner<dim, ord_t, big_ord_t, comm_info_t>;

using big_idx_t        = scfd::static_vec::vec<big_ord_t, dim>;
using periodic_flags_t = scfd::static_vec::vec<bool, dim>;
using rect_t           = scfd::static_vec::rect<ord_t, dim>;
using big_rect_t       = scfd::static_vec::rect<big_ord_t, dim>;

using vec_ops_t     = nmfd::rect_vector_space<scalar, /*dim=*/dim, /*tensor_dim=*/tensor_dim, backend, comm_info_t>;
using vector_t      = typename vec_ops_t::vector_type;
using tensor_t      = scfd::static_vec::vec<scalar, tensor_dim>;
using vector_view_t = typename vector_t::view_type;

using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>;
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

#endif // __CAHN_HILLIARD_COMMON_H__
