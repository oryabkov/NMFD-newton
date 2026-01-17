#include <memory>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <type_traits>
#include <filesystem>

#include <scfd/utils/log.h>
#include <scfd/backend/backend.h>
#include <nmfd/operations/rect_vector_space.h>
#include <nmfd/preconditioners/mg.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/default_monitor.h>
// #include <nmfd/solvers/jacobi.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>

#include "cahn_hilliard_op.h"
#include "cahn_hilliard_jacobi_op.h"
#include "cahn_hilliard_jacobi_pre.h"
// #include "jacobi_pre.h"
#include "solver_logger.h"
#include "prolongator.h"
#include "restrictor.h"
#include "coarsening.h"
#include "identity_op.h"

#include "include/boundary.h"
#include "error_monitor.h"

struct backend
{
    using memory_type       = scfd::backend::memory;
    template <class Ordinal = int>
    using for_each_type     = scfd::backend::template for_each<Ordinal>;
    template <int Dim, class Ordinal = int>
    using for_each_nd_type  = scfd::backend::template for_each_nd<Dim, Ordinal>;
    using reduce_type       = scfd::backend::reduce;
};

/**************************************/


constexpr int dim = 3;
constexpr int tensor_dim = 2;

using scalar = double;
using grid_step_type    = scfd::static_vec::vec<scalar, dim>;
using idx_nd_type       = scfd::static_vec::vec<int   , dim>;

using log_t             = scfd::utils::log_std;

using vec_ops_t         = nmfd::rect_vector_space<scalar,/*dim=*/dim,/*tensor_dim=*/tensor_dim, backend>;

// Monitors
using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>;
using error_monitor_t = tests::error_monitor<vec_ops_t, log_t>;
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

// MG
using prolongator_t             = tests::prolongator<vec_ops_t, log_t>;
using restrictor_t              = tests::restrictor<vec_ops_t, log_t>;
using cahn_hilliard_jacobi_op_t = tests::cahn_hilliard_jacobi_op<vec_ops_t, log_t>;
using ident_op_t                = tests::identity_op<cahn_hilliard_jacobi_op_t, vec_ops_t, log_t>;
using smoother_t                = tests::cahn_hilliard_jacobi_pre<vec_ops_t, log_t>;
using coarsening_t              = tests::coarsening<cahn_hilliard_jacobi_op_t, log_t>;

using precond_interface = nmfd::preconditioners::preconditioner_interface<vec_ops_t, cahn_hilliard_jacobi_op_t>;

using mg_t = nmfd::preconditioners::mg
<
    cahn_hilliard_jacobi_op_t, restrictor_t, prolongator_t,
    smoother_t, ident_op_t, coarsening_t,
    log_t
>;
using mg_params_t = mg_t::params_hierarchy;
using mg_utils_t  = mg_t::utils_hierarchy;


using gmres_solver  = nmfd::solvers::gmres<vec_ops_t, krylov_monitor_t, log_t, cahn_hilliard_jacobi_op_t, precond_interface>;

using vector_t         = typename vec_ops_t::vector_type;

using tensor_t         = scfd::static_vec::vec<scalar, tensor_dim>;
using vector_view_t    = typename vector_t::view_type;

// Newton
using cahn_hilliard_op_t = tests::cahn_hilliard_op<vec_ops_t, cahn_hilliard_jacobi_op_t, log_t>;
using newton_iteration_t = nmfd::solvers::newton_iteration<vec_ops_t, cahn_hilliard_op_t, gmres_solver>;
using newton_solver_t = nmfd::solvers::nonlinear_solver<vec_ops_t, log_t, cahn_hilliard_op_t, newton_iteration_t, error_monitor_t>;

/**************************************/

/* Example 0 */


tensor_t u(scalar x, scalar y, scalar z)
{
    tensor_t res {0., 0.};
    scalar g = x*(1-x)*y*(1-y)*z*(1-z);

    // phi
    res[1] = g * (
        -1
        - x - y - z
        + pow(x, 2) - x*y - x*z + pow(y, 2) - y*z + pow(z, 2)
        + pow(x, 2)*y + pow(x, 2)*z + x*pow(y, 2) - x*y*z + x*pow(z, 2) + pow(y, 2)*z + y*pow(z, 2)
        - pow(x, 2)*pow(y, 2) + pow(x, 2)*y*z - pow(x, 2)*pow(z, 2) + x*pow(y, 2)*z + x*y*pow(z, 2) - pow(y, 2)*pow(z, 2)
        - pow(x, 2)*pow(y, 2)*z - pow(x, 2)*y*pow(z, 2) - x*pow(y, 2)*pow(z, 2)
        + pow(x, 2)*pow(y, 2)*pow(z, 2)
    );

    // psi
    res[0] = g * (
    -35
        - 23*x - 23*y - 23*z
        + 23*pow(x, 2) - 11*x*y - 11*x*z + 23*pow(y, 2) - 11*y*z + 23*pow(z, 2)
        + 11*pow(x, 2)*y + 11*pow(x, 2)*z + 11*x*pow(y, 2) + x*y*z + 11*x*pow(z, 2) + 11*pow(y, 2)*z + 11*y*pow(z, 2)
        - 11*pow(x, 2)*pow(y, 2) - pow(x, 2)*y*z - 11*pow(x, 2)*pow(z, 2) - x*pow(y, 2)*z - x*y*pow(z, 2) - 11*pow(y, 2)*pow(z, 2)
        + pow(x, 2)*pow(y, 2)*z + pow(x, 2)*y*pow(z, 2) + x*pow(y, 2)*pow(z, 2)
        - 2*pow(x, 2)*pow(y, 2)*pow(z, 2)
        - pow(x, 3)*pow(y, 2)*pow(z, 2) - pow(x, 2)*pow(y, 3)*pow(z, 2) - pow(x, 2)*pow(y, 2)*pow(z, 3)
        + 5*pow(x, 4)*pow(y, 2)*pow(z, 2) - pow(x, 3)*pow(y, 3)*pow(z, 2) - pow(x, 3)*pow(y, 2)*pow(z, 3) + 5*pow(x, 2)*pow(y, 4)*pow(z, 2) - pow(x, 2)*pow(y, 3)*pow(z, 3) + 5*pow(x, 2)*pow(y, 2)*pow(z, 4)
        + 2*pow(x, 5)*pow(y, 2)*pow(z, 2) + 5*pow(x, 4)*pow(y, 3)*pow(z, 2) + 5*pow(x, 4)*pow(y, 2)*pow(z, 3) + 5*pow(x, 3)*pow(y, 4)*pow(z, 2) - pow(x, 3)*pow(y, 3)*pow(z, 3) + 5*pow(x, 3)*pow(y, 2)*pow(z, 4) + 2*pow(x, 2)*pow(y, 5)*pow(z, 2) + 5*pow(x, 2)*pow(y, 4)*pow(z, 3) + 5*pow(x, 2)*pow(y, 3)*pow(z, 4) + 2*pow(x, 2)*pow(y, 2)*pow(z, 5)
        - 10*pow(x, 6)*pow(y, 2)*pow(z, 2) + 2*pow(x, 5)*pow(y, 3)*pow(z, 2) + 2*pow(x, 5)*pow(y, 2)*pow(z, 3) - 25*pow(x, 4)*pow(y, 4)*pow(z, 2) + 5*pow(x, 4)*pow(y, 3)*pow(z, 3) - 25*pow(x, 4)*pow(y, 2)*pow(z, 4) + 2*pow(x, 3)*pow(y, 5)*pow(z, 2) + 5*pow(x, 3)*pow(y, 4)*pow(z, 3) + 5*pow(x, 3)*pow(y, 3)*pow(z, 4) + 2*pow(x, 3)*pow(y, 2)*pow(z, 5) - 10*pow(x, 2)*pow(y, 6)*pow(z, 2) + 2*pow(x, 2)*pow(y, 5)*pow(z, 3) - 25*pow(x, 2)*pow(y, 4)*pow(z, 4) + 2*pow(x, 2)*pow(y, 3)*pow(z, 5) - 10*pow(x, 2)*pow(y, 2)*pow(z, 6)
        + 2*pow(x, 7)*pow(y, 2)*pow(z, 2) - 10*pow(x, 6)*pow(y, 3)*pow(z, 2) - 10*pow(x, 6)*pow(y, 2)*pow(z, 3) - 10*pow(x, 5)*pow(y, 4)*pow(z, 2) + 2*pow(x, 5)*pow(y, 3)*pow(z, 3) - 10*pow(x, 5)*pow(y, 2)*pow(z, 4) - 10*pow(x, 4)*pow(y, 5)*pow(z, 2) - 25*pow(x, 4)*pow(y, 4)*pow(z, 3) - 25*pow(x, 4)*pow(y, 3)*pow(z, 4) - 10*pow(x, 4)*pow(y, 2)*pow(z, 5) - 10*pow(x, 3)*pow(y, 6)*pow(z, 2) + 2*pow(x, 3)*pow(y, 5)*pow(z, 3) - 25*pow(x, 3)*pow(y, 4)*pow(z, 4) + 2*pow(x, 3)*pow(y, 3)*pow(z, 5) - 10*pow(x, 3)*pow(y, 2)*pow(z, 6) + 2*pow(x, 2)*pow(y, 7)*pow(z, 2) - 10*pow(x, 2)*pow(y, 6)*pow(z, 3) - 10*pow(x, 2)*pow(y, 5)*pow(z, 4) - 10*pow(x, 2)*pow(y, 4)*pow(z, 5) - 10*pow(x, 2)*pow(y, 3)*pow(z, 6) + 2*pow(x, 2)*pow(y, 2)*pow(z, 7)
        + 7*pow(x, 8)*pow(y, 2)*pow(z, 2) + 2*pow(x, 7)*pow(y, 3)*pow(z, 2) + 2*pow(x, 7)*pow(y, 2)*pow(z, 3) + 50*pow(x, 6)*pow(y, 4)*pow(z, 2) - 10*pow(x, 6)*pow(y, 3)*pow(z, 3) + 50*pow(x, 6)*pow(y, 2)*pow(z, 4) - 4*pow(x, 5)*pow(y, 5)*pow(z, 2) - 10*pow(x, 5)*pow(y, 4)*pow(z, 3) - 10*pow(x, 5)*pow(y, 3)*pow(z, 4) - 4*pow(x, 5)*pow(y, 2)*pow(z, 5) + 50*pow(x, 4)*pow(y, 6)*pow(z, 2) - 10*pow(x, 4)*pow(y, 5)*pow(z, 3) + 125*pow(x, 4)*pow(y, 4)*pow(z, 4) - 10*pow(x, 4)*pow(y, 3)*pow(z, 5) + 50*pow(x, 4)*pow(y, 2)*pow(z, 6) + 2*pow(x, 3)*pow(y, 7)*pow(z, 2) - 10*pow(x, 3)*pow(y, 6)*pow(z, 3) - 10*pow(x, 3)*pow(y, 5)*pow(z, 4) - 10*pow(x, 3)*pow(y, 4)*pow(z, 5) - 10*pow(x, 3)*pow(y, 3)*pow(z, 6) + 2*pow(x, 3)*pow(y, 2)*pow(z, 7) + 7*pow(x, 2)*pow(y, 8)*pow(z, 2) + 2*pow(x, 2)*pow(y, 7)*pow(z, 3) + 50*pow(x, 2)*pow(y, 6)*pow(z, 4) - 4*pow(x, 2)*pow(y, 5)*pow(z, 5) + 50*pow(x, 2)*pow(y, 4)*pow(z, 6) + 2*pow(x, 2)*pow(y, 3)*pow(z, 7) + 7*pow(x, 2)*pow(y, 2)*pow(z, 8)
        - 5*pow(x, 9)*pow(y, 2)*pow(z, 2) + 7*pow(x, 8)*pow(y, 3)*pow(z, 2) + 7*pow(x, 8)*pow(y, 2)*pow(z, 3) - 10*pow(x, 7)*pow(y, 4)*pow(z, 2) + 2*pow(x, 7)*pow(y, 3)*pow(z, 3) - 10*pow(x, 7)*pow(y, 2)*pow(z, 4) + 20*pow(x, 6)*pow(y, 5)*pow(z, 2) + 50*pow(x, 6)*pow(y, 4)*pow(z, 3) + 50*pow(x, 6)*pow(y, 3)*pow(z, 4) + 20*pow(x, 6)*pow(y, 2)*pow(z, 5) + 20*pow(x, 5)*pow(y, 6)*pow(z, 2) - 4*pow(x, 5)*pow(y, 5)*pow(z, 3) + 50*pow(x, 5)*pow(y, 4)*pow(z, 4) - 4*pow(x, 5)*pow(y, 3)*pow(z, 5) + 20*pow(x, 5)*pow(y, 2)*pow(z, 6) - 10*pow(x, 4)*pow(y, 7)*pow(z, 2) + 50*pow(x, 4)*pow(y, 6)*pow(z, 3) + 50*pow(x, 4)*pow(y, 5)*pow(z, 4) + 50*pow(x, 4)*pow(y, 4)*pow(z, 5) + 50*pow(x, 4)*pow(y, 3)*pow(z, 6) - 10*pow(x, 4)*pow(y, 2)*pow(z, 7) + 7*pow(x, 3)*pow(y, 8)*pow(z, 2) + 2*pow(x, 3)*pow(y, 7)*pow(z, 3) + 50*pow(x, 3)*pow(y, 6)*pow(z, 4) - 4*pow(x, 3)*pow(y, 5)*pow(z, 5) + 50*pow(x, 3)*pow(y, 4)*pow(z, 6) + 2*pow(x, 3)*pow(y, 3)*pow(z, 7) + 7*pow(x, 3)*pow(y, 2)*pow(z, 8) - 5*pow(x, 2)*pow(y, 9)*pow(z, 2) + 7*pow(x, 2)*pow(y, 8)*pow(z, 3) - 10*pow(x, 2)*pow(y, 7)*pow(z, 4) + 20*pow(x, 2)*pow(y, 6)*pow(z, 5) + 20*pow(x, 2)*pow(y, 5)*pow(z, 6) - 10*pow(x, 2)*pow(y, 4)*pow(z, 7) + 7*pow(x, 2)*pow(y, 3)*pow(z, 8) - 5*pow(x, 2)*pow(y, 2)*pow(z, 9)
        + pow(x, 10)*pow(y, 2)*pow(z, 2) - 5*pow(x, 9)*pow(y, 3)*pow(z, 2) - 5*pow(x, 9)*pow(y, 2)*pow(z, 3) - 35*pow(x, 8)*pow(y, 4)*pow(z, 2) + 7*pow(x, 8)*pow(y, 3)*pow(z, 3) - 35*pow(x, 8)*pow(y, 2)*pow(z, 4) - 4*pow(x, 7)*pow(y, 5)*pow(z, 2) - 10*pow(x, 7)*pow(y, 4)*pow(z, 3) - 10*pow(x, 7)*pow(y, 3)*pow(z, 4) - 4*pow(x, 7)*pow(y, 2)*pow(z, 5) - 100*pow(x, 6)*pow(y, 6)*pow(z, 2) + 20*pow(x, 6)*pow(y, 5)*pow(z, 3) - 250*pow(x, 6)*pow(y, 4)*pow(z, 4) + 20*pow(x, 6)*pow(y, 3)*pow(z, 5) - 100*pow(x, 6)*pow(y, 2)*pow(z, 6) - 4*pow(x, 5)*pow(y, 7)*pow(z, 2) + 20*pow(x, 5)*pow(y, 6)*pow(z, 3) + 20*pow(x, 5)*pow(y, 5)*pow(z, 4) + 20*pow(x, 5)*pow(y, 4)*pow(z, 5) + 20*pow(x, 5)*pow(y, 3)*pow(z, 6) - 4*pow(x, 5)*pow(y, 2)*pow(z, 7) - 35*pow(x, 4)*pow(y, 8)*pow(z, 2) - 10*pow(x, 4)*pow(y, 7)*pow(z, 3) - 250*pow(x, 4)*pow(y, 6)*pow(z, 4) + 20*pow(x, 4)*pow(y, 5)*pow(z, 5) - 250*pow(x, 4)*pow(y, 4)*pow(z, 6) - 10*pow(x, 4)*pow(y, 3)*pow(z, 7) - 35*pow(x, 4)*pow(y, 2)*pow(z, 8) - 5*pow(x, 3)*pow(y, 9)*pow(z, 2) + 7*pow(x, 3)*pow(y, 8)*pow(z, 3) - 10*pow(x, 3)*pow(y, 7)*pow(z, 4) + 20*pow(x, 3)*pow(y, 6)*pow(z, 5) + 20*pow(x, 3)*pow(y, 5)*pow(z, 6) - 10*pow(x, 3)*pow(y, 4)*pow(z, 7) + 7*pow(x, 3)*pow(y, 3)*pow(z, 8) - 5*pow(x, 3)*pow(y, 2)*pow(z, 9) + pow(x, 2)*pow(y, 10)*pow(z, 2) - 5*pow(x, 2)*pow(y, 9)*pow(z, 3) - 35*pow(x, 2)*pow(y, 8)*pow(z, 4) - 4*pow(x, 2)*pow(y, 7)*pow(z, 5) - 100*pow(x, 2)*pow(y, 6)*pow(z, 6) - 4*pow(x, 2)*pow(y, 5)*pow(z, 7) - 35*pow(x, 2)*pow(y, 4)*pow(z, 8) - 5*pow(x, 2)*pow(y, 3)*pow(z, 9) + pow(x, 2)*pow(y, 2)*pow(z, 10)
        + pow(x, 10)*pow(y, 3)*pow(z, 2) + pow(x, 10)*pow(y, 2)*pow(z, 3) + 25*pow(x, 9)*pow(y, 4)*pow(z, 2) - 5*pow(x, 9)*pow(y, 3)*pow(z, 3) + 25*pow(x, 9)*pow(y, 2)*pow(z, 4) - 14*pow(x, 8)*pow(y, 5)*pow(z, 2) - 35*pow(x, 8)*pow(y, 4)*pow(z, 3) - 35*pow(x, 8)*pow(y, 3)*pow(z, 4) - 14*pow(x, 8)*pow(y, 2)*pow(z, 5) + 20*pow(x, 7)*pow(y, 6)*pow(z, 2) - 4*pow(x, 7)*pow(y, 5)*pow(z, 3) + 50*pow(x, 7)*pow(y, 4)*pow(z, 4) - 4*pow(x, 7)*pow(y, 3)*pow(z, 5) + 20*pow(x, 7)*pow(y, 2)*pow(z, 6) + 20*pow(x, 6)*pow(y, 7)*pow(z, 2) - 100*pow(x, 6)*pow(y, 6)*pow(z, 3) - 100*pow(x, 6)*pow(y, 5)*pow(z, 4) - 100*pow(x, 6)*pow(y, 4)*pow(z, 5) - 100*pow(x, 6)*pow(y, 3)*pow(z, 6) + 20*pow(x, 6)*pow(y, 2)*pow(z, 7) - 14*pow(x, 5)*pow(y, 8)*pow(z, 2) - 4*pow(x, 5)*pow(y, 7)*pow(z, 3) - 100*pow(x, 5)*pow(y, 6)*pow(z, 4) + 8*pow(x, 5)*pow(y, 5)*pow(z, 5) - 100*pow(x, 5)*pow(y, 4)*pow(z, 6) - 4*pow(x, 5)*pow(y, 3)*pow(z, 7) - 14*pow(x, 5)*pow(y, 2)*pow(z, 8) + 25*pow(x, 4)*pow(y, 9)*pow(z, 2) - 35*pow(x, 4)*pow(y, 8)*pow(z, 3) + 50*pow(x, 4)*pow(y, 7)*pow(z, 4) - 100*pow(x, 4)*pow(y, 6)*pow(z, 5) - 100*pow(x, 4)*pow(y, 5)*pow(z, 6) + 50*pow(x, 4)*pow(y, 4)*pow(z, 7) - 35*pow(x, 4)*pow(y, 3)*pow(z, 8) + 25*pow(x, 4)*pow(y, 2)*pow(z, 9) + pow(x, 3)*pow(y, 10)*pow(z, 2) - 5*pow(x, 3)*pow(y, 9)*pow(z, 3) - 35*pow(x, 3)*pow(y, 8)*pow(z, 4) - 4*pow(x, 3)*pow(y, 7)*pow(z, 5) - 100*pow(x, 3)*pow(y, 6)*pow(z, 6) - 4*pow(x, 3)*pow(y, 5)*pow(z, 7) - 35*pow(x, 3)*pow(y, 4)*pow(z, 8) - 5*pow(x, 3)*pow(y, 3)*pow(z, 9) + pow(x, 3)*pow(y, 2)*pow(z, 10) + pow(x, 2)*pow(y, 10)*pow(z, 3) + 25*pow(x, 2)*pow(y, 9)*pow(z, 4) - 14*pow(x, 2)*pow(y, 8)*pow(z, 5) + 20*pow(x, 2)*pow(y, 7)*pow(z, 6) + 20*pow(x, 2)*pow(y, 6)*pow(z, 7) - 14*pow(x, 2)*pow(y, 5)*pow(z, 8) + 25*pow(x, 2)*pow(y, 4)*pow(z, 9) + pow(x, 2)*pow(y, 3)*pow(z, 10)
        - 5*pow(x, 10)*pow(y, 4)*pow(z, 2) + pow(x, 10)*pow(y, 3)*pow(z, 3) - 5*pow(x, 10)*pow(y, 2)*pow(z, 4) + 10*pow(x, 9)*pow(y, 5)*pow(z, 2) + 25*pow(x, 9)*pow(y, 4)*pow(z, 3) + 25*pow(x, 9)*pow(y, 3)*pow(z, 4) + 10*pow(x, 9)*pow(y, 2)*pow(z, 5) + 70*pow(x, 8)*pow(y, 6)*pow(z, 2) - 14*pow(x, 8)*pow(y, 5)*pow(z, 3) + 175*pow(x, 8)*pow(y, 4)*pow(z, 4) - 14*pow(x, 8)*pow(y, 3)*pow(z, 5) + 70*pow(x, 8)*pow(y, 2)*pow(z, 6) - 4*pow(x, 7)*pow(y, 7)*pow(z, 2) + 20*pow(x, 7)*pow(y, 6)*pow(z, 3) + 20*pow(x, 7)*pow(y, 5)*pow(z, 4) + 20*pow(x, 7)*pow(y, 4)*pow(z, 5) + 20*pow(x, 7)*pow(y, 3)*pow(z, 6) - 4*pow(x, 7)*pow(y, 2)*pow(z, 7) + 70*pow(x, 6)*pow(y, 8)*pow(z, 2) + 20*pow(x, 6)*pow(y, 7)*pow(z, 3) + 500*pow(x, 6)*pow(y, 6)*pow(z, 4) - 40*pow(x, 6)*pow(y, 5)*pow(z, 5) + 500*pow(x, 6)*pow(y, 4)*pow(z, 6) + 20*pow(x, 6)*pow(y, 3)*pow(z, 7) + 70*pow(x, 6)*pow(y, 2)*pow(z, 8) + 10*pow(x, 5)*pow(y, 9)*pow(z, 2) - 14*pow(x, 5)*pow(y, 8)*pow(z, 3) + 20*pow(x, 5)*pow(y, 7)*pow(z, 4) - 40*pow(x, 5)*pow(y, 6)*pow(z, 5) - 40*pow(x, 5)*pow(y, 5)*pow(z, 6) + 20*pow(x, 5)*pow(y, 4)*pow(z, 7) - 14*pow(x, 5)*pow(y, 3)*pow(z, 8) + 10*pow(x, 5)*pow(y, 2)*pow(z, 9) - 5*pow(x, 4)*pow(y, 10)*pow(z, 2) + 25*pow(x, 4)*pow(y, 9)*pow(z, 3) + 175*pow(x, 4)*pow(y, 8)*pow(z, 4) + 20*pow(x, 4)*pow(y, 7)*pow(z, 5) + 500*pow(x, 4)*pow(y, 6)*pow(z, 6) + 20*pow(x, 4)*pow(y, 5)*pow(z, 7) + 175*pow(x, 4)*pow(y, 4)*pow(z, 8) + 25*pow(x, 4)*pow(y, 3)*pow(z, 9) - 5*pow(x, 4)*pow(y, 2)*pow(z, 10) + pow(x, 3)*pow(y, 10)*pow(z, 3) + 25*pow(x, 3)*pow(y, 9)*pow(z, 4) - 14*pow(x, 3)*pow(y, 8)*pow(z, 5) + 20*pow(x, 3)*pow(y, 7)*pow(z, 6) + 20*pow(x, 3)*pow(y, 6)*pow(z, 7) - 14*pow(x, 3)*pow(y, 5)*pow(z, 8) + 25*pow(x, 3)*pow(y, 4)*pow(z, 9) + pow(x, 3)*pow(y, 3)*pow(z, 10) - 5*pow(x, 2)*pow(y, 10)*pow(z, 4) + 10*pow(x, 2)*pow(y, 9)*pow(z, 5) + 70*pow(x, 2)*pow(y, 8)*pow(z, 6) - 4*pow(x, 2)*pow(y, 7)*pow(z, 7) + 70*pow(x, 2)*pow(y, 6)*pow(z, 8) + 10*pow(x, 2)*pow(y, 5)*pow(z, 9) - 5*pow(x, 2)*pow(y, 4)*pow(z, 10)
        - 2*pow(x, 10)*pow(y, 5)*pow(z, 2) - 5*pow(x, 10)*pow(y, 4)*pow(z, 3) - 5*pow(x, 10)*pow(y, 3)*pow(z, 4) - 2*pow(x, 10)*pow(y, 2)*pow(z, 5) - 50*pow(x, 9)*pow(y, 6)*pow(z, 2) + 10*pow(x, 9)*pow(y, 5)*pow(z, 3) - 125*pow(x, 9)*pow(y, 4)*pow(z, 4) + 10*pow(x, 9)*pow(y, 3)*pow(z, 5) - 50*pow(x, 9)*pow(y, 2)*pow(z, 6) - 14*pow(x, 8)*pow(y, 7)*pow(z, 2) + 70*pow(x, 8)*pow(y, 6)*pow(z, 3) + 70*pow(x, 8)*pow(y, 5)*pow(z, 4) + 70*pow(x, 8)*pow(y, 4)*pow(z, 5) + 70*pow(x, 8)*pow(y, 3)*pow(z, 6) - 14*pow(x, 8)*pow(y, 2)*pow(z, 7) - 14*pow(x, 7)*pow(y, 8)*pow(z, 2) - 4*pow(x, 7)*pow(y, 7)*pow(z, 3) - 100*pow(x, 7)*pow(y, 6)*pow(z, 4) + 8*pow(x, 7)*pow(y, 5)*pow(z, 5) - 100*pow(x, 7)*pow(y, 4)*pow(z, 6) - 4*pow(x, 7)*pow(y, 3)*pow(z, 7) - 14*pow(x, 7)*pow(y, 2)*pow(z, 8) - 50*pow(x, 6)*pow(y, 9)*pow(z, 2) + 70*pow(x, 6)*pow(y, 8)*pow(z, 3) - 100*pow(x, 6)*pow(y, 7)*pow(z, 4) + 200*pow(x, 6)*pow(y, 6)*pow(z, 5) + 200*pow(x, 6)*pow(y, 5)*pow(z, 6) - 100*pow(x, 6)*pow(y, 4)*pow(z, 7) + 70*pow(x, 6)*pow(y, 3)*pow(z, 8) - 50*pow(x, 6)*pow(y, 2)*pow(z, 9) - 2*pow(x, 5)*pow(y, 10)*pow(z, 2) + 10*pow(x, 5)*pow(y, 9)*pow(z, 3) + 70*pow(x, 5)*pow(y, 8)*pow(z, 4) + 8*pow(x, 5)*pow(y, 7)*pow(z, 5) + 200*pow(x, 5)*pow(y, 6)*pow(z, 6) + 8*pow(x, 5)*pow(y, 5)*pow(z, 7) + 70*pow(x, 5)*pow(y, 4)*pow(z, 8) + 10*pow(x, 5)*pow(y, 3)*pow(z, 9) - 2*pow(x, 5)*pow(y, 2)*pow(z, 10) - 5*pow(x, 4)*pow(y, 10)*pow(z, 3) - 125*pow(x, 4)*pow(y, 9)*pow(z, 4) + 70*pow(x, 4)*pow(y, 8)*pow(z, 5) - 100*pow(x, 4)*pow(y, 7)*pow(z, 6) - 100*pow(x, 4)*pow(y, 6)*pow(z, 7) + 70*pow(x, 4)*pow(y, 5)*pow(z, 8) - 125*pow(x, 4)*pow(y, 4)*pow(z, 9) - 5*pow(x, 4)*pow(y, 3)*pow(z, 10) - 5*pow(x, 3)*pow(y, 10)*pow(z, 4) + 10*pow(x, 3)*pow(y, 9)*pow(z, 5) + 70*pow(x, 3)*pow(y, 8)*pow(z, 6) - 4*pow(x, 3)*pow(y, 7)*pow(z, 7) + 70*pow(x, 3)*pow(y, 6)*pow(z, 8) + 10*pow(x, 3)*pow(y, 5)*pow(z, 9) - 5*pow(x, 3)*pow(y, 4)*pow(z, 10) - 2*pow(x, 2)*pow(y, 10)*pow(z, 5) - 50*pow(x, 2)*pow(y, 9)*pow(z, 6) - 14*pow(x, 2)*pow(y, 8)*pow(z, 7) - 14*pow(x, 2)*pow(y, 7)*pow(z, 8) - 50*pow(x, 2)*pow(y, 6)*pow(z, 9) - 2*pow(x, 2)*pow(y, 5)*pow(z, 10)
        + 10*pow(x, 10)*pow(y, 6)*pow(z, 2) - 2*pow(x, 10)*pow(y, 5)*pow(z, 3) + 25*pow(x, 10)*pow(y, 4)*pow(z, 4) - 2*pow(x, 10)*pow(y, 3)*pow(z, 5) + 10*pow(x, 10)*pow(y, 2)*pow(z, 6) + 10*pow(x, 9)*pow(y, 7)*pow(z, 2) - 50*pow(x, 9)*pow(y, 6)*pow(z, 3) - 50*pow(x, 9)*pow(y, 5)*pow(z, 4) - 50*pow(x, 9)*pow(y, 4)*pow(z, 5) - 50*pow(x, 9)*pow(y, 3)*pow(z, 6) + 10*pow(x, 9)*pow(y, 2)*pow(z, 7) - 49*pow(x, 8)*pow(y, 8)*pow(z, 2) - 14*pow(x, 8)*pow(y, 7)*pow(z, 3) - 350*pow(x, 8)*pow(y, 6)*pow(z, 4) + 28*pow(x, 8)*pow(y, 5)*pow(z, 5) - 350*pow(x, 8)*pow(y, 4)*pow(z, 6) - 14*pow(x, 8)*pow(y, 3)*pow(z, 7) - 49*pow(x, 8)*pow(y, 2)*pow(z, 8) + 10*pow(x, 7)*pow(y, 9)*pow(z, 2) - 14*pow(x, 7)*pow(y, 8)*pow(z, 3) + 20*pow(x, 7)*pow(y, 7)*pow(z, 4) - 40*pow(x, 7)*pow(y, 6)*pow(z, 5) - 40*pow(x, 7)*pow(y, 5)*pow(z, 6) + 20*pow(x, 7)*pow(y, 4)*pow(z, 7) - 14*pow(x, 7)*pow(y, 3)*pow(z, 8) + 10*pow(x, 7)*pow(y, 2)*pow(z, 9) + 10*pow(x, 6)*pow(y, 10)*pow(z, 2) - 50*pow(x, 6)*pow(y, 9)*pow(z, 3) - 350*pow(x, 6)*pow(y, 8)*pow(z, 4) - 40*pow(x, 6)*pow(y, 7)*pow(z, 5) - 1000*pow(x, 6)*pow(y, 6)*pow(z, 6) - 40*pow(x, 6)*pow(y, 5)*pow(z, 7) - 350*pow(x, 6)*pow(y, 4)*pow(z, 8) - 50*pow(x, 6)*pow(y, 3)*pow(z, 9) + 10*pow(x, 6)*pow(y, 2)*pow(z, 10) - 2*pow(x, 5)*pow(y, 10)*pow(z, 3) - 50*pow(x, 5)*pow(y, 9)*pow(z, 4) + 28*pow(x, 5)*pow(y, 8)*pow(z, 5) - 40*pow(x, 5)*pow(y, 7)*pow(z, 6) - 40*pow(x, 5)*pow(y, 6)*pow(z, 7) + 28*pow(x, 5)*pow(y, 5)*pow(z, 8) - 50*pow(x, 5)*pow(y, 4)*pow(z, 9) - 2*pow(x, 5)*pow(y, 3)*pow(z, 10) + 25*pow(x, 4)*pow(y, 10)*pow(z, 4) - 50*pow(x, 4)*pow(y, 9)*pow(z, 5) - 350*pow(x, 4)*pow(y, 8)*pow(z, 6) + 20*pow(x, 4)*pow(y, 7)*pow(z, 7) - 350*pow(x, 4)*pow(y, 6)*pow(z, 8) - 50*pow(x, 4)*pow(y, 5)*pow(z, 9) + 25*pow(x, 4)*pow(y, 4)*pow(z, 10) - 2*pow(x, 3)*pow(y, 10)*pow(z, 5) - 50*pow(x, 3)*pow(y, 9)*pow(z, 6) - 14*pow(x, 3)*pow(y, 8)*pow(z, 7) - 14*pow(x, 3)*pow(y, 7)*pow(z, 8) - 50*pow(x, 3)*pow(y, 6)*pow(z, 9) - 2*pow(x, 3)*pow(y, 5)*pow(z, 10) + 10*pow(x, 2)*pow(y, 10)*pow(z, 6) + 10*pow(x, 2)*pow(y, 9)*pow(z, 7) - 49*pow(x, 2)*pow(y, 8)*pow(z, 8) + 10*pow(x, 2)*pow(y, 7)*pow(z, 9) + 10*pow(x, 2)*pow(y, 6)*pow(z, 10)
        - 2*pow(x, 10)*pow(y, 7)*pow(z, 2) + 10*pow(x, 10)*pow(y, 6)*pow(z, 3) + 10*pow(x, 10)*pow(y, 5)*pow(z, 4) + 10*pow(x, 10)*pow(y, 4)*pow(z, 5) + 10*pow(x, 10)*pow(y, 3)*pow(z, 6) - 2*pow(x, 10)*pow(y, 2)*pow(z, 7) + 35*pow(x, 9)*pow(y, 8)*pow(z, 2) + 10*pow(x, 9)*pow(y, 7)*pow(z, 3) + 250*pow(x, 9)*pow(y, 6)*pow(z, 4) - 20*pow(x, 9)*pow(y, 5)*pow(z, 5) + 250*pow(x, 9)*pow(y, 4)*pow(z, 6) + 10*pow(x, 9)*pow(y, 3)*pow(z, 7) + 35*pow(x, 9)*pow(y, 2)*pow(z, 8) + 35*pow(x, 8)*pow(y, 9)*pow(z, 2) - 49*pow(x, 8)*pow(y, 8)*pow(z, 3) + 70*pow(x, 8)*pow(y, 7)*pow(z, 4) - 140*pow(x, 8)*pow(y, 6)*pow(z, 5) - 140*pow(x, 8)*pow(y, 5)*pow(z, 6) + 70*pow(x, 8)*pow(y, 4)*pow(z, 7) - 49*pow(x, 8)*pow(y, 3)*pow(z, 8) + 35*pow(x, 8)*pow(y, 2)*pow(z, 9) - 2*pow(x, 7)*pow(y, 10)*pow(z, 2) + 10*pow(x, 7)*pow(y, 9)*pow(z, 3) + 70*pow(x, 7)*pow(y, 8)*pow(z, 4) + 8*pow(x, 7)*pow(y, 7)*pow(z, 5) + 200*pow(x, 7)*pow(y, 6)*pow(z, 6) + 8*pow(x, 7)*pow(y, 5)*pow(z, 7) + 70*pow(x, 7)*pow(y, 4)*pow(z, 8) + 10*pow(x, 7)*pow(y, 3)*pow(z, 9) - 2*pow(x, 7)*pow(y, 2)*pow(z, 10) + 10*pow(x, 6)*pow(y, 10)*pow(z, 3) + 250*pow(x, 6)*pow(y, 9)*pow(z, 4) - 140*pow(x, 6)*pow(y, 8)*pow(z, 5) + 200*pow(x, 6)*pow(y, 7)*pow(z, 6) + 200*pow(x, 6)*pow(y, 6)*pow(z, 7) - 140*pow(x, 6)*pow(y, 5)*pow(z, 8) + 250*pow(x, 6)*pow(y, 4)*pow(z, 9) + 10*pow(x, 6)*pow(y, 3)*pow(z, 10) + 10*pow(x, 5)*pow(y, 10)*pow(z, 4) - 20*pow(x, 5)*pow(y, 9)*pow(z, 5) - 140*pow(x, 5)*pow(y, 8)*pow(z, 6) + 8*pow(x, 5)*pow(y, 7)*pow(z, 7) - 140*pow(x, 5)*pow(y, 6)*pow(z, 8) - 20*pow(x, 5)*pow(y, 5)*pow(z, 9) + 10*pow(x, 5)*pow(y, 4)*pow(z, 10) + 10*pow(x, 4)*pow(y, 10)*pow(z, 5) + 250*pow(x, 4)*pow(y, 9)*pow(z, 6) + 70*pow(x, 4)*pow(y, 8)*pow(z, 7) + 70*pow(x, 4)*pow(y, 7)*pow(z, 8) + 250*pow(x, 4)*pow(y, 6)*pow(z, 9) + 10*pow(x, 4)*pow(y, 5)*pow(z, 10) + 10*pow(x, 3)*pow(y, 10)*pow(z, 6) + 10*pow(x, 3)*pow(y, 9)*pow(z, 7) - 49*pow(x, 3)*pow(y, 8)*pow(z, 8) + 10*pow(x, 3)*pow(y, 7)*pow(z, 9) + 10*pow(x, 3)*pow(y, 6)*pow(z, 10) - 2*pow(x, 2)*pow(y, 10)*pow(z, 7) + 35*pow(x, 2)*pow(y, 9)*pow(z, 8) + 35*pow(x, 2)*pow(y, 8)*pow(z, 9) - 2*pow(x, 2)*pow(y, 7)*pow(z, 10)
        - 7*pow(x, 10)*pow(y, 8)*pow(z, 2) - 2*pow(x, 10)*pow(y, 7)*pow(z, 3) - 50*pow(x, 10)*pow(y, 6)*pow(z, 4) + 4*pow(x, 10)*pow(y, 5)*pow(z, 5) - 50*pow(x, 10)*pow(y, 4)*pow(z, 6) - 2*pow(x, 10)*pow(y, 3)*pow(z, 7) - 7*pow(x, 10)*pow(y, 2)*pow(z, 8) - 25*pow(x, 9)*pow(y, 9)*pow(z, 2) + 35*pow(x, 9)*pow(y, 8)*pow(z, 3) - 50*pow(x, 9)*pow(y, 7)*pow(z, 4) + 100*pow(x, 9)*pow(y, 6)*pow(z, 5) + 100*pow(x, 9)*pow(y, 5)*pow(z, 6) - 50*pow(x, 9)*pow(y, 4)*pow(z, 7) + 35*pow(x, 9)*pow(y, 3)*pow(z, 8) - 25*pow(x, 9)*pow(y, 2)*pow(z, 9) - 7*pow(x, 8)*pow(y, 10)*pow(z, 2) + 35*pow(x, 8)*pow(y, 9)*pow(z, 3) + 245*pow(x, 8)*pow(y, 8)*pow(z, 4) + 28*pow(x, 8)*pow(y, 7)*pow(z, 5) + 700*pow(x, 8)*pow(y, 6)*pow(z, 6) + 28*pow(x, 8)*pow(y, 5)*pow(z, 7) + 245*pow(x, 8)*pow(y, 4)*pow(z, 8) + 35*pow(x, 8)*pow(y, 3)*pow(z, 9) - 7*pow(x, 8)*pow(y, 2)*pow(z, 10) - 2*pow(x, 7)*pow(y, 10)*pow(z, 3) - 50*pow(x, 7)*pow(y, 9)*pow(z, 4) + 28*pow(x, 7)*pow(y, 8)*pow(z, 5) - 40*pow(x, 7)*pow(y, 7)*pow(z, 6) - 40*pow(x, 7)*pow(y, 6)*pow(z, 7) + 28*pow(x, 7)*pow(y, 5)*pow(z, 8) - 50*pow(x, 7)*pow(y, 4)*pow(z, 9) - 2*pow(x, 7)*pow(y, 3)*pow(z, 10) - 50*pow(x, 6)*pow(y, 10)*pow(z, 4) + 100*pow(x, 6)*pow(y, 9)*pow(z, 5) + 700*pow(x, 6)*pow(y, 8)*pow(z, 6) - 40*pow(x, 6)*pow(y, 7)*pow(z, 7) + 700*pow(x, 6)*pow(y, 6)*pow(z, 8) + 100*pow(x, 6)*pow(y, 5)*pow(z, 9) - 50*pow(x, 6)*pow(y, 4)*pow(z, 10) + 4*pow(x, 5)*pow(y, 10)*pow(z, 5) + 100*pow(x, 5)*pow(y, 9)*pow(z, 6) + 28*pow(x, 5)*pow(y, 8)*pow(z, 7) + 28*pow(x, 5)*pow(y, 7)*pow(z, 8) + 100*pow(x, 5)*pow(y, 6)*pow(z, 9) + 4*pow(x, 5)*pow(y, 5)*pow(z, 10) - 50*pow(x, 4)*pow(y, 10)*pow(z, 6) - 50*pow(x, 4)*pow(y, 9)*pow(z, 7) + 245*pow(x, 4)*pow(y, 8)*pow(z, 8) - 50*pow(x, 4)*pow(y, 7)*pow(z, 9) - 50*pow(x, 4)*pow(y, 6)*pow(z, 10) - 2*pow(x, 3)*pow(y, 10)*pow(z, 7) + 35*pow(x, 3)*pow(y, 9)*pow(z, 8) + 35*pow(x, 3)*pow(y, 8)*pow(z, 9) - 2*pow(x, 3)*pow(y, 7)*pow(z, 10) - 7*pow(x, 2)*pow(y, 10)*pow(z, 8) - 25*pow(x, 2)*pow(y, 9)*pow(z, 9) - 7*pow(x, 2)*pow(y, 8)*pow(z, 10)
        + 5*pow(x, 10)*pow(y, 9)*pow(z, 2) - 7*pow(x, 10)*pow(y, 8)*pow(z, 3) + 10*pow(x, 10)*pow(y, 7)*pow(z, 4) - 20*pow(x, 10)*pow(y, 6)*pow(z, 5) - 20*pow(x, 10)*pow(y, 5)*pow(z, 6) + 10*pow(x, 10)*pow(y, 4)*pow(z, 7) - 7*pow(x, 10)*pow(y, 3)*pow(z, 8) + 5*pow(x, 10)*pow(y, 2)*pow(z, 9) + 5*pow(x, 9)*pow(y, 10)*pow(z, 2) - 25*pow(x, 9)*pow(y, 9)*pow(z, 3) - 175*pow(x, 9)*pow(y, 8)*pow(z, 4) - 20*pow(x, 9)*pow(y, 7)*pow(z, 5) - 500*pow(x, 9)*pow(y, 6)*pow(z, 6) - 20*pow(x, 9)*pow(y, 5)*pow(z, 7) - 175*pow(x, 9)*pow(y, 4)*pow(z, 8) - 25*pow(x, 9)*pow(y, 3)*pow(z, 9) + 5*pow(x, 9)*pow(y, 2)*pow(z, 10) - 7*pow(x, 8)*pow(y, 10)*pow(z, 3) - 175*pow(x, 8)*pow(y, 9)*pow(z, 4) + 98*pow(x, 8)*pow(y, 8)*pow(z, 5) - 140*pow(x, 8)*pow(y, 7)*pow(z, 6) - 140*pow(x, 8)*pow(y, 6)*pow(z, 7) + 98*pow(x, 8)*pow(y, 5)*pow(z, 8) - 175*pow(x, 8)*pow(y, 4)*pow(z, 9) - 7*pow(x, 8)*pow(y, 3)*pow(z, 10) + 10*pow(x, 7)*pow(y, 10)*pow(z, 4) - 20*pow(x, 7)*pow(y, 9)*pow(z, 5) - 140*pow(x, 7)*pow(y, 8)*pow(z, 6) + 8*pow(x, 7)*pow(y, 7)*pow(z, 7) - 140*pow(x, 7)*pow(y, 6)*pow(z, 8) - 20*pow(x, 7)*pow(y, 5)*pow(z, 9) + 10*pow(x, 7)*pow(y, 4)*pow(z, 10) - 20*pow(x, 6)*pow(y, 10)*pow(z, 5) - 500*pow(x, 6)*pow(y, 9)*pow(z, 6) - 140*pow(x, 6)*pow(y, 8)*pow(z, 7) - 140*pow(x, 6)*pow(y, 7)*pow(z, 8) - 500*pow(x, 6)*pow(y, 6)*pow(z, 9) - 20*pow(x, 6)*pow(y, 5)*pow(z, 10) - 20*pow(x, 5)*pow(y, 10)*pow(z, 6) - 20*pow(x, 5)*pow(y, 9)*pow(z, 7) + 98*pow(x, 5)*pow(y, 8)*pow(z, 8) - 20*pow(x, 5)*pow(y, 7)*pow(z, 9) - 20*pow(x, 5)*pow(y, 6)*pow(z, 10) + 10*pow(x, 4)*pow(y, 10)*pow(z, 7) - 175*pow(x, 4)*pow(y, 9)*pow(z, 8) - 175*pow(x, 4)*pow(y, 8)*pow(z, 9) + 10*pow(x, 4)*pow(y, 7)*pow(z, 10) - 7*pow(x, 3)*pow(y, 10)*pow(z, 8) - 25*pow(x, 3)*pow(y, 9)*pow(z, 9) - 7*pow(x, 3)*pow(y, 8)*pow(z, 10) + 5*pow(x, 2)*pow(y, 10)*pow(z, 9) + 5*pow(x, 2)*pow(y, 9)*pow(z, 10)
        - pow(x, 10)*pow(y, 10)*pow(z, 2) + 5*pow(x, 10)*pow(y, 9)*pow(z, 3) + 35*pow(x, 10)*pow(y, 8)*pow(z, 4) + 4*pow(x, 10)*pow(y, 7)*pow(z, 5) + 100*pow(x, 10)*pow(y, 6)*pow(z, 6) + 4*pow(x, 10)*pow(y, 5)*pow(z, 7) + 35*pow(x, 10)*pow(y, 4)*pow(z, 8) + 5*pow(x, 10)*pow(y, 3)*pow(z, 9) - pow(x, 10)*pow(y, 2)*pow(z, 10) + 5*pow(x, 9)*pow(y, 10)*pow(z, 3) + 125*pow(x, 9)*pow(y, 9)*pow(z, 4) - 70*pow(x, 9)*pow(y, 8)*pow(z, 5) + 100*pow(x, 9)*pow(y, 7)*pow(z, 6) + 100*pow(x, 9)*pow(y, 6)*pow(z, 7) - 70*pow(x, 9)*pow(y, 5)*pow(z, 8) + 125*pow(x, 9)*pow(y, 4)*pow(z, 9) + 5*pow(x, 9)*pow(y, 3)*pow(z, 10) + 35*pow(x, 8)*pow(y, 10)*pow(z, 4) - 70*pow(x, 8)*pow(y, 9)*pow(z, 5) - 490*pow(x, 8)*pow(y, 8)*pow(z, 6) + 28*pow(x, 8)*pow(y, 7)*pow(z, 7) - 490*pow(x, 8)*pow(y, 6)*pow(z, 8) - 70*pow(x, 8)*pow(y, 5)*pow(z, 9) + 35*pow(x, 8)*pow(y, 4)*pow(z, 10) + 4*pow(x, 7)*pow(y, 10)*pow(z, 5) + 100*pow(x, 7)*pow(y, 9)*pow(z, 6) + 28*pow(x, 7)*pow(y, 8)*pow(z, 7) + 28*pow(x, 7)*pow(y, 7)*pow(z, 8) + 100*pow(x, 7)*pow(y, 6)*pow(z, 9) + 4*pow(x, 7)*pow(y, 5)*pow(z, 10) + 100*pow(x, 6)*pow(y, 10)*pow(z, 6) + 100*pow(x, 6)*pow(y, 9)*pow(z, 7) - 490*pow(x, 6)*pow(y, 8)*pow(z, 8) + 100*pow(x, 6)*pow(y, 7)*pow(z, 9) + 100*pow(x, 6)*pow(y, 6)*pow(z, 10) + 4*pow(x, 5)*pow(y, 10)*pow(z, 7) - 70*pow(x, 5)*pow(y, 9)*pow(z, 8) - 70*pow(x, 5)*pow(y, 8)*pow(z, 9) + 4*pow(x, 5)*pow(y, 7)*pow(z, 10) + 35*pow(x, 4)*pow(y, 10)*pow(z, 8) + 125*pow(x, 4)*pow(y, 9)*pow(z, 9) + 35*pow(x, 4)*pow(y, 8)*pow(z, 10) + 5*pow(x, 3)*pow(y, 10)*pow(z, 9) + 5*pow(x, 3)*pow(y, 9)*pow(z, 10) - pow(x, 2)*pow(y, 10)*pow(z, 10)
        - pow(x, 10)*pow(y, 10)*pow(z, 3) - 25*pow(x, 10)*pow(y, 9)*pow(z, 4) + 14*pow(x, 10)*pow(y, 8)*pow(z, 5) - 20*pow(x, 10)*pow(y, 7)*pow(z, 6) - 20*pow(x, 10)*pow(y, 6)*pow(z, 7) + 14*pow(x, 10)*pow(y, 5)*pow(z, 8) - 25*pow(x, 10)*pow(y, 4)*pow(z, 9) - pow(x, 10)*pow(y, 3)*pow(z, 10) - 25*pow(x, 9)*pow(y, 10)*pow(z, 4) + 50*pow(x, 9)*pow(y, 9)*pow(z, 5) + 350*pow(x, 9)*pow(y, 8)*pow(z, 6) - 20*pow(x, 9)*pow(y, 7)*pow(z, 7) + 350*pow(x, 9)*pow(y, 6)*pow(z, 8) + 50*pow(x, 9)*pow(y, 5)*pow(z, 9) - 25*pow(x, 9)*pow(y, 4)*pow(z, 10) + 14*pow(x, 8)*pow(y, 10)*pow(z, 5) + 350*pow(x, 8)*pow(y, 9)*pow(z, 6) + 98*pow(x, 8)*pow(y, 8)*pow(z, 7) + 98*pow(x, 8)*pow(y, 7)*pow(z, 8) + 350*pow(x, 8)*pow(y, 6)*pow(z, 9) + 14*pow(x, 8)*pow(y, 5)*pow(z, 10) - 20*pow(x, 7)*pow(y, 10)*pow(z, 6) - 20*pow(x, 7)*pow(y, 9)*pow(z, 7) + 98*pow(x, 7)*pow(y, 8)*pow(z, 8) - 20*pow(x, 7)*pow(y, 7)*pow(z, 9) - 20*pow(x, 7)*pow(y, 6)*pow(z, 10) - 20*pow(x, 6)*pow(y, 10)*pow(z, 7) + 350*pow(x, 6)*pow(y, 9)*pow(z, 8) + 350*pow(x, 6)*pow(y, 8)*pow(z, 9) - 20*pow(x, 6)*pow(y, 7)*pow(z, 10) + 14*pow(x, 5)*pow(y, 10)*pow(z, 8) + 50*pow(x, 5)*pow(y, 9)*pow(z, 9) + 14*pow(x, 5)*pow(y, 8)*pow(z, 10) - 25*pow(x, 4)*pow(y, 10)*pow(z, 9) - 25*pow(x, 4)*pow(y, 9)*pow(z, 10) - pow(x, 3)*pow(y, 10)*pow(z, 10)
        + 5*pow(x, 10)*pow(y, 10)*pow(z, 4) - 10*pow(x, 10)*pow(y, 9)*pow(z, 5) - 70*pow(x, 10)*pow(y, 8)*pow(z, 6) + 4*pow(x, 10)*pow(y, 7)*pow(z, 7) - 70*pow(x, 10)*pow(y, 6)*pow(z, 8) - 10*pow(x, 10)*pow(y, 5)*pow(z, 9) + 5*pow(x, 10)*pow(y, 4)*pow(z, 10) - 10*pow(x, 9)*pow(y, 10)*pow(z, 5) - 250*pow(x, 9)*pow(y, 9)*pow(z, 6) - 70*pow(x, 9)*pow(y, 8)*pow(z, 7) - 70*pow(x, 9)*pow(y, 7)*pow(z, 8) - 250*pow(x, 9)*pow(y, 6)*pow(z, 9) - 10*pow(x, 9)*pow(y, 5)*pow(z, 10) - 70*pow(x, 8)*pow(y, 10)*pow(z, 6) - 70*pow(x, 8)*pow(y, 9)*pow(z, 7) + 343*pow(x, 8)*pow(y, 8)*pow(z, 8) - 70*pow(x, 8)*pow(y, 7)*pow(z, 9) - 70*pow(x, 8)*pow(y, 6)*pow(z, 10) + 4*pow(x, 7)*pow(y, 10)*pow(z, 7) - 70*pow(x, 7)*pow(y, 9)*pow(z, 8) - 70*pow(x, 7)*pow(y, 8)*pow(z, 9) + 4*pow(x, 7)*pow(y, 7)*pow(z, 10) - 70*pow(x, 6)*pow(y, 10)*pow(z, 8) - 250*pow(x, 6)*pow(y, 9)*pow(z, 9) - 70*pow(x, 6)*pow(y, 8)*pow(z, 10) - 10*pow(x, 5)*pow(y, 10)*pow(z, 9) - 10*pow(x, 5)*pow(y, 9)*pow(z, 10) + 5*pow(x, 4)*pow(y, 10)*pow(z, 10)
        + 2*pow(x, 10)*pow(y, 10)*pow(z, 5) + 50*pow(x, 10)*pow(y, 9)*pow(z, 6) + 14*pow(x, 10)*pow(y, 8)*pow(z, 7) + 14*pow(x, 10)*pow(y, 7)*pow(z, 8) + 50*pow(x, 10)*pow(y, 6)*pow(z, 9) + 2*pow(x, 10)*pow(y, 5)*pow(z, 10) + 50*pow(x, 9)*pow(y, 10)*pow(z, 6) + 50*pow(x, 9)*pow(y, 9)*pow(z, 7) - 245*pow(x, 9)*pow(y, 8)*pow(z, 8) + 50*pow(x, 9)*pow(y, 7)*pow(z, 9) + 50*pow(x, 9)*pow(y, 6)*pow(z, 10) + 14*pow(x, 8)*pow(y, 10)*pow(z, 7) - 245*pow(x, 8)*pow(y, 9)*pow(z, 8) - 245*pow(x, 8)*pow(y, 8)*pow(z, 9) + 14*pow(x, 8)*pow(y, 7)*pow(z, 10) + 14*pow(x, 7)*pow(y, 10)*pow(z, 8) + 50*pow(x, 7)*pow(y, 9)*pow(z, 9) + 14*pow(x, 7)*pow(y, 8)*pow(z, 10) + 50*pow(x, 6)*pow(y, 10)*pow(z, 9) + 50*pow(x, 6)*pow(y, 9)*pow(z, 10) + 2*pow(x, 5)*pow(y, 10)*pow(z, 10)
        - 10*pow(x, 10)*pow(y, 10)*pow(z, 6) - 10*pow(x, 10)*pow(y, 9)*pow(z, 7) + 49*pow(x, 10)*pow(y, 8)*pow(z, 8) - 10*pow(x, 10)*pow(y, 7)*pow(z, 9) - 10*pow(x, 10)*pow(y, 6)*pow(z, 10) - 10*pow(x, 9)*pow(y, 10)*pow(z, 7) + 175*pow(x, 9)*pow(y, 9)*pow(z, 8) + 175*pow(x, 9)*pow(y, 8)*pow(z, 9) - 10*pow(x, 9)*pow(y, 7)*pow(z, 10) + 49*pow(x, 8)*pow(y, 10)*pow(z, 8) + 175*pow(x, 8)*pow(y, 9)*pow(z, 9) + 49*pow(x, 8)*pow(y, 8)*pow(z, 10) - 10*pow(x, 7)*pow(y, 10)*pow(z, 9) - 10*pow(x, 7)*pow(y, 9)*pow(z, 10) - 10*pow(x, 6)*pow(y, 10)*pow(z, 10)
        + 2*pow(x, 10)*pow(y, 10)*pow(z, 7) - 35*pow(x, 10)*pow(y, 9)*pow(z, 8) - 35*pow(x, 10)*pow(y, 8)*pow(z, 9) + 2*pow(x, 10)*pow(y, 7)*pow(z, 10) - 35*pow(x, 9)*pow(y, 10)*pow(z, 8) - 125*pow(x, 9)*pow(y, 9)*pow(z, 9) - 35*pow(x, 9)*pow(y, 8)*pow(z, 10) - 35*pow(x, 8)*pow(y, 10)*pow(z, 9) - 35*pow(x, 8)*pow(y, 9)*pow(z, 10) + 2*pow(x, 7)*pow(y, 10)*pow(z, 10)
        + 7*pow(x, 10)*pow(y, 10)*pow(z, 8) + 25*pow(x, 10)*pow(y, 9)*pow(z, 9) + 7*pow(x, 10)*pow(y, 8)*pow(z, 10) + 25*pow(x, 9)*pow(y, 10)*pow(z, 9) + 25*pow(x, 9)*pow(y, 9)*pow(z, 10) + 7*pow(x, 8)*pow(y, 10)*pow(z, 10)
        - 5*pow(x, 10)*pow(y, 10)*pow(z, 9) - 5*pow(x, 10)*pow(y, 9)*pow(z, 10) - 5*pow(x, 9)*pow(y, 10)*pow(z, 10)
        + pow(x, 10)*pow(y, 10)*pow(z, 10)
    );

    return res;
};

/**************************************/


int main(int argc, char const *argv[])
{
    if (argc < 6)
    {
        std::cout << "USAGE: " << argv[0] << " <solver> <preconditioner> <grid_size> <max_iterations> <run_label>" << std::endl;
        std::cout << std::endl;
        std::cout << "Required arguments:" << std::endl;
        std::cout << "    solver               Solver type: 'jacobi' or 'gmres'" << std::endl;
        std::cout << "    preconditioner       Preconditioner type: 'diag' (diagonal/Jacobi) or 'mg' (multigrid)" << std::endl;
        std::cout << "    grid_size            Number of grid points per dimension (e.g., 32 for 32x32x32 grid)" << std::endl;
        std::cout << "    max_iterations       Maximum number of solver iterations (e.g., 100)" << std::endl;
        std::cout << "    run_label            Label for this run, used in output filenames (e.g., 'cpu', 'gpu')" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    const std::string solver_type         = argv[1];
    const std::string preconditioner_type = argv[2];
    const int         grid_size           = std::stoi(argv[3]);
    const int         max_iterations      = std::stoi(argv[4]);
    const std::string run_label           = argv[5];


    // Solver configuration
    const std::string solver_name  = solver_type;
    const std::string scalar_label = std::is_same_v<float, scalar> ? "float" : "double";

    log_t log;

    auto range = idx_nd_type::make_ones() * grid_size;
    auto step  = grid_step_type::make_ones() / scalar(grid_size);
    auto cond  = boundary_cond<dim>
    {
        {-1, -1, -1}, // left
        {-1, -1, -1}  // right
    }; // -1 = dirichlet, +1 = neuman

    vector_t solution(range), exact_solution(range);

    auto vspace = std::make_shared<vec_ops_t>(range);
    {
        vspace->assign_scalar(0.f, solution);  // Initialize solution to zero
        vector_view_t exact_view(exact_solution, false);

        for(int i=0; i<range[0]; i++) {
            for(int j=0; j<range[1]; j++) {
                for(int k=0; k<range[2]; k++)
                {
                    const auto coord_x = step[0] * (0.5f + i);
                    const auto coord_y = step[1] * (0.5f + j);
                    const auto coord_z = step[2] * (0.5f + k);

                    auto exact_val = u(coord_x, coord_y, coord_z);
                    for (int t = 0; t < tensor_dim; t++) {
                        exact_view(i, j, k, t) = exact_val[t];
                    }
                }
            }
        }

        exact_view.release();
    }


    auto cahn_hilliard_jacobi_op = std::make_shared<cahn_hilliard_jacobi_op_t>(range, step, cond);
    auto cahn_hilliard_op = std::make_shared<cahn_hilliard_op_t>(range, step, cond, cahn_hilliard_jacobi_op);

    std::shared_ptr<precond_interface> precond;

    mg_utils_t    mg_utils;
    mg_params_t   mg_params;

    mg_utils.log               = &log;
    mg_params.direct_coarse    = false;
    mg_params.num_sweeps_pre   = 4;
    mg_params.num_sweeps_post  = 4;

    precond = std::make_shared<mg_t>(mg_utils, mg_params);

    gmres_solver::params params_gmres;
    params_gmres.monitor.rel_tol = std::is_same_v<float, scalar> ? 5e-6f : 1e-10;
    //params_gmres.monitor.rel_tol = 1.0e-6;
    params_gmres.monitor.max_iters_num = max_iterations;
    params_gmres.monitor.save_convergence_history = true;
    params_gmres.do_restart_on_false_ritz_convergence = true;
    params_gmres.basis_size = 25;
    // params_gmres.basis_size = basis_size;
    params_gmres.preconditioner_side = 'L';
    params_gmres.reorthogonalization = true;
    auto lin_solver = std::make_shared<gmres_solver>(cahn_hilliard_jacobi_op, vspace, &log, params_gmres, precond);

    auto newton_iteration = std::make_shared<newton_iteration_t>(vspace, lin_solver);

    // Create error monitor to track ||solution - exact|| at each Newton iteration
    auto error_monitor = std::make_shared<error_monitor_t>(vspace, exact_solution, &log);

    auto newton_solver = std::make_shared<newton_solver_t>(vspace, &log, newton_iteration);
    newton_solver->convergence_strategy()->set_tolerance(std::is_same_v<float, scalar> ? 5e-6f : 1e-10);

    // Solve the system and measure execution time
    std::chrono::duration<double, std::milli> solve_time_ms;
    bool converged;
    std::vector<std::pair<int, scalar>> convergence_history;

    // First, verify that F(exact_solution) is close to zero
    // TODO: Remove
    vector_t F_exact(range);
    cahn_hilliard_op->apply(exact_solution, F_exact);
    scalar F_exact_norm = vspace->norm_l2(F_exact);
    log.info_f("Verification: ||F(exact_solution)||_2 = %le", static_cast<double>(F_exact_norm));


    auto start = std::chrono::steady_clock::now();
    newton_solver->solve(cahn_hilliard_op.get(), error_monitor.get(), nullptr, solution);
    auto end = std::chrono::steady_clock::now();
    solve_time_ms = (end - start);


    // Final comparison
    // TODO: Remove
    vector_t error(range);
    vspace->assign_lin_comb(scalar(1), solution, scalar(-1), exact_solution, error);
    scalar error_norm = vspace->norm_l2(error);
    scalar exact_norm = vspace->norm_l2(exact_solution);
    log.info_f("Final: ||solution - exact||_2 = %le, relative = %le",
               static_cast<double>(error_norm), static_cast<double>(error_norm / exact_norm));
}
