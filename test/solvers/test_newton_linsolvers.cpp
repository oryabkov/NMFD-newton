#include <cmath>
#include <memory>
#include <type_traits>

#include <nmfd/backend/serial_cpu.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/jacobi.h>
#include <nmfd/solvers/monitor_krylov.h>
#include "newton_test_fixture.h"

namespace
{

template <class Space>
struct scalar_problem
{
    using T           = typename Space::scalar_type;
    using vector_type = typename Space::vector_type;

    struct jacobi_operator_type
    {
        T    coefficient = T( 1 );
        bool fail        = false;

        bool apply( const vector_type &x, vector_type &out ) const
        {
            if ( fail )
                return false;
            out( 0 ) = coefficient * x( 0 );
            return true;
        }
    };

    std::shared_ptr<jacobi_operator_type> jacobi = std::make_shared<jacobi_operator_type>();

    void apply( const vector_type &x, vector_type &out ) const
    {
        out( 0 ) = x( 0 ) * x( 0 ) - T( 2 );
    }

    void set_linearization_point( const vector_type &x )
    {
        jacobi->coefficient = T( 2 ) * x( 0 );
    }

    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobi;
    }
};

template <class T>
void test( nmfd::tests::test_report &report )
{
    using backend_type        = nmfd::backend::serial_cpu<T>;
    using space_type          = typename backend_type::vector_space_type;
    using log_type            = typename backend_type::log_type;
    using problem_type        = scalar_problem<space_type>;
    using operator_type       = typename problem_type::jacobi_operator_type;
    using monitor_type        = nmfd::solvers::monitor_krylov<space_type, log_type>;
    using preconditioner_type = nmfd::preconditioners::dummy<space_type, operator_type>;
    using gmres_type = nmfd::solvers::gmres<space_type, monitor_type, log_type, operator_type, preconditioner_type>;
    using base_type =
        nmfd::solvers::iter_solver_base<space_type, monitor_type, log_type, operator_type, preconditioner_type>;
    using iteration_type = nmfd::solvers::newton_iteration<space_type, problem_type, base_type>;
    using newton_type    = nmfd::solvers::nonlinear_solver<space_type, log_type, problem_type, iteration_type>;
    using jacobi_type = nmfd::solvers::jacobi<space_type, operator_type, preconditioner_type, monitor_type, log_type>;

    static_assert( std::is_same_v<typename gmres_type::vector_space_type, space_type> );
    static_assert( std::is_base_of_v<base_type, gmres_type> && std::is_base_of_v<base_type, jacobi_type> );
    backend_type                backend;
    auto                        space = std::make_shared<space_type>( 1 );
    typename gmres_type::params params;
    params.basis_size                       = 2;
    params.batch_size                       = 1;
    params.monitor.rel_tol                  = T( 1e-5 );
    params.monitor.abs_tol                  = T( 0 );
    params.monitor.max_iters_num            = 20;
    params.monitor.save_convergence_history = true;
    typename gmres_type::utils          utilities( backend, space );
    typename preconditioner_type::utils preconditioner_utilities( backend, space );
    report.check<T>(
        utilities.log == &backend.log() && preconditioner_utilities.ops == space,
        "borrowed log and shared space utilities"
    );

    nmfd::detail::vector_wrap<space_type, true, true> rhs( *space ), x( *space ), residual( *space );
    auto                                              problem = std::make_shared<problem_type>();
    auto                                              op      = problem->jacobi;
    ( *rhs )( 0 )                                             = T( 4 );
    ( *x )( 0 )                                               = T( 0 );
    auto                       gmres  = std::make_shared<gmres_type>( op, space, nullptr, params );
    std::shared_ptr<base_type> linear = gmres;
    report.check<T>( linear->solve( *rhs, *x ), "bound virtual GMRES solve" );
    op->apply( *x, *residual );
    space->add_lin_comb( T( -1 ), *rhs, T( 1 ), *residual );
    report.check<T>( space->norm( *residual ) <= T( 1e-5 ), "true GMRES residual" );
    report.check<T>( !linear->monitor().convergence_history().empty(), "monitor history through base" );

    op->coefficient = T( 2 );
    linear->set_operator( op );
    ( *x )( 0 ) = T( 0 );
    report.check<T>(
        linear->solve( *op, *rhs, *x ) && std::abs( ( *x )( 0 ) - T( 2 ) ) < T( 1e-5 ),
        "explicit-operator solve after values change"
    );
    op->coefficient = T( 1 );
    linear          = std::make_shared<jacobi_type>( op, space );
    ( *x )( 0 )     = T( 0 );
    report.check<T>( linear->solve( *rhs, *x ), "Jacobi shares the canonical base" );

    newton_type newton( space, nullptr, std::make_shared<iteration_type>( space, gmres ) );
    newton.set_nonlinear_operator( problem );
    ( *x )( 0 ) = T( 1 );
    report.check<T>( newton.solve( *x ), "Newton with canonical GMRES" );
    report.check<T>( std::abs( ( *x )( 0 ) - std::sqrt( T( 2 ) ) ) < T( 2e-6 ), "Newton root" );
    problem->jacobi->fail = true;
    ( *x )( 0 )           = T( 1 );
    report.check<T>( !newton.solve( *x ) && ( *x )( 0 ) == T( 1 ), "linear failure preserves Newton state" );

    using concrete_iteration_type = nmfd::solvers::newton_iteration<space_type, problem_type, gmres_type>;
    using hierarchy_newton_type =
        nmfd::solvers::nonlinear_solver<space_type, log_type, problem_type, concrete_iteration_type>;
    typename hierarchy_newton_type::utils_hierarchy  hierarchy_utils( backend, space );
    typename hierarchy_newton_type::params_hierarchy hierarchy_params;
    hierarchy_params.iteration_operator.lin_solver = typename gmres_type::params_hierarchy();
    hierarchy_newton_type hierarchy_solver( hierarchy_utils, hierarchy_params );
    problem->jacobi->fail = false;
    ( *x )( 0 )           = T( 1 );
    report.check<T>(
        hierarchy_solver.solve( problem.get(), nullptr, nullptr, *x ), "recursive Newton-GMRES construction"
    );
#ifdef NMFD_ENABLE_NLOHMANN
    auto json = params.to_json();
    for ( const auto *field :
          { "orthogonalization", "reorthogonalization_policy", "dgks_eta", "breakdown_relative_tolerance",
            "max_orthogonalization_passes" } )
        json.erase( field );
    typename gmres_type::params restored;
    restored.from_json( json );
    report.check<T>(
        restored.basis_size == params.basis_size && restored.monitor.rel_tol == params.monitor.rel_tol,
        "original GMRES JSON remains accepted"
    );
#endif
}

}

int main()
{
    nmfd::tests::test_report report;
    test<float>( report );
    test<double>( report );
    return report.finish();
}
