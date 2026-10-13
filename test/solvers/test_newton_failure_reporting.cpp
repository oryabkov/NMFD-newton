#include "newton_test_fixture.h"

using namespace nmfd::tests;

// Forward all convergence operations except the optional linear-failure hook.
template <class Base>
struct strategy_without_linear_status
{
    using params      = typename Base::params;
    using vector_type = typename Base::vector_space_type::vector_type;
    Base base;
    strategy_without_linear_status(
        std::shared_ptr<typename Base::vector_space_type> space, scfd::utils::log_std *log, params p = params()
    )
        : base( space, log, p )
    {
    }
    template <class Problem, class Projection, class Quality>
    bool
    check_convergence( Problem *problem, Projection *projection, Quality *quality, vector_type &x, vector_type &dx )
    {
        return base.check_convergence( problem, projection, quality, x, dx );
    }
    void reset_iterations()
    {
        base.reset_iterations();
    }
    int get_result_status()
    {
        return base.get_result_status();
    }
};

template <class T>
void test( test_report &report )
{
    using fixture     = newton_fixture<T>;
    using vector_type = typename fixture::vector_type;
    for ( bool strict : { false, true } )
    {
        typename fixture::params_type p;
        p.check_linear_solver_fail = strict;
        fixture f( p );
        f.linear->converged = false;
        vector_type x{ T( 2 ) };
        const bool  success = f.solver.solve( &f.problem, nullptr, nullptr, x );
        report.check<T>(
            success == !strict && f.strategy->get_result_status() == ( strict ? 6 : 0 ) &&
                x[0] == ( strict ? T( 2 ) : T( 1 ) ),
            "optional strict linear failure reporting"
        );
    }
    {
        using strategy = strategy_without_linear_status<typename fixture::strategy_type>;
        using solver   = nmfd::solvers::nonlinear_solver<
              typename fixture::space_type, scfd::utils::log_std, typename fixture::problem_type,
              typename fixture::iteration_type, nmfd::operations::ident_operator<typename fixture::space_type>,
              nmfd::operations::zero_functional<typename fixture::space_type>, strategy>;
        static_assert( !nmfd::solvers::has_lin_solver_status<strategy>::value );
        static_assert( nmfd::solvers::has_lin_solver_status<typename fixture::strategy_type>::value );
        fixture f;
        auto    convergence = std::make_shared<strategy>( f.space, nullptr );
        solver  nonlinear( f.space, nullptr, f.iteration, convergence );
        f.linear->converged = false;
        vector_type x{ T( 2 ) };
        report.check<T>(
            nonlinear.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1 ),
            "strategy without linear status compiles and solves"
        );
    }
    for ( int expected : { 2, 3, 7 } )
    {
        typename fixture::params_type p;
        p.max_backtracking_trials = 3;
        fixture f( p );
        f.problem.reference_state  = T( 2 );
        f.problem.invalid_residual = expected == 2;
        f.problem.invalid_trials   = expected == 3;
        f.linear->mode             = expected == 7 ? correction::nonfinite : correction::exact;
        vector_type x{ T( 2 ) };
        const bool  success = f.solver.solve( &f.problem, nullptr, nullptr, x );
        report.check<T>(
            !success && f.strategy->get_result_status() == expected && x[0] == T( 2 ),
            "nonfinite numerical failure returns false without updating state"
        );
        report.check<T>(
            f.problem.trial_calls == ( expected == 3 ? 3u : 0u ),
            "invalid correction bypasses trial evaluation; invalid residual retries are bounded"
        );
    }
    {
        fixture     f;
        vector_type x{ std::numeric_limits<T>::infinity() };
        report.check<T>(
            !f.solver.solve( &f.problem, nullptr, nullptr, x ) && f.strategy->get_result_status() == 2 &&
                f.linear->calls == 0,
            "nonfinite initial residual bypasses Newton correction"
        );
    }
    for ( failure_point phase :
          { failure_point::linearization, failure_point::residual, failure_point::jacobi_access,
            failure_point::operator_setup, failure_point::linear_solve } )
    {
        fixture f;
        f.problem.fail = f.linear->fail = phase;
        vector_type x{ T( 2 ) };
        bool        propagated = false;
        try
        {
            f.solver.solve( &f.problem, nullptr, nullptr, x );
        }
        catch ( const std::runtime_error & )
        {
            propagated = true;
        }
        report.check<T>( propagated && x[0] == T( 2 ), "operator and backend exceptions propagate unchanged" );
        f.problem.fail = f.linear->fail = failure_point::none;
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1 ),
            "solve can be restarted after an exception"
        );
    }
}

int main()
{
    test_report report;
    test<float>( report );
    test<double>( report );
    test<long double>( report );
    return report.finish();
}
