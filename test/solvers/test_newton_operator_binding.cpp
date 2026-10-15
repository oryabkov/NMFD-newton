#include "newton_test_fixture.h"

using namespace nmfd::tests;

template <class T>
void test_binding( test_report &report )
{
    using fixture     = newton_fixture<T>;
    using vector_type = typename fixture::vector_type;
    fixture     f;
    vector_type initial{ T( 2 ) }, x{ T( 9 ) };
    for ( bool out_of_place : { false, true } )
    {
        bool threw = false;
        try
        {
            if ( out_of_place )
                f.solver.solve( initial, x );
            else
                f.solver.solve( x );
        }
        catch ( const std::logic_error & )
        {
            threw = true;
        }
        report.check<T>(
            threw && initial[0] == T( 2 ) && x[0] == T( 9 ) && f.problem.residual_calls == 0 && f.linear->calls == 0,
            "unbound solve fails before touching either vector or evaluating the problem"
        );
    }

    const auto lifecycle_calls = f.space->lifecycle_calls;
    f.solver.set_nonlinear_operator( &f.problem );
    f.solver.set_project_operator( nullptr );
    f.solver.set_quality_functor( nullptr );
    report.check<T>(
        f.problem.residual_calls == 0 && f.linear->calls == 0, "binding does not evaluate or linearize the problem"
    );
    report.check<T>(
        f.solver.solve( initial, x ) && x[0] == T( 1 ) && initial[0] == T( 2 ),
        "bound out-of-place solve uses optional null hooks"
    );
    const auto bound_calls = f.linear->calls;
    x[0]                   = initial[0];
    report.check<T>(
        f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1 ) && f.linear->calls == 2 * bound_calls,
        "bound and explicit solves perform the same corrections"
    );

    typename fixture::problem_type other;
    f.solver.set_nonlinear_operator( &other );
    auto original_calls = f.problem.residual_calls;
    x[0]                = T( 2 );
    report.check<T>(
        f.solver.solve( x ) && x[0] == T( 1 ) && other.residual_calls > 0 && f.problem.residual_calls == original_calls,
        "rebinding selects only the new problem"
    );
    other.invalid_residual = true;
    x[0]                   = T( 2 );
    report.check<T>(
        !f.solver.solve( x ) && f.strategy->get_result_status() == 2 && x[0] == T( 2 ),
        "bound problem updates are visible without rebinding"
    );
    other.invalid_residual = false;
    report.check<T>( f.solver.solve( x ) && x[0] == T( 1 ), "bound solve restarts after a numerical failure" );

    report.check<T>(
        f.solver.solve( &f.problem, nullptr, nullptr, initial, x ),
        "explicit solve remains available with another problem bound"
    );
    original_calls         = f.problem.residual_calls;
    const auto other_calls = other.residual_calls;
    x[0]                   = T( 2 );
    report.check<T>(
        f.solver.solve( x ) && other.residual_calls > other_calls && f.problem.residual_calls == original_calls,
        "explicit solve does not overwrite the stored binding"
    );

    other.fail = failure_point::linearization;
    x[0]       = T( 2 );
    bool threw = false;
    try
    {
        f.solver.solve( x );
    }
    catch ( const std::runtime_error & )
    {
        threw = true;
    }
    report.check<T>( threw && x[0] == T( 2 ), "bound solve preserves exception propagation" );
    other.fail = failure_point::none;
    report.check<T>( f.solver.solve( x ) && x[0] == T( 1 ), "bound solve restarts after an exception" );

    f.solver.set_nonlinear_operator( nullptr );
    threw = false;
    x[0]  = T( 9 );
    try
    {
        f.solver.solve( initial, x );
    }
    catch ( const std::logic_error & )
    {
        threw = true;
    }
    report.check<T>( threw && x[0] == T( 9 ), "clearing the required binding restores the unbound guard" );
    report.check<T>(
        f.solver.solve( &f.problem, nullptr, nullptr, initial, x ), "explicit solve needs no stored binding"
    );
    report.check<T>(
        f.space->lifecycle_calls == lifecycle_calls, "binding and repeated solves allocate no additional vector storage"
    );

    for ( bool out_of_place : { false, true } )
    {
        fixture failing;
        failing.solver.set_nonlinear_operator( &failing.problem );
        failing.linear->mode       = correction::half;
        failing.linear->fail_after = 1;
        x[0]                       = T( 2 );
        const bool success         = out_of_place ? failing.solver.solve( initial, x ) : failing.solver.solve( x );
        report.check<T>(
            !success && failing.strategy->get_result_status() == 6 && initial[0] == T( 2 ) &&
                x[0] == ( out_of_place ? T( 2 ) : T( 1.5 ) ),
            "bound solves retain explicit rollback and accepted-iterate semantics"
        );
    }
}

template <class T>
struct optional_hook
{
    using vector_type             = typename newton_fixture<T>::vector_type;
    unsigned int projection_calls = 0, quality_calls = 0;

    void apply( vector_type & )
    {
        ++projection_calls;
    }
    T calc( const vector_type & )
    {
        ++quality_calls;
        return T( 0 );
    }
};

template <class T>
void test_hooks( test_report &report )
{
    using fixture       = newton_fixture<T>;
    using hook_type     = optional_hook<T>;
    using strategy_type = nmfd::solvers::default_convergence_strategy<
        typename fixture::space_type, scfd::utils::log_std, typename fixture::problem_type, hook_type, hook_type>;
    using solver_type = nmfd::solvers::nonlinear_solver<
        typename fixture::space_type, scfd::utils::log_std, typename fixture::problem_type,
        typename fixture::iteration_type, hook_type, hook_type>;
    fixture                       f;
    auto                          strategy = std::make_shared<strategy_type>( f.space, nullptr );
    solver_type                   solver( f.space, nullptr, f.iteration, strategy );
    hook_type                     first, second;
    typename fixture::vector_type initial{ T( 2 ) }, x{ T( 9 ) };
    solver.set_nonlinear_operator( &f.problem );
    solver.set_project_operator( &first );
    solver.set_quality_functor( &first );
    report.check<T>(
        solver.solve( initial, x ) && first.projection_calls == 1 && first.quality_calls == 1,
        "distinct setters support projection and quality hooks of the same type"
    );

    report.check<T>(
        solver.solve( &f.problem, &second, &second, initial, x ) && first.projection_calls == 1 &&
            first.quality_calls == 1 && second.projection_calls == 1 && second.quality_calls == 1,
        "explicit hooks override stored hooks only for that solve"
    );
    report.check<T>(
        solver.solve( initial, x ) && first.projection_calls == 2 && first.quality_calls == 2 &&
            second.projection_calls == 1 && second.quality_calls == 1,
        "explicit hooks leave stored hook bindings unchanged"
    );

    solver.set_project_operator( nullptr );
    report.check<T>(
        solver.solve( initial, x ) && first.projection_calls == 2 && first.quality_calls == 3,
        "projection can be cleared independently"
    );
    solver.set_project_operator( &second );
    solver.set_quality_functor( nullptr );
    report.check<T>(
        solver.solve( initial, x ) && second.projection_calls == 2 && first.quality_calls == 3 &&
            second.quality_calls == 1,
        "quality can be cleared independently while projection is rebound"
    );
    solver.set_quality_functor( &second );
    report.check<T>(
        solver.solve( initial, x ) && second.projection_calls == 3 && second.quality_calls == 2,
        "quality hook can be rebound"
    );
    solver.set_project_operator( nullptr );
    solver.set_quality_functor( nullptr );
    report.check<T>(
        solver.solve( initial, x ) && second.projection_calls == 3 && second.quality_calls == 2,
        "both optional hooks can be disabled"
    );
}

int main()
{
    test_report report;
    test_binding<float>( report );
    test_binding<double>( report );
    test_binding<long double>( report );
    test_hooks<float>( report );
    test_hooks<double>( report );
    test_hooks<long double>( report );
    return report.finish();
}
