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
    f.solver.set_nonlinear_operator( f.problem_owner );
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

    auto  other_owner = std::make_shared<typename fixture::problem_type>();
    auto &other       = *other_owner;
    f.solver.set_nonlinear_operator( other_owner );
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
        failing.solver.set_nonlinear_operator( failing.problem_owner );
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
void test_shared_binding( test_report &report )
{
    using fixture      = newton_fixture<T>;
    using problem_type = typename fixture::problem_type;
    fixture                       f;
    typename fixture::vector_type initial{ T( 2 ) }, x{ T( 9 ) };
    const auto                    lifecycle_calls = f.space->lifecycle_calls;
    auto                          problem         = std::make_shared<problem_type>();
    std::weak_ptr<problem_type>   retained        = problem;
    f.solver.set_nonlinear_operator( problem );
    report.check<T>(
        problem.use_count() == 2 && problem->residual_calls == 0 && f.linear->calls == 0,
        "shared binding retains the original object without evaluating it"
    );
    report.check<T>(
        f.solver.solve( initial, x ) && x[0] == T( 1 ) && problem.use_count() == 2,
        "shared binding forwards to the existing solve"
    );
    problem.reset();
    report.check<T>(
        !retained.expired() && f.solver.solve( initial, x ),
        "solver retains the problem after the caller releases ownership"
    );
    f.solver.set_nonlinear_operator( retained.lock() );
    report.check<T>(
        !retained.expired() && f.solver.solve( initial, x ), "rebinding the same shared object preserves its owner"
    );
    report.check<T>(
        f.solver.solve( &f.problem, nullptr, nullptr, initial, x ) && !retained.expired() &&
            f.solver.solve( initial, x ),
        "one-shot solves do not release a shared binding"
    );

    auto                        replacement = std::make_shared<problem_type>();
    std::weak_ptr<problem_type> rebound     = replacement;
    f.solver.set_nonlinear_operator( std::move( replacement ) );
    report.check<T>(
        !replacement && retained.expired() && !rebound.expired() && f.solver.solve( initial, x ),
        "shared rebinding releases the old owner and accepts transferred ownership"
    );
    f.solver.set_nonlinear_operator( f.problem_owner );
    report.check<T>(
        rebound.expired() && f.solver.solve( initial, x ), "binding another shared object releases shared ownership"
    );

    for ( bool clear_with_shared : { false, true } )
    {
        problem  = std::make_shared<problem_type>();
        retained = problem;
        f.solver.set_nonlinear_operator( problem );
        problem.reset();
        if ( clear_with_shared )
            f.solver.set_nonlinear_operator( std::shared_ptr<problem_type>{} );
        else
            f.solver.set_nonlinear_operator( nullptr );
        x[0]       = T( 9 );
        bool threw = false;
        try
        {
            f.solver.solve( initial, x );
        }
        catch ( const std::logic_error & )
        {
            threw = true;
        }
        report.check<T>(
            retained.expired() && threw && x[0] == T( 9 ),
            "clearing either clearing form releases the problem and restores the unbound guard"
        );
    }
    report.check<T>(
        f.space->lifecycle_calls == lifecycle_calls, "shared bindings require no additional vector storage"
    );

    problem  = std::make_shared<problem_type>();
    retained = problem;
    std::shared_ptr<problem_type> null_alias( problem, nullptr );
    f.solver.set_nonlinear_operator( std::move( null_alias ) );
    problem.reset();
    report.check<T>( !retained.expired(), "null alias retains its supplied ownership" );
    f.solver.set_nonlinear_operator( nullptr );
    report.check<T>( retained.expired(), "nullptr clearing also releases a null alias owner" );

    {
        typename fixture::solver_type solver( f.space, nullptr, f.iteration, f.strategy );
        problem  = std::make_shared<problem_type>();
        retained = problem;
        solver.set_nonlinear_operator( std::move( problem ) );
        report.check<T>( !retained.expired(), "shared problem remains alive until solver destruction" );
    }
    report.check<T>( retained.expired(), "solver destruction releases the retained problem" );
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
    fixture     f;
    auto        strategy = std::make_shared<strategy_type>( f.space, nullptr );
    solver_type solver( f.space, nullptr, f.iteration, strategy );
    auto        first_owner = std::make_shared<hook_type>(), second_owner = std::make_shared<hook_type>();
    auto       &first  = *first_owner;
    auto       &second = *second_owner;
    typename fixture::vector_type initial{ T( 2 ) }, x{ T( 9 ) };
    solver.set_nonlinear_operator( f.problem_owner );
    solver.set_project_operator( first_owner );
    solver.set_quality_functor( first_owner );
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
    solver.set_project_operator( second_owner );
    solver.set_quality_functor( nullptr );
    report.check<T>(
        solver.solve( initial, x ) && second.projection_calls == 2 && first.quality_calls == 3 &&
            second.quality_calls == 1,
        "quality can be cleared independently while projection is rebound"
    );
    solver.set_quality_functor( second_owner );
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

    auto                     projection          = std::make_shared<hook_type>();
    auto                     quality             = std::make_shared<hook_type>();
    std::weak_ptr<hook_type> retained_projection = projection, retained_quality = quality;
    solver.set_project_operator( projection );
    solver.set_quality_functor( quality );
    report.check<T>(
        projection.use_count() == 2 && quality.use_count() == 2, "shared hooks retain the supplied ownership"
    );
    auto *projection_address = projection.get();
    auto *quality_address    = quality.get();
    projection.reset();
    quality.reset();
    report.check<T>(
        !retained_projection.expired() && !retained_quality.expired() && solver.solve( initial, x ),
        "shared hooks survive release by their caller"
    );
    solver.set_project_operator( retained_projection.lock() );
    solver.set_quality_functor( retained_quality.lock() );
    report.check<T>(
        !retained_projection.expired() && !retained_quality.expired() && solver.solve( initial, x ),
        "rebinding shared hooks preserves their owners"
    );
    report.check<T>(
        projection_address->projection_calls == 2 && quality_address->quality_calls == 2,
        "retained hooks are invoked on the original objects"
    );
    report.check<T>(
        solver.solve( &f.problem, nullptr, nullptr, initial, x ) && !retained_projection.expired() &&
            !retained_quality.expired() && projection_address->projection_calls == 2 &&
            quality_address->quality_calls == 2,
        "one-shot solves leave shared hooks retained but unused"
    );

    projection = std::make_shared<hook_type>();
    quality    = std::make_shared<hook_type>();
    solver.set_project_operator( projection );
    solver.set_quality_functor( quality );
    report.check<T>(
        retained_projection.expired() && retained_quality.expired() && solver.solve( initial, x ) &&
            projection->projection_calls == 1 && quality->quality_calls == 1,
        "shared hook replacement releases old owners and invokes replacements"
    );
    retained_projection = projection;
    retained_quality    = quality;
    projection.reset();
    quality.reset();
    solver.set_project_operator( first_owner );
    solver.set_quality_functor( first_owner );
    report.check<T>(
        retained_projection.expired() && retained_quality.expired() && solver.solve( initial, x ),
        "shared hook replacement releases shared owners"
    );
    for ( bool clear_with_shared : { false, true } )
    {
        projection          = std::make_shared<hook_type>();
        quality             = std::make_shared<hook_type>();
        retained_projection = projection;
        retained_quality    = quality;
        solver.set_project_operator( std::move( projection ) );
        solver.set_quality_functor( std::move( quality ) );
        if ( clear_with_shared )
        {
            solver.set_project_operator( std::shared_ptr<hook_type>{} );
            solver.set_quality_functor( std::shared_ptr<hook_type>{} );
        }
        else
        {
            solver.set_project_operator( nullptr );
            solver.set_quality_functor( nullptr );
        }
        report.check<T>(
            retained_projection.expired() && retained_quality.expired() && solver.solve( initial, x ),
            "clearing either hook clearing form releases both owners"
        );
    }
}

int main()
{
    test_report report;
    test_binding<float>( report );
    test_binding<double>( report );
    test_binding<long double>( report );
    test_shared_binding<float>( report );
    test_shared_binding<double>( report );
    test_shared_binding<long double>( report );
    test_hooks<float>( report );
    test_hooks<double>( report );
    test_hooks<long double>( report );
    return report.finish();
}
