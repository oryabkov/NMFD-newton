#include "newton_test_fixture.h"

using namespace nmfd::tests;

template <class T>
void test( test_report &report )
{
    using fixture     = newton_fixture<T>;
    using strategy    = typename fixture::strategy_type;
    using vector_type = typename fixture::vector_type;
    for ( unsigned int limit : { 2u, 3u } )
    {
        typename fixture::params_type p;
        p.max_backtracking_trials = limit;
        fixture f( p );
        f.problem.model = equation::cubic;
        vector_type x{ T( 1 ) }, dx{ T( -1 ) };
        report.check<T>(
            !f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx ), "initialize cubic baseline"
        );
        const bool finish = f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx );
        report.check<T>(
            finish == ( limit == 2 ) && f.strategy->get_result_status() == ( limit == 2 ? 8 : 1 ) &&
                f.problem.trial_calls == limit && x[0] == ( limit == 2 ? T( 1 ) : T( .75 ) ),
            "cubic accepts weight 0.25 on final allowed trial, otherwise fails at cap"
        );
    }
    {
        fixture f;
        f.problem.model = equation::cubic;
        typename fixture::vector_type x{ T( -2 ) };
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) &&
                std::abs( x[0] - T( -1.7692923542386314L ) ) < T( 1e-5 ),
            "cubic converges from negative seed"
        );
    }
    const T nan = std::numeric_limits<T>::quiet_NaN(), inf = std::numeric_limits<T>::infinity();
    for ( T multiplier : { T( -1 ), T( 0 ), T( 1 ), T( 2 ), nan, inf, -inf } )
    {
        auto                          space = std::make_shared<typename fixture::space_type>();
        typename fixture::params_type p;
        p.newton_weight_mul = multiplier;
        bool rejected       = false;
        try
        {
            strategy convergence( space, nullptr, p );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>( rejected && space->allocations == 0, "invalid multiplier rejected before allocation" );
    }
    struct weights
    {
        T initial, minimum;
    };
    const weights invalid[] = { { nan, T( .125 ) },    { inf, T( .125 ) },     { -inf, T( .125 ) },
                                { T( 0 ), T( .125 ) }, { T( -1 ), T( .125 ) }, { T( .125 ), T( .25 ) },
                                { T( 1 ), nan },       { T( 1 ), inf },        { T( 1 ), -inf },
                                { T( 1 ), T( 0 ) },    { T( 1 ), T( -1 ) } };
    for ( const weights &w : invalid )
    {
        auto                          space = std::make_shared<typename fixture::space_type>();
        typename fixture::params_type p;
        p.newton_weight_initial   = w.initial;
        p.newton_weight_threshold = w.minimum;
        bool rejected             = false;
        try
        {
            strategy convergence( space, nullptr, p );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>( rejected && space->allocations == 0, "invalid weights rejected before allocation" );
        fixture f;
        const T abs_tol = f.strategy->abs_tol(), rel_tol = f.strategy->rel_tol();
        rejected = false;
        try
        {
            f.strategy->set_convergence_constants(
                T( .01 ), 0, T( 2 ), 0, w.initial, true, false, 1, T( .5 ), w.minimum, false
            );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>(
            rejected && f.strategy->abs_tol() == abs_tol && f.strategy->rel_tol() == rel_tol,
            "failed setter leaves tolerances unchanged"
        );
        vector_type x{ T( 2 ) };
        f.linear->mode = correction::half;
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1 ) + std::ldexp( T( 1 ), -20 ) &&
                f.linear->calls == 20 && f.strategy->get_norms_history_handle()->empty(),
            "failed setter leaves iteration limit and damping unchanged"
        );
    }
    for ( unsigned int limit : { 0u, 1u } )
    {
        auto                          space = std::make_shared<typename fixture::space_type>();
        typename fixture::params_type p;
        p.max_backtracking_trials = limit;
        p.newton_weight_initial = p.newton_weight_threshold = T( .25 );
        bool rejected                                       = false;
        try
        {
            strategy convergence( space, nullptr, p );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>(
            rejected == ( limit == 0 ) && space->balanced(), "positive limit and equal initial/minimum weights"
        );
    }
    {
        fixture f;
        f.strategy->set_convergence_constants(
            T( 1e-6 ), 100, T( 1 ), 0, T( .25 ), false, false, 10, T( 0 ), T( .125 )
        );
        vector_type x{ T( 2 ) }, dx{ T( -1 ) };
        f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx );
        report.check<T>(
            !f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx ) && x[0] == T( 1.75 ),
            "valid setter applies new damping"
        );
    }
#ifdef NMFD_ENABLE_NLOHMANN
    {
        typename fixture::params_type p;
        p.from_json( nlohmann::json{ { "max_backtracking_trials", 5 }, { "check_linear_solver_fail", true } } );
        report.check<T>(
            p.to_json().at( "max_backtracking_trials" ) == 5 && p.to_json().at( "check_linear_solver_fail" ) == true,
            "JSON round trip preserves failure policy"
        );
        p.from_json( nlohmann::json{ { "max_backtracking_trials", 0 } } );
        auto space    = std::make_shared<typename fixture::space_type>();
        bool rejected = false;
        try
        {
            strategy convergence( space, nullptr, p );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>( rejected && space->allocations == 0, "invalid JSON rejected before allocation" );
    }
#endif
}

int main()
{
    test_report report;
    test<float>( report );
    test<double>( report );
    test<long double>( report );
    return report.finish();
}
