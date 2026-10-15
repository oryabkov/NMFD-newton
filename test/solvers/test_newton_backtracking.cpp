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
    for ( bool full_check : { false, true } )
    {
        typename fixture::params_type p;
        p.from_json( nlohmann::json{ { "max_backtracking_trials", 5 }, { "perform_full_check", full_check } } );
        report.check<T>(
            p.to_json().at( "max_backtracking_trials" ) == 5 && p.to_json().at( "perform_full_check" ) == full_check,
            "JSON round trip preserves trial limit and vector validation policy"
        );
        fixture     f( p );
        vector_type x{ T( 2 ) };
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && ( f.space->validity_checks != 0 ) == full_check,
            "JSON vector validation policy is used by the solver"
        );
        const auto previous = p.to_json();
        bool       rejected = false;
        try
        {
            p.from_json( nlohmann::json{ { "max_backtracking_trials", 0 } } );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>( rejected && p.to_json() == previous, "invalid JSON leaves parameters unchanged" );
    }
#endif
}

template <class T>
void test_overflow( test_report &report )
{
    using fixture     = newton_fixture<T>;
    using vector_type = typename fixture::vector_type;
    struct problem_type
    {
        unsigned int invalid_state_calls = 0;
        void         apply( const vector_type &x, vector_type &out )
        {
            invalid_state_calls += !std::isfinite( x[0] );
            out[0] = T( 1 ) - x[0] / std::numeric_limits<T>::max();
        }
    };
    struct projection_type
    {
        unsigned int calls = 0;
        void         apply( vector_type & )
        {
            ++calls;
        }
    };
    using strategy_type = nmfd::solvers::default_convergence_strategy<
        typename fixture::space_type, scfd::utils::log_std, problem_type, projection_type>;

    for ( bool full_check : { false, true } )
    {
        auto                           space = std::make_shared<typename fixture::space_type>();
        typename strategy_type::params p;
        p.perform_full_check      = full_check;
        p.max_backtracking_trials = 3;
        strategy_type   strategy( space, nullptr, p );
        problem_type    problem;
        projection_type projection;
        vector_type     x{ T( .75 ) * std::numeric_limits<T>::max() }, dx{ x[0] };
        strategy.check_convergence( &problem, &projection, nullptr, x, dx );
        report.check<T>(
            !strategy.check_convergence( &problem, &projection, nullptr, x, dx ) && strategy.get_result_status() == 1 &&
                x[0] / std::numeric_limits<T>::max() == T( .9375 ),
            "overflowing trials recover at weight 0.25 on the final allowed trial"
        );
        report.check<T>(
            projection.calls == ( full_check ? 1u : 3u ) && problem.invalid_state_calls == ( full_check ? 0u : 2u ),
            "pre-projection validation bypasses both projection and residual for invalid trials"
        );
    }
}

template <class T>
void test_logging( test_report &report )
{
    struct recording_log : scfd::utils::log_std_basic
    {
        std::string messages;
        void        msg( const std::string &text, log_msg_type, int )
        {
            messages += text + '\n';
        }
    };
    using fixture  = newton_fixture<T>;
    using log_type = scfd::utils::log_cformatted<recording_log>;
    struct quality_type
    {
        T calc( const typename fixture::vector_type & ) const
        {
            return T( 3 );
        }
    };
    using strategy_type = nmfd::solvers::default_convergence_strategy<
        typename fixture::space_type, log_type, typename fixture::problem_type,
        nmfd::operations::ident_operator<typename fixture::space_type>, quality_type>;
    auto                           space = std::make_shared<typename fixture::space_type>();
    log_type                       log;
    strategy_type                  strategy( space, &log );
    typename fixture::problem_type problem;
    typename fixture::vector_type  x{ T( 2 ) }, dx{ T( -1 ) };
    quality_type                   quality;
    strategy.check_convergence( &problem, nullptr, nullptr, x, dx );
    report.check<T>(
        strategy.check_convergence( &problem, nullptr, nullptr, x, dx ) && strategy.get_result_status() == 0 &&
            log.messages.find( "returned_ok: true" ) != std::string::npos &&
            log.messages.find( "newton_weight = 1.000000e+00, stagnation = 0" ) != std::string::npos,
        "formatted trial and iteration logs preserve floating and unsigned arguments"
    );
    log.messages.clear();
    typename strategy_type::params p;
    p.stagnation_max = 0;
    strategy_type failing( space, &log, p );
    x[0]  = T( 2 );
    dx[0] = T( 0 );
    failing.check_convergence( &problem, nullptr, &quality, x, dx );
    report.check<T>(
        failing.check_convergence( &problem, nullptr, &quality, x, dx ) && failing.get_result_status() == 5 &&
            log.messages.find( "||x| = 2.000000e+00, relaxed_tol = 1.000000e-06" ) != std::string::npos &&
            log.messages.find( "solution quality = 3.000000e+00" ) != std::string::npos,
        "failure and quality logs preserve floating arguments"
    );
}

int main()
{
    test_report report;
    test<float>( report );
    test<double>( report );
    test<long double>( report );
    test_overflow<float>( report );
    test_overflow<double>( report );
    test_overflow<long double>( report );
    test_logging<float>( report );
    test_logging<double>( report );
    test_logging<long double>( report );
    return report.finish();
}
