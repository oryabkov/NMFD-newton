#include "newton_test_fixture.h"

using namespace nmfd::tests;

template <class T>
void test( test_report &report )
{
    using fixture     = newton_fixture<T>;
    using vector_type = typename fixture::vector_type;
    for ( bool full_check : { false, true } )
    {
        typename fixture::params_type p;
        p.perform_full_check = full_check;
        fixture f( p );
        f.linear->converged = false;
        vector_type x{ T( 2 ) };
        const bool  success = f.solver.solve( &f.problem, nullptr, nullptr, x );
        report.check<T>(
            !success && f.strategy->get_result_status() == 6 && x[0] == T( 2 ) && f.linear->calls == 1,
            "failed increment is rejected independently of vector validation policy"
        );
        f.linear->converged = true;
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1 ),
            "new solve clears the failed increment flag"
        );
        report.check<T>(
            ( f.space->validity_checks != 0 ) == full_check,
            "perform_full_check controls vector scans without disabling residual checks"
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
    for ( bool full_check : { false, true } )
    {
        for ( T value :
              { std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(),
                -std::numeric_limits<T>::infinity() } )
        {
            typename fixture::params_type p;
            p.perform_full_check = full_check;
            fixture     f( p );
            vector_type x{ value };
            report.check<T>(
                !f.solver.solve( &f.problem, nullptr, nullptr, x ) &&
                    f.strategy->get_result_status() == ( full_check ? 11 : 2 ) && f.linear->calls == 0,
                "nonfinite input terminates before requesting a correction"
            );
            report.check<T>(
                f.problem.residual_calls == ( full_check ? 0u : 1u ) &&
                    ( std::isnan( value ) ? std::isnan( x[0] ) : x[0] == value ),
                "full validation bypasses residual evaluation and preserves invalid input"
            );
        }
    }
    {
        typename fixture::params_type p;
        p.perform_full_check      = false;
        p.max_backtracking_trials = 3;
        fixture f( p );
        f.problem.reference_state = T( 2 );
        f.linear->mode            = correction::nonfinite;
        vector_type x{ T( 2 ) };
        report.check<T>(
            !f.solver.solve( &f.problem, nullptr, nullptr, x ) && f.strategy->get_result_status() == 3 &&
                f.problem.trial_calls == 3 && x[0] == T( 2 ),
            "nonfinite trial residuals remain bounded when vector checks are disabled"
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

template <class T>
void test_projection( test_report &report )
{
    using fixture     = newton_fixture<T>;
    using vector_type = typename fixture::vector_type;
    struct projection_type
    {
        unsigned int calls = 0, invalid_trials = 0;
        void         apply( vector_type &x )
        {
            if ( ++calls <= invalid_trials )
                x[0] = std::numeric_limits<T>::infinity();
        }
    };
    using strategy_type = nmfd::solvers::default_convergence_strategy<
        typename fixture::space_type, scfd::utils::log_std, typename fixture::problem_type, projection_type>;

    for ( bool weight_cutoff : { false, true } )
    {
        auto                           space = std::make_shared<typename fixture::space_type>();
        typename strategy_type::params p;
        p.max_backtracking_trials = 2;
        p.newton_weight_threshold = weight_cutoff ? T( .75 ) : T( 1e-12 );
        strategy_type                  strategy( space, nullptr, p );
        typename fixture::problem_type problem;
        problem.reference_state = T( 2 );
        projection_type projection{ 0, 2 };
        vector_type     x{ T( 2 ) }, dx{ T( -1 ) };
        strategy.check_convergence( &problem, &projection, nullptr, x, dx );
        report.check<T>(
            strategy.check_convergence( &problem, &projection, nullptr, x, dx ) && strategy.get_result_status() == 10 &&
                x[0] == T( 2 ) && projection.calls == ( weight_cutoff ? 1u : 2u ),
            "projection failure is preserved at both weight and trial cutoffs"
        );
        report.check<T>(
            problem.residual_calls == 2 && problem.invalid_state_calls == 0,
            "invalid projected states never reach residual evaluation"
        );
    }
    for ( bool invalid_residual : { false, true } )
    {
        auto                           space = std::make_shared<typename fixture::space_type>();
        typename strategy_type::params p;
        p.max_backtracking_trials = 2;
        strategy_type                  strategy( space, nullptr, p );
        typename fixture::problem_type problem;
        problem.reference_state = T( 2 );
        problem.invalid_trials  = invalid_residual;
        projection_type projection{ 0, 1 };
        vector_type     x{ T( 2 ) }, dx{ T( -1 ) };
        strategy.check_convergence( &problem, &projection, nullptr, x, dx );
        const bool finish = strategy.check_convergence( &problem, &projection, nullptr, x, dx );
        report.check<T>(
            finish == invalid_residual && strategy.get_result_status() == ( invalid_residual ? 3 : 1 ) &&
                x[0] == ( invalid_residual ? T( 2 ) : T( 1.5 ) ) && projection.calls == 2 &&
                problem.invalid_state_calls == 0,
            "a new trial replaces projection status with recovery or residual failure"
        );
    }
}

int main()
{
    test_report report;
    test<float>( report );
    test<double>( report );
    test<long double>( report );
    test_projection<float>( report );
    test_projection<double>( report );
    test_projection<long double>( report );
    return report.finish();
}
