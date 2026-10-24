#include "newton_test_fixture.h"
#include <nmfd/operations/pair_vector_space.h>

using namespace nmfd::tests;

template <class T>
void test( test_report &report )
{
    using fixture     = newton_fixture<T>;
    using vector_type = typename fixture::vector_type;
    {
        typename fixture::params_type p;
        p.abs_tol = T( .125 );
        p.rel_tol = T( .5 );
        fixture     f( p );
        vector_type x{ T( 1.125 ) };
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && f.linear->calls == 0 &&
                !f.strategy->used_relaxed_tolerance(),
            "initial absolute tolerance includes equality"
        );
        f.strategy->set_tolerance( T( 0 ) );
        x[0] = T( 1 );
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && f.linear->calls == 0,
            "exact root converges with zero tolerance"
        );
    }
    {
        typename fixture::params_type p;
        p.abs_tol             = T( 0 );
        p.rel_tol             = T( .125 );
        p.store_norms_history = true;
        fixture f( p );
        f.linear->mode = correction::half;
        vector_type x{ T( 9 ) };
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 2 ) && f.linear->calls == 3 &&
                f.strategy->rel_tol_base() == T( 8 ),
            "relative tolerance uses initial solve baseline"
        );
        x[0]             = T( 1.0625 );
        const auto calls = f.linear->calls;
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1.0078125 ) &&
                f.linear->calls == calls + 3 && f.strategy->rel_tol_base() == T( .0625 ) &&
                f.strategy->get_norms_history_handle()->size() == 3,
            "repeated solve clears baseline and norm history"
        );
    }
    for ( int expected : { 0, 4, 5, 8, 9 } )
    {
        typename fixture::params_type p;
        p.max_iters_num           = 1;
        p.max_backtracking_trials = 3;
        if ( expected == 4 )
            p.newton_weight_threshold = T( .75 );
        if ( expected == 5 )
            p.stagnation_max = 0;
        if ( expected == 8 )
            p.newton_weight_mul = T( .999999 );
        fixture f( p );
        f.problem.model = ( expected == 4 || expected == 8 ) ? equation::cubic : equation::linear;
        f.linear->mode  = ( expected == 5 || expected == 9 ) ? correction::zero : correction::exact;
        vector_type x{ f.problem.model == equation::cubic ? T( 1 ) : T( 2 ) };
        const bool  success = f.solver.solve( &f.problem, nullptr, nullptr, x );
        report.check<T>(
            success == ( expected == 0 ) && f.strategy->get_result_status() == expected && f.linear->calls == 1,
            "last correction preserves convergence, failure, and iteration-exhaustion status"
        );
    }
    {
        typename fixture::params_type p;
        p.stagnation_max = 1;
        fixture     f( p );
        vector_type x{ T( 2 ) }, dx{ T( 0 ) };
        f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx );
        report.check<T>(
            !f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx ), "first stagnant correction continues"
        );
        dx[0] = T( -.25 );
        report.check<T>(
            !f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx ),
            "improving correction resets stagnation"
        );
        dx[0] = T( 0 );
        report.check<T>(
            !f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx ),
            "stagnation restarted after improvement"
        );
        report.check<T>(
            f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx ) &&
                f.strategy->get_result_status() == 5,
            "consecutive stagnant corrections reach configured limit"
        );
        f.strategy->reset_iterations();
        x[0] = T( 2 );
        f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx );
        report.check<T>(
            !f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx ), "new solve clears stagnation counter"
        );
    }
    for ( bool out_of_place : { false, true } )
    {
        fixture f;
        f.linear->mode       = correction::half;
        f.linear->fail_after = 1;
        vector_type initial{ T( 2 ) }, x{ T( 2 ) };
        const bool  success = out_of_place ? f.solver.solve( &f.problem, nullptr, nullptr, initial, x )
                                           : f.solver.solve( &f.problem, nullptr, nullptr, x );
        report.check<T>(
            !success && f.strategy->get_result_status() == 6 && initial[0] == T( 2 ) &&
                x[0] == ( out_of_place ? T( 2 ) : T( 1.5 ) ),
            "out-of-place rollback; in-place retains accepted iterate"
        );
        f.linear->calls      = 0;
        f.linear->fail_after = 0;
        f.linear->mode       = correction::exact;
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1 ),
            "failed linear status resets on restart"
        );
    }
}

template <class T>
void test_relaxed_convergence( test_report &report )
{
    using fixture     = newton_fixture<T>;
    using vector_type = typename fixture::vector_type;
    for ( int reason : { 4, 5, 8, 9 } )
    {
        for ( bool allow_relaxed : { false, true } )
        {
            typename fixture::params_type p;
            p.abs_tol                 = T( .125 );
            p.relax_tolerance_factor  = allow_relaxed ? T( 8 ) : T( 4 );
            p.max_iters_num           = 1;
            p.max_backtracking_trials = reason == 8 ? 1 : 2;
            p.newton_weight_threshold = reason == 4 ? T( .75 ) : T( 1e-12 );
            p.stagnation_max          = reason == 5 ? 0 : 10;
            fixture    f( p );
            const bool cubic = reason == 4 || reason == 8;
            f.problem.model  = cubic ? equation::cubic : equation::linear;
            f.linear->mode   = cubic ? correction::exact : correction::zero;
            vector_type x{ cubic ? T( 1 ) : T( 2 ) };
            report.check<T>( !f.strategy->used_relaxed_tolerance(), "relaxed acceptance starts unset" );
            report.check<T>(
                f.solver.solve( &f.problem, nullptr, nullptr, x ) == allow_relaxed &&
                    f.strategy->get_result_status() == ( allow_relaxed ? 0 : reason ) &&
                    f.strategy->used_relaxed_tolerance() == allow_relaxed && x[0] == ( cubic ? T( 1 ) : T( 2 ) ),
                "weight, stagnation, backtracking, and iteration limits require a qualifying saved residual"
            );
            if ( allow_relaxed )
            {
                f.problem.invalid_residual = true;
                report.check<T>(
                    !f.solver.solve( &f.problem, nullptr, nullptr, x ) && !f.strategy->used_relaxed_tolerance(),
                    "failed repeated solve clears relaxed acceptance"
                );
                f.problem.invalid_residual = false;
                f.problem.model            = equation::linear;
                f.linear->mode             = correction::exact;
                x[0]                       = T( 2 );
                report.check<T>(
                    f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1 ) &&
                        !f.strategy->used_relaxed_tolerance(),
                    "strict repeated solve does not inherit relaxed acceptance"
                );
            }
        }
    }
    for ( bool relative : { false, true } )
    {
        typename fixture::params_type p;
        p.abs_tol                = relative ? T( 0 ) : T( .125 );
        p.rel_tol                = relative ? T( .125 ) : T( 0 );
        p.relax_tolerance_factor = T( 4 );
        p.max_iters_num          = 2;
        p.newton_weight_initial  = T( .5 );
        p.store_norms_history    = true;
        fixture     f( p );
        vector_type x{ T( 2 ) };
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1.25 ) && f.linear->calls == 2 &&
                f.strategy->used_relaxed_tolerance() && f.strategy->tol() == T( .125 ) &&
                f.strategy->get_norms_history_handle()->size() == 2 &&
                f.strategy->get_norms_history_handle()->back() == T( .25 ),
            "final improving correction is cached; relaxed tolerance is terminal and does not replace requested "
            "tolerance"
        );
    }
    {
        typename fixture::params_type p;
        p.abs_tol                = T( .125 );
        p.relax_tolerance_factor = T( 2 );
        p.maximum_norm_increase  = T( 2 );
        p.max_iters_num          = 2;
        p.store_norms_history    = true;
        fixture     f( p );
        vector_type x{ T( 2 ) }, dx{ T( -.75 ) };
        f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx );
        f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx );
        dx[0] = T( .25 );
        report.check<T>(
            f.strategy->check_convergence( &f.problem, nullptr, nullptr, x, dx ) &&
                f.strategy->get_result_status() == 0 && f.strategy->used_relaxed_tolerance() && x[0] == T( 1.25 ) &&
                f.strategy->get_norms_history_handle()->back() == T( .25 ),
            "relaxed acceptance restores an earlier best state and reports its residual at equality"
        );
    }
}

template <class T>
void test_relaxed_pair( test_report &report )
{
    using component_space = nmfd::operations::static_vector_space<T, 1, scfd::static_vec::vec<T, 1>>;
    using space_type      = nmfd::operations::pair_vector_space<component_space, component_space>;
    using vector_type     = typename space_type::vector_type;
    struct problem_type
    {
        void apply( const vector_type &x, vector_type &out ) const
        {
            out.first[0]  = x.first[0] - T( 1 );
            out.second[0] = x.second[0] - T( 2 );
        }
    };
    using strategy_type = nmfd::solvers::default_convergence_strategy<space_type, scfd::utils::log_std, problem_type>;
    auto space =
        std::make_shared<space_type>( std::make_shared<component_space>(), std::make_shared<component_space>() );
    typename strategy_type::params p;
    p.abs_tol                = T( .125 );
    p.relax_tolerance_factor = T( 4 );
    p.maximum_norm_increase  = T( 2 );
    p.max_iters_num          = 2;
    strategy_type strategy( space, nullptr, p );
    problem_type  problem;
    vector_type   x{ { T( 2 ) }, { T( 3 ) } }, dx{ { T( -.75 ) }, { T( -.75 ) } };
    strategy.check_convergence( &problem, nullptr, nullptr, x, dx );
    strategy.check_convergence( &problem, nullptr, nullptr, x, dx );
    dx.first[0] = dx.second[0] = T( .25 );
    report.check<T>(
        strategy.check_convergence( &problem, nullptr, nullptr, x, dx ) && strategy.get_result_status() == 0 &&
            strategy.used_relaxed_tolerance() && x.first[0] == T( 1.25 ) && x.second[0] == T( 2.25 ),
        "relaxed restoration keeps paired state and parameter together"
    );
}

int main()
{
    test_report report;
    test<float>( report );
    test<double>( report );
    test<long double>( report );
    test_relaxed_convergence<float>( report );
    test_relaxed_convergence<double>( report );
    test_relaxed_convergence<long double>( report );
    test_relaxed_pair<float>( report );
    test_relaxed_pair<double>( report );
    test_relaxed_pair<long double>( report );
    return report.finish();
}
