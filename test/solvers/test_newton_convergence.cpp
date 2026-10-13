#include "newton_test_fixture.h"

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
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && f.linear->calls == 0,
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
        p.max_iters_num            = 1;
        p.max_backtracking_trials  = 3;
        p.check_linear_solver_fail = true;
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
        typename fixture::params_type p;
        p.check_linear_solver_fail = true;
        fixture f( p );
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

int main()
{
    test_report report;
    test<float>( report );
    test<double>( report );
    test<long double>( report );
    return report.finish();
}
