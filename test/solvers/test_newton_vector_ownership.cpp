#include "newton_test_fixture.h"

using namespace nmfd::tests;

template <class T>
void test( test_report &report )
{
    using fixture = newton_fixture<T, T *>;
    auto linear   = std::make_shared<typename fixture::linear_solver_type>();
    for ( unsigned int event = 1; event <= 2; ++event )
    {
        auto space     = std::make_shared<typename fixture::space_type>();
        space->fail_at = event;
        bool threw     = false;
        try
        {
            typename fixture::iteration_type iteration( space, linear );
        }
        catch ( const std::runtime_error & )
        {
            threw = true;
        }
        report.check<T>( threw && space->balanced(), "partial iteration construction releases scratch" );
    }
    for ( unsigned int event = 1; event <= 6; ++event )
    {
        auto space     = std::make_shared<typename fixture::space_type>();
        space->fail_at = event;
        bool threw     = false;
        try
        {
            typename fixture::strategy_type strategy( space, nullptr );
        }
        catch ( const std::runtime_error & )
        {
            threw = true;
        }
        report.check<T>( threw && space->balanced(), "partial strategy construction releases scratch" );
    }
    for ( bool supplied_strategy : { false, true } )
    {
        const unsigned int events = supplied_strategy ? 2 : 8;
        for ( unsigned int event = 1; event <= events; ++event )
        {
            auto space = std::make_shared<typename fixture::space_type>();
            bool threw = false;
            {
                auto iteration = std::make_shared<typename fixture::iteration_type>( space, linear );
                auto strategy =
                    supplied_strategy ? std::make_shared<typename fixture::strategy_type>( space, nullptr ) : nullptr;
                const auto active = space->allocations;
                space->fail_at    = space->lifecycle_calls + event;
                try
                {
                    typename fixture::solver_type solver( space, nullptr, iteration, strategy );
                }
                catch ( const std::runtime_error & )
                {
                    threw = true;
                }
                report.check<T>(
                    threw && space->allocations - space->frees == active && space->starts - space->stops == active,
                    "failed solver construction releases scratch without disturbing shared dependencies"
                );
            }
            report.check<T>( space->balanced(), "partial solver construction balances all vector lifetimes" );
        }
    }
    {
        auto                          space = std::make_shared<typename fixture::space_type>();
        typename fixture::params_type p;
        p.store_norms_history = true;
        p.max_iters_num       = std::numeric_limits<unsigned int>::max();
        bool threw            = false;
        try
        {
            typename fixture::strategy_type strategy( space, nullptr, p );
        }
        catch ( const std::invalid_argument & )
        {
            threw = true;
        }
        report.check<T>(
            threw && space->allocations == 0 && space->balanced(), "invalid history limit is rejected before allocation"
        );
    }
    auto space = std::make_shared<typename fixture::space_type>();
    {
        typename fixture::iteration_type iteration( space, linear );
        typename fixture::problem_type   problem;
        T                                state = T( 2 ), direction = T( 0 );
        T                               *x = &state, *dx = &direction;
        for ( failure_point phase :
              { failure_point::linearization, failure_point::residual, failure_point::jacobi_access,
                failure_point::operator_setup, failure_point::linear_solve } )
        {
            problem.fail = linear->fail = phase;
            bool threw                  = false;
            try
            {
                iteration.solve( problem, x, dx );
            }
            catch ( const std::runtime_error & )
            {
                threw = true;
            }
            report.check<T>(
                threw && space->allocations == 1 && space->starts == 1 && space->frees == 0 && space->stops == 0,
                "solve exception preserves reusable active scratch"
            );
            problem.fail = linear->fail = failure_point::none;
            report.check<T>(
                iteration.solve( problem, x, dx ) && direction == T( -1 ), "iteration reusable after exception"
            );
        }
    }
    report.check<T>( space->balanced(), "iteration destruction balances vector lifetime" );
    space = std::make_shared<typename fixture::space_type>();
    {
        auto                           strategy  = std::make_shared<typename fixture::strategy_type>( space, nullptr );
        auto                           iteration = std::make_shared<typename fixture::iteration_type>( space, linear );
        typename fixture::solver_type  solver( space, nullptr, iteration, strategy );
        typename fixture::problem_type problem;
        T                              state = T( 2 );
        T                             *x     = &state;
        for ( unsigned int attempt = 0; attempt < 3; ++attempt )
        {
            state = T( 2 );
            report.check<T>(
                solver.solve( &problem, nullptr, nullptr, x ) && state == T( 1 ), "repeated Newton solve succeeds"
            );
        }
        problem.fail = failure_point::residual;
        state        = T( 2 );
        bool threw   = false;
        try
        {
            solver.solve( &problem, nullptr, nullptr, x );
        }
        catch ( const std::runtime_error & )
        {
            threw = true;
        }
        report.check<T>( threw && state == T( 2 ), "residual exception preserves input state" );
        problem.fail = failure_point::none;
        report.check<T>(
            solver.solve( &problem, nullptr, nullptr, x ) && state == T( 1 ), "Newton reusable after exception"
        );
        report.check<T>(
            space->allocations == 5 && space->starts == 5 && space->frees == 0 && space->stops == 0,
            "repeated solves neither allocate nor reactivate scratch"
        );
    }
    report.check<T>( space->balanced(), "Newton owner destruction balances all vector lifetimes" );
}

int main()
{
    test_report report;
    test<float>( report );
    test<double>( report );
    test<long double>( report );
    return report.finish();
}
