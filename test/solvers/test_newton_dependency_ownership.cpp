#include "newton_test_fixture.h"

using namespace nmfd::tests;

template <class T>
void test_dependencies( test_report &report )
{
    using fixture        = newton_fixture<T>;
    using space_type     = typename fixture::space_type;
    using iteration_type = typename fixture::iteration_type;
    using strategy_type  = typename fixture::strategy_type;
    using solver_type    = typename fixture::solver_type;
    using problem_type   = typename fixture::problem_type;
    using log_type       = scfd::utils::log_std;
    static_assert(
        !std::is_invocable_v<decltype( &solver_type::set_nonlinear_operator ), solver_type &, problem_type *>
    );
    static_assert(
        std::is_constructible_v<solver_type, std::shared_ptr<space_type>, log_type *, std::shared_ptr<iteration_type>>
    );
    static_assert(
        !std::is_constructible_v<
            solver_type, std::shared_ptr<space_type>, std::shared_ptr<log_type>, std::shared_ptr<iteration_type>>
    );
    static_assert( std::is_constructible_v<strategy_type, std::shared_ptr<space_type>, log_type *> );
    static_assert( !std::is_constructible_v<strategy_type, std::shared_ptr<space_type>, std::shared_ptr<log_type>> );

    auto       space     = std::make_shared<space_type>();
    auto       linear    = std::make_shared<typename fixture::linear_solver_type>();
    auto       iteration = std::make_shared<iteration_type>( space, linear );
    const auto before    = space->lifecycle_calls;
    for ( unsigned int missing = 0; missing < 5; ++missing )
    {
        bool rejected = false;
        try
        {
            if ( missing < 2 )
                iteration_type invalid( missing == 0 ? nullptr : space, missing == 1 ? nullptr : linear );
            else if ( missing == 2 )
                strategy_type invalid( nullptr, nullptr );
            else
                solver_type invalid( missing == 3 ? nullptr : space, nullptr, missing == 4 ? nullptr : iteration );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>(
            rejected && space->lifecycle_calls == before, "missing dependency fails before vector initialization"
        );
    }

    struct backend_type
    {
        using log_type = scfd::utils::log_std;
        log_type  logger;
        log_type &log()
        {
            return logger;
        }
    };
    for ( unsigned int target = 0; target < 2; ++target )
    {
        bool rejected = false;
        try
        {
            if ( target == 0 )
                typename solver_type::utils invalid( std::shared_ptr<backend_type>{}, space );
            else
                typename strategy_type::utils invalid( std::shared_ptr<backend_type>{}, space );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>(
            rejected && space->lifecycle_calls == before, "missing backend fails before logger access or allocation"
        );
    }
    typename solver_type::utils   default_solver_utils;
    typename strategy_type::utils default_strategy_utils;
    report.check<T>(
        !default_solver_utils.log && !default_strategy_utils.log, "default utilities have no borrowed logger"
    );
    {
        solver_type                   unlogged( space, nullptr, iteration );
        typename fixture::vector_type x{ T( 2 ) };
        auto                          problem = std::make_shared<problem_type>();
        unlogged.set_nonlinear_operator( problem );
        report.check<T>(
            !unlogged.convergence_strategy()->get_log() && unlogged.solve( x ) && x[0] == T( 1 ),
            "Newton supports an absent logger"
        );
    }
    {
        auto        logger = std::make_shared<log_type>();
        solver_type logged( space, logger.get(), iteration );
        report.check<T>(
            logger.use_count() == 1 && logged.convergence_strategy()->get_log() == logger.get(),
            "Newton borrows an externally shared logger without retaining it"
        );
    }
    auto                         backend          = std::make_shared<backend_type>();
    std::weak_ptr<backend_type>  backend_lifetime = backend;
    std::weak_ptr<space_type>    space_lifetime   = space;
    std::weak_ptr<problem_type>  problem_lifetime;
    std::unique_ptr<solver_type> solver;
    {
        typename solver_type::utils_hierarchy utilities( backend, space );
        typename solver_type::utils           borrowed_solver_utils( *backend, space );
        typename strategy_type::utils         borrowed_strategy_utils( *backend, space );
        report.check<T>(
            utilities.log == &backend->log() && utilities.convergence_strategy.log == &backend->log() &&
                borrowed_solver_utils.log == &backend->log() && borrowed_strategy_utils.log == &backend->log() &&
                backend.use_count() == 1,
            "hierarchy utilities borrow loggers from shared and reference backends"
        );
        auto strategy    = std::make_shared<strategy_type>( utilities.convergence_strategy );
        solver           = std::make_unique<solver_type>( space, utilities.log, iteration, strategy );
        auto problem     = std::make_shared<problem_type>();
        problem_lifetime = problem;
        solver->set_nonlinear_operator( problem );
    }
    space.reset();
    linear.reset();
    iteration.reset();
    report.check<T>(
        backend.use_count() == 1 && !space_lifetime.expired() && !problem_lifetime.expired(),
        "Newton retains computational dependencies but borrows its logger"
    );
    typename fixture::vector_type x{ T( 2 ) };
    report.check<T>( solver->solve( x ) && x[0] == T( 1 ), "Newton solves after caller owners are released" );
    {
        auto strategy = solver->convergence_strategy();
        solver.reset();
        report.check<T>(
            problem_lifetime.expired() && backend.use_count() == 1 && strategy->get_log() == &backend->log(),
            "extracted convergence strategy borrows the caller-owned logger"
        );
        report.check<T>(
            space_lifetime.lock()->allocations == space_lifetime.lock()->frees + 3,
            "only convergence workspaces remain after solver destruction"
        );
    }
    backend.reset();
    report.check<T>(
        backend_lifetime.expired() && space_lifetime.expired(), "Newton dependency graph has no ownership cycle"
    );
}

int main()
{
    test_report report;
    test_dependencies<float>( report );
    test_dependencies<double>( report );
    test_dependencies<long double>( report );
    return report.finish();
}
