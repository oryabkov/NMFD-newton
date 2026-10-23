#include "newton_test_fixture.h"

namespace
{

// Change assessment only; Newton still solves the original linear equation x-1=0.
struct weighted_residual_norm
{
    template <class Space, class Operator>
    static auto
    evaluate( Space &space, Operator &op, const typename Space::vector_type &x, typename Space::vector_type &work )
    {
        op.apply( x, work );
        return typename Space::scalar_type( 10 ) * space.norm_l2( work );
    }
};

template <class T>
void test_norm_policy( nmfd::tests::test_report &report )
{
    using fixture_type   = nmfd::tests::newton_fixture<T>;
    using space_type     = typename fixture_type::space_type;
    using problem_type   = typename fixture_type::problem_type;
    using log_type       = scfd::utils::log_std;
    using projector_type = nmfd::operations::ident_operator<space_type>;
    using quality_type   = nmfd::operations::zero_functional<space_type>;
    using strategy_type  = nmfd::solvers::default_convergence_strategy<
         space_type, log_type, problem_type, projector_type, quality_type, weighted_residual_norm>;
    using solver_type = nmfd::solvers::nonlinear_solver<
        space_type, log_type, problem_type, typename fixture_type::iteration_type, projector_type, quality_type,
        strategy_type>;
    fixture_type                   f;
    typename strategy_type::params params;
    params.abs_tol             = T( .04 );
    params.store_norms_history = true;
    auto        strategy       = std::make_shared<strategy_type>( f.space, nullptr, params );
    solver_type solver( f.space, nullptr, f.iteration, strategy );
    solver.set_nonlinear_operator( f.problem_owner );
    typename fixture_type::vector_type state;
    state[0]               = T( 1.01 );
    const auto allocations = f.space->allocations;
    report.check<T>(
        solver.solve( state ) && strategy->get_number_of_iterations() > 0, "custom norm controls initial stopping"
    );
    f.linear->mode = nmfd::tests::correction::half;
    state[0]       = T( 2 );
    report.check<T>(
        solver.solve( state ) && T( 10 ) * std::abs( state[0] - T( 1 ) ) <= params.abs_tol,
        "custom norm controls trial stopping"
    );
    const auto &history = *strategy->get_norms_history_handle();
    report.check<T>(
        !history.empty() && std::abs( history.front() - T( 5 ) ) < T( .001 ), "history uses the selected norm"
    );
    report.check<T>( f.space->allocations == allocations, "custom norm does not allocate workspace" );
    f.problem.invalid_residual = true;
    state[0]                   = T( 2 );
    report.check<T>(
        !solver.solve( state ) && strategy->get_result_status() == 2 && state[0] == T( 2 ),
        "custom norm preserves nonfinite failure reporting"
    );
}

}

int main()
{
    nmfd::tests::test_report report;
    test_norm_policy<float>( report );
    test_norm_policy<double>( report );
    test_norm_policy<long double>( report );
    return report.finish();
}
