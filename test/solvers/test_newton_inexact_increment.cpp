#include "newton_test_fixture.h"
#include <scfd/static_mat/mat.h>

namespace
{
using namespace nmfd::tests;

template <class T>
void test_newton( test_report &report )
{
    using fixture     = newton_fixture<T>;
    using vector_type = typename fixture::vector_type;
    typename fixture::params_type params;
    params.perform_full_check = false;
    fixture strict( params );
    strict.linear->converged = false;
    vector_type state{ T( 2 ) };
    report.check<T>(
        !strict.solver.solve( &strict.problem, nullptr, nullptr, state ) && state[0] == T( 2 ) &&
            strict.strategy->get_result_status() == 6 && strict.space->validity_checks == 0,
        "strict default rejects linear failure without fallback work"
    );

    params.allow_inexact_increment          = true;
    params.maximum_relative_linear_residual = T( .6 );
    fixture inexact( params );
    inexact.linear->converged = false;
    inexact.linear->mode      = correction::half;
    const auto allocations    = inexact.space->allocations;
    report.check<T>(
        inexact.solver.solve( &inexact.problem, nullptr, nullptr, state ) &&
            std::abs( state[0] - T( 1 ) ) <= params.abs_tol && inexact.linear->calls > 1 &&
            inexact.strategy->used_inexact_increment(),
        "accepted inexact corrections still require nonlinear convergence"
    );
    report.check<T>( inexact.space->allocations == allocations, "fallback reuses existing workspace" );
    report.check<T>(
        inexact.solver.solve( &inexact.problem, nullptr, nullptr, state ) &&
            !inexact.strategy->used_inexact_increment(),
        "fresh solve clears fallback diagnostics"
    );
    state[0] = T( 2 );
    inexact.strategy->set_allow_inexact_increment( false );
    report.check<T>(
        !inexact.solver.solve( &inexact.problem, nullptr, nullptr, state ) && state[0] == T( 2 ),
        "disabling fallback restores strict behavior"
    );

    params.maximum_relative_linear_residual = T( .1 );
    fixture rejected( params );
    rejected.linear->converged = false;
    rejected.linear->mode      = correction::half;
    vector_type initial{ T( 2 ) }, result{ T( -5 ) };
    report.check<T>(
        !rejected.solver.solve( &rejected.problem, nullptr, nullptr, initial, result ) && result[0] == initial[0] &&
            !rejected.strategy->used_inexact_increment(),
        "excessive linear defect rejects correction and preserves out-of-place rollback"
    );
    rejected.linear->mode = correction::nonfinite;
    report.check<T>(
        !rejected.solver.solve( &rejected.problem, nullptr, nullptr, state ) && state[0] == T( 2 ),
        "fallback rejects nonfinite corrections even without full checks"
    );

    fixture cubic( params );
    cubic.problem.model = equation::cubic;
    state[0]            = T( 1 );
    vector_type direction{ T( -1 ) };
    cubic.strategy->check_convergence( &cubic.problem, nullptr, nullptr, state, direction );
    cubic.problem.set_linearization_point( state );
    cubic.strategy->increment_ok( false );
    report.check<T>(
        !cubic.strategy->check_convergence( &cubic.problem, nullptr, nullptr, state, direction ) &&
            state[0] == T( .75 ) && cubic.strategy->get_result_status() == 1 &&
            cubic.strategy->used_inexact_increment(),
        "linear fallback preserves ordinary nonlinear backtracking"
    );
}

template <class T>
struct controlled_jacobi
{
    using vector_type          = typename newton_fixture<T>::vector_type;
    mutable unsigned int calls = 0, products = 0;
    bool                 success = true, nonfinite = false, throws = false;
    bool                 apply( const vector_type &in, vector_type &out ) const
    {
        ++calls;
        if ( throws )
            throw std::runtime_error( "injected Jacobian failure" );
        out[0] = nonfinite ? std::numeric_limits<T>::infinity() : in[0];
        return success;
    }
    vector_type operator*( const vector_type &in ) const
    {
        ++products;
        return in;
    }
};

template <class T>
void test_assessment( test_report &report )
{
    using fixture     = newton_fixture<T>;
    using vector_type = typename fixture::vector_type;
    fixture              f;
    controlled_jacobi<T> jacobi;
    vector_type          rhs{ T( 10 ) }, delta{ T( 9 ) };
    report.check<T>(
        !f.strategy->accept_inexact_increment( jacobi, rhs, delta ) && jacobi.calls == 0,
        "disabled assessment does not apply Jacobian"
    );
    f.strategy->set_allow_inexact_increment( true );
    report.check<T>( f.strategy->accept_inexact_increment( jacobi, rhs, delta ), "defect bound includes equality" );
    delta[0] = T( 8 );
    report.check<T>(
        !f.strategy->accept_inexact_increment( jacobi, rhs, delta ), "true defect rejects weak correction"
    );
    delta[0]       = T( 10 );
    jacobi.success = false;
    report.check<T>(
        !f.strategy->accept_inexact_increment( jacobi, rhs, delta ), "failed Jacobian application rejects"
    );
    jacobi.success   = true;
    jacobi.nonfinite = true;
    report.check<T>( !f.strategy->accept_inexact_increment( jacobi, rhs, delta ), "nonfinite defect rejects" );
    jacobi.nonfinite = false;
    rhs[0]           = std::numeric_limits<T>::quiet_NaN();
    report.check<T>( !f.strategy->accept_inexact_increment( jacobi, rhs, delta ), "nonfinite rhs rejects" );
    rhs[0] = delta[0] = T( 0 );
    report.check<T>( f.strategy->accept_inexact_increment( jacobi, rhs, delta ), "zero rhs does not divide by zero" );
    delta[0] = T( 1 );
    report.check<T>( !f.strategy->accept_inexact_increment( jacobi, rhs, delta ), "zero rhs requires zero defect" );
    jacobi.throws   = true;
    bool propagated = false;
    try
    {
        f.strategy->accept_inexact_increment( jacobi, rhs, delta );
    }
    catch ( const std::runtime_error & )
    {
        propagated = true;
    }
    report.check<T>( propagated, "operator exceptions propagate" );
    report.check<T>( jacobi.products == 0, "apply takes precedence over multiplication, including failures" );
}

// Coupled linear equations 2*x+y=4, x+3*y=7 have the root (1,2).
// Like the original direct-solver example, the Jacobian is a raw SCFD matrix.
// An injected linear failure exercises strict rejection and inexact recovery.
template <class T>
struct matrix_problem
{
    using vector_type                            = scfd::static_vec::vec<T, 2>;
    using jacobi_operator_type                   = scfd::static_mat::mat<T, 2, 2>;
    std::shared_ptr<jacobi_operator_type> jacobi = std::make_shared<jacobi_operator_type>();

    matrix_problem()
    {
        ( *jacobi )( 0, 0 ) = T( 2 );
        ( *jacobi )( 0, 1 ) = ( *jacobi )( 1, 0 ) = T( 1 );
        ( *jacobi )( 1, 1 )                       = T( 3 );
    }
    void apply( const vector_type &state, vector_type &result ) const
    {
        result = *jacobi * state;
        result[0] -= T( 4 );
        result[1] -= T( 7 );
    }
    void set_linearization_point( const vector_type & )
    {
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobi;
    }
};

template <class T>
struct matrix_solver
{
    using problem_type = matrix_problem<T>;
    using vector_type  = typename problem_type::vector_type;
    std::shared_ptr<const typename problem_type::jacobi_operator_type> jacobi;
    T                                                                  fraction = T( 19 ) / T( 20 );

    void set_operator( std::shared_ptr<const typename problem_type::jacobi_operator_type> op )
    {
        jacobi = std::move( op );
    }
    bool solve( const vector_type &rhs, vector_type &result ) const
    {
        result = inv( *jacobi ) * rhs;
        result *= fraction;
        return false;
    }
};

template <class T>
void test_matrix_jacobi( test_report &report )
{
    using problem_type   = matrix_problem<T>;
    using vector_type    = typename problem_type::vector_type;
    using space_type     = nmfd::operations::static_vector_space<T, 2, vector_type>;
    using strategy_type  = nmfd::solvers::default_convergence_strategy<space_type, scfd::utils::log_std, problem_type>;
    using iteration_type = nmfd::solvers::newton_iteration<space_type, problem_type, matrix_solver<T>>;
    using solver_type = nmfd::solvers::nonlinear_solver<space_type, scfd::utils::log_std, problem_type, iteration_type>;
    auto         space     = std::make_shared<space_type>();
    auto         linear    = std::make_shared<matrix_solver<T>>();
    auto         iteration = std::make_shared<iteration_type>( space, linear );
    auto         strategy  = std::make_shared<strategy_type>( space, nullptr );
    solver_type  solver( space, nullptr, iteration, strategy );
    problem_type problem;
    for ( bool enabled : { false, true } )
    {
        strategy->set_allow_inexact_increment( enabled );
        vector_type initial{ T( 4 ), T( -3 ) }, state;
        const bool  converged = solver.solve( &problem, nullptr, nullptr, initial, state );
        vector_type residual;
        problem.apply( state, residual );
        report.check<T>(
            enabled
                ? converged && strategy->used_inexact_increment() &&
                      space->norm_l2( residual ) <= strategy->abs_tol() &&
                      std::abs( state[0] - T( 1 ) ) <= T( 10 ) * strategy->abs_tol() &&
                      std::abs( state[1] - T( 2 ) ) <= T( 10 ) * strategy->abs_tol()
                : !converged && strategy->get_result_status() == 6 && state[0] == initial[0] && state[1] == initial[1],
            enabled ? "raw matrix supports inexact Newton convergence"
                    : "raw matrix preserves strict rejection and rollback"
        );
    }
    vector_type rhs{ T( 4 ), T( 7 ) }, delta{ T( .5 ), T( 1 ) };
    report.check<T>(
        !strategy->accept_inexact_increment( *problem.jacobi, rhs, delta ),
        "raw matrix rejects corrections with excessive true defect"
    );
}

template <class T>
void test_configuration( test_report &report )
{
    using fixture = newton_fixture<T>;
    for ( T value :
          { T( 0 ), T( 1 ), T( -1 ), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN() } )
    {
        typename fixture::params_type params;
        params.maximum_relative_linear_residual = value;
        auto space                              = std::make_shared<typename fixture::space_type>();
        bool rejected                           = false;
        try
        {
            typename fixture::strategy_type strategy( space, nullptr, params );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>( rejected && space->allocations == 0, "invalid defect bound rejected before allocation" );
    }
#ifdef NMFD_ENABLE_NLOHMANN
    typename fixture::params_type params;
    params.from_json( { { "allow_inexact_increment", true }, { "maximum_relative_linear_residual", .25 } } );
    typename fixture::params_type restored;
    restored.from_json( params.to_json() );
    report.check<T>(
        restored.allow_inexact_increment && restored.maximum_relative_linear_residual == T( .25 ),
        "inexact settings survive JSON round trip"
    );
    bool rejected = false;
    try
    {
        params.from_json( { { "allow_inexact_increment", false }, { "maximum_relative_linear_residual", 1 } } );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    report.check<T>( rejected && params.allow_inexact_increment, "invalid JSON update is atomic" );
#endif
}

// Strategies used without the default Newton iteration need not expose a Jacobian.
struct residual_only
{
    void apply( const newton_fixture<double>::vector_type &in, newton_fixture<double>::vector_type &out )
    {
        out[0] = in[0];
    }
};

struct opaque_jacobi
{
};
struct opaque_problem : residual_only
{
    std::shared_ptr<const opaque_jacobi> get_jacobi_operator() const
    {
        return std::make_shared<opaque_jacobi>();
    }
};

void test_opaque_jacobi( test_report &report )
{
    using space_type = newton_fixture<double>::space_type;
    nmfd::solvers::default_convergence_strategy<space_type, scfd::utils::log_std, opaque_problem> strategy(
        std::make_shared<space_type>(), nullptr
    );
    opaque_problem op;
    for ( bool enabled : { false, true } )
    {
        strategy.reset_iterations();
        strategy.set_allow_inexact_increment( enabled );
        strategy.increment_ok( false );
        newton_fixture<double>::vector_type state{ 1 }, delta{ -1 };
        report.check<double>(
            strategy.check_convergence( &op, nullptr, nullptr, state, delta ) && strategy.get_result_status() == 6 &&
                state[0] == 1 && !strategy.used_inexact_increment(),
            "unsupported Jacobian compiles and cannot accept an unchecked correction"
        );
    }
}
}

int main()
{
    test_report report;
    test_newton<float>( report );
    test_newton<double>( report );
    test_newton<long double>( report );
    test_assessment<float>( report );
    test_assessment<double>( report );
    test_assessment<long double>( report );
    test_matrix_jacobi<float>( report );
    test_matrix_jacobi<double>( report );
    test_matrix_jacobi<long double>( report );
    test_opaque_jacobi( report );
    test_configuration<float>( report );
    test_configuration<double>( report );
    test_configuration<long double>( report );
    using space_type = newton_fixture<double>::space_type;
    nmfd::solvers::default_convergence_strategy<space_type, scfd::utils::log_std, residual_only> strategy(
        std::make_shared<space_type>(), nullptr
    );
    residual_only                       op;
    newton_fixture<double>::vector_type state{ 1 }, delta{ 0 };
    strategy.increment_ok( false );
    report.check<double>(
        strategy.check_convergence( &op, nullptr, nullptr, state, delta ) && strategy.get_result_status() == 6,
        "strict custom residual-only strategy does not require Jacobian access"
    );
    return report.finish();
}
