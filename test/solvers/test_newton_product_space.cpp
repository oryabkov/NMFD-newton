#include <memory>

#include <nmfd/operations/product_vector_space.h>
#include <nmfd/operations/scfd_vector_space.h>
#include <nmfd/solvers/bicgstabl.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <scfd/backend/serial_cpu.h>

#include "newton_test_fixture.h"

namespace
{

// F(u,p) = (2*(u-target), p^2-target^2), with the positive parameter root.
template <class Space>
struct product_problem
{
    using vector_type = typename Space::vector_type;
    struct jacobi_operator_type
    {
        std::shared_ptr<Space> space;
        double                 parameter_derivative = 2.;
        bool                   fail                 = false;
        bool                   apply( const vector_type &x, vector_type &out ) const
        {
            if ( fail )
                return false;
            space->first_space().assign_lin_comb( 2., x.first, out.first );
            out.second[0] = parameter_derivative * x.second[0];
            return true;
        }
    };
    std::shared_ptr<Space>                space;
    std::shared_ptr<jacobi_operator_type> jacobi;
    double                                target = 2.;

    explicit product_problem( std::shared_ptr<Space> input )
        : space( std::move( input ) ), jacobi( std::make_shared<jacobi_operator_type>() )
    {
        jacobi->space = space;
    }
    void apply( const vector_type &x, vector_type &out ) const
    {
        space->first_space().assign_lin_comb( 2., x.first, out.first );
        space->first_space().add_mul_scalar( -2. * target, 1., out.first );
        out.second[0] = x.second[0] * x.second[0] - target * target;
    }
    void set_linearization_point( const vector_type &x )
    {
        jacobi->parameter_derivative = 2. * x.second[0];
    }
    std::shared_ptr<const jacobi_operator_type> get_jacobi_operator() const
    {
        return jacobi;
    }
};

template <class Space, class Linear>
void check_newton( const std::shared_ptr<Space> &space, nmfd::tests::test_report &test )
{
    using problem_type   = product_problem<Space>;
    using iteration_type = nmfd::solvers::newton_iteration<Space, problem_type, Linear>;
    using solver_type    = nmfd::solvers::nonlinear_solver<Space, scfd::utils::log_std, problem_type, iteration_type>;
    typename Linear::params parameters;
    parameters.basis_size            = 2;
    parameters.monitor.abs_tol       = 1.e-13;
    parameters.monitor.rel_tol       = 1.e-12;
    parameters.monitor.max_iters_num = 50;
    auto        linear               = std::make_shared<Linear>( space, nullptr, parameters );
    auto        problem              = std::make_shared<problem_type>( space );
    solver_type solver( space, nullptr, std::make_shared<iteration_type>( space, linear ) );
    solver.set_nonlinear_operator( problem );
    solver.get_convergence_strategy_handle()->set_tolerance( 1.e-11 );
    nmfd::detail::vector_wrap<Space, false, false> x( *space ), residual( *space );
    x.init();
    x.start_use();
    residual.init();
    residual.start_use();
    for ( double target : { 2., 3. } )
    {
        problem->target = target;
        space->assign_scalar( 1., *x );
        test.check<double>( solver.solve( *x ), "Product Newton convergence and rebinding" );
        problem->apply( *x, *residual );
        test.check<double>( space->norm_l2( *residual ) <= 1.e-10, "Full product nonlinear residual" );
        space->add_mul_scalar( -target, 1., *x );
        test.check<double>( space->norm( *x ) <= 1.e-10, "State and parameter roots" );
    }
    space->assign_scalar( 1., *x );
    problem->jacobi->fail = true;
    test.check<double>( !solver.solve( *x ), "Product Newton reports linear failure" );
    space->add_mul_scalar( -1., 1., *x );
    test.check<double>( space->norm( *x ) == 0., "Linear failure preserves both components" );
}

}

int main()
{
    using state_type     = nmfd::operations::scfd_vector_space<scfd::backend::serial_cpu<>, double>;
    using parameter_type = nmfd::operations::static_vector_space<double, 1, scfd::static_vec::vec<double, 1>>;
    using space_type     = nmfd::operations::product_vector_space<state_type, parameter_type>;
    using log_type       = scfd::utils::log_std;
    using monitor_type   = nmfd::solvers::monitor_krylov<space_type, log_type>;
    using operator_type  = typename product_problem<space_type>::jacobi_operator_type;
    auto space = std::make_shared<space_type>( std::make_shared<state_type>( 2 ), std::make_shared<parameter_type>() );
    nmfd::tests::test_report test;
    check_newton<space_type, nmfd::solvers::gmres<space_type, monitor_type, log_type, operator_type>>( space, test );
    check_newton<space_type, nmfd::solvers::bicgstabl<space_type, monitor_type, log_type, operator_type>>(
        space, test
    );
    return test.finish();
}
