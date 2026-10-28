#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <nmfd/operations/cpu_vector_operations_var_prec.h>
#include <nmfd/solvers/default_monitor.h>
#include <nmfd/solvers/direct_solver.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <scfd/utils/log.h>

namespace
{
int  checks = 0;
void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

template <unsigned int Precision>
void test_solver()
{
    using space_type  = cpu_vector_operations_var_prec<Precision>;
    using T           = typename space_type::scalar_type;
    using vector_type = typename space_type::vector_type;
    struct jacobian
    {
        T    coefficient;
        void apply( const vector_type &in, vector_type &out ) const
        {
            out[0] = coefficient * in[0];
        }
    };
    // x^2 = 1 + 1e-40 has a positive root distinguishable from 1 only above double precision.
    struct problem
    {
        using jacobi_operator_type       = jacobian;
        T                         target = T( 1 ) + T( "1e-40" );
        std::shared_ptr<jacobian> op     = std::make_shared<jacobian>();
        void                      apply( const vector_type &in, vector_type &out )
        {
            out[0] = in[0] * in[0] - target;
        }
        void set_linearization_point( const vector_type &in )
        {
            op->coefficient = T( 2 ) * in[0];
        }
        std::shared_ptr<const jacobian> get_jacobi_operator()
        {
            return op;
        }
    };
    struct inverse
    {
        std::shared_ptr<const jacobian> op;
        void                            set_operator( std::shared_ptr<const jacobian> value )
        {
            op = std::move( value );
        }
        bool apply( vector_type &in ) const
        {
            in[0] /= op->coefficient;
            return true;
        }
    };
    using log_type    = scfd::utils::log_std;
    using linear_type = nmfd::solvers::direct_solver<
        space_type, nmfd::solvers::default_monitor<space_type, log_type>, log_type, jacobian, inverse>;
    using iteration_type               = nmfd::solvers::newton_iteration<space_type, problem, linear_type>;
    using solver_type                  = nmfd::solvers::nonlinear_solver<space_type, log_type, problem, iteration_type>;
    auto                         space = std::make_shared<space_type>( 1 );
    typename linear_type::params linear_parameters;
    linear_parameters.monitor.abs_tol = T( "1e-80" );
    linear_parameters.monitor.rel_tol = T( "1e-80" );
    auto linear    = std::make_shared<linear_type>( space, nullptr, linear_parameters, std::make_shared<inverse>() );
    auto iteration = std::make_shared<iteration_type>( space, linear );
    using strategy_type = nmfd::solvers::default_convergence_strategy<space_type, log_type, problem>;
    typename strategy_type::params parameters;
    parameters.abs_tol   = T( "1e-70" );
    parameters.verbose   = false;
    auto        strategy = std::make_shared<strategy_type>( space, nullptr, parameters );
    solver_type solver( space, nullptr, iteration, strategy );
    auto        op = std::make_shared<problem>();
    solver.set_nonlinear_operator( op );
    vector_type state{ T( 2 ) }, residual( 1 );
    for ( int index = 0; index < 2; ++index )
    {
        state[0] = T( 2 );
        require( solver.solve( state ), "Native-precision Newton correction" );
        op->apply( state, residual );
        require(
            boost::multiprecision::abs( residual[0] ) <= parameters.abs_tol, "Native-precision physical residual"
        );
        require( state[0] != T( 1 ) && static_cast<double>( state[0] ) == 1., "Sub-double information retained" );
    }
    state[0] = std::numeric_limits<T>::infinity();
    require( !solver.solve( state ) && strategy->get_result_status() != 0, "Nonfinite state is rejected" );
}
}

int main()
{
    try
    {
        test_solver<100>();
        test_solver<200>();
        std::cout << "Multiprecision Newton: " << checks << " checks passed\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
