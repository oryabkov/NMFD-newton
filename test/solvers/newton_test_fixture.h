#ifndef NMFD_NEWTON_TEST_FIXTURE_H
#define NMFD_NEWTON_TEST_FIXTURE_H

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/log_std.h>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/nonlinear_solver.h>

namespace nmfd
{
namespace tests
{

struct test_report
{
    unsigned int checks = 0, failures = 0;

    template <class T>
    void check( bool passed, const char *name )
    {
        ++checks;
        if ( !passed )
        {
            ++failures;
            const char *type = std::is_same_v<T, float>    ? "float"
                               : std::is_same_v<T, double> ? "double"
                                                           : "long double";
            std::printf( "FAIL [%s]: %s\n", type, name );
        }
    }

    int finish() const
    {
        std::printf( "%u checks, %u failures\n", checks, failures );
        return failures ? 1 : 0;
    }
};

enum class failure_point
{
    none,
    linearization,
    residual,
    jacobi_access,
    operator_setup,
    linear_solve
};
enum class equation
{
    linear,
    cubic
};
enum class correction
{
    exact,
    zero,
    half,
    nonfinite
};

template <class T, class Vector = scfd::static_vec::vec<T, 1>>
struct newton_fixture
{
    using vector_type = Vector;

    struct space_type : operations::static_vector_space<T, 1, vector_type>
    {
        mutable unsigned int lifecycle_calls = 0, allocations = 0, frees = 0, starts = 0, stops = 0;
        unsigned int         fail_at = 0;

        void init_vector( vector_type &x ) const
        {
            if ( ++lifecycle_calls == fail_at )
                throw std::runtime_error( "injected allocation failure" );
            if constexpr ( std::is_pointer_v<vector_type> )
                x = new T[1]{};
            ++allocations;
        }
        void start_use_vector( vector_type & ) const
        {
            if ( ++lifecycle_calls == fail_at )
                throw std::runtime_error( "injected activation failure" );
            ++starts;
        }
        void stop_use_vector( vector_type & ) const noexcept
        {
            ++stops;
        }
        void free_vector( vector_type &x ) const noexcept
        {
            if constexpr ( std::is_pointer_v<vector_type> )
            {
                delete[] x;
                x = nullptr;
            }
            ++frees;
        }
        bool balanced() const
        {
            return allocations == frees && starts == stops;
        }
    };

    struct jacobi_type
    {
        T    coefficient = T( 1 );
        void apply( const vector_type &x, vector_type &out ) const
        {
            out[0] = coefficient * x[0];
        }
    };

    struct problem_type
    {
        using jacobi_operator_type                    = jacobi_type;
        equation                     model            = equation::linear;
        failure_point                fail             = failure_point::none;
        bool                         invalid_residual = false, invalid_trials = false;
        T                            reference_state = T( 1 );
        unsigned int                 residual_calls = 0, trial_calls = 0;
        std::shared_ptr<jacobi_type> jacobi = std::make_shared<jacobi_type>();

        void apply( const vector_type &x, vector_type &out )
        {
            ++residual_calls;
            if ( fail == failure_point::residual )
                throw std::runtime_error( "injected residual failure" );
            const bool trial = x[0] != reference_state;
            trial_calls += trial;
            if ( invalid_residual || ( invalid_trials && trial ) )
                out[0] = std::numeric_limits<T>::quiet_NaN();
            else
                out[0] = model == equation::cubic ? x[0] * x[0] * x[0] - T( 2 ) * x[0] + T( 2 ) : x[0] - T( 1 );
        }
        void set_linearization_point( const vector_type &x )
        {
            if ( fail == failure_point::linearization )
                throw std::runtime_error( "injected linearization failure" );
            jacobi->coefficient = model == equation::cubic ? T( 3 ) * x[0] * x[0] - T( 2 ) : T( 1 );
        }
        std::shared_ptr<const jacobi_type> get_jacobi_operator() const
        {
            if ( fail == failure_point::jacobi_access )
                throw std::runtime_error( "injected Jacobian access failure" );
            return jacobi;
        }
    };

    struct linear_solver_type
    {
        failure_point                      fail      = failure_point::none;
        correction                         mode      = correction::exact;
        bool                               converged = true;
        unsigned int                       calls = 0, fail_after = 0;
        std::shared_ptr<const jacobi_type> op;

        void set_operator( std::shared_ptr<const jacobi_type> value )
        {
            if ( fail == failure_point::operator_setup )
                throw std::runtime_error( "injected operator setup failure" );
            op = std::move( value );
        }
        bool solve( const vector_type &rhs, vector_type &out )
        {
            ++calls;
            if ( fail == failure_point::linear_solve )
                throw std::runtime_error( "injected linear solve failure" );
            switch ( mode )
            {
            case correction::exact:
                out[0] = rhs[0] / op->coefficient;
                break;
            case correction::zero:
                out[0] = T( 0 );
                break;
            case correction::half:
                out[0] = T( .5 ) * rhs[0] / op->coefficient;
                break;
            case correction::nonfinite:
                out[0] = std::numeric_limits<T>::quiet_NaN();
                break;
            }
            return converged && ( fail_after == 0 || calls <= fail_after );
        }
    };

    using strategy_type  = solvers::default_convergence_strategy<space_type, scfd::utils::log_std, problem_type>;
    using iteration_type = solvers::newton_iteration<space_type, problem_type, linear_solver_type>;
    using solver_type    = solvers::nonlinear_solver<space_type, scfd::utils::log_std, problem_type, iteration_type>;
    using params_type    = typename strategy_type::params;

    std::shared_ptr<space_type>         space     = std::make_shared<space_type>();
    std::shared_ptr<linear_solver_type> linear    = std::make_shared<linear_solver_type>();
    std::shared_ptr<iteration_type>     iteration = std::make_shared<iteration_type>( space, linear );
    std::shared_ptr<strategy_type>      strategy;
    solver_type                         solver;
    problem_type                        problem;

    explicit newton_fixture( const params_type &params = params_type() )
        : strategy( std::make_shared<strategy_type>( space, nullptr, params ) ),
          solver( space, nullptr, iteration, strategy )
    {
    }
};

}
}

#endif
