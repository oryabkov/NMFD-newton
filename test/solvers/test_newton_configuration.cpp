#include "newton_test_fixture.h"
#include <cstdint>

using namespace nmfd::tests;

template <class T>
void check_unchanged( newton_fixture<T> &f, test_report &report )
{
    using vector_type = typename newton_fixture<T>::vector_type;
    report.check<T>(
        f.strategy->abs_tol() == T( 1e-6 ) && f.strategy->rel_tol() == T( 0 ),
        "rejected configuration leaves tolerances unchanged"
    );
    f.linear->mode = correction::half;
    vector_type x{ T( 2 ) };
    report.check<T>(
        f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1 ) + std::ldexp( T( 1 ), -20 ) &&
            f.linear->calls == 20 && f.space->validity_checks != 0 && f.strategy->get_norms_history_handle()->empty(),
        "rejected configuration leaves iteration budget, damping, validation, and history unchanged"
    );
}

template <class T>
void test_tolerances( test_report &report )
{
    using fixture = newton_fixture<T>;
    for ( T value :
          { T( -1 ), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(),
            -std::numeric_limits<T>::infinity() } )
    {
        for ( bool relative : { false, true } )
        {
            typename fixture::params_type p;
            p.abs_tol     = relative ? T( 0 ) : value;
            p.rel_tol     = relative ? value : T( 0 );
            auto space    = std::make_shared<typename fixture::space_type>();
            bool rejected = false;
            try
            {
                typename fixture::strategy_type strategy( space, nullptr, p );
            }
            catch ( const std::invalid_argument & )
            {
                rejected = true;
            }
            report.check<T>(
                rejected && space->allocations == 0, "constructor rejects invalid tolerances before allocation"
            );
            fixture f;
            rejected = false;
            try
            {
                f.strategy->set_tolerance( p.abs_tol, p.rel_tol );
            }
            catch ( const std::invalid_argument & )
            {
                rejected = true;
            }
            report.check<T>( rejected, "tolerance setter rejects either invalid tolerance" );
            rejected = false;
            try
            {
                f.strategy->set_convergence_constants(
                    value, 1, T( 2 ), 0, T( .25 ), true, false, 0, T( .5 ), T( .125 ), !relative, false
                );
            }
            catch ( const std::invalid_argument & )
            {
                rejected = true;
            }
            report.check<T>( rejected, "full setter rejects invalid tolerance before changing any setting" );
            check_unchanged( f, report );
        }
    }
    for ( bool relative : { false, true } )
    {
        fixture f;
        f.strategy->set_convergence_constants(
            T( .125 ), 100, T( 1 ), 0, T( 1 ), true, false, 10, T( 0 ), T( 1e-12 ), !relative, false
        );
        f.linear->mode = correction::half;
        typename fixture::vector_type x{ T( 2 ) };
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1.125 ) && f.linear->calls == 3 &&
                f.strategy->get_norms_history_handle()->size() == 3 && f.space->validity_checks == 0 &&
                f.strategy->abs_tol() == ( relative ? T( 0 ) : T( .125 ) ) &&
                f.strategy->rel_tol() == ( relative ? T( .125 ) : T( 0 ) ),
            "valid full setter commits absolute or relative tolerance and requested controls"
        );
    }
}

template <class T>
void test_limits( test_report &report )
{
    using fixture      = newton_fixture<T>;
    const auto maximum = static_cast<unsigned int>( std::numeric_limits<int>::max() );
    for ( unsigned int limit : { 0u, maximum + 1u, std::numeric_limits<unsigned int>::max() } )
    {
        auto                          space = std::make_shared<typename fixture::space_type>();
        typename fixture::params_type p;
        p.max_iters_num       = limit;
        p.store_norms_history = true;
        bool rejected         = false;
        try
        {
            typename fixture::strategy_type strategy( space, nullptr, p );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>(
            rejected && space->allocations == 0,
            "invalid iteration limit is rejected before scratch and history allocation"
        );
    }
    for ( int limit : { -1, 0 } )
    {
        fixture f;
        bool    rejected = false;
        try
        {
            f.strategy->set_convergence_constants( T( 2 ), limit, T( 2 ), 0, T( .25 ), true, false );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>( rejected, "signed iteration limit is checked before assignment" );
        check_unchanged( f, report );
    }
    for ( unsigned int limit : { 1u, maximum } )
    {
        typename fixture::params_type p;
        p.max_iters_num = limit;
        p.abs_tol = p.rel_tol = T( 0 );
        fixture                       f( p );
        typename fixture::vector_type x{ T( 2 ) };
        report.check<T>(
            f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1 ) && f.linear->calls == 1,
            "iteration range endpoints and exact zero tolerance are valid"
        );
    }
}

template <class T>
void test_globalization( test_report &report )
{
    using fixture = newton_fixture<T>;
    const T nan = std::numeric_limits<T>::quiet_NaN(), inf = std::numeric_limits<T>::infinity();
    struct settings
    {
        T relaxation, increase;
    };
    const settings invalid[] = { { T( -1 ), T( 0 ) }, { T( 0 ), T( 0 ) }, { nan, T( 0 ) }, { inf, T( 0 ) },
                                 { -inf, T( 0 ) },    { T( 1 ), nan },    { T( 1 ), inf }, { T( 1 ), -inf } };
    for ( const auto &value : invalid )
    {
        auto                          space = std::make_shared<typename fixture::space_type>();
        typename fixture::params_type p;
        p.relax_tolerance_factor = value.relaxation;
        p.maximum_norm_increase  = value.increase;
        bool rejected            = false;
        try
        {
            typename fixture::strategy_type strategy( space, nullptr, p );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>(
            rejected && space->allocations == 0,
            "constructor rejects invalid globalization parameters before allocation"
        );
        fixture f;
        rejected = false;
        try
        {
            f.strategy->set_convergence_constants(
                T( 2 ), 1, value.relaxation, 0, T( .25 ), true, false, 0, value.increase, T( .125 ), true, false
            );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>( rejected, "invalid globalization parameters do not partially change the full configuration" );
        check_unchanged( f, report );
    }
}

#ifdef NMFD_ENABLE_NLOHMANN
template <class T>
void test_json( test_report &report )
{
    using fixture        = newton_fixture<T>;
    using json           = nlohmann::json;
    const json invalid[] = {
        { { "max_iters_num", -1 } },
        { { "max_iters_num", 0 } },
        { { "max_iters_num", std::uint64_t( std::numeric_limits<int>::max() ) + 1 } },
        { { "max_iters_num", std::uint64_t( std::numeric_limits<unsigned int>::max() ) + 1 } },
        { { "max_iters_num", std::numeric_limits<std::uint64_t>::max() } },
        { { "max_iters_num", 1.5 } },
        { { "max_iters_num", true } },
        { { "max_iters_num", "100" } },
        { { "max_backtracking_trials", -1 } },
        { { "max_backtracking_trials", 0 } },
        { { "stagnation_max", -1 } },
        { { "abs_tol", -1 } },
        { { "rel_tol", -1 } },
        { { "abs_tol", std::numeric_limits<T>::infinity() } },
        { { "rel_tol", std::numeric_limits<T>::quiet_NaN() } },
        { { "abs_tol", 2 }, { "relax_tolerance_factor", 0 }, { "store_norms_history", true } }
    };
    for ( const auto &input : invalid )
    {
        typename fixture::params_type p;
        const auto                    previous = p.to_json();
        bool                          rejected = false;
        try
        {
            p.from_json( input );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        report.check<T>(
            rejected && p.to_json() == previous, "invalid JSON is rejected without partial parameter updates"
        );
    }
    typename fixture::params_type p;
    p.from_json( json{ { "max_iters_num", 1 }, { "stagnation_max", 0 }, { "abs_tol", 0 }, { "rel_tol", 0 } } );
    fixture                       f( p );
    typename fixture::vector_type x{ T( 2 ) };
    report.check<T>(
        f.solver.solve( &f.problem, nullptr, nullptr, x ) && x[0] == T( 1 ),
        "valid JSON boundary values reach the exact root"
    );
}
#endif

int main()
{
    test_report report;
    test_tolerances<float>( report );
    test_tolerances<double>( report );
    test_tolerances<long double>( report );
    test_limits<float>( report );
    test_limits<double>( report );
    test_limits<long double>( report );
    test_globalization<float>( report );
    test_globalization<double>( report );
    test_globalization<long double>( report );
#ifdef NMFD_ENABLE_NLOHMANN
    test_json<float>( report );
    test_json<double>( report );
    test_json<long double>( report );
#endif
    return report.finish();
}
