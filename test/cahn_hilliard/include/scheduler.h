#ifndef __TESTS_CAHN_HILLIARD_SCHEDULER_H__
#define __TESTS_CAHN_HILLIARD_SCHEDULER_H__

#include <algorithm>
#include <stdexcept>

namespace tests
{

template <class Scalar>
class scheduler
{
public:
    using scalar_type = Scalar;

    scheduler() = delete;

    scheduler( scalar_type dt_inf_initial, int success_threshold = 5 )
        : dt_inf_( dt_inf_initial )
        , success_threshold_( std::max( 1, success_threshold ) )
    {
        if ( !( dt_inf_ > scalar_type( 0 ) ) )
        {
            throw std::invalid_argument( "scheduler: dt_inf_initial must be > 0" );
        }
    }

    scalar_type get_dt_inf() const noexcept
    {
        return dt_inf_;
    }

    int get_success_streak() const noexcept
    {
        return success_streak_;
    }

    void step( bool successful ) noexcept
    {
        if ( successful )
        {
            ++success_streak_;
            if ( success_streak_ >= success_threshold_ )
            {
                dt_inf_ /= scalar_type( 2 );
                success_streak_ = 0;
            }
        }
        else
        {
            success_streak_ = 0;
            dt_inf_ *= scalar_type( 2 );
        }
    }

private:
    scalar_type dt_inf_;
    int         success_streak_     = 0;
    int         success_threshold_  = 5;
};

} // namespace tests

#endif
