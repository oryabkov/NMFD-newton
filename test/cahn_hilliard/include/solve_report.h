#ifndef __TESTS_SOLVE_REPORT_H__
#define __TESTS_SOLVE_REPORT_H__

#include <optional>
#include <string>

namespace tests
{

struct step_report
{
    int index;
    std::optional<double> dt;
    std::optional<double> t;
    std::optional<int>    newton_iters;
    std::optional<double> resid;
    std::optional<double> step_norm;
    std::optional<double> error;
    std::optional<double> rel_error;
    std::optional<double> phobic_energy;
    std::optional<double> philic_energy;
    std::optional<double> step_time_ms;
};

struct final_report
{
    std::optional<bool>        converged;
    std::optional<int>         steps_completed;
    std::optional<int>         steps_total;
    std::optional<std::string> stop_reason;
    std::optional<double>      final_resid;
    std::optional<double>      final_error;
    std::optional<double>      final_rel_error;
    std::optional<double>      avg_newton_iters;
    std::optional<double>      avg_step_time_ms;
    std::optional<double>      total_time_ms;
    std::optional<double>      phobic_energy;
    std::optional<double>      philic_energy;
};

template<class Log>
void log_step_report( Log &log, const step_report &r )
{
    log.info( "----------------------------------------" );
    if ( r.t.has_value() )
    {
        log.info_f( "Time step %d  (dt = %e, t = %e)", r.index, r.dt.value_or( 0.0 ), *r.t );
    }
    else
    {
        log.info_f( "Time step %d", r.index );
    }
    log.info( "----------------------------------------" );
    if ( r.newton_iters.has_value() )
        log.info_f( "  %-28s:  %d", "Newton iterations", *r.newton_iters );
    if ( r.resid.has_value() )
        log.info_f( "  %-28s:  %e", "||F(x)||", *r.resid );
    if ( r.step_norm.has_value() )
        log.info_f( "  %-28s:  %e", "||x - x_prev||", *r.step_norm );
    if ( r.error.has_value() )
        log.info_f( "  %-28s:  %e", "||x - exact||", *r.error );
    if ( r.rel_error.has_value() )
        log.info_f( "  %-28s:  %e", "Relative error", *r.rel_error );
    if ( r.phobic_energy.has_value() )
        log.info_f( "  %-28s:  %e", "Phobic energy", *r.phobic_energy );
    if ( r.philic_energy.has_value() )
        log.info_f( "  %-28s:  %e", "Philic energy", *r.philic_energy );
    if ( r.step_time_ms.has_value() )
        log.info_f( "  %-28s:  %.2f ms", "Step time", *r.step_time_ms );
}

template<class Log>
void log_final_report( Log &log, const final_report &r )
{
    log.info( "" );
    log.info( "========================================" );
    log.info( "Results" );
    log.info( "========================================" );
    if ( r.converged.has_value() )
        log.info_f( "  %-28s:  %s", "Converged", *r.converged ? "yes" : "no" );
    if ( r.steps_completed.has_value() && r.steps_total.has_value() )
        log.info_f( "  %-28s:  %d / %d", "Time steps completed", *r.steps_completed, *r.steps_total );
    if ( r.stop_reason.has_value() )
        log.info_f( "  %-28s:  %s", "Stop reason", r.stop_reason->c_str() );
    if ( r.final_resid.has_value() )
        log.info_f( "  %-28s:  %e", "Final ||F(x)||", *r.final_resid );
    if ( r.final_error.has_value() )
        log.info_f( "  %-28s:  %e", "Final ||x - exact||", *r.final_error );
    if ( r.final_rel_error.has_value() )
        log.info_f( "  %-28s:  %e", "Final relative error", *r.final_rel_error );
    if ( r.avg_newton_iters.has_value() )
        log.info_f( "  %-28s:  %.2f", "Average Newton iters/step", *r.avg_newton_iters );
    if ( r.phobic_energy.has_value() )
        log.info_f( "  %-28s:  %e", "Phobic energy", *r.phobic_energy );
    if ( r.philic_energy.has_value() )
        log.info_f( "  %-28s:  %e", "Philic energy", *r.philic_energy );
    if ( r.avg_step_time_ms.has_value() )
        log.info_f( "  %-28s:  %.2f ms", "Average step time", *r.avg_step_time_ms );
    if ( r.total_time_ms.has_value() )
        log.info_f( "  %-28s:  %.2f ms", "Total solve time", *r.total_time_ms );
    log.info( "========================================" );
}

} // namespace tests

#endif // __TESTS_SOLVE_REPORT_H__
