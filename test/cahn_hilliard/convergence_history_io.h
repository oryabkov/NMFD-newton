#ifndef CONVERGENCE_HISTORY_IO_H
#define CONVERGENCE_HISTORY_IO_H

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <type_traits>

template <class Monitor, class Scalar>
void save_convergence_history(
    const Monitor& monitor,
    const std::string& solver_type,
    const std::string& preconditioner_type,
    int grid_size,
    const std::chrono::duration<double, std::milli>& solve_time_ms,
    const std::string& output_dir )
{
    // Determine architecture
    std::string arch;
#ifdef PLATFORM_CUDA
    arch = "cuda";
#elif defined( PLATFORM_OMP )
    arch = "omp";
#elif defined( PLATFORM_SERIAL_CPU )
    arch = "cpu";
#else
    arch = "unknown";
#endif

    // Determine type (f or d)
    std::string type = std::is_same_v<float, Scalar> ? "f" : "d";

    // Convergence history and times.dat go in the output directory for this run
    std::string conv_file_name = output_dir + "/conv_history.dat";
    std::string exec_time_file_name = output_dir + "/times.dat";

    std::ofstream conv_history( conv_file_name, std::ios::out | std::ios::trunc );
    std::ofstream exec_times( exec_time_file_name, std::ios::out | std::ios::trunc );

    auto res_by_it = monitor.convergence_history();
    std::for_each( begin( res_by_it ), end( res_by_it ),
                   [&]( std::pair<int, Scalar> &pair ) {
                       conv_history << pair.first << " " << pair.second << std::endl;
                   } );

    // Write header to times.dat
    exec_times << "solver,prec,arch,float_type,size,time(ms),iters_n,reduction_rate" << std::endl;

    if ( !res_by_it.empty() )
    {
        auto [i_0, init_res] = res_by_it.front();
        auto [i_n, final_res] = res_by_it.back();

        auto conv_rate = std::pow( final_res / init_res, Scalar( 1 ) / ( i_n - i_0 ) );

        exec_times << std::fixed << std::setprecision( 10 );
        exec_times << solver_type << "," << preconditioner_type << "," << arch << "," << type << ","
                   << grid_size << "," << solve_time_ms.count() << "," << i_n << "," << conv_rate
                   << std::endl;
    }

    conv_history.close();
    exec_times.close();
}

#endif // CONVERGENCE_HISTORY_IO_H
