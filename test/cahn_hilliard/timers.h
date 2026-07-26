#ifndef __TIMER_H__
#define __TIMER_H__

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>

#if defined(PLATFORM_CUDA)
    // CUDA backend
    #include <cuda_runtime.h>
    #include <scfd/utils/cuda_safe_call.h>

    struct Timer {
        cudaEvent_t start, stop;
        std::string name;
        bool auto_print;
        bool stopped;

        Timer(const std::string& timer_name = "Timer", bool print_on_destruct = false)
            : name(timer_name), auto_print(print_on_destruct), stopped(false) {
            CUDA_SAFE_CALL(cudaEventCreate(&start));
            CUDA_SAFE_CALL(cudaEventCreate(&stop));
            CUDA_SAFE_CALL(cudaEventRecord(start));
        }

        ~Timer() {
            if (!stopped) {
                CUDA_SAFE_CALL(cudaEventRecord(stop));
            }
            CUDA_SAFE_CALL(cudaEventSynchronize(stop));
            float ms;
            CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));
            if (auto_print) {
                std::cout << name << ": " << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
            }
            CUDA_SAFE_CALL(cudaEventDestroy(start));
            CUDA_SAFE_CALL(cudaEventDestroy(stop));
        }

        Timer(const Timer&) = delete;
        Timer& operator=(const Timer&) = delete;

        // Method to get elapsed time without destroying the timer
        // Note: This records stop event if not already recorded
        double elapsed_ms() {
            if (!stopped) {
                CUDA_SAFE_CALL(cudaEventRecord(stop));
                stopped = true;
            }
            CUDA_SAFE_CALL(cudaEventSynchronize(stop));
            float ms;
            CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));
            return static_cast<double>(ms);
        }

        // Method to stop timing and get elapsed time
        double stop_and_get_ms() {
            if (!stopped) {
                CUDA_SAFE_CALL(cudaEventRecord(stop));
                stopped = true;
            }
            CUDA_SAFE_CALL(cudaEventSynchronize(stop));
            float ms;
            CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));
            return static_cast<double>(ms);
        }
    };

#elif defined(PLATFORM_HIP)
    // HIP backend not yet supported
    struct Timer {
        Timer(const std::string& = "Timer", bool = false) {
            assert(false && "HIP platform is not yet supported for Timer. Please use CUDA or CPU/OpenMP platform.");
        }
    };

#else
    // CPU/OpenMP/SYCL backend or fallback - use system timer
    #include <scfd/utils/system_timer_event.h>
    #include <chrono>

    struct Timer {
        scfd::utils::system_timer_event start, stop;
        std::string name;
        bool stopped;
        bool auto_print;

        Timer(const std::string& timer_name = "Timer", bool print_on_destruct = false)
            : name(timer_name), stopped(false), auto_print(print_on_destruct) {
            start.record();
        }

        ~Timer() {
            if (!stopped) {
                stop.record();
                double ms = stop.elapsed_time(start);
                if (auto_print) {
                    std::cout << name << ": " << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
                }
            }
        }

        Timer(const Timer&) = delete;
        Timer& operator=(const Timer&) = delete;

        // Method to get elapsed time without destroying the timer
        double elapsed_ms() const {
            scfd::utils::system_timer_event current;
            current.record();
            return current.elapsed_time(start);
        }

        // Method to stop timing and get elapsed time
        double stop_and_get_ms() {
            if (!stopped) {
                stop.record();
                stopped = true;
            }
            return stop.elapsed_time(start);
        }
    };

#endif

#endif // __TIMER_H__
