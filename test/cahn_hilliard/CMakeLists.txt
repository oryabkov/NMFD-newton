# ── cahn_hilliard tests ───────────────────────────────────────────────────────

# Local headers for this directory
set(CAHN_HILLIARD_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})

# Required define for SCFD arrays (enables rect_nd, indexes0_nd, etc.)
set(CH_EXTRA_DEFS SCFD_ARRAYS_ENABLE_INDEX_SHIFT=1)

# Runtime parameters passed to the standalone test executables.
# These mirror the positional args of run_*.sh: <solver> <preconditioner> <grid_size>.
set(NMFD_CH_SOLVER         "gmres" CACHE STRING "Solver used by cahn_hilliard tests: 'jacobi' or 'gmres'")
set(NMFD_CH_PRECONDITIONER "mg"    CACHE STRING "Preconditioner for cahn_hilliard tests: 'diag' or 'mg'")
set(NMFD_CH_GRID_SIZE      "32"    CACHE STRING "Grid points per dimension for cahn_hilliard tests")
mark_as_advanced(NMFD_CH_SOLVER NMFD_CH_PRECONDITIONER NMFD_CH_GRID_SIZE)

# Grid sizes experiment: same loop as run_grid_size_experiment_*.sh.
# One add_test is generated per grid size. Empty list disables the loop.
set(NMFD_CH_GRID_SIZES "32" CACHE STRING "Grid sizes for the experiment, e.g. '2 4 8 16 32 64 128 256'")
mark_as_advanced(NMFD_CH_GRID_SIZES)
string(REPLACE " " ";" NMFD_CH_GRID_SIZES "${NMFD_CH_GRID_SIZES}")

# Tests with all platform variants (cpu, omp, cuda)
set(CH_DUAL_TESTS
    test_biharmonic
    test_cahn_hilliard
    test_time_cahn_hilliard
)

foreach(test_name IN LISTS CH_DUAL_TESTS)
    # CPU variant
    add_executable(${test_name}_cpu_${NMFD_PRECISION_SUFFIX} ${test_name}.cpp)
    target_link_libraries(${test_name}_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers)
    target_include_directories(${test_name}_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE ${CAHN_HILLIARD_INCLUDES})
    target_compile_definitions(${test_name}_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_SERIAL_CPU ${CH_EXTRA_DEFS})

    # OpenMP variant
    add_executable(${test_name}_omp_${NMFD_PRECISION_SUFFIX} ${test_name}.cpp)
    target_link_libraries(${test_name}_omp_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers OpenMP::OpenMP_CXX)
    target_include_directories(${test_name}_omp_${NMFD_PRECISION_SUFFIX} PRIVATE ${CAHN_HILLIARD_INCLUDES})
    target_compile_definitions(${test_name}_omp_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_OMP ${CH_EXTRA_DEFS})

    # CUDA variant
    if(NMFD_CUDA_ENABLED)
        add_executable(${test_name}_cuda_${NMFD_PRECISION_SUFFIX} ${test_name}.cpp)
        target_link_libraries(${test_name}_cuda_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers CUDA::cudart)
        target_include_directories(${test_name}_cuda_${NMFD_PRECISION_SUFFIX} PRIVATE ${CAHN_HILLIARD_INCLUDES})
        target_compile_definitions(${test_name}_cuda_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_CUDA ${CH_EXTRA_DEFS})
        set_source_files_properties(${test_name}.cpp PROPERTIES LANGUAGE CUDA)
    endif()
endforeach()

# Register cahn_hilliard tests
# test_biharmonic/test_cahn_hilliard/test_time_cahn_hilliard take: solver preconditioner grid_size
foreach(grid_size IN LISTS NMFD_CH_GRID_SIZES)
    add_test(NAME test_biharmonic_cpu_${grid_size}
        COMMAND test_biharmonic_cpu_${NMFD_PRECISION_SUFFIX} ${NMFD_CH_SOLVER} ${NMFD_CH_PRECONDITIONER} ${grid_size})
    add_test(NAME test_biharmonic_omp_${grid_size}
        COMMAND test_biharmonic_omp_${NMFD_PRECISION_SUFFIX} ${NMFD_CH_SOLVER} ${NMFD_CH_PRECONDITIONER} ${grid_size})
    add_test(NAME test_cahn_hilliard_cpu_${grid_size}
        COMMAND test_cahn_hilliard_cpu_${NMFD_PRECISION_SUFFIX} ${NMFD_CH_SOLVER} ${NMFD_CH_PRECONDITIONER} ${grid_size})
    add_test(NAME test_cahn_hilliard_omp_${grid_size}
        COMMAND test_cahn_hilliard_omp_${NMFD_PRECISION_SUFFIX} ${NMFD_CH_SOLVER} ${NMFD_CH_PRECONDITIONER} ${grid_size})
    add_test(NAME test_time_cahn_hilliard_cpu_${grid_size}
        COMMAND test_time_cahn_hilliard_cpu_${NMFD_PRECISION_SUFFIX} ${NMFD_CH_SOLVER} ${NMFD_CH_PRECONDITIONER} ${grid_size})
    add_test(NAME test_time_cahn_hilliard_omp_${grid_size}
        COMMAND test_time_cahn_hilliard_omp_${NMFD_PRECISION_SUFFIX} ${NMFD_CH_SOLVER} ${NMFD_CH_PRECONDITIONER} ${grid_size})
endforeach()

# test_random_init_cahn_hilliard takes: grid_size
add_executable(test_random_init_cahn_hilliard_cpu_${NMFD_PRECISION_SUFFIX} test_random_init_cahn_hilliard.cpp)
target_link_libraries(test_random_init_cahn_hilliard_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers)
target_include_directories(test_random_init_cahn_hilliard_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE ${CAHN_HILLIARD_INCLUDES})
target_compile_definitions(test_random_init_cahn_hilliard_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_SERIAL_CPU ${CH_EXTRA_DEFS})

add_executable(test_random_init_cahn_hilliard_omp_${NMFD_PRECISION_SUFFIX} test_random_init_cahn_hilliard.cpp)
target_link_libraries(test_random_init_cahn_hilliard_omp_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers OpenMP::OpenMP_CXX)
target_include_directories(test_random_init_cahn_hilliard_omp_${NMFD_PRECISION_SUFFIX} PRIVATE ${CAHN_HILLIARD_INCLUDES})
target_compile_definitions(test_random_init_cahn_hilliard_omp_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_OMP ${CH_EXTRA_DEFS})

foreach(grid_size IN LISTS NMFD_CH_GRID_SIZES)
    add_test(NAME test_random_init_cahn_hilliard_cpu_${grid_size}
        COMMAND test_random_init_cahn_hilliard_cpu_${NMFD_PRECISION_SUFFIX} ${grid_size})
    add_test(NAME test_random_init_cahn_hilliard_omp_${grid_size}
        COMMAND test_random_init_cahn_hilliard_omp_${NMFD_PRECISION_SUFFIX} ${grid_size})
endforeach()

# MPI variants
if(NMFD_MPI_ENABLED)
    foreach(grid_size IN LISTS NMFD_CH_GRID_SIZES)
        foreach(test_name IN LISTS CH_DUAL_TESTS)
            # MPI + CPU
            add_executable(${test_name}_mpi_cpu_${NMFD_PRECISION_SUFFIX} ${test_name}.cpp)
            target_link_libraries(${test_name}_mpi_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers MPI::MPI_CXX)
            target_include_directories(${test_name}_mpi_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE ${CAHN_HILLIARD_INCLUDES})
            target_compile_definitions(${test_name}_mpi_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_SERIAL_CPU ${CH_EXTRA_DEFS})
            add_test(NAME ${test_name}_mpi_cpu_${grid_size}
                COMMAND ${test_name}_mpi_cpu_${NMFD_PRECISION_SUFFIX} ${NMFD_CH_SOLVER} ${NMFD_CH_PRECONDITIONER} ${grid_size})

            # MPI + OpenMP
            add_executable(${test_name}_mpi_omp_${NMFD_PRECISION_SUFFIX} ${test_name}.cpp)
            target_link_libraries(${test_name}_mpi_omp_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers MPI::MPI_CXX OpenMP::OpenMP_CXX)
            target_include_directories(${test_name}_mpi_omp_${NMFD_PRECISION_SUFFIX} PRIVATE ${CAHN_HILLIARD_INCLUDES})
            target_compile_definitions(${test_name}_mpi_omp_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_OMP ${CH_EXTRA_DEFS})
            add_test(NAME ${test_name}_mpi_omp_${grid_size}
                COMMAND ${test_name}_mpi_omp_${NMFD_PRECISION_SUFFIX} ${NMFD_CH_SOLVER} ${NMFD_CH_PRECONDITIONER} ${grid_size})

            # MPI + CUDA
            if(NMFD_CUDA_ENABLED)
                add_executable(${test_name}_mpi_cuda_${NMFD_PRECISION_SUFFIX} ${test_name}.cpp)
                target_link_libraries(${test_name}_mpi_cuda_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers MPI::MPI_CXX CUDA::cudart)
                target_include_directories(${test_name}_mpi_cuda_${NMFD_PRECISION_SUFFIX} PRIVATE ${CAHN_HILLIARD_INCLUDES})
                target_compile_definitions(${test_name}_mpi_cuda_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_CUDA ${CH_EXTRA_DEFS})
                set_source_files_properties(${test_name}.cpp PROPERTIES LANGUAGE CUDA)
                add_test(NAME ${test_name}_mpi_cuda_${grid_size}
                    COMMAND ${test_name}_mpi_cuda_${NMFD_PRECISION_SUFFIX} ${NMFD_CH_SOLVER} ${NMFD_CH_PRECONDITIONER} ${grid_size})
            endif()
        endforeach()
    endforeach()
endif()