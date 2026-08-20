# ── cahn_hilliard tests ───────────────────────────────────────────────────────

# Local headers for this directory
set(CAHN_HILLIARD_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})

# Required define for SCFD arrays (enables rect_nd, indexes0_nd, etc.)
set(CH_EXTRA_DEFS SCFD_ARRAYS_ENABLE_INDEX_SHIFT=1)

# Tests with all platform variants (cpu, omp, cuda)
set(CH_DUAL_TESTS
    test_biharmonic
    test_cahn_hilliard
    test_time_cahn_hilliard
    test_random_init_cahn_hilliard
)

foreach(test_name IN LISTS CH_DUAL_TESTS)
    # CPU variant
    add_executable(${test_name}_cpu ${test_name}.cpp)
    target_link_libraries(${test_name}_cpu PRIVATE nmfd_headers)
    target_include_directories(${test_name}_cpu PRIVATE ${CAHN_HILLIARD_INCLUDES})
    target_compile_definitions(${test_name}_cpu PRIVATE PLATFORM_SERIAL_CPU ${CH_EXTRA_DEFS})
    add_test(NAME ${test_name}_cpu COMMAND ${test_name}_cpu)

    # OpenMP variant
    add_executable(${test_name}_omp ${test_name}.cpp)
    target_link_libraries(${test_name}_omp PRIVATE nmfd_headers OpenMP::OpenMP_CXX)
    target_include_directories(${test_name}_omp PRIVATE ${CAHN_HILLIARD_INCLUDES})
    target_compile_definitions(${test_name}_omp PRIVATE PLATFORM_OMP ${CH_EXTRA_DEFS})
    add_test(NAME ${test_name}_omp COMMAND ${test_name}_omp)

    # CUDA variant
    if(NMFD_CUDA_ENABLED)
        add_executable(${test_name}_cuda ${test_name}.cpp)
        target_link_libraries(${test_name}_cuda PRIVATE nmfd_headers CUDA::cudart)
        target_include_directories(${test_name}_cuda PRIVATE ${CAHN_HILLIARD_INCLUDES})
        target_compile_definitions(${test_name}_cuda PRIVATE PLATFORM_CUDA ${CH_EXTRA_DEFS})
        set_source_files_properties(${test_name}.cpp PROPERTIES LANGUAGE CUDA)
        add_test(NAME ${test_name}_cuda COMMAND ${test_name}_cuda)
    endif()
endforeach()

# MPI variants
if(NMFD_MPI_ENABLED)
    foreach(test_name IN LISTS CH_DUAL_TESTS)
        # MPI + CPU
        add_executable(${test_name}_mpi_cpu ${test_name}.cpp)
        target_link_libraries(${test_name}_mpi_cpu PRIVATE nmfd_headers MPI::MPI_CXX)
        target_include_directories(${test_name}_mpi_cpu PRIVATE ${CAHN_HILLIARD_INCLUDES})
        target_compile_definitions(${test_name}_mpi_cpu PRIVATE PLATFORM_SERIAL_CPU ${CH_EXTRA_DEFS})
        add_test(NAME ${test_name}_mpi_cpu COMMAND ${test_name}_mpi_cpu)

        # MPI + OpenMP
        add_executable(${test_name}_mpi_omp ${test_name}.cpp)
        target_link_libraries(${test_name}_mpi_omp PRIVATE nmfd_headers MPI::MPI_CXX OpenMP::OpenMP_CXX)
        target_include_directories(${test_name}_mpi_omp PRIVATE ${CAHN_HILLIARD_INCLUDES})
        target_compile_definitions(${test_name}_mpi_omp PRIVATE PLATFORM_OMP ${CH_EXTRA_DEFS})
        add_test(NAME ${test_name}_mpi_omp COMMAND ${test_name}_mpi_omp)

        # MPI + CUDA
        if(NMFD_CUDA_ENABLED)
            add_executable(${test_name}_mpi_cuda ${test_name}.cpp)
            target_link_libraries(${test_name}_mpi_cuda PRIVATE nmfd_headers MPI::MPI_CXX CUDA::cudart)
            target_include_directories(${test_name}_mpi_cuda PRIVATE ${CAHN_HILLIARD_INCLUDES})
            target_compile_definitions(${test_name}_mpi_cuda PRIVATE PLATFORM_CUDA ${CH_EXTRA_DEFS})
            set_source_files_properties(${test_name}.cpp PROPERTIES LANGUAGE CUDA)
            add_test(NAME ${test_name}_mpi_cuda COMMAND ${test_name}_mpi_cuda)
        endif()
    endforeach()
endif()
