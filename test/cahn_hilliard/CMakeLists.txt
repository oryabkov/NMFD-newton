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
)

foreach(test_name IN LISTS CH_DUAL_TESTS)
    # CPU variant
    add_executable(${test_name}_cpu ${test_name}.cpp)
    target_link_libraries(${test_name}_cpu PRIVATE nmfd_headers)
    target_include_directories(${test_name}_cpu PRIVATE ${CAHN_HILLIARD_INCLUDES})
    target_compile_definitions(${test_name}_cpu PRIVATE PLATFORM_SERIAL_CPU ${CH_EXTRA_DEFS})

    # OpenMP variant
    add_executable(${test_name}_omp ${test_name}.cpp)
    target_link_libraries(${test_name}_omp PRIVATE nmfd_headers OpenMP::OpenMP_CXX)
    target_include_directories(${test_name}_omp PRIVATE ${CAHN_HILLIARD_INCLUDES})
    target_compile_definitions(${test_name}_omp PRIVATE PLATFORM_OMP ${CH_EXTRA_DEFS})

    # CUDA variant
    if(NMFD_CUDA_ENABLED)
        add_executable(${test_name}_cuda ${test_name}.cpp)
        target_link_libraries(${test_name}_cuda PRIVATE nmfd_headers CUDA::cudart)
        target_include_directories(${test_name}_cuda PRIVATE ${CAHN_HILLIARD_INCLUDES})
        target_compile_definitions(${test_name}_cuda PRIVATE PLATFORM_CUDA ${CH_EXTRA_DEFS})
        set_source_files_properties(${test_name}.cpp PROPERTIES LANGUAGE CUDA)
    endif()
endforeach()

# Register cahn_hilliard tests with appropriate arguments
# test_biharmonic/test_cahn_hilliard/test_time_cahn_hilliard take: solver preconditioner grid_size
add_test(NAME test_biharmonic_cpu
    COMMAND test_biharmonic_cpu gmres mg 32)
add_test(NAME test_biharmonic_omp
    COMMAND test_biharmonic_omp gmres mg 32)
add_test(NAME test_cahn_hilliard_cpu
    COMMAND test_cahn_hilliard_cpu gmres mg 32)
add_test(NAME test_cahn_hilliard_omp
    COMMAND test_cahn_hilliard_omp gmres mg 32)
add_test(NAME test_time_cahn_hilliard_cpu
    COMMAND test_time_cahn_hilliard_cpu gmres mg 32)
add_test(NAME test_time_cahn_hilliard_omp
    COMMAND test_time_cahn_hilliard_omp gmres mg 32)

# test_random_init_cahn_hilliard takes: grid_size
add_executable(test_random_init_cahn_hilliard_cpu test_random_init_cahn_hilliard.cpp)
target_link_libraries(test_random_init_cahn_hilliard_cpu PRIVATE nmfd_headers)
target_include_directories(test_random_init_cahn_hilliard_cpu PRIVATE ${CAHN_HILLIARD_INCLUDES})
target_compile_definitions(test_random_init_cahn_hilliard_cpu PRIVATE PLATFORM_SERIAL_CPU ${CH_EXTRA_DEFS})
add_test(NAME test_random_init_cahn_hilliard_cpu
    COMMAND test_random_init_cahn_hilliard_cpu 32)

add_executable(test_random_init_cahn_hilliard_omp test_random_init_cahn_hilliard.cpp)
target_link_libraries(test_random_init_cahn_hilliard_omp PRIVATE nmfd_headers OpenMP::OpenMP_CXX)
target_include_directories(test_random_init_cahn_hilliard_omp PRIVATE ${CAHN_HILLIARD_INCLUDES})
target_compile_definitions(test_random_init_cahn_hilliard_omp PRIVATE PLATFORM_OMP ${CH_EXTRA_DEFS})
add_test(NAME test_random_init_cahn_hilliard_omp
    COMMAND test_random_init_cahn_hilliard_omp 32)

# MPI variants
if(NMFD_MPI_ENABLED)
    foreach(test_name IN LISTS CH_DUAL_TESTS)
        # MPI + CPU
        add_executable(${test_name}_mpi_cpu ${test_name}.cpp)
        target_link_libraries(${test_name}_mpi_cpu PRIVATE nmfd_headers MPI::MPI_CXX)
        target_include_directories(${test_name}_mpi_cpu PRIVATE ${CAHN_HILLIARD_INCLUDES})
        target_compile_definitions(${test_name}_mpi_cpu PRIVATE PLATFORM_SERIAL_CPU ${CH_EXTRA_DEFS})
        add_test(NAME ${test_name}_mpi_cpu COMMAND ${test_name}_mpi_cpu gmres mg 32)

        # MPI + OpenMP
        add_executable(${test_name}_mpi_omp ${test_name}.cpp)
        target_link_libraries(${test_name}_mpi_omp PRIVATE nmfd_headers MPI::MPI_CXX OpenMP::OpenMP_CXX)
        target_include_directories(${test_name}_mpi_omp PRIVATE ${CAHN_HILLIARD_INCLUDES})
        target_compile_definitions(${test_name}_mpi_omp PRIVATE PLATFORM_OMP ${CH_EXTRA_DEFS})
        add_test(NAME ${test_name}_mpi_omp COMMAND ${test_name}_mpi_omp gmres mg 32)

        # MPI + CUDA
        if(NMFD_CUDA_ENABLED)
            add_executable(${test_name}_mpi_cuda ${test_name}.cpp)
            target_link_libraries(${test_name}_mpi_cuda PRIVATE nmfd_headers MPI::MPI_CXX CUDA::cudart)
            target_include_directories(${test_name}_mpi_cuda PRIVATE ${CAHN_HILLIARD_INCLUDES})
            target_compile_definitions(${test_name}_mpi_cuda PRIVATE PLATFORM_CUDA ${CH_EXTRA_DEFS})
            set_source_files_properties(${test_name}.cpp PROPERTIES LANGUAGE CUDA)
            add_test(NAME ${test_name}_mpi_cuda COMMAND ${test_name}_mpi_cuda gmres mg 32)
        endif()
    endforeach()
endif()
