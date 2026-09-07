# ── operations tests ──────────────────────────────────────────────────────────

# CPU-only tests
set(OPERATIONS_CPU_TESTS
    test_static_vector_space
    test_pair_vector_space
    test_dense1_extended_operator
    test_rect_vector_space
)

foreach(test_name IN LISTS OPERATIONS_CPU_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE nmfd_headers)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Tests with CUDA variants
set(OPERATIONS_DUAL_TESTS
    test_dense_vector_space
    test_dense_operations
)

foreach(test_name IN LISTS OPERATIONS_DUAL_TESTS)
    # CPU variant
    add_executable(${test_name}_cpu_${NMFD_PRECISION_SUFFIX} ${test_name}.cpp)
    target_link_libraries(${test_name}_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers)
    target_include_directories(${test_name}_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(${test_name}_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_SERIAL_CPU)
    add_test(NAME ${test_name}_cpu COMMAND ${test_name}_cpu_${NMFD_PRECISION_SUFFIX})

    # CUDA variant
    if(NMFD_CUDA_ENABLED)
        add_executable(${test_name}_cuda_${NMFD_PRECISION_SUFFIX} ${test_name}.cpp)
        target_link_libraries(${test_name}_cuda_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers CUDA::cudart)
        target_include_directories(${test_name}_cuda_${NMFD_PRECISION_SUFFIX} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(${test_name}_cuda_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_CUDA)
        set_source_files_properties(${test_name}.cpp PROPERTIES LANGUAGE CUDA)
        add_test(NAME ${test_name}_cuda COMMAND ${test_name}_cuda_${NMFD_PRECISION_SUFFIX})
    endif()
endforeach()

# CPU-only variant of dense_vector_space_static (precision suffix, like make)
add_executable(test_dense_vector_space_static_cpu_${NMFD_PRECISION_SUFFIX} test_dense_vector_space_static.cpp)
target_link_libraries(test_dense_vector_space_static_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers)
target_include_directories(test_dense_vector_space_static_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(test_dense_vector_space_static_cpu_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_SERIAL_CPU)
add_test(NAME test_dense_vector_space_static_cpu COMMAND test_dense_vector_space_static_cpu_${NMFD_PRECISION_SUFFIX})

# cuBLAS variant (precision suffix, like make)
if(NMFD_CUDA_ENABLED)
    add_executable(test_dense_operations_cublas_${NMFD_PRECISION_SUFFIX} test_dense_operations_cublas.cpp)
    target_link_libraries(test_dense_operations_cublas_${NMFD_PRECISION_SUFFIX} PRIVATE nmfd_headers CUDA::cudart CUDA::cublas)
    target_include_directories(test_dense_operations_cublas_${NMFD_PRECISION_SUFFIX} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(test_dense_operations_cublas_${NMFD_PRECISION_SUFFIX} PRIVATE PLATFORM_CUDA)
    set_source_files_properties(test_dense_operations_cublas.cpp PROPERTIES LANGUAGE CUDA)
    add_test(NAME test_dense_operations_cublas COMMAND test_dense_operations_cublas_${NMFD_PRECISION_SUFFIX})
endif()

# BSR matrix tests (host-only)
add_executable(test_bsr_matrix test_bsr_matrix.cpp)
target_link_libraries(test_bsr_matrix PRIVATE nmfd_headers)
target_include_directories(test_bsr_matrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_bsr_matrix COMMAND test_bsr_matrix)