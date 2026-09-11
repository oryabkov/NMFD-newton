# ── operations tests ─────────────────────────────────────────────────────────

set(OPERATIONS_TESTS
        test_static_vector_space
        test_pair_vector_space
        test_dense1_extended_operator
        test_rect_vector_space
        test_dense_vector_space
        test_dense_operations
        test_dense_vector_space_static
        test_bsr_matrix
)

if(PLATFORM STREQUAL "cuda")
    list(APPEND OPERATIONS_TESTS test_dense_operations_cublas)
endif()


foreach(test_name IN LISTS OPERATIONS_TESTS)

    nmfd_add_test(
            ${test_name}
            ${test_name}.cpp
    )

endforeach()