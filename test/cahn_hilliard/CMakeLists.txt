# ── cahn_hilliard tests ───────────────────────────────────────────────────────

set(CAHN_HILLIARD_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CH_EXTRA_DEFS
        SCFD_ARRAYS_ENABLE_INDEX_SHIFT=1
)


# ── Runtime parameters ───────────────────────────────────────────────────────

set(NMFD_CH_SOLVER "gmres"
        CACHE STRING
        "Solver used by cahn_hilliard tests: 'jacobi' or 'gmres'")

set(NMFD_CH_PRECONDITIONER "mg"
        CACHE STRING
        "Preconditioner for cahn_hilliard tests: 'diag' or 'mg'")

set(NMFD_CH_GRID_SIZE "32"
        CACHE STRING
        "Grid points per dimension for cahn_hilliard tests")

set(NMFD_CH_GRID_SIZES "32"
        CACHE STRING
        "Grid sizes for the experiment, e.g. '2 4 8 16 32 64 128 256'")

mark_as_advanced(
        NMFD_CH_SOLVER
        NMFD_CH_PRECONDITIONER
        NMFD_CH_GRID_SIZE
        NMFD_CH_GRID_SIZES
)

string(
        REPLACE " "
        ";"
        NMFD_CH_GRID_SIZES
        "${NMFD_CH_GRID_SIZES}"
)


# ── Tests ─────────────────────────────────────────────────────────────────────

set(NMFD_CH_TESTS
        test_biharmonic
        test_cahn_hilliard
        test_time_cahn_hilliard
        test_random_init_cahn_hilliard
)


foreach(test_name IN LISTS NMFD_CH_TESTS)

    set(TEST_TARGET
            ${test_name}_${NMFD_PLATFORM_SUFFIX}_${NMFD_PRECISION_SUFFIX})

    add_executable(${TEST_TARGET} ${test_name}.cpp)
    set_target_properties(${TEST_TARGET} PROPERTIES SUFFIX ".bin")
    target_link_libraries(${TEST_TARGET} PRIVATE nmfd_headers
            ${NMFD_PLATFORM_LIBS})

    target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_compile_definitions(${TEST_TARGET} PRIVATE ${NMFD_PLATFORM_DEFINITIONS}
            ${NMFD_PRECISION_DEFINE})

    if(PLATFORM STREQUAL "cuda")
        set_target_properties(${TEST_TARGET} PROPERTIES CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON)
    endif()

endforeach()


# ── Additional definitions ───────────────────────────────────────────────────

foreach(test_name IN LISTS NMFD_CH_TESTS)

    set(test_target
            ${test_name}_${NMFD_PLATFORM_SUFFIX}_${NMFD_PRECISION_SUFFIX})

    target_compile_definitions(
            ${test_target}
            PRIVATE
            ${CH_EXTRA_DEFS}
    )

    target_include_directories(
            ${test_target}
            PRIVATE
            ${CAHN_HILLIARD_INCLUDES}
    )

endforeach()


# ── MPI ──────────────────────────────────────────────────────────────────────

if(NMFD_WITH_MPI)

    foreach(test_name IN LISTS NMFD_CH_TESTS)

        set(test_target
                ${test_name}_${NMFD_PLATFORM_SUFFIX}_${NMFD_PRECISION_SUFFIX})

        target_link_libraries(
                ${test_target}
                PRIVATE
                MPI::MPI_CXX
        )

    endforeach()

endif()


# ── Cahn-Hilliard experiments ────────────────────────────────────────────────
#
# The executable is created only once.
# Different grid sizes are different CTest invocations.
#

foreach(test_name IN LISTS NMFD_CH_TESTS)

    foreach(grid_size IN LISTS NMFD_CH_GRID_SIZES)

        add_test(
                NAME ${test_name}_${NMFD_PLATFORM_SUFFIX}_${NMFD_PRECISION_SUFFIX}_${grid_size}
                COMMAND
                ${test_name}_${NMFD_PLATFORM_SUFFIX}_${NMFD_PRECISION_SUFFIX}
                ${NMFD_CH_SOLVER}
                ${NMFD_CH_PRECONDITIONER}
                ${grid_size}
        )

    endforeach()

endforeach()