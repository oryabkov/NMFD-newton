# ── solvers tests ─────────────────────────────────────────────────────────────

set(SOLVER_TESTS
        test_gmres
        test_gmres_mg
        test_nonlinear_solver
        test_dense1_extended_solver
        test_nonlinear_solver_circle
)


foreach(test_name IN LISTS SOLVER_TESTS)

    nmfd_add_test(
            ${test_name}
            ${test_name}.cpp
    )

endforeach()