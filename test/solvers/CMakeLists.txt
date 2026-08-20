# ── solvers tests ─────────────────────────────────────────────────────────────

set(SOLVER_TESTS
    test_gmres
    test_gmres_mg
    test_nonlinear_solver
    test_dense1_extended_solver
    test_nonlinear_solver_circle
)

foreach(test_name IN LISTS SOLVER_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE nmfd_headers)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
