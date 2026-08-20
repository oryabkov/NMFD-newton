# ── detail tests ──────────────────────────────────────────────────────────────

add_executable(test_algo_hierarchy_macro test_algo_hierarchy_macro.cpp)
target_link_libraries(test_algo_hierarchy_macro PRIVATE nmfd_headers)
target_include_directories(test_algo_hierarchy_macro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME test_algo_hierarchy_macro COMMAND test_algo_hierarchy_macro)
