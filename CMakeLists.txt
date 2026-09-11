cmake_minimum_required(VERSION 3.18)

project(NMFD LANGUAGES C CXX)

include(CheckLanguage)

# ── Build configuration ──────────────────────────────────────────────────────

set(PLATFORM
    "cpu"
    CACHE STRING "Build platform: cpu, omp or cuda")

set_property(CACHE PLATFORM PROPERTY STRINGS cpu omp cuda)

option(NMFD_WITH_MPI "Enable MPI support" OFF)

option(NMFD_WITH_TESTS "Build tests" ON)

option(NMFD_USE_APPLE_OMP "Use Homebrew OpenMP on macOS" OFF)

set(NMFD_FLOAT_TYPE
    "float"
    CACHE STRING "Floating point type: float or double")

set_property(CACHE NMFD_FLOAT_TYPE PROPERTY STRINGS float double)

# ── Validate configuration ───────────────────────────────────────────────────

if(NOT PLATFORM STREQUAL "cpu"
   AND NOT PLATFORM STREQUAL "omp"
   AND NOT PLATFORM STREQUAL "cuda")

  message(FATAL_ERROR "Unknown PLATFORM='${PLATFORM}'. "
                      "Supported platforms are: cpu, omp, cuda")
endif()

if(NOT NMFD_FLOAT_TYPE STREQUAL "float" AND NOT NMFD_FLOAT_TYPE STREQUAL
                                            "double")

  message(FATAL_ERROR "NMFD_FLOAT_TYPE must be 'float' or 'double', "
                      "got '${NMFD_FLOAT_TYPE}'")
endif()

# ── C++ standard ─────────────────────────────────────────────────────────────

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── SCFD ─────────────────────────────────────────────────────────────────────

set(SCFD_WITH_TESTS
    OFF
    CACHE BOOL "" FORCE)

add_subdirectory(contrib/SCFD)

# ── NMFD headers ─────────────────────────────────────────────────────────────

add_library(nmfd_headers INTERFACE)

target_include_directories(
  nmfd_headers
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/contrib/SCFD/include
            ${CMAKE_CURRENT_SOURCE_DIR}/contrib/json)

# ── Platform configuration ───────────────────────────────────────────────────

set(NMFD_PLATFORM_SUFFIX "")
set(NMFD_PLATFORM_DEFINITIONS "")
set(NMFD_PLATFORM_LIBS "")
set(NMFD_PLATFORM_LANGUAGE CXX)

if(PLATFORM STREQUAL "cpu")

  set(NMFD_PLATFORM_SUFFIX "cpu")
  set(NMFD_PLATFORM_DEFINITIONS PLATFORM_SERIAL_CPU)
  set(NMFD_PLATFORM_LANGUAGE CXX)

elseif(PLATFORM STREQUAL "omp")

  find_package(OpenMP REQUIRED)

  set(NMFD_PLATFORM_SUFFIX "omp")
  set(NMFD_PLATFORM_DEFINITIONS PLATFORM_OMP)
  set(NMFD_PLATFORM_LIBS OpenMP::OpenMP_CXX)
  set(NMFD_PLATFORM_LANGUAGE CXX)

elseif(PLATFORM STREQUAL "cuda")

  check_language(CUDA)

  if(NOT CMAKE_CUDA_COMPILER)
    message(
      FATAL_ERROR "PLATFORM=cuda was requested, but no CUDA compiler was found")
  endif()

  enable_language(CUDA)

  find_package(CUDAToolkit REQUIRED)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES
        "70;75;80;86;89"
        CACHE STRING "CUDA architectures")
  endif()

  set(NMFD_PLATFORM_SUFFIX "cuda")
  set(NMFD_PLATFORM_DEFINITIONS PLATFORM_CUDA)
  set(NMFD_PLATFORM_LIBS CUDA::cudart)
  set(NMFD_PLATFORM_LANGUAGE CUDA)

endif()

# ── MPI configuration ────────────────────────────────────────────────────────

if(NMFD_WITH_MPI)

  find_package(
    MPI
    COMPONENTS CXX
    REQUIRED)

  set(NMFD_WITH_MPI_VALUE 1)

else()

  set(NMFD_WITH_MPI_VALUE 0)

endif()

# ── Precision configuration ──────────────────────────────────────────────────

if(NMFD_FLOAT_TYPE STREQUAL "double")
  set(NMFD_PRECISION_SUFFIX "d")
  set(NMFD_PRECISION_DEFINE "USE_DOUBLE_PRECISION")
elseif(NMFD_FLOAT_TYPE STREQUAL "float")
  set(NMFD_PRECISION_SUFFIX "f")
  set(NMFD_PRECISION_DEFINE "")
else()
  message(FATAL_ERROR "NMFD_FLOAT_TYPE must be 'float' or 'double'")
endif()

# ── Global platform definitions ───────────────────────────────────────────────
#
# Every target receives the same platform configuration.
#
# PLATFORM_CPU / PLATFORM_OMP / PLATFORM_CUDA NMFD_WITH_MPI=0/1
#

if(PLATFORM STREQUAL "cpu")
  add_compile_definitions(PLATFORM_CPU)
elseif(PLATFORM STREQUAL "omp")
  add_compile_definitions(PLATFORM_OMP)
elseif(PLATFORM STREQUAL "cuda")
  add_compile_definitions(PLATFORM_CUDA)
endif()

add_compile_definitions(NMFD_WITH_MPI=${NMFD_WITH_MPI_VALUE})

# ── CUDA source configuration ────────────────────────────────────────────────
#
# There is only one platform in one CMake configuration. Therefore marking the
# source as CUDA here does not conflict with CPU/OMP targets as it did in the
# previous multi-platform approach.
#

if(PLATFORM STREQUAL "cuda")

  file(GLOB_RECURSE NMFD_TEST_CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)

  foreach(source_file IN LISTS NMFD_TEST_CPP_FILES)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
  endforeach()

endif()

# ── Helper for tests ─────────────────────────────────────────────────────────
#
# Creates exactly one target for the selected platform.
#

function(nmfd_add_test TEST_NAME)

  set(TEST_TARGET
      ${TEST_NAME}_${NMFD_PLATFORM_SUFFIX}_${NMFD_PRECISION_SUFFIX})

  add_executable(${TEST_TARGET} ${ARGN})
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

  add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})

endfunction()

# ── Tests ────────────────────────────────────────────────────────────────────

if(NMFD_WITH_TESTS)

  enable_testing()

  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/contrib/googletest/CMakeLists.txt)

    message(STATUS "Using local googletest from contrib/googletest")

    add_subdirectory(contrib/googletest)

  else()

    message(STATUS "Fetching googletest via FetchContent")

    include(FetchContent)

    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/release-1.14.0.zip)

    set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

  endif()

  add_subdirectory(test)

endif()

# ── Configuration summary ─────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "NMFD configuration:")
message(STATUS "  PLATFORM              = ${PLATFORM}")
message(STATUS "  NMFD_WITH_MPI         = ${NMFD_WITH_MPI}")
message(STATUS "  NMFD_PRECISION_SUFFIX = ${NMFD_PRECISION_SUFFIX}")
message(STATUS "  TESTS                 = ${NMFD_WITH_TESTS}")
message(STATUS "")
