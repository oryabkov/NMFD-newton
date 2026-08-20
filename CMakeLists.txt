cmake_minimum_required(VERSION 3.18)
project(NMFD LANGUAGES C CXX)

include(CheckLanguage)

# ── Options ──────────────────────────────────────────────────────────────────
option(NMFD_WITH_CUDA       "Enable CUDA support"                      OFF)
option(NMFD_WITH_MPI        "Enable MPI support"                       OFF)
option(NMFD_WITH_TESTS      "Enable tests"                             ON)
option(NMFD_USE_APPLE_OMP   "Use Homebrew OpenMP on macOS"             OFF)

set(NMFD_FLOAT_TYPE "float" CACHE STRING "Floating point type: float or double")

# ── C++ standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── SCFD submodule (already in contrib/SCFD) ─────────────────────────────────
set(SCFD_WITH_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(contrib/SCFD)

# ── NMFD headers (header-only library) ───────────────────────────────────────
add_library(nmfd_headers INTERFACE)
target_include_directories(nmfd_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/contrib/SCFD/include
    ${CMAKE_CURRENT_SOURCE_DIR}/contrib/json
)

# ── OpenMP ───────────────────────────────────────────────────────────────────
if(NMFD_USE_APPLE_OMP)
    find_package(OpenMP COMPONENTS CXX REQUIRED)
else()
    find_package(OpenMP REQUIRED)
endif()

# ── CUDA ─────────────────────────────────────────────────────────────────────
if(NMFD_WITH_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "NMFD with CUDA support")
        message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

        # FindCUDAToolkit for libraries (cublas, cusparse, etc.)
        find_package(CUDAToolkit REQUIRED)

        # CUDA architectures
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89")
        endif()

        set(NMFD_CUDA_ENABLED 1)
    else()
        message(WARNING "No CUDA compiler found!")
    endif()
endif()

# ── MPI ──────────────────────────────────────────────────────────────────────
if(NMFD_WITH_MPI)
    find_package(MPI COMPONENTS CXX REQUIRED)
    message(STATUS "NMFD with MPI support")
    message(STATUS "MPI CXX compiler: ${MPI_CXX_COMPILER}")
    set(NMFD_MPI_ENABLED 1)
endif()

# ── Float type define ────────────────────────────────────────────────────────
if(NMFD_FLOAT_TYPE STREQUAL "double")
    add_compile_definitions(USE_DOUBLE_PRECISION)
    set(NMFD_PRECISION_SUFFIX "d")
else()
    set(NMFD_PRECISION_SUFFIX "f")
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
if(NMFD_WITH_TESTS)
    enable_testing()

    # GoogleTest — submodule in contrib/googletest (preferred) or FetchContent
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/contrib/googletest/CMakeLists.txt)
        message(STATUS "Using local googletest from contrib/googletest")
        add_subdirectory(contrib/googletest)
    else()
        message(STATUS "Fetching googletest via FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/release-1.14.0.zip
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_subdirectory(test)
endif()
