# Operations Dependency

Initialize with `git submodule update --init --recursive`.
Public solver/backend headers require these include paths:

```
include
contrib/nmfd-operations/include
contrib/nmfd-operations/contrib/scfd/include
contrib/json
```

`contrib/nmfd-operations` owns the supported operations, spaces, block operators,
and `nmfd/detail/vector_wrap.h`. Its SCFD pin is used by default; the historical
direct `contrib/SCFD` checkout remains only for older external builds.
`NMFD_OPERATIONS_ROOT` and `SCFD_ROOT` can override the default paths.
HYPRE reference headers are in `legacy/operations`, outside the public include
tree. No supported operations are duplicated in this repository.

## Verification

```
make -j2 test_cpu
make check_headers_cpu
make -j2 test_cpu CONFIG_FILE=build_configs/config_clang.inc BUILD_DIR=build/clang
make -j2 test_cpu CONFIG_FILE=build_configs/config_sanitizers.inc BUILD_DIR=build/sanitizers
make check_cuda check_headers_cuda BUILD_DIR=build/cuda CUDA_ARCH=75
```

CPU tests need a C++17 compiler and OpenMP. CUDA tests need NVCC, a matching GPU,
cuBLAS, and cuSOLVER. The Makefile compiles the operations dependency's compensated
reduction kernels once per build directory. Both float and double dense solves
are checked; Newton regressions also cover long double.

Header checks compile each core backend/detail/solver/preconditioner header in
isolation. CPU checks cover both JSON modes; CUDA checks select the CUDA backend.
HIP, MPI, HYPRE, mesh/DG, and Cahn-Hilliard application builds are not certified by
these core checks. Existing `test/*/Makefile` entry points remain available.

Solver algorithms and public interfaces are unchanged by this extraction.
Computational dependencies retain shared ownership; loggers remain borrowed
`Log*` and must outlive their users. Internal linear-solver replacement is a
separate migration.
