# Solver Dependencies

Initialize with `git submodule update --init --recursive`.
Public solver/backend headers require these include paths:

```
include
contrib/nmfd-linsolvers/include
contrib/nmfd-operations/include
contrib/nmfd-operations/contrib/scfd/include
contrib/json
```

`contrib/nmfd-operations` owns the supported operations, spaces, block operators,
and `nmfd/detail/vector_wrap.h`. Its SCFD pin is used by default; the historical
direct `contrib/SCFD` checkout remains only for older external builds.
`contrib/nmfd-linsolvers` owns GMRES, the iterative-solver base, monitors,
preconditioners, and shared hierarchy helpers. There are no overlapping public
headers between the two solver repositories, so include order cannot select a
different implementation. Jacobi and the bordered solver remain here until their
own migration; they use the canonical linear-solver base and contracts.

`NMFD_LINSOLVERS_ROOT`, `NMFD_OPERATIONS_ROOT`, and `SCFD_ROOT` can override the
default paths. Keep operations and SCFD pins compatible across dependencies.
HYPRE reference headers are in `legacy/operations`, outside the public include
tree. No supported operations are duplicated in this repository.

## Verification

```
make -j2 test_cpu
make check_headers_cpu
make check_linsolvers check_ch_serial check_ch_omp
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
The linear-solver compatibility test checks old constructors, virtual-base
dispatch, operator rebinding, utility hierarchies, JSON input, Newton failures,
and both include orders. Cahn-Hilliard tests use the existing coupled residual,
Jacobian, and diagonal preconditioner on a 4x4x4 grid, float/double, and left/right
GMRES on serial CPU, OMP, and CUDA.

These are single-process kernel/solver tests, not certification of all historical
Cahn-Hilliard programs. The old serial entry points already omit distributor
arguments required by their MPI-era helpers; this migration does not rewrite
those applications. HIP, MPI, HYPRE, and mesh/DG builds remain uncertified.
Existing `test/*/Makefile` entry points remain available with updated dependencies.

Newton's public interfaces and algorithms are unchanged. GMRES uses the
NMFD-linsolvers implementation, including its existing numerical safeguards.
Computational dependencies retain shared ownership; loggers remain borrowed
`Log*` and must outlive their users. Vector spaces expose `is_valid_number()`;
Newton calls this single validity-check interface directly.
