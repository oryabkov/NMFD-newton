# Solver Dependencies

Initialize direct dependencies with `git submodule update --init`.
Public solver/backend headers require these include paths:

```
include
contrib/nmfd-linsolvers/include
contrib/nmfd-operations/include
contrib/SCFD/include
contrib/json
```

`contrib/nmfd-operations` owns the supported operations, spaces, block operators,
and `nmfd/detail/vector_wrap.h`. Standalone builds use the direct `contrib/SCFD`
pin. Dependency repositories' own submodules need not be initialized.
`contrib/nmfd-linsolvers` owns GMRES, Jacobi, the iterative-solver base, monitors,
preconditioners, and shared hierarchy helpers. There are no overlapping public
headers between the two solver repositories, so include order cannot select a
different implementation. The existing `dense1_extended_solver` now also lives
in NMFD-linsolvers, retaining its public header and interface. Jacobi keeps its
existing template order and constructors; include
`nmfd/solvers/jacobi.h` from NMFD-linsolvers without changing client code.

`NMFD_LINSOLVERS_ROOT`, `NMFD_OPERATIONS_ROOT`, and `SCFD_ROOT` can override the
default paths. Keep operations and SCFD pins compatible across dependencies.
HYPRE reference headers are in `legacy/operations`, outside the public include
tree. No supported operations are duplicated in this repository.

## CI Modes

Pinned jobs use committed direct gitlinks. The separate `latest-dependencies`
job selects the branches declared in `.gitmodules`, logs the tested revisions,
and verifies CPU solvers, public headers, original CH programs, and MPI. It runs
on pushes, pull requests, manual dispatch, and daily from the default branch.
It never commits or pushes changed pins. Run remote updates only in an isolated
checkout, not over a developer's working trees.

Integrating projects supply `SCFD_ROOT`, `NMFD_OPERATIONS_ROOT`, and
`NMFD_LINSOLVERS_ROOT` to select one canonical checkout of each library. Ordinary
builds do not fetch, switch branches, or modify dependency metadata.

## Verification

```
make -j2 test_cpu
make check_headers_cpu
make check_linsolvers check_ch_serial check_ch_omp
make -j2 check_ch_original
make -j2 check_ch_mpi MPI_NP=4 MPIEXEC_FLAGS=--oversubscribe
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

The original CH entry points have interface-only updates for SCFD's templated
backends, distributor construction, halo-aware allocation, and explicit mobility
binding. Their equations, algorithms, parameters and internal criteria are unchanged.
`check_ch_original` builds and smoke-tests all six programs. These short runs
check successful execution and finite output, not long-time convergence.
`check_ch_mpi` needs an MPI compiler/launcher and runs the original collective
vector-space test and biharmonic GMRES solves with diagonal and multigrid
preconditioners. It requires reported convergence and compares one-rank and
multi-rank solution errors. Use `MPICXX`, `MPIEXEC`, `MPIEXEC_FLAGS` and `MPI_NP`
to configure MPI. Distributed nonlinear CH evolution is not covered by this gate.
HIP, HYPRE, and mesh/DG builds remain uncertified.
Existing `test/*/Makefile` entry points remain available with updated dependencies.

Newton's public interfaces and algorithms are unchanged. GMRES uses the
NMFD-linsolvers implementation, including its existing numerical safeguards.
Computational dependencies retain shared ownership; loggers remain borrowed
`Log*` and must outlive their users. Vector spaces expose `is_valid_number()`;
Newton calls this single validity-check interface directly.
