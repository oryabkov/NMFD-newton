# Legacy Operations

The HYPRE headers are preserved unchanged for reference. They require external
project-specific HYPRE/profiling infrastructure and are not supported public
headers or part of the standalone build.

Supported vector spaces, operations, operators, and `nmfd/detail/vector_wrap.h`
come exclusively from the pinned `contrib/nmfd-operations` dependency.
The old CUDA-only `dense_operations_cuda` implementation is superseded by
`dense_operations_cuda_hip<T, Backend, Ordinal>` from that dependency.
