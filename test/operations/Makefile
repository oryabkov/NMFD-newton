-include ../common.mk

all: test_static_vector_space test_pair_vector_space test_dense1_extended_operator

test_scfd_backend_cpu: test_scfd_backend.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $< -o test_scfd_backend_cpu.bin

test_scfd_backend_cuda: test_scfd_backend.cpp ../solvers/dense_solvers_cuda_inst.cu $(CUDA_OPERATIONS_OBJECTS) $(CUDA_SOLVER_OBJECTS)
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) -DPLATFORM_CUDA -x cu -c test_scfd_backend.cpp -o test_scfd_backend_cuda.o
	$(CUDACOMPILER) $(CUDAFLAGS) test_scfd_backend_cuda.o $(CUDA_OPERATIONS_OBJECTS) $(CUDA_SOLVER_OBJECTS) -o test_scfd_backend_cuda.bin $(CUDA_SOLVER_LIBS)

test_static_vector_space: test_static_vector_space.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) test_static_vector_space.cpp -o test_static_vector_space.bin

test_pair_vector_space: test_pair_vector_space.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) test_pair_vector_space.cpp -o test_pair_vector_space.bin

test_dense1_extended_operator: test_dense1_extended_operator.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) test_dense1_extended_operator.cpp -o test_dense1_extended_operator.bin

test_rect_vector_space: test_rect_vector_space.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) test_rect_vector_space.cpp -o test_rect_vector_space.bin

# test_dense_vector_space

test_dense_vector_space_all: test_dense_vector_space_cpu test_dense_vector_space_cuda

test_dense_vector_space_cpu: test_dense_vector_space.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_dense_vector_space.cpp -o test_dense_vector_space_cpu_$(PRECISION_SUFFIX).bin

test_dense_vector_space_cuda: test_dense_vector_space.cpp $(CUDA_OPERATIONS_OBJECTS)
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) -c test_dense_vector_space.cpp -o test_dense_vector_space_cuda_$(PRECISION_SUFFIX).o
	$(CUDACOMPILER) $(CUDAFLAGS) test_dense_vector_space_cuda_$(PRECISION_SUFFIX).o $(CUDA_OPERATIONS_OBJECTS) -o test_dense_vector_space_cuda_$(PRECISION_SUFFIX).bin $(CUDA_SOLVER_LIBS)

# test_dense_vector_space_static

test_dense_vector_space_static_cpu: test_dense_vector_space_static.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_dense_vector_space_static.cpp -o test_dense_vector_space_static_cpu_$(PRECISION_SUFFIX).bin

# test_dense_operations

test_dense_operations_all: test_dense_operations_cpu test_dense_operations_cuda

test_dense_operations_cpu: test_dense_operations.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_dense_operations.cpp -o test_dense_operations_cpu_$(PRECISION_SUFFIX).bin

test_dense_operations_cuda: test_dense_operations.cpp $(CUDA_OPERATIONS_OBJECTS)
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) -c test_dense_operations.cpp -o test_dense_operations_cuda_$(PRECISION_SUFFIX).o
	$(CUDACOMPILER) $(CUDAFLAGS) test_dense_operations_cuda_$(PRECISION_SUFFIX).o $(CUDA_OPERATIONS_OBJECTS) -o test_dense_operations_cuda_$(PRECISION_SUFFIX).bin $(CUDA_SOLVER_LIBS)

# test_dense_operations_cublas (cuBLAS-accelerated)

test_dense_operations_cublas: test_dense_operations_cublas.cpp $(CUDA_OPERATIONS_OBJECTS)
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) -c test_dense_operations_cublas.cpp -o test_dense_operations_cublas_$(PRECISION_SUFFIX).o
	$(CUDACOMPILER) $(CUDAFLAGS) test_dense_operations_cublas_$(PRECISION_SUFFIX).o $(CUDA_OPERATIONS_OBJECTS) -o test_dense_operations_cublas_$(PRECISION_SUFFIX).bin $(CUDA_SOLVER_LIBS)
