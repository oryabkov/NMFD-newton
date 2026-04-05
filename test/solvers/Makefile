-include ../common.mk

all: test_gmres.bin test_gmres_mg.bin test_nonlinear_solver.bin test_dense1_extended_solver.bin

test:
	./test_gmres.bin
	./test_gmres_mg.bin
	./test_nonlinear_solver.bin
	./test_dense1_extended_solver.bin

test_gmres.bin: test_gmres.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_gmres.cpp -o test_gmres.bin
test_gmres_mg.bin: test_gmres_mg.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_gmres_mg.cpp -o test_gmres_mg.bin
test_nonlinear_solver.bin: test_nonlinear_solver.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_nonlinear_solver.cpp -o test_nonlinear_solver.bin
test_dense1_extended_solver.bin: test_dense1_extended_solver.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_dense1_extended_solver.cpp -o test_dense1_extended_solver.bin

# test_gmres_dense

test_gmres_dense_cpu: test_gmres_dense.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_gmres_dense.cpp -o test_gmres_dense_cpu_$(PRECISION_SUFFIX).bin

test_gmres_dense_cuda: test_gmres_dense.cpp
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) test_gmres_dense.cpp -o test_gmres_dense_cuda_$(PRECISION_SUFFIX).bin

# test_nonlinear_solver_dense

test_nonlinear_solver_dense_all: test_nonlinear_solver_dense_cpu test_nonlinear_solver_dense_cuda

test_nonlinear_solver_dense_cpu: test_nonlinear_solver_dense.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_nonlinear_solver_dense.cpp -o test_nonlinear_solver_dense_cpu_$(PRECISION_SUFFIX).bin

test_nonlinear_solver_dense_cuda: test_nonlinear_solver_dense.cpp
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) test_nonlinear_solver_dense.cpp -o test_nonlinear_solver_dense_cuda_$(PRECISION_SUFFIX).bin

# test_direct_solver_dense

test_direct_solver_dense_all: test_direct_solver_dense_cpu test_direct_solver_dense_cuda

test_direct_solver_dense_cpu: test_direct_solver_dense.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_direct_solver_dense.cpp -o test_direct_solver_dense_cpu_$(PRECISION_SUFFIX).bin

test_direct_solver_dense_cuda: test_direct_solver_dense.cpp
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) test_direct_solver_dense.cpp -o test_direct_solver_dense_cuda_$(PRECISION_SUFFIX).bin

# test_newton_direct_solver_dense

test_newton_direct_solver_dense_all: test_newton_direct_solver_dense_cpu test_newton_direct_solver_dense_cuda

test_newton_direct_solver_dense_cpu: test_newton_direct_solver_dense.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_newton_direct_solver_dense.cpp -o test_newton_direct_solver_dense_cpu_$(PRECISION_SUFFIX).bin

test_newton_direct_solver_dense_cuda: test_newton_direct_solver_dense.cpp
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) test_newton_direct_solver_dense.cpp -o test_newton_direct_solver_dense_cuda_$(PRECISION_SUFFIX).bin
