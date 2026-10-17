-include ../common.mk

NEWTON_REGRESSION_TESTS = test_newton_failure_reporting.bin test_newton_backtracking.bin test_newton_convergence.bin test_newton_vector_ownership.bin test_newton_backtracking_json.bin test_newton_configuration.bin test_newton_configuration_json.bin test_newton_operator_binding.bin test_newton_dependency_ownership.bin
NEWTON_REGRESSION_HEADERS = newton_test_fixture.h $(wildcard ../../include/nmfd/solvers/*.h ../../include/nmfd/detail/*.h) ../../include/nmfd/operations/static_vector_space.h

all: test_gmres.bin test_gmres_mg.bin test_nonlinear_solver.bin test_dense1_extended_solver.bin $(NEWTON_REGRESSION_TESTS)

test:
	./test_gmres.bin
	./test_gmres_mg.bin
	./test_nonlinear_solver.bin
	./test_dense1_extended_solver.bin
	$(MAKE) test-newton-regressions

.PHONY: newton-regressions test-newton-regressions
newton-regressions: $(NEWTON_REGRESSION_TESTS)

test-newton-regressions: newton-regressions
	@set -e; for binary in $(NEWTON_REGRESSION_TESTS); do ./$$binary; done

$(filter-out %_json.bin,$(NEWTON_REGRESSION_TESTS)): %.bin: %.cpp $(NEWTON_REGRESSION_HEADERS)
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $< -o $@

test_newton_%_json.bin: test_newton_%.cpp $(NEWTON_REGRESSION_HEADERS) ../../contrib/json/nlohmann/json.hpp
	$(HOSTCOMPILER) $(HOSTFLAGS) -DNMFD_ENABLE_NLOHMANN $(INCLUDE_CONTRIB) -I../../contrib/json $< -o $@

test_gmres.bin: test_gmres.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_gmres.cpp -o test_gmres.bin
test_gmres_mg.bin: test_gmres_mg.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_gmres_mg.cpp -o test_gmres_mg.bin
test_nonlinear_solver.bin: test_nonlinear_solver.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_nonlinear_solver.cpp -o test_nonlinear_solver.bin
test_dense1_extended_solver.bin: test_dense1_extended_solver.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_dense1_extended_solver.cpp -o test_dense1_extended_solver.bin

# test_gmres_dense

test_gmres_dense_all: test_gmres_dense_cpu test_gmres_dense_cuda

test_gmres_dense_cpu: test_gmres_dense.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_gmres_dense.cpp -o test_gmres_dense_cpu_$(PRECISION_SUFFIX).bin

test_gmres_dense_cuda: test_gmres_dense.cpp dense_solvers_cuda_inst.cu
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA $(PRECISION_DEFINE) -x cu test_gmres_dense.cpp dense_solvers_cuda_inst.cu -o test_gmres_dense_cuda_$(PRECISION_SUFFIX).bin $(CUDA_SOLVER_LIBS)

# test_nonlinear_solver_dense

test_nonlinear_solver_dense_all: test_nonlinear_solver_dense_cpu test_nonlinear_solver_dense_cuda

test_nonlinear_solver_dense_cpu: test_nonlinear_solver_dense.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_nonlinear_solver_dense.cpp -o test_nonlinear_solver_dense_cpu_$(PRECISION_SUFFIX).bin

test_nonlinear_solver_dense_cuda: test_nonlinear_solver_dense.cpp dense_solvers_cuda_inst.cu
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA $(PRECISION_DEFINE) -x cu test_nonlinear_solver_dense.cpp dense_solvers_cuda_inst.cu -o test_nonlinear_solver_dense_cuda_$(PRECISION_SUFFIX).bin $(CUDA_SOLVER_LIBS)

# test_direct_solver_dense

test_direct_solver_dense_all: test_direct_solver_dense_cpu test_direct_solver_dense_cuda

test_direct_solver_dense_cpu: test_direct_solver_dense.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_direct_solver_dense.cpp -o test_direct_solver_dense_cpu_$(PRECISION_SUFFIX).bin

test_direct_solver_dense_cuda: test_direct_solver_dense.cpp dense_solvers_cuda_inst.cu
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA $(PRECISION_DEFINE) -x cu test_direct_solver_dense.cpp dense_solvers_cuda_inst.cu -o test_direct_solver_dense_cuda_$(PRECISION_SUFFIX).bin $(CUDA_SOLVER_LIBS)

# test_newton_direct_solver_dense

test_newton_direct_solver_dense_all: test_newton_direct_solver_dense_cpu test_newton_direct_solver_dense_cuda

test_newton_direct_solver_dense_cpu: test_newton_direct_solver_dense.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_newton_direct_solver_dense.cpp -o test_newton_direct_solver_dense_cpu_$(PRECISION_SUFFIX).bin

test_newton_direct_solver_dense_cuda: test_newton_direct_solver_dense.cpp dense_solvers_cuda_inst.cu
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA $(PRECISION_DEFINE) -x cu test_newton_direct_solver_dense.cpp dense_solvers_cuda_inst.cu -o test_newton_direct_solver_dense_cuda_$(PRECISION_SUFFIX).bin $(CUDA_SOLVER_LIBS)
