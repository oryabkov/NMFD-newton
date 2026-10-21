CONFIG_FILE ?= build_configs/config_cpu.inc
include $(CONFIG_FILE)

BUILD_DIR ?= build
NMFD_OPERATIONS_ROOT ?= contrib/nmfd-operations
NMFD_LINSOLVERS_ROOT ?= contrib/nmfd-linsolvers
SCFD_ROOT ?= contrib/SCFD
CUDA_ROOT ?= /usr/local/cuda
NVCC ?= $(CUDA_ROOT)/bin/nvcc
CUDA_ARCH ?= 75
NVCCFLAGS ?= -std=$(CPPSTD) -O2 --extended-lambda -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
MPICXX ?= mpicxx
MPIEXEC ?= mpiexec
MPIEXEC_FLAGS ?=
MPI_NP ?= 2
MPI_CXXFLAGS ?=
MPI_LDFLAGS ?=

INCLUDES := -Iinclude -I$(NMFD_LINSOLVERS_ROOT)/include -I$(NMFD_OPERATIONS_ROOT)/include -Icontrib/json -isystem $(SCFD_ROOT)/include
CXXFLAGS := -std=$(CPPSTD) $(TARGET_CXX) $(WARNING_FLAGS) $(SANITIZER_FLAGS) $(INCLUDES)
CORE_HEADERS := $(shell find include/nmfd/backend include/nmfd/detail include/nmfd/solvers -name '*.h')
CPU_HEADERS = $(filter-out include/nmfd/backend/cuda.h include/nmfd/backend/hip.h,$(CORE_HEADERS)) $(LINSOLVER_HEADERS)
OPERATIONS_HEADERS := $(shell find $(NMFD_OPERATIONS_ROOT)/include -type f \( -name '*.h' -o -name '*.cuh' \) 2>/dev/null)
LINSOLVER_HEADERS := $(shell find $(NMFD_LINSOLVERS_ROOT)/include -type f -name '*.h' 2>/dev/null)
TEST_HEADERS := $(CORE_HEADERS) $(LINSOLVER_HEADERS) $(OPERATIONS_HEADERS) $(wildcard test/solvers/*.h)

REGRESSIONS := failure_reporting backtracking convergence vector_ownership configuration operator_binding dependency_ownership linsolvers
REGRESSION_BINS := $(addprefix $(BUILD_DIR)/test_newton_,$(addsuffix .bin,$(REGRESSIONS)))
JSON_BINS := $(BUILD_DIR)/test_newton_backtracking_json.bin $(BUILD_DIR)/test_newton_configuration_json.bin $(BUILD_DIR)/test_newton_linsolvers_json.bin
DENSE_TESTS := gmres nonlinear_solver direct_solver newton_direct_solver
DENSE_CPU_BINS := $(foreach precision,float double,$(addprefix $(BUILD_DIR)/$(precision)/test_,$(addsuffix _dense_cpu.bin,$(DENSE_TESTS))))
DENSE_CUDA_BINS := $(foreach precision,float double,$(addprefix $(BUILD_DIR)/$(precision)/test_,$(addsuffix _dense_cuda.bin,$(DENSE_TESTS))))
OPERATIONS_TESTS := static_vector_space pair_vector_space dense1_extended_operator
OPERATIONS_BINS := $(addprefix $(BUILD_DIR)/test_,$(addsuffix .bin,$(OPERATIONS_TESTS)))
SOLVER_TESTS := gmres gmres_mg nonlinear_solver dense1_extended_solver
SOLVER_BINS := $(addprefix $(BUILD_DIR)/test_,$(addsuffix .bin,$(SOLVER_TESTS)))
CH_SOURCE := test/cahn_hilliard/test_solver_compatibility.cpp
CH_HEADERS := $(wildcard test/cahn_hilliard/*.h test/cahn_hilliard/include/*.h test/cahn_hilliard/kernels/*.h)
CH_CPU_BINS := $(BUILD_DIR)/test_ch_serial.bin $(BUILD_DIR)/test_ch_omp.bin
CH_ORIGINAL_PROGRAMS := test_biharmonic test_cahn_hilliard test_time_cahn_hilliard test_cahn_hilliard_nonlinear test_random_init_cahn_hilliard
CH_ORIGINAL_BINS := $(addprefix $(BUILD_DIR)/original/,$(addsuffix .bin,$(CH_ORIGINAL_PROGRAMS)))
CPU_BINS := $(REGRESSION_BINS) $(JSON_BINS) $(OPERATIONS_BINS) $(SOLVER_BINS) $(DENSE_CPU_BINS) $(BUILD_DIR)/test_scfd_backend_cpu.bin $(BUILD_DIR)/test_algo_hierarchy_macro.bin $(BUILD_DIR)/test_newton_linsolvers_reverse.bin $(CH_CPU_BINS)

.PHONY: all check_dependencies tests_cpu test_cpu check_headers_cpu check_headers_cuda check_regressions_cpu check_operations_cpu check_solvers_cpu check_dense_cpu check_hierarchy_cpu check_cuda check_backtracking_json check_configuration_json check_linsolvers_json check_ch_serial check_ch_omp check_ch_cuda $(addprefix check_, $(REGRESSIONS))
all: tests_cpu

check_dependencies:
	@test -f $(NMFD_LINSOLVERS_ROOT)/include/nmfd/solvers/gmres.h || \
		{ echo 'Initialize recursive submodules or set NMFD_LINSOLVERS_ROOT'; exit 1; }
	@test -f $(NMFD_LINSOLVERS_ROOT)/include/nmfd/solvers/jacobi.h || \
		{ echo 'NMFD-linsolvers must include migrated Jacobi; update the dependency or set NMFD_LINSOLVERS_ROOT'; exit 1; }
	@test -f $(NMFD_LINSOLVERS_ROOT)/include/nmfd/solvers/dense1_extended_solver.h || \
		{ echo 'NMFD-linsolvers must include the migrated extended solver; update the dependency or set NMFD_LINSOLVERS_ROOT'; exit 1; }
	@test -f $(NMFD_OPERATIONS_ROOT)/include/nmfd/detail/vector_wrap.h || \
		{ echo 'Initialize recursive submodules or set NMFD_OPERATIONS_ROOT'; exit 1; }
	@test -f $(SCFD_ROOT)/include/scfd/utils/log.h || \
		{ echo 'SCFD dependency is not initialized'; exit 1; }
	@test -z "$(shell find include/nmfd/operations -type f 2>/dev/null)" && test ! -f include/nmfd/detail/vector_wrap.h || \
		{ echo 'Duplicated operations headers must not shadow NMFD-operations'; exit 1; }
	@set -e; for header in $(LINSOLVER_HEADERS); do \
		test ! -f "include/$${header#$(NMFD_LINSOLVERS_ROOT)/include/}" || \
			{ echo "Duplicated linear-solver dependency: $$header"; exit 1; }; \
	done

$(BUILD_DIR)/test_newton_%_json.bin: test/solvers/test_newton_%.cpp $(TEST_HEADERS) contrib/json/nlohmann/json.hpp | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -DNMFD_ENABLE_NLOHMANN $< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/test_newton_%.bin: test/solvers/test_newton_%.cpp $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) -o $@

$(OPERATIONS_BINS): $(BUILD_DIR)/%.bin: test/operations/%.cpp $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) -o $@

$(SOLVER_BINS): $(BUILD_DIR)/%.bin: test/solvers/%.cpp test/solvers/cpu_vector_space.h $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/test_algo_hierarchy_macro.bin: test/detail/test_algo_hierarchy_macro.cpp $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/test_newton_linsolvers_reverse.bin: test/solvers/test_newton_linsolvers.cpp $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) -I$(NMFD_LINSOLVERS_ROOT)/include $(CXXFLAGS) $< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/test_ch_serial.bin: $(CH_SOURCE) $(CH_HEADERS) $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -DPLATFORM_SERIAL_CPU $< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/test_ch_omp.bin: $(CH_SOURCE) $(CH_HEADERS) $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -DPLATFORM_OMP $< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/test_ch_cuda.bin: $(CH_SOURCE) $(CH_HEADERS) $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DPLATFORM_CUDA -x cu $< -o $@

$(BUILD_DIR)/test_scfd_backend_cpu.bin: test/operations/test_scfd_backend.cpp $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -DPLATFORM_SERIAL_CPU $< $(OPENMP_FLAGS) -o $@

.SECONDEXPANSION:
$(DENSE_CPU_BINS): $$(addprefix test/solvers/,$$(notdir $$(subst _cpu.bin,.cpp,$$@))) $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -DPLATFORM_SERIAL_CPU $(if $(findstring /double/,$@),-DUSE_DOUBLE_PRECISION) \
		$< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/operations_cuda_reductions.o: $(NMFD_OPERATIONS_ROOT)/include/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_kernels.cu $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/dense_solvers_cuda_inst.o: test/solvers/dense_solvers_cuda_inst.cu | check_dependencies
	mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test_scfd_backend_cuda.bin: test/operations/test_scfd_backend.cpp $(TEST_HEADERS) $(BUILD_DIR)/operations_cuda_reductions.o $(BUILD_DIR)/dense_solvers_cuda_inst.o | check_dependencies
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DPLATFORM_CUDA -x cu -c $< -o $(@:.bin=.o)
	$(NVCC) $(NVCCFLAGS) $(@:.bin=.o) $(BUILD_DIR)/operations_cuda_reductions.o $(BUILD_DIR)/dense_solvers_cuda_inst.o -lcublas -lcusolver -o $@

$(DENSE_CUDA_BINS): $$(addprefix test/solvers/,$$(notdir $$(subst _cuda.bin,.cpp,$$@))) $(TEST_HEADERS) $(BUILD_DIR)/operations_cuda_reductions.o $(BUILD_DIR)/dense_solvers_cuda_inst.o | check_dependencies
	mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DPLATFORM_CUDA $(if $(findstring /double/,$@),-DUSE_DOUBLE_PRECISION) \
		-x cu -c $< -o $(@:.bin=.o)
	$(NVCC) $(NVCCFLAGS) $(@:.bin=.o) $(BUILD_DIR)/operations_cuda_reductions.o $(BUILD_DIR)/dense_solvers_cuda_inst.o -lcublas -lcusolver -o $@

check_headers_cpu: check_dependencies
	@set -e; for json in '' '-DNMFD_ENABLE_NLOHMANN'; do \
		for header in $(CPU_HEADERS); do \
			echo "Checking $$header $$json"; \
			include_name=$${header#include/}; include_name=$${include_name#$(NMFD_LINSOLVERS_ROOT)/include/}; \
			printf '#include <%s>\nint main() {}\n' "$$include_name" | \
				$(HOST_CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -DPLATFORM_SERIAL_CPU $$json -x c++ -fsyntax-only -; \
		done; \
	done

check_headers_cuda: check_dependencies
	@set -e; for header in $(filter-out include/nmfd/backend/hip.h,$(CORE_HEADERS)) $(LINSOLVER_HEADERS); do \
		echo "Checking $$header (CUDA)"; \
		include_name=$${header#include/}; include_name=$${include_name#$(NMFD_LINSOLVERS_ROOT)/include/}; \
		printf '#include <%s>\nint main() {}\n' "$$include_name" | \
			$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DPLATFORM_CUDA -x cu -c -o /dev/null -; \
	done

$(addprefix check_,$(filter-out linsolvers,$(REGRESSIONS))): check_%: $(BUILD_DIR)/test_newton_%.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $<

check_backtracking: check_backtracking_json
check_backtracking_json: $(BUILD_DIR)/test_newton_backtracking_json.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(BUILD_DIR)/test_newton_backtracking_json.bin

check_configuration: check_configuration_json
check_configuration_json: $(BUILD_DIR)/test_newton_configuration_json.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(BUILD_DIR)/test_newton_configuration_json.bin

check_linsolvers: $(BUILD_DIR)/test_newton_linsolvers.bin check_linsolvers_json $(BUILD_DIR)/test_newton_linsolvers_reverse.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(BUILD_DIR)/test_newton_linsolvers.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(BUILD_DIR)/test_newton_linsolvers_reverse.bin

check_linsolvers_json: $(BUILD_DIR)/test_newton_linsolvers_json.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $(BUILD_DIR)/test_newton_linsolvers_json.bin

check_ch_serial: $(BUILD_DIR)/test_ch_serial.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $<

check_ch_omp: $(BUILD_DIR)/test_ch_omp.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $<

check_ch_cuda: $(BUILD_DIR)/test_ch_cuda.bin
	$<

.PHONY: check_ch_original check_ch_mpi
$(CH_ORIGINAL_BINS): $(BUILD_DIR)/original/%.bin: test/cahn_hilliard/%.cpp $(CH_HEADERS) $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -DSCFD_ARRAYS_ENABLE_INDEX_SHIFT -DPLATFORM_SERIAL_CPU -DUSE_DOUBLE_PRECISION \
		$< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/original/test_rect.bin: test/operations/test_rect_vector_space.cpp $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) $< $(OPENMP_FLAGS) -o $@

$(BUILD_DIR)/original/test_rect_mpi.bin: test/operations/test_rect_vector_space_mpi.cpp $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(MPICXX) $(CXXFLAGS) $(MPI_CXXFLAGS) $< $(OPENMP_FLAGS) $(MPI_LDFLAGS) -o $@

$(BUILD_DIR)/original/test_biharmonic_mpi.bin: test/cahn_hilliard/test_biharmonic_mpi.cpp $(CH_HEADERS) $(TEST_HEADERS) | check_dependencies
	mkdir -p $(@D)
	$(MPICXX) $(CXXFLAGS) $(MPI_CXXFLAGS) -DSCFD_ARRAYS_ENABLE_INDEX_SHIFT -DPLATFORM_SERIAL_CPU -DUSE_DOUBLE_PRECISION \
		$< $(OPENMP_FLAGS) $(MPI_LDFLAGS) -o $@

check_ch_original: $(CH_ORIGINAL_BINS) $(BUILD_DIR)/original/test_rect.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) python3 test/cahn_hilliard/check_original_programs.py --build-dir $(BUILD_DIR)/original

check_ch_mpi: $(BUILD_DIR)/original/test_rect_mpi.bin $(BUILD_DIR)/original/test_biharmonic_mpi.bin
	OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) python3 test/cahn_hilliard/check_original_programs.py --build-dir $(BUILD_DIR)/original \
		--mpi-exec "$(MPIEXEC) $(MPIEXEC_FLAGS)" --ranks $(MPI_NP)

check_regressions_cpu: $(addprefix check_,$(REGRESSIONS))

check_operations_cpu: $(OPERATIONS_BINS) $(BUILD_DIR)/test_scfd_backend_cpu.bin
	@set -e; for binary in $^; do OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $$binary; done

check_dense_cpu: $(DENSE_CPU_BINS)
	@set -e; for binary in $^; do OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $$binary; done

check_solvers_cpu: $(SOLVER_BINS)
	@set -e; for binary in $^; do OMP_NUM_THREADS=$(TEST_OMP_NUM_THREADS) $$binary; done

check_hierarchy_cpu: $(BUILD_DIR)/test_algo_hierarchy_macro.bin
	$<

tests_cpu: $(CPU_BINS)
test_cpu: check_regressions_cpu check_operations_cpu check_solvers_cpu check_dense_cpu check_hierarchy_cpu check_ch_serial check_ch_omp

check_cuda: $(BUILD_DIR)/test_scfd_backend_cuda.bin $(DENSE_CUDA_BINS) $(BUILD_DIR)/test_ch_cuda.bin
	@set -e; for binary in $^; do $$binary; done
