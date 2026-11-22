-include ../common.mk

.DEFAULT_GOAL := all

.PHONY: all test test_gmres test_gmres_mg test_nonlinear_solver test_dense1_extended_solver test_nonlinear_solver_circle test_gmres_dense test_nonlinear_solver_dense test_direct_solver_dense test_newton_direct_solver_dense

all: test_gmres test_gmres_mg test_nonlinear_solver test_dense1_extended_solver test_nonlinear_solver_circle test_gmres_dense test_nonlinear_solver_dense test_direct_solver_dense test_newton_direct_solver_dense

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test:
	$(BUILD_DIR)/test_gmres_$(PRECISION_SUFFIX).bin
	$(BUILD_DIR)/test_gmres_mg_$(PRECISION_SUFFIX).bin
	$(BUILD_DIR)/test_nonlinear_solver_$(PRECISION_SUFFIX).bin
	$(BUILD_DIR)/test_dense1_extended_solver_$(PRECISION_SUFFIX).bin
	$(BUILD_DIR)/test_nonlinear_solver_circle_$(PRECISION_SUFFIX).bin

test_gmres: test_gmres.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_gmres.cpp -o $(BUILD_DIR)/test_gmres_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_gmres_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_gmres_$(PRECISION_SUFFIX).bin

test_gmres_mg: test_gmres_mg.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_gmres_mg.cpp -o $(BUILD_DIR)/test_gmres_mg_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_gmres_mg_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_gmres_mg_$(PRECISION_SUFFIX).bin

test_nonlinear_solver: test_nonlinear_solver.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_nonlinear_solver.cpp -o $(BUILD_DIR)/test_nonlinear_solver_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_nonlinear_solver_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_nonlinear_solver_$(PRECISION_SUFFIX).bin

test_dense1_extended_solver: test_dense1_extended_solver.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_dense1_extended_solver.cpp -o $(BUILD_DIR)/test_dense1_extended_solver_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_dense1_extended_solver_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_dense1_extended_solver_$(PRECISION_SUFFIX).bin

test_nonlinear_solver_circle: test_nonlinear_solver_circle.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_nonlinear_solver_circle.cpp -o $(BUILD_DIR)/test_nonlinear_solver_circle_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_nonlinear_solver_circle_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_nonlinear_solver_circle_$(PRECISION_SUFFIX).bin

test_gmres_dense: test_gmres_dense.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_gmres_dense.cpp -o $(BUILD_DIR)/test_gmres_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_gmres_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_gmres_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_nonlinear_solver_dense: test_nonlinear_solver_dense.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_nonlinear_solver_dense.cpp -o $(BUILD_DIR)/test_nonlinear_solver_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_nonlinear_solver_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_nonlinear_solver_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_direct_solver_dense: test_direct_solver_dense.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_direct_solver_dense.cpp -o $(BUILD_DIR)/test_direct_solver_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_direct_solver_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_direct_solver_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_newton_direct_solver_dense: test_newton_direct_solver_dense.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_newton_direct_solver_dense.cpp -o $(BUILD_DIR)/test_newton_direct_solver_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_newton_direct_solver_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_newton_direct_solver_dense_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin
