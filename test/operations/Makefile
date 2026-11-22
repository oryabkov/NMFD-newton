-include ../common.mk

.DEFAULT_GOAL := all

EXTRA_FLAGS = -DSCFD_ARRAYS_ENABLE_INDEX_SHIFT=1

.PHONY: all test_static_vector_space test_pair_vector_space test_dense1_extended_operator test_rect_vector_space test_dense_vector_space test_dense_vector_space_static test_dense_operations test_dense_operations_benchmarks

all: test_static_vector_space test_pair_vector_space test_dense1_extended_operator test_rect_vector_space test_dense_vector_space test_dense_vector_space_static test_dense_operations test_dense_operations_benchmarks

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test_static_vector_space: test_static_vector_space.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_static_vector_space.cpp -o $(BUILD_DIR)/test_static_vector_space_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_static_vector_space_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_static_vector_space_$(PRECISION_SUFFIX).bin

test_pair_vector_space: test_pair_vector_space.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_pair_vector_space.cpp -o $(BUILD_DIR)/test_pair_vector_space_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_pair_vector_space_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_pair_vector_space_$(PRECISION_SUFFIX).bin

test_dense1_extended_operator: test_dense1_extended_operator.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_dense1_extended_operator.cpp -o $(BUILD_DIR)/test_dense1_extended_operator_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_dense1_extended_operator_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_dense1_extended_operator_$(PRECISION_SUFFIX).bin

test_rect_vector_space: test_rect_vector_space.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_rect_vector_space.cpp -o $(BUILD_DIR)/test_rect_vector_space_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_rect_vector_space_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_rect_vector_space_$(PRECISION_SUFFIX).bin

test_dense_vector_space: test_dense_vector_space.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_dense_vector_space.cpp -o $(BUILD_DIR)/test_dense_vector_space_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_dense_vector_space_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_dense_vector_space_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_dense_vector_space_static: test_dense_vector_space_static.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_dense_vector_space_static.cpp -o $(BUILD_DIR)/test_dense_vector_space_static_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_dense_vector_space_static_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_dense_vector_space_static_$(PRECISION_SUFFIX).bin

test_dense_operations: test_dense_operations.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_dense_operations.cpp -o $(BUILD_DIR)/test_dense_operations_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_dense_operations_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_dense_operations_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_dense_operations_benchmarks: test_dense_operations_benchmarks.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_dense_operations_benchmarks.cpp -o $(BUILD_DIR)/test_dense_operations_benchmarks_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_dense_operations_benchmarks_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_dense_operations_benchmarks_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin
