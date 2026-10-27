-include ../common.mk

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

EXTRA_INCLUDE = -I./headers

all: test_jacobi

test_jacobi: test_jacobi.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(EXTRA_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) -c test_jacobi.cpp -o $(BUILD_DIR)/test_jacobi_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_jacobi_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_jacobi_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_jacobi_mpi:
	$(MAKE) test_jacobi PLATFORM_MPI=1