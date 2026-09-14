-include ../common.mk

EXTRA_FLAGS = -DSCFD_ARRAYS_ENABLE_INDEX_SHIFT=1

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

all: test_biharmonic test_cahn_hilliard test_time_cahn_hilliard test_random_init_cahn_hilliard test_cahn_hilliard_nonlinear

# ----- Biharmonic -----

test_biharmonic: test_biharmonic.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_biharmonic.cpp -o $(BUILD_DIR)/test_biharmonic_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_biharmonic_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_biharmonic_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_biharmonic_mpi:
	$(MAKE) test_biharmonic PLATFORM_MPI=1

# ----- Cahn-Hilliard -----

test_cahn_hilliard: test_cahn_hilliard.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_cahn_hilliard.cpp -o $(BUILD_DIR)/test_cahn_hilliard_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_cahn_hilliard_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_cahn_hilliard_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_cahn_hilliard_mpi:
	$(MAKE) test_cahn_hilliard PLATFORM_MPI=1


# ----- Time depentent Cahn-Hilliard -----

test_time_cahn_hilliard: test_time_cahn_hilliard.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_time_cahn_hilliard.cpp -o $(BUILD_DIR)/test_time_cahn_hilliard_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_time_cahn_hilliard_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_time_cahn_hilliard_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_time_cahn_hilliard_mpi:
	$(MAKE) test_time_cahn_hilliard PLATFORM_MPI=1


# ----- Random init time depentent Cahn-Hilliard -----

test_random_init_cahn_hilliard: test_random_init_cahn_hilliard.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_random_init_cahn_hilliard.cpp -o $(BUILD_DIR)/test_random_init_cahn_hilliard_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_random_init_cahn_hilliard_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_random_init_cahn_hilliard_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_random_init_cahn_hilliard_mpi:
	$(MAKE) test_random_init_cahn_hilliard PLATFORM_MPI=1


# ----- Cahn-Hilliard with nonlinear boundary conditions -----

test_cahn_hilliard_nonlinear: test_cahn_hilliard_nonlinear.cpp | $(BUILD_DIR)
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_INCLUDE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) $(EXTRA_FLAGS) -c test_cahn_hilliard_nonlinear.cpp -o $(BUILD_DIR)/test_cahn_hilliard_nonlinear_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) $(BUILD_DIR)/test_cahn_hilliard_nonlinear_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o $(BUILD_DIR)/test_cahn_hilliard_nonlinear_$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_cahn_hilliard_nonlinear_mpi:
	$(MAKE) test_cahn_hilliard_nonlinear PLATFORM_MPI=1
