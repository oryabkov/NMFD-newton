-include ../common.mk



# ----- Biharmonic -----

test_biharmonic: test_biharmonic.cpp
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_MPI_INCLUDE) $(PLATFORM_MPI_DEFINE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) -c test_biharmonic.cpp -o test_biharmonic$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) test_biharmonic$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o test_biharmonic$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_biharmonic_all:
	$(MAKE) test_biharmonic PLATFORM=cpu
	$(MAKE) test_biharmonic PLATFORM=omp
	$(MAKE) test_biharmonic PLATFORM=cuda

test_biharmonic_mpi:
	$(MAKE) test_biharmonic PLATFORM_MPI=1

test_biharmonic_mpi_all:
	$(MAKE) test_biharmonic_mpi PLATFORM=cpu
	$(MAKE) test_biharmonic_mpi PLATFORM=omp
	$(MAKE) test_biharmonic_mpi PLATFORM=cuda

# ----- Cahn-Hilliard -----

test_cahn_hilliard: test_cahn_hilliard.cpp
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_MPI_INCLUDE) $(PLATFORM_MPI_DEFINE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) -c test_cahn_hilliard.cpp -o test_cahn_hilliard$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) test_cahn_hilliard$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o test_cahn_hilliard$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_cahn_hilliard_all:
	$(MAKE) test_cahn_hilliard PLATFORM=cpu
	$(MAKE) test_cahn_hilliard PLATFORM=omp
	$(MAKE) test_cahn_hilliard PLATFORM=cuda


# ----- Time depentent Cahn-Hilliard -----

test_time_cahn_hilliard: test_time_cahn_hilliard.cpp
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_MPI_INCLUDE) $(PLATFORM_MPI_DEFINE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) -c test_time_cahn_hilliard.cpp -o test_time_cahn_hilliard$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) test_time_cahn_hilliard$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o test_time_cahn_hilliard$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_time_cahn_hilliard_all:
	$(MAKE) test_time_cahn_hilliard PLATFORM=cpu
	$(MAKE) test_time_cahn_hilliard PLATFORM=omp
	$(MAKE) test_time_cahn_hilliard PLATFORM=cuda


# ----- Random init time depentent Cahn-Hilliard -----

test_random_init_cahn_hilliard: test_random_init_cahn_hilliard.cpp
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_MPI_INCLUDE) $(PLATFORM_MPI_DEFINE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) -c test_random_init_cahn_hilliard.cpp -o test_random_init_cahn_hilliard$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) test_random_init_cahn_hilliard$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o test_random_init_cahn_hilliard$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin


# ----- Cahn-Hilliard with nonlinear boundary conditions -----

test_cahn_hilliard_nonlinear: test_cahn_hilliard_nonlinear.cpp
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_MPI_INCLUDE) $(PLATFORM_MPI_DEFINE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) -c test_cahn_hilliard_nonlinear.cpp -o test_cahn_hilliard_nonlinear$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) test_cahn_hilliard_nonlinear$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o test_cahn_hilliard_nonlinear$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_cahn_hilliard_nonlinear_all:
	$(MAKE) test_cahn_hilliard_nonlinear PLATFORM=cpu
	$(MAKE) test_cahn_hilliard_nonlinear PLATFORM=omp
	$(MAKE) test_cahn_hilliard_nonlinear PLATFORM=cuda
