-include ../common.mk

# ----- Biharmonic -----

test_biharmonic: test_biharmonic.cpp
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_MPI_INCLUDE) $(PLATFORM_MPI_DEFINE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) -c test_biharmonic.cpp -o bin/test_biharmonic$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) bin/test_biharmonic$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o bin/test_biharmonic$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

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
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_MPI_INCLUDE) $(PLATFORM_MPI_DEFINE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) -c test_cahn_hilliard.cpp -o bin/test_cahn_hilliard$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) bin/test_cahn_hilliard$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o bin/test_cahn_hilliard$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_cahn_hilliard_all:
	$(MAKE) test_cahn_hilliard PLATFORM=cpu
	$(MAKE) test_cahn_hilliard PLATFORM=omp
	$(MAKE) test_cahn_hilliard PLATFORM=cuda

test_cahn_hilliard_mpi:
	$(MAKE) test_cahn_hilliard PLATFORM_MPI=1

test_cahn_hilliard_mpi_all:
	$(MAKE) test_cahn_hilliard_mpi PLATFORM=cpu
	$(MAKE) test_cahn_hilliard_mpi PLATFORM=omp
	$(MAKE) test_cahn_hilliard_mpi PLATFORM=cuda


# ----- Time depentent Cahn-Hilliard -----

test_time_cahn_hilliard: test_time_cahn_hilliard.cpp
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_MPI_INCLUDE) $(PLATFORM_MPI_DEFINE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) -c test_time_cahn_hilliard.cpp -o bin/test_time_cahn_hilliard$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) bin/test_time_cahn_hilliard$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o bin/test_time_cahn_hilliard$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_time_cahn_hilliard_all:
	$(MAKE) test_time_cahn_hilliard PLATFORM=cpu
	$(MAKE) test_time_cahn_hilliard PLATFORM=omp
	$(MAKE) test_time_cahn_hilliard PLATFORM=cuda

test_time_cahn_hilliard_mpi:
	$(MAKE) test_time_cahn_hilliard PLATFORM_MPI=1

test_time_cahn_hilliard_mpi_all:
	$(MAKE) test_time_cahn_hilliard_mpi PLATFORM=cpu
	$(MAKE) test_time_cahn_hilliard_mpi PLATFORM=omp
	$(MAKE) test_time_cahn_hilliard_mpi PLATFORM=cuda


# ----- Random init time depentent Cahn-Hilliard -----

test_random_init_cahn_hilliard: test_random_init_cahn_hilliard.cpp
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_MPI_INCLUDE) $(PLATFORM_MPI_DEFINE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) -c test_random_init_cahn_hilliard.cpp -o bin/test_random_init_cahn_hilliard$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) bin/test_random_init_cahn_hilliard$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o bin/test_random_init_cahn_hilliard$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_random_init_cahn_hilliard_all:
	$(MAKE) test_random_init_cahn_hilliard PLATFORM=cpu
	$(MAKE) test_random_init_cahn_hilliard PLATFORM=omp
	$(MAKE) test_random_init_cahn_hilliard PLATFORM=cuda

test_random_init_cahn_hilliard_mpi:
	$(MAKE) test_random_init_cahn_hilliard PLATFORM_MPI=1

test_random_init_cahn_hilliard_mpi_all:
	$(MAKE) test_random_init_cahn_hilliard_mpi PLATFORM=cpu
	$(MAKE) test_random_init_cahn_hilliard_mpi PLATFORM=omp
	$(MAKE) test_random_init_cahn_hilliard_mpi PLATFORM=cuda


# ----- Cahn-Hilliard with nonlinear boundary conditions -----

test_cahn_hilliard_nonlinear: test_cahn_hilliard_nonlinear.cpp
	$(PLATFORMCXX) $(PLATFORMCXX_FLAGS) $(INCLUDE_CONTRIB) $(PLATFORM_MPI_INCLUDE) $(PLATFORM_MPI_DEFINE) $(PLATFORM_DEFINE) $(PRECISION_DEFINE) $(PROFILE_DEFINE) -c test_cahn_hilliard_nonlinear.cpp -o bin/test_cahn_hilliard_nonlinear$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o
	$(PLATFORMLINKER) bin/test_cahn_hilliard_nonlinear$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).o $(PLATFORMLINKER_FLAGS) -o bin/test_cahn_hilliard_nonlinear$(MPI_SUFFIX)$(PLATFORM_SUFFIX)_$(PRECISION_SUFFIX).bin

test_cahn_hilliard_nonlinear_all:
	$(MAKE) test_cahn_hilliard_nonlinear PLATFORM=cpu
	$(MAKE) test_cahn_hilliard_nonlinear PLATFORM=omp
	$(MAKE) test_cahn_hilliard_nonlinear PLATFORM=cuda

test_cahn_hilliard_nonlinear_mpi:
	$(MAKE) test_cahn_hilliard_nonlinear PLATFORM_MPI=1

test_cahn_hilliard_nonlinear_mpi_all:
	$(MAKE) test_cahn_hilliard_nonlinear_mpi PLATFORM=cpu
	$(MAKE) test_cahn_hilliard_nonlinear_mpi PLATFORM=omp
	$(MAKE) test_cahn_hilliard_nonlinear_mpi PLATFORM=cuda
