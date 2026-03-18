-include ../common.mk

EXTRAFLAGS = -DSCFD_ARRAYS_ENABLE_INDEX_SHIFT=1


# ----- Biharmonic -----

test_biharmonic_cpu: test_biharmonic.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_biharmonic.cpp -o test_biharmonic_cpu_$(PRECISION_SUFFIX).bin

test_biharmonic_omp: test_biharmonic.cppw
	$(HOSTCOMPILER) $(OMP_FLAGS) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_OMP $(PRECISION_DEFINE) test_biharmonic.cpp -o test_biharmonic_omp_$(PRECISION_SUFFIX).bin

test_biharmonic_cuda: test_biharmonic.cpp
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) test_biharmonic.cpp -o test_biharmonic_cuda_$(PRECISION_SUFFIX).bin


test_biharmonic_all: test_biharmonic_cpu test_biharmonic_omp test_biharmonic_cuda


# ----- Cahn-Hilliard -----

test_cahn_hilliard_cpu: test_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_cahn_hilliard.cpp -o test_cahn_hilliard_cpu_$(PRECISION_SUFFIX).bin

test_cahn_hilliard_omp: test_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(OMP_FLAGS) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_OMP $(PRECISION_DEFINE) test_cahn_hilliard.cpp -o test_cahn_hilliard_omp_$(PRECISION_SUFFIX).bin

test_cahn_hilliard_cuda: test_cahn_hilliard.cpp
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) test_cahn_hilliard.cpp -o test_cahn_hilliard_cuda_$(PRECISION_SUFFIX).bin


test_cahn_hilliard_all: test_cahn_hilliard_cpu test_cahn_hilliard_omp test_cahn_hilliard_cuda


# ----- Time depentent Cahn-Hilliard -----

test_time_cahn_hilliard_cpu: test_time_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_time_cahn_hilliard.cpp -o test_time_cahn_hilliard_cpu_$(PRECISION_SUFFIX).bin

test_time_cahn_hilliard_omp: test_time_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(OMP_FLAGS) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_OMP $(PRECISION_DEFINE) test_time_cahn_hilliard.cpp -o test_time_cahn_hilliard_omp_$(PRECISION_SUFFIX).bin

test_time_cahn_hilliard_cuda: test_time_cahn_hilliard.cpp
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) test_time_cahn_hilliard.cpp -o test_time_cahn_hilliard_cuda_$(PRECISION_SUFFIX).bin


test_time_cahn_hilliard_all: test_time_cahn_hilliard_cpu test_time_cahn_hilliard_omp test_time_cahn_hilliard_cuda


# ----- Random init time depentent Cahn-Hilliard -----

test_random_init_cahh_hilliard_cpu: test_random_init_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_random_init_cahn_hilliard.cpp -o test_random_init_cahn_hilliard_cpu_$(PRECISION_SUFFIX).bin

test_random_init_cahn_hilliard_cuda: test_random_init_cahn_hilliard.cpp
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) test_random_init_cahn_hilliard.cpp -o test_random_init_cahn_hilliard_cuda_$(PRECISION_SUFFIX).bin


# ----- Cahn-Hilliard with nonlinear boundary conditions -----

test_cahn_hilliard_nonlinear_cpu: test_cahn_hilliard_nonlinear.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_cahn_hilliard_nonlinear.cpp -o test_cahn_hilliard_nonlinear_cpu_$(PRECISION_SUFFIX).bin

test_cahn_hilliard_nonlinear_omp: test_cahn_hilliard_nonlinear.cpp
	$(HOSTCOMPILER) $(OMP_FLAGS) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_OMP $(PRECISION_DEFINE) test_cahn_hilliard_nonlinear.cpp -o test_cahn_hilliard_nonlinear_omp_$(PRECISION_SUFFIX).bin

test_cahn_hilliard_nonlinear_cuda: test_cahn_hilliard_nonlinear.cpp
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) test_cahn_hilliard_nonlinear.cpp -o test_cahn_hilliard_nonlinear_cuda_$(PRECISION_SUFFIX).bin


test_cahn_hilliard_nonlinear_all: test_cahn_hilliard_nonlinear_cpu test_cahn_hilliard_nonlinear_omp test_cahn_hilliard_nonlinear_cuda
