-include ../common.mk

EXTRA_INCLUDE = -I./headers

# ----- Poisson Jacobi -----

test_poisson_jacobi_cpu: test_poisson_jacobi.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRA_INCLUDE) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_poisson_jacobi.cpp -o test_poisson_jacobi_cpu_$(PRECISION_SUFFIX).bin

test_poisson_jacobi_omp: test_poisson_jacobi.cpp
	$(HOSTCOMPILER) $(OMP_FLAGS) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRA_INCLUDE) -DPLATFORM_OMP $(PRECISION_DEFINE) test_poisson_jacobi.cpp -o test_poisson_jacobi_omp_$(PRECISION_SUFFIX).bin

test_poisson_jacobi_cuda: test_poisson_jacobi.cpp
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) $(EXTRA_INCLUDE) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) test_poisson_jacobi.cpp -o test_poisson_jacobi_cuda_$(PRECISION_SUFFIX).bin


test_poisson_jacobi_all: test_poisson_jacobi_cpu test_poisson_jacobi_omp test_poisson_jacobi_cuda

# ----- Poissson GMRES -----

test_poisson_gmres_cpu: test_poisson_gmres.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRA_INCLUDE) -DPLATFORM_SERIAL_CPU $(PRECISION_DEFINE) test_poisson_gmres.cpp -o test_poisson_gmres_cpu_$(PRECISION_SUFFIX).bin

test_poisson_gmres_omp: test_poisson_gmres.cpp
	$(HOSTCOMPILER) $(OMP_FLAGS) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRA_INCLUDE) -DPLATFORM_OMP $(PRECISION_DEFINE) test_poisson_gmres.cpp -o test_poisson_gmres_omp_$(PRECISION_SUFFIX).bin

test_poisson_gmres_cuda: test_poisson_gmres.cpp
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) $(EXTRA_INCLUDE) -DPLATFORM_CUDA -x cu $(PRECISION_DEFINE) test_poisson_gmres.cpp -o test_poisson_gmres_cuda_$(PRECISION_SUFFIX).bin


test_poisson_gmres_all: test_poisson_gmres_cpu test_poisson_gmres_omp test_poisson_gmres_cuda

