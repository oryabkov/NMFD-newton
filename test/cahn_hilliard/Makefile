-include ../common.mk

EXTRAFLAGS = -DSCFD_ARRAYS_ENABLE_INDEX_SHIFT=1

all: test_biharmonic

# ----- Biharmonic -----

# Single precision (float)
test_biharmonic_cpu_f: test_biharmonic.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_SERIAL_CPU test_biharmonic.cpp -o test_biharmonic_cpu_f.bin

test_biharmonic_omp_f: test_biharmonic.cpp
	$(HOSTCOMPILER) $(OMP_FLAGS) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_OMP test_biharmonic.cpp -o test_biharmonic_omp_f.bin

test_biharmonic_cuda_f: test_biharmonic.cpp
	nvcc $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu test_biharmonic.cpp -o test_biharmonic_cuda_f.bin

# Double precision (double)
test_biharmonic_cpu_d: test_biharmonic.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_SERIAL_CPU -DUSE_DOUBLE_PRECISION test_biharmonic.cpp -o test_biharmonic_cpu_d.bin

test_biharmonic_omp_d: test_biharmonic.cpp
	$(HOSTCOMPILER) $(OMP_FLAGS) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DUSE_DOUBLE_PRECISION -DPLATFORM_OMP test_biharmonic.cpp -o test_biharmonic_omp_d.bin

test_biharmonic_cuda_d: test_biharmonic.cpp
	nvcc $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu -DUSE_DOUBLE_PRECISION test_biharmonic.cpp -o test_biharmonic_cuda_d.bin


# ----- Cahn-Hilliard -----

test_cahn_hilliard_cpu: test_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_SERIAL_CPU test_cahn_hilliard.cpp -o test_cahn_hilliard_cpu.bin

test_cahn_hilliard_omp: test_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(OMP_FLAGS) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_OMP test_cahn_hilliard.cpp -o test_cahn_hilliard_omp.bin

test_cahn_hilliard_cuda: test_cahn_hilliard.cpp
	nvcc $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu test_cahn_hilliard.cpp -o test_cahn_hilliard_cuda.bin

# ----- Time depentent Cahn-Hilliard -----

test_time_cahn_hilliard_cpu: test_time_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_SERIAL_CPU test_time_cahn_hilliard.cpp -o test_time_cahn_hilliard_cpu.bin

test_time_cahn_hilliard_omp: test_time_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(OMP_FLAGS) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_OMP test_time_cahn_hilliard.cpp -o test_time_cahn_hilliard_omp.bin

test_time_cahn_hilliard_cuda: test_time_cahn_hilliard.cpp
	nvcc $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu test_time_cahn_hilliard.cpp -o test_time_cahn_hilliard_cuda.bin

# ----- Random init time depentent Cahn-Hilliard -----

test_random_init_hilliard_cpu: test_random_init_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_SERIAL_CPU test_random_init_cahn_hilliard.cpp -o test_random_init_cahn_hilliard_cpu.bin

test_random_init_hilliard_cuda: test_random_init_cahn_hilliard.cpp
	nvcc $(HOSTFLAGS) $(INCLUDE_CONTRIB) $(EXTRAFLAGS) -DPLATFORM_CUDA -x cu test_random_init_cahn_hilliard.cpp -o test_random_init_cahn_hilliard_cuda.bin
