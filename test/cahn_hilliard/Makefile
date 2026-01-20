-include ../common.mk

all: test_biharmonic

test_biharmonic: test_biharmonic.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU test_biharmonic.cpp -o test_biharmonic.bin

test_cahn_hilliard: test_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU test_cahn_hilliard.cpp -o test_cahn_hilliard.bin

test_cahn_hilliard_cpu: test_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU test_cahn_hilliard.cpp -o test_cahn_hilliard_cpu.bin

test_cahn_hilliard_omp: test_cahn_hilliard.cpp
	$(HOSTCOMPILER) $(OMP_FLAGS) $(HOSTFLAGS) $(INCLUDE_CONTRIB) -DPLATFORM_OMP test_cahn_hilliard.cpp -o test_cahn_hilliard_omp.bin $(OMP_LIBS)
