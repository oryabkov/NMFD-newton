-include ../common.mk

all: test_gmres.bin test_gmres_dense.bin test_gmres_mg.bin test_nonlinear_solver.bin test_dense1_extended_solver.bin

test:
	./test_gmres.bin
	./test_gmres_dense.bin
	./test_gmres_mg.bin
	./test_nonlinear_solver.bin
	./test_dense1_extended_solver.bin

test_gmres.bin: test_gmres.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_gmres.cpp -o test_gmres.bin
test_gmres_dense.bin: test_gmres_dense.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) -DPLATFORM_SERIAL_CPU test_gmres_dense.cpp -o test_gmres_dense.bin
test_gmres_mg.bin: test_gmres_mg.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_gmres_mg.cpp -o test_gmres_mg.bin
test_nonlinear_solver.bin: test_nonlinear_solver.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_nonlinear_solver.cpp -o test_nonlinear_solver.bin
test_dense1_extended_solver.bin: test_dense1_extended_solver.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_dense1_extended_solver.cpp -o test_dense1_extended_solver.bin
