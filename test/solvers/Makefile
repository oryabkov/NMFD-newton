-include ../common.mk

all: test_gmres.bin test_gmres_mg.bin test_nonlinear_solver.bin

test: 
	./test_gmres.bin 
	./test_gmres_mg.bin 
	./test_nonlinear_solver.bin

test_gmres.bin: test_gmres.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_gmres.cpp -o test_gmres.bin
test_gmres_mg.bin: test_gmres_mg.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_gmres_mg.cpp -o test_gmres_mg.bin
test_nonlinear_solver.bin: test_nonlinear_solver.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_nonlinear_solver.cpp -o test_nonlinear_solver.bin
