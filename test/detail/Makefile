-include ../common.mk

all: test_algo_hierarchy_macro.bin

test: 
	./test_algo_hierarchy_macro.bin

test_algo_hierarchy_macro.bin: test_algo_hierarchy_macro.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_algo_hierarchy_macro.cpp -o test_algo_hierarchy_macro.bin
