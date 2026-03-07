-include ../common.mk

all: test_static_vector_space test_pair_vector_space test_dense1_extended_operator

test_static_vector_space: test_static_vector_space.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) test_static_vector_space.cpp -o test_static_vector_space.bin

test_pair_vector_space: test_pair_vector_space.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) test_pair_vector_space.cpp -o test_pair_vector_space.bin

test_dense1_extended_operator: test_dense1_extended_operator.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) test_dense1_extended_operator.cpp -o test_dense1_extended_operator.bin

test_dense_vector_space: test_dense_vector_space.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) test_dense_vector_space.cpp -o test_dense_vector_space.bin

test_dense_vector_space_static: test_dense_vector_space_static.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) test_dense_vector_space_static.cpp -o test_dense_vector_space_static.bin

test_dense_operations: test_dense_operations.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_CONTRIB) test_dense_operations.cpp -o test_dense_operations.bin
