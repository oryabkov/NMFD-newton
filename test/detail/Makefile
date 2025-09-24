ifndef CONFIG_FILE
CONFIG_FILE = config.inc
endif

ifeq (,$(wildcard $(CONFIG_FILE)))
$(info config file $(CONFIG_FILE) does not exist.)
$(error Create $(CONFIG_FILE) from example or specify another config via: make <target> CONFIG_FILE=<config_filename> )
endif

-include $(CONFIG_FILE)

PROJECT_ROOT_PATH = ../..
SCFD_INCLUDE = $(PROJECT_ROOT_PATH)/contrib/SCFD/include
JSON_INCLUDE = $(PROJECT_ROOT_PATH)/contrib/json
NMFD_INCLUDE = $(PROJECT_ROOT_PATH)/include
#INCLUDE_ROOT = -I$(PROJECT_ROOT_PATH)/sourse
#INCLUDE_LOCAL = -I$(PROJECT_ROOT_PATH)/sourse/solver
INCLUDE_CONTRIB = -I$(JSON_INCLUDE) -I$(SCFD_INCLUDE) -I$(NMFD_INCLUDE)
HOSTFLAGS = $(TARGET_GCC) -DNMFD_ENABLE_NLOHMANN=1 -std=c++17
HOSTCOMPILER = g++
#CUDAFLAGS = $(TARGET_NVCC)
#MPICOMPILER = $(MPI_ROOT_PATH)/bin/mpic++
#CUDA = $(CUDA_ROOT_PATH)#/opt/cuda#/opt/cuda_all/cuda_11.2
#SM = $(CUDA_ARCH)
#MPI = $(MPI_ROOT_PATH)
#CUDACOMPILER = ${CUDA}/bin/nvcc
#HYPRELIBRARY = -lHYPRE
#CUDALIBRARIES = -lcudart -lcurand -lcusparse -lcublas
IPROJECT = ${INCLUDE_CONTRIB}
LPROJECT = -ldl



test_algo_hierarchy_macro.bin: test_algo_hierarchy_macro.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_algo_hierarchy_macro.cpp -o test_algo_hierarchy_macro.bin
