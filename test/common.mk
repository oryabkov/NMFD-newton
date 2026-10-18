ifndef CONFIG_FILE
CONFIG_FILE = config.inc
endif

ifeq (,$(wildcard $(CONFIG_FILE)))
$(info config file $(CONFIG_FILE) does not exist.)
$(error Create $(CONFIG_FILE) from example or specify another config via: make <target> CONFIG_FILE=<config_filename> )
endif

-include $(CONFIG_FILE)

ifndef TARGET_GCC
TARGET_GCC =
endif

ifndef TARGET_NVCC
TARGET_NVCC =
endif

ifndef CUDA_ROOT_PATH
CUDA_ROOT_PATH =
endif

ifndef BOOST_INCLUDE
BOOST_INCLUDE =
endif

ifndef AMGCL_INCLUDE
AMGCL_INCLUDE =
endif

ifndef FLOAT_TYPE
FLOAT_TYPE = float
endif

ifeq ($(FLOAT_TYPE),double)
PRECISION_SUFFIX = d
PRECISION_DEFINE = -DUSE_DOUBLE_PRECISION
else
PRECISION_SUFFIX = f
PRECISION_DEFINE =
endif

ifndef USE_APPLE_OMP
USE_APPLE_OMP = False
endif

PROJECT_ROOT_PATH = ../..
NMFD_OPERATIONS_ROOT ?= $(PROJECT_ROOT_PATH)/contrib/nmfd-operations
SCFD_ROOT ?= $(NMFD_OPERATIONS_ROOT)/contrib/scfd
SCFD_INCLUDE = $(SCFD_ROOT)/include
NMFD_INCLUDE = $(PROJECT_ROOT_PATH)/include
#INCLUDE_ROOT = -I$(PROJECT_ROOT_PATH)/sourse
#INCLUDE_LOCAL = -I$(PROJECT_ROOT_PATH)/sourse/solver
INCLUDE_CONTRIB = -I$(NMFD_INCLUDE) -I$(NMFD_OPERATIONS_ROOT)/include -I$(SCFD_INCLUDE) -I$(PROJECT_ROOT_PATH)/contrib/json

ifeq ($(USE_APPLE_OMP),True)
	OMP_FLAGS = -Xpreprocessor -fopenmp -lomp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib
else
	OMP_FLAGS = -fopenmp
endif

HOSTFLAGS = $(TARGET_GCC) -std=c++17
HOSTCOMPILER ?= g++

ifneq ($(strip $(CUDA_ARCH)),)
CUDA_ARCH_FLAG = -arch=$(CUDA_ARCH)
endif
CUDAFLAGS = $(TARGET_NVCC) -std=c++17 --extended-lambda $(CUDA_ARCH_FLAG)
ifneq ($(strip $(CUDA_ROOT_PATH)),)
CUDACOMPILER = $(CUDA_ROOT_PATH)/bin/nvcc
else
CUDACOMPILER = nvcc
endif


#MPICOMPILER = $(MPI_ROOT_PATH)/bin/mpic++
#SM = $(CUDA_ARCH)
#MPI = $(MPI_ROOT_PATH)
#HYPRELIBRARY = -lHYPRE
#CUDALIBRARIES = -lcudart -lcurand -lcusparse -lcublas
IPROJECT = ${INCLUDE_CONTRIB}
LPROJECT = -ldl

CUDA_SOLVER_LIBS ?= -lcublas -lcusolver
CUDA_OPERATIONS_OBJECTS = $(PROJECT_ROOT_PATH)/build/legacy_tests/operations_cuda_reductions.o
CUDA_SOLVER_OBJECTS = $(PROJECT_ROOT_PATH)/build/legacy_tests/dense_solvers_cuda_inst.o

# Included before the leaf Makefile's all target.
.DEFAULT_GOAL := all
$(CUDA_OPERATIONS_OBJECTS): $(NMFD_OPERATIONS_ROOT)/include/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_kernels.cu
	mkdir -p $(@D)
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) -c $< -o $@

$(CUDA_SOLVER_OBJECTS): $(PROJECT_ROOT_PATH)/test/solvers/dense_solvers_cuda_inst.cu
	mkdir -p $(@D)
	$(CUDACOMPILER) $(CUDAFLAGS) $(INCLUDE_CONTRIB) -c $< -o $@
