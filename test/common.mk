ifndef CONFIG_FILE
CONFIG_FILE = config.inc
endif

ifeq (,$(wildcard $(CONFIG_FILE)))
$(info config file $(CONFIG_FILE) does not exist.)
$(error Create $(CONFIG_FILE) from example or specify another config via: make <target> CONFIG_FILE=<config_filename> )
endif

-include $(CONFIG_FILE)


# ----- Config defaults -----

TARGET_GCC ?=
TARGET_NVCC ?=
CUDA_ROOT_PATH ?=
BOOST_INCLUDE ?=
AMGCL_INCLUDE ?=
FLOAT_TYPE ?= float
USE_APPLE_OMP ?= False
PLATFORM ?= omp
PLATFORM_MPI ?= 0
CUDA_AWARE_MPI ?= False
ENABLE_PROFILING ?= False
BUILD_DIR ?= bin


# ----- Paths -----

PROJECT_ROOT_PATH = ../..
SCFD_INCLUDE = $(PROJECT_ROOT_PATH)/contrib/SCFD/include
NMFD_INCLUDE = $(PROJECT_ROOT_PATH)/include
CLI11_INCLUDE = $(PROJECT_ROOT_PATH)/contrib/CLI11/include
INCLUDE_CONTRIB = -I$(SCFD_INCLUDE) -I$(NMFD_INCLUDE) -I$(CLI11_INCLUDE)


# ----- Precision -----

ifeq ($(FLOAT_TYPE),double)
PRECISION_SUFFIX = d
PRECISION_DEFINE = -DUSE_DOUBLE_PRECISION
else
PRECISION_SUFFIX = f
PRECISION_DEFINE =
endif


# ----- Profiling -----

ifeq ($(ENABLE_PROFILING),True)
PROFILE_DEFINE = -DSCFD_ENABLE_PROFILING
else
PROFILE_DEFINE =
endif


# ----- Host -----

HOSTCOMPILER = g++
HOSTFLAGS = $(TARGET_GCC) -std=c++17

ifeq ($(USE_APPLE_OMP),True)
OMP_FLAGS = -Xpreprocessor -fopenmp -lomp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib
else
OMP_FLAGS = -fopenmp
endif


# ----- CUDA -----

ifneq ($(strip $(CUDA_ARCH)),)
# CUDA_ARCH_FLAG = -arch=$(CUDA_ARCH)
# CUDA_ARCH may list several sm_XX targets (e.g. "sm_70 sm_80") to build one fat binary
# that runs on all of them, instead of a single -arch=sm_XX tied to one GPU generation.
CUDA_ARCH_FLAG = $(foreach arch,$(CUDA_ARCH),-gencode arch=compute_$(patsubst sm_%,%,$(arch)),code=$(arch))
endif
CUDAFLAGS = $(TARGET_NVCC) -std=c++17 $(CUDA_ARCH_FLAG)

ifneq ($(strip $(CUDA_ROOT_PATH)),)
CUDACOMPILER = $(CUDA_ROOT_PATH)/bin/nvcc
CUDA_LIB_PATH = -L$(CUDA_ROOT_PATH)/lib64
else
CUDACOMPILER = nvcc
CUDA_LIB_PATH =
endif


# ----- MPI -----

MPICXX ?= mpic++
INCLUDE_MPI := $(filter -I%,$(shell $(MPICXX) -show 2>/dev/null))


# ----- Platform -----

ifeq ($(PLATFORM),cpu)
PLATFORMCXX = $(HOSTCOMPILER)
PLATFORMCXX_FLAGS = $(HOSTFLAGS)
PLATFORMCXX_LINK_FLAGS =
PLATFORM_DEFINE = -DPLATFORM_SERIAL_CPU
PLATFORM_SUFFIX = cpu
MPICXX_FLAGS =
else ifeq ($(PLATFORM),omp)
PLATFORMCXX = $(HOSTCOMPILER)
PLATFORMCXX_FLAGS = $(HOSTFLAGS) $(OMP_FLAGS)
PLATFORMCXX_LINK_FLAGS = $(OMP_FLAGS)
PLATFORM_DEFINE = -DPLATFORM_OMP
PLATFORM_SUFFIX = omp
MPICXX_FLAGS = $(OMP_FLAGS)
else ifeq ($(PLATFORM),cuda)
PLATFORMCXX = $(CUDACOMPILER)
PLATFORMCXX_FLAGS = $(CUDAFLAGS) -x cu
PLATFORMCXX_LINK_FLAGS =
PLATFORM_DEFINE = -DPLATFORM_CUDA
PLATFORM_SUFFIX = cuda
MPICXX_FLAGS = $(CUDA_LIB_PATH) -lcudart
else
$(error Unknown PLATFORM '$(PLATFORM)'. Supported values: cpu, omp, cuda)
endif


# ----- Platform linker -----

ifeq ($(PLATFORM_MPI),1)
PLATFORMLINKER = $(MPICXX)
PLATFORMLINKER_FLAGS = $(MPICXX_FLAGS)
PLATFORM_INCLUDE = $(INCLUDE_MPI)
PLATFORM_DEFINE += -DSCFD_BACKEND_ENABLE_MPI
ifeq ($(CUDA_AWARE_MPI),True)
PLATFORM_DEFINE += -DSCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
endif
PLATFORM_SUFFIX := $(PLATFORM_SUFFIX)_mpi
else
PLATFORMLINKER = $(PLATFORMCXX)
PLATFORMLINKER_FLAGS = $(PLATFORMCXX_LINK_FLAGS)
PLATFORM_INCLUDE =
endif
