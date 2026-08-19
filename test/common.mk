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


# ----- Paths -----

PROJECT_ROOT_PATH = ../..
SCFD_INCLUDE = $(PROJECT_ROOT_PATH)/contrib/SCFD/include
NMFD_INCLUDE = $(PROJECT_ROOT_PATH)/include
CLI11_INCLUDE = $(PROJECT_ROOT_PATH)/contrib/CLI11/include
INCLUDE_CONTRIB = -I$(SCFD_INCLUDE) -I$(NMFD_INCLUDE) -I$(CLI11_INCLUDE)

# Required by nmfd/operations/rect_vector_space.h (it allocates its vectors index-shifted)
ARRAYS_FLAGS = -DSCFD_ARRAYS_ENABLE_INDEX_SHIFT=1


# ----- Precision -----

ifeq ($(FLOAT_TYPE),double)
PRECISION_SUFFIX = d
PRECISION_DEFINE = -DUSE_DOUBLE_PRECISION
else
PRECISION_SUFFIX = f
PRECISION_DEFINE =
endif


# ----- Profiling -----

PROFILE_DEFINE = -DSCFD_ENABLE_PROFILING


# ----- Host -----

HOSTCOMPILER = g++
HOSTFLAGS = $(TARGET_GCC) -std=c++17 $(ARRAYS_FLAGS)

ifeq ($(USE_APPLE_OMP),True)
OMP_FLAGS = -Xpreprocessor -fopenmp -lomp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib
else
OMP_FLAGS = -fopenmp
endif


# ----- CUDA -----

ifneq ($(strip $(CUDA_ARCH)),)
CUDA_ARCH_FLAG = -arch=$(CUDA_ARCH)
endif
CUDAFLAGS = $(TARGET_NVCC) -std=c++17 $(CUDA_ARCH_FLAG) $(ARRAYS_FLAGS)

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
PLATFORM_SUFFIX = _cpu
MPICXX_FLAGS =
else ifeq ($(PLATFORM),omp)
PLATFORMCXX = $(HOSTCOMPILER)
PLATFORMCXX_FLAGS = $(HOSTFLAGS) $(OMP_FLAGS)
PLATFORMCXX_LINK_FLAGS = $(OMP_FLAGS)
PLATFORM_DEFINE = -DPLATFORM_OMP
PLATFORM_SUFFIX = _omp
MPICXX_FLAGS = $(OMP_FLAGS)
else ifeq ($(PLATFORM),cuda)
PLATFORMCXX = $(CUDACOMPILER)
PLATFORMCXX_FLAGS = $(CUDAFLAGS) -x cu
PLATFORMCXX_LINK_FLAGS =
PLATFORM_DEFINE = -DPLATFORM_CUDA
PLATFORM_SUFFIX = _cuda
MPICXX_FLAGS = $(CUDA_LIB_PATH) -lcudart
else
$(error Unknown PLATFORM '$(PLATFORM)'. Supported values: cpu, omp, cuda)
endif


# ----- Platform linker -----

ifeq ($(PLATFORM_MPI),1)
PLATFORMLINKER = $(MPICXX)
PLATFORMLINKER_FLAGS = $(MPICXX_FLAGS)
PLATFORM_MPI_INCLUDE = $(INCLUDE_MPI)
PLATFORM_MPI_DEFINE = -DSCFD_BACKEND_ENABLE_MPI
MPI_SUFFIX = _mpi
else
PLATFORMLINKER = $(PLATFORMCXX)
PLATFORMLINKER_FLAGS = $(PLATFORMCXX_LINK_FLAGS)
PLATFORM_MPI_INCLUDE =
PLATFORM_MPI_DEFINE =
MPI_SUFFIX =
endif
