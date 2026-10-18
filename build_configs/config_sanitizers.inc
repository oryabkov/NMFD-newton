HOST_CXX = g++
CPPSTD = c++17
TARGET_CXX = -O1 -g
WARNING_FLAGS = -Wall -Wextra -Wpedantic
OPENMP_FLAGS = -fopenmp -pthread
SANITIZER_FLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer
TEST_OMP_NUM_THREADS = 2
