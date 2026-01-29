#!/bin/bash

# Grandiose timing measurement script for test_biharmonic
# Measures execution time across CPU, OMP (with different thread counts), and CUDA platforms
# to analyze acceleration
#
# Usage: ./time_prod_biharmonic.sh [gpu_device]
#   gpu_device: GPU device ID to use (for CUDA). Default: 0
#
# This script runs:
#   - Platforms: CPU, OMP (2, 4, 8, 16, 32, 64 threads), CUDA
#   - Solvers: jacobi, gmres
#   - Preconditioners: diag, mg
#   - Precision: float, double
#   - Grid size: 256
#   - Max iterations: 100
#
# Results are saved to time_biharmonic_prod/{platform}/{threads}/
# Each run creates its own times.dat file in its output folder

# Tolerance settings
TOLERANCE_FLOAT=1e-5
TOLERANCE_DOUBLE=1e-10

# Fixed parameters
GRID_SIZE=256
MAX_ITERATIONS=100
OUTPUT_BASE="time_biharmonic_prod"

# Parse CLI arguments
GPU_DEVICE="${1:-0}"

# OMP thread counts to test
OMP_THREADS=(32 64 96 128)

echo "=========================================="
echo "Grandiose Timing Measurement"
echo "=========================================="
echo "Grid size: ${GRID_SIZE}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "Output base: ${OUTPUT_BASE}"
echo "=========================================="
echo ""

# Function to run a single test
run_test() {
    local arch=$1
    local threads=$2
    local solver=$3
    local prec=$4
    local float_type=$5

    # Determine binary name and tolerance
    if [ "$float_type" == "f" ]; then
        binary="test_biharmonic_${arch}_f.bin"
        tolerance="$TOLERANCE_FLOAT"
        type_label="float"
    else
        binary="test_biharmonic_${arch}_d.bin"
        tolerance="$TOLERANCE_DOUBLE"
        type_label="double"
    fi

    # Create output directory structure
    if [ "$arch" == "omp" ]; then
        prefix="${OUTPUT_BASE}/${arch}/${threads}/${solver}_${prec}_${arch}_${type_label}_${GRID_SIZE}"
    else
        prefix="${OUTPUT_BASE}/${arch}/${solver}_${prec}_${arch}_${type_label}_${GRID_SIZE}"
    fi

    echo "Running: arch=${arch}, threads=${threads}, solver=${solver}, prec=${prec}, type=${type_label}"

    # Check if binary exists
    if [ ! -f "$binary" ]; then
        echo "ERROR: Binary '$binary' not found. Skipping..."
        return 1
    fi

    # Set OMP_NUM_THREADS if using OMP
    if [ "$arch" == "omp" ]; then
        export OMP_NUM_THREADS="${threads}"
    fi

    # Run the solver (without --verbose for clean timing)
    if ./"$binary" "$solver" "$prec" "$GRID_SIZE" "$prefix" --tolerance "$tolerance" --max-iterations "$MAX_ITERATIONS" ; then
        echo "SUCCESS: arch=${arch}, threads=${threads}, solver=${solver}, prec=${prec}, type=${type_label}"
        return 0
    else
        echo "ERROR: Execution failed for arch=${arch}, threads=${threads}, solver=${solver}, prec=${prec}, type=${type_label}"
        return 1
    fi
}

# Set CUDA_VISIBLE_DEVICES for CUDA runs
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"

# ==========================================
# CPU Platform
# ==========================================
# echo "=========================================="
# echo "Testing CPU Platform"
# echo "=========================================="
# for solver in jacobi gmres; do
#     for prec in diag mg; do
#         for float_type in f d; do
#             run_test "cpu" "1" "$solver" "$prec" "$float_type"
#             echo ""
#         done
#     done
# done

# ==========================================
# OMP Platform (with different thread counts)
# ==========================================
echo "=========================================="
echo "Testing OMP Platform"
echo "=========================================="
for threads in "${OMP_THREADS[@]}"; do
    echo "----------------------------------------"
    echo "OMP with ${threads} threads"
    echo "----------------------------------------"
    for solver in jacobi gmres; do
        for prec in diag mg; do
            for float_type in f d; do
                run_test "omp" "$threads" "$solver" "$prec" "$float_type"
                echo ""
            done
        done
    done
done

# ==========================================
# CUDA Platform
# ==========================================
echo "=========================================="
echo "Testing CUDA Platform"
echo "=========================================="
echo "Using GPU device: ${GPU_DEVICE}"
for solver in jacobi gmres; do
    for prec in diag mg; do
        for float_type in f d; do
            run_test "cuda" "1" "$solver" "$prec" "$float_type"
            echo ""
        done
    done
done

echo "=========================================="
echo "Grandiose Timing Measurement Completed!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_BASE}/"
echo "  - CPU: ${OUTPUT_BASE}/cpu/"
echo "  - OMP: ${OUTPUT_BASE}/omp/{2,4,8,16,32,64}/"
echo "  - CUDA: ${OUTPUT_BASE}/cuda/"
echo ""
echo "Each run has its own times.dat file in its output folder"
echo "=========================================="
