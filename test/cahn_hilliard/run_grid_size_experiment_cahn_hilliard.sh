#!/bin/bash

# Grid size experiment script for test_cahn_hilliard
# Runs production runs for different grid sizes to study convergence behavior
#
# Usage: ./run_grid_size_experiment_cahn_hilliard.sh [arch] [max_iterations] [gpu_device] [grid_sizes...]
#   arch: Architecture to use (cpu, omp, or cuda). Default: cuda
#   max_iterations: Maximum number of iterations. Default: 100000
#   gpu_device: GPU device ID to use (for CUDA). Default: 0
#   grid_sizes: Space-separated list of grid sizes. Default: 2 4 8 16 32 64
#
# Example: ./run_grid_size_experiment_cahn_hilliard.sh cuda 100000 0 16 32 64
# Example: ./run_grid_size_experiment_cahn_hilliard.sh cuda 50000 1 32 64
#
# Results are saved to data/{grid_size}/conv_cahn_hilliard_grid_prod/
# Each grid size gets its own folder hierarchy
# This script runs both jacobi+diag and gmres+mg combinations

# Parse CLI arguments
ARCH="${1:-cuda}"
MAX_ITERATIONS="${2:-100000}"
GPU_DEVICE="${3:-0}"

# Default grid sizes if not provided
if [ $# -lt 4 ]; then
    # GRID_SIZES=(4 8 16 32 64 128)
    GRID_SIZES=(256)
else
    # Shift to skip arch, max_iterations, and gpu_device arguments
    shift
    shift
    shift
    GRID_SIZES=("$@")
fi

echo "=========================================="
echo "Grid Size Experiment - Cahn-Hilliard"
echo "=========================================="
echo "Architecture: ${ARCH}"
echo "Max iterations: ${MAX_ITERATIONS}"
if [ "$ARCH" == "cuda" ]; then
    echo "GPU device: ${GPU_DEVICE}"
fi
echo "Grid sizes: ${GRID_SIZES[*]}"
echo "Solver combinations: jacobi+diag, gmres+mg"
echo "=========================================="
echo ""

# Run experiments for each grid size
for GRID_SIZE in "${GRID_SIZES[@]}"; do
    echo "=========================================="
    echo "Running experiments for grid size: ${GRID_SIZE}"
    echo "=========================================="

    # Create output directory structure: data/{grid_size}/conv_cahn_hilliard_grid_prod
    OUTPUT_BASE="conv_cahn_hilliard_grid_prod/${GRID_SIZE}"

    # Run jacobi + diag combination
    echo ""
    echo "--- Running jacobi + diag ---"
    ./run_prod_cahn_hilliard.sh "${ARCH}" "${GRID_SIZE}" "${OUTPUT_BASE}" "${MAX_ITERATIONS}" "${GPU_DEVICE}" \
        "jacobi" "diag"

    # Run gmres + mg combination
    # echo ""
    # echo "--- Running gmres + mg ---"
    # ./run_prod_cahn_hilliard.sh "${ARCH}" "${GRID_SIZE}" "${OUTPUT_BASE}" "${MAX_ITERATIONS}" "${GPU_DEVICE}" \
    #     "gmres" "mg"

    echo ""
    echo "Completed grid size: ${GRID_SIZE}"
    echo ""
done

echo "=========================================="
echo "Grid size experiment completed!"
echo "Results saved to: data/{grid_size}/conv_cahn_hilliard_grid_prod/"
echo "=========================================="
