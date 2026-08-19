#!/bin/bash

# Grid size experiment script for test_time_cahn_hilliard
# Runs production runs for different grid sizes to study convergence behavior
#
# Usage: ./run_grid_size_experiment_time_cahn_hilliard.sh [arch] [max_iterations] [gpu_device] [dt_inf] [max_time_steps] [time_tol] [grid_sizes...]
#   arch: Architecture to use (cpu, omp, or cuda). Default: cuda
#   max_iterations: Maximum number of linear solver iterations. Default: 100000
#   gpu_device: GPU device ID to use (for CUDA). Default: 0
#   dt_inf: dt_inf parameter (1/dt). Default: 100.0
#   max_time_steps: Maximum number of time steps. Default: 1000
#   time_tol: Time convergence tolerance. Default: 5e-6 (float) or 1e-10 (double)
#   grid_sizes: Space-separated list of grid sizes. Default: 2 4 8 16 32 64 128 256
#
# Example: ./run_grid_size_experiment_time_cahn_hilliard.sh cuda 100000 0 1.0 1000 5e-6 16 32 64
# Example: ./run_grid_size_experiment_time_cahn_hilliard.sh cuda 50000 1 0.5 500 1e-6 32 64
#
# Results are saved to data/{grid_size}/conv_time_cahn_hilliard_grid_prod/
# Each grid size gets its own folder hierarchy
# This script runs both jacobi+diag and gmres+mg combinations

# Parse CLI arguments
ARCH="${1:-cuda}"
MAX_ITERATIONS="${2:-100000}"
GPU_DEVICE="${3:-0}"
DT_INF="${4:-1000}"
MAX_TIME_STEPS="${5:-100}"
TIME_TOL="${6:-}"

# Default grid sizes if not provided
if [ $# -lt 7 ]; then
    # GRID_SIZES=(4 8 16 32 64 128 256)
    GRID_SIZES=(16)
else
    # Shift to skip arch, max_iterations, gpu_device, dt_inf, max_time_steps, and time_tol arguments
    shift
    shift
    shift
    shift
    shift
    shift
    GRID_SIZES=("$@")
fi

echo "=========================================="
echo "Grid Size Experiment - Time-Dependent Cahn-Hilliard"
echo "=========================================="
echo "Architecture: ${ARCH}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "dt_inf: ${DT_INF}"
echo "Max time steps: ${MAX_TIME_STEPS}"
if [ -n "$TIME_TOL" ]; then
    echo "Time tolerance: ${TIME_TOL}"
else
    echo "Time tolerance: default (5e-6 for float, 1e-10 for double)"
fi
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

    # Create output directory structure: data/{grid_size}/conv_time_cahn_hilliard_grid_prod
    OUTPUT_BASE="conv_time_cahn_hilliard_grid_prod/${GRID_SIZE}"

    # Run jacobi + diag combination
    echo ""
    echo "--- Running jacobi + diag ---"
    ./run_prod_time_cahn_hilliard.sh "${ARCH}" "${GRID_SIZE}" "${OUTPUT_BASE}" "${MAX_ITERATIONS}" "${GPU_DEVICE}" \
        "jacobi" "diag" "${DT_INF}" "${MAX_TIME_STEPS}" "${TIME_TOL}"

    # Run gmres + mg combination
    echo ""
    echo "--- Running gmres + mg ---"
    ./run_prod_time_cahn_hilliard.sh "${ARCH}" "${GRID_SIZE}" "${OUTPUT_BASE}" "${MAX_ITERATIONS}" "${GPU_DEVICE}" \
        "gmres" "mg" "${DT_INF}" "${MAX_TIME_STEPS}" "${TIME_TOL}"

    echo ""
    echo "Completed grid size: ${GRID_SIZE}"
    echo ""
done

echo "=========================================="
echo "Grid size experiment completed!"
echo "Results saved to: data/{grid_size}/conv_time_cahn_hilliard_grid_prod/"
echo "=========================================="
