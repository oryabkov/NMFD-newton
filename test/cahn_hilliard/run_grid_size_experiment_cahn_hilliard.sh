#!/bin/bash

# Grid size experiment script for test_cahn_hilliard
# Runs production runs for different grid sizes to study convergence behavior
#
# Usage: ./run_grid_size_experiment_cahn_hilliard.sh [arch] [max_iterations] [gpu_device] [solver] [preconditioner] [grid_sizes...]
#   arch: Architecture to use (cpu, omp, or cuda). Default: cuda
#   max_iterations: Maximum number of iterations. Default: 100000
#   gpu_device: GPU device ID to use (for CUDA). Default: 0
#   solver: Solver type (jacobi or gmres). Default: gmres
#   preconditioner: Preconditioner type (diag or mg). Default: mg
#   grid_sizes: Space-separated list of grid sizes. Default: 2 4 8 16 32 64
#
# Example: ./run_grid_size_experiment_cahn_hilliard.sh cuda 100000 0 gmres mg 16 32 64
# Example: ./run_grid_size_experiment_cahn_hilliard.sh cuda 50000 1 gmres mg 32 64
#
# Results are saved to data/{grid_size}/conv_cahn_hilliard_grid_prod/
# Each grid size gets its own folder hierarchy

# Parse CLI arguments
ARCH="${1:-cuda}"
MAX_ITERATIONS="${2:-100000}"
GPU_DEVICE="${3:-0}"
SOLVER="${4:-gmres}"
PRECONDITIONER="${5:-mg}"

# Default grid sizes if not provided
if [ $# -lt 6 ]; then
    GRID_SIZES=(2 4 8 16 32 64 128 256)
    # GRID_SIZES=(64)
else
    # Shift to skip arch, max_iterations, gpu_device, solver, and preconditioner arguments
    shift
    shift
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
echo "Solver: ${SOLVER}"
echo "Preconditioner: ${PRECONDITIONER}"
if [ "$ARCH" == "cuda" ]; then
    echo "GPU device: ${GPU_DEVICE}"
fi
echo "Grid sizes: ${GRID_SIZES[*]}"
echo "=========================================="
echo ""

# Run experiments for each grid size
for GRID_SIZE in "${GRID_SIZES[@]}"; do
    echo "=========================================="
    echo "Running experiments for grid size: ${GRID_SIZE}"
    echo "=========================================="

    # Create output directory structure: data/{grid_size}/conv_cahn_hilliard_grid_prod
    OUTPUT_BASE="conv_cahn_hilliard_grid_prod/${GRID_SIZE}"

    # Call the production script with grid size, output base, max iterations, GPU device, solver, and preconditioner
    ./run_prod_cahn_hilliard.sh "${ARCH}" "${GRID_SIZE}" "${OUTPUT_BASE}" "${MAX_ITERATIONS}" "${GPU_DEVICE}" "${SOLVER}" "${PRECONDITIONER}"

    echo ""
    echo "Completed grid size: ${GRID_SIZE}"
    echo ""
done

echo "=========================================="
echo "Grid size experiment completed!"
echo "Results saved to: data/{grid_size}/conv_cahn_hilliard_grid_prod/"
echo "=========================================="
