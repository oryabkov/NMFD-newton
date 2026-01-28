#!/bin/bash

# Grid size experiment script for test_biharmonic
# Runs production runs for different grid sizes to study convergence behavior
#
# Usage: ./run_grid_size_experiment.sh [arch] [max_iterations] [grid_sizes...]
#   arch: Architecture to use (cpu, omp, or cuda). Default: cuda
#   max_iterations: Maximum number of iterations. Default: 100000
#   grid_sizes: Space-separated list of grid sizes. Default: 2 4 8 16 32 64
#
# Example: ./run_grid_size_experiment.sh cuda 100000 16 32 64
# Example: ./run_grid_size_experiment.sh cuda 50000 32 64
#
# Results are saved to data/{grid_size}/conv_biharmonic_prod/
# Each grid size gets its own folder hierarchy

# Parse CLI arguments
ARCH="${1:-cuda}"
MAX_ITERATIONS="${2:-100000}"

# Default grid sizes if not provided
if [ $# -lt 3 ]; then
    # GRID_SIZES=(2 4 8 16 32 64)
    GRID_SIZES=(64)
else
    # Shift to skip arch and max_iterations arguments
    shift
    shift
    GRID_SIZES=("$@")
fi

echo "=========================================="
echo "Grid Size Experiment"
echo "=========================================="
echo "Architecture: ${ARCH}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "Grid sizes: ${GRID_SIZES[*]}"
echo "=========================================="
echo ""

# Run experiments for each grid size
for GRID_SIZE in "${GRID_SIZES[@]}"; do
    echo "=========================================="
    echo "Running experiments for grid size: ${GRID_SIZE}"
    echo "=========================================="

    # Create output directory structure: data/{grid_size}/conv_biharmonic_prod
    OUTPUT_BASE="conv_biharmonic_grid_prod/${GRID_SIZE}"

    # Call the production script with grid size, output base, and max iterations
    ./run_prod_biharmonic.sh "${ARCH}" "${GRID_SIZE}" "${OUTPUT_BASE}" "${MAX_ITERATIONS}"

    echo ""
    echo "Completed grid size: ${GRID_SIZE}"
    echo ""
done

echo "=========================================="
echo "Grid size experiment completed!"
echo "Results saved to: data/{grid_size}/conv_biharmonic_prod/"
echo "=========================================="
