#!/bin/bash

# Production run script for test_biharmonic
# Runs all combinations of solver, preconditioner, and precision
#
# Usage: ./run_prod_biharmonic.sh [arch] [grid_size] [output_base] [max_iterations] [gpu_device]
#   arch: Architecture to use (cpu, omp, or cuda). Default: cuda
#   grid_size: Grid size (number of points per dimension). Default: 32
#   output_base: Base output directory. Default: conv_biharmonic_prod
#   max_iterations: Maximum number of iterations. Default: 100000
#   gpu_device: GPU device ID to use (for CUDA). Default: 0
#
# This script runs 8 combinations:
#   - Solvers: jacobi, gmres
#   - Preconditioners: diag, mg
#   - Precision: float, double
#
# Results are saved to data/{output_base}/
# Each run creates its own times.dat and conv_history.dat files in its output folder

# Tolerance settings (adjust these as needed)
TOLERANCE_FLOAT=1e-5
TOLERANCE_DOUBLE=1e-10

# Parse CLI arguments with defaults
ARCH="${1:-cuda}"
GRID_SIZE="${2:-32}"
OUTPUT_BASE="${3:-conv_biharmonic_prod}"
MAX_ITERATIONS="${4:-100000}"
GPU_DEVICE="${5:-0}"

# Set CUDA_VISIBLE_DEVICES if using CUDA
if [ "$ARCH" == "cuda" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
    echo "Using GPU device: ${GPU_DEVICE}"
fi

# Create output directory (ensure data/ prefix if not present and not absolute path)
# if [[ "$OUTPUT_BASE" != /* ]] && [[ "$OUTPUT_BASE" != data/* ]]; then
#     OUTPUT_BASE="data/${OUTPUT_BASE}"
# fi
# mkdir -p "${OUTPUT_BASE}"

# Run all combinations
for solver in jacobi gmres; do
    for prec in diag mg; do
        for float_type in f d; do
            # Determine binary name and tolerance
            if [ "$float_type" == "f" ]; then
                binary="test_biharmonic_${ARCH}_f.bin"
                tolerance="$TOLERANCE_FLOAT"
                type_label="float"
            else
                binary="test_biharmonic_${ARCH}_d.bin"
                tolerance="$TOLERANCE_DOUBLE"
                type_label="double"
            fi

            # Create descriptive folder name (without timestamp, C++ will add it)
            folder_name="${solver}_${prec}_${ARCH}_${type_label}_${GRID_SIZE}"
            prefix="${OUTPUT_BASE}/${folder_name}"

            echo "=========================================="
            echo "Running: solver=${solver}, prec=${prec}, arch=${ARCH}, type=${type_label}, size=${GRID_SIZE}"
            if [ "$ARCH" == "cuda" ]; then
                echo "GPU device: ${GPU_DEVICE}"
            fi
            echo "=========================================="

            # Check if binary exists
            if [ ! -f "$binary" ]; then
                echo "ERROR: Binary '$binary' not found. Skipping..."
                continue
            fi

            # Run the solver
            if ./"$binary" "$solver" "$prec" "$GRID_SIZE" "$prefix" --tolerance "$tolerance" --max-iterations "$MAX_ITERATIONS" --verbose ; then
                echo "SUCCESS: solver=${solver}, prec=${prec}, arch=${ARCH}, type=${type_label}, size=${GRID_SIZE}"
            else
                echo "ERROR: Execution failed for solver=${solver}, prec=${prec}, arch=${ARCH}, type=${type_label}, size=${GRID_SIZE}"
            fi

            echo ""
        done
    done
done

echo "=========================================="
echo "Production run completed!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "Each run has its own times.dat file in its output folder"
echo "=========================================="
