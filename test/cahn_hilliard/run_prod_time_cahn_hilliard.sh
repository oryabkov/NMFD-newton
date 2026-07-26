#!/bin/bash

# Production run script for test_time_cahn_hilliard
# Runs time-dependent Cahn-Hilliard solver for different solver/preconditioner combinations
#
# Usage: ./run_prod_time_cahn_hilliard.sh [arch] [grid_size] [output_base] [max_iterations] [gpu_device] [solver] [preconditioner] [dt_inf] [max_time_steps] [time_tol]
#   arch: Architecture to use (cpu, omp, or cuda). Default: cuda
#   grid_size: Grid size (number of points per dimension). Default: 32
#   output_base: Base output directory. Default: conv_time_cahn_hilliard_prod
#   max_iterations: Maximum number of linear solver iterations. Default: 100000
#   gpu_device: GPU device ID to use (for CUDA). Default: 0
#   solver: Solver type (jacobi or gmres). Default: gmres
#   preconditioner: Preconditioner type (diag or mg). Default: mg
#   dt_inf: dt_inf parameter (1/dt). Default: 1.0
#   max_time_steps: Maximum number of time steps. Default: 1000
#   time_tol: Time convergence tolerance. Default: 5e-6 (float) or 1e-10 (double)
#
# This script runs 2 combinations by default (float and double):
#   - Precision: float, double
#   - Solver: gmres (default)
#   - Preconditioner: mg (default)
#
# Results are saved to data/{output_base}/
# Each run creates its own step_N folders with times.dat, nonlinear_history.dat, and convergence_history_N.dat files

# Tolerance settings (adjust these as needed)
TOLERANCE_FLOAT=2e-4
# TOLERANCE_DOUBLE=1e-10
TOLERANCE_DOUBLE=1e-3

NEWTON_TOL_FLOAT=9e-1
# NEWTON_TOL_DOUBLE=2e-9
NEWTON_TOL_DOUBLE=2e-6

TIME_TOL_FLOAT=9e-1
# TIME_TOL_DOUBLE=2e-9
TIME_TOL_DOUBLE=2e-6

# Parse CLI arguments with defaults
ARCH="${1:-cuda}"
GRID_SIZE="${2:-32}"
OUTPUT_BASE="${3:-conv_time_cahn_hilliard_prod}"
MAX_ITERATIONS="${4:-100000}"
GPU_DEVICE="${5:-0}"
SOLVER="${6:-gmres}"
PRECONDITIONER="${7:-mg}"
DT_INF="${8:-100}"
MAX_TIME_STEPS="${9:-10}"
TIME_TOL="${10:-}"

# Set CUDA_VISIBLE_DEVICES if using CUDA
if [ "$ARCH" == "cuda" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
    echo "Using GPU device: ${GPU_DEVICE}"
fi

# Run for both float and double precision
for float_type in f d; do
    # Determine binary name and tolerance
    if [ "$float_type" == "f" ]; then
        binary="test_time_cahn_hilliard_${ARCH}_f.bin"
        tolerance="$TOLERANCE_FLOAT"
        newton_tol="$NEWTON_TOL_FLOAT"
        time_tol="${TIME_TOL:-$TIME_TOL_FLOAT}"
        type_label="float"
    else
        binary="test_time_cahn_hilliard_${ARCH}_d.bin"
        tolerance="$TOLERANCE_DOUBLE"
        newton_tol="$NEWTON_TOL_DOUBLE"
        time_tol="${TIME_TOL:-$TIME_TOL_DOUBLE}"
        type_label="double"
    fi

    # Create descriptive folder name (without timestamp, C++ will add it)
    folder_name="${SOLVER}_${PRECONDITIONER}_${ARCH}_${type_label}_${GRID_SIZE}"
    prefix="${OUTPUT_BASE}/${folder_name}"

    echo "=========================================="
    echo "Running: solver=${SOLVER}, prec=${PRECONDITIONER}, arch=${ARCH}, type=${type_label}, size=${GRID_SIZE}"
    echo "  dt_inf=${DT_INF}, max_time_steps=${MAX_TIME_STEPS}, time_tol=${time_tol}"
    if [ "$ARCH" == "cuda" ]; then
        echo "  GPU device: ${GPU_DEVICE}"
    fi
    echo "=========================================="

    # Check if binary exists
    if [ ! -f "$binary" ]; then
        echo "ERROR: Binary '$binary' not found. Skipping..."
        continue
    fi

    # Run the solver
    if ./"$binary" "$SOLVER" "$PRECONDITIONER" "$GRID_SIZE" "$prefix" \
        --tolerance "$tolerance" \
        --newton-tol "$newton_tol" \
        --max-iterations "$MAX_ITERATIONS" \
        --dt-inf "$DT_INF" \
        --max-time-steps "$MAX_TIME_STEPS" \
        --time-tol "$time_tol" ; then
        echo "SUCCESS: solver=${SOLVER}, prec=${PRECONDITIONER}, arch=${ARCH}, type=${type_label}, size=${GRID_SIZE}"
    else
        echo "ERROR: Execution failed for solver=${SOLVER}, prec=${PRECONDITIONER}, arch=${ARCH}, type=${type_label}, size=${GRID_SIZE}"
    fi

    echo ""
done

echo "=========================================="
echo "Production run completed!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "Each run has its own step_N folders with times.dat, nonlinear_history.dat, and convergence_history_N.dat files"
echo "=========================================="
