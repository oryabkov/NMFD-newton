#!/bin/bash

# Production profiling script for test_biharmonic
# Profiles all combinations of solver, preconditioner, and precision using nvprof
#
# Usage: ./profile_prod_biharmonic.sh [grid_size] [max_iterations] [output_base] [gpu_device] [nvprof_path]
#   grid_size: Grid size (number of points per dimension). REQUIRED
#   max_iterations: Maximum number of iterations. REQUIRED
#   output_base: Base output directory. Default: profile_biharmonic_prod
#   gpu_device: GPU device ID to use (for CUDA). Default: 0
#   nvprof_path: Path to nvprof (default: /usr/local/cuda/bin/nvprof or nvprof in PATH)
#
# This script profiles 8 combinations:
#   - Solvers: jacobi, gmres
#   - Preconditioners: diag, mg
#   - Precision: float, double
#
# Profiling results (CSV files) are saved to each run's output folder

# Tolerance settings
TOLERANCE_FLOAT=1e-5
TOLERANCE_DOUBLE=1e-10

# Parse CLI arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <grid_size> <max_iterations> [output_base] [gpu_device] [nvprof_path]"
    echo ""
    echo "Example:"
    echo "  $0 32 100000 profile_biharmonic_prod 0"
    echo "  $0 32 100000 profile_biharmonic_prod 0 /usr/local/cuda/bin/nvprof"
    exit 1
fi

GRID_SIZE="$1"
MAX_ITERATIONS="$2"
OUTPUT_BASE="${3:-profile_biharmonic_prod}"
GPU_DEVICE="${4:-0}"
NVPROF_PATH="${5:-/usr/local/cuda/bin/nvprof}"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
echo "Using GPU device: ${GPU_DEVICE}"
echo ""

# Check if nvprof is available
if [ ! -f "$NVPROF_PATH" ]; then
    # Try to find nvprof in PATH
    if command -v nvprof &> /dev/null; then
        NVPROF_PATH="nvprof"
    else
        echo "ERROR: nvprof not found at '$NVPROF_PATH' and not in PATH"
        echo "Please specify the path to nvprof or ensure it's in your PATH"
        exit 1
    fi
fi

echo "Using nvprof: $NVPROF_PATH"
echo ""

# Run all combinations
for solver in jacobi gmres; do
    for prec in diag mg; do
        for float_type in f d; do
            # Determine binary name and tolerance
            if [ "$float_type" == "f" ]; then
                binary="test_biharmonic_cuda_f.bin"
                tolerance="$TOLERANCE_FLOAT"
                type_label="float"
            else
                binary="test_biharmonic_cuda_d.bin"
                tolerance="$TOLERANCE_DOUBLE"
                type_label="double"
            fi

            # Create descriptive folder name (without timestamp, C++ will add it)
            folder_name="${solver}_${prec}_cuda_${type_label}_${GRID_SIZE}"
            prefix="${OUTPUT_BASE}/${folder_name}"

            echo "=========================================="
            echo "Profiling: solver=${solver}, prec=${prec}, type=${type_label}, size=${GRID_SIZE}"
            echo "GPU device: ${GPU_DEVICE}"
            echo "=========================================="

            # Check if binary exists
            if [ ! -f "$binary" ]; then
                echo "ERROR: Binary '$binary' not found. Skipping..."
                echo ""
                continue
            fi

            echo "Profiling with nvprof (this may take a while)..."
            echo ""

            # Determine metric based on precision
            if [ "$float_type" == "f" ]; then
                METRIC="flop_count_sp"
            else
                METRIC="flop_count_dp"
            fi

            # Create temporary file to capture program output (to extract directory)
            TEMP_OUTPUT=$(mktemp)

            # Create temporary CSV file (nvprof will write to this, then we'll move it)
            TEMP_CSV=$(mktemp)

            echo "Running nvprof with metric: $METRIC"
            echo ""

            # Run with nvprof profiling
            # nvprof writes CSV to the log file, program output goes to stderr
            # Format: nvprof -m <metric> --csv --log-file <output> <binary> <args>
            # We capture stderr to extract the output directory
            if sudo "$NVPROF_PATH" -m "$METRIC" --csv --log-file "$TEMP_CSV" \
                ./"$binary" "$solver" "$prec" "$GRID_SIZE" "$prefix" \
                --tolerance "$tolerance" --max-iterations "$MAX_ITERATIONS" \
                2> "$TEMP_OUTPUT"; then

                # Extract output directory from program output (look for "Directory:" line)
                OUTPUT_DIR=$(grep "Directory:" "$TEMP_OUTPUT" | sed 's/.*Directory:[[:space:]]*//' | head -1)

                if [ -z "$OUTPUT_DIR" ]; then
                    echo "WARNING: Could not determine output directory from program output."
                    echo "Trying to find most recent matching directory..."
                    # Try to find the most recent directory matching the pattern
                    OUTPUT_DIR=$(ls -td data/${OUTPUT_BASE}/${folder_name}_* 2>/dev/null | head -1)
                fi

                if [ -z "$OUTPUT_DIR" ] || [ ! -d "$OUTPUT_DIR" ]; then
                    echo "ERROR: Output directory not found. Saving CSV to current directory..."
                    CSV_FILE="nvprof_${solver}_${prec}_${type_label}_${GRID_SIZE}.csv"
                else
                    echo "Output directory: $OUTPUT_DIR"
                    CSV_FILE="${OUTPUT_DIR}/nvprof_${solver}_${prec}_${type_label}.csv"
                fi

                # Move CSV to the output directory
                if [ -f "$TEMP_CSV" ] && [ -s "$TEMP_CSV" ]; then
                    mv "$TEMP_CSV" "$CSV_FILE"
                    echo "SUCCESS: Profiling completed for solver=${solver}, prec=${prec}, type=${type_label}"
                    echo "CSV saved to: $CSV_FILE"
                else
                    echo "WARNING: CSV file was not created or is empty"
                    rm -f "$TEMP_CSV"
                fi
            else
                echo "ERROR: Profiling failed for solver=${solver}, prec=${prec}, type=${type_label}"
                # Try to extract directory and save CSV anyway
                OUTPUT_DIR=$(grep "Directory:" "$TEMP_OUTPUT" | sed 's/.*Directory:[[:space:]]*//' | head -1)
                if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ] && [ -f "$TEMP_CSV" ] && [ -s "$TEMP_CSV" ]; then
                    CSV_FILE="${OUTPUT_DIR}/nvprof_${solver}_${prec}_${type_label}.csv"
                    mv "$TEMP_CSV" "$CSV_FILE"
                    echo "Note: Partial CSV saved to: $CSV_FILE"
                else
                    rm -f "$TEMP_CSV"
                fi
            fi

            # Clean up
            rm -f "$TEMP_OUTPUT"

            echo ""
        done
    done
done

echo "=========================================="
echo "Profiling completed!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "Each run has its own nvprof_*.csv file in its output folder"
echo "=========================================="
