#!/bin/bash

# Production profiling script for test_biharmonic
# Profiles all combinations of solver, preconditioner, and precision using nvprof
#
# Usage: ./profile_prod_biharmonic.sh <grid_size> <max_iterations> [output_base] [gpu_device] [nvprof_path]

# Tolerance settings
TOLERANCE_FLOAT=1e-5
TOLERANCE_DOUBLE=1e-10

# Parse CLI arguments
GRID_SIZE="$1"
MAX_ITERATIONS="$2"
OUTPUT_BASE="${3:-profile_biharmonic_prod}"
GPU_DEVICE="${4:-0}"
NVPROF_PATH="${5:-/usr/local/cuda/bin/nvprof}"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"

# Check if nvprof is available
if [ ! -f "$NVPROF_PATH" ] && ! command -v nvprof &> /dev/null; then
    echo "ERROR: nvprof not found"
    exit 1
fi

if [ ! -f "$NVPROF_PATH" ]; then
    NVPROF_PATH="nvprof"
fi

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Run all combinations
for solver in jacobi gmres; do
    for prec in diag mg; do
        for float_type in f d; do
            if [ "$float_type" == "f" ]; then
                binary="test_biharmonic_cuda_f.bin"
                tolerance="$TOLERANCE_FLOAT"
                type_label="float"
                METRIC="flop_count_sp"
            else
                binary="test_biharmonic_cuda_d.bin"
                tolerance="$TOLERANCE_DOUBLE"
                type_label="double"
                METRIC="flop_count_dp"
            fi

            CSV_FILE="data/${OUTPUT_BASE}/nvprof_${solver}_${prec}_${type_label}.csv"
            prefix="${OUTPUT_BASE}/${solver}_${prec}_cuda_${type_label}_${GRID_SIZE}"

            echo "Profiling: solver=${solver}, prec=${prec}, type=${type_label}"

            if [ ! -f "$binary" ]; then
                echo "  ERROR: Binary '$binary' not found. Skipping..."
                continue
            fi

            sudo "$NVPROF_PATH" -m "$METRIC" --csv --log-file "$CSV_FILE" \
                ./"$binary" "$solver" "$prec" "$GRID_SIZE" "$prefix" \
                --tolerance "$tolerance" --max-iterations "$MAX_ITERATIONS" \
                > /dev/null 2>&1

            sudo chown $(whoami):$(whoami) "$CSV_FILE" 2>/dev/null || true
            echo "  CSV saved to: $CSV_FILE"
            echo ""
        done
    done
done

echo "Profiling completed! Results saved to: ${OUTPUT_BASE}/"
