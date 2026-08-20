#!/usr/bin/env bash
#
# Generic wrapper for the CLI solver test binaries in this directory: turns
# a "prefix" argument into a real, pre-created results directory and tees
# combined stdout+stderr into it. Also handles mpirun/OpenMP/GPU selection
# so the binary itself never has to know about any of that.
#
# Usage:
#   ./run.sh [wrapper flags] <binary> <solver> <preconditioner> <grid_size> <prefix> [binary flags...]
#
# Wrapper flags (-np/-omp/-gpus are accepted as aliases for --np/--omp/--gpus;
# must appear before the binary name; everything after the binary name is
# forwarded to it unchanged):
#   --np, -np N     run under mpirun with N ranks. Defaults to 1 if the
#                    binary name contains "mpi" and this is not given.
#   --omp, -omp T    sets OMP_NUM_THREADS=T
#   --gpus, -gpus G  must be 1 (all ranks share GPU 0) or equal to --np (one
#                    dedicated GPU per rank: GPU i for rank i)
#
# Examples:
#   ./run.sh --omp 4 binary_omp_d.bin gmres mg 16 test --verbose
#   ./run.sh -np 4 --omp 2 --gpus 4 binary_mpi_cuda_d.bin gmres mg 16 test --verbose
#
# The 4th non-flag token after the binary name (solver preconditioner
# grid_size prefix) is treated as the prefix and replaced in place with
# data/<prefix>_<timestamp>, which is created before the run.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

NP=""
OMP=""
GPUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --np|-np)
            NP="$2"
            shift 2
            ;;
        --omp|-omp)
            OMP="$2"
            shift 2
            ;;
        --gpus|-gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "error: missing binary name" >&2
    echo "usage: ./run.sh [--np N] [--omp T] [--gpus G] <binary_name> <solver> <preconditioner> <grid_size> <prefix> [binary flags...]" >&2
    exit 1
fi

BINARY_NAME="$1"
shift
BINARY="./${BINARY_NAME}"

if [[ -z "$NP" && "$BINARY_NAME" == *mpi* ]]; then
    NP=1
fi

if [[ -n "$GPUS" ]]; then
    EFFECTIVE_NP="${NP:-1}"
    if [[ "$GPUS" -ne 1 && "$GPUS" -ne "$EFFECTIVE_NP" ]]; then
        echo "error: --gpus must be 1 (shared) or equal to --np (one GPU per rank), got --gpus $GPUS with np=$EFFECTIVE_NP" >&2
        exit 1
    fi
fi

if [[ ! -x "$BINARY" ]]; then
    echo "error: binary $BINARY_NAME not found or not executable in $(pwd)" >&2
    exit 1
fi

ARGS=("$@")

PREFIX_INDEX=-1
NON_DASH_COUNT=0
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" != -* ]]; then
        NON_DASH_COUNT=$((NON_DASH_COUNT + 1))
        if [[ "$NON_DASH_COUNT" -eq 4 ]]; then
            PREFIX_INDEX="$i"
            break
        fi
    fi
done

if [[ "$PREFIX_INDEX" -eq -1 ]]; then
    echo "error: expected <solver> <preconditioner> <grid_size> <prefix> after the binary name" >&2
    echo "usage: ./run.sh [--np N] [--omp T] [--gpus G] <binary_name> <solver> <preconditioner> <grid_size> <prefix> [binary flags...]" >&2
    exit 1
fi

PREFIX="${ARGS[$PREFIX_INDEX]}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
output_dir="data/${PREFIX}_${TIMESTAMP}"
ARGS[$PREFIX_INDEX]="$output_dir"

mkdir -p "$output_dir"
echo "output_dir: $output_dir"

if [[ -n "$NP" ]]; then
    MPIRUN_ARGS=(--bind-to none -np "$NP")
    if [[ -n "$OMP" ]]; then
        MPIRUN_ARGS+=(-x "OMP_NUM_THREADS=$OMP")
    fi

    if [[ "$GPUS" == "1" ]]; then
        MPIRUN_ARGS+=(-x "CUDA_VISIBLE_DEVICES=0")
        mpirun "${MPIRUN_ARGS[@]}" "$BINARY" "${ARGS[@]}" \
            2>&1 | tee "$output_dir/log.txt"
    elif [[ -n "$GPUS" ]]; then
        mpirun "${MPIRUN_ARGS[@]}" \
            bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_RANK; exec "$0" "$@"' \
            "$BINARY" "${ARGS[@]}" \
            2>&1 | tee "$output_dir/log.txt"
    else
        mpirun "${MPIRUN_ARGS[@]}" "$BINARY" "${ARGS[@]}" \
            2>&1 | tee "$output_dir/log.txt"
    fi
else
    if [[ -n "$GPUS" ]]; then
        export CUDA_VISIBLE_DEVICES=0
    fi
    if [[ -n "$OMP" ]]; then
        export OMP_NUM_THREADS="$OMP"
    fi
    "$BINARY" "${ARGS[@]}" \
        2>&1 | tee "$output_dir/log.txt"
fi

echo "output_dir: $output_dir"
