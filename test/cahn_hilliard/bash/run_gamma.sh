#!/bin/bash
#SBATCH --job-name=ch_gamma
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=6

set -euo pipefail

# Load CUDA module (adjust if your cluster uses a different version)
# module load cuda/12.0

# gammas=(13e-4 11e-4 9e-4 7e-4 5e-4 3e-4 1e-4)
gammas=(1e0 1e-1 1e-2 1e-3 1e-4 1e-5)
num_gpus=1

STEPS=100

pids=()
for ((i=0; i<${#gammas[@]}; i++)); do
    gamma="${gammas[i]}"
    gpu_id=$((i % num_gpus))
    folder="gamma_NoC_test/gamma_${gamma}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" ./test_cahn_hilliard_nonlinear_cuda_d.bin \
        gmres mg 128 "$folder" \
        --tolerance 1e-9 \
        --newton-tol 1e-9 \
        --max-time-steps "$STEPS" \
        --dt-inf 1e0 \
        --gamma "$gamma" \
        --cos-theta -0.5 \
        --save-coords &
    pids+=("$!")
done

wait "${pids[@]}"
