#!/bin/bash
#SBATCH --job-name=ch_cos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=10

set -euo pipefail

# Load CUDA module (adjust if your cluster uses a different version)
# module load cuda/12.0

cos_thetas=(-1.0 -0.8660254038 -0.7071067812 -0.5 0 0.5 0.7071067812 0.8660254038 1.0)
num_gpus=1

pids=()
for ((i=0; i<${#cos_thetas[@]}; i++)); do
    cos_theta="${cos_thetas[i]}"
    gpu_id=$((i % num_gpus))
    folder="fixed_bc_cos_theta_test/cos_theta_${cos_theta}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" ./test_cahn_hilliard_nonlinear_cuda_d.bin \
        gmres mg 128 "$folder" \
        --tolerance 1e-9 \
        --newton-tol 1e-9 \
        --max-time-steps 100 \
        --dt-inf 1e0 \
        --gamma 1e-4 \
        --cos-theta "$cos_theta" \
        --save-coords &
    pids+=("$!")
done

wait "${pids[@]}"
