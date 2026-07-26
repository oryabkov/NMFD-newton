#!/bin/bash
#SBATCH --job-name=biharmonic_timing
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load CUDA module
module load cuda/12.0

# Run the timing script
./time_prod_biharmonic.sh
