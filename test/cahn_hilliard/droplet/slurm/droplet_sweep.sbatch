#!/bin/bash
#SBATCH --job-name=droplet
#SBATCH --partition=AMD7763-A100-CCP
#SBATCH --nodelist=node208-24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=droplet_%x_%j.out

# Runs one named sweep from ../sweep.sh on the A100. Submit from test/cahn_hilliard/droplet/slurm:
#
#   sbatch --job-name=r0_dw  --export=ALL,SWEEP=r0_dw  droplet_sweep.sbatch
#   sbatch --job-name=faceavg --export=ALL,SWEEP=faceavg,GRID=64 droplet_sweep.sbatch
#
# Environment knobs forwarded to sweep.sh: SWEEP (required), GRID, DT, STEPS, SOLVER, PRECOND,
# R0_LIST.

set -euo pipefail

module load nvhpc/26.3 cuda/12.9

cd "$SLURM_SUBMIT_DIR/.."

: "${SWEEP:?set SWEEP=<name>, see ./sweep.sh list}"

make -C .. test_time_cahn_hilliard PLATFORM=cuda > "slurm/build_${SWEEP}_${SLURM_JOB_ID}.log" 2>&1 \
    || { echo "BUILD FAILED"; grep -E "error" "slurm/build_${SWEEP}_${SLURM_JOB_ID}.log" | head -30; exit 1; }

echo "starting sweep=${SWEEP} on $(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

./sweep.sh "$SWEEP"

echo "done: see ../data/${SWEEP}_*"
