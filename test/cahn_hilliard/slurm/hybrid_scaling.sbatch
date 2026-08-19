#!/bin/bash
#SBATCH --job-name=hybrid_scaling
#SBATCH --partition=AMD7763-A100-CCP
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=01:00:00
#SBATCH --output=hybrid_scaling_%j.out

# Hybrid MPI + OpenMP sweep for the biharmonic test problem under a fixed budget of 32 cores:
# every combination with ranks * threads <= 32 is run on each grid, plus a serial reference.
# Runs go through run.sh, which resolves the binary and writes log.txt into its own directory
# under results/$STUDY.

set -euo pipefail

module load nvhpc/26.3
cd "$SLURM_SUBMIT_DIR"

BUDGET=32
STUDY="hybrid_${SLURM_JOB_ID}"
echo "study directory: results/$STUDY"

for N in 32 64; do
    echo "=== N=$N serial reference ==="
    ./run.sh test_biharmonic_cpu_d.bin gmres mg "$N" "$STUDY/N${N}_serial" --verbose

    for t in 1 2 4 8 16 32; do
        for p in 1 2 4 8 16 32; do
            if [ $(( p * t )) -gt "$BUDGET" ]; then continue; fi

            echo "=== N=$N ranks=$p threads=$t ==="
            ./run.sh --np "$p" --omp "$t" test_biharmonic_mpi_omp_d.bin gmres mg "$N" \
                "$STUDY/N${N}_p${p}_t${t}" --verbose
        done
    done
done

echo "done: results/$STUDY"
