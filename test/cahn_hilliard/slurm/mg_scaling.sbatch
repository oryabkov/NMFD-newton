#!/bin/bash
#SBATCH --job-name=mg_scaling
#SBATCH --partition=AMD7763-A100-CCP
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=01:00:00
#SBATCH --output=mg_scaling_%j.out

# Multigrid strong-scaling check for the biharmonic test problem: the serial binary is the
# reference, the MPI binary is run over a range of process counts on the same grid. Every run
# goes through run.sh, which resolves the binary and writes log.txt into its own directory
# under results/$STUDY.

set -euo pipefail

module load nvhpc/26.3
cd "$SLURM_SUBMIT_DIR"

STUDY="mg_scaling_${SLURM_JOB_ID}"
echo "study directory: results/$STUDY"

# Built here rather than on the login node, which has no libatomic for nvc++.
make -B test_biharmonic PLATFORM=cpu > "build_test_biharmonic_cpu.log" 2>&1 \
    || { echo "test_biharmonic BUILD FAILED"; tail -5 build_test_biharmonic_cpu.log; exit 1; }
make -B test_biharmonic_mpi PLATFORM=cpu > "build_test_biharmonic_mpi_cpu.log" 2>&1 \
    || { echo "test_biharmonic_mpi BUILD FAILED"; tail -5 build_test_biharmonic_mpi_cpu.log; exit 1; }

for N in 16 32; do
    echo "=== N=$N serial reference ==="
    ./run.sh test_biharmonic_cpu_d.bin gmres mg "$N" "$STUDY/N${N}_serial" --save-coords --verbose

    for np in 1 2 8 16 32; do
        echo "=== N=$N np=$np ==="
        # Coordinates are only dumped on a single rank; the solution is not gathered otherwise.
        coords=""
        if [ "$np" -eq 1 ]; then coords="--save-coords"; fi

        ./run.sh --np "$np" test_biharmonic_mpi_cpu_d.bin gmres mg "$N" "$STUDY/N${N}_np${np}" $coords --verbose
    done
done

echo "done: results/$STUDY"
