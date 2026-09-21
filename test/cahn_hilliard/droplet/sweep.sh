#!/usr/bin/env bash
#
# Droplet-shrinkage parameter sweeps.
#
# Usage (from test/cahn_hilliard/droplet, on node208-24 after
#        `module load nvhpc/26.3 cuda/12.9`):
#
#   ./sweep.sh <sweep-name> [extra binary flags...]
#   ./sweep.sh list
#
# Each case becomes one ../data/<sweep>_<case>_<timestamp>/ directory containing log.txt,
# which py/parse_runs.py reads. Predictions come from py/predict.py.
#
# Workhorse configuration: unit box, 64^3, eps = 0.04 (about 2.6 cells of decay length,
# interface spanning ~10 cells), which puts the predicted critical radius at R_c ~ 0.21..0.29
# depending on the potential. Confirmation runs use GRID=128 with every gamma divided by 4
# (eps = 0.02); see notes/03_shrinkage_criterion.md.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

BIN=${BIN:-test_time_cahn_hilliard_cuda_d.bin}
RUN=../run.sh
SOLVER=${SOLVER:-gmres}
PRECOND=${PRECOND:-mg}
GRID=${GRID:-64}
DT=${DT:-2e-3}
STEPS=${STEPS:-1500}

# gamma = eps^2 * f''(phi_eq) at eps = 0.04 keeps the interface width matched across potentials
GAMMA_DW=3.200e-3     # R_c = 0.2865
GAMMA_W25=2.461e-3    # R_c = 0.2777
GAMMA_W30=7.373e-3    # R_c = 0.2686
GAMMA_W40=3.207e-2    # R_c = 0.2497
GAMMA_W60=3.050e-1    # R_c = 0.2109

R0_LIST=${R0_LIST:-"0.15 0.20 0.24 0.26 0.28 0.30 0.33"}

common() {
    echo "--init sphere --bc neumann --dt $DT --max-time-steps $STEPS"
}

one() {
    local tag="$1"; shift
    echo "=== ${SWEEP}_${tag}"
    $RUN "$BIN" "$SOLVER" "$PRECOND" "$GRID" "${SWEEP}_${tag}" "$@" ${EXTRA[@]+"${EXTRA[@]}"}
}

sweep_pilot() {
    STEPS=50 one "pilot" --init sphere --bc neumann --dt $DT --max-time-steps 50 \
        --r0 0.25 --gamma $GAMMA_DW --potential double_well --verbose
}

sweep_r0_dw() {
    for r0 in $R0_LIST; do
        one "r0${r0}" $(common) --r0 "$r0" --gamma $GAMMA_DW --potential double_well
    done
}

sweep_r0_log() {
    for r0 in $R0_LIST; do
        one "r0${r0}" $(common) --r0 "$r0" --gamma $GAMMA_W30 --potential logarithmic --omega 3
    done
}

sweep_omega() {
    # R0 grids bracket the *exact* (nonlinear) predictions of py/criterion.py at eps = 0.04:
    #   omega=2.5 -> R_c 0.310,  omega=3 -> 0.317,  omega=4 -> 0.352,  omega=6 -> 0.576.
    # The linearised Lambda formula predicts 0.278 / 0.269 / 0.250 / 0.211 instead, so these
    # grids discriminate between the two. For omega=6 the exact criterion says nothing survives
    # in a unit box at all.
    for spec in "2.5 $GAMMA_W25 0.26 0.29 0.31 0.33" \
                "3 $GAMMA_W30 0.31 0.32 0.34" \
                "4 $GAMMA_W40 0.28 0.32 0.34 0.36 0.38" \
                "6 $GAMMA_W60 0.30 0.36 0.42"; do
        set -- $spec
        local w=$1 g=$2; shift 2
        for r0 in "$@"; do
            one "w${w}_r0${r0}" $(common) --r0 "$r0" --gamma "$g" --potential logarithmic --omega "$w"
        done
    done
}

sweep_gamma() {
    # R_c ~ (eps V)^{1/4}: gamma /16 is eps /4 is R_c /1.41
    for g in 2.000e-4 8.000e-4 3.200e-3 1.280e-2; do
        for r0 in 0.16 0.20 0.24 0.28 0.32; do
            one "g${g}_r0${r0}" $(common) --r0 "$r0" --gamma "$g" --potential double_well
        done
    done
}

sweep_faceavg() {
    # r0 = 0.26 sits just below R_c = 0.2865: the drop is still doomed thermodynamically but
    # lingers long enough for a rate difference to be measurable. r0 = 0.20 was useless here --
    # it dies inside the initial profile relaxation, where M ~ D for every averaging rule.
    local r0=${FACEAVG_R0:-0.26}
    for rule in midpoint arithmetic geometric harmonic; do
        for floor in 1e-2 1e-3 1e-4; do
            one "${rule}_f${floor}" $(common) --r0 "$r0" --gamma $GAMMA_DW \
                --potential double_well --mobility parabolic \
                --mobility-floor "$floor" --face-avg "$rule"
        done
    done
    one "constant_ref" $(common) --r0 "$r0" --gamma $GAMMA_DW --potential double_well
}

sweep_frontier() {
    # Запирание задаётся значением на грани, а не порогом: у гармонического оно ~2f,
    # у геометрического ~sqrt(f). Проходим по порогам так, чтобы сравнить два правила
    # при СОПОСТАВИМОМ запирании и посмотреть, какое из них решатель переносит лучше:
    # геометрическое меняется по пространству плавно, гармоническое -- скачком по минимуму.
    for f in 1e-2 1e-3 1e-4 1e-5 1e-6 1e-8; do
        one "geom_f${f}" $(common) --r0 0.26 --gamma $GAMMA_DW --potential double_well \
            --mobility parabolic --mobility-floor "$f" --face-avg geometric
    done
    for f in 1e-4 1e-5; do
        one "harm_f${f}" $(common) --r0 0.26 --gamma $GAMMA_DW --potential double_well \
            --mobility parabolic --mobility-floor "$f" --face-avg harmonic
    done
}

sweep_solverprobe() {
    # Гармоническое среднее при floor=1e-4 не сходится: 80 % шагов упираются в потолок
    # итераций Ньютона. Проверяем, в чём причина -- в слишком большом шаге по времени
    # (сильная нелинейность на шаге) или в самом перепаде коэффициента.
    one "constant_ref" $(common) --r0 0.26 --gamma $GAMMA_DW --potential double_well
    for dt in 2e-3 5e-4 1e-4; do
        local n
        n=$(awk -v d="$dt" 'BEGIN{printf "%d", 0.30/d}')
        one "harm1e-4_dt${dt}" --init sphere --bc neumann --dt "$dt" --max-time-steps "$n" \
            --r0 0.26 --gamma $GAMMA_DW --potential double_well --mobility parabolic \
            --mobility-floor 1e-4 --face-avg harmonic
    done
    for sw in 4 8; do
        one "harm1e-4_sweeps${sw}" $(common) --r0 0.26 --gamma $GAMMA_DW \
            --potential double_well --mobility parabolic --mobility-floor 1e-4 \
            --face-avg harmonic --mg-sweeps-pre "$sw" --mg-sweeps-post "$sw" \
            --max-time-steps 150
    done
}

sweep_bc() {
    for bc in neumann dirichlet periodic; do
        one "$bc" --init sphere --dt $DT --max-time-steps $STEPS \
            --r0 0.30 --gamma $GAMMA_DW --potential double_well --bc "$bc"
    done
}

sweep_resolution() {
    # The plateau is reached by t ~ 0.3, so 500 steps at dt = 2e-3 is ample for the grid study.
    # 256^3 costs ~64x a 64^3 step; it is opt-in via RESOLUTION_MAX_GRID=256.
    local grids="64 128"
    [[ "${RESOLUTION_MAX_GRID:-128}" == "256" ]] && grids="64 128 256"
    for g in $grids; do
        GRID=$g one "grid${g}" --init sphere --bc neumann --dt 2e-3 --max-time-steps 500 \
            --r0 0.30 --gamma $GAMMA_DW --potential double_well
    done
    # dt study at a fixed end time t_end = 1.2
    for dt in 5e-4 1e-3 2e-3 4e-3; do
        local n
        n=$(awk -v d="$dt" 'BEGIN{printf "%d", 1.2/d}')
        one "dt${dt}" --init sphere --bc neumann --dt "$dt" \
            --max-time-steps "$n" --r0 0.30 --gamma $GAMMA_DW --potential double_well
    done
}

sweep_cube() {
    for r0 in 0.20 0.24 0.28 0.32; do
        one "cube_r0${r0}" --init cube --bc neumann --dt $DT --max-time-steps $STEPS \
            --r0 "$r0" --gamma $GAMMA_DW --potential double_well
        one "sphere_r0${r0}" $(common) --r0 "$r0" --gamma $GAMMA_DW --potential double_well
    done
}

sweep_eta() {
    # Smoothed obstacle, f = a(eta) (sqrt((1-phi^2)^2 + eta^4) - eta^2) with a(eta) pinned so the
    # barrier is 1/4 for every eta. Same gamma, grid and dt as sweep_r0_dw, so the double well's
    # measured bracket (0.28, 0.30) is the control. Exact criterion (py/potential_zoo.py) predicts
    #   eta = 1 -> 0.295,  0.5 -> 0.253,  0.3 -> 0.215,  0.2 -> 0.189,
    #   0.15  -> 0.174,  0.1 -> 0.158,  0.05 -> 0.140.
    # Each grid brackets its prediction and adds one point well above it.
    for spec in "1    0.26 0.28 0.30 0.33" \
                "0.5  0.22 0.24 0.26 0.30" \
                "0.3  0.18 0.20 0.22 0.26" \
                "0.2  0.16 0.18 0.20 0.24" \
                "0.15 0.14 0.16 0.18 0.22" \
                "0.1  0.12 0.15 0.17 0.20" \
                "0.05 0.10 0.13 0.15 0.18"; do
        set -- $spec
        local e=$1; shift
        for r0 in "$@"; do
            one "e${e}_r0${r0}" $(common) --r0 "$r0" --gamma $GAMMA_DW \
                --potential smoothed_obstacle --eta "$e"
        done
    done
}

sweep_etaprobe() {
    # Short solver probe: one radius safely above every prediction, eta walking down until the
    # Newton/MG solve gives out. Conditioning degrades as eta^-2, so this is the run that sets the
    # usable eta, not sweep_eta.
    for e in 1 0.5 0.3 0.2 0.15 0.1 0.05 0.02 0.01; do
        STEPS=100 one "e${e}" --init sphere --bc neumann --dt $DT --max-time-steps 100 \
            --r0 0.30 --gamma $GAMMA_DW --potential smoothed_obstacle --eta "$e" --verbose
    done
}

if [[ $# -lt 1 || "$1" == "list" ]]; then
    echo "sweeps: pilot r0_dw r0_log omega gamma faceavg frontier solverprobe bc resolution cube eta etaprobe"
    exit 0
fi

SWEEP="$1"; shift
EXTRA=("$@")

if ! declare -F "sweep_${SWEEP}" > /dev/null; then
    echo "unknown sweep '${SWEEP}'; try: $0 list" >&2
    exit 1
fi

echo "sweep=${SWEEP} grid=${GRID} dt=${DT} steps=${STEPS} solver=${SOLVER}/${PRECOND}"
"sweep_${SWEEP}"
echo "sweep ${SWEEP} done"
