#!/usr/bin/env python3
"""Measured survival threshold of the smoothed obstacle against the exact criterion.

Data: sweep_eta (slurm job 96523), 28 runs, 64^3, gamma = 3.2e-3, Neumann, dt = 2e-3.
Each entry is (eta, largest R0 that evaporated, smallest R0 that survived).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from potential_zoo import DIM, Candidate, R_c, dw_pair, obstacle_pair, smooth_obstacle_pair

GAMMA = 3.2e-3
OUT = "../figs/eta_threshold.png"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

# eta, R0 that evaporated, R0 that survived
MEASURED = [
    (1.00, 0.26, 0.28),
    (0.50, 0.22, 0.24),
    (0.30, 0.18, 0.20),
    (0.20, 0.16, 0.18),
    (0.15, 0.14, 0.16),
    (0.10, 0.12, 0.15),
    (0.05, 0.13, 0.15),
]
DW_DEAD, DW_ALIVE = 0.28, 0.30


def theory_curve(etas):
    out = []
    for e in etas:
        f, df = smooth_obstacle_pair(e)
        c = Candidate("", f, df, 1.0, 4.0)
        out.append(R_c(c, GAMMA))
    return np.array(out)


def main():
    f, df = dw_pair(2.0)
    dw = Candidate("", f, df, 1.0, 4.0)
    rc_dw = R_c(dw, GAMMA)

    # eta -> 0 limit: the double obstacle, whose threshold is set by the spinodal ceiling alone.
    fo, dfo = obstacle_pair(1.0)
    obs = Candidate("", fo, dfo, 1.0, 1.0, obstacle=True)
    floor = (DIM - 1) * obs.sigma_hat() * np.sqrt(GAMMA) / (2.0 * obs.spinodal()[1])

    etas = np.array([1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03])
    rc_th = theory_curve(etas)

    fig, ax = plt.subplots(figsize=(8.6, 5.9))

    ax.axhspan(DW_DEAD, DW_ALIVE, color="0.75", alpha=0.45, zorder=1)
    ax.axhline(rc_dw, color="0.35", lw=2.0, ls="-", zorder=2,
               label=f"полиномиальный: теория {rc_dw:.3f}, измерено ({DW_DEAD}, {DW_ALIVE})")
    ax.axhline(floor, color="#7570b3", lw=1.6, ls=":", zorder=2,
               label=rf"предел семейства (двойное препятствие): {floor:.3f}")

    ax.plot(etas, rc_th, color="#e7298a", lw=2.2, zorder=3,
            label="сглаженное препятствие: точный критерий")

    e = np.array([m[0] for m in MEASURED])
    dead = np.array([m[1] for m in MEASURED])
    alive = np.array([m[2] for m in MEASURED])
    mid = 0.5 * (dead + alive)
    ax.errorbar(e, mid, yerr=[mid - dead, alive - mid], fmt="o", ms=6,
                color="#1f78b4", ecolor="#1f78b4", elinewidth=1.8, capsize=4, zorder=4,
                label="измеренная вилка (испарилась / выжила)")

    ax.set_xscale("log")
    ax.set_xticks([0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(0.026, 1.15)
    ax.set_ylim(0.105, 0.345)
    ax.set_xlabel(r"параметр сглаживания $\eta$")
    ax.set_ylabel(r"пороговый радиус $R_c$")
    ax.set_title("Порог выживания капли: сглаженное препятствие против полиномиального потенциала\n"
                 r"куб $1\times1\times1$, $64^3$, $\gamma=3{,}2\cdot10^{-3}$, стенки Неймана",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)

    print(f"\n{'eta':>6} {'вилка':>14} {'середина':>9} {'теория':>8} {'отн. изм.':>10} {'отн. теор.':>11}")
    print(f"{'—':>6} {f'({DW_DEAD}, {DW_ALIVE})':>14} {0.29:9.3f} {rc_dw:8.3f} {1.0:10.3f} {1.0:11.3f}")
    for (ee, d, a), th in zip(MEASURED, theory_curve([m[0] for m in MEASURED])):
        m = 0.5 * (d + a)
        print(f"{ee:6.2f} {f'({d}, {a})':>14} {m:9.3f} {th:8.3f} {m / 0.29:10.3f} {th / rc_dw:11.3f}")


if __name__ == "__main__":
    main()
