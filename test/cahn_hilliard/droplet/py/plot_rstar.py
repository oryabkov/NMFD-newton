#!/usr/bin/env python3
"""Stationary radius against the conserved mean, R_*(m).

Every run in sweep_landscape conserves m exactly, so its final radius is a measurement of a
stationary point of E(R) at that m.  Runs that settle give the stable branch directly; the
unstable branch is not reachable, but it is bracketed between the largest R_0 that collapsed
and the smallest R_0 that survived.

Both are roots of h(R) = A R^4 - m R + sigma/k [+ C gamma R^2], which exists only for
m > m_min and has the two branches meeting there in a saddle-node.

Data: sweep_landscape, selected by LANDSCAPE_GAMMA and LANDSCAPE_GRID.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_bifurcation import GAMMA, GRIDN, h_m, load, m_min_num, roots_m, sci, sides
from verify_sharp_interface import A_COEF, _bisect, sigma

# RSTAR_DELTA=1 adds the delta << 1 line, m = A R^3: all the excess mass sits in the droplet and
# the matrix stays at -1.  That is the note's m = 2 R^3 rewritten for a unit cube, A = 8 pi / 3.
DELTA = os.environ.get("RSTAR_DELTA", "") not in ("", "0")
# RSTAR_MMAX=2 draws the whole admissible range: m = 1 + mean(phi) with phi in [-1, 1].
MMAX = float(os.environ.get("RSTAR_MMAX", "0")) or None
OUT = (f"../figs/rstar_g{GAMMA:.1e}".replace("e-0", "e-")
       + ("_delta" if DELTA else "") + ("_full" if MMAX else "") + ".png")
C_SHARP, C_CORR, C_DELTA = "0.55", "#7570b3", "#d95f02"

plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "axes.titlesize": 16,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})


def roots(m, g, rmax):
    """roots_m of plot_bifurcation stops at R = 0.45, which is short of the m -> 2 end."""
    Rs = np.linspace(1e-4, rmax, 200001)
    v = h_m(Rs, m, g, False)
    idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    return [_bisect(lambda R: h_m(R, m, g, False), Rs[i], Rs[i + 1]) for i in idx]


def main():
    runs = load()
    if not runs:
        print("no landscape runs found")
        return
    g = runs[0]["gamma"]
    xi = np.sqrt(2.0 * g)
    ms = sorted({r["m"] for r in runs})

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    m_hi = MMAX if MMAX else ms[-1] * 1.08
    m_lo = 0.0 if MMAX else ms[0] * 0.92
    r_cap = 1.1 * (m_hi / A_COEF) ** (1.0 / 3.0)

    ax.axhspan(0.0, 3.0 * xi, color="0.85", alpha=0.6, zorder=0)
    ax.text(0.5 * (m_lo + m_hi) if not MMAX else 0.25 * m_hi, 1.5 * xi, r"$R<3\sqrt{2\gamma}$",
            ha="center", va="center", fontsize=12, color="0.35")

    mm = m_min_num(g, False)
    grid = np.linspace(mm * 1.0001, m_hi, 600)
    ax.plot(grid, [roots(m, g, r_cap)[-1] for m in grid], "-", color=C_CORR, lw=3.0, zorder=3,
            label=rf"резкая граница: $m_{{\min}}={mm:.3f}$")
    ax.plot(grid, [roots(m, g, r_cap)[0] for m in grid], "--", color=C_CORR, lw=3.0, zorder=3)
    ax.axvline(mm, color=C_CORR, lw=1.2, ls=":", zorder=2)

    if DELTA:
        md = np.linspace(max(m_lo, 0.0), m_hi, 600)
        ax.plot(md, (md / A_COEF) ** (1.0 / 3.0), "-", color=C_DELTA, lw=3.0, zorder=3,
                label=r"$\delta\ll1$: $m=\frac{8\pi}{3}R^{3}$")

    if MMAX:                                  # a sphere stops fitting in the cube at R = 1/2
        m_fit = A_COEF * 0.125
        ax.axhline(0.5, color="0.35", lw=2.0, ls="-.", zorder=2)
        ax.axvspan(m_fit, m_hi, color="0.85", alpha=0.6, zorder=0)
        ax.text(0.5 * (m_fit + m_hi), 0.5 * r_cap, "сфера\nне влезает", ha="center", va="center",
                fontsize=12, color="0.35")

    stable_m, stable_r, br_m, br_lo, br_hi = [], [], [], [], []
    for m in ms:
        below, above, settled = sides(runs, m)
        if settled:
            stable_m.append(m)
            stable_r.append(float(np.mean([r["Rend"] for r in settled])))
        if below and above:
            br_m.append(m)
            br_lo.append(max(below))
            br_hi.append(min(above))

    ax.plot(stable_m, stable_r, "o", color="k", ms=10, zorder=5, label=r"измеренный $R_*$")
    br_m, br_lo, br_hi = np.array(br_m), np.array(br_lo), np.array(br_hi)
    mid = 0.5 * (br_lo + br_hi)
    ax.errorbar(br_m, mid, yerr=[mid - br_lo, br_hi - mid], fmt="none", ecolor="k", elinewidth=2.4,
                capsize=7, capthick=2.4, zorder=5, label=r"измеренная вилка на $R_c$")

    ax.set_xlim(m_lo, m_hi)
    ax.set_ylim(0.0, r_cap if MMAX else max(stable_r) * 1.18)
    ax.set_xlabel(r"$m=1+\bar\phi$")
    ax.set_ylabel(r"$R_*$")
    ax.set_title(rf"$\gamma={sci(g)}$, ${GRIDN}^3$")
    ax.legend(loc="upper left", framealpha=0.94)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.5, zorder=1)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)

    print(f"\n{'m':>6} {'R_inf изм':>10} {'R_inf теория':>13} {'ошибка':>8} | "
          f"{'вилка R_c':>16} {'R_c теория':>11} {'R_c/xi':>7}")
    for m in ms:
        below, above, settled = sides(runs, m)
        rk = roots_m(m, g, False)
        meas = np.mean([r["Rend"] for r in settled]) if settled else float("nan")
        br = f"({max(below):.3f}, {min(above):.3f})" if below and above else "—"
        print(f"{m:6.3f} {meas:10.4f} {rk[-1] if rk else float('nan'):11.4f} "
              f"{(meas / rk[-1] - 1 if rk else float('nan')):7.2%} | {br:>16} "
              f"{rk[0] if rk else float('nan'):9.4f} {(rk[0] / xi if rk else float('nan')):7.2f}")


if __name__ == "__main__":
    main()
