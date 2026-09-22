#!/usr/bin/env python3
"""Four panels of the sharp-interface theory against the runs on disk."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify_sharp_interface import (A_COEF, K_DW, load_all, m_of_r0, r0_crit,
                                    roots_of, sigma)

OUT = "../figs/sharp_interface.png"
GAMMAS = (2.0e-4, 8.0e-4, 3.2e-3, 1.28e-2)
GDW = 3.2e-3
COL = {2.0e-4: "#1b9e77", 8.0e-4: "#d95f02", 3.2e-3: "#7570b3", 1.28e-2: "#e7298a"}

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25})

f_bulk = lambda p: 0.25 * (p * p - 1.0) ** 2


def E_quad(R, m, g):
    return sigma(g) * 4 * np.pi * R**2 + 0.5 * K_DW * (m - A_COEF * R**3) ** 2


def brackets(runs):
    out = {}
    for cfg, d in runs:
        if cfg["grid"] != 64 or cfg["dt"] != 2e-3:
            continue
        out.setdefault(cfg["gamma"], set()).add((cfg["r0"], d["drop_volume"][-1] > 0))
    return out


def main():
    runs = load_all()
    br = brackets(runs)

    fig, ax = plt.subplots(2, 2, figsize=(12.4, 9.2))
    (a, b), (c, e) = ax

    # ---- A: threshold vs gamma --------------------------------------------------
    gg = np.logspace(np.log10(1e-4), np.log10(2e-2), 200)
    a.plot(gg, [r0_crit(g) for g in gg], "-", color="0.25", lw=2.2,
           label=r"резкая граница: $R_0^4=0.5053\,\sigma V/(2\phi_{eq}^2 f'')$")
    a.plot(gg, [r0_crit(g, True) for g in gg], "--", color="0.55", lw=1.6,
           label=r"с поправкой $O(\gamma)$ на ширину границы")
    for g in sorted(br):
        dead = [r for r, al in br[g] if not al]
        live = [r for r, al in br[g] if al]
        lo, hi = (max(dead) if dead else np.nan), (min(live) if live else np.nan)
        if not np.isnan(hi):
            a.plot([g, g], [lo, hi], color=COL[g], lw=6, alpha=0.35, solid_capstyle="butt")
        a.plot([g], [lo], "v", color=COL[g], ms=7)
        if not np.isnan(hi):
            a.plot([g], [hi], "^", color=COL[g], ms=7)
    a.plot([], [], "v", color="0.3", ms=7, label="испарилась")
    a.plot([], [], "^", color="0.3", ms=7, label="выжила")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel(r"$\gamma$")
    a.set_ylabel(r"начальный радиус $R_0$")
    a.set_title(r"A. Порог выживания: $R_{0,\mathrm{крит}}\propto\gamma^{1/8}$" "\n"
                r"наблюдённые вилки, куб $1^3$, $64^3$", fontsize=10)
    a.legend(fontsize=8, loc="upper left")

    # ---- B: saddle node at gamma = 3.2e-3 ---------------------------------------
    r0s = np.linspace(0.281, 0.345, 400)
    for corr, ls, lab in ((False, "-", "резкая граница"), (True, "--", r"с поправкой $O(\gamma)$")):
        y = []
        for r0 in r0s:
            rr = roots_of(GDW, r0, corr)
            y.append(rr[-1] if rr else np.nan)
        b.plot(r0s, y, ls, color="0.25" if not corr else "0.55",
               lw=2.2 if not corr else 1.6, label=lab)
    rc = r0_crit(GDW)
    b.axvline(rc, color="#7570b3", lw=1.2, ls=":")
    b.text(rc + 0.0015, 0.175, rf"$R_{{0,\mathrm{{крит}}}}={rc:.4f}$", color="#7570b3", fontsize=8.5)
    seen = set()
    for cfg, d in runs:
        if cfg["gamma"] != GDW or cfg["grid"] != 64 or cfg["r0"] in seen:
            continue
        seen.add(cfg["r0"])
        alive = d["drop_volume"][-1] > 0
        b.plot([cfg["r0"]], [d["R_eff"][-1] if alive else 0.145],
               "o" if alive else "v", color="#1f78b4" if alive else "0.4", ms=7, zorder=5)
    b.plot([], [], "o", color="#1f78b4", ms=7, label="измеренный $R_\\infty$")
    b.plot([], [], "v", color="0.4", ms=7, label="испарилась (отложено внизу)")
    b.set_xlim(0.274, 0.345)
    b.set_ylim(0.138, 0.325)
    b.set_xlabel(r"$R_0$")
    b.set_ylabel(r"стационарный радиус $R_\infty$")
    b.set_title("B. Седло-узловая бифуркация, " r"$\gamma=3{,}2\cdot10^{-3}$" "\n"
                r"$R_\infty-R_f\simeq R_0\sqrt{\Delta R_0/2R_f}$", fontsize=10)
    b.legend(fontsize=8, loc="lower right")

    # ---- C: bulk shift to second order ------------------------------------------
    for cfg, d in runs:
        if d["drop_volume"][-1] <= 0 or cfg["grid"] != 64:
            continue
        g, R = cfg["gamma"], d["R_eff"][-1]
        psi = sigma(g) / R
        c.plot([psi], [d["phi_min"][-1] + 1.0], "o", color=COL[g], ms=7, zorder=4)
        if R / np.sqrt(g / K_DW) > 10:
            c.plot([psi], [d["phi_max"][-1] - 1.0], "s", color=COL[g], ms=6, mfc="none", zorder=4)
    p = np.linspace(0, 0.22, 100)
    c.plot(p, p / 2, "-", color="0.6", lw=1.6, label=r"1-й порядок: $\psi/2$")
    c.plot(p, p / 2 + 0.375 * p**2, "-", color="0.15", lw=2.0,
           label=r"2-й порядок: $\psi/2+\frac{3}{8}\psi^2$ (матрица)")
    c.plot(p, p / 2 - 0.375 * p**2, "--", color="0.15", lw=2.0,
           label=r"2-й порядок: $\psi/2-\frac{3}{8}\psi^2$ (капля)")
    c.plot([], [], "o", color="0.3", ms=7, label=r"измерено: $\phi_{min}+1$")
    c.plot([], [], "s", color="0.3", ms=6, mfc="none", label=r"измерено: $\phi_{max}-1$")
    c.set_xlabel(r"$\psi=\sigma/R$")
    c.set_ylabel(r"сдвиг объёмной фазы $\delta_\pm$")
    c.set_title("C. Сдвиг обеих объёмных фаз: второй порядок" "\n"
                r"разводит каплю и матрицу, $\delta_-+\delta_+=\psi$", fontsize=10)
    c.legend(fontsize=8, loc="upper left")

    # ---- D: the landscape --------------------------------------------------------
    Rs = np.linspace(0.0, 0.34, 600)
    for r0, col in ((0.284, "#d95f02"), (0.300, "#1f78b4"), (0.330, "#1b9e77")):
        m = m_of_r0(GDW, r0)
        e.plot(Rs, E_quad(Rs, m, GDW) - 0.5 * K_DW * m**2, "-", color=col, lw=2.0,
               label=rf"$R_0={r0:.3f}$  ($m={m:.3f}$)")
        rr = roots_of(GDW, r0)
        if rr:
            e.plot([rr[0]], [E_quad(rr[0], m, GDW) - 0.5 * K_DW * m**2], "x", color=col, ms=9, mew=2)
            e.plot([rr[-1]], [E_quad(rr[-1], m, GDW) - 0.5 * K_DW * m**2], "o", color=col, ms=7)
    e.axhline(0.0, color="0.5", lw=1.0)
    e.text(0.005, -0.0022, r"однородное состояние $E(0)$", fontsize=8, color="0.35")
    e.plot([], [], "x", color="0.3", ms=9, mew=2, label=r"$R_c$ — критический зародыш")
    e.plot([], [], "o", color="0.3", ms=7, label=r"$R_\infty$ — устойчивая капля")
    e.set_xlabel(r"$R$")
    e.set_ylabel(r"$E(R)-E(0)$")
    e.set_ylim(-0.012, 0.022)
    e.set_title("D. Энергетический ландшафт, " r"$\gamma=3{,}2\cdot10^{-3}$" "\n"
                r"барьер $R_c$ недостижим: наше НУ даёт $h(R_0)=\sigma/k>0$", fontsize=10)
    e.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
