#!/usr/bin/env python3
"""The energy landscape E(R), measured.

Along one run the mean value is conserved, so the trajectory sweeps R at fixed m -- which is
exactly the coordinate the landscape is a function of.  Plotting the logged phobic + philic
against R_eff therefore *is* a measurement of E(R), with no extra runs needed.

The overlaid curve keeps the bulk thermodynamics exact (the quadratic expansion of f about -1
is off by 10 % already at m = 0.27) and carries the O(gamma) interface mass, C gamma R:

    x = v(R)/V,   x phi_in + (1-x) phi_out + C gamma R = phibar,   f'(phi_in) = f'(phi_out) = psi
    E(R) = 4 pi sigma R^2 + x f(phi_in) + (1-x) f(phi_out)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify_sharp_interface import A_COEF, C_COEF, K_DW, load_all, roots_of, sigma

OUT = "../figs/landscape.png"
SKIP = 10                      # the first steps relax the initial tanh, not the radius
COLS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#1f78b4", "#a6761d"]

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25})

f_bulk = np.vectorize(lambda p: 0.25 * (p * p - 1.0) ** 2)


def branches(psi):
    r = np.roots([1.0, 0.0, -1.0, -psi])
    r = np.sort(np.real(r[np.abs(r.imag) < 1e-12]))
    return r[0], r[-1]


def E_exact(R, phibar, g, corr=True):
    x = A_COEF * R**3 / 2.0
    shift = C_COEF * g * R if corr else 0.0
    lo, hi = -0.45, 0.45
    for _ in range(80):
        psi = 0.5 * (lo + hi)
        po, pi = branches(psi)
        if x * pi + (1 - x) * po + shift > phibar:
            hi = psi
        else:
            lo = psi
    po, pi = branches(0.5 * (lo + hi))
    return sigma(g) * 4 * np.pi * R**2 + x * f_bulk(pi) + (1 - x) * f_bulk(po)


def E_quad(R, phibar, g, corr=True):
    m = 1.0 + phibar
    d = m - A_COEF * R**3 - (C_COEF * g * R if corr else 0.0)
    return sigma(g) * 4 * np.pi * R**2 + 0.5 * K_DW * d**2


def runs_at(gamma, r0s):
    out = []
    for cfg, d in load_all():
        if cfg["gamma"] != gamma or cfg["grid"] != 64 or cfg["dt"] != 2e-3:
            continue
        if round(cfg["r0"], 3) in r0s and not any(c["r0"] == cfg["r0"] for c, _ in out):
            out.append((cfg, d))
    return sorted(out, key=lambda cd: cd[0]["r0"])


def draw(ax, axr, gamma, r0s, title):
    for n, (cfg, d) in enumerate(runs_at(gamma, r0s)):
        col = COLS[n % len(COLS)]
        pb = d["mass"][0]
        R = d["R_eff"][SKIP:]
        F = (d["phobic"] + d["philic"])[SKIP:]
        ok = R > 2.0 * np.sqrt(2 * gamma)
        R, F = R[ok], F[ok]
        if len(R) < 3:
            continue

        rr = np.linspace(R.min() * 0.92, max(R.max(), cfg["r0"]) * 1.02, 400)
        Eth = np.array([E_exact(x, pb, gamma) for x in rr])
        ax.plot(rr, Eth, "-", color=col, lw=1.7, zorder=3)
        ax.plot(rr, [E_quad(x, pb, gamma) for x in rr], ":", color=col, lw=1.1, zorder=2)

        step = max(1, len(R) // 45)
        alive = d["drop_volume"][-1] > 0
        ax.plot(R[::step], F[::step], "o", color=col, ms=3.6, mfc="none", mew=0.9, zorder=5,
                label=rf"$R_0={cfg['r0']:.2f}$, $m={1+pb:.3f}$ — "
                      + ("выжила" if alive else "испарилась"))

        rts = roots_of(gamma, cfg["r0"])
        if rts:
            ax.plot([rts[-1]], [E_exact(rts[-1], pb, gamma)], "*", color=col, ms=13,
                    zorder=6, mec="0.2", mew=0.5)

        Ei = np.array([E_exact(x, pb, gamma) for x in R])
        axr.plot(R[::step], (F[::step] - Ei[::step]) / Ei[::step], "-", color=col, lw=1.3)

    ax.plot([], [], "*", color="0.35", ms=13, mec="0.2", mew=0.5, label=r"предсказанный минимум $R_\infty$")
    ax.set_ylabel(r"$F/V$")
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(labelbottom=False)
    ax.legend(fontsize=8.1, loc="upper left", framealpha=0.92)

    axr.axhline(0.0, color="0.7", lw=0.6)
    axr.set_xlabel(r"$R$")
    axr.set_ylabel("отн. невязка")
    axr.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))


def main():
    fig = plt.figure(figsize=(13.4, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.1], hspace=0.07, wspace=0.16,
                          left=0.06, right=0.995, top=0.895, bottom=0.075)
    a, ar = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    b, br = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    draw(a, ar, 3.2e-3, {0.24, 0.26, 0.28, 0.30, 0.33},
         r"$\gamma=3{,}2\cdot10^{-3}$  ($\xi=0{,}080$, $\xi/R\approx0{,}3$)")
    draw(b, br, 8.0e-4, {0.16, 0.20, 0.24, 0.28, 0.32},
         r"$\gamma=8\cdot10^{-4}$  ($\xi=0{,}040$, $\xi/R\approx0{,}15$)")

    fig.suptitle(
        "Энергетический ландшафт, измеренный: вдоль одного расчёта масса сохраняется, поэтому "
        r"траектория и есть $E(R)$ при фиксированном $m$." "\n"
        r"Точки — $(\mathrm{phobic}+\mathrm{philic})$ против $R_{\rm eff}$, сплошная — теория с точной "
        r"объёмной термодинамикой, пунктир — с квадратичной. Капля катится справа налево.",
        fontsize=10.3, y=0.978)

    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
