#!/usr/bin/env python3
"""Radial profile of a stationary droplet, numerical against the asymptotic solution

    phi(r) = -tanh(zeta) + (sigma / 2R) tanh^2(zeta),     zeta = (r - R) / sqrt(2 gamma)

Left column: the stationary radius held at R = 0.28 while gamma varies, so the interface
smears at a fixed droplet size.  Right column: gamma held at 3.2e-3 while R varies, so the
curvature correction sigma/2R is what changes.  Each column carries a residual strip.

Data: sweep_profile, runs under ../data/profile_R<R>_g<gamma>_*/solution/numerical_final.bin.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from field_io import radial_profile, read_field, zero_crossing
from verify_sharp_interface import parse, sigma

OUT = "../figs/profile.png"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

PANEL_A = [(0.28, 2.000e-4, "#1b9e77"),
           (0.28, 8.000e-4, "#d95f02"),
           (0.28, 3.200e-3, "#7570b3"),
           (0.28, 1.280e-2, "#e7298a")]
PANEL_B = [(0.22, 3.200e-3, "#1b9e77"),
           (0.25, 3.200e-3, "#d95f02"),
           (0.28, 3.200e-3, "#7570b3"),
           (0.31, 3.200e-3, "#e7298a")]

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25})


def analytic(r, R, gamma, corrected=True):
    z = (r - R) / np.sqrt(2.0 * gamma)
    out = -np.tanh(z)
    if corrected:
        out = out + sigma(gamma) / (2.0 * R) * np.tanh(z) ** 2
    return out


def load(target_R, gamma):
    pat = os.path.join(DATA, f"profile_R{target_R}_g{gamma:.3e}".replace("e-0", "e-") + "_*")
    dirs = sorted(glob.glob(pat))
    if not dirs:
        return None
    run = dirs[-1]
    fld = os.path.join(run, "solution", "numerical_final.bin")
    if not os.path.exists(fld):
        return None
    cfg, data = parse(run)
    r, mean, lo, hi = radial_profile(read_field(fld)["phi"])
    return {"r": r, "phi": mean, "lo": lo, "hi": hi,
            "R": zero_crossing(r, mean), "R_eff": data["R_eff"][-1],
            "gamma": cfg["gamma"], "r0": cfg["r0"], "grid": int(cfg["grid"]),
            "steps": int(data["step"][-1])}


def draw(ax, axr, cases, label_of, title):
    for target_R, gamma, col in cases:
        c = load(target_R, gamma)
        if c is None:
            print(f"missing: R={target_R} gamma={gamma:.3e}")
            continue
        R = c["R"]
        rr = np.linspace(0.0, 0.5, 1200)

        ax.fill_between(c["r"], c["lo"], c["hi"], color=col, alpha=0.18, lw=0)
        step = max(1, len(c["r"]) // 70)
        ax.plot(c["r"][::step], c["phi"][::step], "o", color=col, ms=3.4, mfc="none", mew=0.9,
                zorder=4, label=label_of(c, target_R, gamma))
        ax.plot(rr, analytic(rr, R, gamma), "-", color=col, lw=1.7, zorder=3)
        ax.plot(rr, analytic(rr, R, gamma, corrected=False), ":", color=col, lw=1.2, zorder=2)

        axr.plot(c["r"], c["phi"] - analytic(c["r"], R, gamma), "-", color=col, lw=1.4)
        axr.plot(c["r"], c["phi"] - analytic(c["r"], R, gamma, corrected=False), ":", color=col, lw=1.0)

    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(-1.2, 1.25)
    ax.set_ylabel(r"$\phi$")
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(labelbottom=False)
    ax.legend(fontsize=8.2, loc="lower left", framealpha=0.9)

    axr.axhline(0.0, color="0.7", lw=0.6)
    axr.set_xlim(0.0, 0.5)
    axr.set_xlabel(r"$r$")
    axr.set_ylabel("невязка")


def main():
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.15], hspace=0.07, wspace=0.17,
                          left=0.055, right=0.995, top=0.90, bottom=0.075)
    a, ar = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    b, br = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    draw(a, ar, PANEL_A,
         lambda c, R, g: rf"$\gamma={g:.1e}$".replace("e-0", r"\cdot10^{-") + "}$"
                         + rf",  $\xi/R={np.sqrt(2*g)/c['R']:.2f}$",
         r"A. Фиксированный $R\simeq0.28$, разные $\gamma$")
    draw(b, br, PANEL_B,
         lambda c, R, g: rf"$R={c['R']:.3f}$,  $\sigma/2R={sigma(g)/(2*c['R']):.3f}$",
         r"B. Фиксированная $\gamma=3{,}2\cdot10^{-3}$, разные $R$")

    fig.suptitle(
        r"Радиальный профиль стационарной капли: точки — расчёт ($128^3$, сферическое среднее, "
        r"полоса — разброс по слою), сплошная — $-\tanh\zeta+\frac{\sigma}{2R}\tanh^2\zeta$, "
        r"пунктир — $-\tanh\zeta$",
        fontsize=10.5, y=0.975)

    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
