#!/usr/bin/env python3
"""The (R, m) landscape, measured.

Data: sweep_landscape -- a grid of runs with the conserved mean set independently of the droplet
radius (--phi-mean).  Because m is conserved exactly, every run is a horizontal line in the (R, m)
plane, and the theory's two branches predict which way it moves:

    R < R_c(m)              collapse to zero
    R_c(m) < R < R_inf(m)   growth to R_inf
    R > R_inf(m)            shrinkage back to R_inf

Left panel is that phase portrait.  Right panel samples E(R) directly: the energy logged at step 0
is the landscape evaluated at the initial radius, so the grid measures the barrier itself -- the
part no trajectory can reach.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify_sharp_interface import (A_COEF, C_COEF, K_DW, _bisect, h, m_min,
                                    parse, sigma)

OUT = "../figs/bifurcation.png"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

C_DEAD, C_GROW, C_SHRINK = "#7f7f7f", "#1b9e77", "#d95f02"

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25})

f_bulk = np.vectorize(lambda p: 0.25 * (p * p - 1.0) ** 2)


def roots_m(m, g):
    Rs = np.linspace(1e-4, 0.45, 200001)
    v = h(Rs, m, g)
    idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    return [_bisect(lambda R: h(R, m, g), Rs[i], Rs[i + 1]) for i in idx]


def branches(psi):
    r = np.roots([1.0, 0.0, -1.0, -psi])
    r = np.sort(np.real(r[np.abs(r.imag) < 1e-12]))
    return r[0], r[-1]


def E_exact(R, phibar, g):
    x = A_COEF * R**3 / 2.0
    shift = C_COEF * g * R
    lo, hi = -0.6, 0.6
    for _ in range(80):
        psi = 0.5 * (lo + hi)
        po, pi = branches(psi)
        if x * pi + (1 - x) * po + shift > phibar:
            hi = psi
        else:
            lo = psi
    po, pi = branches(0.5 * (lo + hi))
    return sigma(g) * 4 * np.pi * R**2 + x * f_bulk(pi) + (1 - x) * f_bulk(po)


def load():
    runs = []
    for d in sorted(glob.glob(os.path.join(DATA, "landscape_m*_R*_*"))):
        got = parse(d)
        if got is None:
            continue
        cfg, data = got
        m = 1.0 + cfg["phi_mean"]
        R0, Rend = data["R_eff"][0], data["R_eff"][-1]
        if Rend <= 0.0:
            kind = "dead"
        elif Rend > R0 * 1.02:
            kind = "grow"
        else:
            kind = "shrink"
        runs.append({"m": m, "R0": R0, "Rend": Rend, "kind": kind, "gamma": cfg["gamma"],
                     "E0": data["phobic"][0] + data["philic"][0], "r0_nom": cfg["r0"],
                     "steps": int(data["step"][-1])})
    return runs


def main():
    runs = load()
    if not runs:
        print("no landscape runs found")
        return
    g = runs[0]["gamma"]
    print(f"{len(runs)} runs at gamma = {g:.3e};  "
          f"dead {sum(r['kind']=='dead' for r in runs)}, "
          f"grow {sum(r['kind']=='grow' for r in runs)}, "
          f"shrink {sum(r['kind']=='shrink' for r in runs)}")

    fig = plt.figure(figsize=(13.6, 6.6))
    gs = fig.add_gridspec(1, 2, wspace=0.2, left=0.055, right=0.99, top=0.87, bottom=0.095)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- phase portrait -----------------------------------------------------------------
    mm = m_min(g)
    ms = np.linspace(mm * 1.00001, 0.40, 400)
    rc, ri = [], []
    for m in ms:
        r = roots_m(m, g)
        rc.append(r[0] if r else np.nan)
        ri.append(r[-1] if r else np.nan)
    a.plot(rc, ms, "--", color="#7570b3", lw=2.0, label=r"теория: $R_c(m)$ — барьер (неустойчив)")
    a.plot(ri, ms, "-", color="#7570b3", lw=2.4, label=r"теория: $R_\infty(m)$ — устойчивая капля")
    a.axhline(mm, color="0.45", lw=1.2, ls=":")
    a.axhspan(0.0, mm, color="0.85", alpha=0.5, zorder=0)
    a.text(0.345, mm - 0.008, rf"$m_{{\min}}={mm:.3f}$ — ниже капель нет",
           fontsize=8.5, color="0.35", ha="right", va="top")

    for r in runs:
        col = {"dead": C_DEAD, "grow": C_GROW, "shrink": C_SHRINK}[r["kind"]]
        end = max(r["Rend"], 0.004)
        a.annotate("", xy=(end, r["m"]), xytext=(r["R0"], r["m"]),
                   arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, alpha=0.85,
                                   shrinkA=0, shrinkB=0), zorder=4)
        a.plot([r["R0"]], [r["m"]], "o", color=col, ms=3.2, zorder=5)

    for lab, col in (("испарилась", C_DEAD), ("выросла", C_GROW), ("сжалась до полки", C_SHRINK)):
        a.plot([], [], "-", color=col, lw=2.0, marker="o", ms=4, label=lab)
    a.set_xlim(0.0, 0.35)
    a.set_ylim(0.15, 0.36)
    a.set_xlabel(r"радиус капли $R$")
    a.set_ylabel(r"сохраняющееся среднее $m=1+\bar\phi$")
    a.set_title("Фазовый портрет: точка — старт, стрелка — куда пришло\n"
                r"$m$ сохраняется точно, поэтому каждый расчёт — горизонтальная линия", fontsize=10)
    a.legend(fontsize=8.4, loc="lower right", framealpha=0.93)

    # ---- E(R), barrier included ---------------------------------------------------------
    sel_m = sorted({round(r["m"], 3) for r in runs})
    show = [sel_m[i] for i in (0, len(sel_m) // 3, 2 * len(sel_m) // 3, len(sel_m) - 1)]
    cols = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    for m, col in zip(show, cols):
        pts = sorted([r for r in runs if abs(r["m"] - m) < 1e-6], key=lambda r: r["R0"])
        if not pts:
            continue
        rr = np.linspace(0.02, 0.36, 300)
        b.plot(rr, [E_exact(x, m - 1.0, g) for x in rr], "-", color=col, lw=1.6,
               label=rf"$m={m:.2f}$")
        b.plot([p["R0"] for p in pts], [p["E0"] for p in pts], "o", color=col, ms=5,
               mfc="none", mew=1.3, zorder=4)
        r = roots_m(m, g)
        if r:
            b.plot([r[0]], [E_exact(r[0], m - 1.0, g)], "x", color=col, ms=9, mew=2, zorder=5)
            b.plot([r[-1]], [E_exact(r[-1], m - 1.0, g)], "*", color=col, ms=12, zorder=5,
                   mec="0.2", mew=0.5)
    b.plot([], [], "o", color="0.35", ms=5, mfc="none", mew=1.3, label="измерено (шаг 0)")
    b.plot([], [], "x", color="0.35", ms=9, mew=2, label=r"$R_c$ — вершина барьера")
    b.plot([], [], "*", color="0.35", ms=12, mec="0.2", mew=0.5, label=r"$R_\infty$")
    b.set_xlabel(r"$R$")
    b.set_ylabel(r"$F/V$")
    b.set_title("Ландшафт по точкам, включая барьер:\n"
                "энергия на нулевом шаге и есть $E(R)$ при заданном $m$", fontsize=10)
    b.legend(fontsize=8.4, loc="upper left", framealpha=0.93)

    fig.suptitle(
        rf"Сетка по $(R,m)$ при $\gamma={g:.1e}$".replace("e-0", r"\cdot10^{-") + "}$, "
        r"куб $1^3$, $64^3$, среднее задано независимо от радиуса",
        fontsize=11, y=0.965)

    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)

    print(f"\n{'m':>6} {'R_c теор':>9} {'R_inf теор':>11} | старт -> финиш")
    for m in sel_m:
        r = roots_m(m, g)
        rcs = f"{r[0]:9.4f}" if r else f"{'—':>9}"
        ris = f"{r[-1]:11.4f}" if r else f"{'—':>11}"
        pts = sorted([x for x in runs if abs(x["m"] - m) < 1e-6], key=lambda x: x["R0"])
        tags = " ".join(f"{p['R0']:.3f}{'↓' if p['kind']=='dead' else ('↑' if p['kind']=='grow' else '→')}"
                        f"{p['Rend']:.3f}" for p in pts)
        print(f"{m:6.2f} {rcs} {ris} | {tags}")


if __name__ == "__main__":
    main()
