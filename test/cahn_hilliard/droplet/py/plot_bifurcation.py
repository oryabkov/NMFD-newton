#!/usr/bin/env python3
"""The (R, m) landscape, measured.

Data: sweep_landscape -- a grid of runs with the conserved mean set independently of the droplet
radius (--phi-mean).  Because m is conserved exactly, every run is a horizontal line in the (R, m)
plane, and the theory's two branches predict which way it moves:

    R < R_c(m)              collapse to zero
    R_c(m) < R < R_inf(m)   growth to R_inf
    R > R_inf(m)            shrinkage back to R_inf

Left panel is that phase portrait, with both candidate theories drawn: the sharp-interface
h(R) = A R^4 - m R + sigma/k and the same with the O(gamma) interface mass, h + C gamma R^2.

Right panel samples E(R) directly: the energy logged at step 0 is the landscape evaluated at the
initial radius, so the grid measures the barrier itself -- the part no trajectory can reach.  Two
curves are drawn there because the initial condition is not the constrained minimiser: it shifts
both phases by the same constant, while the true minimiser puts them on the two branches of
f'(phi) = psi, which costs less.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify_sharp_interface import A_COEF, C_COEF, K_DW, _bisect, sigma

OUT = "../figs/bifurcation.png"
DATA = os.environ.get(
    "DROPLET_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

C_DEAD, C_GROW, C_SHRINK = "#9a9a9a", "#1b9e77", "#d95f02"
C_SHARP, C_CORR = "0.55", "#7570b3"

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25})

f_bulk = np.vectorize(lambda p: 0.25 * (p * p - 1.0) ** 2)


def sci(v):
    mant, exp = f"{v:.1e}".split("e")
    return mant.replace(".", "{,}") + r"\cdot10^{" + str(int(exp)) + "}"


def h_m(R, m, g, corr):
    return A_COEF * R**4 - m * R + sigma(g) / K_DW + (C_COEF * g * R**2 if corr else 0.0)


def roots_m(m, g, corr):
    Rs = np.linspace(1e-4, 0.45, 100001)
    v = h_m(Rs, m, g, corr)
    idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    return [_bisect(lambda R: h_m(R, m, g, corr), Rs[i], Rs[i + 1]) for i in idx]


def m_min_num(g, corr):
    Rs = np.linspace(1e-4, 0.45, 20001)
    return _bisect(lambda m: h_m(Rs, m, g, corr).min(), 0.45, 0.05)


def branches(psi):
    r = np.roots([1.0, 0.0, -1.0, -psi])
    r = np.sort(np.real(r[np.abs(r.imag) < 1e-12]))
    return r[0], r[-1]


def E_equil(R, m, g):
    """Landscape: both phases on the two branches of f'(phi) = psi, mass conserved."""
    x = A_COEF * R**3 / 2.0
    lo, hi = -0.6, 0.6
    for _ in range(80):
        psi = 0.5 * (lo + hi)
        po, pi = branches(psi)
        if x * pi + (1 - x) * po + C_COEF * g * R > m - 1.0:
            hi = psi
        else:
            lo = psi
    po, pi = branches(0.5 * (lo + hi))
    return sigma(g) * 4 * np.pi * R**2 + x * f_bulk(pi) + (1 - x) * f_bulk(po)


def E_shifted(R, m, g):
    """What the initial condition actually carries: both phases moved by the same constant."""
    x = A_COEF * R**3 / 2.0
    d = m - A_COEF * R**3 - C_COEF * g * R
    return sigma(g) * 4 * np.pi * R**2 + x * f_bulk(1.0 + d) + (1 - x) * f_bulk(-1.0 + d)


def parse_log(run):
    cfg, rows = {}, []
    with open(os.path.join(run, "log.txt")) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("INFO:"):
                line = line[5:].strip()
            if line.startswith("DROPLET_CONFIG"):
                for tok in line.split()[1:]:
                    k, _, v = tok.partition("=")
                    try:
                        cfg[k] = float(v)
                    except ValueError:
                        cfg[k] = v
            elif line.startswith("DROPLET "):
                rows.append({k: float(v) for k, _, v in (t.partition("=") for t in line.split()[1:])})
    if not rows:
        return None
    return cfg, {k: np.array([r[k] for r in rows]) for k in rows[0]}


def load():
    runs = []
    for d in sorted(glob.glob(os.path.join(DATA, "landscape_m*_R*_*"))):
        got = parse_log(d)
        if got is None:
            continue
        cfg, data = got
        R0, Rend = data["R_eff"][0], data["R_eff"][-1]
        kind = "dead" if Rend <= 0.0 else ("grow" if Rend > R0 * 1.02 else "shrink")
        runs.append({"m": round(1.0 + cfg["phi_mean"], 4), "R0": R0, "Rend": Rend, "kind": kind,
                     "gamma": cfg["gamma"], "E0": data["phobic"][0] + data["philic"][0]})
    return runs


def main():
    runs = load()
    if not runs:
        print("no landscape runs found")
        return
    g = runs[0]["gamma"]
    ms = sorted({r["m"] for r in runs})
    print(f"{len(runs)} runs at gamma = {g:.3e};  "
          f"dead {sum(r['kind']=='dead' for r in runs)}, grow {sum(r['kind']=='grow' for r in runs)}, "
          f"shrink {sum(r['kind']=='shrink' for r in runs)}")

    fig = plt.figure(figsize=(13.8, 6.9))
    gs = fig.add_gridspec(1, 2, wspace=0.19, left=0.055, right=0.99, top=0.855, bottom=0.095)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- phase portrait ------------------------------------------------------------------
    for corr, col, lw, lab in ((False, C_SHARP, 1.5, "резкая граница"),
                               (True, C_CORR, 2.3, r"с поправкой $O(\gamma)$")):
        mm = m_min_num(g, corr)
        mgrid = np.linspace(mm * 1.0001, 0.37, 300)
        rc = [roots_m(m, g, corr)[0] for m in mgrid]
        ri = [roots_m(m, g, corr)[-1] for m in mgrid]
        a.plot(rc, mgrid, "--", color=col, lw=lw)
        a.plot(ri, mgrid, "-", color=col, lw=lw, label=rf"{lab}: $m_{{\min}}={mm:.3f}$")
        a.axhline(mm, color=col, lw=0.9, ls=":")

    for r in runs:
        col = {"dead": C_DEAD, "grow": C_GROW, "shrink": C_SHRINK}[r["kind"]]
        a.annotate("", xy=(max(r["Rend"], 0.012), r["m"]), xytext=(r["R0"], r["m"]),
                   arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, alpha=0.9,
                                   shrinkA=0, shrinkB=0), zorder=4)
        a.plot([r["R0"]], [r["m"]], "o", color=col, ms=3.0, zorder=5)

    for m in ms:                                     # measured R_c bracket and R_inf
        row = sorted([r for r in runs if r["m"] == m], key=lambda r: r["R0"])
        dead = [r["R0"] for r in row if r["kind"] == "dead"]
        alive = [r["R0"] for r in row if r["kind"] != "dead"]
        if dead and alive:
            a.plot([max(dead), min(alive)], [m, m], "-", color="k", lw=3.2, alpha=0.5, zorder=6)
        ends = {round(r["Rend"], 4) for r in row if r["kind"] != "dead"}
        for e in ends:
            a.plot([e], [m], "D", color="k", ms=4.5, zorder=7)
    a.plot([], [], "-", color="k", lw=3.2, alpha=0.5, label=r"измеренная вилка на $R_c$")
    a.plot([], [], "D", color="k", ms=4.5, label=r"измеренный $R_\infty$")
    for lab, col in (("испарилась", C_DEAD), ("выросла", C_GROW), ("сжалась", C_SHRINK)):
        a.plot([], [], "-", color=col, lw=1.8, marker="o", ms=3.5, label=lab)
    a.set_xlim(0.0, 0.36)
    a.set_ylim(0.155, 0.355)
    a.set_xlabel(r"радиус капли $R$")
    a.set_ylabel(r"сохраняющееся среднее $m=1+\bar\phi$")
    a.set_title("Фазовый портрет: точка — старт, стрелка — куда пришло;\n"
                r"$m$ сохраняется точно, поэтому расчёт — горизонтальная линия", fontsize=10.5)
    a.legend(fontsize=8.2, loc="lower right", framealpha=0.94, ncol=1)

    # ---- E(R) ----------------------------------------------------------------------------
    show = [ms[0], ms[3], ms[6], ms[-1]]
    cols = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    rr = np.linspace(0.02, 0.36, 260)
    for m, col in zip(show, cols):
        pts = sorted([r for r in runs if r["m"] == m], key=lambda r: r["R0"])
        b.plot(rr, [E_shifted(x, m, g) for x in rr], "-", color=col, lw=1.7,
               label=rf"$m={m:.2f}$")
        b.plot(rr, [E_equil(x, m, g) for x in rr], "--", color=col, lw=1.1, alpha=0.8)
        b.plot([p["R0"] for p in pts], [p["E0"] for p in pts], "o", color=col, ms=5,
               mfc="none", mew=1.3, zorder=4)
        r = roots_m(m, g, True)
        if r:
            b.plot([r[0]], [E_equil(r[0], m, g)], "x", color=col, ms=9, mew=2, zorder=5)
            b.plot([r[-1]], [E_equil(r[-1], m, g)], "*", color=col, ms=12, zorder=5,
                   mec="0.2", mew=0.5)
    b.plot([], [], "o", color="0.35", ms=5, mfc="none", mew=1.3, label="измерено (шаг 0)")
    b.plot([], [], "-", color="0.35", lw=1.7, label="энергия того же НУ (общий сдвиг фаз)")
    b.plot([], [], "--", color="0.35", lw=1.1, label=r"ландшафт $E(R)$ (равновесные фазы)")
    b.plot([], [], "x", color="0.35", ms=9, mew=2, label=r"$R_c$")
    b.plot([], [], "*", color="0.35", ms=12, mec="0.2", mew=0.5, label=r"$R_\infty$")
    b.set_xlabel(r"$R$")
    b.set_ylabel(r"$F/V$")
    b.set_title("Ландшафт по точкам, включая барьер:\n"
                r"энергия на нулевом шаге и есть $E(R)$ при заданном $m$", fontsize=10.5)
    b.legend(fontsize=8.2, loc="upper left", framealpha=0.94)

    fig.suptitle(
        rf"Сетка по $(R,m)$ при $\gamma={sci(g)}$, куб $1^3$, $64^3$: "
        r"среднее задано независимо от радиуса, поэтому достижима и неустойчивая ветвь",
        fontsize=11.5, y=0.965)
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)

    print(f"\n{'m':>6} | {'R_c резк':>8} {'R_c O(g)':>9} {'вилка':>15} | "
          f"{'R_inf резк':>10} {'R_inf O(g)':>10} {'измерено':>9}")
    for m in ms:
        row = sorted([r for r in runs if r["m"] == m], key=lambda r: r["R0"])
        dead = [r["R0"] for r in row if r["kind"] == "dead"]
        alive = [r["R0"] for r in row if r["kind"] != "dead"]
        rs, rk = roots_m(m, g, False), roots_m(m, g, True)
        br = f"({max(dead):.3f}, {min(alive):.3f})" if dead and alive else (
            "всё выжило" if alive else "всё умерло")
        meas = f"{np.mean([r['Rend'] for r in row if r['kind'] != 'dead']):.4f}" if alive else "—"
        print(f"{m:6.2f} | {rs[0] if rs else float('nan'):8.4f} {rk[0] if rk else float('nan'):9.4f} "
              f"{br:>15} | {rs[-1] if rs else float('nan'):10.4f} "
              f"{rk[-1] if rk else float('nan'):10.4f} {meas:>9}")


if __name__ == "__main__":
    main()
