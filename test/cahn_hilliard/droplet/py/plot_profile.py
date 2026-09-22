#!/usr/bin/env python3
"""Radial profile of a stationary droplet against the first-order uniformly valid solution

    phi(r) = -tanh(zeta) + (sigma / 2R) tanh^2(zeta),   zeta = (r - R)/sqrt(2 gamma)

-- a linear and a quadratic term in tanh, nothing else.  R is read off the computed field as the
zero crossing of the spherically averaged profile.

Top row: the profiles.  Left holds the stationary radius at R = 0.28 and varies gamma, right holds
gamma = 2e-4 and varies R.

Bottom row: the error norms over the inscribed ball, against gamma on the left and against R on
the right.  The first neglected term is O(psi^2) with psi = sigma/R, and psi^2 ~ gamma/R^2, so the
norm should go as gamma^1 at fixed R and as R^-2 at fixed gamma.  Those two slopes are the test.

Data: sweep_profile, ../data/profile_R<R>_g<gamma>_*/solution/numerical_final.bin.
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
DATA = os.environ.get(
    "DROPLET_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

GRID = 256
GAMMA_A = [2.000e-4, 4.000e-4, 8.000e-4, 1.600e-3, 3.200e-3, 6.400e-3]
R_A = 0.28
GAMMA_B = 2.000e-4
R_B = [0.15, 0.20, 0.25, 0.28, 0.33]
COLS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#1f78b4", "#a6761d"]

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25})


def analytic(r, R, gamma):
    z = (r - R) / np.sqrt(2.0 * gamma)
    return -np.tanh(z) + sigma(gamma) / (2.0 * R) * np.tanh(z) ** 2


def sci(v):
    mant, exp = f"{v:.1e}".split("e")
    return mant.replace(".", "{,}") + r"\cdot10^{" + str(int(exp)) + "}"


def norms(phi3d, R, gamma, rmax=0.5):
    """L1 and L2 of (numerical - analytic) over the inscribed ball, normalised by its volume."""
    nz, ny, nx = phi3d.shape
    z = (np.arange(nz) + 0.5) / nz - 0.5
    y = (np.arange(ny) + 0.5) / ny - 0.5
    x = (np.arange(nx) + 0.5) / nx - 0.5
    r = np.sqrt(z[:, None, None] ** 2 + y[None, :, None] ** 2 + x[None, None, :] ** 2)
    m = r <= rmax
    e = phi3d[m] - analytic(r[m], R, gamma)
    return np.abs(e).mean(), np.sqrt((e**2).mean())


def load(target_R, gamma, grid=GRID):
    pat = os.path.join(DATA, f"profile_R{target_R:.2f}_g{gamma:.3e}".replace("e-0", "e-") + "_*")
    for run in sorted(glob.glob(pat), reverse=True):
        fld = os.path.join(run, "solution", "numerical_final.bin")
        if not os.path.exists(fld):
            continue
        cfg, data = parse(run)
        if int(cfg["grid"]) != grid or data["drop_volume"][-1] <= 0.0:
            continue
        phi = read_field(fld)["phi"]
        r, mean, lo, hi = radial_profile(phi)
        R = zero_crossing(r, mean)
        l1, l2 = norms(phi, R, gamma)
        return {"r": r, "phi": mean, "lo": lo, "hi": hi, "R": R, "gamma": gamma,
                "l1": l1, "l2": l2, "grid": int(cfg["grid"])}
    return None


def draw_profiles(ax, cases, label_of, title):
    out = []
    for key, gamma, col in cases:
        c = load(key, gamma)
        if c is None:
            print(f"missing: R={key} gamma={gamma:.3e} grid={GRID}")
            continue
        out.append(c)
        rr = np.linspace(0.0, 0.5, 1500)
        ax.fill_between(c["r"], c["lo"], c["hi"], color=col, alpha=0.16, lw=0)
        step = max(1, len(c["r"]) // 55)
        ax.plot(c["r"][::step], c["phi"][::step], "o", color=col, ms=3.6, mfc="none", mew=1.0,
                zorder=4, label=label_of(c))
        ax.plot(rr, analytic(rr, c["R"], gamma), "-", color=col, lw=1.6, zorder=3)
    ax.axhline(0.0, color="0.75", lw=0.6)
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(-1.18, 1.22)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=8.6, loc="lower left", framealpha=0.93)
    return out


def draw_scaling(ax, xs, res, xlabel, power, powlabel, extra=None):
    l1 = np.array([c["l1"] for c in res])
    l2 = np.array([c["l2"] for c in res])
    xs = np.array(xs)
    for v, mark, lab, col in ((l2, "o", r"$L_2$", "#1f78b4"), (l1, "s", r"$L_1$", "#e7298a")):
        k, b = np.polyfit(np.log(xs), np.log(v), 1)
        ax.plot(xs, v, mark + "-", color=col, ms=6, lw=1.4,
                label=f"{lab}, наклон {k:.2f}")
    ref = l2[len(l2) // 2] * (xs / xs[len(xs) // 2]) ** power
    ax.plot(xs, ref, ":", color="0.35", lw=1.6, label=powlabel)
    if extra:
        ax.plot([e[0] for e in extra], [e[1] for e in extra], "v", color="0.45", ms=7,
                mfc="none", mew=1.4, label=r"то же на $128^3$ (контроль сетки)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("норма невязки")
    ax.legend(fontsize=8.3, loc="best", framealpha=0.93)


def main():
    fig = plt.figure(figsize=(13.2, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.30, wspace=0.18,
                          left=0.06, right=0.99, top=0.885, bottom=0.065)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ca, cb = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    res_a = draw_profiles(a, [(R_A, g, COLS[i]) for i, g in enumerate(GAMMA_A)],
                          lambda c: rf"$\gamma={sci(c['gamma'])}$",
                          r"A. $R\simeq0{,}28$ закреплён, меняется $\gamma$")
    res_b = draw_profiles(b, [(R, GAMMA_B, COLS[i]) for i, R in enumerate(R_B)],
                          lambda c: rf"$R={c['R']:.3f}$",
                          rf"B. $\gamma={sci(GAMMA_B)}$ закреплена, меняется $R$")

    ctrl = []
    for g in GAMMA_A:
        c = load(R_A, g, grid=128)
        if c is not None:
            ctrl.append((g, c["l2"]))

    if res_a:
        draw_scaling(ca, [c["gamma"] for c in res_a], res_a, r"$\gamma$", 1.0,
                     r"$\propto\gamma$ — ожидаемое $O(\psi^2)$", extra=ctrl)
        ca.set_title(r"A. Невязка против $\gamma$ при $R\simeq0{,}28$", fontsize=10.5)
    if res_b:
        draw_scaling(cb, [c["R"] for c in res_b], res_b, r"$R$", -2.0,
                     r"$\propto R^{-2}$ — ожидаемое $O(\psi^2)$")
        cb.set_title(rf"B. Невязка против $R$ при $\gamma={sci(GAMMA_B)}$", fontsize=10.5)

    fig.suptitle(
        r"$\phi(r)=-\tanh\zeta+\dfrac{\sigma}{2R}\tanh^{2}\zeta,\qquad "
        r"\zeta=\dfrac{r-R}{\sqrt{2\gamma}},\qquad \sigma=\dfrac{2\sqrt2}{3}\sqrt\gamma$"
        "\n"
        rf"точки — расчёт ${GRID}^3$ (сферическое среднее, полоса — разброс внутри слоя), "
        r"линия — формула; $R$ взят из расчёта как ноль профиля",
        fontsize=11.5, y=0.985)

    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)

    print(f"\n{'R':>6} {'gamma':>10} {'grid':>5} {'L1':>10} {'L2':>10}")
    for c in res_a + res_b:
        print(f"{c['R']:6.3f} {c['gamma']:10.3e} {c['grid']:5d} {c['l1']:10.3e} {c['l2']:10.3e}")


if __name__ == "__main__":
    main()
