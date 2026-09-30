#!/usr/bin/env python3
"""Radial profile of a stationary droplet against

    phi(r) = -tanh(zeta) + (sigma / 2R) tanh^2(zeta),   zeta = (r - R)/sqrt(2 gamma)

R is read off the computed field as the zero crossing of the spherically averaged profile.

Top row: profiles.  Left holds R = 0.28 and varies gamma, right holds gamma = 2e-4 and varies R.
Bottom row: the mean absolute error over the whole domain, with a line of fixed slope (1 against
gamma, -2 against R) whose offset is the only fitted number, since the first neglected term is
(3/8) psi^2 with psi = sigma/R.

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
R_A = 0.28
GAMMA_SHOW = [2.000e-4, 8.000e-4, 3.200e-3, 6.400e-3]
GAMMA_ALL = [2.000e-4, 4.000e-4, 8.000e-4, 1.600e-3, 3.200e-3, 6.400e-3]
GAMMA_B = 2.000e-4
R_B = [0.20, 0.25, 0.28, 0.33]
COLS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#1f78b4", "#a6761d"]

plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "axes.titlesize": 16,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.5, zorder=1)


def analytic(r, R, gamma):
    z = (r - R) / np.sqrt(2.0 * gamma)
    return -np.tanh(z) + sigma(gamma) / (2.0 * R) * np.tanh(z) ** 2


def sci(v):
    mant, exp = f"{v:.1e}".split("e")
    return mant.replace(".", "{,}") + r"\cdot10^{" + str(int(exp)) + "}"


def mae(phi3d, R, gamma, rmax=0.5):
    """Mean absolute error over the whole domain shown."""
    nz, ny, nx = phi3d.shape
    ax = lambda n: (np.arange(n) + 0.5) / n - 0.5
    r = np.sqrt(ax(nz)[:, None, None] ** 2 + ax(ny)[None, :, None] ** 2 + ax(nx)[None, None, :] ** 2)
    m = r <= rmax
    return float(np.abs(phi3d[m] - analytic(r[m], R, gamma)).mean())


def load(target_R, gamma):
    pat = os.path.join(DATA, f"profile_R{target_R:.2f}_g{gamma:.3e}".replace("e-0", "e-") + "_*")
    for run in sorted(glob.glob(pat), reverse=True):
        fld = os.path.join(run, "solution", "numerical_final.bin")
        if not os.path.exists(fld):
            continue
        cfg, data = parse(run)
        if int(cfg["grid"]) != GRID or data["drop_volume"][-1] <= 0.0:
            continue
        phi = read_field(fld)["phi"]
        r, mean, lo, hi = radial_profile(phi)
        R = zero_crossing(r, mean)
        return {"r": r, "phi": mean, "lo": lo, "hi": hi, "R": R, "gamma": gamma,
                "mae": mae(phi, R, gamma)}
    return None


def draw_profiles(ax, cases, label_of, title):
    for key, gamma, col in cases:
        c = load(key, gamma)
        if c is None:
            print(f"missing: R={key} gamma={gamma:.3e}")
            continue
        rr = np.linspace(0.0, 0.5, 1500)
        ax.fill_between(c["r"], c["lo"], c["hi"], color=col, alpha=0.16, lw=0)
        step = max(1, len(c["r"]) // 55)
        ax.plot(c["r"][::step], c["phi"][::step], "o", color=col, ms=5.0, mfc="none", mew=1.6,
                zorder=4, label=label_of(c))
        ax.plot(rr, analytic(rr, c["R"], gamma), "-", color=col, lw=2.4, zorder=3)
    ax.axhline(0.0, color="0.75", lw=1.0)
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(-1.18, 1.22)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\phi(r)$")
    ax.set_title(title)
    ax.legend(loc="lower left", framealpha=0.93)
    style(ax)


def draw_residual(ax, xs, ys, slope, xlabel, slabel):
    xs, ys = np.asarray(xs), np.asarray(ys)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    b = np.mean(np.log(ys) - slope * np.log(xs))          # only the offset is fitted
    ax.plot(xs, ys, "o", color="#1f78b4", ms=10, zorder=4)
    ax.plot(xs, np.exp(b) * xs**slope, "-", color="0.25", lw=2.4, zorder=3, label=slabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("residue")
    ax.legend(loc="best", framealpha=0.93)
    style(ax)


def main():
    fig = plt.figure(figsize=(16.0, 12.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0])
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ca, cb = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    draw_profiles(a, [(R_A, g, COLS[i]) for i, g in enumerate(GAMMA_SHOW)],
                  lambda c: rf"$\gamma={sci(c['gamma'])}$", r"$R=0.28$")
    draw_profiles(b, [(R, GAMMA_B, COLS[i]) for i, R in enumerate(R_B)],
                  lambda c: rf"$R={c['R']:.3f}$", rf"$\gamma={sci(GAMMA_B)}$")

    ga = [(g, load(R_A, g)) for g in GAMMA_ALL]
    ga = [(g, c) for g, c in ga if c is not None]
    draw_residual(ca, [g for g, _ in ga], [c["mae"] for _, c in ga], 1.0,
                  r"$\gamma$", r"$\propto\gamma$")

    gb = [load(R, GAMMA_B) for R in R_B]
    gb = [c for c in gb if c is not None]
    draw_residual(cb, [c["R"] for c in gb], [c["mae"] for c in gb], -2.0,
                  r"$R$", r"$\propto R^{-2}$")

    fig.suptitle(
        r"$\phi(r)=-\tanh\zeta+\dfrac{\sigma}{2R}\tanh^{2}\zeta,\qquad "
        r"\zeta=\dfrac{r-R}{\sqrt{2\gamma}},\qquad \sigma=\dfrac{2\sqrt{2}}{3}\sqrt{\gamma}$",
        fontsize=16)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)

    print(f"\n{'R':>7} {'gamma':>10} {'MAE':>11}")
    for g, c in ga:
        print(f"{c['R']:7.3f} {g:10.3e} {c['mae']:11.4e}")
    for c in gb:
        print(f"{c['R']:7.3f} {GAMMA_B:10.3e} {c['mae']:11.4e}")


if __name__ == "__main__":
    main()
