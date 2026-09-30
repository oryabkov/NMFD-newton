#!/usr/bin/env python3
"""The stationary chemical potential against Gibbs-Thomson, section 3.7 of the note:

    psi_0 = sigma / R + O(gamma^{3/2}),   sigma = (2 sqrt2 / 3) sqrt(gamma)

psi_0 needs no fitting: in a stationary state it is a constant (its spatial spread in the
saved field is float32 roundoff), so every run contributes one number.  R is the zero
crossing of the spherically averaged phi, the same definition the profile figure uses.

Top row: psi_0 itself.  Left holds R = 0.28 and varies gamma (theory ~ sqrt(gamma)),
right holds gamma = 2e-4 and varies R (theory ~ 1/R).
Bottom row: |psi_0 - sigma/R| with a line of fixed slope whose offset is the only fitted
number: 3/2 against gamma and -3 against R.

Data: sweep_profile, ../data/profile_R<R>_g<gamma>_*/solution/numerical_final.bin.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

from field_io import radial_profile, read_field, zero_crossing
from verify_sharp_interface import parse, sigma

OUT = "../figs/psi0.png"
DATA = os.environ.get(
    "DROPLET_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

GRID = 256
R_A = 0.28
GAMMA_ALL = [2.000e-4, 4.000e-4, 8.000e-4, 1.600e-3, 3.200e-3, 6.400e-3]
GAMMA_B = 2.000e-4
R_B = [0.20, 0.25, 0.28, 0.33]

plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "axes.titlesize": 16,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.5, zorder=1)


def logticks(ax, xs):
    """Inside one decade the default log labelling overlaps; tick the data instead."""
    xs = np.asarray(xs)
    if xs.max() / xs.min() < 10.0:
        ax.set_xticks(np.sort(xs))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())


def sci(v):
    mant, exp = f"{v:.1e}".split("e")
    return mant.replace(".", "{,}") + r"\cdot10^{" + str(int(exp)) + "}"


def load(target_R, gamma, grid=None):
    pat = os.path.join(DATA, f"profile_R{target_R:.2f}_g{gamma:.3e}".replace("e-0", "e-") + "_*")
    for run in sorted(glob.glob(pat), reverse=True):
        fld = os.path.join(run, "solution", "numerical_final.bin")
        if not os.path.exists(fld):
            continue
        cfg, data = parse(run)
        if int(cfg["grid"]) != (grid or GRID) or data["drop_volume"][-1] <= 0.0:
            continue
        f = read_field(fld)
        r, mean, _, _ = radial_profile(f["phi"])
        R = zero_crossing(r, mean)
        R_vol = (3.0 * data["drop_volume"][-1] / (4.0 * np.pi)) ** (1.0 / 3.0)
        psi = float(f["psi"].mean())
        return {"R": R, "R_vol": R_vol, "gamma": gamma, "psi": psi, "gt": sigma(gamma) / R,
                "gt_vol": sigma(gamma) / R_vol, "grid": int(cfg["grid"])}
    return None


def draw_psi(ax, cases, xkey, xlabel, title, curve):
    xs = np.array([c[xkey] for c in cases])
    ax.plot(xs, [c["psi"] for c in cases], "o", color="#1f78b4", ms=10, zorder=4, label=r"$\psi_0$")
    t = np.linspace(xs.min() * 0.85, xs.max() * 1.18, 400)
    ax.plot(t, curve(t), "-", color="0.25", lw=2.4, zorder=3, label=r"$\sigma/R$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\psi_0$")
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.93)
    logticks(ax, xs)
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
    logticks(ax, xs)
    style(ax)


def main():
    ga = [c for c in (load(R_A, g) for g in GAMMA_ALL) if c is not None]
    gb = [c for c in (load(R, GAMMA_B) for R in R_B) if c is not None]

    fig = plt.figure(figsize=(16.0, 12.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0])
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ca, cb = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    draw_psi(a, ga, "gamma", r"$\gamma$", rf"$R={R_A}$",
             lambda g: sigma(g) / np.mean([c["R"] for c in ga]))
    draw_psi(b, gb, "R", r"$R$", rf"$\gamma={sci(GAMMA_B)}$", lambda R: sigma(GAMMA_B) / R)

    draw_residual(ca, [c["gamma"] for c in ga], [abs(c["psi"] - c["gt"]) for c in ga], 1.5,
                  r"$\gamma$", r"$\propto\gamma^{3/2}$")
    draw_residual(cb, [c["R"] for c in gb], [abs(c["psi"] - c["gt"]) for c in gb], -3.0,
                  r"$R$", r"$\propto R^{-3}$")

    fig.suptitle(
        r"$\psi_0=\dfrac{\sigma}{R}+O\!\left(\gamma^{3/2}\right),\qquad "
        r"\sigma=\dfrac{2\sqrt{2}}{3}\sqrt{\gamma}$", fontsize=16)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)

    print(f"\n{'R':>7} {'gamma':>10} {'N':>4} {'psi_0':>12} {'sigma/R':>12} {'ratio':>9} "
          f"{'residue':>11} {'/gamma^3/2':>11}")
    for c in ga + gb:
        d = c["psi"] - c["gt"]
        print(f"{c['R']:7.4f} {c['gamma']:10.3e} {c['grid']:4d} {c['psi']:12.6e} {c['gt']:12.6e} "
              f"{c['psi'] / c['gt']:9.5f} {d:11.3e} {d / c['gamma'] ** 1.5:11.3f}")


if __name__ == "__main__":
    main()
