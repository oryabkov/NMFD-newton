#!/usr/bin/env python3
"""One figure, one panel per candidate: the polynomial double well (same colour everywhere)
against one alternative, with a single shared legend carrying names and formulas."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from potential_zoo import (WIDTH, Candidate, dw_pair, log_pair, obstacle_pair,
                           smooth_obstacle_pair, table)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

OUT = "../figs/potential_zoo.png"

DW_COLOR = "0.45"
XLIM = (-1.26, 1.26)
YLIM = (-0.012, 0.30)
ZOOM_X = (0.90, 1.12)
ZOOM_Y = (-0.0016, 0.036)


def candidates():
    """(short title, colour, linestyle, formula, Candidate) for the six alternatives."""
    out = []

    f, df, pe = log_pair(3.0)
    out.append((r"логарифмический, $\omega=3$", "#d95f02", "-",
                r"$f\propto(1{+}\phi)\ln(1{+}\phi)+(1{-}\phi)\ln(1{-}\phi)-\frac{\omega}{2}\phi^2$",
                Candidate("", f, df, pe, 1.0 - 1e-13)))

    for p, col in ((1.5, "#1b9e77"), (1.2, "#7570b3")):
        f, df = dw_pair(p)
        out.append((f"со сломом, $p={p}$", col, "-",
                    rf"$f\propto\left|1-\phi^{{2}}\right|^{{{p}}}$",
                    Candidate("", f, df, 1.0, 4.0)))

    for e, col in ((0.15, "#e7298a"), (0.05, "#1f78b4")):
        f, df = smooth_obstacle_pair(e)
        out.append((rf"сглаженное препятствие, $\eta={e}$", col, "-",
                    rf"$f\propto\sqrt{{(1-\phi^{{2}})^{{2}}+\eta^{{4}}}}-\eta^{{2}},\;\eta={e}$",
                    Candidate("", f, df, 1.0, 4.0)))

    f, df = obstacle_pair(1.0)
    out.append(("двойное препятствие", "k", "--",
                r"$f\propto(1-\phi^{2})+I_{[-1,1]}(\phi)$",
                Candidate("", f, df, 1.0, 1.0, obstacle=True)))
    return out


def main():
    f, df = dw_pair(2.0)
    dw = Candidate("", f, df, 1.0, 4.0)
    rc_dw = table([dw], width=WIDTH)[0]["Rc"]

    cands = candidates()
    rcs = [table([c], width=WIDTH)[0]["Rc"] for *_, c in cands]

    fig = plt.figure(figsize=(14.6, 7.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.06,
                             left=0.045, right=0.995, top=0.865, bottom=0.075)
    grid = outer[0, 0].subgridspec(2, 3, hspace=0.30, wspace=0.13)

    x = np.linspace(XLIM[0], XLIM[1], 4001)
    axes = []

    for n, ((title, col, ls, _, c), rc) in enumerate(zip(cands, rcs)):
        ax = fig.add_subplot(grid[n // 3, n % 3])
        axes.append(ax)

        axin = ax.inset_axes([0.030, 0.45, 0.40, 0.50], zorder=6)
        axin.set_facecolor("white")
        axin.patch.set_alpha(1.0)
        for sp in axin.spines.values():
            sp.set_color("0.55")
        xz = np.linspace(*ZOOM_X, 3001)

        ax.plot(x, dw.f(x), color=DW_COLOR, lw=2.6, zorder=2)
        axin.plot(xz, dw.f(xz), color=DW_COLOR, lw=2.2)
        if c.obstacle:
            xo = np.linspace(-1.0, 1.0, 1201)
            ax.plot(xo, c.f(xo), color=col, lw=2.2, ls=ls, zorder=3)
            for s in (-1.0, 1.0):
                ax.plot([s, s], [0.0, YLIM[1]], color=col, lw=2.2, ls=ls, zorder=3)
            xoz = xz[xz <= 1.0]
            axin.plot(xoz, c.f(xoz), color=col, lw=1.9, ls=ls)
            axin.plot([1.0, 1.0], [0.0, ZOOM_Y[1]], color=col, lw=1.9, ls=ls)
        else:
            ax.plot(x, c.f(x), color=col, lw=2.2, ls=ls, zorder=3)
            axin.plot(xz, c.f(xz), color=col, lw=1.9, ls=ls)

        axin.set_xlim(*ZOOM_X)
        axin.set_ylim(*ZOOM_Y)
        axin.set_xticks([0.90, 1.00, 1.10])
        axin.set_yticks([0.0, 0.02])
        axin.yaxis.tick_right()
        axin.tick_params(labelsize=7, length=2.5, pad=1.5)
        axin.grid(alpha=0.2)

        ax.axhline(0.0, color="0.7", lw=0.6)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_title(title, fontsize=10.5, pad=6)
        ax.text(0.968, 0.955,
                f"$R_c={rc:.3f}$" + "\n" + f"({rc / rc_dw:.2f}" + r"$\times$)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=col, zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

        if n // 3 == 1:
            ax.set_xlabel(r"$\phi$")
        else:
            ax.tick_params(labelbottom=False)
        if n % 3 == 0:
            ax.set_ylabel(r"$f(\phi)$")
        else:
            ax.tick_params(labelleft=False)

    handles = [Line2D([], [], color=DW_COLOR, lw=2.6,
                      label="полиномиальный (двойная яма)\n" + r"$f\propto(1-\phi^{2})^{2}$" + "\n")]
    for title, col, ls, formula, _ in cands:
        handles.append(Line2D([], [], color=col, lw=2.2, ls=ls,
                              label=f"{title}\n{formula}\n"))

    lax = fig.add_subplot(outer[0, 1])
    lax.axis("off")
    lax.legend(handles=handles, loc="center left", frameon=False, fontsize=9.3,
               handlelength=1.9, handletextpad=0.8, labelspacing=1.05,
               borderaxespad=0.0)

    fig.suptitle(
        "Потенциалы, приведённые к общей нормировке "
        r"($\phi_{\text{равн}}=\pm1$, высота барьера $1/4$); серая кривая всюду одна и та же; "
        r"во врезке — окрестность $\phi=+1$"
        "\n"
        r"$R_c$ — порог выживания капли в кубе $1\times1\times1$ при одинаковой ширине границы "
        f"$W={WIDTH}$; в скобках — отношение к полиномиальному потенциалу",
        fontsize=11.5, y=0.975)

    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
