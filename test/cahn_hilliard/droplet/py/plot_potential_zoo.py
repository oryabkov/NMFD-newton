#!/usr/bin/env python3
"""One figure: candidate bulk potentials, normalised to a common scale, with the
threshold radius each of them yields at a fixed resolved interface width."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from potential_zoo import WIDTH, table

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

OUT = "../figs/potential_zoo.png"


def main():
    rows = table(width=WIDTH)
    ref = rows[0]["Rc"]

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    axin = ax.inset_axes([0.585, 0.455, 0.385, 0.38])

    x = np.linspace(-1.35, 1.35, 4001)
    xin = np.linspace(0.88, 1.12, 6001)

    for r in rows:
        c = r["cand"]
        rc = r["Rc"]
        lab = f"{c.name}:  $R_c={rc:.3f}$  ({rc / ref:.2f}$\\times$)"
        if c.obstacle:
            xo = np.linspace(-1.0, 1.0, 1001)
            ax.plot(xo, c.f(xo), label=lab, **c.style)
            axin.plot(xo[xo > 0.85], c.f(xo[xo > 0.85]), **c.style)
            for s in (-1.0, 1.0):
                ax.plot([s, s], [0.0, 0.62], color=c.style["color"], lw=c.style["lw"],
                        ls=c.style.get("ls", "-"))
            axin.plot([1.0, 1.0], [0.0, 0.05], color=c.style["color"],
                      lw=c.style["lw"], ls=c.style.get("ls", "-"))
        else:
            ax.plot(x, c.f(x), label=lab, **c.style)
            axin.plot(xin, c.f(xin), **c.style)

    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.annotate(r"$f=+\infty$", xy=(1.025, 0.15), fontsize=9.5)
    ax.annotate(r"$f=+\infty$", xy=(-1.20, 0.36), fontsize=9.5)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.02, 0.62)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$f(\phi)$")
    ax.set_title("Потенциалы, приведённые к общей нормировке "
                 r"($\phi_{\text{равн}}=\pm1$, высота барьера $1/4$)" "\n"
                 f"$R_c$ — порог выживания капли в кубе $1{chr(215)}1{chr(215)}1$ "
                 f"при одинаковой ширине границы $W={WIDTH}$", fontsize=11)
    ax.legend(fontsize=8.6, loc="upper left", framealpha=0.92)

    axin.set_xlim(0.88, 1.12)
    axin.set_ylim(-0.0025, 0.05)
    axin.set_title(r"окрестность равновесного значения $\phi=+1$", fontsize=8.5)
    axin.tick_params(labelsize=7.5)
    axin.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
