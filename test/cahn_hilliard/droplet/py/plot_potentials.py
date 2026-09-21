#!/usr/bin/env python3
"""
Supervisor item #1: what the logarithmic potential actually looks like as omega varies,
and whether it is genuinely "stiffer" than the double well or just a rescaling of it.

Writes figures into ../figs and a markdown table into ../notes/02_potential_shapes_table.md.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from potentials import (
    OMEGAS,
    dw_ddf,
    dw_df,
    dw_f,
    dw_summary,
    log_ddf,
    log_df,
    log_f,
    phi_eq,
    summary,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figs")
NOTES = os.path.join(HERE, "..", "notes")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(NOTES, exist_ok=True)

CMAP = plt.get_cmap("viridis")
COLORS = {w: CMAP(i / max(1, len(OMEGAS) - 1)) for i, w in enumerate(OMEGAS)}
DWC = "crimson"


def fig_raw():
    p = np.linspace(-1.0 + 1e-9, 1.0 - 1e-9, 4001)
    pw = np.linspace(-1.25, 1.25, 4001)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    ax[0, 0].plot(pw, dw_f(pw), color=DWC, lw=2.2, label="double well")
    ax[0, 1].plot(pw, dw_df(pw), color=DWC, lw=2.2, label="double well")
    ax[1, 0].plot(pw, dw_ddf(pw), color=DWC, lw=2.2, label="double well")

    for w in OMEGAS:
        pe = phi_eq(w)
        c = COLORS[w]
        lbl = rf"$\omega={w:g}$ ($\phi_{{eq}}={pe:.3f}$)"
        ax[0, 0].plot(p, log_f(p, w) - log_f(np.array([pe]), w)[0], color=c, label=lbl)
        ax[0, 1].plot(p, log_df(p, w), color=c, label=lbl)
        ax[1, 0].plot(p, log_ddf(p, w), color=c, label=lbl)
        ax[0, 0].plot([pe], [0.0], "o", color=c, ms=4)

    ax[0, 0].set(xlabel=r"$\phi$", ylabel=r"$f(\phi)-f(\phi_{eq})$", title="bulk potential (shifted to min 0)")
    ax[0, 0].set_ylim(-0.05, 1.2)
    ax[0, 1].set(xlabel=r"$\phi$", ylabel=r"$f'(\phi)$", title="chemical-potential part $f'$")
    ax[0, 1].set_ylim(-3, 3)
    ax[0, 1].axhline(0, color="k", lw=0.6)
    ax[1, 0].set(xlabel=r"$\phi$", ylabel=r"$f''(\phi)$", title="curvature $f''$")
    ax[1, 0].set_ylim(-11, 20)
    ax[1, 0].axhline(0, color="k", lw=0.6)

    # zoom: slope at / above the equilibrium point, the quantity that sets the
    # Gibbs-Thomson shift  delta_phi = mu / f''(phi_eq)
    ax[1, 1].plot(pw[pw > 0.5], dw_df(pw[pw > 0.5]), color=DWC, lw=2.2, label="double well")
    for w in OMEGAS:
        m = p > 0.5
        ax[1, 1].plot(p[m], log_df(p[m], w), color=COLORS[w])
    ax[1, 1].axhline(0, color="k", lw=0.6)
    ax[1, 1].set(xlabel=r"$\phi$", ylabel=r"$f'(\phi)$", title=r"zoom near the upper equilibrium")
    ax[1, 1].set_xlim(0.5, 1.25)
    ax[1, 1].set_ylim(-1.0, 3.0)

    for a in ax.ravel():
        a.grid(alpha=0.25)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Raw potentials: logarithmic (varying $\\omega$) vs double well", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "potential_raw.png"), dpi=150)
    plt.close(fig)


def fig_normalized():
    """
    Remove the trivial rescaling the supervisor pointed at: put the equilibrium of every
    potential at u = phi/phi_eq = 1 and scale the energy so that g''(1) = 2, exactly as for
    the double well. Whatever difference survives is a real difference in shape.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    u = np.linspace(-1.35, 1.35, 4001)
    ax[0].plot(u, dw_f(u), color=DWC, lw=2.2, label="double well")
    ax[1].plot(u, dw_df(u), color=DWC, lw=2.2, label="double well")
    du = np.linspace(1e-4, 0.35, 2000)
    ax[2].plot(du, dw_f(1 + du) - dw_f(np.array(1.0)), color=DWC, lw=2.2, label="double well")

    rows = []
    for w in OMEGAS:
        pe = phi_eq(w)
        ddf = 2.0 / (1.0 - pe ** 2) - w
        c = 2.0 / (pe ** 2 * ddf)
        f0 = log_f(np.array([pe]), w)[0]

        uu = np.linspace(-1.0 / pe + 1e-9, 1.0 / pe - 1e-9, 4001)
        g = c * (log_f(pe * uu, w) - f0)
        dg = c * pe * log_df(pe * uu, w)
        ax[0].plot(uu, g, color=COLORS[w], label=rf"$\omega={w:g}$")
        ax[1].plot(uu, dg, color=COLORS[w])

        d = np.linspace(1e-4, 1.0 / pe - 1.0 - 1e-6, 2000)
        ax[2].plot(d, c * (log_f(pe * (1 + d), w) - f0), color=COLORS[w], label=rf"$\omega={w:g}$")
        rows.append((w, pe, 1.0 / pe - 1.0))

    ax[0].set(xlabel=r"$u=\phi/\phi_{eq}$", ylabel=r"$g(u)$", title="curvature-matched potential")
    ax[0].set_ylim(-0.05, 1.5)
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel=r"$u$", ylabel=r"$g'(u)$", title="curvature-matched $g'$ ($g''(1)=2$ for all)")
    ax[1].set_ylim(-3, 4)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[2].set(
        xlabel=r"overshoot $\delta = u-1$",
        ylabel=r"$g(1+\delta)-g(1)$",
        title="energy price of pushing $\\phi$ past equilibrium",
    )
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle(
        "After matching $\\phi_{eq}\\to 1$ and $f''(\\phi_{eq})\\to 2$: what shape difference is left",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "potential_normalized.png"), dpi=150)
    plt.close(fig)


def fig_scalars():
    ws = np.linspace(2.02, 12.0, 400)
    s = [summary(w) for w in ws]
    dw = dw_summary()

    keys = [
        ("phi_eq", r"$\phi_{eq}$", False),
        ("ddf_eq", r"$f''(\phi_{eq})$", False),
        ("barrier", r"barrier $f(0)-f(\phi_{eq})$", True),
        ("sigma_hat", r"$\sigma/\sqrt{\gamma}$", True),
        ("eps_hat", r"$\epsilon/\sqrt{\gamma}=1/\sqrt{f''(\phi_{eq})}$", True),
        ("headroom", r"headroom $1/\phi_{eq}-1$", True),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    for a, (k, lbl, logy) in zip(ax.ravel(), keys):
        a.plot(ws, [x[k] for x in s], color="teal", lw=2)
        a.axhline(dw[k], color=DWC, ls="--", lw=1.6, label="double well")
        for w in OMEGAS:
            a.plot([w], [summary(w)[k]], "o", color=COLORS[w], ms=6)
        a.set(xlabel=r"$\omega$", title=lbl)
        if logy:
            a.set_yscale("log")
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
    fig.suptitle("Logarithmic potential: derived quantities vs $\\omega$", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "potential_scalars.png"), dpi=150)
    plt.close(fig)


def table():
    dw = dw_summary()
    lines = [
        "# Potential shapes: derived quantities",
        "",
        "Generated by `py/plot_potentials.py`. Energy functional "
        "$F=\\int f(\\phi)+\\tfrac{\\gamma}{2}|\\nabla\\phi|^2$.",
        "",
        "* `phi_eq` — positive root of $f'$;",
        "* `f''(phi_eq)` — curvature at equilibrium; sets the Gibbs-Thomson shift "
        "$\\delta\\phi=\\mu/f''(\\phi_{eq})$;",
        "* `sigma/sqrt(gamma)` — $\\int_{-\\phi_{eq}}^{\\phi_{eq}}\\sqrt{2\\Delta f}\\,d\\phi$;",
        "* `eps/sqrt(gamma)` — interface decay length $1/\\sqrt{f''(\\phi_{eq})}$;",
        "* `gamma for eps=1` — the $\\gamma$ that gives a unit interface length, i.e. "
        "$f''(\\phi_{eq})$; use it to compare potentials at *matched* interface width;",
        "* `sigma at eps=1` — resulting surface tension at that matched width;",
        "* `headroom` — $1/\\phi_{eq}-1$, the relative overshoot available before $f=+\\infty$;",
        "* `Lambda` — $\\sigma/(2\\phi_{eq}^2 f''(\\phi_{eq})\\epsilon)$, a pure shape number "
        "(invariant under rescaling of $\\phi$ and of $\\gamma$): the Gibbs-Thomson overshoot is "
        "$\\delta\\phi/\\phi_{eq}=(d-1)\\Lambda\\,\\epsilon/R$. Smaller is better;",
        "* `Rc/L at Cn=0.02` — critical initial radius below which the drop evaporates entirely, "
        "$R_c/L=(0.505\\,\\Lambda\\,Cn)^{1/4}$ (see `03_shrinkage_criterion.md`).",
        "",
        "| potential | phi_eq | f''(phi_eq) | barrier | sigma/sqrt(gamma) | eps/sqrt(gamma) "
        "| gamma for eps=1 | sigma at eps=1 | headroom | Lambda |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(name, d):
        g1 = d["ddf_eq"]
        lam = d["sigma_hat"] / (d["phi_eq"] * np.sqrt(g1))
        return (
            f"| {name} | {d['phi_eq']:.4f} | {d['ddf_eq']:.4f} | {d['barrier']:.4f} | "
            f"{d['sigma_hat']:.4f} | {d['eps_hat']:.4f} | {g1:.4f} | "
            f"{d['sigma_hat'] * np.sqrt(g1):.4f} | "
            + ("inf" if not np.isfinite(d["headroom"]) else f"{d['headroom']:.4f}")
            + f" | {lam:.4f} |"
        )

    lines.append(row("double well", dw))
    for w in OMEGAS:
        lines.append(row(f"log, omega={w:g}", summary(w)))
    lines += ["", "![raw](../figs/potential_raw.png)", "", "![normalized](../figs/potential_normalized.png)", "", "![scalars](../figs/potential_scalars.png)", ""]

    path = os.path.join(NOTES, "02_potential_shapes_table.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines[12:22]))
    print("wrote", path)


if __name__ == "__main__":
    fig_raw()
    fig_normalized()
    fig_scalars()
    table()
