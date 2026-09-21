#!/usr/bin/env python3
"""
The two headline figures:
  figs/Rc_vs_omega.png  -- measured critical radius vs omega against both predictions
  figs/Rc_vs_eps.png    -- R_c ~ (eps V)^(1/4) over a factor 8 in eps
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from criterion import R_c as Rc_ex
from parse_runs import load_run
from predict import R_c as Rc_lin, gamma_for_eps

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")


def brackets(paths, key):
    """{group: (largest R0 that died, smallest R0 that survived)}"""
    g = {}
    for p in sorted(paths):
        r = load_run(p)
        if not r:
            continue
        c = r["cfg"]
        g.setdefault(key(c), []).append((c["r0"], r["R_eff"][-1]))
    out = {}
    for k, v in g.items():
        died = [r0 for r0, R in v if R == 0]
        surv = [r0 for r0, R in v if R > 0]
        out[k] = (max(died) if died else None, min(surv) if surv else None)
    return out


def fig_omega():
    b = brackets(glob.glob("../data/omega_*"), lambda c: c["omega"])
    # the omega=3 lower edge comes from the r0_log sweep, which used a coarser R0 grid
    b3 = brackets(glob.glob("../data/r0_log_*"), lambda c: c["omega"])
    if 3.0 in b and 3.0 in b3:
        lo = max([x for x in (b[3.0][0], b3[3.0][0]) if x is not None], default=None)
        hi = min([x for x in (b[3.0][1], b3[3.0][1]) if x is not None], default=None)
        b[3.0] = (lo, hi)
    bdw = brackets(glob.glob("../data/r0_dw_*"), lambda c: 2.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ws = np.array([2.5, 3.0, 4.0, 6.0])
    gam = {w: gamma_for_eps("log", 0.04, w) for w in ws}
    ax.plot(ws, [Rc_lin("log", gam[w], 1.0, w) for w in ws], "o--", color="steelblue", lw=2,
            label=r"linearised $\Lambda$ theory (notes/03)")
    ax.plot(ws, [Rc_ex("log", gam[w], 1.0, w) for w in ws], "s-", color="seagreen", lw=2,
            label="exact nonlinear criterion (notes/06)")

    for w, (lo, hi) in sorted(b.items()):
        if lo is None or hi is None:
            if hi is None and lo is not None:
                ax.annotate("", xy=(w, lo + 0.13), xytext=(w, lo),
                            arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.5))
                ax.plot([w], [lo], "_", color="crimson", ms=22, mew=3)
            continue
        ax.plot([w, w], [lo, hi], color="crimson", lw=8, alpha=0.45,
                solid_capstyle="butt")
    ax.plot([], [], color="crimson", lw=8, alpha=0.45, label="measured bracket (dies / survives)")

    gdw = 3.2e-3
    lo, hi = bdw[2.0]
    ax.plot([2.0, 2.0], [lo, hi], color="crimson", lw=8, alpha=0.45, solid_capstyle="butt")
    ax.plot([2.0], [Rc_lin("dw", gdw)], "o", color="steelblue", ms=8)
    ax.plot([2.0], [Rc_ex("dw", gdw)], "s", color="seagreen", ms=8)
    ax.annotate("double well\n($\\omega\\to2$ limit)", xy=(2.0, 0.20), ha="center", fontsize=9,
                color="dimgray")

    ax.set(xlabel=r"$\omega$  (logarithmic potential;  $\omega\to2$ = double well)",
           ylabel=r"critical radius $R_c$",
           title="Stiffening the potential makes the drop LESS robust, not more\n"
                 r"(all at matched interface width $\epsilon=0.04$, unit box)")
    ax.set_ylim(0.15, 0.65)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIGS, "Rc_vs_omega.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def fig_eps():
    b = brackets(glob.glob("../data/gamma_*"), lambda c: c["gamma"])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    gs = np.logspace(np.log10(1e-4), np.log10(2e-2), 60)
    eps = np.sqrt(gs / 2.0)
    ax.plot(eps, [Rc_lin("dw", g) for g in gs], "--", color="steelblue", lw=2,
            label=r"linearised: $R_c=(0.505\,\Lambda\epsilon V)^{1/4}$")
    ax.plot(eps, [Rc_ex("dw", g) for g in gs], "-", color="seagreen", lw=2,
            label="exact nonlinear criterion")
    for g, (lo, hi) in sorted(b.items()):
        e = np.sqrt(g / 2.0)
        if lo is None or hi is None:
            if lo is not None:
                ax.plot([e], [lo], "_", color="crimson", ms=20, mew=3)
                ax.annotate("", xy=(e, lo * 1.35), xytext=(e, lo),
                            arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.5))
            continue
        ax.plot([e, e], [lo, hi], color="crimson", lw=8, alpha=0.45, solid_capstyle="butt")
    ax.plot([], [], color="crimson", lw=8, alpha=0.45, label="measured bracket")
    ax.set(xscale="log", yscale="log", xlabel=r"$\epsilon=\sqrt{\gamma/f''(\phi_{eq})}$",
           ylabel=r"$R_c$", title=r"double well: $R_c\propto\epsilon^{1/4}$ over a factor 8 in $\epsilon$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIGS, "Rc_vs_eps.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_omega()
    fig_eps()
