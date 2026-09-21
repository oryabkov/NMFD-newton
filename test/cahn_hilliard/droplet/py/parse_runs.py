#!/usr/bin/env python3
"""
Parse droplet runs into a table and plot R(t), mass(t), max(phi)(t).

Each run directory produced by ../../run.sh contains a log with one line per time step:

    DROPLET step=1 t=2.000000e-03 mass=... drop_volume=... R_eff=... phi_max=...
            phi_min=... phi_centre=... phobic=... philic=... newton=3 resid=...

and one configuration line:

    DROPLET_CONFIG init=sphere R0=2.400000e-01 gamma=8.000000e-04 potential=double_well
                   omega=3 mobility=constant D=1 floor=0 face_avg=midpoint bc=neumann
                   grid=128 dt=2.000000e-03

Usage:
    python3 parse_runs.py ../data/sweep_R0_*        # table + figures
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from predict import R_c, R_inf_over_R0, derived

STEP_RE = re.compile(r"^DROPLET\s+(.*)$")
CFG_RE = re.compile(r"^DROPLET_CONFIG\s+(.*)$")


def _kv(text):
    out = {}
    for tok in text.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def load_run(path):
    logs = glob.glob(os.path.join(path, "*.log")) + glob.glob(os.path.join(path, "*.txt"))
    if not logs:
        logs = [path] if os.path.isfile(path) else []
    cfg, steps = None, []
    for lg in logs:
        with open(lg, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                m = CFG_RE.match(line)
                if m:
                    cfg = _kv(m.group(1))
                    continue
                m = STEP_RE.match(line)
                if m:
                    steps.append(_kv(m.group(1)))
    if cfg is None or not steps:
        return None
    rec = {k: np.array([s[k] for s in steps]) for k in steps[0]}
    return dict(path=path, cfg=cfg, **rec)


def potential_key(cfg):
    return ("dw", 3.0) if cfg.get("potential", "").startswith("double") else ("log", cfg.get("omega", 3.0))


def summarise(run):
    cfg = run["cfg"]
    pot, om = potential_key(cfg)
    gamma = cfg["gamma"]
    R0 = cfg["r0"]
    d = derived(pot, gamma, om)
    rc = R_c(pot, gamma, 1.0, om)
    R = run["R_eff"]
    survived = R[-1] > 0.25 * R0
    mass = run["mass"]
    return dict(
        path=os.path.basename(run["path"]),
        potential=cfg.get("potential"),
        omega=om,
        face=cfg.get("face_avg"),
        mobility=cfg.get("mobility"),
        grid=int(cfg.get("grid", 0)),
        R0=R0,
        gamma=gamma,
        eps=d["eps"],
        Lambda=d["Lambda"],
        R_c=rc,
        R0_over_Rc=R0 / rc,
        R_final=R[-1],
        R_final_over_R0=R[-1] / R0,
        R_inf_pred=R_inf_over_R0(pot, gamma, R0, 1.0, om),
        survived=survived,
        mass_drift=abs(mass[-1] - mass[0]) / max(abs(mass[0]), 1e-300),
        phi_max=run["phi_max"].max(),
        phi_max_pred=d["phi_eq"] + 2.0 * d["Lambda"] * d["phi_eq"] * d["eps"] / R0,
        t_end=run["t"][-1],
        newton_mean=run["newton"].mean(),
    )


def table(rows):
    cols = ["path", "potential", "omega", "mobility", "face", "grid", "R0", "eps", "Lambda",
            "R_c", "R0_over_Rc", "R_final_over_R0", "R_inf_pred", "mass_drift",
            "phi_max", "phi_max_pred", "newton_mean"]
    widths = {c: max(len(c), 11) for c in cols}
    print(" ".join(c.rjust(widths[c]) for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append((v if isinstance(v, str) else
                          f"{v:.4g}" if isinstance(v, float) else str(v)).rjust(widths[c]))
        print(" ".join(cells))


def figures(runs, out):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cmap = plt.get_cmap("viridis")
    n = max(1, len(runs) - 1)
    for i, run in enumerate(sorted(runs, key=lambda r: r["cfg"]["r0"])):
        c = cmap(i / n)
        lbl = f"R0={run['cfg']['r0']:.3f}"
        ax[0, 0].plot(run["t"], run["R_eff"] / run["cfg"]["r0"], color=c, label=lbl)
        ax[0, 1].plot(run["t"], run["phi_max"], color=c)
        ax[1, 0].plot(run["t"], run["mass"] - run["mass"][0], color=c)
        ax[1, 1].plot(run["t"], run["phobic"] + run["philic"], color=c)

    ax[0, 0].set(xlabel="t", ylabel=r"$R(t)/R_0$", title="drop radius")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].set(xlabel="t", ylabel=r"$\max\phi$", title="interior overshoot")
    ax[1, 0].set(xlabel="t", ylabel=r"$\int\phi-\int\phi_0$", title="mass drift (must be ~0)")
    ax[1, 1].set(xlabel="t", ylabel=r"$F$", title="total free energy (must decrease)")
    for a in ax.ravel():
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def master_curve(x):
    """
    R_inf/R0 as a function of R0/R_c alone.

    r(1-r^3) = A with A = 3 Lambda eps V / (4 pi R0^4), and R_c^4 = 3 Lambda eps V / (4 pi * 0.47247),
    so A = 0.47247 * (R_c/R0)^4 -- every potential, gamma and domain size collapses onto one curve.
    """
    A = 0.47247 / x ** 4
    if A > 0.47247:
        return 0.0
    lo, hi = 4.0 ** (-1.0 / 3.0), 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * (1.0 - mid ** 3) - A > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def figure_collapse(rows, out):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = np.linspace(0.4, 1.8, 600)
    ax.plot(x, [master_curve(xi) for xi in x], "k-", lw=2,
            label=r"theory: $r(1-r^3)=0.472\,(R_c/R_0)^4$")
    seen_lbl = set()
    for r in rows:
        lbl = f"{r['potential']}/{r['mobility']}/{r['face']}"
        ax.plot(r["R0_over_Rc"], r["R_final_over_R0"], "o", ms=9, color="crimson",
                label=lbl if lbl not in seen_lbl else None)
        seen_lbl.add(lbl)
    ax.axvline(1.0, color="steelblue", ls="--", lw=1.5, label=r"$R_0=R_c$ (predicted)")
    ax.set(xlabel=r"$R_0/R_c$", ylabel=r"$R_\infty/R_0$", ylim=(-0.05, 1.05),
           title="measured final radius vs the generalised criterion")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="../figs/runs.png")
    args = ap.parse_args()

    runs = [r for p in args.paths for r in [load_run(p)] if r]
    if not runs:
        raise SystemExit("no parseable runs found")
    rows = [summarise(r) for r in runs]
    table(rows)
    figures(runs, args.out)
    figure_collapse(rows, args.out.replace(".png", "_collapse.png"))
