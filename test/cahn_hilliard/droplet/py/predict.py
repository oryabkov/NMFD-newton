#!/usr/bin/env python3
"""
Quantitative predictions of notes/03_shrinkage_criterion.md, and the run matrix they imply.

    eps    = sqrt(gamma / f''(phi_eq))
    Lambda = sigma / (2 phi_eq^2 f''(phi_eq) eps)
    R_c    = (0.5053 Lambda eps V)^(1/4)                       (3D)
    R_inf  = R0 * larger root of  r(1-r^3) = 3 Lambda eps V / (4 pi R0^4)

Usage:
    python3 predict.py                 # run matrix for the planned sweeps
    python3 predict.py --table         # R_c for a grid of (potential, gamma)
"""

import argparse

import numpy as np

from potentials import dw_summary, phi_eq, summary

C3 = 0.5053
C2 = 0.4135


def shape(potential, omega=3.0):
    d = dw_summary() if potential == "dw" else summary(omega)
    k = d["ddf_eq"]
    return dict(phi_eq=d["phi_eq"], k=k, sigma_hat=d["sigma_hat"], headroom=d["headroom"])


def derived(potential, gamma, omega=3.0):
    s = shape(potential, omega)
    eps = np.sqrt(gamma / s["k"])
    sigma = s["sigma_hat"] * np.sqrt(gamma)
    lam = sigma / (2.0 * s["phi_eq"] ** 2 * s["k"] * eps)
    return dict(eps=eps, sigma=sigma, Lambda=lam, **s)


def gamma_for_eps(potential, eps, omega=3.0):
    return eps ** 2 * shape(potential, omega)["k"]


def R_c(potential, gamma, V=1.0, omega=3.0, dim=3):
    d = derived(potential, gamma, omega)
    if dim == 3:
        return (C3 * d["Lambda"] * d["eps"] * V) ** 0.25
    return (C2 * d["Lambda"] * d["eps"] * V) ** (1.0 / 3.0)


def R_inf_over_R0(potential, gamma, R0, V=1.0, omega=3.0):
    """Larger root of r(1-r^3) = A; returns 0.0 (full evaporation) when no root exists."""
    d = derived(potential, gamma, omega)
    A = 3.0 * d["Lambda"] * d["eps"] * V / (4.0 * np.pi * R0 ** 4)
    if A > 4.0 ** (-1.0 / 3.0) * (1.0 - 0.25):
        return 0.0
    lo, hi = 4.0 ** (-1.0 / 3.0), 1.0
    g = lambda r: r * (1.0 - r ** 3) - A
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def overshoot(potential, gamma, R, omega=3.0, dim=3):
    d = derived(potential, gamma, omega)
    return (dim - 1) * d["Lambda"] * d["phi_eq"] * d["eps"] / R


def run_matrix():
    print("=" * 96)
    print("Run matrix: unit box L=1 (V=1), interface resolved with eps/h >= 2")
    print("=" * 96)
    print(f"{'potential':>16} {'omega':>6} {'eps':>8} {'gamma':>10} {'sigma':>9} "
          f"{'Lambda':>8} {'R_c':>7} {'min grid':>9} {'overshoot@R_c':>14}")
    for eps in (0.04, 0.02, 0.01):
        print("-" * 96)
        for pot, om in [("dw", None), ("log", 2.5), ("log", 3.0), ("log", 4.0), ("log", 6.0)]:
            om_ = 3.0 if om is None else om
            g = gamma_for_eps(pot, eps, om_)
            d = derived(pot, g, om_)
            rc = R_c(pot, g, 1.0, om_)
            n = int(np.ceil(2.0 / eps))
            n = 1 << int(np.ceil(np.log2(n)))
            ov = overshoot(pot, g, rc, om_)
            name = "double well" if pot == "dw" else "logarithmic"
            head = "-" if pot == "dw" else f"{d['headroom']:.3f}"
            print(f"{name:>16} {'-' if om is None else f'{om:g}':>6} {eps:8.4f} {g:10.3e} "
                  f"{d['sigma']:9.5f} {d['Lambda']:8.4f} {rc:7.4f} {n:9d} "
                  f"{ov:9.4f} (hr {head})")
    print("=" * 96)
    print("R_c is the 3D critical radius: a drop with R0 < R_c evaporates completely.")
    print("'overshoot@R_c' is max(phi)-phi_eq predicted at R = R_c; 'hr' is the headroom")
    print("1/phi_eq-1 of the logarithmic potential (its singular wall bites only when")
    print("overshoot > hr).")


def sweep_prediction(potential="dw", eps=0.02, omega=3.0, V=1.0):
    g = gamma_for_eps(potential, eps, omega)
    rc = R_c(potential, g, V, omega)
    print()
    print(f"Predicted R_inf/R0 sweep  [{potential} omega={omega:g}, eps={eps}, gamma={g:.4e}, "
          f"R_c={rc:.4f}]")
    print(f"{'R0':>8} {'R0/R_c':>8} {'R_inf/R0':>10} {'R_inf':>8} {'outcome':>12}")
    for R0 in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.32, 0.36):
        r = R_inf_over_R0(potential, g, R0, V, omega)
        out = "evaporates" if r == 0.0 else "survives"
        print(f"{R0:8.3f} {R0 / rc:8.3f} {r:10.4f} {r * R0:8.4f} {out:>12}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.02)
    args = ap.parse_args()
    run_matrix()
    sweep_prediction("dw", args.eps)
    sweep_prediction("log", args.eps, 3.0)
