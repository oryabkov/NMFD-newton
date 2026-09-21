#!/usr/bin/env python3
"""
Bulk (phobic) potentials used by the Cahn-Hilliard solver.

Double well (kernels/phobic_energy.h, double_well_potential):
    f(p)   = 1/4 (p^2 - 1)^2
    f'(p)  = p^3 - p
    f''(p) = 3 p^2 - 1

Logarithmic / regular-solution (kernels/phobic_energy.h, logarithmic_potential):
    f(p)   = (1+p) ln(1+p) + (1-p) ln(1-p) - w/2 p^2
    f'(p)  = ln((1+p)/(1-p)) - w p
    f''(p) = 2/(1-p^2) - w

Total free energy is F = int [ f(phi) + gamma/2 |grad phi|^2 ].
"""

import numpy as np

OMEGAS = (2.1, 2.2, 2.5, 3.0, 4.0, 6.0, 10.0)


def dw_f(p):
    return 0.25 * (p * p - 1.0) ** 2


def dw_df(p):
    return p ** 3 - p


def dw_ddf(p):
    return 3.0 * p * p - 1.0


def log_f(p, w):
    p = np.asarray(p, dtype=float)
    out = np.full(np.shape(p), np.nan)
    m = np.abs(p) < 1.0
    q = p[m] if out.ndim else p
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (1 + q) * np.log(1 + q) + (1 - q) * np.log(1 - q) - 0.5 * w * q * q
    if out.ndim:
        out[m] = val
        out[np.abs(p) == 1.0] = np.log(2.0) * 2.0 * 0.5 * 2 - 0.5 * w  # 2 ln2 - w/2
        return out
    return val


def log_df(p, w):
    p = np.asarray(p, dtype=float)
    out = np.full(np.shape(p), np.nan)
    m = np.abs(p) < 1.0
    out[m] = np.log((1 + p[m]) / (1 - p[m])) - w * p[m]
    return out


def log_ddf(p, w):
    p = np.asarray(p, dtype=float)
    out = np.full(np.shape(p), np.nan)
    m = np.abs(p) < 1.0
    out[m] = 2.0 / (1.0 - p[m] ** 2) - w
    return out


def phi_eq(w, tol=1e-15):
    """Positive root of f'(p) = 0, i.e. ln((1+p)/(1-p)) = w p. Exists for w > 2."""
    if w <= 2.0:
        return 0.0
    lo, hi = 1e-12, 1.0 - 1e-15
    g = lambda p: np.log((1 + p) / (1 - p)) - w * p
    # g(lo) < 0 for w>2 (slope 2-w<0), g(hi) -> +inf
    assert g(lo) < 0 < g(hi)
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def surface_tension_coeff(f, pe, n=200001):
    """sigma = sqrt(gamma) * int_{-pe}^{pe} sqrt(2 (f(p)-f(pe))) dp  -> returns the integral."""
    p = np.linspace(-pe, pe, n)
    d = f(p) - f(pe)
    d = np.clip(d, 0.0, None)
    return np.trapezoid(np.sqrt(2.0 * d), p)


def interface_halfwidth(f, pe, frac=0.9, n=200001):
    """x at which phi = frac*pe on the planar profile, in units of sqrt(gamma)."""
    p = np.linspace(0.0, frac * pe, n)
    d = np.clip(f(p) - f(pe), 1e-300, None)
    return np.trapezoid(1.0 / np.sqrt(2.0 * d), p)


def summary(w):
    pe = phi_eq(w)
    f = lambda p: log_f(p, w)
    return {
        "omega": w,
        "phi_eq": pe,
        "ddf_eq": 2.0 / (1.0 - pe ** 2) - w,
        "barrier": log_f(np.array([0.0]), w)[0] - log_f(np.array([pe]), w)[0],
        "sigma_hat": surface_tension_coeff(f, pe),
        "eps_hat": 1.0 / np.sqrt(2.0 / (1.0 - pe ** 2) - w),
        "w90_hat": interface_halfwidth(f, pe),
        "headroom": 1.0 / pe - 1.0,
    }


def dw_summary():
    pe = 1.0
    return {
        "omega": np.nan,
        "phi_eq": pe,
        "ddf_eq": 2.0,
        "barrier": 0.25,
        "sigma_hat": surface_tension_coeff(dw_f, pe),
        "eps_hat": 1.0 / np.sqrt(2.0),
        "w90_hat": interface_halfwidth(dw_f, pe),
        "headroom": np.inf,
    }
