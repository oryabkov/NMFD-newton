#!/usr/bin/env python3
"""
Exact (nonlinear) shrinkage criterion.

The linearised criterion of notes/03 assumes both bulk phases respond with the same tangent
stiffness f''(phi_eq). That is fine for the double well but fails for the logarithmic potential,
whose f'' rises steeply towards the singular wall and falls steeply towards 0: the drop interior
(shifting towards +1) stiffens, while the *exterior* (shifting towards 0) softens -- and it is the
exterior, with essentially the whole domain volume, that absorbs the released mass.

Exact statement. With mu the (uniform) chemical potential,

    f'(phi_eq + d_in)  = mu          (drop interior)
    f'(-phi_eq + d_out) = mu         (exterior)
    mu = (d-1) sigma / (2 phi_eq R)  (Gibbs-Thomson)

and mass conservation

    2 phi_eq (V_d0 - V_d) = d_in V_d + d_out (V - V_d).

An equilibrium radius is the largest R < R0 solving this; if none exists the drop evaporates.

The exterior branch also has a ceiling: f' on (-phi_eq, 0) peaks at the spinodal point. If mu
exceeds that peak the ambient phase cannot hold the supersaturation at all.
"""

import functools

import numpy as np

from potentials import dw_df, log_df, phi_eq as log_phi_eq


@functools.lru_cache(maxsize=None)
def potential(kind, omega=3.0):
    if kind == "dw":
        return dict(
            df=lambda p: np.asarray(p, dtype=float) ** 3 - np.asarray(p, dtype=float),
            phi_eq=1.0,
            k=2.0,
            spinodal=1.0 / np.sqrt(3.0),
            sigma_hat=2.0 * np.sqrt(2.0) / 3.0,
            phi_top=4.0,
        )
    pe = log_phi_eq(omega)
    return dict(
        df=lambda p: np.log((1.0 + np.asarray(p, dtype=float)) / (1.0 - np.asarray(p, dtype=float)))
        - omega * np.asarray(p, dtype=float),
        phi_eq=pe,
        k=2.0 / (1.0 - pe ** 2) - omega,
        spinodal=np.sqrt(max(0.0, 1.0 - 2.0 / omega)),
        sigma_hat=None,
        phi_top=1.0 - 1e-14,
    )


def _bisect(f, lo, hi, n=60):
    flo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (flo > 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def shifts(pot, mu):
    """(d_in, d_out) for a given chemical potential; d_out is None past the spinodal."""
    pe, df, sp = pot["phi_eq"], pot["df"], pot["spinodal"]

    d_in = _bisect(lambda x: df(x) - mu, pe, pot["phi_top"]) - pe

    mu_max = df(-sp) if -sp > -pe else df(np.array(-pe + 1e-9))
    mu_max = float(np.atleast_1d(mu_max)[0]) if np.ndim(mu_max) else float(mu_max)
    if mu >= mu_max:
        return d_in, None
    d_out = _bisect(lambda x: df(x) - mu, -pe, -sp) + pe
    return d_in, d_out


@functools.lru_cache(maxsize=None)
def sigma_of(kind, gamma, omega=3.0):
    from potentials import dw_f, dw_summary, log_f, summary
    d = dw_summary() if kind == "dw" else summary(omega)
    return d["sigma_hat"] * np.sqrt(gamma)


def residual(kind, gamma, R0, R, V=1.0, omega=3.0, dim=3):
    """>0: released mass exceeds what the Gibbs-Thomson state needs (drop would regrow)."""
    pot = potential(kind, omega)
    pe = pot["phi_eq"]
    sigma = sigma_of(kind, gamma, omega)
    mu = (dim - 1) * sigma / (2.0 * pe * R)
    d_in, d_out = shifts(pot, mu)
    if d_out is None:
        return -np.inf
    Vd0 = 4.0 * np.pi * R0 ** 3 / 3.0
    Vd = 4.0 * np.pi * R ** 3 / 3.0
    return 2.0 * pe * (Vd0 - Vd) - d_in * Vd - d_out * (V - Vd)


def R_inf(kind, gamma, R0, V=1.0, omega=3.0, dim=3, n=240):
    """Largest equilibrium radius below R0; 0.0 if the drop evaporates."""
    Rs = np.linspace(R0 * (1.0 - 1e-6), 1e-4, n)
    g = np.array([residual(kind, gamma, R0, r, V, omega, dim) for r in Rs])
    sign = np.sign(g)
    idx = np.nonzero((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    if len(idx) == 0:
        return 0.0
    a, b = Rs[idx[0]], Rs[idx[0] + 1]
    return _bisect(lambda r: residual(kind, gamma, R0, r, V, omega, dim), a, b, 40)


def R_c(kind, gamma, V=1.0, omega=3.0, dim=3, lo=0.05, hi=0.60):
    """Smallest R0 that still admits an equilibrium."""
    if R_inf(kind, gamma, hi, V, omega, dim) == 0.0:
        return np.nan
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if R_inf(kind, gamma, mid, V, omega, dim) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    from predict import R_c as R_c_linear, gamma_for_eps

    eps = 0.04
    print(f"{'potential':>18} {'gamma':>10} {'R_c linear':>11} {'R_c exact':>10} {'ratio':>7}")
    for kind, om in [("dw", 3.0), ("log", 2.5), ("log", 3.0), ("log", 4.0), ("log", 6.0)]:
        g = gamma_for_eps(kind, eps, om)
        lin = R_c_linear(kind, g, 1.0, om)
        ex = R_c(kind, g, 1.0, om)
        name = "double well" if kind == "dw" else f"log omega={om:g}"
        print(f"{name:>18} {g:10.4e} {lin:11.4f} {ex:10.4f} {ex / lin:7.3f}")
