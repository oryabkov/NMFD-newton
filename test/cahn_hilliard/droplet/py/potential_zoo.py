#!/usr/bin/env python3
"""
Candidate bulk potentials for suppressing spontaneous drop evaporation.

Design target (notes/08). At fixed *resolved interface width* W the exact
threshold radius depends on the potential only through the dimensionless
shape number

    K[f] = sigma_hat / ( phi_eq^2 * f''(phi_eq) * w_hat ),

    sigma_hat = int_{-phi_eq}^{phi_eq} sqrt(2 (f - f_min)) dphi     (sigma = sigma_hat sqrt(gamma))
    w_hat     = int_0^{0.9 phi_eq} dphi / sqrt(2 (f - f_min))       (half width, units sqrt(gamma))

and R_c^4 = 0.2527 V W K. K is invariant under f -> A f and phi -> a phi, so every
potential below is first normalised to phi_eq = 1 and barrier f(0) - f(1) = 1/4;
that changes nothing physical and makes the shapes directly comparable on one plot.

The exact (nonlinear) criterion of criterion.py is re-implemented here generically,
because the candidates have f'' -> inf at the bulk values and the linear response
delta = psi / f''(phi_eq) degenerates.
"""

import numpy as np

from potentials import log_f, log_df, phi_eq as log_phi_eq

DIM = 3
VOL = 1.0
WIDTH = 0.1          # full 90 % interface width, in units of the unit box
PHI_SAMP = 400001


# ----------------------------------------------------------------- potentials


def dw_pair(p=2.0):
    """f = |1 - phi^2|^p / (2p).  p = 2 is the classical double well (up to a factor).

    p < 2 puts a cusp at phi = +-1: f''(+-1) = inf and the equilibrium profile
    reaches +-1 at a *finite* distance (compact support).  p -> 1 is the tent.
    """

    def f(x):
        x = np.asarray(x, dtype=float)
        return np.abs(1.0 - x * x) ** p / (2.0 * p)

    def df(x):
        x = np.asarray(x, dtype=float)
        s = 1.0 - x * x
        return -x * np.abs(s) ** (p - 1.0) * np.sign(s)

    return f, df


def obstacle_pair(theta=1.0):
    """Double obstacle, Oono-Puri / Blowey-Elliott: f = theta/2 (1 - phi^2) + I_[-1,1].

    Concave inside, minima pinned at the constraint.  Returned f is the smooth part;
    the caller must treat phi = +-1 as a corner (subdifferential), see `Candidate`.
    """

    def f(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * theta * (1.0 - x * x)

    def df(x):
        x = np.asarray(x, dtype=float)
        return -theta * x

    return f, df


def smooth_obstacle_pair(eta=0.15):
    """Smoothed tent: f = 1/2 ( sqrt((1-phi^2)^2 + eta^4) - eta^2 ).

    Minima exactly at +-1 with f''(+-1) = 2/eta^2.  eta -> 0 recovers the obstacle;
    eta = 1 is a soft flat-bottomed well.  Everywhere C^infty, so an ordinary Newton
    /multigrid solver applies -- no variational inequality.
    """

    e4 = eta ** 4

    def f(x):
        x = np.asarray(x, dtype=float)
        s = 1.0 - x * x
        return 0.5 * (np.sqrt(s * s + e4) - eta * eta)

    def df(x):
        x = np.asarray(x, dtype=float)
        s = 1.0 - x * x
        return -x * s / np.sqrt(s * s + e4)

    return f, df


def log_pair(omega):
    pe = log_phi_eq(omega)

    def f(x):
        return log_f(np.asarray(x, dtype=float), omega)

    def df(x):
        return log_df(np.asarray(x, dtype=float), omega)

    return f, df, pe


# ------------------------------------------------------------- normalisation


class Candidate:
    """A potential normalised to phi_eq = 1, f(+-1) = 0, f(0) = 1/4."""

    def __init__(self, name, f, df, phi_eq_raw, phi_top_raw, obstacle=False, style=None):
        self.name = name
        self.obstacle = obstacle
        self.style = style or {}
        pe = phi_eq_raw
        fmin = float(np.atleast_1d(f(pe))[0])
        barrier = float(np.atleast_1d(f(0.0))[0]) - fmin
        self.scale = 1.0 / (4.0 * barrier)
        self._f, self._df, self._pe = f, df, pe
        self.phi_top = phi_top_raw / pe
        self._cache = {}

    def f(self, x):
        x = np.asarray(x, dtype=float)
        return (self._f(x * self._pe) - self._f(self._pe)) * self.scale

    def df(self, x):
        x = np.asarray(x, dtype=float)
        return self._df(x * self._pe) * self._pe * self.scale

    # ---- shape scalars

    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def sigma_hat(self):
        def go():
            x = np.linspace(-1.0, 1.0, PHI_SAMP)
            return np.trapezoid(np.sqrt(2.0 * np.clip(self.f(x), 0.0, None)), x)
        return self._memo("sigma_hat", go)

    def w_hat(self, frac=0.9):
        def go():
            x = np.linspace(0.0, frac, PHI_SAMP)
            return np.trapezoid(1.0 / np.sqrt(2.0 * np.clip(self.f(x), 1e-300, None)), x)
        return self._memo(("w_hat", frac), go)

    def gamma(self, width=WIDTH):
        return (0.5 * width / self.w_hat()) ** 2

    def sigma(self, width=WIDTH):
        return self.sigma_hat() * np.sqrt(self.gamma(width))

    def spinodal(self):
        """(phi*, psi_max) -- the peak of f' on (-1, 0); past psi_max the ambient
        cannot hold the supersaturation at all."""
        def go():
            if self.obstacle:
                return -1.0, float(self.df(-1.0 + 1e-14))
            x = np.linspace(-1.0 + 1e-12, -1e-9, 200001)
            g = self.df(x)
            i = int(np.nanargmax(g))
            lo, hi = x[max(i - 1, 0)], x[min(i + 1, len(x) - 1)]
            xx = np.linspace(lo, hi, 2001)
            gg = self.df(xx)
            j = int(np.nanargmax(gg))
            return float(xx[j]), float(gg[j])
        return self._memo("spinodal", go)

    # ---- response of each bulk phase to a chemical potential psi

    def shifts(self, psi):
        """(d_in, d_out); d_out = None when psi exceeds the ambient's ceiling."""
        _, psi_max = self.spinodal()
        if self.obstacle:
            # corner at +-1: the subdifferential spans an interval, so *no* mass
            # moves in or out while psi stays inside it.
            return (0.0, 0.0) if psi <= psi_max else (0.0, None)
        d_in = _bisect(lambda x: self.df(x) - psi, 1.0, self.phi_top) - 1.0
        if psi >= psi_max:
            return d_in, None
        sp, _ = self.spinodal()
        return d_in, _bisect(lambda x: self.df(x) - psi, -1.0, sp) + 1.0


def _bisect(fn, lo, hi, n=80):
    flo = fn(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if (fn(mid) > 0) == (flo > 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ criterion


def gamma_for(c, width=None, eps=None):
    """gamma fixed either by the resolved 90 % interface width, or by the tail length
    eps = sqrt(gamma / f''(phi_eq)) used elsewhere in this study."""
    if width is not None:
        return (0.5 * width / c.w_hat()) ** 2
    return eps ** 2 * curvature(c)


def residual(c, R0, R, gamma, V=VOL, dim=DIM):
    psi = (dim - 1) * (c.sigma_hat() * np.sqrt(gamma)) / (2.0 * R)
    d_in, d_out = c.shifts(psi)
    if d_out is None:
        return -np.inf
    Vd0 = 4.0 * np.pi * R0 ** 3 / 3.0
    Vd = 4.0 * np.pi * R ** 3 / 3.0
    return 2.0 * (Vd0 - Vd) - d_in * Vd - d_out * (V - Vd)


def R_inf(c, R0, gamma, V=VOL, dim=DIM, n=400):
    Rs = np.linspace(R0 * (1.0 - 1e-7), 1e-5, n)
    g = np.array([residual(c, R0, r, gamma, V, dim) for r in Rs])
    if g[0] >= 0.0:
        return R0          # the ambient demands nothing: the drop does not shrink at all
    s = np.sign(g)
    idx = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(idx) == 0:
        return 0.0
    return _bisect(lambda r: residual(c, R0, r, gamma, V, dim), Rs[idx[0]], Rs[idx[0] + 1], 50)


def R_c(c, gamma, V=VOL, dim=DIM, lo=1e-4, hi=0.62):
    """Smallest R0 that still admits an equilibrium; 0 if every drop survives."""
    if R_inf(c, hi, gamma, V, dim) == 0.0:
        return np.nan
    if R_inf(c, lo, gamma, V, dim) > 0.0:
        return 0.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if R_inf(c, mid, gamma, V, dim) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def curvature(c, h=1e-6):
    """f''(1^-) by a one-sided difference of f'; inf for cusped/obstacle wells."""
    if c.obstacle:
        return np.inf
    return float((c.df(1.0) - c.df(1.0 - h)) / h)


def build(width=WIDTH):
    cs = []
    f, df = dw_pair(2.0)
    cs.append(Candidate("двойная яма  $p=2$", f, df, 1.0, 4.0,
                        style=dict(color="0.35", lw=3.0, zorder=2)))
    f, df, pe = log_pair(3.0)
    cs.append(Candidate(r"логарифмический  $\omega=3$", f, df, pe, 1.0 - 1e-13,
                        style=dict(color="#d95f02", lw=2.0)))
    for p, col in ((1.5, "#1b9e77"), (1.2, "#7570b3")):
        f, df = dw_pair(p)
        cs.append(Candidate(f"со сломом  $p={p}$", f, df, 1.0, 4.0,
                            style=dict(color=col, lw=2.0)))
    f, df = smooth_obstacle_pair(0.15)
    cs.append(Candidate(r"сглаженное препятствие  $\eta=0.15$", f, df, 1.0, 4.0,
                        style=dict(color="#e7298a", lw=2.0, ls="-.")))
    f, df = obstacle_pair(1.0)
    cs.append(Candidate("двойное препятствие", f, df, 1.0, 1.0, obstacle=True,
                        style=dict(color="k", lw=2.2, ls="--")))
    return cs


def table(cands=None, width=WIDTH, eps=None):
    rows = []
    for c in (cands or build()):
        k = curvature(c)
        g = gamma_for(c, width=None if eps else width, eps=eps)
        sig = c.sigma_hat() * np.sqrt(g)
        rows.append(dict(
            name=c.name,
            sigma_hat=c.sigma_hat(),
            w_hat=c.w_hat(),
            ddf=k,
            K=c.sigma_hat() / (k * c.w_hat()) if np.isfinite(k) else 0.0,
            gamma=g,
            sigma=sig,
            width=2.0 * c.w_hat() * np.sqrt(g),
            eps=np.sqrt(g / k) if np.isfinite(k) else 0.0,
            psi_max=c.spinodal()[1],
            R_floor=sig * (DIM - 1) / (2.0 * c.spinodal()[1]),
            Rc=R_c(c, g),
            cand=c,
        ))
    return rows


def _show(rows, title):
    ref = rows[0]["Rc"]
    print(f"\n{title}\n")
    print(f"{'потенциал':>36} {'sigma^':>7} {'w^':>7} {'f\"(1)':>9} {'K':>8} "
          f"{'eps':>8} {'psi_max':>8} {'R_потолок':>9} {'R_c':>8} {'отн.':>7} {'отн.объём':>9}")
    for r in rows:
        rc = r["Rc"]
        rcs = "испар." if np.isnan(rc) else ("0 (нет)" if rc == 0.0 else f"{rc:8.4f}")
        rel = "—" if np.isnan(rc) else f"{rc / ref:7.3f}"
        vol = "—" if np.isnan(rc) else f"{(rc / ref) ** 3:9.4f}"
        print(f"{r['name']:>36} {r['sigma_hat']:7.4f} {r['w_hat']:7.4f} {r['ddf']:9.3g} "
              f"{r['K']:8.4f} {r['eps']:8.5f} {r['psi_max']:8.4f} {r['R_floor']:9.4f} "
              f"{rcs:>8} {rel:>7} {vol:>9}")


if __name__ == "__main__":
    import sys

    _show(table(width=WIDTH),
          f"A. при фиксированной разрешаемой ширине границы W = {WIDTH} (куб 1x1x1, 3D)")
    _show(table(eps=0.04),
          "B. при фиксированной длине хвоста eps = 0.04 — сверка с criterion.py")

    if "--sweep" in sys.argv:
        print("\nC. развёртка по параметру регуляризации (фикс. W = 0.1)\n")
        print(f"{'семейство':>28} {'параметр':>9} {'f\"(1)':>10} {'R_c':>9} {'отн.':>7} {'отн.объём':>10}")
        ref = table(width=WIDTH)[0]["Rc"]
        for p in (2.0, 1.8, 1.6, 1.4, 1.2, 1.1, 1.05):
            f, df = dw_pair(p)
            c = Candidate("", f, df, 1.0, 4.0)
            r = table([c], width=WIDTH)[0]
            print(f"{'слом  |1-phi^2|^p':>28} {p:9.2f} {r['ddf']:10.4g} {r['Rc']:9.4f} "
                  f"{r['Rc'] / ref:7.3f} {(r['Rc'] / ref) ** 3:10.4f}")
        for e in (1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02):
            f, df = smooth_obstacle_pair(e)
            c = Candidate("", f, df, 1.0, 4.0)
            r = table([c], width=WIDTH)[0]
            print(f"{'сглаж. препятствие  eta':>28} {e:9.2f} {r['ddf']:10.4g} {r['Rc']:9.4f} "
                  f"{r['Rc'] / ref:7.3f} {(r['Rc'] / ref) ** 3:10.4f}")
