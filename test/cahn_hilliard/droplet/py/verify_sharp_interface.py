#!/usr/bin/env python3
"""Check the sharp-interface theory of notes/cahn_hilliard_spherical_droplet_analysis.md
against every double-well run on disk.

The note states the theory for a unit ball; the cube version is the same algebra with the
area/volume ratios put back explicitly.  With V = 1, a(R) = 4 pi R^2, v(R) = 4/3 pi R^3:

    m       = 1 + phibar                                     excess over the pure -1 phase
    delta   = m - A R^3           - C gamma R                bulk shift, A = 8 pi/3, C = 4 pi^3/3
    E(R)    = 4 pi sigma R^2 + (k/2) delta^2
    E'(R)=0 <=> h(R) = A R^4 - m R + sigma/k [+ C gamma R^2] = 0

The bracketed C-terms are the O(gamma) interface-mass correction (note section 3.6, pi^2 R gamma
in ball units).  They must be switched on or off in BOTH places at once: m comes from the initial
profile and h from the stationary one, and mixing the two is what produces a spurious 10% error.
"""

import glob
import os

import numpy as np

ROOT = os.environ.get(
    "DROPLET_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

A_COEF = 8.0 * np.pi / 3.0            # 2 v(R) / V for the unit cube
C_COEF = 4.0 * np.pi**3 / 3.0         # interface mass correction, phibar += C gamma R
K_DW = 2.0                            # f''(+-1) for f = (phi^2 - 1)^2 / 4
SIG_DW = 2.0 * np.sqrt(2.0) / 3.0     # sigma / sqrt(gamma)


def sigma(gamma):
    return SIG_DW * np.sqrt(gamma)


def m_of_r0(gamma, r0, corr=False):
    return A_COEF * r0**3 + (C_COEF * gamma * r0 if corr else 0.0)


def h(R, m, gamma, corr=False):
    return A_COEF * R**4 - m * R + sigma(gamma) / K_DW + (C_COEF * gamma * R**2 if corr else 0.0)


def _bisect(f, a, b, n=200):
    fa = f(a)
    for _ in range(n):
        c = 0.5 * (a + b)
        if fa * f(c) <= 0.0:
            b = c
        else:
            a, fa = c, f(c)
    return 0.5 * (a + b)


def roots_of(gamma, r0, corr=False):
    m = m_of_r0(gamma, r0, corr)
    Rs = np.linspace(1e-4, 0.5, 200001)
    v = h(Rs, m, gamma, corr)
    out = []
    for i in np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
        out.append(_bisect(lambda R: h(R, m, gamma, corr), Rs[i], Rs[i + 1]))
    return out


def r0_crit(gamma, corr=False):
    Rs = np.linspace(1e-4, 0.5, 20001)

    def min_h(r0):
        return h(Rs, m_of_r0(gamma, r0, corr), gamma, corr).min()

    return _bisect(min_h, 0.05, 0.48)


def m_min(gamma):
    B = sigma(gamma) / K_DW
    return (256.0 * A_COEF * B**3 / 27.0) ** 0.25


def m_eq(gamma):
    """E(R_*) = E(0): boundary between a metastable and a globally stable droplet."""
    R = (sigma(gamma) / (K_DW * A_COEF)) ** 0.25
    return 2.0 * sigma(gamma) / (K_DW * R)


def r0_of_m(m):
    return (m / A_COEF) ** (1.0 / 3.0)


def parse(run):
    cfg, rows = {}, []
    with open(os.path.join(run, "log.txt")) as fh:
        for line in fh:
            # adaptive-dt runs go through the logger, which prefixes every line with its level
            line = line.strip()
            if line.startswith("INFO:"):
                line = line[len("INFO:"):].strip()
            if line.startswith("DROPLET_CONFIG"):
                for tok in line.split()[1:]:
                    key, _, val = tok.partition("=")
                    try:
                        cfg[key] = float(val)
                    except ValueError:
                        cfg[key] = val
            elif line.startswith("DROPLET "):
                rows.append({k: float(v) for k, _, v in (t.partition("=") for t in line.split()[1:])})
    if not rows:
        return None
    cfg["run"] = os.path.basename(run.rstrip("/"))
    return cfg, {k: np.array([r[k] for r in rows]) for k in rows[0]}


def load_all():
    out = []
    for run in sorted(glob.glob(os.path.join(ROOT, "*/"))):
        got = parse(run)
        if got is None:
            continue
        cfg, data = got
        if cfg.get("potential") == "double_well" and cfg.get("bc") == "neumann" \
                and cfg.get("mobility") == "constant":
            out.append((cfg, data))
    return out


def head(n, title):
    print(f"\n{'=' * 104}\n{n}  {title}\n{'=' * 104}")


def main():
    runs = load_all()
    print(f"{len(runs)} double-well / neumann / constant-mobility runs on disk")

    head("T1", "initial mean value: phibar(0) vs A R0^3 - 1 + C gamma R0    [note 3.6]")
    print(f"{'run':38} {'R0':>6} {'gamma':>9} {'phibar':>11} {'sharp err':>11} {'+C g R0 err':>12}")
    for cfg, d in sorted({(c["r0"], c["gamma"], c["grid"]): (c, x) for c, x in runs}.values(),
                         key=lambda cd: (cd[0]["gamma"], cd[0]["r0"])):
        r0, g = cfg["r0"], cfg["gamma"]
        pb = d["mass"][0]
        sharp = m_of_r0(g, r0) - 1.0
        corr = m_of_r0(g, r0, True) - 1.0
        print(f"{cfg['run']:38} {r0:6.3f} {g:9.2e} {pb:11.6f} {pb - sharp:11.2e} {pb - corr:12.2e}")

    head("T2", "survival threshold vs the observed bracket")
    print(f"{'gamma':>9} {'xi':>7} {'xi/h(64)':>9} {'R0c sharp':>10} {'R0c O(g)':>9} "
          f"{'died<=':>8} {'lived>=':>8} {'verdict':>9}")
    by_g = {}
    for cfg, d in runs:
        if cfg["grid"] != 64 or cfg["dt"] != 2e-3:
            continue
        by_g.setdefault(cfg["gamma"], set()).add((cfg["r0"], d["drop_volume"][-1] > 0))
    for g in sorted(by_g):
        dead = [r for r, a in by_g[g] if not a]
        live = [r for r, a in by_g[g] if a]
        lo = max(dead) if dead else float("nan")
        hi = min(live) if live else float("nan")
        a, b = r0_crit(g), r0_crit(g, True)
        ok = "ok" if (np.isnan(lo) or lo < a) and (np.isnan(hi) or hi > a) else "MISS"
        print(f"{g:9.2e} {np.sqrt(2 * g):7.4f} {np.sqrt(2 * g) * 64:9.2f} {a:10.4f} {b:9.4f} "
              f"{lo:8.3f} {hi:8.3f} {ok:>9}")

    head("T3", "stationary radius of the survivors, both branches consistently")
    print(f"{'run':38} {'R0':>6} {'gamma':>9} {'R meas':>8} {'sharp':>8} {'err':>7} "
          f"{'O(g)':>8} {'err':>7} {'xi/R':>6}")
    for cfg, d in runs:
        if d["drop_volume"][-1] <= 0:
            continue
        g, r0, R = cfg["gamma"], cfg["r0"], d["R_eff"][-1]
        ra, rb = roots_of(g, r0), roots_of(g, r0, True)
        sa = ra[-1] if ra else float("nan")
        sb = rb[-1] if rb else float("nan")
        print(f"{cfg['run']:38} {r0:6.3f} {g:9.2e} {R:8.4f} {sa:8.4f} {R / sa - 1:7.2%} "
              f"{sb:8.4f} {R / sb - 1:7.2%} {np.sqrt(2 * g) / R:6.3f}")

    head("T4", "bulk shifts to second order: delta_+- = psi/2 -+ (3/8) psi^2,  psi = sigma/R")
    print(f"{'run':38} {'gamma':>9} {'R/eps':>6} | {'phimax-1':>9} {'1st':>8} {'2nd':>8} {'err':>7}"
          f" | {'phimin+1':>9} {'2nd':>8} {'err':>7}")
    for cfg, d in runs:
        if d["drop_volume"][-1] <= 0 or cfg["grid"] != 64:
            continue
        g, R = cfg["gamma"], d["R_eff"][-1]
        psi = sigma(g) / R
        dp, dm = psi / 2 - 0.375 * psi**2, psi / 2 + 0.375 * psi**2
        mp, mn = d["phi_max"][-1] - 1.0, d["phi_min"][-1] + 1.0
        print(f"{cfg['run']:38} {g:9.2e} {R / np.sqrt(g / K_DW):6.2f} | {mp:9.5f} {psi/2:8.5f} "
              f"{dp:8.5f} {(mp - dp) / dp:7.2%} | {mn:9.5f} {dm:8.5f} {(mn - dm) / dm:7.2%}")

    head("T5", "Gibbs-Thomson without logging psi:  psi = (phi_max - 1) + (phi_min + 1)")
    print(f"{'run':38} {'gamma':>9} {'R':>7} {'psi meas':>9} {'sigma/R':>9} {'ratio':>7} "
          f"{'(xi/R)^2':>9} {'dev/(xi/R)^2':>13}")
    for cfg, d in runs:
        if d["drop_volume"][-1] <= 0 or cfg["grid"] != 64:
            continue
        g, R = cfg["gamma"], d["R_eff"][-1]
        if R / np.sqrt(g / K_DW) < 10:
            continue
        psi = (d["phi_max"][-1] - 1.0) + (d["phi_min"][-1] + 1.0)
        pr, xi2 = sigma(g) / R, 2.0 * g / R**2
        print(f"{cfg['run']:38} {g:9.2e} {R:7.4f} {psi:9.5f} {pr:9.5f} {psi / pr:7.4f} "
              f"{xi2:9.5f} {(psi / pr - 1) / xi2:13.2f}")

    head("T6", "the landscape: m against m_min (existence) and m_eq (global stability)")
    print(f"{'gamma':>9} {'m_min':>8} {'m_eq':>8} {'R0 crit':>8} {'R0 eq':>7} {'window':>7} "
          f"{'R_f':>7} {'R_eq':>7}")
    for g in sorted({c["gamma"] for c, _ in runs}):
        mm, me = m_min(g), m_eq(g)
        print(f"{g:9.2e} {mm:8.5f} {me:8.5f} {r0_of_m(mm):8.4f} {r0_of_m(me):7.4f} "
              f"{r0_of_m(me) / r0_of_m(mm) - 1:7.2%} "
              f"{(mm / (4 * A_COEF)) ** (1/3.):7.4f} {(sigma(g)/(K_DW*A_COEF))**0.25:7.4f}")
    print("\n  free energy of the survivors against the homogeneous state:")
    print(f"{'run':38} {'m':>8} {'F meas':>9} {'(k/2)m^2':>9} {'f(phibar)':>10} {'verdict':>22}")
    for cfg, d in runs:
        if d["drop_volume"][-1] <= 0 or cfg["grid"] != 64:
            continue
        m = 1.0 + d["mass"][0]
        F = d["phobic"][-1] + d["philic"][-1]
        quad = 0.5 * K_DW * m**2
        exact = 0.25 * (d["mass"][0] ** 2 - 1.0) ** 2
        print(f"{cfg['run']:38} {m:8.5f} {F:9.5f} {quad:9.5f} {exact:10.5f} "
              f"{('droplet wins' if F < exact else 'homogeneous wins'):>22}")

    head("T7", "does anything grow?  h(R0) = sigma/k > 0 identically, so nothing should")
    grew = [cfg["run"] for cfg, d in runs if np.any(np.diff(d["R_eff"]) > 1e-4)]
    print(f"  runs with a growing radius: {len(grew)} of {len(runs)}" + (f" -> {grew}" if grew else ""))


if __name__ == "__main__":
    main()
