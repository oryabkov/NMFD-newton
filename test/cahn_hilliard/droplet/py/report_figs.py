#!/usr/bin/env python3
"""Рисунки для отчёта (подписи по-русски). Пишет в ../figs/rep_*.png."""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from criterion import R_c as Rc_ex, R_inf as Rinf_ex
from parse_runs import load_run, master_curve
from potentials import OMEGAS, dw_f, log_f, phi_eq
from predict import R_c as Rc_lin, R_inf_over_R0 as R_inf_lin, gamma_for_eps

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def runs(pattern, key=lambda c: c["r0"]):
    out = []
    for p in sorted(glob.glob(pattern)):
        r = load_run(p)
        if r:
            out.append((key(r["cfg"]), r))
    return sorted(out, key=lambda x: x[0])


def fig_mechanism():
    rs = runs("../data/r0_dw_r0*")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    cmap = plt.get_cmap("viridis")
    n = max(1, len(rs) - 1)
    for i, (r0, r) in enumerate(rs):
        c = cmap(i / n)
        surv = r["R_eff"][-1] > 0
        ls = "-" if surv else "--"
        ax[0].plot(r["t"], r["R_eff"] / r0, ls, color=c, label=f"$R_0$={r0:.2f}")
        ax[1].plot(r["t"], r["phi_max"], ls, color=c)
        ax[2].plot(r["t"], r["phi_min"], ls, color=c)

    ax[0].set(xlabel="время", ylabel=r"$R(t)\,/\,R_0$", title="радиус капли\n(пунктир — капля исчезает)")
    ax[0].legend(fontsize=8)
    ax[0].set_xlim(0, 0.8)
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].set(xlabel="время", ylabel=r"$\max\phi$", title="наибольшее значение поля\n(оно достигается в центре капли)")
    ax[1].set_xlim(0, 0.8)
    ax[2].axhline(-1.0, color="k", lw=0.8)
    ax[2].set(xlabel="время", ylabel=r"$\min\phi$", title="уровень окружающей среды\n(поднимается — среда впитывает массу)")
    ax[2].set_xlim(0, 0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "rep_mechanism.png"), dpi=150)
    plt.close(fig)


def fig_master():
    """
    Слева: универсальная кривая линейного приближения (полиномиальный потенциал на неё ложится).
    Справа: то же в абсолютных радиусах, отдельной кривой на потенциал, плюс пороги точной теории.
    """
    GAM = {"dw": 3.200e-3, "log": 7.373e-3}
    OM = {"dw": 3.0, "log": 3.0}
    STYLE = [("dw", "../data/r0_dw_r0*", "crimson", "полиномиальный"),
             ("log", "../data/r0_log_r0*", "royalblue", "логарифмический, ω=3")]

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

    x = np.linspace(0.45, 1.45, 500)
    ax[0].plot(x, [master_curve(xi) for xi in x], "k-", lw=2, label="теория, линейный отклик")
    for kind, pat, col, lbl in STYLE:
        rc = Rc_lin(kind, GAM[kind], 1.0, OM[kind])
        first = True
        for r0, r in runs(pat):
            ax[0].plot(r0 / rc, r["R_eff"][-1] / r0, "o", ms=9, color=col, mec="k", mew=0.5,
                       label=lbl if first else None)
            first = False
    ax[0].axvline(1.0, color="gray", ls="--", lw=1.4)
    ax[0].set(xlabel=r"$R_0/R_c$   ($R_c$ из линейного отклика)", ylabel=r"$R_\infty\,/\,R_0$",
              ylim=(-0.05, 1.03),
              title="Полиномиальный потенциал ложится на универсальную кривую,\nлогарифмический смещён вправо")
    ax[0].legend(fontsize=9, loc="upper left")

    rr = np.linspace(0.13, 0.36, 220)
    for kind, pat, col, lbl in STYLE:
        g, om = GAM[kind], OM[kind]
        ax[1].plot(rr, [R_inf_lin(kind, g, float(v), 1.0, om) for v in rr], "-", lw=2, color=col,
                   label=f"линейный отклик: {lbl}")
        ax[1].axvline(Rc_ex(kind, g, 1.0, om), color=col, ls=":", lw=1.8)
        first = True
        for r0, r in runs(pat):
            ax[1].plot(r0, r["R_eff"][-1] / r0, "o", ms=9, color=col, mec="k", mew=0.5,
                       label=f"расчёт: {lbl}" if first else None)
            first = False
    ax[1].plot([], [], ":", color="gray", lw=1.8, label="порог по точной теории")
    ax[1].set(xlabel=r"начальный радиус $R_0$", ylabel=r"$R_\infty\,/\,R_0$", ylim=(-0.05, 1.03),
              title="Порог точнее предсказывает нелинейная теория,\nконечный радиус — линейная")
    ax[1].legend(fontsize=8.5, loc="upper left")

    for a_ in ax:
        a_.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "rep_master.png"), dpi=150)
    plt.close(fig)


def _bracket(pattern, keyf):
    g = {}
    for p in sorted(glob.glob(pattern)):
        r = load_run(p)
        if not r:
            continue
        c = r["cfg"]
        g.setdefault(keyf(c), []).append((c["r0"], r["R_eff"][-1]))
    out = {}
    for k, v in g.items():
        died = [a for a, R in v if R == 0]
        surv = [a for a, R in v if R > 0]
        out[k] = (max(died) if died else None, min(surv) if surv else None)
    return out


def fig_eps():
    b = _bracket("../data/gamma_*", lambda c: c["gamma"])
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    gs = np.logspace(-4, np.log10(2e-2), 50)
    eps = np.sqrt(gs / 2.0)
    ax.plot(eps, [Rc_lin("dw", g) for g in gs], "-", color="seagreen", lw=2,
            label=r"теория: $R_c\propto\varepsilon^{1/4}$")
    first = True
    for g, (lo, hi) in sorted(b.items()):
        e = np.sqrt(g / 2.0)
        if lo is not None and hi is not None:
            ax.plot([e, e], [lo, hi], color="crimson", lw=9, alpha=0.45, solid_capstyle="butt",
                    label="расчёт: между гибелью и выживанием" if first else None)
            first = False
        elif lo is not None:
            ax.plot([e], [lo], "_", color="crimson", ms=20, mew=3)
            ax.annotate("", xy=(e, lo * 1.3), xytext=(e, lo),
                        arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.2))
    ax.set(xscale="log", yscale="log",
           xlabel=r"толщина переходного слоя $\varepsilon$", ylabel=r"критический радиус $R_c$",
           title="Утоньшение переходного слоя помогает,\nно лишь как корень четвёртой степени")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "rep_eps.png"), dpi=150)
    plt.close(fig)


def fig_potentials():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    p = np.linspace(-1 + 1e-9, 1 - 1e-9, 3000)
    pw = np.linspace(-1.3, 1.3, 3000)
    ax[0].plot(pw, dw_f(pw), color="crimson", lw=2.4, label="полиномиальный")
    cmap = plt.get_cmap("viridis")
    show = [2.5, 3.0, 4.0, 6.0]
    for i, w in enumerate(show):
        pe = phi_eq(w)
        ax[0].plot(p, log_f(p, w) - log_f(np.array([pe]), w)[0], color=cmap(i / 3),
                   label=f"логарифмический, ω={w:g}")
        ax[0].plot([pe], [0], "o", color=cmap(i / 3), ms=5)
    ax[0].set(xlabel=r"$\phi$", ylabel="энергия (минимум сдвинут в 0)", ylim=(-0.05, 1.0),
              title="Как выглядят потенциалы")
    ax[0].legend(fontsize=8)

    du = np.linspace(1e-4, 0.4, 800)
    ax[1].plot(du, dw_f(1 + du) - dw_f(np.array(1.0)), color="crimson", lw=2.4, label="полиномиальный")
    for i, w in enumerate(show):
        pe = phi_eq(w)
        ddf = 2.0 / (1 - pe ** 2) - w
        cst = 2.0 / (pe ** 2 * ddf)
        f0 = log_f(np.array([pe]), w)[0]
        d = np.linspace(1e-4, 1.0 / pe - 1.0 - 1e-6, 800)
        ax[1].plot(d, cst * (log_f(pe * (1 + d), w) - f0), color=cmap(i / 3), label=f"ω={w:g}")
    ax[1].set(xscale="log", yscale="log", xlabel="относительный перегрев сверх равновесия",
              ylabel="цена перегрева (энергия)",
              title="После приведения к общему масштабу кривые совпадают:\n"
                    "логарифмический просто обрывается раньше")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "rep_potentials.png"), dpi=150)
    plt.close(fig)


def fig_omega():
    b = _bracket("../data/omega_*", lambda c: c["omega"])
    b3 = _bracket("../data/r0_log_*", lambda c: c["omega"])
    if 3.0 in b and 3.0 in b3:
        lo = max([x for x in (b[3.0][0], b3[3.0][0]) if x is not None], default=None)
        hi = min([x for x in (b[3.0][1], b3[3.0][1]) if x is not None], default=None)
        b[3.0] = (lo, hi)
    bdw = _bracket("../data/r0_dw_*", lambda c: 2.0)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ws = np.array([2.5, 3.0, 4.0, 6.0])
    gam = {w: gamma_for_eps("log", 0.04, w) for w in ws}
    ax.plot(ws, [Rc_lin("log", gam[w], 1.0, w) for w in ws], "o--", color="steelblue", lw=2,
            label="упрощённая теория (линейный отклик)")
    ax.plot(ws, [Rc_ex("log", gam[w], 1.0, w) for w in ws], "s-", color="seagreen", lw=2,
            label="точная теория (нелинейный отклик)")
    first = True
    for w, (lo, hi) in sorted(list(b.items()) + [(2.0, bdw[2.0])]):
        if lo is not None and hi is not None:
            ax.plot([w, w], [lo, hi], color="crimson", lw=9, alpha=0.45, solid_capstyle="butt",
                    label="расчёт" if first else None)
            first = False
        elif lo is not None:
            ax.plot([w], [lo], "_", color="crimson", ms=20, mew=3)
            ax.annotate("", xy=(w, lo + 0.12), xytext=(w, lo),
                        arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.2))
    g = 3.2e-3
    ax.plot([2.0], [Rc_lin("dw", g)], "o", color="steelblue", ms=8)
    ax.plot([2.0], [Rc_ex("dw", g)], "s", color="seagreen", ms=8)
    ax.annotate("полиномиальный\n(предел ω→2)", xy=(2.05, 0.185), fontsize=9, color="dimgray")
    ax.set(xlabel=r"параметр $\omega$ логарифмического потенциала",
           ylabel=r"критический радиус $R_c$", ylim=(0.16, 0.64),
           title="Чем «жёстче» потенциал, тем ХУЖЕ: критический радиус растёт\n"
                 "(все точки при одинаковой толщине переходного слоя ε=0.04)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "rep_omega.png"), dpi=150)
    plt.close(fig)


def fig_mobility():
    def pick(pat):
        c = sorted(glob.glob(pat))
        return load_run(c[0]) if c else None

    h = pick("../data/faceavg_harmonic_f1e-4_*")
    m = pick("../data/faceavg_midpoint_f1e-4_*")
    a = pick("../data/faceavg_arithmetic_f1e-4_*")
    const = pick("../data/faceavg_constant_ref_*") or pick("../data/r0_dw_r00.26_*")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for r, col, lbl in [(m, "crimson", "среднее по φ (как сейчас)"),
                        (a, "darkorange", "среднее арифметическое"),
                        (h, "seagreen", "среднее гармоническое")]:
        if r is None:
            continue
        ax[0].plot(r["t"], r["R_eff"], color=col, lw=2, label=lbl)
        ax[1].plot(r["t"], r["phi_min"], color=col, lw=2, label=lbl)
    if const is not None:
        ax[0].plot(const["t"], const["R_eff"], color="gray", lw=2, ls=":",
                   label="постоянная подвижность")
    ax[0].set(xlabel="время", ylabel=r"радиус капли $R(t)$", xlim=(0, 0.5),
              title="Три способа усреднения дают одну кривую;\nнелинейная диффузия продлевает жизнь в 2.7 раза")
    ax[0].legend(fontsize=8)
    ax[1].axhline(-1.0, color="k", lw=0.8)
    ax[1].set(xlabel="время", ylabel=r"$\min\phi$ (уровень среды)", xlim=(0, 0.5),
              title="Среда держится у φ=-1 лишь до t≈0.2, затем впитывает массу\nи подвижность в ней перестаёт быть малой")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "rep_mobility.png"), dpi=150)
    plt.close(fig)


def fig_numerics():
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for p in sorted(glob.glob("../data/resolution_dt*")):
        r = load_run(p)
        ax[0].plot(r["t"], r["R_eff"], lw=1.8, label=f"$dt$ = {r['cfg']['dt']:.0e}")
    ax[0].set(xlabel="время", ylabel=r"$R(t)$", title="Шаг по времени: кривые неразличимы")
    ax[0].legend(fontsize=8)
    for p in sorted(glob.glob("../data/resolution_grid*")):
        r = load_run(p)
        ax[1].plot(r["t"], r["R_eff"], lw=1.8, label=f"сетка ${int(r['cfg']['grid'])}^3$")
    ax[1].set(xlabel="время", ylabel=r"$R(t)$", title="Сетка: расхождение 0.24 %")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "rep_numerics.png"), dpi=150)
    plt.close(fig)


def fig_theory_compare():
    """Логарифмический потенциал при разных omega: кривые обеих версий критерия и точки расчёта."""
    from criterion import R_inf as Rex_abs, R_c as Rc_ex

    PANELS = [
        (2.5, 2.461e-3, (0.22, 0.38)),
        (3.0, 7.373e-3, (0.22, 0.38)),
        (4.0, 3.207e-2, (0.22, 0.42)),
        (6.0, 3.050e-1, (0.18, 0.62)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    for ax, (om, g, xlim) in zip(axes.ravel(), PANELS):
        rr = np.linspace(xlim[0], xlim[1], 140)
        ax.plot(rr, [R_inf_lin("log", g, float(v), 1.0, om) for v in rr], "-", lw=2.2,
                color="steelblue", label="теория: линейный отклик")
        ax.plot(rr, [Rex_abs("log", g, float(v), 1.0, om) / v for v in rr], "-", lw=2.2,
                color="seagreen", label="теория: нелинейный отклик")

        pts = {}
        for pat in (f"../data/omega_w{om:g}_r0*", "../data/r0_log_r0*"):
            for pth in sorted(glob.glob(pat)):
                r = load_run(pth)
                if r and abs(r["cfg"].get("omega", 0) - om) < 1e-9:
                    pts[r["cfg"]["r0"]] = r["R_eff"][-1] / r["R_eff"][0]
        ax.plot(sorted(pts), [pts[k] for k in sorted(pts)], "o", ms=11, color="crimson",
                mec="k", mew=0.8, zorder=5, label="расчёт")

        lin, ex = Rc_lin("log", g, 1.0, om), Rc_ex("log", g, 1.0, om)
        ax.axvline(lin, color="steelblue", ls=":", lw=1.8)
        ax.axvline(ex, color="seagreen", ls=":", lw=1.8)
        ax.set(xlabel=r"начальный радиус $R_0$", ylabel=r"$R_\infty\,/\,R_0$",
               ylim=(-0.05, 1.05), xlim=xlim,
               title=f"$\\omega$ = {om:g}   (порог $R_c$: линейн. {lin:.3f},  нелинейн. {ex:.3f})")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle("Логарифмический потенциал: нелинейная версия критерия описывает расчёт,\n"
                 "линейная систематически занижает порог — и тем сильнее, чем больше ω",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIGS, "rep_theory_compare.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_mechanism()
    fig_master()
    fig_eps()
    fig_potentials()
    fig_omega()
    fig_mobility()
    fig_numerics()
    fig_theory_compare()
    print("готово:", sorted(os.path.basename(f) for f in glob.glob(os.path.join(FIGS, "rep_*.png"))))
