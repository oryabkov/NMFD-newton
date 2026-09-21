# Designing a potential that does not let the drop evaporate

Follow-up to [06](06_nonlinear_correction.md) and [07](07_practical_guide.md). There the conclusion
was negative: raising $\omega$ on the logarithmic potential makes $R_c$ *worse*, because the
stiffness it adds sits at $\pm\phi_{eq}$ while the ambient relaxes along the flat flank between
$-\phi_{eq}$ and $0$. This note turns that into a design target and enumerates the potentials that
meet it.

Code: `py/potential_zoo.py` (scalars + exact criterion), `py/plot_potential_zoo.py` (the figure
`figs/potential_zoo.png`). Panel B of `potential_zoo.py` reproduces `criterion.py` to four digits
at matched $\epsilon$, so the generalised implementation is checked against the one already
validated against measurements.

## 1. The design target

From [03](03_shrinkage_criterion.md), quasi-static balance of a drop of radius $R$ in a closed
domain of volume $V$, dimension $d$:

$$\psi=\frac{(d-1)\sigma}{2\phi_{eq}R},\qquad
2\phi_{eq}\bigl(V_d(R_0)-V_d(R)\bigr)=\delta_{in}V_d+\delta_{out}(V-V_d),$$

with $f'(\phi_{eq}+\delta_{in})=f'(-\phi_{eq}+\delta_{out})=\psi$. Since $V_d\ll V$, only
$\delta_{out}$ matters. Linearising it, $\delta_{out}=\psi/f''(-\phi_{eq})$, gives in 3D

$$R_c^4=0.2527\,\frac{\sigma V}{\phi_{eq}^{2}f''(\phi_{eq})}.$$

**Only $\sigma/(\phi_{eq}^2 f'')$ is under our control.** Now fix what a practitioner actually
fixes — the *resolved* interface width $W$ (the distance over which $\phi$ runs from $-0.9\phi_{eq}$
to $+0.9\phi_{eq}$), not $\gamma$ and not $\epsilon$. Write

$$\hat\sigma=\int_{-\phi_{eq}}^{\phi_{eq}}\!\!\sqrt{2\bigl(f-f_{min}\bigr)}\,d\phi,\qquad
\hat w=\int_0^{0.9\phi_{eq}}\!\!\frac{d\phi}{\sqrt{2(f-f_{min})}},$$

so that $\sigma=\hat\sigma\sqrt\gamma$ and $W=2\hat w\sqrt\gamma$. Eliminating $\gamma$,

$$\boxed{\;R_c^4=0.2527\,V\,W\,K[f],\qquad
K[f]=\frac{\hat\sigma}{\phi_{eq}^{2}\,f''(\phi_{eq})\,\hat w}\;}$$

$K$ is a pure shape number: invariant under $f\to Af$ and $\phi\to a\phi$, and free of $\gamma$.
$K=0.2264$ for the double well. **Minimising $K$ at fixed $W$ is the whole design problem.**

Two consequences read straight off:

* $\hat\sigma$ and $\hat w$ are integrals over the *whole* barrier and vary by at most 20 % across
  any sane potential. $f''(\phi_{eq})$ has no upper bound at all. So the only real lever is
  **the curvature of the well at the bulk value**, and the target is to make it as large as
  possible *without* widening the interface.
* Raising $f''$ everywhere (what $\omega$ does) also raises $\hat\sigma$ and lowers $\hat w$, and
  it changes $\phi_{eq}$. Those partly cancel. The win only comes from curvature concentrated in a
  neighbourhood of $\pm\phi_{eq}$ that is thin compared with $W$.

A potential with $f''(\pm\phi_{eq})=\infty$ has $K=0$. Then the linear criterion is vacuous and the
threshold is set instead by the **spinodal ceiling**: the ambient can hold no supersaturation past
$\psi_{max}=\max_{(-\phi_{eq},0)}f'$, so

$$R\;\ge\;R_{floor}=\frac{(d-1)\sigma}{2\,\psi_{max}}.$$

That floor is $\approx 0.7\,W$ for every candidate below — i.e. *the drop merely has to be wider
than the interface*, which is already required for resolution. The thermodynamic constraint stops
binding.

## 2. Candidate functional forms

All normalised to $\phi_{eq}=1$, $f(\pm1)=0$, $f(0)=1/4$ — a rescaling that leaves $R_c$ unchanged
(§1) and makes the shapes comparable on one axes.

### (a) Double obstacle — the exact answer

$$f(\phi)=\frac{\theta}{2}\bigl(1-\phi^{2}\bigr)+I_{[-1,1]}(\phi),\qquad
I_{[-1,1]}=\begin{cases}0,&|\phi|\le1\\ +\infty,&|\phi|>1\end{cases}$$

Concave inside; the minima are the *constraint*, not a stationary point. The equilibrium condition
becomes the inclusion $\psi\in\partial f(\phi)$, and at $\phi=-1$ the subdifferential is the whole
half-line $(-\infty,\theta]$. Therefore

$$\delta_{out}\equiv 0\quad\text{for every }\psi\le\theta .$$

The ambient absorbs **exactly zero** mass. There is nowhere for the drop's mass to go, so it does
not shrink at all until $\psi=\sigma/R$ exceeds $\theta$, i.e. until $R<R_{floor}=0.70\,W$.

The equilibrium profile is $\phi=\sin\!\bigl(x/\sqrt{\gamma\theta}\bigr)$ on a band of width
$\pi\sqrt{\gamma\theta}$ and exactly $\pm1$ outside — **compact support, no exponential tails**.
The tails are precisely the mechanism of the leak, so removing them removes the leak.

Price: a variational inequality, not a smooth nonlinear system. Newton on $f'$ does not apply.

### (b) Cusped power family — a smooth homotopy towards it

$$f_p(\phi)=\frac{1}{2p}\bigl|1-\phi^{2}\bigr|^{p},\qquad 1<p\le 2$$

$$f_p'(\phi)=-\phi\,|1-\phi^2|^{p-1}\operatorname{sign}(1-\phi^2),\qquad
f_p''=-s^{p-1}+2(p-1)\phi^{2}s^{p-2},\;\;s=1-\phi^2 .$$

$p=2$ is the double well exactly. For $p<2$: $f''(\pm1)=\infty$, the profile has compact support,
and the ambient response is sublinear — near $\phi=-1$, $\psi=(2\delta_{out})^{p-1}$, i.e.
$\delta_{out}=\tfrac12\psi^{1/(p-1)}$, which for $p=1.2$ is $\psi^5/2$. $p\to1$ is the tent
$\tfrac12|1-\phi^2|$, the smooth part of (a).

No variational inequality, but $f'$ has unbounded derivative at $\pm1$ and $\phi$ can still leave
$[-1,1]$ (where $f$ grows only like $|\phi|^{2p}$, weaker than the double well for $p<2$).

### (c) Smoothed obstacle — the one to implement first

$$f_\eta(\phi)=a_\eta\Bigl(\sqrt{(1-\phi^{2})^{2}+\eta^{4}}-\eta^{2}\Bigr),\qquad
a_\eta=\frac{1}{4\bigl(\sqrt{1+\eta^{4}}-\eta^{2}\bigr)}$$

$$f_\eta'(\phi)=\frac{-2a_\eta\,\phi\,(1-\phi^{2})}{D},\qquad
f_\eta''(\phi)=a_\eta\Bigl(\frac{-2(1-\phi^2)}{D}+\frac{4\phi^{2}\eta^{4}}{D^{3}}\Bigr),\quad
D=\sqrt{(1-\phi^{2})^{2}+\eta^{4}}$$

$C^\infty$ everywhere, minima exactly at $\pm1$, $f_\eta(\pm1)=0$, and

$$f_\eta''(\pm1)=\frac{4a_\eta}{\eta^{2}}\;\xrightarrow[\eta\to0]{}\;\frac{1}{\eta^{2}} .$$

$a_\eta$ pins the barrier at $f_\eta(0)=1/4$ for every $\eta$, so $\gamma$ buys the same interface
width as it does for the double well and the two are directly comparable at equal `--gamma`. It
also makes the initial profile scale $\sqrt{2\gamma}$ for every $\eta$, the double well's value.

$\eta\to0$ gives the tent (hence the obstacle); $\eta=1$ is a soft flat-bottomed well. Same
concave-centre / convex-wings structure as the double well, so the existing convex splitting and
the existing Newton + multigrid apply unchanged — only the constant $2/\eta^2$ in the diagonal
changes. **$\eta$ is a single knob that interpolates continuously between what we have now and the
obstacle limit**, which makes it the right thing to sweep.

Bonus: the overshoot shrinks with the same knob, $\max\phi-1\simeq\psi\eta^{2}/2$, so the
$\phi>1$ complaint disappears without a singular potential.

### (d) One-sided stiffening

§1 says only $f''(-\phi_{eq})$ enters, because $V_d\ll V$. If the drop phase is always the minority
phase, it is enough to stiffen the *ambient* well and leave the drop well alone — e.g. apply (c)
only for $\phi<0$ and keep the quartic for $\phi>0$. Cheaper, and it keeps a familiar potential
where $\phi$ overshoots. Untested; listed for completeness, and it cannot beat (a)–(c), only
approach them at lower cost.

## 3. Numbers

Exact nonlinear criterion, 3D, unit box, fixed resolved width $W=0.1$:

| potential | $\hat\sigma$ | $\hat w$ | $f''(1)$ | $K$ | $\psi_{max}$ | $R_c$ | rel. | min. drop vol. fraction |
|---|---|---|---|---|---|---|---|---|
| double well $p=2$ | 0.9428 | 2.082 | 2 | 0.2264 | 0.385 | 0.2388 | 1.00 | 5.70 % |
| logarithmic $\omega=3$ | 0.9893 | 1.897 | 3.69 | 0.1415 | 0.387 | 0.2198 | 0.92 | 4.45 % |
| cusped $p=1.5$ | 1.0167 | 1.802 | $\infty$ | 0 | 0.375 | 0.1953 | 0.82 | 3.12 % |
| cusped $p=1.2$ | 1.0701 | 1.665 | $\infty$ | 0 | 0.395 | 0.1416 | 0.59 | 1.19 % |
| smoothed obstacle $\eta=0.15$ | 1.1003 | 1.599 | 45.5 | 0.0151 | 0.473 | 0.1309 | 0.55 | 0.94 % |
| **double obstacle** | 1.1107 | 1.584 | $\infty$ | 0 | 0.500 | **0.0701** | **0.29** | **0.144 %** |

$\eta$-sweep for (c), same $W$:

| $\eta$ | 1.0 | 0.5 | 0.3 | 0.2 | 0.15 | 0.1 | 0.05 | 0.02 |
|---|---|---|---|---|---|---|---|---|
| $f''(1)$ | 2.4 | 5.1 | 12 | 26 | 45 | 101 | 401 | 2501 |
| $R_c$ | 0.232 | 0.203 | 0.171 | 0.146 | 0.131 | 0.113 | 0.092 | 0.078 |
| $R_c/R_c^{dw}$ | 0.97 | 0.85 | 0.72 | 0.61 | 0.55 | 0.47 | 0.39 | 0.33 |

$p$-sweep for (b): $R_c/R_c^{dw}$ = 1.00, 0.94, 0.87, 0.76, 0.59, 0.47, 0.39 at
$p$ = 2.0, 1.8, 1.6, 1.4, 1.2, 1.1, 1.05.

Both families saturate at $R_{floor}\approx0.70\,W$ — a factor $3.4$ in radius, $\mathbf{40}$ in
volume, against the double well. Nothing beats that floor: it is the point where the sharp-interface
picture stops being meaningful anyway.

### Why "at fixed $W$" and not "at fixed $\epsilon$"

Panel B of `potential_zoo.py` shows what happens if $\epsilon=\sqrt{\gamma/f''(\phi_{eq})}$ is
matched instead: the cusped potentials have $\epsilon$ three to four orders of magnitude below
their own interface width, so matching $\epsilon$ would force an absurdly wide interface and every
one of them "evaporates". $\epsilon$ is the tail decay length; once the tails are gone it is no
longer the interface width and no longer the right thing to hold fixed. The comparison at matched
$\epsilon$ is still the right one *within* the double-well/logarithmic pair, which is how
[05](05_results_phase1.md)–[07](07_practical_guide.md) used it — there, log $\omega=3$ is 3.4 %
worse than the double well ($R_c$ 0.3170 vs 0.3066), whereas at matched $W$ it is 8 % better. Both
are true; they answer different questions.

## 4. Who has done this

* **Yue, Zhou & Feng**, *Spontaneous shrinkage of drops and mass conservation in phase-field
  simulations*, J. Comput. Phys. **223** (2007) 1–9. The diagnosis. No potential redesign — the
  recommendation is a smaller $\epsilon$ and a mobility tuned so that $t_{evap}$ exceeds the
  simulated time.
* **Donaldson, Kirpalani & Macchi**, *Diffuse interface tracking of immiscible fluids: improving
  phase continuity through free energy density selection*, Int. J. Multiphase Flow **37** (2011)
  777. **The closest published work to §2:** a modified double-obstacle chosen specifically to
  remove spontaneous drop shrinkage and the accompanying limits on mobility, benchmarked against
  the double well and against VOF, at a reported cost only slightly above the double well.
* **Oono & Puri** (deep-quench cell-dynamics, Phys. Rev. A **38** (1988) 434) and
  **Blowey & Elliott**, Eur. J. Appl. Math. **2** (1991) 233–280 (Part I, analysis) and **3**
  (1992) 147–179 (Part II, numerics). Origin and rigorous theory of the double obstacle, including
  the compact-support property.
* **Elliott & Garcke**, SIAM J. Math. Anal. **27** (1996) 404–423 — degenerate mobility, the
  companion knob (the one that buys time rather than stability).
* Solvers for the obstacle case: **Baňas & Nürnberg**, *A multigrid method for the Cahn–Hilliard
  equation with obstacle potential*, Appl. Math. Comput. **213** (2009) 290–303 (mesh-independent,
  arbitrary $dt$, robust down to small $\gamma$); **Hintermüller, Hinze & Tber**, *Solving the
  Cahn–Hilliard variational inequality with a semi-smooth Newton method*, ESAIM COCV (2011).
* Different routes to the same symptom, for contrast:
  **Kwakkel, Fernandino & Dorao**, *A redefined energy functional to prevent mass loss in
  phase-field methods*, AIP Advances **10** (2020) 065124 — rebuild the functional out of
  $\nabla_n C$ so the curvature-driven source term is absent;
  **Gurin**, arXiv:2606.24295 (2026) — keep the double well but shift it by a curvature-dependent
  correction, $r=2/\kappa$ from $\kappa=\nabla\!\cdot(\nabla\phi/|\nabla\phi|)$.
  Both are corrections applied on top of a leaking model; §2 removes the leak instead.

## 4a. First measurements (`etaprobe`, job 96522)

Implemented as `smoothed_obstacle_potential` with the prefactor
$a_\eta=\bigl[4(\sqrt{1+\eta^4}-\eta^2)\bigr]^{-1}$ pinning the barrier at $1/4$ for every $\eta$, so
`--gamma` means the same thing as for the double well. 64³, $\gamma=3.2\times10^{-3}$,
$R_0=0.30$, Neumann, 100 steps.

| $\eta$ | 1 | 0.5 | 0.3 | 0.2 | 0.15 | 0.1 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|---|---|---|---|---|
| $f''(1)$ measured | 2.414 | 5.123 | 12.16 | 26.02 | 45.46 | 101.0 | 401.0 | 2501 | 10001 |
| $\max\phi-1$ | 5.7e-2 | 3.7e-2 | 1.7e-2 | 8.1e-3 | 4.7e-3 | 2.1e-3 | 5.4e-4 | 8.7e-5 | 2.2e-5 |
| Newton its | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| final residual | 4.9e-2 | 1.4e-4 | 2.1e-9 | 8.8e-11 | 8.7e-11 | 8.9e-11 | 9.1e-11 | 8.1e-11 | 9.2e-11 |

* $f''(1)$ reproduces $4a_\eta/\eta^2$ exactly, and $\ell=\sqrt{2\gamma}=0.08$ for every $\eta$, as
  designed.
* **The overshoot follows $\max\phi-1=\psi/f''(1)$ to within 6 %** across four decades. At
  $\eta=0.01$ it is $2.2\times10^{-5}$, against $8.9\times10^{-2}$ for the double well at the same
  radius — a factor of 4000. The $\phi>1$ complaint that started this study is settled by the same
  knob that is supposed to settle evaporation.
* **§5.3 below was wrong.** The predicted $\eta^{-2}$ conditioning failure did not appear: Newton
  takes two iterations and reaches $10^{-10}$ at $\eta=0.01$, i.e. $f''=10^4$. Nothing in the range
  we care about is solver-limited, so $\eta$ is free to be chosen on physical grounds. (The runs at
  $\eta\le0.2$ stop before step 100 on `time_tol` — the drop stops changing, which is the outcome
  being tested.)
* $R_{\text{eff}}$ at $R_0=0.30$ settles at 0.303 for $\eta\le0.05$ and does not decrease. Whether
  that holds down at the threshold is what `sweep_eta` (job 96523) measures.

## 5. What to do here

1. ~~Implement (c)~~ **done** — `smoothed_obstacle_potential` in `kernels/phobic_energy.h`,
   `--potential smoothed_obstacle --eta <x>`. `get_phi_eq()` returns 1 exactly and
   `get_curvature()` returns $4a_\eta/\eta^2$, so the degenerate-mobility fix and the smoother
   diagonal pick it up unchanged. Also added `get_profile_scale( gamma )` per potential, because
   the old initial condition used $2\epsilon$ and $\epsilon$ is meaningless here; the double well
   and the logarithmic potential keep their previous value exactly.
2. `sweep_eta` (job 96523, running): $\eta\in\{1,0.5,0.3,0.2,0.15,0.1,0.05\}$, four radii each,
   at the *same* $\gamma=3.2\times10^{-3}$, grid and $dt$ as the validated `r0_dw` run, so the
   double well's measured bracket $(0.28,0.30)$ is the control. Exact criterion predicts
   $R_c$ = 0.295, 0.253, 0.215, 0.189, 0.174, 0.158, 0.140. This is the falsifiable statement.
3. ~~Conditioning will degrade like $\eta^{-2}$ and set the usable $\eta$.~~ **Measured false**
   (§4a): two Newton iterations to $10^{-10}$ at $\eta=0.01$, i.e. $f''=10^4$. $\eta$ is not
   solver-limited in the useful range.
4. Since (3) fell, the remaining question is whether the *criterion* saturates where theory says it
   does. $R_{floor}=(d-1)\sigma/(2\psi_{max})=0.126$ at this $\gamma$, and the predicted $R_c$ is
   already 0.140 at $\eta=0.05$ — so the family has essentially converged and $\eta\lesssim0.05$
   buys nothing further. A three-point run at $\eta=0.02$ around $R_0\in\{0.11,0.13,0.15\}$ would
   confirm the floor. **(a)** is then only worth its variational-inequality machinery if that floor
   turns out to be wrong; Baňas & Nürnberg is the recipe.