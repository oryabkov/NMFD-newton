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

$$f_\eta(\phi)=\tfrac12\Bigl(\sqrt{(1-\phi^{2})^{2}+\eta^{4}}-\eta^{2}\Bigr),\qquad
f_\eta'(\phi)=\frac{-\phi\,(1-\phi^{2})}{\sqrt{(1-\phi^{2})^{2}+\eta^{4}}}$$

$C^\infty$ everywhere, minima exactly at $\pm1$, $f_\eta(\pm1)=0$, and

$$f_\eta''(\pm1)=\frac{2}{\eta^{2}} .$$

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

## 5. What to do here

1. Implement (c) as a third `phobic_energy` alongside `double_well_potential` and
   `logarithmic_potential`, with $\eta$ from the command line. `get_phi_eq()` returns 1 exactly and
   `get_curvature()` returns $2/\eta^2$, so the degenerate-mobility fix and the smoother diagonal
   both pick it up unchanged.
2. Sweep $\eta\in\{1,0.5,0.3,0.2,0.15,0.1,0.05\}$ at $R_0$ around the double well's measured
   bracket $(0.28,0.30)$, matched $W$ rather than matched $\gamma$. Predicted: the bracket walks
   down to $\approx0.09$. This is the falsifiable statement of this note.
3. Watch the solver, not the physics. $f''$ in the bulk grows as $2/\eta^2$ while $\gamma/h^2$ is
   fixed, so the Newton/multigrid conditioning degrades like $\eta^{-2}$. The measured failure mode
   from the mobility study — Newton hitting its iteration cap with GMRES stalling — is the one to
   expect, and it is what will set the usable $\eta$, not the criterion.
4. Only if $\eta$ bottoms out on solver grounds is (a) worth the variational-inequality machinery;
   Baňas & Nürnberg is then the recipe.
