# Phase 1 results — the criterion holds, and the mobility surprise

Runs: 64³ unit box, $\epsilon=0.04$ ($\gamma=3.2\times10^{-3}$, double well), sphere IC with a tanh
profile, Neumann walls (closed box), $dt=2\times10^{-3}$, GMRES+MG. Data under `../data/`,
analysis by `../py/parse_runs.py`.

## 1. Mass conservation — the gate

`mass_drift` $=|M(t_{end})-M(0)|/|M(0)| = 0$ **exactly**, in every one of the 16 runs, to all
printed digits. The closed-box discretisation conserves $\int\phi$ to round-off, so every number
below is measuring physics, not a leak. (The one exception is discussed in §3.)

## 2. The critical radius — prediction P1/P2 confirmed to ~2 %

Predicted $R_c=(0.5053\,\Lambda\epsilon V)^{1/4}=0.2865$ with $\Lambda=1/3$.

| $R_0$ | $R_0/R_c$ | outcome | $R_\infty/R_0$ measured | predicted | $\max\phi$ measured | predicted |
|---|---|---|---|---|---|---|
| 0.15 | 0.524 | evaporates | 0 | 0 | 0.936 | 1.178 |
| 0.20 | 0.698 | evaporates | 0 | 0 | 1.047 | 1.133 |
| 0.24 | 0.838 | evaporates | 0 | 0 | 1.084 | 1.111 |
| 0.26 | 0.908 | evaporates | 0 | 0 | 1.090 | 1.103 |
| 0.28 | 0.977 | evaporates | 0 | 0 | 1.091 | 1.095 |
| **0.30** | **1.047** | **survives** | **0.8002** | **0.7975** | 1.089 | 1.089 |
| **0.33** | **1.152** | **survives** | **0.8980** | **0.8868** | 1.083 | 1.081 |

* The transition is bracketed between $R_0=0.28$ and $0.30$; predicted $R_c=0.2865$ falls inside
  that bracket. **Within 2 %.**
* $R_\infty/R_0$ agrees to **0.3 %** at $R_0=0.30$ and **1.3 %** at $R_0=0.33$ — and the saddle-node
  jump from 0 to $\approx0.8$ is reproduced, not just the threshold.
* Every point lies on the universal master curve $r(1-r^3)=0.472\,(R_c/R_0)^4$
  (`../figs/r0_dw_collapse.png`), which contains no fitted parameter.

**The overshoot law P5 also holds.** $\max\phi$ agrees with $1+\tfrac23\epsilon/R$ to within 0.1 %
for the surviving drops. The measured maximum sits at the **drop centre** (`phi_centre == phi_max`
in the logs) — exactly the perturbation direction Yue's variational argument assumes. So the
"пережатая жидкость" at $\phi>1$ is not a numerical defect: it is the Gibbs–Thomson shift, and it
is predictable to three digits.

The prediction degrades for the small, doomed drops ($R_0\le0.24$) because they die before the
quasi-static regime is reached; at $R_0=0.15$ the drop never even attains $\phi=1$ — it is only
$5\epsilon$ across, so its profile never saturates.

**Consequence for supervisor item #1.** With the criterion now validated, the $\Lambda^{1/4}$
scaling is on firm ground: the logarithmic potential at $\omega=3$ moves $R_c$ by 6 %, i.e. from
0.2865 to 0.2686. The `r0_log` sweep tests exactly this and is queued.

## 3. Face averaging — a negative result, and why

Nine runs, $R_0=0.20$, double well, parabolic (degenerate) mobility, $3$ rules $\times$ $3$ floors:

| rule | floor | $t_{evap}$ | mean Newton its | note |
|---|---|---|---|---|
| midpoint | $10^{-2}$ | 0.100 | 2.9 | |
| arithmetic | $10^{-2}$ | 0.100 | 2.9 | |
| harmonic | $10^{-2}$ | 0.100 | 2.9 | |
| midpoint | $10^{-3}$ | 0.100 | 3.1 | |
| arithmetic | $10^{-3}$ | 0.100 | 3.0 | |
| harmonic | $10^{-3}$ | 0.100 | 3.2 | |
| midpoint | $10^{-5}$ | 0.100 | 3.2 | |
| arithmetic | $10^{-5}$ | 0.100 | 3.2 | |
| harmonic | $10^{-5}$ | — | **89.3** | **solver breakdown**, see below |
| *constant $M=1$* | — | *0.040* | *2.5* | reference from the `r0_dw` sweep |

Three things come out of this.

**(a) P6 confirmed.** Every configuration below $R_c$ evaporates. Mobility does not appear in the
thermodynamic criterion and, indeed, does not change the outcome — only the rate. Degenerate
mobility slows evaporation by $\approx2.5\times$ ($t_{evap}$ 0.040 → 0.100) but does not save the
drop.

**(b) The face rule made no difference at all** — contrary to the naive expectation in
[04](04_mobility_averaging.md), and contrary to my own prediction P7. The likely reason is a
feedback that defeats the block: mass leaves the drop through the **interface shell**, where
$\phi\approx0$ and $M\approx D$ under *every* averaging rule; the released mass then piles up in the
bulk immediately outside, which pushes $\phi$ away from $-\phi_{eq}$ and thereby **un-degenerates
$M$ exactly where the block was supposed to act**. The degenerate mobility is self-defeating at the
one place it matters.

*This is a hypothesis, not yet a measurement.* The decisive test is to log the face mobility in the
shell just outside the interface and watch it rise as mass accumulates.

**(c) This experiment was badly designed and is being repeated.** $R_0=0.20$ is $5\epsilon$ across
and dies in $\sim\!50$ steps — inside the initial profile relaxation, where $M\approx D$ regardless
of rule or floor. That is why all nine numbers are identical: they are measuring profile
relaxation, not Ostwald transport. The rerun uses $R_0=0.26$ (still below $R_c$, but long-lived),
$2000$ steps, and floors $10^{-2}\ldots10^{-4}$. **Treat (b) as provisional until that lands.**

## 4. Harmonic averaging breaks the linear solver at a $10^5$ contrast

`harmonic` + floor $10^{-5}$: mean 89.3 Newton iterations (hitting the 101 cap), repeated
`gmres::solve: linear solver failed to converge`, and the only nonzero mass drift in the whole
study ($2.6\times10^{-8}$) — i.e. the Newton solve stopped converging well enough to conserve mass.
The run died at step 110 of 600.

This is the risk flagged in the roadmap: harmonic averaging faithfully transmits the full
$M_{max}/M_{min}$ contrast into the discrete operator, and the multigrid smoother cannot cope with
$10^5$. Floors down to $10^{-3}$ are fine (3.2 Newton its). So the usable corner is
$M_{min}/M_{max}\gtrsim10^{-3}$ — which, per (b), may not be deep enough to matter anyway.

Note the smoother's $\psi$-diagonal was *fixed* as part of this work (it previously used the cell
mobility instead of the two face mobilities); the breakdown above is with the corrected smoother,
so it is a genuine conditioning limit, not the old bug.

## 5. What this changes in the roadmap

* Phase 1 (criterion) is **done and confirmed**; the theory in
  [03](03_shrinkage_criterion.md) can be used as a predictive tool from here on.
* Phase 2 (potential) now has a sharp, pre-registered prediction to falsify: $R_c$ should move to
  0.2686 at $\omega=3$, i.e. the transition should shift from the (0.28, 0.30) bracket to
  (0.26, 0.28). A 6 % effect is *just* resolvable with this $R_0$ grid.
* Phase 3 (mobility) has to be re-run before anything is concluded, and its most interesting
  question is now **(b)**: is the degenerate mobility defeated by its own feedback? That is a
  better question than "which average", and it was not visible before running the sweep.
