# The spherical-droplet asymptotics against our runs

*Reading of [`cahn_hilliard_spherical_droplet_analysis.md`](cahn_hilliard_spherical_droplet_analysis.md):
how it relates to [`03_shrinkage_criterion.md`](03_shrinkage_criterion.md), what the runs already on
disk say about it, and what is left to measure.*
*Numbers reproduced by [`py/verify_sharp_interface.py`](../py/verify_sharp_interface.py);
figure [`figs/sharp_interface.png`](../figs/sharp_interface.png).*

## 1. The two derivations are the same equation

Note 03 starts from a Gibbs–Thomson mass balance in a closed box and asks whether the mass released
by shrinking can supply the supersaturation the shrunken radius demands. The new note starts from
the constrained energy $E(R)$ and asks for its stationary points. They coincide identically.

| | note 03 | new note (unit ball) |
|---|---|---|
| stationarity | $r(1-r^{3})=\dfrac{\Lambda\epsilon V}{c_3R_0^{4}}$, $r=R/R_0$ | $h(R)=2R^{4}-mR+\dfrac{\sigma}{2}=0$ |
| existence threshold | $R_0^{4}\ge0.5053\,\Lambda\epsilon V$ | $m_{\min}=\left(\tfrac{4\sigma}{3}\right)^{3/4}$ |
| smallest stable drop | $R_\infty=4^{-1/3}R_0$ at threshold | $R_f=(\sigma/12)^{1/4}$ |
| chemical potential | $\mu=\dfrac{(d-1)\sigma}{2\phi_{eq}R}$ | $\psi_0=\dfrac{\sigma}{R}$ |
| overshoot | $\max\phi-1=\tfrac23\epsilon/R$ | $\max\phi-1=\sigma/2R$ |

The bridge is the initial condition: a sharp drop of radius $R_0$ in a box of volume $V$ carries
$m=1+\bar\phi=2v(R_0)/V$. Substituting that into $h(R)=0$ and writing $r=R/R_0$ gives note 03's
equation term for term; putting $\Lambda=1/3$, $\epsilon=\sqrt{\gamma/2}$, $V=4\pi/3$ into the
threshold gives $R_0^4\ge0.52915\,\sigma$, and $(m_{\min}/2)^{4/3}=0.52915\,\sigma$ as well. Same
for $R_f$: both give $0.5373\,\sigma^{1/4}$. The last three rows are term-by-term identities
($\sigma=\tfrac43\epsilon$ for the double well).

**Notation clash to watch.** Note 03 calls $R_c$ the *initial* radius at which survival switches on;
the new note calls $R_c$ the *unstable stationary root* (the critical nucleus). They are different
objects — the first is $R_{0,\mathrm{crit}}$ below, the second is the smaller root of $h$.

## 2. What the new note adds

1. **The landscape, not just the threshold.** $E(R)=3\sigma R^{2}+(m-2R^{3})^{2}$ has a minimum at
   $R=0$, a maximum at $R_c$ and a minimum at $R_\infty$. Note 03 only asked whether roots exist.
2. **Zero Tolman length** ($\psi_2=0$ by the $\phi\to-\phi$ symmetry of $f$), so Gibbs–Thomson is
   accurate to $O((\xi/R)^{2})$, not $O(\xi/R)$.
3. **Second-order bulk shifts** $\delta_\pm=\frac{\psi}{2}\mp\frac38\psi^{2}$ — the droplet and the
   matrix stop shifting by the same amount. Note 03 had only the symmetric first order.
4. **The explicit profile correction** $\Phi_1=\frac{\psi_1}{2}\Phi_0^{2}$, i.e.
   $\phi\approx-\tanh\zeta+\frac{\sigma}{2R}\tanh^{2}\zeta$.
5. **The $O(\gamma)$ mass correction** $\bar\phi=2R^3-1+\delta+\pi^{2}R\gamma$ — the finite interface
   carries net mass because the sphere weights its outer side more.
6. **Global stability** $m_{\mathrm{eq}}=\sqrt2\,\sigma^{3/4}$: between $m_{\min}$ and
   $m_{\mathrm{eq}}$ the droplet exists but costs more than the homogeneous state.
7. The stability condition $R^{4}>\sigma/12$ and the barrier $\Delta F^{*}=\pi\sigma^{3}/3m^{2}$.

## 3. What note 03 has that the new note does not

Arbitrary bulk potentials through the single shape number
$\Lambda=\sigma/(2\phi_{eq}^{2}f''\epsilon)$ (the new note treats the double well plus a remark on
the logarithmic one); the 2D case; the **rate** $t_{evap}$ and the $R^{3}$ law; the fact that
mobility cannot move $R_{0,\mathrm{crit}}$ at all, only the lifetime; the explicit $V$ dependence;
and the spinodal ceiling that caps the whole family when $f''(\phi_{eq})\to\infty$
([08](08_potential_design.md)).

## 4. Cube version

Geometry enters only through $a(R)/V$ and $v(R)/V$, so every formula transfers with
$V=1$, $a=4\pi R^{2}$, $v=\frac43\pi R^{3}$, $A\equiv8\pi/3$, $C\equiv4\pi^{3}/3$, $k=f''(1)=2$:

$$\delta=m-AR^{3}-C\gamma R,\qquad
E(R)=4\pi\sigma R^{2}+\tfrac{k}{2}\delta^{2},\qquad
h(R)=AR^{4}-mR+\tfrac{\sigma}{k}+C\gamma R^{2}=0,$$

with $m=AR_0^{3}+C\gamma R_0$ from the initial profile. Dropping both $C$-terms is the sharp-interface
limit. **They must be switched on or off together** — $m$ comes from the initial profile and $h$ from
the stationary one, and mixing them manufactures a spurious 10 % error (this is what my earlier
"criterion overpredicts by 5–9 %" actually was).

$$m_{\min}=\left(\tfrac{256}{27}AB^{3}\right)^{1/4}=1.6983\,\gamma^{3/8},\quad B=\tfrac{\sigma}{k},
\qquad
m_{\mathrm{eq}}=\tfrac{2\sigma}{kR_{\mathrm{eq}}}=1.9359\,\gamma^{3/8},\quad
R_{\mathrm{eq}}=\left(\tfrac{\sigma}{kA}\right)^{1/4}.$$

| $\gamma$ | $\xi=\sqrt{2\gamma}$ | $\xi/h$ at $64^3$ | $R_{0,\mathrm{crit}}$ sharp | $R_{0,\mathrm{crit}}$ with $O(\gamma)$ | $R_f$ | $R_{0,\mathrm{eq}}$ (quad.) | $R_{0,\mathrm{eq}}$ (exact bulk) |
|---|---|---|---|---|---|---|---|
| $2\cdot10^{-4}$ | 0.0200 | 1.28 | 0.2026 | 0.2020 | 0.1276 | 0.2116 | 0.2172 |
| $8\cdot10^{-4}$ | 0.0400 | 2.56 | 0.2409 | 0.2389 | 0.1518 | 0.2517 | 0.2639 |
| $3.2\cdot10^{-3}$ | 0.0800 | 5.12 | 0.2865 | 0.2797 | 0.1805 | 0.2993 | 0.3293 |
| $1.28\cdot10^{-2}$ | 0.1600 | 10.24 | 0.3407 | 0.3183 | 0.2146 | 0.3559 | 0.4500 |

$R_{0,\mathrm{crit}}\propto\gamma^{1/8}$ exactly; consecutive rows differ by $4^{1/8}=1.1892$.

**An isotropic box-size sweep would test nothing new.** Under $x\to Lx$ the functional on $[0,L]^3$
with coupling $\gamma$ is $L^{3}\times$ the unit-cube functional with $\gamma/L^{2}$, so
$R_{0,\mathrm{crit}}(L,\gamma)=L\,R_{0,\mathrm{crit}}(1,\gamma/L^{2})$ identically. Only an
**anisotropic** box changes $V$ independently of $\gamma$.

## 5. What the runs on disk already say

34 double-well / Neumann / constant-mobility runs: a $\gamma$ sweep (4 values × 5 radii), an $R_0$
sweep at $\gamma=3.2\cdot10^{-3}$ (7 radii), a grid study (64/128) and a $dt$ study (4 values).

**T1 — the $O(\gamma)$ mass correction is exact.** $\bar\phi(0)$ against $AR_0^{3}-1+C\gamma R_0$:
error $10^{-8}$ at $\gamma=2\cdot10^{-4}$, $10^{-4}$ at $3.2\cdot10^{-3}$, $3\cdot10^{-2}$ at
$1.28\cdot10^{-2}$ (where $O(\gamma^{2})$ takes over). Without the $C$-term the error is
$10^{-1}$. §3.6 confirmed.

**T2 — the threshold brackets all four $\gamma$.** Predicted 0.2026 / 0.2409 / 0.2865 / 0.3407
against observed (0.20, 0.24) / (0.24, 0.28) / (0.28, 0.30) / (0.32, —). Four for four, but the
brackets are 0.04 wide and cannot separate the sharp value from its $O(\gamma)$ correction.

**T3 — $R_\infty$ to better than 1.5 %,** provided both branches are treated consistently:

| $\gamma$ | $R_0$ | measured | sharp | err | with $O(\gamma)$ | err |
|---|---|---|---|---|---|---|
| $2\cdot10^{-4}$ | 0.32 | 0.3117 | 0.3115 | +0.06 % | 0.3115 | +0.05 % |
| $8\cdot10^{-4}$ | 0.28 | 0.2507 | 0.2498 | +0.38 % | 0.2505 | +0.09 % |
| $3.2\cdot10^{-3}$ | 0.30 | 0.2401 | 0.2393 | +0.34 % | 0.2462 | −2.5 % |
| $3.2\cdot10^{-3}$ | 0.32 | 0.2812 | 0.2771 | +1.47 % | 0.2804 | +0.29 % |
| $3.2\cdot10^{-3}$ | 0.33 | 0.2963 | 0.2926 | +1.27 % | 0.2951 | +0.42 % |

Mesh-independent: $64^{3}$ gives 0.2401 and $128^{3}$ gives 0.2395 at the same point, so the
residual is physics, not discretisation. Which of the two columns wins is still ambiguous at this
resolution — see R1.

**T4 — the second-order bulk shifts are measured, not just bounded.** At $\gamma=8\cdot10^{-4}$:

| | measured | $\psi/2$ | $\psi/2\mp\frac38\psi^{2}$ |
|---|---|---|---|
| droplet, $\phi_{\max}-1$ | 0.04877 | 0.05318 (−8 %) | 0.04894 (**−0.34 %**) |
| matrix, $\phi_{\min}+1$ | 0.05737 | 0.05318 (+8 %) | 0.05742 (**−0.09 %**) |

The first order is off by 8 % in opposite directions and the second order fixes both to 0.3 %.
At $\gamma=3.2\cdot10^{-3}$ the matrix still matches to 0.3 % while the droplet does not: at
$R/\epsilon=6$ the drop centre never reaches its bulk plateau, so $\phi_{\max}$ stops being the
bulk value. At $\gamma=2\cdot10^{-4}$ both drift to 1.5 % — that is the grid ($\xi=1.28h$).

**T5 — Gibbs–Thomson to 0.2 %, without logging $\psi$.** The $\frac38\psi^{2}$ terms cancel in the
sum, so $\psi=(\phi_{\max}-1)+(\phi_{\min}+1)$ to second order. Measured $\psi R/\sigma$ = 0.9980
and 0.9979 at $\gamma=8\cdot10^{-4}$. The residual divided by $(\xi/R)^{2}$ is −0.08 and −0.12,
i.e. consistent with zero: the $(\xi/R)^{2}$ coefficient is below what two points can resolve.

**T6 — the quadratic global-stability threshold is wrong, and the data says so.** At
$\gamma=3.2\cdot10^{-3}$, $R_0=0.30$ the droplet survives with $F/V=0.05401$ against
$f(\bar\phi)=0.05299$ — the homogeneous state is cheaper, i.e. the droplet is **metastable**. The
quadratic $m_{\mathrm{eq}}$ puts the crossover at $R_0=0.2993$ and so predicts the opposite;
keeping $f$ exact in the bulk moves it to 0.3293 and predicts what is observed. The note's own
caveat ("requires $m\ll1$") bites at $m=0.27$. Margin is only 2 % against a 9 % model gap, so this
is suggestive, not decisive — see R2.

**T7 — nothing ever grows, and that is exactly right.** In all 34 runs $R(t)$ is monotonically
decreasing. The reason is an identity: our initial condition ties $m$ to $R_0$, so
$h(R_0)=AR_0^{4}-mR_0+\sigma/k=\sigma/k>0$ **for every $R_0$**, i.e. the start is always on the
outer branch, above $R_\infty$. **The unstable root $R_c$, the barrier, and the whole growth branch
have never been touched by any run in this study.**

**Not confirmed: the rate law.** Note 03's $\dot R=-Mk\,h(R)/(2R^{2})$ (the monopole estimate)
fits the measured $R(t)$ with slopes 2–7 instead of 1 and $r^{2}<0.4$. The capacitance $4\pi R$ is
an open-space result and the box is only 3–4 radii across. This is a gap in note 03, not in the new
note, which makes no dynamical claims.

## 6. Test plan

### R1 — the threshold to 0.2 %, at fixed interface resolution *(no code change)*

The one measurement that separates the sharp prediction from its $O(\gamma)$ correction: they differ
by 0.3 % at $\gamma=2\cdot10^{-4}$ and 6.6 % at $\gamma=1.28\cdot10^{-2}$, while the current
brackets are 14 % wide.

Hold $\xi/h=5.12$ fixed so that the sweep is physics and not resolution:
$(\gamma,N)=(1.28\cdot10^{-2},32)$, $(3.2\cdot10^{-3},64)$, $(8\cdot10^{-4},128)$,
$(2\cdot10^{-4},256)$. Scan $R_0$ on a 0.002 grid spanning $\pm8$ % of each prediction.
Run to $|\dot R|<10^{-5}$ or $v=0$, with a step budget $\ge5\times$ the naive $t_{evap}$ — near the
threshold the lifetime diverges ($t_{evap}$ went 0.092 → 0.154 → 0.350 for $R_0=0.24,0.26,0.28$).

Fit the saddle node rather than reading off a bracket: $R_\infty-R_f=R_0\sqrt{\Delta R_0/2R_f}$, so
$(R_\infty-R_f)^{2}$ is linear in $R_0$ and its intercept gives $R_{0,\mathrm{crit}}$ from the
survivors alone, far more sharply than the last dead point. Then check $\gamma^{1/8}$ over the 64×
range and whether the residual matches $C\gamma$.

Cost: the two coarse $\gamma$ are minutes; $128^{3}$ is ~0.7 s/step, so ~2.5 GPU-h; $256^{3}$ is a
separate opt-in job (~12 GPU-h) and can be cut to five points bracketing 0.2026.

### R2 — the metastability window *(no code change)*

Predicted window $R_0\in[R_{0,\mathrm{crit}},R_{0,\mathrm{eq}}]$, and the exact-bulk and quadratic
versions disagree strongly: at $\gamma=8\cdot10^{-4}$, [0.2409, **0.2639**] against [0.2409, 0.2517].
Run $R_0=0.245,0.250,0.255,0.260,0.270,0.280$ at $\gamma=8\cdot10^{-4}$, $N=128$; for each survivor
compare the converged $F/V$ with $f(\bar\phi)$. Prediction: all survive, those below 0.2639 end
**above** $f(\bar\phi)$, those above end below. $\gamma=8\cdot10^{-4}$ rather than $3.2\cdot10^{-3}$
because $\xi/R\approx0.16$ there and the sharp-interface energy is good to ~1 %, against the 9 %
gap seen at the coarser $\gamma$.

Worth fixing first: `philic` uses a central difference for $\nabla\phi$, which underestimates the
gradient energy by $\sim\frac23(h/\xi)^{2}$ — 2.5 % at $\xi/h=5.12$. A one-sided-consistent or
staggered gradient would remove a known bias from exactly the quantity this test compares.

### R3 — the critical nucleus and the growth branch *(needs `--phi-mean`)*

The highest-value experiment, because nothing on disk probes it. It needs $m$ decoupled from $R_0$:
build $\phi=\phi_{eq}\tanh((R_{\rm init}-r)/\ell)$ as now, then add the uniform constant that makes
$\int\phi\,dV$ equal a requested value. One option, no change to the operator.

Design: $\gamma=8\cdot10^{-4}$, $N=128$, $m=0.1256$ — chosen so that $R_c=0.1200=3\xi$ is resolved
($m_{\min}=0.1171$, so the pair of roots is $R_c=0.1200$, $R_\infty=0.1860$).
Start at $R_{\rm init}=0.10,0.11,0.115,0.125,0.13,0.15$. Prediction: below 0.12 dissolve, above
0.12 **grow** to $R_\infty=0.186$. A growing droplet would be the first in this study, and the crossing
point measures $R_c$ directly. Also fit the growth rate against
$\dot R=-Mk\,h(R)/(2R^{2})$, which changes sign there — the cleanest available test of the rate law,
since $h$ is small and the prefactor drops out of the zero crossing.

### R4 — log $\psi$ *(two lines in `droplet_stats.h`)*

$\psi$ is already slot 0 of the state vector; `compute()` only reads slot 1. Adding
`psi_centre`, `psi_drop` (mean over $\phi>0$) and `psi_matrix` makes Gibbs–Thomson a direct
measurement instead of an inference from $\phi_{\max}+\phi_{\min}$, and it keeps working at small
$R/\epsilon$ where $\phi_{\max}$ stops being the bulk value. Cheapest useful change in the list.
Note also that `phi_max`/`phi_min`/`phi_centre` are **not** reduced across ranks — fine for the
single-GPU droplet runs, wrong for anything multi-rank.

### R5 — the order of the Gibbs–Thomson error *(after R4)*

The note's most distinctive claim, §3.7: $\psi_0R/\sigma-1=O((\xi/R)^{2})$ with no $O(\xi/R)$ term,
because $\psi_2=0$ by symmetry. Fix $R_\infty/\xi\approx8$ and sweep $\gamma$ over four decades at
fixed $\xi/h$; fit $a(\xi/R)+b(\xi/R)^{2}$ and check $a=0$. T5 shows the current two points are
consistent with $a=0$ but cannot resolve $b$. A falsifiable side test: the logarithmic potential is
also even, so it must give $a=0$ too, while an asymmetric potential must not.

### R6 — the radial profile *(no code change)*

`--save-coords` already dumps the field. Radially average the converged solution and compare with
$-\tanh\zeta+\frac{\sigma}{2R}\tanh^{2}\zeta$ — tests §3.5 including $\Phi_1$, which nothing else
here touches. Free: one run plus a Python script.

### R7 — anisotropic box *(needs a non-cubic domain)*

$V$ at fixed $\gamma$ is only reachable with an anisotropic box (§4). A $1\times1\times2$ domain
should move $R_{0,\mathrm{crit}}$ by $2^{1/4}=1.189$ — the same factor as $16\times$ in $\gamma$,
for a much smaller change. Low priority, but it is the only independent check of the $V^{1/4}$ law.

Order: **R1 + R2** (ready to launch, no code), then **R4 → R5**, then **R3** (most new physics),
with R6 free alongside and R7 last.

## 6a. The (R, m) grid: the landscape measured directly

*81 runs, `sweep_landscape`, job 96831, gamma = 3.2e-3, 64^3, with `--phi-mean` setting the
conserved mean independently of the radius. Validated against 9 runs at a pinned dt = 2e-3
(`sweep_landscape_ctrl`, job 96834): identical outcomes, and the same stationary radius to five
digits. Figure `figs/bifurcation.png`.*

Setting m independently is what finally reaches the unstable branch, and 15 of the 81 runs **grew** —
the first growing droplet in this study.

**The stable branch settles the O(gamma) question.**

| m | measured $R_\infty$ | with $O(\gamma)$ | err | sharp | err |
|---|---|---|---|---|---|
| 0.27 | 0.2451 | 0.2495 | −1.8 % | 0.2742 | −10.6 % |
| 0.30 | 0.2701 | 0.2704 | **−0.1 %** | 0.2921 | −7.5 % |
| 0.34 | 0.2948 | 0.2927 | **+0.7 %** | 0.3120 | −5.5 % |

The correction is not a refinement here, it is the difference between 1 % and 10 %.

**The existence threshold is still under-predicted.** Everything dies at m ≤ 0.24 and survives from
m = 0.27, so $m_{\min}\in(0.24,0.27)$ against 0.197 sharp and 0.220 corrected. Both are low; the
corrected one by ~10 %.

**The barrier is 30–40 % wider than predicted.** Measured brackets (0.140, 0.168) at m = 0.27 and
(0.113, 0.142) at m = 0.30, against 0.109 and 0.095. That is expected rather than surprising:
$R_c\simeq\xi/(3m)$, so at m ≈ 0.3 the critical nucleus is 1.4–1.9 interface widths across and the
thin-interface expansion has nothing left to stand on — exactly the note's own §5 caveat
"$R_c$ meaningful only for $m\ll1/3$". It also explains the $m_{\min}$ miss, since $m_{\min}$ is
where $R_c$ and $R_\infty$ merge.

**Two things the grid taught us about method:**

* The adaptive scheduler is safe here. Step counts fell from ~600 to ~15 with no change in the
  answer, in every control case.
* The energy sampled at step 0 follows the energy of the *initial condition* (both phases moved by
  one constant), not the constrained minimiser (phases on the two branches of $f'=\psi$). At these
  mean values the two differ visibly, so the landscape panel draws both.

## 7. Errata and nits in the note

- §5's caveat "$R_c$ meaningful only for $m\ll1/3$" is the binding one for us: at
  $\gamma=3.2\cdot10^{-3}$ and $m=0.4$ the nucleus is $R_c=0.067<\xi=0.08$. R3 is designed around it.
- §4.6's $m_{\mathrm{eq}}$ inherits the quadratic bulk and is off by 10 % at $m=0.27$ (T6). Worth
  stating the exact-bulk version, which is a two-line numerical solve.
- §6 is missing (the text jumps §5 → §7).
- Note 03 §2 says the left side of $r(1-r^{d})$ peaks at $d^{-1/d}$; it is $(d+1)^{-1/(d+1)}$, i.e.
  $4^{-1/3}=2^{-2/3}$ in 3D. The numerical constant 0.5053 that follows is correct — only the label
  was wrong.
