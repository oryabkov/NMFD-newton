# Spontaneous droplet shrinkage in Cahn–Hilliard — research roadmap

**Living document.** Updated as data arrives. Last update: 2026-09-18.

## Question

A drop of one phase in a bulk of the other spontaneously loses mass and, below a critical size,
evaporates entirely. This is a known artefact of Cahn–Hilliard dynamics (Yue, Zhou & Feng, JCP 223
(2007) 1–9). Observations in our code so far:

* with the standard double-well potential the drop interior overshoots, $\phi>1$;
* with the logarithmic potential $\phi$ stays in $[-1,1]$ but **the drop still evaporates**;
* nonlinear (degenerate) mobility suppresses the effect.

We want to know *why*, quantitatively, and which knob actually controls it.

## Layout

```
droplet/
  README.md                        this roadmap
  notes/01_yue2007_theory.md       distilled reference theory
  notes/02_potential_shapes_table.md  derived numbers for every potential
  notes/03_shrinkage_criterion.md  generalised criterion + predictions P1..P7
  notes/04_mobility_averaging.md   face-averaging analysis (item #2)
  notes/05_results_phase1.md       phase 1 measurements
  notes/06_nonlinear_correction.md exact criterion; why omega backfires
  py/                              analysis + plotting
  figs/                            generated figures
  slurm/                           sweep jobs
  data/                            run outputs (gitignored)
```

## Current state of the theory

Derived in [notes/03](notes/03_shrinkage_criterion.md) and verified to reproduce Yue's 2D and 3D
results **exactly**:

$$R_c=\Big(0.5053\;\Lambda\,\epsilon\,V\Big)^{1/4}\ (3\mathrm D),
\qquad
\Lambda=\frac{\sigma}{2\phi_{eq}^{2}f''(\phi_{eq})\,\epsilon},
\qquad \epsilon=\sqrt{\gamma/f''(\phi_{eq})}.$$

A drop with $R_0<R_c$ evaporates; otherwise it shrinks to $R_\infty$, the larger root of
$r(1-r^3)=3\Lambda\epsilon V/(4\pi R_0^4)$.

This linearised form is **confirmed for the double well** (see notes/05) but **fails across
potentials**: it predicts the logarithmic potential lowers $R_c$ by 6 %, whereas measurement shows
it *raises* it by ~10 %. The nonlinear correction — the two bulk phases respond with different
stiffnesses, and the exterior softens — is in
[notes/06](notes/06_nonlinear_correction.md) and `py/criterion.py`, and it reverses the conclusion:
**stiffening the potential backfires.** Mobility still does not enter $R_c$ at all; it only sets
the *rate*.

## Roadmap

Legend: `[ ]` todo `[~]` in progress `[x]` done `[!]` blocked/surprising result

### Phase 0 — instrumentation

* `[x]` **0.1** Potential shapes vs $\omega$ (supervisor item #1). `py/plot_potentials.py`,
  `figs/potential_{raw,normalized,scalars}.png`, `notes/02`.
  **Result:** after matching $\phi_{eq}\to1$ and $f''(\phi_{eq})\to2$, the logarithmic potential
  lies *on top of* the double well until it hits its singular wall; the only real difference is the
  hard wall at relative overshoot $1/\phi_{eq}-1$ (16 % at $\omega=3$, 0.5 % at $\omega=6$). The
  supervisor's reading is correct: at $\omega=3$ we mostly just moved the equilibria to $\pm0.859$.
* `[x]` **0.2** Yue theory extracted → `notes/01`.
* `[x]` **0.3** Generalised criterion + $\Lambda$ → `notes/03`. Cross-checks against Yue exactly.
* `[x]` **0.4** Solver instrumentation (see *Code work* below): droplet IC, droplet diagnostics,
  runtime choice of potential / mobility / face-averaging / BC.
* `[x]` **0.5** Post-processing: $R(t)$, $M_{drop}(t)$, $\max\phi(t)$, energy split from the logs.

### Phase 1 — baseline and criterion test

* `[x]` **1.1** Cube IC, double well, constant mobility, **Neumann** walls (closed box, mass
  conserved to machine precision). Verify total mass conservation first — this is the sanity gate
  for everything else.
* `[x]` **1.2** Sweep $R_0$ across the predicted $R_c$ at fixed $\gamma$, $L$, grid. **Tests P1.**
* `[x]` **1.3** Measure $R_\infty/R_0$ against the predicted root. **Tests P2.**
* `[~]` **1.4** Sweep $\gamma$ (hence $\epsilon$) and $L$. **Tests P3** ($R_c\propto(\epsilon V)^{1/4}$).
* `[~]` **1.5** Grid convergence at fixed $\epsilon$ ($h/\epsilon$ = 4, 2, 1, 0.5) and $dt$ study —
  separate the *physical* artefact from the *discretisation* error. Non-negotiable before any
  conclusion.
* `[~]` **1.6** Dirichlet-wall control run. Our `-1` BC sets $\phi=\psi=0$ at the wall and therefore
  **leaks mass**; quantify the leak so the user's requested Dirichlet runs can be interpreted.

### Phase 2 — the potential lever (supervisor item #1)

* `[~]` **2.1** Logarithmic potential, $\omega\in\{2.5,3,4,6\}$, each with $\gamma=\epsilon^2f''(\omega)$
  so that $\epsilon$ is **matched**. Prediction P4: $R_c$ ratios $0.968/0.938/0.876/0.736$.
  If confirmed → the potential is definitively not the fix, with a number attached.
* `[ ]` **2.2** Overshoot law P5: $\max\phi-\phi_{eq}=(d-1)\Lambda\phi_{eq}\epsilon/R(t)$. For the
  double well this predicts the observed $\phi>1$; for the log potential it predicts the wall is
  only reached at $R\lesssim3\epsilon$.
* `[ ]` **2.3** *If* 2.1/2.2 confirm the theory, the follow-up is a potential designed to minimise
  $\Lambda$ at matched $\sigma$ and $\epsilon$ — i.e. stiff wells with a low flat barrier
  ($\Lambda=\frac{1}{2\sqrt2}\int\sqrt{2\Delta g}\,du$ in curvature-matched units). Expected to be
  a weak lever too (1/4 power) and to degrade the interface profile; worth one run to close the
  question rather than leave it open.

### Phase 3 — the mobility lever (supervisor item #2)

* `[!]` **3.1** Face-averaging comparison at fixed everything else: midpoint (current) vs
  arithmetic vs harmonic, with the degenerate parabolic mobility. **Tests P7.**
  Expected: midpoint passes $M(\phi_{eq}/2)\approx0.75\,M_{max}$ across a pure/interface cell pair,
  arithmetic $0.5\,M_{max}$, harmonic $\approx2M_{min}$ — i.e. only the harmonic mean actually
  blocks the bulk. Harmonic is also the *more accurate* face rule for a sharply varying
  coefficient (exact for 1D series resistance), so this is a correctness fix, not only a knob.
* `[!]` **3.2** Mobility floor sweep $M_{min}/M_{max}\in\{10^{-2},10^{-3},10^{-4},10^{-5}\}$ with
  harmonic averaging. Expect drop lifetime $\propto M_{min}^{-1}$ and solver conditioning
  $\propto M_{min}^{-1}$ — find the usable corner.
* `[x]` **3.3** Confirm P6: $R_c$ (the *thermodynamic* threshold) is unchanged by mobility; only
  $t_{sh}$ moves. Run long enough to show the drop still dies, just slowly.
* `[x]` **3.4** Solver health under harmonic + degenerate mobility: Newton and GMRES/MG iteration
  counts. The smoother's $\psi$-diagonal currently uses the *cell* mobility instead of the face
  mobilities — wrong by orders of magnitude once $M$ is degenerate, and a likely cause of past
  smoother trouble. Fixed as part of 0.4; verify it pays off here.

### Phase 4 — synthesis

* `[ ]` **4.1** One figure: measured $R_\infty/R_0$ vs $R_0/R_c$, all potentials and mobilities
  collapsed onto the single predicted curve.
* `[ ]` **4.2** Write-up with the practical recipe: given a target drop size and domain, what
  $\epsilon$, $\gamma$, mobility and face rule keep the drop alive for the simulation time.

## Code work (task 0.4)

All in `test/cahn_hilliard/`. Default behaviour of every existing test must stay **bit-identical**.

1. `include/kernels/mobility.h`
   * `parabolic_mobility` currently ignores its constructor arguments (`D_(1), offset_(1e-5)`) —
     fix, and expose the floor as a real knob.
   * add a face-averaging rule (`midpoint` = current default, `arithmetic`, `harmonic`) with
     `face(a,b)`, `face_diff_a(a,b)`, `face_diff_b(a,b)`.
2. `include/kernels/{cahn_hilliard_op,jacobi_op,jacobi_pre}.h` — use the face interface; drop the
   hard-coded $\tfrac12$ chain-rule factors; fix the smoother's $\psi$-diagonal to use face
   mobilities.
3. `include/{cahn_hilliard_op,jacobi_op,jacobi_pre}.h` — `set_phobic_energy` so $\omega$ is settable.
4. `test_time_cahn_hilliard.cpp` — `--init cube|sphere|trig` (trig = today's behaviour), droplet
   diagnostics, and `--potential/--omega/--mobility/--mobility-floor/--face-avg/--gamma/--bc`.
   `run()` templated on the potential and mobility policies, dispatched on the CLI strings
   (a full CUDA build is 31 s, so 4 instantiations are affordable).

## Results so far

See [notes/05_results_phase1.md](notes/05_results_phase1.md) for the full write-up.

* **Criterion confirmed to ~2 %.** Transition bracketed in (0.28, 0.30) against a predicted
  $R_c=0.2865$; $R_\infty/R_0$ matched to 0.3 % and 1.3 %; every point on the parameter-free master
  curve (`figs/r0_dw_collapse.png`). Mass conserved to round-off in all 16 runs.
* **Overshoot law confirmed.** $\max\phi=1+\tfrac23\epsilon/R$ to 0.1 %, with the maximum at the
  drop centre — the $\phi>1$ "overcompressed" interior is the Gibbs–Thomson shift, not a bug.
* **P6 confirmed:** mobility changes the rate (2.5x), never the outcome.
* **Surprise (provisional):** the face-averaging rule and the mobility floor made *no* difference.
  Working hypothesis: mass escapes through the interface shell where $M\approx D$ under every rule,
  and then un-degenerates $M$ in the bulk it accumulates in. Rerun at $R_0=0.26$ is queued.
* **Harmonic + floor $10^{-5}$ breaks the MG/GMRES solve** (89 Newton its, convergence failures).
  Usable corner is $M_{min}/M_{max}\gtrsim10^{-3}$.

## Open questions / risks

* **Mass conservation is the gate.** If the discrete scheme does not conserve $\int\phi$ to
  round-off under Neumann walls, the whole shrinkage measurement is meaningless. Check first.
* Harmonic averaging with $M_{min}/M_{max}=10^{-5}$ gives a coefficient contrast of $10^5$ inside
  the linear system. Multigrid may stall. If it does, that is a result in itself (and points back
  at `ADAPTIVE_ALPHA_SMOOTHER.md`).
* The cube IC has sharp corners and no tanh profile; the first few steps will be dominated by
  profile relaxation, which itself consumes drop mass. Yue's $\delta r$ formulas assume a relaxed
  tanh profile. Either start from a tanh-profiled sphere or discard the initial transient — this
  materially affects P2. Plan: implement **both** `cube` and `sphere` (tanh-profiled) and use
  `sphere` for the quantitative tests, `cube` for the qualitative one the user asked for.

## Phase 5 — potential design (open)

Driven by [notes/08_potential_design.md](notes/08_potential_design.md). The shape functional

$$R_c^4=0.2527\,V\,W\,K[f],\qquad K[f]=\hat\sigma\,/\,(\phi_{eq}^2\,f''(\phi_{eq})\,\hat w)$$

at fixed *resolved* interface width $W$ identifies the only real lever: the curvature of the
well at the bulk value, concentrated in a neighbourhood thin compared with $W$. Candidates and
predicted thresholds are in notes/08 and `figs/potential_zoo.png`.

* [x] `phobic_energy.h`: `smoothed_obstacle_potential`,
  $f_\eta=a_\eta(\sqrt{(1-\phi^2)^2+\eta^4}-\eta^2)$ with $a_\eta$ pinning the barrier at $1/4$,
  $f''(\pm1)=4a_\eta/\eta^2$, `--potential smoothed_obstacle --eta`. $C^\infty$, so convex
  splitting / Newton / MG are unchanged. Also `get_profile_scale( gamma )` per potential.
* [x] `sweep_eta` (job 96523, 28 runs): the bracket walks from (0.28, 0.30) down to (0.12, 0.15),
  i.e. $R_c$ falls $2.1\times$ and the smallest surviving drop volume $10\times$ — 10.2 % of the
  box down to 1.0 %. Matches the exact criterion to 5–9 %, the same bias the double well shows.
  Table in [notes/08](notes/08_potential_design.md) §4b, figure `figs/eta_threshold.png`.
* [x] `etaprobe` (job 96522): **no** $\eta^{-2}$ conditioning failure — 2 Newton iterations to
  $10^{-10}$ at $\eta=0.01$ ($f''=10^4$). The prediction that the solver would set the usable
  $\eta$ was wrong; $\eta$ is limited by saturation of the criterion instead.
* [x] overshoot: $\max\phi-1=\psi/f''(1)$ to 6 % over four decades; $2.2\times10^{-5}$ at
  $\eta=0.01$ against $8.9\times10^{-2}$ for the double well at the same radius.
* [ ] the true double obstacle is **not** worth its variational-inequality machinery here: the
  measured brackets at $\eta=0.1$ and $0.05$ already straddle the family's floor $0.126$.
* [ ] open: repeat two points at $128^3$ to separate the 5–9 % criterion bias from discretisation.
