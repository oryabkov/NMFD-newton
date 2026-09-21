# Practical guide: how to keep a drop alive

Everything below is measured in this study unless marked otherwise. Configuration: 3D, closed unit
box, `test_time_cahn_hilliard` with `--init sphere`.

## 0. The one design rule

A drop survives iff it is **big enough relative to the domain**, and the threshold is set by a
single dimensionless comparison:

$$\boxed{\ \nu \;\gtrsim\; 0.705\,\frac{\epsilon}{R_0}\ }\qquad
\nu=\frac{V_{drop}}{V_{domain}},\quad \epsilon=\sqrt{\gamma/f''(\phi_{eq})}$$

i.e. **the drop's volume fraction must exceed about 0.7 times its Cahn number.** Equivalently
$R_c=(0.505\,\Lambda\epsilon V)^{1/4}$ with $\Lambda=1/3$ for the double well. Check it before
running: `py/predict.py` and `py/criterion.py`.

Measured thresholds (double well, unit box):

| $\epsilon$ | $R_c$ predicted | $R_c$ measured | drop must be $\ge$ |
|---|---|---|---|
| 0.08 | 0.341 / 0.388 | $>0.32$ | ~17 % of the box |
| 0.04 | 0.287 / 0.305 | (0.28, 0.30) | ~10 % of the box |
| 0.02 | 0.241 / 0.249 | (0.24, 0.28) | ~6 % of the box |
| 0.01 | 0.203 / 0.206 | (0.20, 0.24) | ~3.5 % of the box |

(predicted = linear / exact; both bracket correctly)

## 1. Free wins — do these always

**Use Neumann or periodic walls. Never Dirichlet.** Measured at $R_0=0.30$, $\epsilon=0.04$:

| BC | mass drift | outcome |
|---|---|---|
| Neumann | 0 (round-off) | survives, $R_\infty/R_0=0.800$ |
| periodic | 0 (round-off) | survives, $R_\infty/R_0=0.800$ |
| **Dirichlet** | **6.0 %** | **evaporates completely** |

Our `-1` boundary code sets $\phi=\psi=0$ at the wall, which both pins a spurious interface there
and opens a mass sink. It destroys a drop that is comfortably above $R_c$.

**Do not over-refine at fixed $\epsilon$.** $\epsilon/h=2.5$ (64³) and $\epsilon/h=5.1$ (128³) gave
$R_\infty/R_0 = 0.8002$ vs $0.7983$ — a **0.24 %** difference. Cells spent going beyond ~2.5 per
decay length buy nothing. Spend them on a smaller $\epsilon$ instead.

**Do not agonise over $dt$.** $R_\infty/R_0=0.8002$ for $dt=5\times10^{-4},10^{-3},2\times10^{-3},
4\times10^{-3}$ — identical to four decimals over an 8-fold range. The fully implicit scheme is
free here; use the largest $dt$ your other accuracy requirements allow.

**Start from a tanh profile, not a sharp step.** A sharp initial interface relaxes by consuming
drop mass before any quasi-static regime begins; at $R_0=0.20$ that transient alone dominated the
whole run. (Qualitative — the quantitative cube-vs-sphere comparison is not yet run.)

**Keep the domain tight.** $R_c\propto V^{1/4}$, so this is the cheapest real lever available:
halving the box in each direction shrinks $R_c$ by $1.68\times$. An over-sized domain is a bigger
mass sponge and kills drops for free.

## 2. The expensive lever, and its price

Reducing $\epsilon$ works — $R_c\propto\epsilon^{1/4}$, confirmed over a factor 8 in $\epsilon$
(`figs/Rc_vs_eps.png`) — but the fourth root is punishing. Cost of keeping a drop of a given size
alive, at $\epsilon/h=2.5$:

| $R_0/L$ | drop vol. fraction | $\epsilon/L$ needed | cells per dimension | total cells |
|---|---|---|---|---|
| 0.30 | 11 % | $4.8\times10^{-2}$ | 52 | $1.4\times10^5$ |
| 0.25 | 6.5 % | $2.3\times10^{-2}$ | 108 | $1.3\times10^6$ |
| 0.20 | 3.4 % | $9.5\times10^{-3}$ | 263 | $1.8\times10^7$ |
| 0.15 | 1.4 % | $3.0\times10^{-3}$ | 832 | $5.8\times10^8$ |
| **0.10** | **0.4 %** | $5.9\times10^{-4}$ | **4211** | $7.5\times10^{10}$ |
| 0.05 | 0.1 % | $3.7\times10^{-5}$ | 67373 | $3\times10^{14}$ |

**Read the 0.10 row.** A drop one tenth the size of the box — an entirely ordinary thing to want —
needs a $4200^3$ grid to be thermodynamically stable. That is the real severity of this problem,
and it is why parameter tuning alone cannot save small drops.

A $256^3$ run supports a drop down to $R_0/L\approx0.2$. That is the honest practical ceiling for
a single-GPU study.

## 3. What NOT to do

**Do not stiffen the bulk potential to "punish" the density rise.** Measured, at matched
$\epsilon=0.04$: $R_c$ *rises* monotonically with $\omega$ — (0.28,0.30) for the double well →
(0.29,0.31) at $\omega=2.5$ → (0.30,0.31) at $\omega=3$ → (0.36,0.38) at $\omega=4$ → $>0.42$ at
$\omega=6$. It makes things strictly worse. Reason in
[06](06_nonlinear_correction.md): the stiffness lands on the drop interior (a few % of the volume)
while the *ambient*, which absorbs the mass and occupies the whole domain, gets **softer**.

**Do not switch to the logarithmic potential expecting help with evaporation.** It does keep
$\phi$ inside $[-1,1]$, which is a legitimate reason to want it — but it costs ~10 % in $R_c$ at
$\omega=3$, and its singular wall only engages at $R\lesssim3\epsilon$, by which point the drop is
unresolved anyway. At well-resolved interfaces ($\epsilon=0.02$) the potentials differ by under
3 % and the choice is immaterial for survival.

**Do not read $\phi>1$ as a bug.** The overshoot is the Gibbs–Thomson shift and is predictable:

$$\max\phi \simeq 1+\tfrac23\,\frac{\epsilon}{R}\quad\text{(double well, 3D)}$$

measured to within 0.1 % on the surviving drops, with the maximum sitting exactly at the drop
centre. If you need $\max\phi<1.01$, that is a constraint $R>67\epsilon$ — far more demanding than
survival itself.

## 4. When you cannot satisfy the criterion

Which, per §2, is most of the time. Then accept that **Cahn–Hilliard has no stable single drop at
that size** and switch from a thermodynamic to a kinetic argument: make the drop's lifetime exceed
the simulated time.

$$t_{evap}=\frac{2\phi_{eq}^{2}R_0^{3}}{3M\sigma}$$

Mobility does not appear in $R_c$ — confirmed: every configuration below $R_c$ evaporated
regardless of mobility model, floor or averaging rule. It only buys time, $t_{evap}\propto1/M$.
A degenerate (parabolic) mobility bought $2.5\times$ at $R_0=0.20$.

Two caveats, both measured:
* the degenerate mobility must vanish at the potential's **actual** bulk values. Ours was hard-wired
  to $\pm1$, so with the logarithmic potential (bulk at $\pm0.859$) it left $M\approx0.26$ — no
  degeneracy at all. Fixed; check this if you port the trick elsewhere.
* harmonic face averaging with a floor of $10^{-5}$ **breaks the multigrid solve** (89 Newton
  iterations, GMRES failures). Usable floors are $\gtrsim10^{-3}$.

*The `faceavg` sweep quantifying how much time the averaging rule actually buys is still running;
§4's numbers may sharpen.* Beyond this, the structural fix is an explicit mass-conservation
constraint (Lagrange multiplier / mass-conserving CH variants) — not tested here, and not discussed
by Yue et al. either.

## 5. Checklist

1. Compute $\nu$ and $\epsilon/R_0$. Is $\nu \ge 0.7\,\epsilon/R_0$? If yes, the drop is stable —
   you are done.
2. If no: shrink the domain until it is, if the physics lets you.
3. If still no: reduce $\epsilon$ using §2's table, up to what you can afford.
4. If still no: compute $t_{evap}$ and check it against your simulation time. Degenerate mobility
   to buy the factor you need, floor no smaller than $10^{-3}$.
5. Use Neumann/periodic walls, a tanh initial profile, $\epsilon/h\approx2.5$, and the largest $dt$
   you can justify.
6. Report the drop's mass loss over the run. It is never zero, and now it is predictable.
