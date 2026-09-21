# The linearised criterion fails across potentials — and the correction reverses the conclusion

## 1. The discrepancy

The `r0_log` sweep (logarithmic potential, $\omega=3$, $\gamma$ chosen so $\epsilon=0.04$ matches the
double-well runs exactly) does **not** follow the linearised criterion of
[03](03_shrinkage_criterion.md):

| potential | $R_c$ predicted (linear $\Lambda$) | $R_c$ measured | verdict |
|---|---|---|---|
| double well | 0.2865 | in (0.28, 0.30) | ✓ |
| logarithmic $\omega=3$ | 0.2686 | **in (0.30, 0.33)** | ✗ — and in the *wrong direction* |

The linear theory says the logarithmic potential should make the drop **more** robust
($R_c$ smaller by 6 %). The measurement says it makes it **less** robust ($R_c$ larger by ~10 %).

This is not a run-length artefact. The double-well survivors reach an exact plateau
($dR/dt\sim10^{-16}$) by $t\approx0.3$, long before their runs end at $t=1.2$, while the
logarithmic drops at $R_0=0.28$ and $0.30$ die at $t=0.2$ and $t=0.32$ — well inside the window.
Both sweeps are converged.

## 2. What the linearisation got wrong

The linearised criterion assumes both bulk phases respond with the same tangent stiffness
$f''(\phi_{eq})$, so the shift is $\delta=\mu/f''(\phi_{eq})$ on both sides. That is wrong whenever
the potential is strongly asymmetric about its well — which is exactly what the logarithmic
potential is:

* the **drop interior** shifts from $+\phi_{eq}$ towards $+1$, i.e. **into** the singular wall, where
  $f''$ rises steeply — it stiffens;
* the **exterior** shifts from $-\phi_{eq}$ towards $0$, i.e. **away** from the wall and down the
  flank towards the spinodal, where $f''$ *falls* — it softens.

And it is the exterior that matters: it occupies essentially the whole domain, so it absorbs
essentially all the released mass ($V_d\ll V$). A softer exterior is a **bigger mass sponge**: for a
given released mass it develops *less* supersaturation, so the ambient never reaches the level the
drop's Gibbs–Thomson potential demands, and the drop keeps shrinking.

The logarithmic potential concentrates its stiffness on the side that does not matter and removes
it from the side that does.

## 3. Exact criterion

Implemented in [`py/criterion.py`](../py/criterion.py). With $\mu$ the uniform chemical potential,

$$f'(\phi_{eq}+\delta_{in})=\mu,\qquad f'(-\phi_{eq}+\delta_{out})=\mu,\qquad
\mu=\frac{(d-1)\sigma}{2\phi_{eq}R},$$

$$2\phi_{eq}\big(V_d(R_0)-V_d(R)\big)=\delta_{in}V_d(R)+\delta_{out}\big(V-V_d(R)\big).$$

$R_\infty$ is the largest root below $R_0$; no root means evaporation. Note the exterior branch has
a hard ceiling: $f'$ on $(-\phi_{eq},0)$ peaks at the **spinodal** point
$\phi_s=-\sqrt{1-2/\omega}$ (or $-1/\sqrt3$ for the double well). If the Laplace-driven $\mu$
exceeds $f'(\phi_s)$, the ambient phase cannot hold the supersaturation at any level and the drop
is doomed regardless of mass budget.

| potential ($\epsilon=0.04$) | $\sigma$ | $R_c$ linear | $R_c$ exact | $R_c$ measured |
|---|---|---|---|---|
| double well | 0.0533 | 0.2865 | 0.3066 | (0.28, 0.30) — точная **выше** вилки |
| log $\omega=2.5$ | 0.0183 | 0.2777 | **0.3096** | **(0.29, 0.31)** ✓ |
| log $\omega=3$ | 0.0700 | 0.2686 | 0.3170 | (0.30, 0.31) |
| log $\omega=4$ | 0.2829 | 0.2497 | 0.3517 | (0.36, 0.38) |
| log $\omega=6$ | 1.4789 | 0.2109 | 0.5759 | $>0.42$ |

The exact criterion lands **inside** the measured bracket for the logarithmic potential and gets the
ordering right; the linear one is outside it and has the sign of the effect backwards. For the
double well the exact value is 1.6 % above the observed bracket — both versions are within the
$O(\epsilon/R)\approx13\%$ error one should expect from sharp-interface asymptotics at these radii,
so the double-well agreement does not discriminate between them. **The logarithmic runs do.**

## 4. The conclusion for supervisor item #1, revised

Previously: "stiffening the potential is a weak lever, $R_c\propto\Lambda^{1/4}$, worth 6 %."

Now: **stiffening the potential this way actively backfires.** $R_c$ *increases* with $\omega$:
0.307 (double well) → 0.317 ($\omega=3$) → 0.352 ($\omega=4$) → 0.576 ($\omega=6$). At $\omega=6$
the predicted critical diameter is 1.15 — larger than the box — so *no* drop survives in a unit
domain at all.

Two effects compound:

1. **The soft-flank sponge** of §2 — raising $\omega$ narrows the wells and widens the soft region
   the ambient relaxes into.
2. **Matched-$\epsilon$ comparison forces $\sigma$ up.** Holding the interface width fixed while
   raising $\omega$ requires $\gamma=\epsilon^2f''(\phi_{eq})$, so $\sigma$ climbs from 0.053 to
   1.479 between the double well and $\omega=6$ — a 28-fold larger Laplace pressure driving the
   drop's dissolution. The linear $\Lambda$ did account for this and still predicted net
   improvement, because $f''$ grows faster than $\sigma$; the nonlinear softening of the exterior is
   what tips the balance the other way.

So the supervisor's hypothesis — "штрафовать её посильнее за увеличение плотности в центре" — is
correct about *where* the penalty lands but wrong about the consequence. Penalising the centre does
suppress $\delta_{in}$, but $\delta_{in}$ was never the bottleneck: it acts on $V_d$, a few percent
of the domain. What controls survival is $\delta_{out}$ acting on $V$, and the same potential change
makes that *worse*.

**This predicts what a potential that actually helps must look like:** stiff on the flank the
*ambient* relaxes into (i.e. between $-\phi_{eq}$ and 0), not at the wells. That is the opposite of
adding a singular wall at $\pm1$, and it is a concrete design target — see Phase 2.3.

## 5. Verdict — the $\omega$ sweep has landed

15 runs, $\omega\in\{2.5,3,4,6\}$, all at matched $\epsilon=0.04$. See `figs/Rc_vs_omega.png`.

$R_c$ **increases monotonically with $\omega$**: (0.28, 0.30) for the double well → (0.29, 0.31) at
$\omega=2.5$ → (0.30, 0.31) at $\omega=3$ → **(0.36, 0.38)** at $\omega=4$ → $>0.42$ at $\omega=6$.

* The linearised $\Lambda$ theory predicts a *decreasing* $R_c$ (0.278 → 0.269 → 0.250 → 0.211).
  It is outside the measured bracket at **every** $\omega$, and has the sign of the trend wrong.
* The exact criterion gets the **sign** of the trend right, which is what matters here, but it is
  not quantitatively reliable: of five potentials only $\omega=2.5$ lands inside the measured
  bracket; $\omega=3$ is 2.2 % high, $\omega=4$ is 2.3 % **low** (opposite sign of error), and for
  the double well it gives $R_c=0.3066$, i.e. it predicts death for the $R_0=0.30$ drop that in
  fact survived at $R_\infty/R_0=0.80$. Both versions inherit an $O(\epsilon/R)$ error: a direct
  check of the mass budget against the measured fields leaves a 13 % residue, which sits in the
  diffuse layer the sharp-interface accounting ignores.

At $\omega=6$ the drops die in 48–57 steps ($t\approx0.1$) rather than the several hundred steps
typical elsewhere — consistent with $\sigma$ being 28$\times$ larger, hence a far stronger Laplace
drive.

**§4 is now a measured result, not a hypothesis.** Raising $\omega$ to "punish" the density rise in
the drop centre makes the drop strictly easier to destroy, and by $\omega=6$ no drop survives in a
unit box at all.

The $\epsilon$-sweep matters too: at $\epsilon=0.02$ the exact criterion gives 0.249 / 0.247 / 0.255
/ 0.335 for double well / $\omega=3$ / $\omega=4$ / $\omega=6$ — i.e. the whole effect largely
**washes out as the interface thins**, as it must, since it is driven by $\delta_{out}\propto\sigma/R$.
That is itself a falsifiable prediction and a useful practical statement: the potential choice
matters most exactly where the interface is least resolved.
