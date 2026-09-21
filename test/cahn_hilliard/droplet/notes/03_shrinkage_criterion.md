# A generalised shrinkage criterion, and why the potential is a weak lever

*Derived independently from an Ostwald/Gibbs–Thomson mass balance, then cross-checked against
Yue, Zhou & Feng (JCP 223, 2007) — see [01_yue2007_theory.md](01_yue2007_theory.md). It reproduces
their 2D **and** 3D critical radii exactly, and generalises them to an arbitrary bulk potential.*

## 0. Conventions

Our solver (`test/cahn_hilliard/include/kernels/cahn_hilliard_op.h`) discretises

$$\mathscr F[\phi]=\int_\Omega\Big[f(\phi)+\tfrac{\gamma}{2}|\nabla\phi|^2\Big]\,d\Omega,
\qquad
\partial_t\phi=\nabla\!\cdot\!\big(M(\phi)\nabla\psi\big),\quad
\psi=f'(\phi)-\gamma\Delta\phi .$$

so `gamma` in the code is $\gamma$ and `phobic_energy` is $f$. Derived quantities:

| symbol | definition | meaning |
|---|---|---|
| $\phi_{eq}$ | positive root of $f'$ | bulk equilibrium value |
| $k\equiv f''(\phi_{eq})$ | | bulk stiffness |
| $\epsilon=\sqrt{\gamma/k}$ | | interface decay length, $\phi\simeq\phi_{eq}-Ae^{-x/\epsilon}$ |
| $\sigma=\int_{-\phi_{eq}}^{\phi_{eq}}\sqrt{2\gamma\,\Delta f}\,d\phi$ | $\Delta f=f-f(\phi_{eq})$ | surface tension |

Yue et al. use $\mathscr F=\int\lambda[\tfrac12|\nabla\phi|^2+(\phi^2-1)^2/4\epsilon_Y^2]$, so
$\gamma=\lambda$, $k=2\lambda/\epsilon_Y^2$ and **$\epsilon_Y=\sqrt2\,\epsilon$**. Every comparison
below uses that translation.

## 1. The shape number $\Lambda$

A spherical drop of radius $R$ carries a Gibbs–Thomson chemical potential. Moving the interface
inward by $\delta R$ removes mass $A\,\delta R\,\Delta\phi$ with $\Delta\phi=2\phi_{eq}$ the jump,
and releases interfacial energy $\sigma\,(d-1)A\,\delta R/R$, so

$$\mu=\frac{(d-1)\,\sigma}{2\phi_{eq}R},
\qquad
\delta\phi=\frac{\mu}{k}=(d-1)\,\Lambda\,\phi_{eq}\,\frac{\epsilon}{R},$$

$$\boxed{\ \Lambda\;\equiv\;\frac{\sigma}{2\,\phi_{eq}^{2}\,f''(\phi_{eq})\,\epsilon}\ }$$

$\Lambda$ is a **pure shape number**: it is invariant under rescaling $\phi\to c\phi$ and under
rescaling $\gamma$. Equivalently, in curvature-matched normalised units
($u=\phi/\phi_{eq}$, energy scaled so $g''(1)=2$),

$$\Lambda=\frac{1}{2\sqrt2}\int_{-1}^{1}\sqrt{2\,\Delta g(u)}\,du,$$

i.e. **$\Lambda$ measures the barrier area relative to the well stiffness, and nothing else.**

| potential | $\phi_{eq}$ | $f''(\phi_{eq})$ | $\sigma/\sqrt\gamma$ | $\Lambda$ | $\Lambda^{1/4}$ rel. to DW |
|---|---|---|---|---|---|
| double well | 1.0000 | 2.000 | 0.9428 | **0.3333** | 1.000 |
| log, $\omega=2.5$ | 0.7104 | 1.538 | 0.3684 | 0.2943 | 0.968 |
| log, $\omega=3$ (**our default**) | 0.8586 | 4.608 | 0.8154 | **0.2576** | **0.938** |
| log, $\omega=4$ | 0.9575 | 20.04 | 1.5798 | 0.1925 | 0.876 |
| log, $\omega=6$ | 0.9949 | 190.6 | 2.6778 | 0.0980 | 0.736 |
| log, $\omega=10$ | 0.9999 | 10994 | 4.1265 | 0.0197 | 0.493 |

As $\omega\to2^+$ the logarithmic potential *is* the double well (Landau expansion
$f\simeq-a\phi^2+\phi^4/6$, $a=\omega/2-1$), and indeed $\Lambda\to1/3$.

## 2. Critical radius

Mass balance in a closed box of volume $V$: shrinking from $R_0$ to $R=rR_0$ releases
$2\phi_{eq}\,(V_d(R_0)-V_d(R))$, which raises the ambient level by $\delta\phi_{out}$. An
equilibrium radius exists iff the released mass can supply the Gibbs–Thomson supersaturation
$\delta\phi=\mu/k$ that radius demands:

$$r\,(1-r^{d})\;=\;\frac{\Lambda\,\epsilon\,V}{c_d\,R_0^{\,d+1}},
\qquad c_2=2\pi,\quad c_3=\tfrac{4\pi}{3}.$$

The left side peaks at $r_i=d^{-1/d}$ — **exactly Yue's inflexion point**
($r_0/\sqrt3$ in 2D, $r_0/2^{2/3}$ in 3D). A root exists, i.e. the drop survives at some
$R_\infty>0$, iff

$$\boxed{\;R_0^{\,4}\;\ge\;0.5053\;\Lambda\,\epsilon\,V\;=\;0.5053\,\frac{\sigma V}{2\phi_{eq}^{2}f''(\phi_{eq})}\;}\qquad(3\mathrm D)$$

$$R_0^{\,3}\;\ge\;0.4135\,\Lambda\,\epsilon\,V \qquad(2\mathrm D)$$

**Verification against Yue.** Setting $\Lambda=1/3$ (double well) and $\epsilon=\epsilon_Y/\sqrt2$:

| | this derivation | Yue et al. | 
|---|---|---|
| 3D | $r_c^4=0.16843\,V\epsilon_Y$ | $r_c^4=\dfrac{2^{1/6}}{3\pi}V\epsilon_Y=0.16843\,V\epsilon_Y$ |
| 2D | $r_c^3=0.13783\,V\epsilon_Y$ | $r_c^3=\dfrac{\sqrt6}{8\pi}V\epsilon_Y=0.13783\,V\epsilon_Y$ |
| small-$\delta r$ | $\delta r/r_0=-\dfrac19\dfrac{V}{V_d}\dfrac{\epsilon}{r_0}$ | $-\dfrac{\sqrt2}{18}\dfrac{V}{V_d}\dfrac{\epsilon_Y}{r_0}$ — identical |

Exact agreement in all three, from a completely different route. The generalisation is that **every
property of the bulk potential enters only through the single combination
$\Lambda\epsilon=\sigma/(2\phi_{eq}^2f'')$, and only at the power $1/4$.**

## 3. Consequences — the three claims to test

**(C1) The potential is a fourth-root lever, so it cannot fix the explosion.**
Switching double well $\to$ logarithmic at $\omega=3$ changes $\Lambda$ from $0.333$ to $0.258$:
the critical radius moves by $(0.258/0.333)^{1/4}=0.94$, a **6 % effect**. Even $\omega=10$ —
which needs $f''=1.1\times10^4$, hence $\gamma\sim10^4\epsilon^2$ — buys only a factor $2$.
This quantitatively explains the observation *"с логарифмическим потенциалом капля всё равно
взрывается"*.

**(C2) The singularity of the log potential only bites at $R\lesssim3\epsilon$.**
The required overshoot is $\delta\phi=(d-1)\Lambda\phi_{eq}\epsilon/R$; the available headroom is
$1-\phi_{eq}$. At $\omega=3$: $\delta\phi=0.443\,\epsilon/R$ vs headroom $0.141$, so the wall is
reached only when $R<3.1\,\epsilon$ — a drop three interface widths across, i.e. already
unresolved. **Above that radius the logarithmic potential is dynamically indistinguishable from the
double well.** This is precisely the supervisor's intuition, made quantitative: matching $g''$ at
equilibrium collapses the two potentials onto the *same* curve (right panel of
`figs/potential_normalized.png`) until the hard wall, which is far away.

**(C3) The double-well overshoot is predictable:** $\max\phi\simeq1+\tfrac23\,\epsilon/R$.
That is the "пережатая жидкость" — it is not a bug, it is the Gibbs–Thomson shift.

## 4. What is *not* a fourth-root lever

$\Lambda$, $\epsilon$ and $V$ all enter the **thermodynamic** criterion at power $1/4$. The
mobility does **not appear at all** — degenerate mobility cannot change $r_c$. What it changes is
the **rate**. Yue's estimate (their §5): $t_{sh}\approx100\,r_0\Delta r\,\epsilon/(\gamma_M\sigma)$,
i.e. the escape is kinetic — make the drop's lifetime exceed the simulation time. With
$M(\phi)\to0$ in the bulk, the flux $-M\nabla\psi$ that carries mass away from the drop is
switched off, and $t_{sh}\to\infty$ while $r_c$ is untouched.

This is why nonlinear diffusion works where the potential does not, **and it predicts that the
face-averaging rule matters more than the potential**: with $M_{i+1/2}$ an arithmetic-type average,
one "open" cell adjacent to a "blocked" one still passes $\sim M_{max}/2$; with a harmonic average
it passes $\sim 2M_{min}$. The harmonic mean is the one that actually enforces the block —
supervisor item #2. See [`04_mobility_averaging.md`](04_mobility_averaging.md).

## 4a. The rate, and hence the run length

Quasi-static Ostwald transport: a sphere of radius $R$ acts as a diffusive monopole of
"capacitance" $4\pi R$, so the mass flux out of the drop is $Q=4\pi R\,M\,(\mu_{drop}-\mu_\infty)$
and

$$\frac{d}{dt}\Big[2\phi_{eq}\tfrac43\pi R^3\Big]=-Q
\quad\Longrightarrow\quad
\frac{dR}{dt}=-\frac{M\,(\mu_{drop}-\mu_\infty)}{2\phi_{eq}R}.$$

Early on ($\mu_\infty\approx0$, $\mu_{drop}=\sigma/(\phi_{eq}R)$ in 3D):

$$\frac{dR}{dt}=-\frac{M\sigma}{2\phi_{eq}^{2}R^{2}}
\quad\Longrightarrow\quad
R^3(t)=R_0^3-\frac{3M\sigma}{2\phi_{eq}^{2}}\,t,
\qquad
\boxed{\;t_{evap}=\frac{2\phi_{eq}^{2}R_0^{3}}{3M\sigma}\;}$$

**An $R^3$ law** — that is the sharpest signature to look for in $R(t)$, and it fixes the run
length. For the planned baseline ($R_0=0.25$, double well, $\epsilon=0.04$, $\gamma=3.2\times10^{-3}$,
$\sigma=0.0533$, $M=1$): $t_{evap}\approx0.20$, i.e. $\sim\!100$ steps at $dt=2\times10^{-3}$.
Drops above $R_c$ approach $R_\infty$ exponentially and need $\sim\!10\times$ longer.

$t_{evap}\propto1/M$ is the whole of the mobility lever: with a degenerate $M$ the effective $M$
seen by the bulk is the *face* value, which is why §4's averaging rule sets the prefactor.

## 5. Falsifiable predictions for the runs

| # | prediction | how to measure |
|---|---|---|
| P1 | drop survives iff $R_0>R_c=(0.5053\,\Lambda\epsilon V)^{1/4}$ | sweep $R_0$, watch survival |
| P2 | $R_\infty/R_0$ = larger root of $r(1-r^3)=3\Lambda\epsilon V/(4\pi R_0^4)$ | measure $R_\infty$ |
| P3 | $R_c\propto(\epsilon V)^{1/4}$ | sweep $\gamma$ and $L$ |
| P4 | $R_c(\omega{=}3)/R_c(\mathrm{DW})=0.938$ at matched $\epsilon$ | sweep $\omega$ with $\gamma=\epsilon^2f''(\omega)$ |
| P5 | $\max\phi-\phi_{eq}\simeq(d-1)\Lambda\phi_{eq}\epsilon/R(t)$ throughout | log $\max\phi$ and $R(t)$ |
| P6 | degenerate mobility leaves $R_c$ unchanged but $t_{sh}\to\infty$ | same sweep, two mobilities |
| P7 | harmonic averaging suppresses the bulk flux by $\sim M_{min}/M_{max}$ vs arithmetic | compare $dM_{drop}/dt$ |
| P8 | $R^3(t)=R_0^3-\tfrac{3M\sigma}{2\phi_{eq}^2}t$ while $R\gg R_\infty$ | fit $R^3$ vs $t$ |

Caveat on P1–P4: the derivation assumes a *closed* box ($\partial_n\phi=\partial_n\psi=0$). Our
`-1` boundary code is **not** that — it sets the ghost to minus the interior, i.e. $\phi=\psi=0$ at
the wall, which both pins an artificial interface at the wall and **leaks mass**
($\psi$-flux $=-2M\psi_i/h^2\ne0$). Dirichlet runs therefore test something different and must be
compared against Neumann, not used as the baseline. See the roadmap.
