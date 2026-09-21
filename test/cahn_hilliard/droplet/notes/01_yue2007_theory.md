# Yue, Zhou & Feng (2007), "Spontaneous shrinkage of drops and mass conservation in phase-field simulations", JCP 223, 1–9

Source: `/Users/alexey/Zotero/storage/4TBFTKHB/Yue и др. - 2007 - ...pdf`. Short note (9 pages, no appendix). Full paper read end to end; there is no additional content beyond what's transcribed below (references list is standard, not reproduced).

## 0. Scope and what this paper is / is not

This is a *quiescent-drop* analysis: a single circular (2D) or spherical (3D) drop sitting in a finite computational domain, **no imposed flow**, standard Cahn–Hilliard (not "mass-conserving" or Lagrange-multiplier-modified CH). They explicitly restrict to this case "for ease of discussion and without loss of generality," then argue the conclusions carry over qualitatively to flow situations (and verify this with one flow example, see §5 below).

Important terminology mismatch versus what you might expect from other Yue/Feng papers: **this note does not define a Peclet number or a symbol "S" for a diffusion/mobility parameter.** The symbol $S$ that appears here (Eq. 5–6) is simply *the perimeter of the drop*, $S = 2\pi r_0$, not a dimensionless diffusive parameter. The only transport parameter is the dimensional CH mobility $\gamma$, discussed via scaling relations in §5 below. If your other reference material defines a Peclet number $S$, it does not come from this paper — flag that explicitly rather than conflating the two.

## 1. Governing equations and definitions

**Cahn–Hilliard equation** (Eq. 1):
$$\frac{\partial \phi}{\partial t} + \mathbf v\cdot\nabla\phi = \gamma\,\Delta\mu,$$
where
- $\phi$ = phase-field / order parameter (bulk values $\pm 1$),
- $\mathbf v$ = flow velocity,
- $\gamma$ = **mobility parameter** (dimensional; not nondimensionalized in this paper),
- $\mu = \delta\mathscr F/\delta\phi$ = chemical potential, the variational (functional) derivative of the mixing energy $\mathscr F$ with respect to $\phi$.

Boundary conditions used throughout: $\mathbf n\cdot\mathbf v|_{\partial\Omega}=0$ (no penetration) and $\mathbf n\cdot\nabla\mu|_{\partial\Omega}=0$ (no diffusive flux through the domain boundary), $\mathbf n$ = outward normal of $\partial\Omega$.

**Global mass conservation** (Eq. 2), obtained by integrating Eq. (1) over $\Omega$ and using the BCs above:
$$\frac{d}{dt}\int_\Omega \phi\, d\Omega = 0.$$
This is *exact* for the continuous CH system — total $\int\phi$ is conserved identically. The paper's entire point is that this global conservation does **not** imply conservation of the volume of an individual drop (i.e., of $|\{\phi>0\}|$), because a finite interfacial thickness lets $\phi$ redistribute between "shrinking the $\phi=0$ level set" and "raising bulk $\phi$ slightly above/below $\pm1$" while leaving $\int\phi\,d\Omega$ exactly unchanged.

**Mixing (free) energy — Ginzburg–Landau functional** (Eq. 3):
$$\mathscr F = \int_\Omega \lambda\Big[\tfrac12|\nabla\phi|^2 + f(\phi)\Big]\,d\Omega,$$
with
- $\lambda$ = mixing energy density,
- $f(\phi) = \dfrac{(\phi^2-1)^2}{4\epsilon^2}$ = double-well bulk potential,
- $\epsilon$ = **capillary width**, indicative of interfacial thickness (this is the paper's definition/notation for what elsewhere is often called the interface-thickness parameter).

**1D planar equilibrium** (unbounded domain, energy minimization): $\phi(x) = \tanh\!\left(\dfrac{x}{\sqrt2\,\epsilon}\right)$ across an interface located at $x=0$. Taking $-0.9<\phi<0.9$ as "the extent of the interface," this profile gives an **interfacial thickness of $4.164\,\epsilon$**.

**Energy law** (Eq. 4), from multiplying Eq. (1) by $\mu$ and integrating (using the same natural BCs):
$$\frac{d\mathscr F}{dt} = -\gamma\int_\Omega(\nabla\mu)^2\,d\Omega \;\le\; 0.$$
CH dynamics is a gradient flow that always decreases $\mathscr F$; this is the root cause of shrinkage — a relaxed (lower-energy) drop shape/profile has lower $\mathscr F$ than the imposed initial hyperbolic-tangent profile, even at the cost of volume.

**Interfacial tension** (sharp-interface asymptotic relation used throughout): $\sigma = \dfrac{2\sqrt2}{3}\dfrac{\lambda}{\epsilon}$.

**Cahn number.** Defined in the paper (§2.1, just after Eq. 13) as
$$Cn = \frac{\epsilon}{r_0},$$
i.e. capillary width over the **initial drop radius** $r_0$ — *not* over a domain length $L$. (In §2.3 they separately reference Fabbri & Voller's rule using $\epsilon$ relative to *domain size*; that is a different, related but not identical, ratio — see §5.)

**Drop-radius / volume notation:**
- $r_0$ = initial drop radius,
- $r$ = instantaneous/shrunk drop radius,
- $\delta r = r - r_0$ = radius perturbation (Section 2.1, small-perturbation regime),
- $V$ = volume (area, in 2D) of the *entire computational domain* $\Omega$,
- $V_d$ = volume of the drop: $V_d=\pi r_0^2$ in 2D, $V_d = \frac43\pi r_0^3$ in 3D,
- $\delta\phi$ = uniform shift of the bulk value of $\phi$ away from $\pm1$ inside/outside the drop.

## 2. Physical mechanism of spontaneous shrinkage

Consider a circular/spherical drop in a *finite* domain, initialized with $\phi=+1$ strictly inside, $\phi=-1$ strictly outside, and a 1D hyperbolic-tangent profile imposed radially across the interface (Fig. 1 in the paper). For this initial condition the bulk energy contribution $\mathscr F_2=0$ exactly (since $f(\pm1)=0$), and all the energy sits in the interfacial term $\mathscr F_1=\sigma S$ (perimeter $\times$ tension).

The key geometric fact: this hyperbolic-tangent-with-exact-bulk-values profile is the true energy minimizer *only* for a planar interface in an **unbounded** domain, where the bulk volume is infinite and thus contributes zero energy density regardless of $\phi$'s exact value there. In a **finite** domain, the total free energy can instead be lowered by simultaneously (a) shrinking the drop radius (reducing interfacial length/area, hence reducing $\mathscr F_1$) and (b) shifting the bulk $\phi$ slightly away from $\pm1$ inside and outside (which raises $\mathscr F_2$ from zero, since $f(\phi)>0$ for $\phi\ne\pm1$) — and mass conservation (Eq. 2) *forces* this coupling: any change in interfacial position/drop volume must be compensated by an opposite bulk $\phi$ shift so that the total $\int\phi\,d\Omega$ stays fixed.

This is exactly the mechanism the user described: reducing interfacial area (lowering $\mathscr F_1$, a linear-in-$\delta r$ gain) is energetically cheaper than the bulk energy penalty (a quadratic-in-$\delta\phi\propto\delta r$ cost, i.e. Gibbs–Thomson-like curvature stiffness), so CH dynamics spontaneously trades drop volume for lower total energy — the drop's chemical potential is elevated by curvature (surface-tension/Laplace-pressure analogue: $\mu \sim \sigma/r_0$, used later in §5 for timing estimates) which drives outward diffusive flux $-\gamma\nabla\mu$, i.e. loses mass to the bulk. The paper explicitly frames this as "reducing the interfacial energy at the expense of raising the bulk energy, which is perfectly permissible within the Cahn–Hilliard framework but would violate mass conservation for the drop in the physical context." They also note this mechanism is independent of the specific tanh initial condition — any relaxation of an imposed interfacial profile (e.g., in shear-driven drop deformation, Yue et al. refs [5,9]) triggers the same interfacial/bulk energy exchange, so it is "a fundamental mechanism inherent to the Cahn–Hilliard dynamics," also previously noted by Jacqmin [1].

## 3. Variational (perturbation) derivation — small-$\delta r$ regime (Section 2.1)

This is exactly the Gateaux/first-variation argument the user expected, done for a 2D circular drop (3D spherical result given by direct analogy).

**Ansatz:** the drop radius perturbs by $\delta r$ (drop shrinks radially: $\delta r<0$ is the shrinking direction), and — enforced by the constraint that total $\int_\Omega\phi\,d\Omega$ is conserved (Eq. 2) — the bulk value of $\phi$ shifts by a **uniform** amount $\delta\phi$ in both bulk phases (interior and exterior), assuming the interface stays thin, $\epsilon\ll r_0$. Uniformity of $\delta\phi$ follows because $\phi$ is spatially uniform in each bulk phase and $\mu=\lambda(\phi^2-1)\phi/\epsilon^2$; spatial uniformity of $\mu$ in equilibrium requires that the small shifts from $\phi=\pm1$ be equal in the two bulk phases.

**Mass-conservation constraint** (Eq. 5), leading order:
$$\delta\phi \approx -\frac{2S\,\delta r}{V} = -\frac{4\pi r_0\,\delta r}{V},$$
with $S=2\pi r_0$ the drop perimeter and $V$ the domain volume.

**Interfacial-energy change** (Eq. 6), from the change in interfacial length, neglecting the $O(\delta\phi^2)$ variation of $\sigma$ itself:
$$\delta\mathscr F_1 \approx \sigma\,\delta S = 2\pi\sigma\,\delta r.$$

**Bulk-energy change** (Eq. 7), from $\phi$ shifting away from $\pm1$ over the whole domain volume $V$:
$$\delta\mathscr F_2 \approx \int_\Omega \lambda\,\delta f\, d\Omega \approx \lambda\,\frac{\delta\phi^2}{\epsilon^2}\,V.$$

**Total energy variation** (Eq. 8), substituting Eq. (5) into Eq. (7) and adding Eq. (6):
$$\delta\mathscr F \approx 2\pi\sigma\,\delta r \;+\; \lambda\,\frac{(4\pi r_0)^2}{V\epsilon^2}\,\delta r^2.$$
(Only leading-order terms are kept throughout.) The paper remarks that $\lambda/\epsilon\sim\sigma=O(1)$ and that $\delta r$ turns out to be small (consistent with $\delta r=O(\epsilon)$ from Eq. 9 below — *note: the printed sentence in the source connecting these two orders is compressed/ambiguous in the scanned text; the conclusion that the quadratic bulk term $\delta\mathscr F_2$ ends up the same asymptotic order as the linear interfacial term $\delta\mathscr F_1$ is unambiguous from the equations themselves and from what follows*).

**Sign / energy-balance argument — why the drop shrinks:**
$$\left.\frac{\partial(\delta\mathscr F)}{\partial(\delta r)}\right|_{\delta r=0} = 2\pi\sigma > 0.$$
Since this slope at $\delta r=0$ is strictly positive, decreasing $\delta r$ (i.e. $\delta r<0$, drop **shrinking**) *lowers* the total energy — this is the exact statement of "shrinking is energetically favorable," confirming the qualitative mechanism of §2. It is also consistent with $\delta\mathscr F_2 \sim O(\delta r)$ (not just $O(\delta r^2)$ at leading order once Eq. 5's coupling is substituted) being the *same order* as the interfacial-energy gain $O(\delta r)$, despite $\mathscr F_2$'s naively quadratic form in $\delta\phi$.

**Critical-point condition — the equilibrium/quantitative shrinkage amount.** Setting $\partial(\delta\mathscr F)/\partial(\delta r)=0$ gives the state of *lowest* energy, i.e. the final (equilibrium) shrinkage:
$$\boxed{\delta r = -\frac{\sigma V}{16\pi\lambda}\left(\frac{\epsilon}{r_0}\right)^2 = -\frac{\sqrt2\,V}{24\pi}\,\frac{\epsilon}{r_0^2}} \qquad (\text{2D, Eq. 9})$$
or in dimensionless form (Eq. 10):
$$\frac{\delta r}{r_0} = -\frac{\sqrt2}{24}\left(\frac{V}{V_d}\right)\left(\frac{\epsilon}{r_0}\right), \qquad V_d=\pi r_0^2.$$
Substituting Eq. (9) back into the constraint Eq. (5) gives the corresponding bulk shift (Eq. 11):
$$\delta\phi = \frac{\sqrt2}{6}\,\frac{\epsilon}{r_0}.$$

**3D spherical drop**, by the analogous derivation (Eqs. 12–13):
$$\left(\frac{\delta r}{r_0}\right) = -\frac{\sqrt2}{18}\left(\frac{V}{V_d}\right)\left(\frac{\epsilon}{r_0}\right), \qquad V_d=\tfrac43\pi r_0^3,$$
$$\delta\phi = \frac{\sqrt2}{3}\,\frac{\epsilon}{r_0}.$$

**Validity / regime of applicability.** These formulas require $\delta r/r_0\ll1$ (needed to keep only leading-order terms) — this is stated to be *more restrictive* than either $Cn=\epsilon/r_0\ll1$ alone or $V/V_d>1$ alone; both smallness conditions are simultaneously needed. Sign convention: results assume $\phi=+1$ inside the drop, $\phi=-1$ outside; if reversed, $\delta\phi$ in Eqs. (11)/(13) flips sign, while $\delta r$ in Eqs. (10)/(12) is unaffected.

**Critical condition for "drop shrinks" in this small-perturbation regime:** the paper does not phrase it as an inequality on parameters here (that comes in Section 3, see §4 below) — in this small-$\delta r$ regime the drop *always* shrinks (the linear term's positive slope guarantees it), the only question addressed here is *by how much* (Eqs. 9/12) before it settles at a new equilibrium radius $r_0+\delta r$. The question of *whether the drop vanishes completely* is a separate, large-perturbation ($\delta r\sim r_0$) analysis, given next.

## 4. Critical drop radius — large-perturbation ("Ostwald-ripening-like") analysis (Section 3)

The small-perturbation result (Eqs. 10, 12) hints that a drop could disappear entirely if $(V/V_d)(\epsilon/r_0)$ is large enough, but the paper states this is **inconclusive** because Eqs. (9)–(13) hold only for $\delta r\ll r_0$. They therefore redo the energy analysis allowing the shrunk radius $r$ to differ from $r_0$ by an amount that is *not* small (only assuming $\epsilon\ll r_0$, i.e. $Cn\ll1$).

**Setup:** 2D drop, initial radius $r_0$ ($\phi=1$ inside, $-1$ outside), shrinks to radius $r$; again a uniform bulk shift $\delta\phi$ inside and outside (but now $r_0^2-r^2$ is finite, not infinitesimal).

**Exact mass constraint** (Eq. 14):
$$\delta\phi = \frac{2\pi(r_0^2-r^2)}{V}.$$
A posteriori (using Eq. 17 below) one confirms $\delta\phi \sim r_0^2/V \sim (V\epsilon)^{2/3}/V = \epsilon^{2/3}/V^{1/3} \ll 1$, justifying dropping cubic/quartic terms in $\delta\phi$ in the energy expansion.

**Total free energy as a function of the (finite) shrunk radius $r$** (Eq. 15), substituting the double-well potential evaluated at the shifted bulk values $1+\delta\phi$ (drop interior, area $\pi r^2$) and $-1+\delta\phi$ (exterior, area $V-\pi r^2$), plus the interfacial term $2\pi r\sigma$ (using the asymptotic $\sigma=2\sqrt2\lambda/3\epsilon$, valid since $r\gg\epsilon$ is assumed for this part of the curve):
$$\mathscr F(r) \approx 2\pi r\sigma + \lambda\frac{[(1+\delta\phi)^2-1]^2}{4\epsilon^2}(\pi r^2) + \lambda\frac{[(-1+\delta\phi)^2-1]^2}{4\epsilon^2}(V-\pi r^2)$$
$$\approx \frac{4\sqrt2\pi}{3}\frac{\lambda}{\epsilon}\,r \;+\; 4\pi^2\frac{\lambda}{V\epsilon}\left(r_0^4 - 2r_0^2 r^2 + r^4\right). \qquad (\text{Eq. 15})$$
(Note: although a fully vanished drop physically has $r\sim\epsilon$, the interesting part of this curve — where the qualitative shrink/no-shrink decision is made — is at $r=O(r_0)$.)

**Derivative** (Eq. 16):
$$\frac{\partial\mathscr F}{\partial r} = \frac{4\sqrt2\pi}{3}\frac{\lambda}{\epsilon} + \frac{4\pi^2\lambda}{V\epsilon}\left(4r^3-4r_0^2 r\right).$$
Evaluating at both endpoints gives the **same** value, $\left.\partial\mathscr F/\partial r\right|_{r=0} = \left.\partial\mathscr F/\partial r\right|_{r=r_0} = \dfrac{4\sqrt2\pi}{3}\dfrac{\lambda}{\epsilon} > 0$ (the bracket term vanishes at both $r=0$ and $r=r_0$).

**Shape of $\mathscr F(r)$ (Fig. 4 schematic):** since the slope is positive and equal at both ends of $[0,r_0]$, the curve must have an **inflexion point** at some $r_i\in[0,r_0]$ (found from $\partial^2\mathscr F/\partial r^2=0$):
$$r_i = \frac{\sqrt3}{3}\,r_0 \quad\text{(2D)}.$$
- If $\partial\mathscr F/\partial r\big|_{r=r_i} < 0$, then $\mathscr F(r)$ develops a **local potential well** at some $r_w>r_i$: the shrinking drop gets *trapped* at $r_w$ (a metastable, nonzero equilibrium radius) and will **not** fully vanish, even if $\mathscr F(r{=}0) < \mathscr F(r{=}r_0)$ (i.e., even if disappearance is globally favorable, there's a local energy barrier preventing it kinetically/via this gradient-flow-style argument — see Fig. 4's schematic, dashed vs solid curves).
- **A vanishing drop therefore requires** $\partial\mathscr F/\partial r\big|_{r=r_i}\ge 0$ (no potential well — monotone descent to $r=0$ is possible).

**Resulting critical-radius condition** — this is the exact "vanish" criterion (Eq. 17, 2D):
$$\boxed{r_0 \le r_c = \left(\frac{\sqrt6}{8\pi}\,V\epsilon\right)^{1/3}} \qquad \text{(drops with } r_0<r_c \text{ eventually disappear entirely).}$$

**3D spherical case**, by the analogous derivation:
$$r_i = \frac{r_0}{2^{2/3}}, \qquad r_c = \left(\frac{2^{1/6}}{3\pi}\,V\epsilon\right)^{1/4}. \qquad (\text{Eq. 18})$$

Both formulas were verified against finite-element computations (Fig. 5): "the range of $V$ is achieved by using $L=1,2,4,8$ in the computational domains"; numerically, discrete $r_0$ values were tested to bracket $r_c$ to within 3%.

**Scaling remarks on $r_c$:** $r_c \propto (V\epsilon)^{1/3}$ in 2D and $\propto(V\epsilon)^{1/4}$ in 3D — i.e. $r_c$ is **largely determined by domain size $V$** and depends only **weakly** on the capillary width $\epsilon$ (cube-root / fourth-root dependence). Consequence stated explicitly: "Once the domain size is chosen, there is little room for raising $r_c$ by reducing the capillary width $\epsilon$."

## 5. Numerical verification, guidelines, and remedies (Sections 2.2–2.3 and 3)

### Numerical setups used to verify Eqs. (9)–(18)
- **2D spectral (Fourier–Chebyshev Galerkin):** rectangular box $2\pi\times2$, periodic in the horizontal direction. Drops with initial radius $r_0 = 0.25,\,0.4,\,0.5,\,0.52,\,0.6,\,0.7,\,0.8,\,0.9$; capillary width $\epsilon=0.01$ and $0.02$.
- **Finite-element (2D and axisymmetric 3D), adaptive meshing:** quarter-domain with symmetry BCs on both axes, $L=2$ (Fig. 2), giving $V=16$ (2D) and $V=16\pi$ (3D). Initial drop radius fixed at $r_0=1$; capillary widths tested $\epsilon = 0.005,\,0.01,\,0.02,\,0.05,\,0.1,\,0.2$. $\phi=+1$ inside, $-1$ outside initially. Mesh size at the interface kept at roughly $\epsilon/2$ to guarantee numerical accuracy.
- **Critical-radius sweep (Fig. 5):** domain sizes via $L=1,2,4,8$; both 2D ($\epsilon=0.01,0.02$) and 3D ($\epsilon=0.01,0.02$) finite-element results plotted against theory over $V\epsilon \in [10^{-2},10^{2}]$, with $r_c$ ranging roughly $10^{-1}$ to a few units on the plot.

**No explicit numeric table is printed in the paper** — Figs. 3 and 5 are log–log comparison plots (numerics vs. the boxed theoretical formulas above), not tabulated values, so I cannot transcribe exact numbers beyond the parameter lists above and the qualitative agreement statement. Specifically stated quantitative facts from the text (not read off a plot):
- At $Cn=0.05$, the spatial variation of $\delta\phi$ over the 2D computational domain is below **3.1%** — i.e. $\delta\phi$ is numerically defined/measured as $\delta\phi \equiv \tfrac12[(\phi_{\max}-1)+(\phi_{\min}+1)]$, and this measured quantity matches the uniform-shift assumption well at small $Cn$, with growing deviation as $Cn$ increases (theory loses accuracy toward the upper end of tested $Cn$).
- Agreement between numerical and theoretical $\delta\phi$ and $-\delta r/r_0$ (Fig. 3) is described as "excellent" for small $Cn$.

### Guidelines for conserving mass (Section 2.3)
1. **Use a small Cahn number, $Cn=\epsilon/r_0 \ll 1$** — i.e. $\epsilon$ much smaller than drop size.
   - Cited external guideline: Fabbri & Voller [13] (1D solidification calculations) suggest $\epsilon$ should be smaller than **0.0025 of the domain size**.
   - The authors' own experience/recommendation: **$Cn \lesssim 0.01$ is generally sufficient.**
2. **Avoid using very large computational domains relative to the size of the dispersed phase** ($V/V_d$ should not be too large) — if $V/V_d$ is large, a drop may shrink considerably even for small $Cn$, and drops with $r_0$ below the critical size $r_c$ (Section 3) will disappear altogether.
3. These guidelines were derived for the no-external-flow case but are stated to apply generally to flow situations as well.

### Two further comments on $Cn\ll1$ (Section 2.3, end)
- **Sharp-interface limit:** $Cn\ll1$ is also the condition for the diffuse interface to approximate the sharp interface. The error in interfacial tension is $O(Cn^2)$, and the error in mixture properties (density, viscosity) is $O(Cn^2)$. So the mass-conservation requirement *partially overlaps* with the sharp-interface-approximation requirement — but thinner interfaces are more computationally costly to resolve (motivating adaptive meshing, ref [6]).
- **Cn should be understood more generally** as the ratio of $\epsilon$ to the *smallest length scale of interest*. In singular events (interfacial rupture, coalescence) the local radius of curvature approaches zero, so effectively $Cn\to1$ no matter how small $\epsilon$ is — the diffuse-interface method's convergence to the sharp-interface limit necessarily breaks down there. However, such singular events are dominated by short-range forces that the sharp-interface model can't represent either (it has no regularization), whereas the phase-field model contains a phenomenological "short-range force" resembling the van der Waals force [14] — in this sense the phase-field method is *superior* for treating such singular events despite Cn formally being large there.

### Mobility scaling and shrinkage-time estimate (Section 3, second half) — the key mitigation via $\gamma$
This is the paper's main practical remedy beyond "shrink $\epsilon$/domain": even if $r_0<r_c$ (drop is doomed to eventually vanish), shrinkage can be made so slow that it's irrelevant on the simulation's timescale of interest.

- CH diffusive flux: $\mathbf F = \gamma\nabla\mu$.
- Chemical potential estimated as the phase-field analogue of interfacial-tension-times-curvature (Jacqmin [1]): $\mu\sim\sigma/r_0$.
- Length scale $l$ over which $\mu$ varies is much larger than $\epsilon$ (estimated by scaling arguments, ref [15] Briant & Yeomans, but ultimately fixed from numerical data): $\mathbf F\sim\gamma\sigma/(r_0 l)$.
- Equating the "mass" (radius) lost, $\Delta r = F\,t_{sh}$, gives the **shrinkage-time estimate** (Eq. 19):
$$t_{sh} \sim \frac{r_0\,\Delta r\, l}{\gamma\sigma}.$$
- Comparing this scaling with their numerical data (across the few $\epsilon$ values tested) shows $l/\epsilon \sim O(100)$, giving (Eq. 20):
$$t_{sh} \approx 100\,\frac{r_0\,\Delta r\,\epsilon}{\gamma\sigma}.$$
- **Mobility scaling choice:** Jacqmin suggested $\gamma\sim\epsilon^n$ with $n$ between 1 and 2 (comparing diffusive vs. convective fluxes). For their own flow-induced drop-deformation simulations, they found
$$\gamma \sim \frac{G\,r_0\,\epsilon^2}{\sigma}$$
to be a good choice, where $G$ is the characteristic strain rate of the flow. Substituting this $\gamma$ into Eq. (20) gives (Eq. 21):
$$t_{sh} \approx 100\,\frac{\Delta r}{r_0}\,\frac{r_0}{\epsilon}\,\frac1G = 100\,\frac{\Delta r}{r_0}\,Cn^{-1}\,G^{-1}.$$
- **Conclusion:** the time needed for the drop to shrink by a significant fraction ($\Delta r/r_0=O(1)$) is **at least two orders of magnitude longer than the flow time $t_f=G^{-1}$**. The smaller $Cn$, the greater the disparity between $t_{sh}$ and $t_f$ — i.e., with a small enough $Cn$ and a judicious mobility $\gamma$, drop shrinkage can be kept under control even for drops with $r_0<r_c$.

**Worked example (elongational-flow validation, referencing their FE work [6]):** axisymmetric domain $L=10$; $r_0=1$, $\epsilon=0.01$, $G=1$, $\sigma=10$. Chosen $\gamma = 2.0\times10^{-5} \sim Gr_0\epsilon^2/\sigma$. From Eq. (18), $r_c=1.65>r_0=1$, so this drop is formally "subject to disappearance" by the critical-radius criterion. A drop of equal viscosity to the external fluid reaches a steady spheroidal shape at flow time $t\approx5$. Eq. (20) gives $t_{sh}\approx5000 \gg t_f$. At $t=5$, drop volume had decreased by only **1.4%** (equivalent to an effective-radius decrease of **0.5%**) — i.e. mass was conserved to a great degree of accuracy despite simulating a drop nominally smaller than the critical radius, and the numerical result agreed well with the sharp-interface benchmark.

### Remedies / mitigations — explicit list as stated in the paper
The paper does **not** discuss Lagrange-multiplier mass-correction schemes or "mass-conserving CH" variants at all (no such remedy is mentioned or evaluated) — the sole remedies proposed are:
1. Small Cahn number $Cn=\epsilon/r_0\ll1$ (target $Cn\lesssim0.01$; or $\epsilon<0.0025\times$domain size per Fabbri & Voller).
2. Keep the domain-to-drop volume ratio $V/V_d$ modest (don't over-resolve a small drop in a huge domain).
3. Choose the mobility $\gamma$ so that the shrinkage timescale $t_{sh}$ (Eq. 19–21) is much longer than the physical/simulation timescale of interest $t_f$ — specifically the scaling $\gamma\sim Gr_0\epsilon^2/\sigma$ (with $G$ a characteristic strain rate) worked well for their flow problems; more generally Jacqmin's $\gamma\sim\epsilon^n$, $n\in[1,2]$.
4. Adaptive meshing (ref. [6]) is mentioned as "an attractive strategy for reaching down to smaller values of $Cn$" computationally (i.e., it's an *enabler* for guideline 1, not a separate remedy in itself).

They explicitly do **not** claim these guidelines eliminate the critical radius $r_c$ — $r_c$ is a hard threshold set essentially by domain size (Eqs. 17–18); the mitigation is entirely about making the *approach* to $r_c$ slow enough (via $\gamma$) to be irrelevant within the run's duration, not preventing eventual vanishing of very small drops in principle.

## 6. What this predicts for our runs — dimensionless groups to sweep

Per the paper's own analysis, the following dimensionless (or paper-defined) groups control mass loss / shrinkage, and should be varied independently in a parameter study:

1. **$Cn = \epsilon/r_0$** (their definition — capillary width over drop radius, not domain length). Controls the leading-order shift $\delta\phi$ and relative shrinkage $\delta r/r_0$ linearly (Eqs. 10–13): $\delta\phi\propto Cn$, $\delta r/r_0 \propto (V/V_d)\,Cn$. Target $Cn\lesssim0.01$ for negligible mass loss in the small-perturbation regime.
2. **$V/V_d$** — domain volume over drop volume. Appears multiplicatively with $Cn$ in the small-perturbation shrinkage formula (Eqs. 10, 12), and sets the critical radius via $r_c\propto (V\epsilon)^{1/3}$ (2D) or $(V\epsilon)^{1/4}$ (3D) — note $r_c$ depends on the *combination* $V\epsilon$, not $Cn$ directly, and only weakly (cube/fourth-root) on $\epsilon$ once $V$ is fixed.
3. **$r_0/r_c$** (equivalently, whether $r_0 \lessgtr r_c$ from Eqs. 17–18) — the binary "will this drop eventually fully vanish" criterion, independent of how slowly it does so.
4. **Mobility $\gamma$** (dimensional, or its dimensionless combination $\gamma\sigma/(r_0 l)$ implicit in Eq. 19) — sets the *rate* $t_{sh}$ at which shrinkage/vanishing proceeds; the ratio of interest is $t_{sh}/t_f$ (shrinkage time vs. simulation/flow time of interest), which per Eq. (21) scales like $Cn^{-1}G^{-1}$ once $\gamma\sim Gr_0\epsilon^2/\sigma$ is adopted — so for a quiescent-drop study (no imposed $G$), $t_{sh}$ should be computed directly from Eq. (19)/(20) using whatever mobility law is actually used in the solver, and compared against the total simulated time.
5. **$l/\epsilon$** — the empirical ratio (found $\approx100$ in their tests) between the chemical-potential decay length $l$ and the capillary width $\epsilon$; this entered as a fitted constant in Eq. (20) rather than a controlled/derived parameter, so if reproducing their timing estimate, this ratio may need to be recalibrated for a different solver/mobility model rather than assumed universal.
6. **Sign/orientation of $\phi$** (which phase is $+1$) — flips the sign of $\delta\phi$ but not of $\delta r$; worth confirming this convention matches our solver's initialization before comparing numbers directly.

Practical sweep design implication: to test the shrinkage/critical-radius theory cleanly, vary $r_0$ at fixed $\epsilon,V$ to locate $r_c$ empirically and compare to Eq. (17)/(18); separately vary $\epsilon$ at fixed $r_0,V$ to confirm the weak $r_c\propto\epsilon^{1/3\text{ or }1/4}$ scaling and the linear $Cn$ scaling of $\delta\phi,\delta r$ in the small-perturbation regime; and separately vary the mobility $\gamma$ (holding geometry fixed) to test the $t_{sh}\propto\gamma^{-1}$ scaling of Eq. (19)/(20).
