# Stationary spherical droplet in the Cahn–Hilliard equation: analytical results

This document collects the analytical results for a stationary, spherically symmetric droplet described by the Cahn–Hilliard (CH) equation in a closed ball. It is intended as a reference for verifying a numerical solver. All quantities are dimensionless.

---

## 1. Problem statement

### 1.1 Equation

$$
\frac{\partial\phi}{\partial t} = D\,\nabla^2\psi, \qquad \psi = f'(\phi) - \gamma\,\nabla^2\phi, \qquad f(\phi) = \frac14\left(\phi^2 - 1\right)^2 .
$$

- $\phi$ is the order parameter; the two bulk phases correspond to $\phi \approx \pm 1$.
- $\psi$ is the chemical potential.
- $f'(\phi) = \phi^3 - \phi$, $f''(\phi) = 3\phi^2 - 1$, $f''(\pm 1) = 2$.

The CH equation is the $H^{-1}$ gradient flow of the free energy

$$
F[\phi] = \int_V\left(f(\phi) + \frac{\gamma}{2}|\nabla\phi|^2\right)dV, \qquad \psi = \frac{\delta F}{\delta \phi},
$$

so that, with the boundary conditions below,

$$
\frac{dF}{dt} = -D\int_V |\nabla\psi|^2\,dV \le 0, \qquad \frac{d}{dt}\int_V \phi\,dV = 0 .
$$

### 1.2 Geometry and boundary conditions

- Domain: ball of radius 1.
- Spherical symmetry: $\phi = \phi(r)$, $\psi = \psi(r)$.
- Radial Laplacian: $\nabla^2 u = u'' + \frac{2}{r}u' = \frac{1}{r^2}\left(r^2 u'\right)'$.
- Regularity at the centre: $\phi'(0) = \psi'(0) = 0$.
- Neumann at the wall: $\phi'(1) = 0$ (neutral wetting, 90° contact angle), $\psi'(1) = 0$ (no flux).
- At $r = 0$ the term $\frac{2}{r}\phi'$ is resolved by L'Hôpital: $\nabla^2\phi(0) = 3\phi''(0)$.

### 1.3 Mass conservation

Integrating the CH equation over the ball and using $\psi'(1) = 0$ shows that $\int_V\phi\,dV$ is conserved. The mean value is set by the initial condition:

$$
\bar\phi = \frac{1}{V}\int_V \phi\,dV = 3\int_0^1 \phi(r)\,r^2\,dr, \qquad V = \frac{4\pi}{3}.
$$

The factor 3 is $4\pi / V$.

### 1.4 Stationary problem

In a stationary state $(r^2\psi')' = 0$, so $\psi = A - B/r$; regularity gives $B = 0$, hence $\psi = \psi_0 = \text{const}$. The problem becomes

$$
\gamma\left(\phi'' + \frac{2}{r}\phi'\right) = \phi^3 - \phi - \psi_0, \qquad \phi'(0) = \phi'(1) = 0, \qquad 3\int_0^1\phi\,r^2\,dr = \bar\phi .
$$

The constant $\psi_0$ is a Lagrange multiplier fixed by the mass constraint. There is no closed-form solution (the $2\phi'/r$ term destroys the first integral), so the solution is constructed asymptotically in the thin-interface limit. We look for a droplet of phase $+1$ in a matrix of phase $-1$.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $\varepsilon = \sqrt{\gamma}$ | small expansion parameter |
| $\xi = \sqrt{2\gamma}$ | interface width |
| $R$ | droplet radius, defined by $\phi(R) = 0$ |
| $z = (r - R)/\varepsilon$ | stretched inner coordinate |
| $\zeta = (r - R)/\xi = z/\sqrt2$ | inner coordinate normalised by the interface width |
| $\sigma = \frac{2\sqrt2}{3}\sqrt{\gamma}$ | surface tension of a flat interface |
| $\Delta\phi = 2$ | jump of $\phi$ between the bulk phases |
| $\bar\phi$ | mean value of $\phi$ over the ball (from the initial condition) |
| $m = 1 + \bar\phi$ | excess of material over the pure phase $-1$ |
| $\delta$ | common shift of both bulk phases from $\pm 1$ |
| $\psi_0$ | constant chemical potential of the stationary state |
| $L = \frac{d^2}{dz^2} - f''(\Phi_0)$ | linearised inner operator; its kernel is $\Phi_0' \propto 1 - \Phi_0^2$ |
| $R_c$ | critical nucleus radius (unstable stationary droplet) |
| $R_*$ | stable stationary droplet radius |

Surface tension follows from the 1D profile: $\sigma = \int \gamma\,\phi'^2\,dr = \int_{-1}^{1}\sqrt{2\gamma f(\phi)}\,d\phi = \frac{2\sqrt2}{3}\sqrt\gamma$.

---

## 3. Thin-interface asymptotic solution

Assumptions: $\varepsilon \ll R$ and $\varepsilon \ll 1 - R$ (the interface touches neither the centre nor the wall). Expansions:

$$
\Phi = \Phi_0 + \varepsilon\Phi_1 + \varepsilon^2\Phi_2 + \dots, \qquad \psi_0 = \psi_{00} + \varepsilon\psi_1 + \varepsilon^2\psi_2 + \dots
$$

### 3.1 Outer solution (bulk phases)

Away from the interface the gradient term is negligible and $f'(\phi) = \psi_0$. Linearising around the minima, $f'(\pm1 + \delta) \approx 2\delta$, gives

$$
\phi_{\text{in}} = 1 + \frac{\psi_0}{2}\ (r < R), \qquad \phi_{\text{out}} = -1 + \frac{\psi_0}{2}\ (r > R).
$$

Both phases are shifted by the same amount $\delta = \psi_0/2$ (consequence of the symmetry of $f$). At second order the shifts become asymmetric: $\delta_\pm = \frac{\psi_0}{2} \mp \frac{3}{8}\psi_0^2$.

### 3.2 Inner problem

With $\phi(r) = \Phi(z)$ the stationary equation becomes

$$
\Phi_{zz} + \frac{2\varepsilon}{R + \varepsilon z}\Phi_z = f'(\Phi) - \psi_0 .
$$

### 3.3 Zero order

$$
\Phi_0'' = f'(\Phi_0) - \psi_{00}, \qquad \Phi_0(-\infty) = \phi_+, \quad \Phi_0(+\infty) = \phi_- .
$$

A heteroclinic solution exists only if (i) $f'(\phi_\pm) = \psi_{00}$ and (ii) $\int_{\phi_-}^{\phi_+}\big(f'(\phi) - \psi_{00}\big)d\phi = 0$ (Maxwell equal-area / common-tangent construction). The function $I(\psi) = \int_{\phi_1(\psi)}^{\phi_3(\psi)}(f' - \psi)\,d\phi$ satisfies $dI/d\psi = -(\phi_3 - \phi_1) < 0$ and $I(0) = 0$ by oddness of $f'$, hence

$$
\psi_{00} = 0, \qquad \phi_\pm = \pm 1, \qquad \Phi_0(z) = -\tanh\frac{z}{\sqrt2} = -\tanh\zeta .
$$

This is exactly the planar 1D interface wrapped onto a sphere of radius $R$. It proves $\psi_0 = O(\varepsilon)$.

### 3.4 First order

$$
L\Phi_1 = -\frac{2}{R}\Phi_0' - \psi_1 .
$$

**Solvability condition.** Differentiating the zero-order equation gives $L\Phi_0' = 0$, so $\Phi_0'$ is in the kernel of the self-adjoint operator $L$. Multiplying by $\Phi_0'$ and integrating (boundary terms vanish because $\Phi_0', \Phi_0''$ decay exponentially and $\Phi_1$ is bounded):

$$
-\frac{2}{R}\int\Phi_0'^2\,dz - \psi_1\int\Phi_0'\,dz = 0, \qquad \int\Phi_0'^2\,dz = \frac{2\sqrt2}{3}, \qquad \int\Phi_0'\,dz = -2,
$$

$$
\psi_1 = \frac{2\sqrt2}{3R} \qquad\Longrightarrow\qquad \boxed{\psi_0 = \varepsilon\psi_1 = \frac{\sigma}{R}}
$$

This is the Gibbs–Thomson relation $\psi_0 = \dfrac{2\sigma}{R\,\Delta\phi}$ with $\Delta\phi = 2$.

**Explicit first-order correction.** Using $L(1) = L(\Phi_0^2) = 1 - 3\Phi_0^2$, a particular solution is the constant $\psi_1/2$. The homogeneous solutions are $y_1 = 1 - \Phi_0^2 = \text{sech}^2\zeta$ (bounded) and, from Liouville's formula ($W = \text{const}$ since there is no first-derivative term), $y_2 = y_1\int dz/y_1^2 = \frac{\sqrt2}{8}\left(3\zeta\,\text{sech}^2\zeta + 3\tanh\zeta + \sinh 2\zeta\right)$, which grows as $e^{\sqrt2|z|}$ and is discarded. The condition $\Phi_1(0) = 0$ (definition of $R$) gives

$$
\Phi_1 = \frac{\psi_1}{2}\,\Phi_0^2 = \frac{\sqrt2}{3R}\tanh^2\zeta ,
$$

which matches the outer shift $\psi_0/2$ as $\zeta \to \pm\infty$.

### 3.5 Uniformly valid profile (to first order)

$$
\boxed{\;\phi(r) \approx -\tanh\zeta + \frac{\sigma}{2R}\tanh^2\zeta, \qquad \zeta = \frac{r - R}{\sqrt{2\gamma}}, \qquad \psi_0 = \frac{\sigma}{R}\;}
$$

Limits: $\phi \to \pm 1 + \frac{\sigma}{2R}$ far from the interface; $\phi(R) = 0$ exactly. Neumann conditions are satisfied up to exponentially small terms $O(e^{-R/\xi})$ and $O(e^{-(1-R)/\xi})$.

### 3.6 Droplet radius from mass conservation

Replacing the profile by a step (corrections are $O(\gamma)$, the leading one being $\pi^2 R\gamma$):

$$
\bar\phi = 2R^3 - 1 + \frac{\psi_0}{2} \quad\Longleftrightarrow\quad h(R) \equiv 2R^4 - mR + \frac{\sigma}{2} = 0 .
$$

Perturbative solution for the large (stable) root:

$$
R_* \approx R_0 - \frac{\sigma}{6m}, \qquad R_0 = \left(\frac{m}{2}\right)^{1/3}.
$$

### 3.7 Second order (brief)

At $O(\varepsilon^2)$:

$$
L\Phi_2 = 3\Phi_0\Phi_1^2 - \frac{2}{R}\Phi_1' + \frac{2z}{R^2}\Phi_0' - \psi_2 .
$$

All three integrands in the solvability condition ($\Phi_0'\Phi_0\Phi_1^2$, $\Phi_0'\Phi_1'$, $z\Phi_0'^2$) are odd in $z$, so

$$
\psi_2 = 0, \qquad \psi_0 = \frac{\sigma}{R} + O\!\left(\gamma^{3/2}\right).
$$

Physically this is a zero Tolman length, a consequence of the $\phi \to -\phi$ symmetry of $f$. The relative error of Gibbs–Thomson is therefore $O(\gamma/R^2) = O\big((\xi/R)^2\big)$, not $O(\xi/R)$. Second order also produces an asymmetric (odd) part of the interface profile and asymmetric bulk shifts, but does not change $\psi_0$. Exponentially small terms ($e^{-R/\xi}$, $e^{-(1-R)/\xi}$) are beyond all orders of the expansion.

---

## 4. Energy analysis: existence, stability and critical radius

### 4.1 Principle

Since $dF/dt \le 0$ at fixed mass, stable stationary states are local minima of $F$ under the mass constraint; maxima and saddles are unstable.

### 4.2 Energy of a droplet of radius $R$

Consider the family of states "droplet of radius $R$ with the tanh interface, both bulk phases shifted by $\delta$".

- **Interface:** energy per unit area of the tanh profile is $\sigma$, so $F_{\text{int}} = 4\pi R^2\sigma$.
- **Bulk:** $f(\pm1 + \delta) \approx \frac12 f''(\pm1)\delta^2 = \delta^2$ in both phases, so $F_{\text{bulk}} = \frac{4\pi}{3}\delta^2$.
- **Mass:** $\bar\phi = R^3(1 + \delta) + (1 - R^3)(-1 + \delta) = 2R^3 - 1 + \delta$, hence $\delta = m - 2R^3$.

(A direct evaluation of $F[\phi]$ on this profile confirms these two terms; all neglected contributions are $O(\xi^3)$.)

Dividing by $V = 4\pi/3$:

$$
\boxed{\;E(R) = \frac{F}{V} = 3\sigma R^2 + \left(m - 2R^3\right)^2\;}
$$

$E(0) = m^2 \approx f(\bar\phi)$ is the energy of the homogeneous state. Relative to it,

$$
E(R) - E(0) = 3\sigma R^2 - 4mR^3 + 4R^6 ,
$$

where the terms are, respectively, the interface cost, the gain from condensing the excess into the new phase, and the depletion of the matrix in a finite volume.

### 4.3 Stationary points

$$
E'(R) = 6\sigma R - 12mR^2 + 24R^5 = 12R\,h(R), \qquad h(R) = 2R^4 - mR + \frac{\sigma}{2}.
$$

For $R > 0$ the sign of $E'$ equals the sign of $h$. The stationary points are the roots of $h$, i.e. exactly the radius equation of Section 3.6 (Gibbs–Thomson plus mass conservation). Equivalently, $E'(R) = 6R^2(\psi_d - \psi_m)$ with $\psi_d = \sigma/R$ at the droplet surface and $\psi_m = 2(m - 2R^3)$ in the matrix.

### 4.4 Shape of the landscape

$h(0) = \sigma/2 > 0$, $h \to +\infty$ as $R \to \infty$, $h'' = 24R^2 \ge 0$ (convex), minimum at $R_f = (m/8)^{1/3}$, where

$$
h(R_f) = \frac{\sigma}{2} - \frac{3m}{4}R_f .
$$

- If $h(R_f) > 0$: $E$ increases monotonically, the only minimum is $R = 0$; any droplet dissolves.
- If $h(R_f) < 0$: $h$ has two roots $R_c < R_f < R_*$ and the sign pattern $+,-,+$. Hence $R = 0$ is a local minimum, $R_c$ is a maximum (critical nucleus), $R_*$ is a minimum (stable droplet).

Dynamics: a droplet with $R < R_c$ collapses to $R = 0$; a droplet with $R > R_c$ relaxes to $R_*$.

Stability condition of a root (from $E'' = 12R\,h'$ at a root and $m = 2R^3 + \sigma/(2R)$): $R^4 > \sigma/12$.

### 4.5 Existence threshold

$h(R_f) < 0 \iff m^{4/3} > \frac{4\sigma}{3}$:

$$
\boxed{\;m_{\min} = \left(\frac{4\sigma}{3}\right)^{3/4} \approx 1.19\,\gamma^{3/8}, \qquad R_f = \left(\frac{\sigma}{12}\right)^{1/4}\;}
$$

At $m = m_{\min}$ the two roots merge at $R_f$ (saddle-node bifurcation); $R_f$ is the smallest possible stable droplet.

### 4.6 Global stability (droplet vs homogeneous state)

Solving $E(R) = E(0)$ together with $h(R) = 0$ (i.e. $4R^4 - 4mR + 3\sigma = 0$ and $4R^4 - 2mR + \sigma = 0$) gives $mR = \sigma$, $4R^4 = \sigma$:

$$
\boxed{\;m_{\text{eq}} = \sqrt2\,\sigma^{3/4} \approx 1.35\,\gamma^{3/8}, \qquad R_{\text{eq}} = \left(\frac{\sigma}{4}\right)^{1/4}\;}
$$

Always $m_{\min} < m_{\text{eq}}$. This is the finite-volume droplet evaporation/condensation transition (Binder & Kalos 1980; Biskup, Chayes & Kotecký 2002), with the known scaling $m \propto \sigma^{3/4}$.

### 4.7 Critical radius and barrier

For $m \gg m_{\min}$, neglecting $2R^4$ in $h$:

$$
R_c \approx \frac{\sigma}{2m}, \qquad \Delta F^* = V\left[E(R_c) - E(0)\right] \approx \frac{4\pi}{3}\sigma R_c^2 = \frac{\pi\sigma^3}{3m^2}.
$$

This coincides with classical nucleation theory, $R^* = 2\sigma/\Delta g$, $\Delta G^* = 16\pi\sigma^3/(3\Delta g^2)$, with driving force $\Delta g = \psi_{\text{hom}}\Delta\phi = 4m$. The extra term $4R^6$ (matrix depletion) is what distinguishes a closed volume: it creates the stable minimum $R_*$ and, for small $m$, removes the barrier together with the minimum.


---

## 5. Validity and caveats

- Thin interface for the stable droplet: $\xi \ll R_*$ and $\xi \ll 1 - R_*$. At the existence threshold $\xi/R_f \approx 2.7\,\gamma^{3/8}$ (0.08 for $\gamma = 10^{-4}$, 0.2 for $\gamma = 10^{-3}$).
- Thin interface for the critical nucleus: $R_c \approx \xi/(3m)$, so the $R_c$ formula is meaningful only for $m \ll 1/3$. Near the spinodal the nucleus is diffuse (Cahn–Hilliard 1959 nucleation theory), and the thin-interface $R_c$ is only qualitative.
- The energy analysis linearises $f$ around $-1$: requires $m \ll 1$.
- Only spherically symmetric perturbations are considered. With $\partial_n\phi = 0$ (90° contact angle) a droplet attached to the wall has a smaller interface area than a centred one of equal volume; the attraction to the wall is exponentially small, $\propto e^{-(1-R)/\xi}$. In a 1D radial solver this mode is excluded by symmetry.
- The CH equation is deterministic: it does not cross the barrier $\Delta F^*$ by itself. Metastable states persist indefinitely without noise.
- Logarithmic potential $f = (1+\phi)\ln(1+\phi) + (1-\phi)\ln(1-\phi) - \frac{\omega}{2}\phi^2$ (even): same structure with $\pm1 \to \pm\phi_b$ ($\phi_b = \tanh(\omega\phi_b/2)$), $a = \Delta\phi = 2\phi_b$, $c = f''(\phi_b) = \frac{2}{1-\phi_b^2} - \omega$, $\sigma$ from quadrature, $m = \bar\phi + \phi_b$, and $E(R) = 3\sigma R^2 + \frac{c}{2}(m - aR^3)^2$; $\psi_2 = 0$ still holds.


---

## 7. Suggested numerical checks

1. **Profile.** Converged stationary $\phi(r)$ vs $-\tanh\zeta + \frac{\sigma}{2R}\tanh^2\zeta$, with $R$ taken from $\phi(R) = 0$.
2. **Gibbs–Thomson.** $\psi_0 R/\sigma \to 1$; the deviation should scale as $(\xi/R)^2$ (zero Tolman length), not $\xi/R$.
3. **Bulk shift.** Bulk values $\pm1 + \psi_0/2$ in both phases.
4. **Radius.** Measured $R_*$ vs the large root of $h(R) = 0$; error expected $O(\gamma)$.
5. **Mass conservation.** $3\int_0^1\phi\,r^2dr$ constant in time to solver precision.
6. **Critical radius.** At fixed $\bar\phi$ with $m_{\min} < m$ (and $m \ll 1/3$), start from droplets with initial radius slightly below and above $R_c$ (matrix set consistently with mass); the first should dissolve, the second should relax to $R_*$.
7. **Existence threshold.** Scan $\bar\phi$ across $\bar\phi_{\min} = m_{\min} - 1$: below it every droplet dissolves, above it a droplet survives; the smallest surviving radius should approach $R_f$.
8. **Energy.** Compare the computed $F/V$ of the stationary droplet with $E(R_*)$ and with $f(\bar\phi)$ to locate $m_{\text{eq}}$.
