# Face averaging of a degenerate mobility (supervisor item #2)

## 1. Why it matters

The flux form of the first equation is $\partial_t\phi=\nabla\!\cdot\!(M(\phi)\nabla\psi)$,
discretised as

$$\big[\,M_{i+\frac12}(\psi_{i+1}-\psi_i)-M_{i-\frac12}(\psi_i-\psi_{i-1})\,\big]/h^2 .$$

The whole point of a degenerate mobility is that the **bulk** phases cannot transport mass, so the
drop cannot leak into them. That only works if $M_{i+\frac12}$ is actually small on a face between a
pure-bulk cell and its neighbour. Take the parabolic mobility
$M(\phi)=\sqrt{A(1-\phi^2)^2+M_{min}^2}$ with $M_{min}=10^{-5}$, and a face between a pure cell
($\phi=-1$, $M=M_{min}$) and an interface cell ($\phi=0$, $M\approx1$):

| rule | $M_{i+1/2}$ | value | verdict |
|---|---|---|---|
| **midpoint** (current code) | $M\big(\tfrac{\phi_i+\phi_j}{2}\big)=M(-0.5)$ | $\approx0.75$ | wide open |
| arithmetic | $\tfrac12(M_i+M_j)$ | $\approx0.5$ | wide open |
| geometric | $\sqrt{M_iM_j}$ | $\approx3\times10^{-3}$ | mostly closed |
| **harmonic** | $2M_iM_j/(M_i+M_j)$ | $\approx2\times10^{-5}$ | closed |

The rule we ship is the *leakiest of all four*. This is almost certainly the reason the supervisor
sees the degenerate mobility "всё равно быстро спадает": the numerical mobility never actually
degenerates, it just dips for one cell and then averages its way back to $O(1)$.

Harmonic averaging is not merely more restrictive, it is **more accurate**: for a 1D steady
diffusion problem with a piecewise-constant coefficient it reproduces the exact flux (series
resistances add), while arithmetic/midpoint averaging is $O(1)$ wrong when the coefficient jumps
across the face. For a *smooth* $M$ all four agree to $O(h^2)$, so switching costs nothing in the
constant-mobility runs.

Caveat: harmonic averaging makes $M_{i+1/2}\le 2\min(M_i,M_j)$, so the discrete operator inherits
the full $M_{max}/M_{min}$ contrast. Expect the linear solve to get harder in exactly the regime
where the physics gets better. That trade-off is what Phase 3.2 measures.

## 2. Face rules and their derivatives

With $M_a=M(\phi_a)$, $M_b=M(\phi_b)$, and $\bar\phi=(\phi_a+\phi_b)/2$:

| rule | $F(\phi_a,\phi_b)$ | $\partial F/\partial\phi_a$ | $\partial F/\partial\phi_b$ |
|---|---|---|---|
| midpoint | $M(\bar\phi)$ | $\tfrac12M'(\bar\phi)$ | $\tfrac12M'(\bar\phi)$ |
| arithmetic | $\tfrac12(M_a+M_b)$ | $\tfrac12M'(\phi_a)$ | $\tfrac12M'(\phi_b)$ |
| harmonic | $\dfrac{2M_aM_b}{M_a+M_b}$ | $\dfrac{2M_b^2}{(M_a+M_b)^2}M'(\phi_a)$ | $\dfrac{2M_a^2}{(M_a+M_b)^2}M'(\phi_b)$ |

All three are symmetric: $F(a,b)=F(b,a)$ and $\partial_aF(a,b)=\partial_bF(b,a)$, which the
implementation should preserve exactly so that the discrete operator stays symmetric in $\psi$.

## 3. Consistent Jacobian

Residual (one axis $j$, cell $i$, $M_p=F(\phi_i,\phi_{i+1})$, $M_m=F(\phi_{i-1},\phi_i)$):

$$F^{(0)}_i \supset \frac{M_p(\psi_{i+1}-\psi_i)-M_m(\psi_i-\psi_{i-1})}{h^2}.$$

Linearising about $(\psi^L,\phi^L)$ in the direction $(\delta\psi,\delta\phi)$, and writing
$g_p=\psi^L_{i+1}-\psi^L_i$, $g_m=\psi^L_i-\psi^L_{i-1}$:

$$\delta F^{(0)}_i=\frac{M_p(\delta\psi_{i+1}-\delta\psi_i)-M_m(\delta\psi_i-\delta\psi_{i-1})}{h^2}
+\frac{\delta M_p\,g_p-\delta M_m\,g_m}{h^2},$$

$$\delta M_p=\partial_aF(\phi^L_i,\phi^L_{i+1})\,\delta\phi_i+\partial_bF(\phi^L_i,\phi^L_{i+1})\,\delta\phi_{i+1},$$
$$\delta M_m=\partial_aF(\phi^L_{i-1},\phi^L_i)\,\delta\phi_{i-1}+\partial_bF(\phi^L_{i-1},\phi^L_i)\,\delta\phi_i.$$

so the $\delta\phi$ block is

$$\frac{1}{h^2}\Big[\big(\partial_aF_p\,g_p-\partial_bF_m\,g_m\big)\delta\phi_i
+\partial_bF_p\,g_p\,\delta\phi_{i+1}
-\partial_aF_m\,g_m\,\delta\phi_{i-1}\Big].$$

**This reduces exactly to the code's present "VARIANT 2" discrete linearisation** when
$F$ = midpoint, since then $\partial_aF_p=\partial_bF_p=\tfrac12M'(\bar\phi_p)$ — the explicit
$\tfrac12$ factors in `kernels/jacobi_op.h` are that chain-rule factor. They must be *removed* when
the derivative is supplied by `face_diff_*`, otherwise the factor is applied twice.

## 4. The smoother diagonal is currently wrong

`kernels/jacobi_pre.h` builds the $(0,0)$ entry as

```
mat(0,0) += mobility(lin_curr[1]) * diag_j[0] / (hj*hj);
```

i.e. the **cell** mobility times $(-2+\text{ghost coef})$. The true diagonal of the term above is

$$\frac{\partial F^{(0)}_i}{\partial\psi_i}=\frac{-(M_p+M_m)+c_p M_p+c_m M_m}{h^2},$$

with $c=+1$ at a Neumann ghost, $-1$ at a Dirichlet ghost, $0$ for an interior or halo neighbour.
With a constant mobility the two coincide. With a degenerate one they differ by orders of
magnitude — e.g. an interface cell sitting next to bulk has $M_i\approx1$ but $M_p+M_m\approx
M_{min}$ under harmonic averaging, so the preconditioner under-relaxes by $10^5$. Fixing this is
part of the same change, and Phase 3.4 checks whether it is what has been hurting the smoother.

The $(0,1)$ entry likewise becomes $\big(\partial_aF_p\,g_p-\partial_bF_m\,g_m\big)/h^2$,
matching §3.

## 5. What this does and does not fix

From [03](03_shrinkage_criterion.md): the mobility does not appear in the critical radius at all.
Harmonic averaging cannot make the drop thermodynamically stable — it buys **time**, by making the
drop's lifetime $t_{sh}$ long compared with the simulation. That is exactly the escape route Yue et
al. recommend (their §5), and it is the honest framing for the write-up: we are not removing the
artefact, we are pushing it past the horizon of the computation. Prediction P6 is the test.
