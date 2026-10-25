# Continuous stopped-generator reference certificate

The new generator workflow keeps the **5% maximum pointwise error** target and
the **1.25% uncertainty/reference contribution** budget. It uses independent
deterministic moment calculations plus an explicit stopping correction bound.
These calculations are reference numerics, not neural forecasts or labels
supplied to the neural value map.

The manually audited certification scope is the noninteracting periodic native
system with **G=32, N=2048, D=1, T≤1**, the uniform or `exp(0.2 cos(x))` IID
histogram initialization, and the four cell-integrated sine/cosine observables.
Calls outside that scope return an unresolved certificate. A valid certificate
states that the returned absolute bounds enclose the target; it does not state
that a network passes the accuracy gate.

## Interior moments

Write `alpha=D/h²`, let `B` be the periodic incidence matrix, and set

\[
K=-\alpha BB^\top,\qquad
A(m)=\alpha B\operatorname{diag}(m_i+m_{i+1})B^\top.
\]

The stopped diffusion has drift `Km` and covariance `A(m)/N` before its first
boundary hit. Its interior generator preserves polynomials of degree at most
two. For a polynomial `f(m)=mᵀQm+lᵀm`, the backward coefficient equations are

\[
\dot Q=K^\top Q+QK,\qquad
\dot l=K^\top l+\operatorname{tr}(Q A(e_i))_{i=1}^G/N.
\]

The reference solves this full coefficient system independently of the learned
Fourier representation. With Poisson rate `4 alpha`, one uniformization step is

\[
Q'_{ij}=\tfrac14(Q_{i-1,j}+Q_{i+1,j}+Q_{i,j-1}+Q_{i,j+1}),
\]

\[
l'_i=\tfrac12l_i+\tfrac14(l_{i-1}+l_{i+1})
 +[d_i(Q)+d_{i-1}(Q)]/(4N),
\quad d_i(Q)=Q_{ii}+Q_{i+1,i+1}-Q_{i,i+1}-Q_{i+1,i}.
\]

The finite Poisson mixture has an explicit geometric upper bound on its omitted
tail. `||Q_n||max≤||Q_0||max` and
`||l_n||max≤||l_0||max+2n||Q_0||max/N` control both probability and first-moment
tails. The implementation encloses the Poisson weights with 80-digit decimal
arithmetic and binary conversion bounds. IEEE gamma bounds cover the coefficient
updates, mixture sums, and final polynomial evaluation. A separate positive
radius-construction allowance covers rounding while evaluating those bounds.

For IID histogram starts with cell probabilities `p`, evaluation uses the exact
identity

\[
\mathbb E[m_i m_j]=(1-N^{-1})p_i p_j+N^{-1}p_i\mathbf1_{i=j}.
\]

It therefore includes initial sampling covariance without Monte Carlo.

## Why this also bounds the stopped target

No global unstopped DK process is assumed. If `U_f(t,m)` is the interior backward
polynomial above, stopped Dynkin's identity gives

\[
\mathbb E f(m_{T\wedge\tau})=U_f(T,m_0)
 +\mathbb E[\mathbf1_{\tau\le T}
 \{f(m_\tau)-U_f(T-\tau,m_\tau)\}].
\]

The remaining term is bounded with a first-hitting probability certificate.
For `u(0)=lambda e_i`, define

\[
\dot u_i=\alpha(u_{i-1}-2u_i+u_{i+1})
 -\frac\alpha2[(u_i-u_{i-1})^2+(u_i-u_{i+1})^2].
\]

Then `exp(-N u(H-t)ᵀm_t)`, stopped at `tau`, is a bounded positive martingale.
For `0<lambda≤1`, the unit cube is invariant. Its Jacobian has nonnegative
off-diagonal entries and zero row sums, so the flow is nonexpansive in the
maximum norm. Also `u(s)≤exp(Ks) lambda e_i` componentwise.

Every off-diagonal heat-kernel entry is at most `alpha s`: its derivative is
at most `alpha` because the two neighboring probabilities are distinct and
their sum is at most one. On a first boundary hit at cell `i` during the window
`[H-epsilon,H]`, with unit mass and boundary mass `r`,

\[
u(H-\tau)^\top m_\tau\le
\lambda[r+\alpha\epsilon(1-r)]=b.
\]

Optional stopping and a union bound over windows and cells give

\[
\mathbb P(\tau\le T\mid m_0)
\le\sum_{H\in\{\epsilon,2\epsilon,\ldots,T\}}\sum_i
\exp\{Nb-Nu^{(i)}(H)^\top m_0\}.
\]

For IID initialization the initial exponential is integrated exactly:

\[
\mathbb E\exp[-N u^\top m_0]
=\left(\sum_j p_j e^{-u_j}\right)^N.
\]

Probability intervals are propagated upward in this expression. Empty initial
histogram cells, which stop at time zero, are included. Fixed initial states that
are certainly already stopped are evaluated as frozen states with zero
conditional covariance. A state ambiguous within the enclosure of the boundary
constant receives an invalid certificate.

## Validating the Riccati calculation

On the invariant unit cube, the maximum-norm derivative bounds are

\[
M_1=3\alpha,\quad M_2=24\alpha^2,\quad M_3=264\alpha^3,
\quad M_4=3840\alpha^4,\quad M_5=69888\alpha^5.
\]

They follow from `||J||∞≤8 alpha` and
`||H(v,w)||∞≤8 alpha ||v||∞||w||∞`, with the constant Hessian of this quadratic
vector field. A fourth-order Taylor step has local remainder at most
`M5 delta^5/120`. Nonexpansivity lets these local errors add without a growing
Gronwall factor. The code projects only this auxiliary proof ODE onto its known
invariant cube; this does not clip or modify any DK trajectory.

The per-step floating-point allowance uses a 512-operation gamma factor times
the absolute Taylor polynomial bound. Parameter sensitivity adds
`3 T delta_alpha`. At the default `delta=epsilon=2^-16`, the global enclosure
is below `1e-8`. The packet remains unresolved if the requested numerical
certificate tolerance is not attained.

## Converting stopping probability to observable errors

For normalized observables `Y=sqrt(N) y/s`, where `s=sinc(k/G)`,
`|Y|≤sqrt(2N)`. The formal heat mean has the same bound. Its conditional
covariance entries have magnitude at most two: the scalar quadratic variation
is at most `4 lambda_k`, and the damped integral is at most two.

If the certified stopping probability is `P`, the normalized mean correction
is at most `2 sqrt(2N) P`. The raw second-moment correction is at most
`(4N+2)P`. The mean-product perturbation contributes at most `8NP`, yielding

\[
|\delta\operatorname{Cov}(Y)|\le(12N+2)P.
\]

The implementation converts these bounds back to physical means and `N` times
physical covariance. It adds the independently enclosed moment calculation and
initial-profile integration errors. Nearly unit input masses are accepted only
within `1e-12`; outward upper mass bounds and a conservative numerical radius
factor account for that tolerance. States are never renormalized.

## Geometry, initialization and OU arithmetic

Machin's arctangent identity encloses mathematical pi. Decimal trigonometric
Taylor series enclose the fixed observation functions. Explicit differences
between `D/(2*pi/G)^2` and the binary64 coefficient used by the numerical
algorithms enter both moment and Riccati error budgets.

Composite Simpson integration uses an explicit fourth-derivative bound for the
exponential profile and an interval for its normalization. The two specified
profiles are covered by the manual audit. Large or different profile families
are not silently certified by this workflow.

The OU reference uses positive exponential Taylor terms, a geometric remainder,
outward arithmetic, and reciprocal intervals to enclose `exp(-x)`. Thus its
conditional means and diagonal covariance have deterministic numerical bounds
and no reference sampling error.

The arithmetic audit assumes IEEE binary64 round-to-nearest for basic
operations and Python Decimal's documented correctly rounded transcendental
operations at 80 digits. Bound evaluation itself is inflated with explicit
operation-count gamma allowances. These are finite-dimensional references for
the stated stopped system, not a continuum DK convergence theorem.
