# Dean–Kawasaki background and the numerical target

The Dean–Kawasaki (DK) equation describes the fluctuations of an empirical
particle density using conservative, density-dependent noise. NN-DK studies
neural approximations of specified finite-dimensional dynamics and statistics.
The continuum equation explains the coefficients and their particle-number
scaling; it does not by itself certify the numerical or neural approximation.

This guide adapts selected exposition from Xiaohao Ji's doctoral thesis,
*Stochastic PDEs and Mean-Field Particle Systems*, Freie Universität Berlin,
2026, including local revisions of its introduction and weak-error chapter.
The interacting weak-error results are joint work of Ana Djurdjevac, Xiaohao Ji,
and Nicolas Perkowski. See [source provenance](provenance.md) for the inspected
snapshot and [the published paper](https://doi.org/10.1016/j.matpur.2026.103937)
for the full hypotheses and proofs.

## From particles to a conservative stochastic equation

Use the periodic interval $\mathbb T=[0,2\pi)$, diffusion constant $D>0$, and
independent standard Brownian motions $B^1,\ldots,B^N$. For an interaction
potential $V$, consider

```math
dX_t^i=b(X_t^i,\mu_t^N)\,dt+\sqrt{2D}\,dB_t^i,
\qquad
\mu_t^N=\frac1N\sum_{i=1}^N\delta_{X_t^i},
\qquad
b(x,\mu)=-(V'*\mu)(x).
```

The empirical measure has total mass one. For a smooth periodic test function
$\varphi$, Itô's formula gives the exact identity

```math
d\langle\varphi,\mu_t^N\rangle
=\langle D\varphi''+b(\cdot,\mu_t^N)\varphi',\mu_t^N\rangle\,dt
+dM_t^\varphi,
\qquad
dM_t^\varphi=\frac{\sqrt{2D}}N
\sum_{i=1}^N\varphi'(X_t^i)\,dB_t^i.
```

Independence of the particle drivers determines the cross variation:

```math
d[M^\varphi,M^\psi]_t
=\frac{2D}{N}\langle\varphi'\psi',\mu_t^N\rangle\,dt.
```

Formally replacing the empirical measure by a density $\rho$ gives

```math
d\rho_t
=\left[D\partial_{xx}\rho_t
-\partial_x\big(b(\cdot,\rho_t)\rho_t\big)\right]dt
+\partial_x\left(\sqrt{\frac{2D\rho_t}{N}}\,dW_t\right),
```

where $W$ is a cylindrical Wiener process on $L^2(\mathbb T)$, the time
integral of space-time white noise. Testing the noise against $\varphi$ gives
the required cross variation. The divergence makes the constant test function
have zero drift and zero noise, so mass is conserved. Its overall sign can be
absorbed into $W$. This is the structure in
[Dean's derivation](https://doi.org/10.1088/0305-4470/29/24/001)
([preprint](https://arxiv.org/abs/cond-mat/9611104)).

For a probability density, the noise amplitude is proportional to
$N^{-1/2}$ and its covariance to $N^{-1}$. For the unnormalized number density
$n=N\rho$, the same equation has noise amplitude $\sqrt{2Dn}$; changing density
normalization changes the displayed coefficient. The probability convention
is used throughout NN-DK. Initial IID cell histograms with cell probabilities
$p$ also satisfy

```math
\operatorname{Cov}(m_0)
=\frac1N\left[\operatorname{diag}(p)-pp^\mathsf T\right].
```

The formal density notation needs care: $\mu_t^N$ is atomic, and its square root
is not an ordinary density function. In the setting analyzed by
[Konarovskyi, Lehmann, and von Renesse](https://doi.org/10.1214/19-ECP208)
([preprint](https://arxiv.org/abs/1806.05018)), the unregularized martingale
problem has solutions only in discrete admissible parameter regimes, and those
solutions are empirical particle measures. One cannot infer a freely evolving
smooth-density model from the formal equation.

## Regularization and weak approximation

A continuum approximation can instead smooth both the square root and the
spatial noise. In the constant-diffusion setting above, its form is

```math
dm_t^N
=\left[D\partial_{xx}m_t^N
-\partial_x\big(b(\cdot,m_t^N)m_t^N\big)\right]dt
+\sqrt{\frac{2D}{N}}\,\partial_x\big(f_{\delta_N}(m_t^N)\,dW_t^N\big).
```

Here $f_\delta(0)=0$ is a Lipschitz approximation with
$|f_\delta(r)^2-r|\lesssim\delta$ for $r\geq0$, and
$`\|f_\delta'\|_\infty\lesssim\delta^{-1/2}`$. The colored noise has a Fourier
representation

```math
W_t^N=\sum_k\theta_k^N e_k B_t^k,
\qquad
\theta_k^N=\widehat\eta(k/L_N)^{1/2},
```

with the usual conjugacy conditions giving a real field. The covariance kernel
is chosen so that $\widehat\eta$ is smooth, nonnegative, compactly supported,
and $\widehat\eta(0)=1$. Increasing $L_N$ admits more spatial noise modes.
The parameters cannot be removed independently: stochastic parabolicity
requires a bound of the form

```math
\frac{\|\theta^N\|_{\ell^2}^2}{N\delta_N}\leq c,
```

where $c$ depends on the ellipticity and diffusion bounds. Diffusive coercivity
must dominate the regularized noise's contribution.

Djurdjevac, Ji, and Perkowski treat more general smooth mean-field drift and
diffusion coefficients in
[*Weak Error of Dean–Kawasaki Equation with Smooth Mean-Field Interactions*](https://doi.org/10.1016/j.matpur.2026.103937),
*Journal de Mathématiques Pures et Appliquées* **213** (2026), 103937
([preprint](https://arxiv.org/abs/2502.20929)). Their weak-error theorem assumes
the specified covariance construction, uniform ellipticity and coefficient
bounds, stochastic parabolicity, and sufficiently smooth coefficients and
test functionals with controlled measure derivatives through order four and
the stated spatial regularity. Initial data are prepared by mollifying the
initial empirical measure at scale $\varepsilon$.

Under those hypotheses, for a suitable smooth functional $F$ and fixed horizon
$T$, the continuum comparison has the form

```math
\sup_{t\in[0,T]}
\left|\mathbb E F(\mu_t^N)-\mathbb E F(m_t^{N,\varepsilon})\right|
\lesssim_T
\varepsilon+\frac{\delta_N}{N}
+\frac{\log(\varepsilon^{-1})}{NL_N^2}.
```

In dimension $d$, balancing terms subject to parabolicity with
$\varepsilon\asymp N^{-2}$, $\delta_N\asymp N^{-2/(d+2)}$, and
$L_N\asymp\delta_N^{-1/2}$ yields order
$N^{-1-2/(d+2)}\log N$; in one dimension this is $N^{-5/3}\log N$.
This is a comparison of expectations for the specified regularized SPDE and
particle system. It is not a pathwise error rate, a time-stepping error rate,
or a neural-network error bound.

## The stopped finite-volume law implemented here

The native state in [the implementation](../src/nn_dk/native.py) is a vector
of probability **masses** $m_i$ in $G$ periodic cells of width $h=2\pi/G$.
The associated piecewise-constant cell density is $m_i/h$. With face $i$
between cells $i$ and $i+1$, the face density is
$\rho_{i+1/2}=(m_i+m_{i+1})/(2h)$. The implementation integrates the force of
this piecewise-constant density exactly for
$V(x)=-\kappa\cos x+a_2\cos(2x)$ to obtain face velocity $v_{i+1/2}$.
Its deterministic flux and drift are

```math
J_i=\rho_{i+1/2}v_{i+1/2}
-\frac{D}{h^2}(m_{i+1}-m_i),
\qquad
(b_h(m))_i=J_{i-1}-J_i.
```

Let $B$ denote the periodic incidence matrix with $`(Bq)_i=q_{i-1}-q_i`$.
One independent Brownian driver is assigned to each face, shared by its two
neighboring cells with opposite signs:

```math
dm=b_h(m)\,dt+N^{-1/2}S_h(m)\,dW,
\qquad
S_h(m)=B\operatorname{diag}(\sigma(m)),
\qquad
\sigma_i(m)^2=\frac{D(m_i+m_{i+1})}{h^2}.
```

Consequently $\mathbf 1^\mathsf T b_h=0$,
$\mathbf 1^\mathsf T S_h=0$, and the cell covariance per unit time is
$S_hS_h^\mathsf T/N$. `native_coefficients` returns $S_hS_h^\mathsf T$
**before** division by $N$. Neighboring cells have negative noise cross
variation; replacing their shared face driver with independent cell noise
would change the law. At positive masses the covariance has rank $G-1$.

The numerical target is the specified Euler recursion, stopped at the first
crossing of density $\rho_{\rm stop}=0.01/(2\pi)$ by its piecewise-linear Euler
polygon. The crossing state is interpolated and then frozen. Paths that begin
at or below this threshold stop immediately. Every path remains in the
ensemble; no cell mass is clipped and no state is renormalized. Thus the
threshold and stopping convention are part of the target law.

The [causal neural path solver](PATH_SOLVER.md) approximates that same stopped
Euler law using the supplied physical face Brownian increments. Its learned
response kernel and finite Picard depth introduce errors that must be measured
against the native reference. Comparisons with a finer Euler step assess an
additional time-discretization effect.

These experiments do not establish a continuum limit as $h\to0$, a match
between the face-noise discretization and the theorem's chosen colored noise,
the required coupled $N,\delta_N,L_N$ limit, or the theorem's prepared initial
data. They also do not resolve continuous-time boundary-hitting bias. The
finite-grid stopping rule differs from the continuum square-root
regularization. Smooth trigonometric interactions are compatible with the
motivation for the weak theory, but that observation supplies none of these
missing approximation estimates. Accuracy claims in this repository therefore
refer to each experiment's declared particle, discrete, or neural target.
