# Interacting density paths and statistics

The interacting experiments use native finite-volume masses on the periodic
torus, with G=32, N=2048, D=1 and
`V(x) = -kappa cos(x) + a2 cos(2x)`. Defaults are `kappa=0.75`, `a2=0.1`.
Both uniform and `exp(0.2 cos(x))` initial laws use IID particle histograms.
Every path freezes at its first numerical boundary crossing; stopped paths
remain in all statistics. No clipping or renormalization is performed.

## Density-path model

The causal, noise-conditioned Volterra/Picard solver already retained the
known nonlinear interaction in its integrated Itô residual. This extension
adds a uniform interacting case, configurable interaction strengths and paired
uniform/nonuniform presets. The exact physical interaction and mobility remain
part of the solver; the learned component is the response kernel. The network
does not learn unknown physics.

The same physical face Brownian increments drive neural and Euler paths.
Training randomness is separate. No auxiliary Brownian forcing is added by the
residual formulation. All-cell strong errors, full joint weak diagnostics,
causality, conservation and stopping checks remain those of the
[density-path workflow](PATH_SOLVER.md).

```bash
uv run nn-dk path-solver --preset path-interacting-smoke \
  --output runs/path-interacting-smoke
uv run nn-dk path-solver --preset path-interacting-pilot \
  --output runs/path-interacting-pilot
```

The short preset uses T=1/16, step 2^-8, fine reference 2^-11, one optimizer
seed and 64 test paths. Heat and additive-noise controls run alongside the two
interacting cases. The T=1 preset uses three seeds and 512 test paths. Comparing
with Euler at the same step measures approximation of that numerical map;
comparison with finer Euler also contains time discretization error.

## Full-state statistics model

The earlier noninteracting time-only moment family is retained with its
interaction rejection. Interactions destroy its finite moment closure: the
first moments couple to second moments, which couple to higher moments.
Even two densities with identical modes 1 and 2 can have different instantaneous
mode-2 drift. The new model therefore takes all G-1 mass-zero coordinates and
the remaining horizon. Physics is fixed for each fitted model; initial-profile
descriptors only define the initial law, not the conditional state map.

Let `y(m)` be the four normalized, sqrt(N)-scaled observables. The model uses

```
mu(m,tau) = y(m) + tau g(m) f_theta(m,tau)
C(m,tau)  = tau g(m) L_theta(m,tau) L_theta(m,tau)^T
g(m) = product_i tanh((m_i - m_stop) / epsilon), epsilon = 0.05/G
```

Absorbed states return `mu=y`, `C=0`. The gate is an approximation ansatz and
does not change the physical stopping threshold. Three SiLU hidden layers of
width 32 take all tangent coordinates. A triangular covariance head gives PSD
same-time covariance without a fixed covariance floor.

Training minimizes the backward-generator residuals

```
partial_tau mu - L mu
partial_tau C - L C - (grad mu) A (grad mu)^T
```

where `A=S S^T/N`. Exact directional Itô jets retain every physical face
direction and propagate second-order chain rules through the network. There
is no stochastic trace estimator and no future payoff/moment supervision.
The initial covariance-head bias uses the known instantaneous uniform-state
quadratic variation, not future statistics.

State collocation comes from independent native DK paths at fixed physical
snapshots, with independently sampled remaining horizons. Training and
validation split by whole trajectory families. AdamW is followed by chunked
full-bank LBFGS, with validation selection and resumable optimizer/RNG state.
Completion of the declared optimizer allocation does not certify accuracy.

```bash
uv run nn-dk interacting-moments --preset interacting-moments-smoke \
  --output runs/interacting-moments-smoke
uv run nn-dk interacting-moments --preset interacting-moments-pilot \
  --output runs/interacting-moments-pilot
```

Stages are `calibrate`, `generate`, `fit`, `evaluate`, `report` and `all`.
Resume with the identical command; exit 75 preserves completed work. Package
source, configuration, target, bank contents and checkpoint identities are
checked. Use a fresh run directory after source or configuration changes.

## Evaluation and interpretation

The statistics API returns physical conditional means and N-scaled physical
covariance for the four fixed, exactly cell-integrated sine/cosine modes.
Covariance is full 4x4 at each time; cross-time covariance is not supported by
this model. Ensemble statistics use independent multinomial initial samples
and the full law of total covariance, `E[C] + N Cov(conditional means)`.

All declared optimizer seeds must finish before final references unlock.
Conditional continuations, ensemble references, neural initial integration,
training and validation use separate family/RNG namespaces. Whole-family
bootstrap intervals separately track reference sampling and initial-integration
uncertainty; reported intervals are pointwise, not simultaneous certificates.

Reference Euler refinements share literal sums of the finest face Brownian
increments. Every conditional anchor has its own refinement allocation.
Paired mean/covariance and stopping diagnostics expose numerical sensitivity.
Small finite-step differences are not a continuous-time reference-error bound.
The earlier noninteracting stopping certificate is never reused here.

Short runs test execution and identify learning failures. Their small reference
ensembles cannot resolve a 1.25% fluctuation-uncertainty budget. A 5% target,
continuous stopped-DK convergence, rare-event stopping accuracy and a continuum
SPDE limit all require further evidence. Reports keep these claims unresolved.

Historical checkpoints retain their own source freezes and target identifiers.
The development checkout is isolated from the source-frozen T=1 density-path
batch while that batch is running.
