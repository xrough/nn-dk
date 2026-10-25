# Project A: a causal neural density-path solver

This implementation generates every finite-volume cell mass along a stochastic
trajectory. Its inputs are an initial histogram and the actual physical face
Brownian increments. Weights remain fixed during a realization. The separate
[generator-moment model](GENERATOR_MOMENTS.md) predicts statistics and is not
used as a density-path solver.

## Numerical target

The first scientific configuration uses the periodic interval `[0,2*pi)`,
G=32, N=2048, D=1 and T=1. Three DK cases use IID particle histograms:

1. Uniform initial law, no interaction.
2. Initial density proportional to `exp(0.2*cos(x))`, no interaction.
3. The same nonuniform initial law, with `V(x)=-0.75*cos(x)+0.1*cos(2*x)`.

The native finite-volume Itô system is

\[
dm=b_h(m)dt+N^{-1/2}S_h(m)dW,
\qquad S_h(m)=B\operatorname{diag}(\sigma(m)),
\]

where B is periodic conservative divergence and
`sigma_i²=D*(m_i+m_(i+1))/h²`. All 32 independent face drivers are retained.
Inside the positive simplex their cell covariance has rank 31. A deterministic
heat equation and an unstopped additive-noise field provide preceding controls.
These controls do not use the nonlinear DK stopping rule.

The implemented training target is the specified **Euler discretization**.
DK paths stop at the first crossing of density `0.01/(2*pi)` by an Euler
segment, interpolate to that crossing and freeze. No density is clipped or
renormalized. Every stopped path remains in the ensemble. A finer Euler
reference provides a separate time-discretization diagnostic. Its finite
difference does not certify continuous-time boundary-hitting bias or a
continuum DK limit.

## Neural ansatz

`CausalPathOperator` learns a full-resolution spectral Volterra kernel
`G_theta(lag, physics, dt)`. A three-hidden-layer SiLU MLP receives frequency,
squared sine frequency, physical lag, kappa, a2, D and dt. Every real Fourier
mode is retained, including Nyquist. The constant mode is fixed at one.

Positive Euler poles and a bounded learned lag correction parameterize the
kernel. Initially the poles correspond to **half** the known instantaneous
Laplacian rates; the initial network is not an exact future solution. The
per-mode pole parameterization admits the exact linear Euler response at the
supported grid/step. The finite MLP must jointly approximate the required
frequency corrections, and training must actually find them. Admissible Euler
poles alone do not establish a successful neural fit.
The positive-pole constraint requires `dt*lambda_max<1`; the G32 ladder
`2^-7,2^-8,2^-9` satisfies it.

Heat and additive-noise trajectories apply this learned response to the
initial field and supplied physical noise. DK adds fixed-depth causal Picard
blocks. Let \(\bar m_j^\ell\) be the stopped preceding iterate and
\(a_j^\ell\) its active-at-start indicator:

\[
\widehat m^{\ell+1}_n=G_\theta(n)m_0+
\sum_{j<n}G_\theta(n-1-j)a_j^\ell
\left[\Delta f_{\rm int}(\bar m^\ell_j)
 +N^{-1/2}S_h(\bar m^\ell_j)\Delta W_j\right].
\]

The interaction and mobility are the **known native formulas**, not another
learned coefficient network. Intermediate Picard paths are stopped before
mobility evaluation; their densities are not repaired. Forcing after their
absorption is zeroed. The outer path retains its raw crossing proposal before
applying the same stopping rule; raw outputs after absorption are not physical
frozen states.

The lower-triangular contraction has exactly zero future-noise dependency in
the computation graph. Its current cost is O(K²G), where K is the number of
time steps. This deliberately transparent implementation is not an assumed
speed improvement. Blocked/scan implementations are later performance work.

With the exact Euler kernel, each Picard block recovers an additional correct
time-step prefix, and K blocks recover the K-step stopped states in exact
arithmetic. This statement covers active proposals up to absorption; it makes
no promise about raw proposals after absorption.
Useful accuracy at depth four is an experimental observation, not a global
contraction theorem. Learned-kernel error and finite-depth error must both be
measured. Stable poles alone do not prove nonlinear stochastic stability.

## Training and noise

Training uses complete uncensored pre-stop proposals in the integrated Itô
residual, supplemented by a local residual:

\[
r_k=\widehat m^{\rm proposal}_{k+1}-\widehat m_k
       -b_h(\widehat m_k)\Delta
       -N^{-1/2}S_h(\widehat m_k)\Delta W_k,
\qquad R_n=\sum_{k<n}r_k.
\]

The loss sums every cell, hence all 31 mass-zero directions, with fixed
uniform-histogram fluctuation normalization. Origins after absorption have
zero exposure; the crossing proposal contributes its **full** residual.
The outer path/time denominator includes every allocated trajectory. Absorbed
increments are not fitted as ordinary uncensored Gaussian increments.

The ansatz and residual use the same physical Brownian realization. AdamW
minibatch shuffling has an independent RNG; no auxiliary Brownian forcing is
introduced by the variational formulation. Deterministic spatial summation
avoids stochastic quadrature at this grid size. Training and validation banks
contain only initial histograms and noise inputs, with no future-path labels.
Fine simulated references appear only after all final fits are locked.

Checkpoints preserve current/best weights, AdamW and LBFGS states, shuffle and
Torch RNG states, source/configuration/data fingerprints, histories and costs.
Selection uses the independent validation residual. Completion of the declared
optimizer allocation is distinct from optimization convergence or accuracy.

## Evaluation

Full-state strong error uses the same initial masses and literal shared
Brownian increments:

\[
E_{\rm path}=\sqrt{\frac{N}{1-1/G}
\mathbb E\max_n\sum_j
|\widehat m_{n,j}-m^{\rm reference}_{n,j}|^2}.
\]

This is density-L2 error divided by the analytic uniform-IID histogram
fluctuation energy; the cell-width factor cancels. All stored times and cells
enter the maximum. No four-mode surrogate can substitute for this metric.

Reports also retain all-cell means, full same-time and two-time covariance at
the configured physical queries, fixed cell-integrated Fourier modes 1 and 2,
scalar distribution diagnostics, stopping counts/times, conservation and
invalid paths. One invalid path prevents survivor-conditioned estimates.
Quadratic variation uses uncensored innovations and predictable active-at-start
exposure, including exiting proposals.

Whole trajectory families are bootstrapped. The 5% strong/weak research gates
use pointwise paired contrasts and a 1.25% uncertainty budget; they are not
simultaneous confidence guarantees or certification of each marginal law's
Monte Carlo precision. Zero reference bias is used only for the explicitly
specified same-step Euler target. Fine-reference comparisons retain an
unresolved continuous-time bias flag.

The three recorded comparisons are NN versus same-step Euler, same-step Euler
versus fine Euler, and NN versus fine Euler. Their errors are not conflated.
Future-noise perturbations and same-noise replay checks accompany every final
model. All fits finish and are byte-locked before final inputs are generated.

## Commands and saved paths

```bash
uv sync --locked --group dev
uv run pytest
uv run nn-dk path-solver --preset path-smoke --output runs/path-smoke
uv run nn-dk path-solver --preset path-first-batch --output runs/path-first-batch
```

`path-smoke` runs all five cases at G32, N2048, T=1/16, with 32/16/64
training/validation/test paths, one seed and step 2^-7. `path-first-batch` uses
T=1, 256/64/256 paths, three seeds, width32/depth4, 1000 AdamW steps and 100
LBFGS outer calls: 15 fits. `path-controls` runs the two controls over three
steps; `path-pilot` adds the three DK cases for a 45-fit resolution study.
Each numerical step gets a separate fit.

Stages are `calibrate`, `generate`, `fit`, `evaluate`, `report`, or `all`.
Controls are fitted and checked before DK regardless of case ordering.
An unresolved control gate leaves final inputs locked. Exit 75 means the
invocation budget expired with resumable work. Rerun the identical command;
source/configuration changes require a new directory.

HDF5 stores initial/noise banks, fine reference paths, and each model's full
density paths and raw proposals. Content hashes protect completed rows, saved
trajectories, checkpoints and evaluation caches. Reports export Markdown,
JSON, CSV and separate strong/weak error figures.

To sample a saved checkpoint directly:

```python
from nn_dk.path_training import load_path_model

model = load_path_model("runs/path-first-batch/checkpoints/<checkpoint>.pt")
context = model.path_config.context(model.case)
sample = model.sample(context, times=(0., .25, .5, 1.), n_paths=16, seed=123)
print(sample["states"].shape)  # (16, 4, 32)
```

Supply `initial_masses=` and `brownian_increments=` to reproduce a declared
physical driver exactly. Loaded samplers reject a different context, numerical
step or horizon beyond the fitted interval. Control fields and DK have distinct
problem/target identifiers.

## Literature and boundaries

The implementation follows the approved idea of a neural trial operator with
an integrated Itô residual. Relevant foundations are
[Deep Ritz](https://arxiv.org/abs/1710.00211),
[Neural Galerkin](https://arxiv.org/abs/2203.01360),
[Neural SPDE](https://arxiv.org/abs/2110.10249), and the 2026
[Wiener-chaos neural operator](https://proceedings.mlr.press/v306/shi26i.html).
This implementation uses a learned Volterra kernel, rather than a truncated
Wiener-chaos expansion or an online SDE for neural parameters. It is our
validated implementation candidate, not an assertion of a best DK method.

Recent DK numerical work emphasizes preserving discrete cross variation and
quantifying positivity effects:
[Arora–Cornalba–Shardlow, September 2026](https://arxiv.org/abs/2609.06681) and
[Damnjanović–Djurdjevac–Perkowski, August 2026](https://arxiv.org/abs/2608.00838).
These do not automatically prove the neural ansatz's convergence. Moving from
fixed-grid stopped dynamics to a continuum target requires an explicit noise
regularization, consistent with the
[smooth-interaction weak-error framework](https://arxiv.org/abs/2502.20929).
