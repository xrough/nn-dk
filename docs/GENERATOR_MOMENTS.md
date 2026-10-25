# Neural moments from a known generator

This track learns conditional means and full same-time covariances for the
noninteracting, stopped native DK system. It follows the unsuccessful
[trajectory BSDE experiments](experiments/bsde_anchor_study_2026-10-24.md), but
uses a different training objective. It is a statistics solver, with no density
path or cross-time covariance output.

## Target and outputs

The target is the continuous-time finite-volume SDE on the periodic torus,
with 32 cells, 2048 particles, diffusion 1 and horizon 1. Its native state is
the vector of cell masses. A path freezes at the first continuous hit of the
specified positive density threshold `0.01/(2*pi)`. Initial masses are IID
particle histograms from either the uniform law or the normalized
`exp(0.2*cos(x))` law. Every stopped path belongs to the target law.

The four physical test functions are `sqrt(2)*cos(k*x)` and
`sqrt(2)*sin(k*x)`, for k=1,2. Their averages over each cell are integrated
exactly against the reconstructed constant density. Predictions are physical
means and N-scaled physical covariance matrices at times 1/4, 1/2 and 1.
Conditional inputs are complete initial histograms; ensemble outputs integrate
the known multinomial initial law.

## Network and loss

The interior generator preserves polynomials of degree at most two. Translation
symmetry therefore supplies a much smaller state dependence than a generic
32-input MLP. In normalized coordinates the model has the form

\[
\mu_\theta(t,m)=\sqrt N D_\theta(t)Wm,\qquad
C_\theta(t,m)=\sum_jm_j R_j K_\theta(t)R_j^\top.
\]

Here W contains the fixed cell observations, R_j are their known rotations,
and a three-hidden-layer, width-32 SiLU network supplies the time-dependent
propagators and kernel factors. The Gram ansatz

\[
K_\theta(t)=t(F_0+tA_\theta(t))(F_0+tA_\theta(t))^\top
             +t^2B_\theta(t)B_\theta(t)^\top
\]

enforces positive semidefiniteness and zero initial conditional covariance.
The known instantaneous noise factor F_0 sets the derivative at zero. No fixed
covariance floor is added. All future corrections and propagators are neural
outputs; an exact future moment solution is not inserted at inference.

Training minimizes the continuous generator residuals for the propagators and
covariance kernel. The configuration declares 65 collocation times, 257 offset
validation times, 1000 Adam steps and 100 LBFGS outer calls of at most 10
iterations. Validation selects the checkpoint with the smallest maximum raw
residual. A completed optimizer budget is distinct from attaining the internal
`1e-6` residual tolerance, and both are distinct from statistical accuracy.
Three optimizer seeds are evaluated separately.

This is known-mechanism physics-informed training, rather than trajectory-only
BSDE learning. It needs no forward trajectory labels, stochastic forcing draws,
or auxiliary martingale sampling noise. Consequently it does not settle how to
represent independent physical and variational noises in a learned path solver.

## Independent references and acceptance

The reference uses the full degree-two native backward operator with Poisson
uniformization. A separate exponential-martingale argument bounds continuous
boundary hitting and the resulting stopping correction. Numerical, geometry
and initial-profile integration errors are enclosed explicitly. See the
[derivation](GENERATOR_REFERENCE_CERTIFICATE.md) and
[independent mathematical audit](GENERATOR_REFERENCE_AUDIT.md).

The neural interior moment ansatz does not solve the stopped boundary condition
globally. Its discrepancy from the stopped target must fit inside the same
error gate after the reference stopping bound is included. Initially stopped
conditional states are frozen exactly. Ensemble predictions use the known
multinomial mean and covariance, with initial and future stopping effects
covered by the reference bound.

Acceptance checks every mean and every full same-time covariance entry, over
every declared test initial state, time, profile, scope and optimizer seed.
For the fixed observation scale s_k=sinc(k*pi/G), errors are normalized as

\[
e_\mu=\sqrt N(\mu_\theta-\mu_{\rm ref})/s,\qquad
e_C=(C_\theta-C_{\rm ref})/(ss^\top).
\]

Each absolute error plus its reference error bound must be at most 0.05.
Each reference contribution must also be at most 0.0125. These are fractions
of fixed observation scales, not relative errors divided by a possibly zero
mean or off-diagonal covariance. Deterministic reference enclosures remove
reference Monte Carlo noise; optimizer variation remains separate. No RMSE
substitution or best-seed selection is allowed.

The run evaluates development states first. All six completed fits must pass
development and be locked before fresh final inputs are generated. Source,
configuration, checkpoints and cached arrays are fingerprinted. Changing source
or configuration requires a new run directory.

## Reproduction and scope

```bash
uv sync --locked --group dev
uv run nn-dk generator --config configs/generator_accuracy.toml \
  --output runs/generator-accuracy
```

Rerun the identical command to resume. `--budget-seconds 600` sets an invocation
budget; exit 75 means resumable work remains. Stages are `calibrate`, `generate`,
`fit`, `evaluate`, `report`, or `all`. A failed development gate leaves final
inputs locked. `generator-smoke` is a small execution check whose two optimizer
steps are not expected to satisfy scientific accuracy.

Passing this track establishes the declared moment accuracy at a fixed grid
and particle number for the tested cases and times. It establishes neither a
continuum DK limit nor accuracy of density paths, interacting dynamics, rare
events, an entire cross-time joint law, or arbitrary initial profiles. The
time-only network can be queried between test times, but those additional times
are not covered by the finite test-bank accuracy result.
