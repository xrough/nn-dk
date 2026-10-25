# Causal neural density-path solver — 25 October 2026

The approved Project A implementation now generates **full stochastic density
paths**, with all 32 native cell masses and all physical face-noise drivers.
The completed short-horizon experiments show accurate learning of the declared
Euler path map. Their much larger differences from fine Euler show that
same-step neural accuracy does not establish continuous-time DK accuracy.

## What was implemented

A three-hidden-layer SiLU network learns a spectral Volterra response kernel.
The known conservative interaction and multiplicative face mobility enter
four causal Picard blocks in the scientific configuration. Network weights
remain fixed during a realization. This is a learned response operator, not
the earlier moment predictor or an online learned-coefficient Euler solver.

Training minimizes full-cell integrated and local left-endpoint Itô residuals.
The same physical Brownian increments appear in the ansatz and residual.
Optimizer shuffling has an independent RNG. No additional Brownian forcing
comes from the variational objective, and no simulated future density labels
enter training or validation banks.

The native system has G=32, N=2048 and D=1 on the periodic interval `[0,2*pi)`.
DK initial states are IID particle histograms from the uniform profile or the
normalized `exp(0.2*cos(x))` profile. The interacting case uses
`V(x)=-0.75*cos(x)+0.1*cos(2*x)`. DK paths freeze at the first Euler-segment
crossing of density `0.01/(2*pi)`. Crossing proposals remain uncensored in the
residual and diagnostics. Every absorbed path remains in statistics; failed
ensembles cannot report survivor averages.

The [method guide](../PATH_SOLVER.md) and
[mathematical audit](../PATH_OPERATOR_AUDIT.md) disclose the parameterization,
active forcing mask, finite Picard error and stopping limitations. Positive
linear poles alone do not prove stochastic stability. Each timestep receives
its own fit.

## Completed execution and short refinement checks

Both completed runs used T=1/16, width16/depth3, 32 training, 16 validation and
64 locked-final independent paths per case. Both fitted heat, additive-noise,
uniform DK, nonuniform DK and interacting DK. They used optimizer seed zero,
200 AdamW steps and 20 LBFGS outer calls. Final input generation followed the
complete fit lock.

The first smoke has five fits at step 2^-7 and fine reference 2^-10. Every
same-step strong and weak gate passed. The largest full-state strong RMS was
**0.2692%**, with largest 95% upper endpoint **0.2766%**. Maximum mean and full
joint covariance interval endpoints were **0.8957%** and **0.8506%**. Its
largest neural/fine-reference strong difference was **35.95%**.

The independent refinement smoke has 15 fits on nested physical noise inputs:

| Step | Largest NN/same-step strong RMS | Largest 95% upper endpoint | Largest Euler/fine RMS | Largest NN/fine RMS |
|---|---:|---:|---:|---:|
| 2^-7 | 0.3161% | 0.3235% | 36.53% | 36.53% |
| 2^-8 | 0.02392% | 0.02515% | 15.15% | 15.15% |
| 2^-9 | 0.08564% | 0.09330% | 5.702% | 5.702% |

The table takes maxima across all five cases, retaining every individual
comparison in the [15-row archive table](artifacts/path_refinement_smoke_2026-10-25/path_table.csv).
Every declared same-step strong/weak gate passed. The largest mean and joint
covariance interval endpoints were 1.394% and 0.8558%. Neural approximation
error is not monotone under refinement, consistent with finite optimization
and Picard errors; no learning or temporal order is fitted.

All errors are normalized by fixed fluctuation units, not by the possibly
zero mean or a covariance entry. Strong RMS is
`sqrt(N/(1-1/G) * mean_paths(max_times(sum_cells(error**2))))`.
Weak gates inspect full cell means and full same-time/two-time joint
covariances at both queries, plus fixed cell-integrated modes 1 and 2.
Intervals are paired whole-trajectory **pointwise** bootstrap intervals, not
simultaneous confidence bands or certification of marginal Monte Carlo
precision. The stored time grid differs between refinement levels.

The stochastic fine-reference errors at the smallest smoke step still exceed
5%. A 2^-10 finite reference is not a bound on continuous-time stopping bias.
These small experiments validate execution and identify a dominant temporal
error; they do not demonstrate the intended T=1 DK convergence claim.

Both runs had zero observed exits or invalid paths, with maximum mass error
2.23e-16. Zero exits in 64 paths per profile does not test rare stopping events.
Boundary interpolation and uncensored exit handling are instead covered by
deliberate boundary tests. Full-field saved paths, physical-noise replay,
future-noise perturbations, exact family disjointness and Brownian aggregation
were independently audited; see the [evidence audit](../PATH_SOLVER_EVIDENCE_AUDIT.md).

## Calibration and reproducibility

Source freeze is `bd256de`, package identity
`a37ebd4fa82c8c590922417004d47150b1ea71d75be635a316b670506053223a`.
Verification at this source passed **580 tests** and Ruff. Tests include all
grid modes, conservative noise covariance, actual independent held-out
learning, causal values/gradients, boundary interpolation, proposal storage,
resume equivalence, immutable completed checkpoints and final-input locks.

The initial T=1 calibration caught a numerical problem in an independent
Gaussian reference calculation: a Van Loan block containing `-A` grew
exponentially, causing cancellation when multiplied by `exp(A*T)`. This was
corrected by an independent dissipative Lyapunov covariance ODE. T=1 agreement
with the spectral covariance is 6.34e-14. A regression test now covers this
horizon. The failed calibration had not fitted models or opened final inputs;
it was not a neural rollout instability. Its original run remains untouched.

The complete smoke evidence is archived separately:

- [Five-fit smoke](artifacts/path_smoke_2026-10-25/README.md).
- [Fifteen-fit refinement smoke](artifacts/path_refinement_smoke_2026-10-25/README.md).

Both archives preserve exact designs, source/configuration hashes, all
checkpoint hashes, complete compressed report data and actual density-path
figures. Full restorable checkpoints and physical input/output HDF5 banks
remain in the named local `runs/` directories.

```bash
uv sync --locked --group dev
uv run pytest
uv run nn-dk path-solver --preset path-smoke --output runs/path-smoke-reproduction
uv run nn-dk path-solver --config configs/path_refinement_smoke.toml \
  --output runs/path-refinement-smoke-reproduction
uv run nn-dk path-solver --preset path-first-batch --output runs/path-first-batch-reproduction
```

Rerun an identical command to resume; exit 75 records budget exhaustion.
Source or configuration changes require a fresh output directory. Stage
commands support calibration, generation, fitting, evaluation and reporting.
The default eight-hour budget is an execution budget, not a promised accuracy.

## Next scientific gates

The T=1 first-batch design uses 256/64/256 train/validation/final paths per
case, width32/depth4, three optimizer seeds and step 2^-7: 15 fits. A full
three-step design is available as `path-pilot`, with a separate fit at each
resolution. Validation selection and final tests remain distinct.

After establishing T=1 behavior, prioritize temporal refinement with exact
coefficients and the neural operator on the same physical drivers. Audit
reference refinement independently before estimating an order. Then vary
independent training families and capacity separately. Stopping stress cases,
particle-count changes and particle comparisons remain later scientific
questions. Spatial resolution studies need an explicit continuum noise
regularization before being described as continuum convergence.
