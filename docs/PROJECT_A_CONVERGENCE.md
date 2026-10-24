# Project A: convergence experiments for a neural DK solver

Project A's first convergence claim concerns a **stopped, spatially discrete Dean–Kawasaki system at fixed grid and particle count**. It does not claim convergence to an unregularized density-valued DK equation. The solver learns autonomous coefficients and is then integrated at several timesteps; learning error and integration error are measured separately.

## Run the workflow

Use the locked Python 3.12 environment:

```bash
uv sync --locked --group dev
uv run nn-dk convergence --preset a-smoke --output runs/a-smoke
```

The equivalent file-based configuration is:

```bash
uv run nn-dk convergence --config configs/a_smoke.toml --output runs/a-smoke
```

Choose a preset or a configuration file. The shipped TOML files match their named presets. Use the same source, configuration, seeds, and output directory when resuming; provenance checks reject mixing different experiments. Outputs are local run artifacts rather than source files.

| Preset | Purpose | Main allocation |
|---|---|---|
| `a-smoke` | Exercise the complete workflow | 8 cells, 128 particles, horizon 1/8; 8/16 training families, 32 diagnostic paths, three epochs |
| `a-first-batch` | Numerical calibration and first coefficient study | Uniform noninteracting case, 32 cells, 2048 particles, horizon 1; 64 training families, width 32, three optimizer seeds |
| `a-full` | Learning, capacity, temporal, and supervision studies | Three cases, 64/256/1024/4096 training families, widths 16/32/64, oracle and increment modes, 8192 diagnostic paths |
| `a-final` | Selected high-precision follow-up configuration | Same study axes as `a-full`, with 65,536 diagnostic paths |

The scientific presets allocate **eight hours per invocation**. This is a resumable execution budget, not a promise that a full study finishes in eight hours. The first batch prioritizes A0 and the uniform oracle-supervised case; larger learning and final-precision studies can require many batches. Checkpointed training retains optimizer and RNG states. Reusing an output directory after changing source or configuration is intentionally rejected; use a new run directory.

```bash
uv run nn-dk convergence --preset a-first-batch --output runs/a-first-batch
uv run nn-dk convergence --preset a-full --output runs/a-full
uv run nn-dk convergence --preset a-final --output runs/a-final
```

## The native target

On the periodic torus of length \(L=2\pi\), let \(h=L/G\) and let \(m_i\) be a cell probability mass. The scientific defaults are \(G=32\), \(N=2048\), \(D=1\), and \(T=1\). The target is

\[
dm=b_h(m)\,dt+N^{-1/2}B\,\mathrm{diag}(\sigma_h(m))\,dW,
\qquad
\sigma_{h,i}(m)^2=\frac{2D}{h}\rho_{i+1/2},
\]

where \((Bf)_i=f_{i-1}-f_i\), \(\rho_{i+1/2}=(m_i+m_{i+1})/(2h)\), and the deterministic flux uses centered diffusion and the cell-integrated interaction force. Explicit stepping checks the diffusion/advection stability condition. Initial masses are **raw IID particle histograms**, with no renormalization or replacement by a deterministic profile.

The three cases are:

1. Uniform initial particles with no interaction.
2. IID initial density \(p_0(x)\propto e^{0.2\cos x}\), with no interaction.
3. The same nonuniform initial law with \(V(x)=-0.75\cos x+0.1\cos2x\).

Stopping is part of the target law. Stop when the reconstructed piecewise-constant density first reaches \(\rho_*=0.01/(2\pi)\). A numerical step follows its Euler polygon to the first crossing, freezes that boundary state, and records the interpolated stopping time. A path initially below the boundary stops at zero. **Every absorbed path remains in all ensemble estimates.** This operation defines the stopped law; it does not repair a negative proposal, average survivors, or assert positivity of an unstopped Gaussian scheme.

The native state is the complete finite-volume mass vector. It is distinct from the legacy spline-projected particle state. Target metadata records the grid, histogram initialization, state representation, and first-crossing convention. Native and legacy checkpoint/target identifiers are incompatible by design.

Evaluation uses fixed physical functions \(\sqrt2\cos(kx)\) and \(\sqrt2\sin(kx)\), \(k=1,2\). They are integrated exactly against each reconstructed cell density; the cell averages contain the factor \(\operatorname{sinc}(k/G)\). Queries occur at \(t=1/4,1/2,1\) for scientific presets. Thus the test-function definition does not change with the grid. The legacy B-spline observation map remains available for the original particle benchmark.

## Coefficients and training data

`DKCoefficientModel.coefficients(m, physics)` returns a conservative drift and covariance independently of timestep:

\[
b_\theta=Bj_\theta,\qquad
A_\theta=B\,\mathrm{diag}(\sigma_\theta^2)B^\top.
\]

The three-layer periodic CNN uses current masses, physical parameters, and current-state Fourier information. Initial-profile coefficients and elapsed time are absent from this autonomous generator. The positive diagonal face factor contains the native DK noise structure. Computations use double precision, and there is **no fixed covariance floor** in this track. Particle number enters the integrator's explicit \(N^{-1/2}\) factor.

Two objectives use the same underlying trajectory-family pools:

- **Oracle supervision** fits known native drift and covariance on simulated pre-stop states.
- **Increment supervision** fits the tangent-space Gaussian likelihood of raw Euler proposals, without coefficient labels. The pool stores proposals before applying stopping, including proposals crossing the boundary. Only whether the origin is active determines training eligibility; the proposal's outcome never selects surviving training increments. Absorbed increments are not fitted as ordinary Gaussian draws.

Training/validation/coefficient-test families are disjoint. Each scientific training family supplies 32 states at fixed physical times. The data ladder uses nested prefixes of 64, 256, 1024, and 4096 **independent parent trajectories**, rather than counting correlated state samples as independent paths. Coupled refinements and derived samples stay in the same family. Oracle provenance is stored separately from inference features; increment fitting does not consume the oracle coefficient labels.

At width 32, increase training families while holding architecture fixed. At the largest dataset, vary widths 16, 32, and 64 while holding data fixed. Increment supervision initially uses the largest dataset and base width. Keep optimizer seeds 0, 1, and 2 separate. Start with AdamW, learning rate \(10^{-3}\), batch size 128, at most 100 epochs, and validation patience 10. These allocations measure learning and capacity independently rather than conflating the two.

## Experimental ladder

| Stage | Measurement | Interpretation |
|---|---|---|
| A0 | Linear Gaussian matrix-exponential moments, native DK refinement, conservation, initialization, noise and stopping checks | Validate numerical implementation before interpreting neural errors |
| A1 | Oracle/oracle, learned drift/oracle noise, oracle drift/learned noise, and both learned | Locate drift or covariance approximation error |
| A2 | Nested training-family curve at width 32; separate width curve at largest data pool | Measure data improvement or approximation/optimization plateaus |
| A3 | Freeze a checkpoint and integrate it over the timestep ladder; repeat with exact coefficients | Distinguish temporal discretization from a neural error floor |
| A4 | Increment supervision on selected configurations | Measure additional statistical estimation error |
| A5 | Compare DK and NN with particle observables at selected configurations | Distinguish neural error from DK-to-particle discrepancy |

The scientific timestep ladder is

\[
\Delta=2^{-7},2^{-8},2^{-9},2^{-10},
\qquad \delta=2^{-13},\delta/2,\delta/4.
\]

Fine Brownian face increments are summed to construct coarse increments, and coupled paths share their initial histograms. Canonical positive diagonal face factors define the declared coupling for the neural/oracle comparisons. This reduces variance for **weak-statistic contrasts** and does not itself establish strong convergence. Family-aware bootstrap resampling preserves this coupling; genuinely independent pools are resampled independently and may not overlap in IDs.

For each query statistic, retain the decomposition

\[
q_{\theta,\Delta}-q_{h,\delta}
=\underbrace{q_{\theta,\Delta}-q_{h,\Delta}}_{\text{neural coefficient error}}
+\underbrace{q_{h,\Delta}-q_{h,\delta}}_{\text{time discretization difference}}.
\]

Both components, the total contrast, reference refinement, and sampling uncertainty must be visible. Do not retrain a network for each integration step when constructing a temporal curve. Stopping-boundary effects can alter the observed weak order, so first order is not assumed.

## Reports and acceptance

Saved artifacts include run/target/configuration/source metadata, HDF5 trajectory-family pools and raw proposals, resumable coefficient checkpoints, coefficient diagnostics, and report JSON/Markdown/PNG. The report presents learning, width, timestep, stopping, and cost curves. It retains optimizer seeds separately; seed ranges are not Monte Carlo confidence intervals.

At each physical query time measure:

- \(\sqrt N\)-scaled mean errors, \(N\)-scaled variances, and every two-time covariance pair.
- Scalar empirical Wasserstein distances, stopping probabilities, and stopping-time distributions.
- Conservation, minimum mass, drift/covariance errors, and training/inference costs.
- For increment fitting, validation likelihood and whitened tangent-space residual diagnostics.

Normalization is fixed analytically by uniform-IID histogram fluctuations: each mode's standard deviation is \(\operatorname{sinc}(k/G)\), and covariance units are products of these standard deviations. Held-out outcomes do not define normalization or center a discrepancy away.

Whole-family pointwise 95% bootstrap intervals distinguish finite-ensemble sampling error from training variation and reference bias. The default normalized-error target is **5%**, with each sampling/reference contribution targeted below **1.25%**. Diagnostic ensembles start at 8192 paths; final selected comparisons can use 65,536. Counts do not guarantee resolution. Scalar Wasserstein distances are descriptive unless separately equipped with a justified uncertainty analysis.

Finite-level reference differences are diagnostic evidence, not automatically a certified bias bound. An accuracy gate remains **unresolved** unless its contrast intervals, individual-pool precision, and separately justified reference bias all meet their budgets. A temporal slope is reported only for at least three distinguishable decreasing refinement levels, each above three contrast standard errors and the reference budget. Otherwise report a plateau or unresolved rate. Passing software tests is compatible with an unresolved or negative scientific result.

Cost records separate generation, training, initialization, prediction, and total wall time. Shared training/data costs repeated across ablations are not additive. Compare methods at matched resolved accuracy before claiming speedup; an undertrained fast rollout does not demonstrate a solver advantage.

Paired accuracy audits also construct fine Brownian increments for common-noise comparisons. Their recorded initialization and total costs include this overhead. Separate single-path latency and batched throughput measurements generate the neural solver's own coarse noise; their timestep and coefficient ablation are recorded explicitly. Evaluation records are committed individually before building the aggregate report, so resuming a large study avoids repeatedly rewriting the complete result history.

After fixed-grid calibration succeeds, examine particle counts 512, 2048, and 8192, retaining stopping effects. Spatial refinement at fixed particle count is a resolution study until a separate noise/regularization target is specified. Particle-law discrepancy is not neural discretization error. Memory, non-Gaussian generators, and continuum convergence remain later research branches.

A5 is implemented as `nn_dk.particle_audit.audit_particles` and runs from the full workflow only after the selected fit's same-step neural contrast is within tolerance. It reports direct continuous particle observables, histogram projection error, native time error, and neural error separately. The particle law is unstopped; stopping contributions to the native discrepancy remain visible. Further mathematical comparison must check the regularization assumptions in [Djurdjevac, Ji, and Perkowski's weak-error framework](https://arxiv.org/abs/2502.20929).
