# NN-DK

Conservative neural stochastic solvers for Dean–Kawasaki dynamics and coarse interacting-particle systems.

The **Project A convergence workflow** learns autonomous drift and noise coefficients for a stopped, spatially discrete DK system. It separates coefficient approximation, time discretization, reference bias, and Monte Carlo uncertainty at a fixed grid. A convergence rate or neural speed advantage is an experimental result, not an assumption.

The original particle-observation benchmark remains available: its Project A samples smooth coarse observations at a fixed learned lag, while Project B predicts means and joint covariances. Native DK coefficient checkpoints and particle-observation checkpoints have distinct target/model identifiers.

The **conditional Deep BSDE workflow** learns observable means and same-time covariances using known forward dynamics. Its value network supplies predictions without a future-path rollout; a separate martingale network is used during training. This is the first stage toward comparing learned SPDE paths with directly learned statistics. See [the implementation and noise conventions](docs/BSDE_STATISTICS.md).

The [first scientific BSDE batch](docs/experiments/bsde_first_batch_2026-10-24.md) completed with substantial conditional prediction errors. Its findings distinguish conditional accuracy, ensemble averages, variance errors, and execution cost.

## Install and run

Python **3.12** is required. The exact environment is recorded in `uv.lock`.

```bash
uv sync --locked --group dev
uv run pytest
uv run nn-dk convergence --preset a-smoke --output runs/a-smoke
uv run nn-dk bsde --preset bsde-smoke --output runs/bsde-smoke
```

The A-only smoke workflow exercises numerical calibration, both coefficient-training objectives, ablations, learning/capacity axes, temporal refinement, and reporting. Its small ensembles and three training epochs validate execution; they cannot certify 5% fluctuation accuracy. The scientific configuration starts with:

```bash
uv run nn-dk convergence --preset a-first-batch --output runs/a-first-batch
uv run nn-dk convergence --config configs/a_first_batch.toml --output runs/a-first-batch
```

These two commands select the same configuration. Rerun the identical command to resume the same run. `a-first-batch` allocates an eight-hour invocation budget to numerical calibration and the uniform, noninteracting oracle-supervised case. `a-full` adds the three cases, nested data and width studies, increment supervision, and particle-number follow-ups; `a-final` increases diagnostic ensembles to 65,536 paths. Use a fresh output directory when changing configuration or source. See [the Project A convergence guide](docs/PROJECT_A_CONVERGENCE.md) for the mathematical target, experiment ladder, and interpretation of unresolved results.

The first scientific batch completed on 24 October 2026: **40 comparisons, three optimizer seeds, and 8192 evaluation families per configuration** at G=32, N=2048, T=1. Finest-step normalized covariance RMSE against the fine DK reference was 0.200%, 0.178%, and 0.449%. Refining the time step exposed a remaining neural coefficient error, dominated by drift in the ablations. Individual sampling precision and continuous-time reference bias remain unresolved, so this is not a convergence certificate. Read [the experiment findings](docs/experiments/a_first_batch_2026-10-24.md) for calibration, uncertainty, stopping, costs, saved artifacts, and follow-up decisions.

Run the original combined A/B benchmark with:

```bash
uv run nn-dk run --config configs/smoke.toml --output runs/smoke
```

This checkout also has a provisioned local environment. Without a global `uv` command, use `.venv/bin/python -m pytest` and `.venv/bin/nn-dk` directly, or invoke `.venv/bin/uv`.

The smoke run has 12 contexts and 8 independent particle trajectories per context. It trains both neural models for three epochs, fits classical regressions, evaluates every method, and writes Markdown, strict JSON, and a PNG accuracy–cost figure under `runs/smoke/report/`. Small ensembles support software validation; their confidence intervals do not certify 5% statistical accuracy.

Run stages independently against the same saved configuration:

```bash
uv run nn-dk calibrate --config configs/pilot.toml --output runs/pilot
uv run nn-dk generate --output runs/pilot
uv run nn-dk fit --output runs/pilot
uv run nn-dk evaluate --output runs/pilot
uv run nn-dk report --output runs/pilot
```

Or run all stages with `uv run nn-dk run --config configs/pilot.toml --output runs/pilot`.

Each invocation has an eight-hour default budget. `--budget-seconds 600` sets a shorter invocation budget. Expiration is checked between atomic trajectory, epoch, or evaluation units, so a running unit can overrun the boundary. Exit code **75** means work was checkpointed; rerun the identical command to continue. Configuration and source hashes prevent mixing results from different implementations. Use a new output directory after changing source code. Evaluation records already written are reused without resimulation.

## Legacy particle benchmark: model and target

The reference system is

\[
dX_i=-N^{-1}\sum_j V'(X_i-X_j)dt+\sqrt{2D}\,dB_i,
\qquad V(x)=-\kappa\cos x+a_2\cos2x.
\]

The default is (D=1,N=512,M=32). Initial particles are sampled independently from a three-mode exponential Fourier profile using exact rejection sampling. Forces use empirical trigonometric moments in (O(NK)) work.

The learned state is the vector of periodic cubic B-spline masses, centered at (j2\pi/M). Particle observations and piecewise-constant field observations use the same spline functions. Finite-volume masses are projected by integrating those functions over each cell; they are never identified directly with the learned state.

The shared observable vector contains normalized sine/cosine modes 1 and 2 at three positive times. Covariances are reported after multiplication by (N). Both neural methods receive the initial-profile descriptor and known physics. Project A additionally receives its present coarse state and elapsed time.

## Legacy particle benchmark: methods

| Method | Behavior |
| --- | --- |
| Particle MC | Independent particle paths at the reference timestep |
| Mean field | Deterministic finite-volume mean; zero fluctuation covariance |
| Linear fluctuations | Mean-field linearization, Gaussian forcing, IID initial covariance, and multitime cross covariances |
| Discrete DK | Centered conservative fluxes and shared Gaussian face noise |
| Autoregression | Gaussian linear transition in mass-zero coordinates |
| Polynomial/kernel ridge | Direct mean and paired-covariance regressions |
| Neural A | Three-layer periodic CNN, conservative current and banded noise factor |
| Neural B | Three-layer MLP, means and a joint Cholesky covariance head |

Neural A uses a fixed lag (2^{-6}). Its likelihood is computed in (M-1) orthonormal tangent coordinates using double-precision Cholesky. A physically scaled covariance floor acts only on this subspace. Neural B uses disjoint independent replicate pairs for unbiased covariance labels, avoiding cancellation between two fitted order-one moments. Regression baselines use the same path pool and validation-only regularization selection.

For this legacy particle benchmark, Gaussian transitions can produce negative masses. A raw mass below (-10^{-12}) invalidates a trajectory. Paths are stopped without clipping or renormalization. **One failed path makes the corresponding unconditional ensemble estimate unavailable.** Reports show failure rates and attempted minimum masses; surviving paths are never silently averaged. The new native convergence target instead declares boundary stopping as part of its law and includes every absorbed path in its estimates.

## Data, reproducibility, and cost

HDF5 stores whole paths, seeds, trajectory identifiers, completion markers, costs, and grouped context splits. Oracle provenance and learner inputs occupy separate groups. Training normalization is fitted on training contexts only. Checkpoints contain optimizer state, shuffle/RNG state, selected model state, supported particle counts, and training-data fingerprints.

Reports distinguish reference Monte Carlo intervals, online sampling intervals for simulators, optimizer-seed variation, and numerical calibration. Pointwise bootstrap intervals are not simultaneous coverage guarantees. The free-particle oracle uses the actual filtered spline observables, including aliases, rather than bare particle Fourier modes.

Cost records include data generation, training, initialization, complete-query latency, and sampling throughput; Neural B also has a vectorized multi-context timing. Each standalone learned method is charged the shared training/validation pool. The benchmark's independent reference audit is recorded separately. The combined mean-field/linearized computation is a declared upper timing bound for each of those methods. Online timing differences do not establish matched-accuracy end-to-end savings.

## Configurations and boundaries

- `a_smoke.toml`: short A-only integration check; all three cases, oracle/increment training, tiny learning/capacity curves.
- `a_first_batch.toml`: fixed-grid scientific starting point; uniform case, oracle supervision, three optimizer seeds, eight-hour invocation budget.
- `a_full.toml`: nested 64–4096-path learning curves, widths 16/32/64, three cases, and increment-supervision comparison. These allocations can require multiple resumed batches.
- `smoke.toml`: 8 training, 2 validation, and 2 test contexts; eight replicas each; three training epochs; one optimizer seed.
- `pilot.toml`: 64×16 training paths, 16×32 validation paths, 8×512 reference test paths; three optimizer seeds.
- `n2048.toml`, `resolution64.toml`, `long-horizon.toml`: separately trained follow-ups. Transfer and extrapolation are not inferred from their names.

The discrete DK baseline uses its own specified spatial/noise approximation. Initialization, observation, and spatial projection discrepancies are reported separately. The numerical calibration gate must pass before training. Research certification can remain unresolved even when software checks pass.

Memory closures, non-Gaussian generators, unknown mechanisms, noise-conditioned continuum solution operators, rare events, and two-dimensional systems are later branches.

See [numerical conventions](docs/methods.md), [legacy provenance](docs/provenance.md), and [verification](docs/verification.md).
