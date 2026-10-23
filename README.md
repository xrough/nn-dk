# NN-DK

Conservative neural coarse dynamics for interacting Brownian particles, with a direct-statistics neural comparator and classical fluctuation baselines.

The primary target is the law of **smooth observations of finite particle systems** on the one-dimensional torus. Project A samples coarse paths at a fixed learned lag; Project B predicts a fixed bank of means and joint covariances. This release is a research benchmark, with no assumed neural accuracy or speed advantage.

## Install and run

Python **3.12** is required. The exact environment is recorded in `uv.lock`.

```bash
uv sync --locked --group dev
uv run pytest
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

## Model and shared target

The reference system is

\[
dX_i=-N^{-1}\sum_j V'(X_i-X_j)dt+\sqrt{2D}\,dB_i,
\qquad V(x)=-\kappa\cos x+a_2\cos2x.
\]

The default is (D=1,N=512,M=32). Initial particles are sampled independently from a three-mode exponential Fourier profile using exact rejection sampling. Forces use empirical trigonometric moments in (O(NK)) work.

The learned state is the vector of periodic cubic B-spline masses, centered at (j2\pi/M). Particle observations and piecewise-constant field observations use the same spline functions. Finite-volume masses are projected by integrating those functions over each cell; they are never identified directly with the learned state.

The shared observable vector contains normalized sine/cosine modes 1 and 2 at three positive times. Covariances are reported after multiplication by (N). Both neural methods receive the initial-profile descriptor and known physics. Project A additionally receives its present coarse state and elapsed time.

## Methods

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

Gaussian transitions can produce negative masses. A raw mass below (-10^{-12}) invalidates a trajectory. Paths are stopped without clipping or renormalization. **One failed path makes the corresponding unconditional ensemble estimate unavailable.** Reports show failure rates and attempted minimum masses; surviving paths are never silently averaged.

## Data, reproducibility, and cost

HDF5 stores whole paths, seeds, trajectory identifiers, completion markers, costs, and grouped context splits. Oracle provenance and learner inputs occupy separate groups. Training normalization is fitted on training contexts only. Checkpoints contain optimizer state, shuffle/RNG state, selected model state, supported particle counts, and training-data fingerprints.

Reports distinguish reference Monte Carlo intervals, online sampling intervals for simulators, optimizer-seed variation, and numerical calibration. Pointwise bootstrap intervals are not simultaneous coverage guarantees. The free-particle oracle uses the actual filtered spline observables, including aliases, rather than bare particle Fourier modes.

Cost records include data generation, training, initialization, complete-query latency, and sampling throughput; Neural B also has a vectorized multi-context timing. Each standalone learned method is charged the shared training/validation pool. The benchmark's independent reference audit is recorded separately. The combined mean-field/linearized computation is a declared upper timing bound for each of those methods. Online timing differences do not establish matched-accuracy end-to-end savings.

## Configurations and boundaries

- `smoke.toml`: 8 training, 2 validation, and 2 test contexts; eight replicas each; three training epochs; one optimizer seed.
- `pilot.toml`: 64×16 training paths, 16×32 validation paths, 8×512 reference test paths; three optimizer seeds.
- `n2048.toml`, `resolution64.toml`, `long-horizon.toml`: separately trained follow-ups. Transfer and extrapolation are not inferred from their names.

The discrete DK baseline uses its own specified spatial/noise approximation. Initialization, observation, and spatial projection discrepancies are reported separately. The numerical calibration gate must pass before training. Research certification can remain unresolved even when software checks pass.

Memory closures, non-Gaussian generators, unknown mechanisms, noise-conditioned continuum solution operators, rare events, and two-dimensional systems are later branches.

See [numerical conventions](docs/methods.md), [legacy provenance](docs/provenance.md), and [verification](docs/verification.md).
