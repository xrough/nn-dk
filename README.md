# NN-DK

Neural numerical experiments for Dean–Kawasaki dynamics and coarse particle systems.

**The structured neural moment solver met the 5% accuracy target on 25 October 2026.** All 15 final comparisons passed, covering three optimizer seeds, an OU control, and conditional and ensemble DK moments. The largest error bounds were **0.1373% for means** and **0.180% for full same-time covariance**, including the independent reference error. This result concerns observable statistics of a fixed-grid stopped system; density-path simulation remains a separate track.

## Current result

The DK target is the **continuous-time, noninteracting finite-volume system** on the periodic interval `[0, 2π)`, with **32 cells, N=2048, D=1, T=1**. Paths freeze at the first continuous hit of density `0.01/(2π)`. Initial masses are IID particle histograms from the uniform or normalized `exp(0.2 cos x)` profile.

We tested **32 fresh initial histograms per profile**, **1024 fresh OU states**, and the known IID ensemble laws at times **1/4, 1/2 and 1**. The four observables are exactly cell-integrated normalized sine/cosine modes 1 and 2. All six fits were locked before generating final inputs; 512 separate development histograms supported the preceding development check. The final states come from the specified profile laws, not new physics or profiles.

The table retains every optimizer seed. DK rows take the maximum across both profiles and conditional/ensemble outputs; the [full report](docs/experiments/generator_accuracy_2026-10-25.md) retains all 15 comparisons.

| Problem | Seed | Maximum mean bound | Maximum full-covariance bound | Result |
|---|---:|---:|---:|---|
| DK | 0 | 0.0567% | 0.0839% | Pass |
| DK | 1 | 0.1373% | 0.0765% | Pass |
| DK | 2 | 0.0834% | 0.1258% | Pass |
| OU | 0 | 0.0221% | 0.0869% | Pass |
| OU | 1 | 0.0748% | 0.0100% | Pass |
| OU | 2 | 0.0492% | 0.1797% | Pass |

Every absolute normalized error **plus its reference bound** must be ≤5%; each reference contribution must also be ≤1.25%. The largest final reference contribution was **0.0628%**. Means use `sqrt(N)` scaling and covariance uses `N` scaling, divided by fixed observable scales. These percentages are not relative errors divided by possibly zero means or covariance entries. Acceptance checks every entry, including off-diagonal covariance; RMSE and best-seed selection cannot substitute for it. Displayed table values are rounded upward.

The reference independently solves the full native degree-two moment equations and bounds the correction from continuous stopping. It encloses numerical, geometry and profile-integration errors. Ensemble initialization uses exact known multinomial moments, so reference Monte Carlo uncertainty is absent. See the [certificate derivation](docs/GENERATOR_REFERENCE_CERTIFICATE.md) and [independent mathematical audit](docs/GENERATOR_REFERENCE_AUDIT.md) for the arithmetic assumptions and audited scope.

## What changed

The network builds linear mean dependence and translation-equivariant covariance dependence on initial masses into its architecture. A three-hidden-layer, width-32 SiLU network learns time propagators and positive-semidefinite covariance kernels. Training uses **known continuous-generator residuals**, with Adam followed by LBFGS. Known instantaneous forcing enters the residual and initialization; exact future moment forecasts are not supplied to the predictor.

This changes both the architecture and training objective from the earlier trajectory BSDE fits. It requires no stochastic training trajectories or auxiliary sampling noise. The known IID initialization law is integrated analytically. The [method guide](docs/GENERATOR_MOMENTS.md) states the model, loss, normalization and noise conventions.

The six fits took **86.19 seconds** in total; the initial end-to-end invocation took **219.50 seconds** on the local CPU. All exhausted their declared optimizer budget without attaining the stricter internal `1e-6` residual tolerance. They nevertheless passed the separately evaluated statistical accuracy gates. These timings do not establish a matched-accuracy neural speed advantage over classical moment methods.

This noninteracting problem has closed interior moment equations, making it a useful validation case. The result establishes the stated finite-bank moment accuracy, not a convergence rate, continuum DK limit, density-path accuracy, cross-time joint law, or accuracy for interacting dynamics. A useful next experiment is the interacting case, where this simple moment closure fails.

## Reproduce

Python **3.12** and the committed `uv.lock` define the environment.

```bash
uv sync --locked --group dev
uv run pytest
uv run nn-dk generator --config configs/generator_accuracy.toml \
  --output runs/generator-accuracy
```

Use `.venv/bin/pytest` and `.venv/bin/nn-dk` in the provisioned local environment if `uv` is not on your path. For a small execution check:

```bash
uv run nn-dk generator --preset generator-smoke --output runs/generator-smoke
```

The smoke configuration has two optimizer steps per phase and is not expected to meet scientific accuracy. A failed development gate leaves final inputs locked. `--stage` supports `calibrate`, `generate`, `fit`, `evaluate`, `report`, and `all`.

Rerun the identical command to resume; exit **75** means the invocation budget expired with resumable work. `--budget-seconds 600` changes that budget. Source/configuration checks and content hashes protect checkpoints, input arrays and cached evaluations. Use a new output directory after changing source or configuration.

The completed local experiment is in `runs/generator-accuracy-2026-10-25/`, including all six restorable checkpoints. An identical completed-run replay passed without changing the locked artifacts. Compact results, complete compressed report data, test inputs, reference packets and checksums are preserved in the [experiment archive](docs/experiments/artifacts/generator_accuracy_2026-10-25/README.md). Source freeze: **`49a6528`**; verification: **378 tests passed**, Ruff passed.

## Other experiment tracks

Earlier findings remain available and retain their original targets and unresolved conclusions:

- [Initial-state coverage and structured BSDE study](docs/experiments/bsde_anchor_study_2026-10-24.md): 36 fits; broader initial-state coverage helped, but 5% accuracy remained unresolved. [BSDE implementation](docs/BSDE_STATISTICS.md).
- [Autonomous coefficient-learning experiment](docs/experiments/a_first_batch_2026-10-24.md): native DK coefficient learning, ablations and time refinement. [Convergence workflow](docs/PROJECT_A_CONVERGENCE.md).
- [First trajectory BSDE batch](docs/experiments/bsde_first_batch_2026-10-24.md): conditional/ensemble errors and noise diagnostics.
- Original particle A/B benchmark: `uv run nn-dk run --config configs/smoke.toml --output runs/smoke`. [Numerical methods](docs/methods.md), [provenance](docs/provenance.md), [verification](docs/verification.md).

Native DK states, legacy spline observations, finite-lag transitions, BSDE value maps and continuous-generator moment maps have distinct target/model identifiers. Their checkpoints and accuracy claims are not interchangeable.
