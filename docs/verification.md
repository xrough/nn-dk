# Verification workflow

```bash
uv sync --locked --group dev
uv run ruff check .
uv run pytest
uv run nn-dk convergence --preset a-smoke --output runs/a-smoke
uv run nn-dk run --config configs/smoke.toml --output runs/smoke
```

The tests cover analytic numerics, statistical contracts, independent data units, neural geometry, checkpoint resumption, reference-certification guards, failure handling, report generation, and the executable pipeline.

Calibration writes `calibration.json`. Its top-level `status` tests numerical implementation invariants; `reference_certification.status` separately records whether Monte Carlo and refinement uncertainties resolve the configured tolerance. A passed implementation check does not establish trained-model accuracy.

Each run records exact dependency versions, Python/platform information, source/configuration hashes, fixed test allocations, training history, and immutable per-method/per-context evaluation records. Repeating a completed run reuses those outputs. Rebuilding its report does not fit models or resimulate trajectories.

Tests should be run before a new source hash is frozen. Use a new run directory for scientific comparisons after code changes. Do not combine outputs across configuration hashes or silently replace failed methods by their surviving paths.

## Initial implementation evidence — 24 October 2026

- Locked Python 3.12 environment: offline `uv sync --locked --group dev` check succeeded for 30 packages, including PyTorch 2.14.1.
- Ruff: all checks passed. Pytest: **32 passed**; the repeated full run took 20.57 seconds.
- Configured smoke run: **96 independent reference paths**, both neural fits, nine methods across two test contexts, and **18 evaluation records**; total pipeline wall time approximately 25 seconds.
- Outcomes: 14 available comparisons, one unavailable neural-A ensemble after one of 16 generated paths became negative, and three explicitly invalid ridge covariance predictions. No clipping, covariance repair, or survivor-conditioned estimate was substituted.
- Calibration implementation checks passed. Reference certification remained **unresolved**, as expected for these small ensembles.
- Standalone wheel and source archive built successfully. Report generation and immutable-record replay are covered by the integration test.

The generated smoke artifacts are under the ignored `runs/smoke/` directory in this checkout. The larger eight-hour pilot is configured but has not been executed. These checks establish executable software and honest diagnostics, not a neural accuracy or speedup result.

## Native Project A implementation — 24 October 2026

The separate native workflow adds stopped finite-volume DK states, autonomous double-precision coefficient networks, oracle and uncensored-proposal objectives, and an A-only experiment runner. Legacy target and checkpoint formats remain separate.

- Full suite: **105 tests pass**, including the legacy tests. Ruff passes.
- A-only smoke: all three cases, 12 coefficient checkpoints, and 117 complete comparison records. It exercises both objectives, coefficient ablations, nested data and width studies, fixed-checkpoint timestep refinement, and reports. No Project B fit is invoked.
- Maximum observed mass-conservation error: **6.7e-16**. Stopping frequencies range from zero to one of 32 paths; every frozen path remains in every statistic.
- A0's independent matrix and spectral moment calculations agree. Coupled native reference refinements preserve initial samples and Brownian sums. Finite reference differences remain explicitly uncertified.
- Tests cover first-boundary interpolation, full native observations, raw crossing proposals, autonomous/equivariant inputs, covariance geometry without a floor, disjoint trajectory families, exact generation/training resume, incomplete-report rejection, and checkpoint round trips. Separate particle-audit tests exercise distinct target laws and retained stopping contributions.
- Offline wheel and source-distribution builds succeed and include all native modules.

Smoke results are under `runs/a-smoke/`. Scientific defaults remain G=32, N=2048, T=1, 8192 diagnostic paths, three optimizer seeds, and an eight-hour resumable invocation budget. The full scientific allocation has **not** been executed. The small smoke study establishes software acceptance, not a convergence rate, 5% accuracy certification, or neural speed advantage.
