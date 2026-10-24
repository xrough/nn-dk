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

Smoke results are under `runs/a-smoke/`. Scientific defaults remain G=32, N=2048, T=1, 8192 diagnostic paths, three optimizer seeds, and an eight-hour resumable invocation budget. The small smoke study establishes software acceptance, not a convergence rate, 5% accuracy certification, or neural speed advantage.

## First scientific Project A batch — 24 October 2026

The configured `a-first-batch` run completed successfully in **5548.91 seconds (92.48 minutes)**. The frozen package source hash is `865faee0d24b982db69fd1890ae2378ded36129f3c51d623232d2563eb710780`; configuration hash is `cd1627a2a1ea14eb64974f0f3aec7b330ef82c4709952f955905a56fb63bd4af`.

- A0 numerical calibration completed, including independent matrix/spectral heat moments and coupled native reference refinement.
- All **40 declared comparisons** and **three coefficient checkpoints** completed. Each evaluation configuration contains 8192 trajectory families; repeated methods and steps share those families.
- Finest-step total normalized covariance RMSE was 0.199933%, 0.178158%, and 0.448901% by optimizer seed. Same-step neural components were 0.183620%, 0.175678%, and 0.435076%.
- Every total paired statistical contrast satisfied the 5% pointwise tolerance check. All overall accuracy gates remain **unresolved**: individual covariance sampling halfwidths are approximately 3.3%, above the 1.25% budget, and finite reference refinements do not certify remaining continuous-time bias.
- No stops occurred in the evaluated ensembles; every path was retained. Maximum prediction saved-state mass error was 1.9984e-15. Zero stops do not validate boundary-sensitive behavior or imply zero stopping probability.
- Four new artifact-analysis tests pass; Ruff passes. They cover incomplete-run rejection, declared allocations, separate variance/two-time covariance diagnostics, and nonzero zero-event stopping bounds. The numerical package was unchanged from the previously tested source revision.

The [experiment note](experiments/a_first_batch_2026-10-24.md) documents the findings and limitations. Compact tables, figures, provenance, and timing records are archived beside it; full datasets and checkpoints remain under `runs/a-first-batch/`. Analysis commands are documented in [analysis/README.md](../analysis/README.md).

This completes the uniform, noninteracting oracle-supervised first batch. The larger data/capacity ladder, nonuniform and interacting cases, increment-supervised study, 65,536-path precision follow-ups, and particle comparisons have **not** been run at their scientific allocations.
