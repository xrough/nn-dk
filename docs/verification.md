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

## Conditional Deep BSDE implementation — 24 October 2026

The separate `nn-dk bsde` workflow predicts conditional means and same-time four-by-four covariances, then applies the law of total covariance for IID histogram starts. Known forward dynamics generate grouped training traces; value-network prediction requires no future rollout. Legacy particle, coefficient-learning and checkpoint formats remain distinct.

- Full suite: **161 tests pass**. Ruff passes.
- Saved `bsde-smoke`: numerical Brownian/OU calibration and coupled native-DK reference audit passed; both neural problems completed training, evaluation and reports in **2.4033 seconds** of recorded run time.
- The exact smoke uses G=32, N=2048, T=1/8, one optimizer seed, three training epochs, 16 training anchors with two continuations each, eight validation anchors with two continuations each, and 32 ensemble reference paths per DK profile. It is smaller than the configured scientific first batch.
- Tests cover predictable integrand inputs, shared forward/martingale increments, full crossing-step noise, uncensored proposal capture, time-zero and initially stopped outputs, positive covariances without a floor, anchor isolation, training-only normalization, complete-epoch resumption, target compatibility, immutable completed checkpoints, and rejection of incomplete or incompatible reports.
- Analytic OU moment errors distinguish the trained Euler law from continuous-time truth. Conditional and ensemble bootstrap tests preserve trajectory families and include variation between initial histograms. Numerical refinement retains every stopped path and never treats finite reference contrasts as certified bias bounds.
- A separate width-64, T=1 performance probe used 64 training anchors × two continuations, 16 validation anchors × two continuations, batch 32 and one epoch. The epoch including validation took **0.42390 seconds**. Conditional prediction at three horizons took median **0.51438 ms** for one precomputed state and **1.53692 ms** for 64 states, excluding histogram initialization/loading. It establishes execution cost only; its one-epoch model has no scientific accuracy claim.
- Offline wheel and source-distribution builds succeed with the BSDE modules. Completed saved-CLI replay and immutable checkpoint hashes pass.
- Saved smoke ensemble mean/covariance RMSEs were **10.32% / 12.85%** for uniform DK and **28.91% / 19.86%** for nonuniform DK. These undertrained predictions do **not** meet the 5% target. All DK accuracy gates remain unresolved; software acceptance is the only conclusion.

The [BSDE guide](BSDE_STATISTICS.md) gives APIs, noise conventions, staged commands and scientific allocations. The [saved smoke note](experiments/bsde_smoke_2026-10-24.md) records exact provenance, calibration, errors and costs, with compact archived results and figures. The full scientific first batch, forward-step refinement, interacting case, multitime covariance and pathwise SPDE formulation have **not** been run by this implementation smoke.

## First scientific conditional BSDE batch — 24 October 2026

The configured `bsde-first-batch` run completed in **3161.6944 seconds (52.6949 minutes)**, with all six OU/DK optimizer checkpoints, six model evaluations and **39 derived comparison rows**. The frozen package source hash is `7d24dcb7b6c80a1f6856ee08b379c8c46fb564188bc3c59eca1ce9debdcd61ef`; configuration hash is `20e7fdae25f659c999ba443fc8ab99cfe01c8fbb9347cb5d7d59b9c9d045aaf3`.

- G=32, N=2048, T=1, no interaction; uniform and exp(0.2 cos x) IID histogram starts. Each problem uses 128 training anchors × 16 continuations, 32 separate validation anchors × 32 continuations, width 64, and optimizer seeds 0, 1, 2. Conditional test banks contain eight anchors per profile × 2048 continuations; independent ensemble banks contain 8192 histograms per profile.
- Numerical Brownian/OU calibration and both 32-family native reference audits passed. Finest reference-refinement pointwise confidence endpoints were approximately 0.153% and 0.161% in fixed fluctuation units. Remaining continuous-time reference bias remains uncertified.
- OU same-step average mean/covariance RMSEs were approximately 1.36–1.52% / 1.81–2.07%, but individual mean/covariance errors reached 10.8–23.5% / 18.2–25.3%. Small terminal loss and average error did not establish uniform conditional accuracy.
- DK conditional mean RMSEs against the fine reference were **11.11–21.13%**; ensemble mean RMSEs were **1.03–4.75%**. Full ensemble covariance RMSEs of **3.55–5.75%** concealed variance-only errors of **6.35–11.13%**. Many pointwise intervals for errors against the training-step Euler law lay wholly outside the 5% band. Every overall DK accuracy gate remains unresolved; this is a negative conditional-generalization result for the tested allocation, not a convergence rate.
- Audits verified 49,152 distinct reference continuation families across four completed banks, with no stopping events. Conditional children share fixed anchors and do not count as independent initial-law samples. Saved-state mass error remained below 5.6e-15; all learned conditional covariance checks passed. Reference integrity, conservation and positive covariance do not certify predictive accuracy.
- Conditional value-head prediction was inexpensive on precomputed states, but whole ensemble prediction was dominated by histogram initialization with substantial recorded timing variability. Shared coupled audits and training are accounted separately. No matched-accuracy speedup is established.
- The numerical package is unchanged from the implementation tested by **161 passing tests**. Separately, **nine artifact-analysis tests** and Ruff pass. Strict completed-run extraction validates source/configuration hashes, every checkpoint byte hash, declared allocations, and signed neural/time error decomposition.

The [scientific experiment note](experiments/bsde_first_batch_2026-10-24.md) documents all seeds, uncertainty, stopping, cost variability, and follow-up priorities. Compact archived summaries, figures, OU and DK diagnostics, reference-bank audits and file hashes accompany it. Larger initial-anchor learning curves, a structural OU control, forward-step refinement, interacting DK, joint temporal statistics and a pathwise SPDE representation remain follow-up work.
