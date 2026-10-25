# Analysis of completed Project A runs

## Causal full-density path operator

```bash
.venv/bin/python analysis/plot_density_paths.py runs/path-first-batch
.venv/bin/python analysis/benchmark_path_solver.py --run runs/path-first-batch \
  --output runs/path-first-batch/report/path_timings.json
.venv/bin/python analysis/archive_density_paths.py runs/path-first-batch \
  docs/experiments/artifacts/path_first_batch
```

These tools require completed models and unchanged source/configuration. The
plot selects trajectory zero, the first declared optimizer seed and coarsest
step before reading values. It shows actual full-grid density paths and their
same-noise Euler differences. The archive verifies the final artifact lock and
copies lossless compressed report data, checkpoint metadata and hashes, plots,
environment and execution records; large input/path banks and complete binary
checkpoints remain in the original local run.

Run timing only after other training/evaluation jobs have finished. It uses
validation inputs, not final inputs, and measures full warm trajectories at
P=1 and P=64 with randomized method order. Native NumPy Euler and Torch Euler
are distinct backend comparisons. Checkpoint restoration and IID histogram
initialization are timed separately. Inference excludes generation, loading,
training, bootstrap and file I/O. The accuracy check concerns a strong point
error on those validation inputs, not a joint weak or continuous-time gate.

`fit_path_seed_batch.py` is an optional manual execution helper, not a concurrent
scheduler. **Stop the sequential CLI first**, finish its control-development
caches, and assign exactly one process per declared optimizer seed. Wait for
every worker process to exit before resuming the identical original CLI. The
CLI does not consult worker lock files. Preserve previous execution records
before any worker restart. Workers reuse the same fitting implementation and
existing physical inputs; they never unlock final inputs or finalize reports.
The original CLI validates every completed fit before final evaluation.

Worker wall intervals overlap and are absent from the CLI manifest's recorded
elapsed time. Preserve their execution records and helper SHA; distinguish
UTC wall span, recorded monotonic execution duration, summed per-fit durations,
and worker process CPU time. Use the remaining monotonic execution budget when
resuming; a UTC gap alone does not establish active-budget exhaustion. The ordinary sequential
CLI is sufficient for reproduction and avoids this manual ownership protocol.

`finish_path_batch.py` is the one-shot continuation for the recorded three-seed,
15-fit batch. It waits for actual process exit and released locks, checks the
frozen helper/source/configuration hashes and restores every fit before resuming
the original workflow. It then saves density plots, separate warm timings and
the compact archive. `--check-only` reads the gate without launching evaluation.
Its frozen plan is in that run's `execution/parallel_plan.json`; this helper is
not a recurring scheduler. Permission to inspect the recorded PIDs is required.

The [path method](../docs/PATH_SOLVER.md) and
[experiment findings](../docs/experiments/path_solver_2026-10-25.md) define the
new target. The following older analyses concern distinct experiment tracks.

## Autonomous coefficient solver

Run these commands from the repository root, using its locked Python environment:

```bash
.venv/bin/python analysis/summarize_first_batch.py runs/a-first-batch
.venv/bin/python analysis/probe_uniform_generator.py runs/a-first-batch --plot
.venv/bin/python analysis/time_native_solver.py runs/a-first-batch --include-neural
```

The summary reads saved JSON only. It checks completion and the exact declared allocation, preserves individual optimizer seeds, and writes `report/findings_summary.json` and `report/findings_table.csv`. Whole-ensemble uncertainty, paired contrasts, reference bias, and overall accuracy gates remain distinct. It never fits or resimulates a model.

The uniform-state probe loads completed coefficient checkpoints and differentiates their generators. Its Fourier eigenvalues and frozen-noise covariance describe a local interior linearization. They do not certify nonlinear rollouts, stopped dynamics, or continuum convergence. Source, configuration, and checkpoint hashes identify the calculation.

Standalone timing requires a completed primary run and matching numerical source. Run it after other CPU work has finished. It measures three repeats at each declared native timestep and, optionally, every complete neural checkpoint at the finest step. Costs include histogram initialization, internal random-number generation, and the current shared simulation backend. Neural rollouts also incur that backend's native-coefficient calculation before the neural coefficients override it. Model loading is recorded separately. These are specified-step timings, without an accuracy-matched speedup claim.

Numerical source and configuration must stay unchanged while an experiment runs. Derived analysis files do not change the package source hash. The saved experiment note links the compact archived findings; full trajectory caches and checkpoints remain under the ignored `runs/` directory.

## Conditional BSDE execution profile

```bash
.venv/bin/python analysis/profile_bsde_batch.py runs/bsde-batch-profile
```

This independent diagnostic uses the planned width 64, G=32, N=2048, horizon 1,
forward step 2^-7, and path batch 32. It fits one epoch on 64 initial anchors with
two continuations each, with 16 separate validation anchors and two continuations
each. It records pool generation/loading, epoch cost, parameter counts, checkpoint
loading, and three repeats of conditional value inference for one and 64 initial
states. These reduced allocations measure execution cost; they provide no
scientific accuracy or speedup conclusion. Timings exclude histogram generation
at inference because the initial masses are already available.

`profile.json` is immutable on replay and has its own configuration/source hashes.
Run it without competing CPU work. The full configured BSDE workflow and its
statistical targets are documented in [the guide](../docs/BSDE_STATISTICS.md).

## Conditional BSDE findings

After the complete scientific run finishes:

```bash
.venv/bin/python analysis/summarize_bsde_batch.py runs/bsde-first-batch
```

The script reads completed JSON artifacts and hashes checkpoint bytes without
loading executable checkpoints, fitting, or simulating paths. It validates the
frozen/current package source, configuration content and hash, every declared
problem/optimizer seed, checkpoint and individual-record hashes, training and
reference allocations, stopping/conservation diagnostics, and signed neural/time
error decomposition. It writes `report/findings_summary.json` and
`report/findings_table.csv`. `--output PATH` selects another analysis directory.
Incomplete runs are rejected before aggregate outputs are read.

Validation concerns immutable metadata and checkpoint bytes; HDF5 training traces
and reference caches are not reread. Each optimizer seed remains separate.
Conditional intervals quantify continuation sampling within fixed initial-state
anchors; ensemble intervals also include initial-histogram sampling. Intervals
are pointwise, and finite DK reference refinements do not certify remaining
continuous-time bias. Same-time variances and cross-mode covariances are separated;
no joint temporal covariance is implied. Signed total errors equal neural plus
forward-time contrasts, while their RMSE magnitudes are not additive. Shared data
and reference-audit costs must not be summed across seeds, and the paired audit is
not a standalone matched-accuracy timing comparison.

Completed BSDE runs also support the following diagnostics:

```bash
.venv/bin/python analysis/audit_bsde_references.py runs/bsde-first-batch --case all
.venv/bin/python analysis/probe_bsde_ou_generalization.py runs/bsde-first-batch
.venv/bin/python analysis/assess_bsde_dk_results.py runs/bsde-first-batch
.venv/bin/python analysis/plot_bsde_findings.py runs/bsde-first-batch
```

The reference audit reads completed NPZ banks and checks declared family counts,
IDs, disjoint stream keys, finite observations, stopping inclusion, and recorded
conservation diagnostics. It does not reread full mass histories or Brownian
increments, certify neural accuracy, or establish remaining reference bias. The
OU probe uses the fixed analytic test bank and completed checkpoints to inspect
local errors and the finite-step martingale objective floor. The DK assessment
reads completed evaluation JSON, separating conditional from ensemble errors,
variance from cross-mode covariance, and pointwise interval violations. These
posthoc diagnostics do not alter training, refit models, change test allocations,
or justify tuning a new method on the final test bank. They may explicitly report
an incomplete optimizer-seed allocation; the strict findings summary still
requires the entire primary run to finish.

The supplemental findings plot places conditional and ensemble errors side by side,
including variance-only errors and maximum pointwise discrepancies. It requires a
complete allocation and checks source/configuration/checkpoint hashes. It does not
fit, simulate, or turn the plotted point estimates into confidence bands.
