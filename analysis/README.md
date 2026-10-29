# Analysis of completed Project A runs

## Continuing the README research tasks

The [task/evidence ledger](../docs/README_TASKS.md) separates completed historical
studies from the open priorities in [the findings README](../README_FINDINGS.md).
The next diagnostics use frozen v3 models and validation-state anchors, with
fresh continuation families. They do not fit a model or reopen historical final
outcomes, and their completion does not finish the proposed sampling or learned
Gaussian-correction studies.
Both diagnostics require an available C++ compiler (default `clang++`) for the
independent native reference, in addition to the locked Python environment.

```bash
.venv/bin/python analysis/audit_payoff_mean_gradients.py prepare --root runs/payoff-mean-gradients-2026-10-29
.venv/bin/python analysis/audit_payoff_mean_gradients.py run --root runs/payoff-mean-gradients-2026-10-29
.venv/bin/python analysis/report_payoff_mean_gradients.py --root runs/payoff-mean-gradients-2026-10-29
.venv/bin/python analysis/audit_payoff_residual_transport.py prepare --root runs/payoff-residual-transport-2026-10-29
.venv/bin/python analysis/audit_payoff_residual_transport.py run --root runs/payoff-residual-transport-2026-10-29
.venv/bin/python analysis/audit_payoff_residual_transport.py verify --root runs/payoff-residual-transport-2026-10-29
.venv/bin/python analysis/report_payoff_residual_transport.py --root runs/payoff-residual-transport-2026-10-29
```

Preparation freezes complete plans, checkpoint bytes, state-only anchor rules,
source/helper hashes and independent continuation streams. Both diagnostics
retain all six cases and three optimizer seeds by default. Rerun the same
`run` invocation to resume checksummed chunks; `--max-chunks` supports bounded
software checks in a **separate** output root. `--plan` on preparation accepts a
complete diagnostic-plan JSON. Changing code, allocation or input identities
requires a new root. Full local historical weights and state banks are required.

The mean-gradient audit computes every noise-face directional derivative using
native JVPs, including the derivative of killed-integral exposure at stopping.
It retains paired Euler refinements, a quadratic plug-in covariance source and
an independent-half cross-Gram source estimate. The latter avoids the plug-in's
positive Monte Carlo bias but may be indefinite. Signed componentwise bootstrap
contrasts are pointwise; positive Frobenius-distance ranges are **not** tests
of a nonzero discrepancy. Derivative interchange and remaining stopping/time
bias are not certified.

The transport diagnostic integrates mean and raw-second-moment residuals along
fresh native trajectories. The raw-second residual includes both mean-residual
cross terms and the accumulated killed running reward. It reports observed
Euler/generator/stopping defects, coupled time refinement and stride/half-stride
quadrature changes. A covariance residual alone is not an error certificate;
raw seconds and unbiased empirical covariance remain separate quantities.

`payoff_collocation_sampling.py` supplies deterministic boundary/time strata,
importance-corrected row proposals and parent-grouped synthetic stress states.
The importance weight multiplies each **per-row loss after** any nonlinear
robust transform; it is not inserted inside that transform or self-normalized.
Uniform and stratified proposals therefore target the same original-pool
objective. Synthetic transforms preserve mass and their parent-family IDs;
they are conditional probes, not independent physical trajectories. These
utilities still require a controlled, fresh-data fitting/evaluation study.

The [29 October diagnostic report](../docs/experiments/payoff_diagnostics_2026-10-29.md)
records the completed allocations and their limitations. Diagnostic metadata,
numeric contrasts and exact source snapshots are preserved in its compact
archive; raw native polygons, Brownian arrays, JVP chunks and weights remain local.

### Matched-pool sampling and loss comparison

`run_payoff_collocation_study.py` implements a separate, fresh six-regime study
with **72 fits**: three optimizer seeds crossed with uniform/stratified proposals
and pseudo-Huber/squared losses. All four variants share the literal state pools,
initial network weights and train-only scales within each case/seed. Every
variant selects on the same natural-occupancy validation MSE; the selected mean
is frozen before covariance fitting. No future-payoff labels enter this study.

```bash
.venv/bin/python analysis/run_payoff_collocation_study.py prepare --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/run_payoff_collocation_study.py fit --seed 0 --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/run_payoff_collocation_study.py fit --seed 1 --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/run_payoff_collocation_study.py fit --seed 2 --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/run_payoff_collocation_study.py lock --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/evaluate_payoff_collocation_study.py prepare --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/run_payoff_collocation_study.py freeze-evaluation --root runs/payoff-collocation-study-2026-10-29
```

Sequential fitting is sufficient. Parallel execution requires exactly one owner
per declared optimizer seed; source, pool, sampler and optimizer identities remain
fixed. Exit 75 preserves incomplete checkpoints for the identical invocation.
Full preparation/fit invocation wall and process-CPU costs are recorded alongside
their nonadditive components and overlapping worker spans.

The evaluation gate verifies every fitted model and the complete matched-input
proof, then freezes the evaluator/reporter source, criteria, all method/group/query
slots, and independent initial-family identifier plans before final reference
anchors or outcomes can be generated. Missing or stale plans/helpers fail closed.
Future evaluator/report implementation may be finished during fitting; changing
the frozen training runner, engine or sampler requires a new scientific root.

After locking evaluation, generate each of `T0p25_k0p75`, `T0p25_k3`,
`T0p25_k6`, `T1_k0p75`, `T1_k3` and `T1_k6` with:

```bash
.venv/bin/python analysis/run_payoff_collocation_study.py reference --case T0p25_k0p75 --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/evaluate_payoff_collocation_study.py evaluate --root runs/payoff-collocation-study-2026-10-29
```

Evaluation retains all 72 models, untrained controls, both Gaussian steps and an
independent native-MC control. Independent initial-law integration replaces the
shared-initial NN ensemble estimates in primary comparisons; paired-initial
results remain explicit diagnostics. Full allocation is 576 primary and 732
retained accuracy records. Do not change models after opening these outcomes.

Only after all fitting, reference, evaluation and other heavy CPU work has exited:

```bash
.venv/bin/python analysis/evaluate_payoff_collocation_study.py benchmark --final-timing --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/report_payoff_collocation_study.py --root runs/payoff-collocation-study-2026-10-29
```

The report requires all 696 literal timing workloads and 156 restoration rows;
it separates full-study costs, one-model deployment curves and shared preparation.
Pointwise uncertainty, reference refinement and accuracy-qualified payback remain
distinct. Software fixtures have passed the full fit/lock/reference/evaluation/
timing/report/archive workflow. **The scientific comparison remains unfinished
until its declared real allocation, audits and final archive are complete.**

### Supervised nonlinear-LNA correction and joint final-data gate

The separate R4 candidate reuses the existing nonlinear mean-field/linear-noise
ODE and learns corrections to selected conditional payoff moments. Its labels
are **supervised stopped-Euler moments**, not generator-residual targets. This
tests a new formulation; it is not an established accuracy or cost improvement.
Supplied numerical ODE outputs are data for the heads. Differentiating those
heads alone does not give the complete initial-state derivative of the ODE;
this candidate provides no generator-residual or input-sensitivity interface.
The analysis-only engine, runner, batched baseline and joint evaluator have
separate identities. Existing package APIs and frozen scientific helpers remain
unchanged.

The prescribed training allocation has all six `(T, κ)` cases and optimizer
seeds `(0, 1, 2)`, giving **18 fits**. For each case:

- Training uses 80 natural IID histograms, 40 per initial profile, plus 40
  synthetic stress states derived from the first 20 training parents/profile.
  Stress distances above the stopping mass are cyclically `.25`, `1`, `4`
  times the boundary width `.05/G`, constructed at remaining time `T` in the
  parent's minimum cell. They preserve mass and parent groups; they are not
  additional independent draws from the natural initial law.
- Validation uses 40 independent natural parents, 20/profile. The training
  mixture is explicitly two-thirds natural and one-third stress; it is not an
  importance-corrected natural loss. Validation selection uses natural rows.
- Every training/validation state has 256/512 independent continuations at
  Euler step `2^-12`. All three query times share each complete trajectory.
  The full label allocation is **307,200 continuation families**, retaining
  every stopped path with interpolated crossing and killed running rewards.
  Targets are sample means and unbiased `N`-scaled sample covariance.
- The baseline cache evolves each literal histogram's nonlinear conditional
  Gaussian ODE at step `2^-7`. Its zero initial covariance is conditional on
  that histogram. Analytic IID covariance propagation is not a replacement.
- Two separate width-32, depth-2 correction heads use 1,000 Adam updates and
  30 L-BFGS updates per stage. The selected mean is frozen before covariance
  fitting. Natural validation MSE and the physical payoff-bank units are fixed;
  no final-data hyperparameter or optimizer-seed selection is allowed.

The mean correction vanishes quadratically at zero remaining time. A moving
boundary gate enforces exact initially absorbed values. A time-ordered
congruence transforms the baseline covariance; zero heads recover the gated
baseline without a covariance floor or eigenvalue repair. Raw LNA, gated
zero-correction and all trained seeds remain separate controls. Deterministic
boundary flags and numerical failures are retained. These constraints do not
prove accuracy for the stopped nonlinear law or the continuum equation.

The independent namespace is `lna-supervised-correction-2026-10-30-v1`.
Training initial/forcing seeds are `2026103001/2026103002`; validation uses
`2026103004/2026103005`. Exact RNG addresses include the case, role, profile,
parent/anchor and continuation. Preparation freezes source, configuration,
parent groups and data identities before labels. Full label-generation,
baseline-cache, preparation and fit invocation costs are recorded.

```bash
.venv/bin/python analysis/run_lna_payoff_correction_study.py prepare --root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/run_lna_payoff_correction_study.py data --root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/audit_lna_supervised_labels.py run --root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/audit_lna_supervised_labels.py verify --root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/run_lna_payoff_correction_study.py fit --seed 0 --root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/run_lna_payoff_correction_study.py fit --seed 1 --root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/run_lna_payoff_correction_study.py fit --seed 2 --root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/run_lna_payoff_correction_study.py lock --root runs/lna-payoff-correction-2026-10-30
```

Only one process may own each data stage or optimizer seed. This training/data
runner has an eight-hour invocation budget and checkpoints on exit 75. Resume the identical
invocation; never restart a live worker because observation timed out.
Raw label chunks and full weights remain local in the ignored study root.
The replay audit reconstructs continuation zero at the first natural and stress
training anchors for both profiles in every case: 24 predetermined families in
the full study. It compares saved forcing hashes, stopping times and all three
payoff readouts against the independent package simulator, without new families
or access to final outcomes. A partial receipt cannot satisfy the joint gate.

**The joint comparison must lock all 72 collocation fits and all 18 correction
fits before any final reference or independent-IID outcome is opened.** The new
joint evaluator freezes both fit locks, the exact criteria and method inventory,
all evaluation/report source, future family-ID plans and their disjoint-stream
audit. It refuses already-materialized final outcomes. For this joint study,
generate the shared reference banks through its guarded wrapper; the standalone
collocation reference recipe above is not the joint unlock protocol.

After **both full fit allocations** complete and their worker processes exit:

```bash
.venv/bin/python analysis/run_payoff_collocation_study.py lock --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/evaluate_lna_payoff_correction.py prepare --root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/evaluate_lna_payoff_correction.py freeze --root runs/lna-payoff-correction-2026-10-30
```

Joint preparation creates and checks the original collocation evaluation packet
as well as the correction's plans. Only then generate **each of the six cases**
through the wrapper, substituting each case name for the example below:

```bash
.venv/bin/python analysis/evaluate_lna_payoff_correction.py reference --case T0p25_k0p75 --root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/evaluate_payoff_collocation_study.py evaluate --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/evaluate_lna_payoff_correction.py evaluate --root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/evaluate_lna_payoff_correction.py contrasts --root runs/lna-payoff-correction-2026-10-30
```

The planned final inventory retains the 576 collocation primary records and
216 new correction/control records. New IID-law predictions integrate **16,384
fresh histograms/profile**, independent of native references, by total
covariance. Both per-histogram raw LNA steps (`2^-7`, `2^-10`), gated zero heads
and all three trained heads are evaluated. Batched sparse/rank-four RK4 evolves
the nonlinear ODE separately for every histogram; it supplies the same-time
payoffs and no analytic-IID shortcut. Invalid rows stay aligned and make the
whole method/group unavailable; survivor averaging is forbidden.

Only after every heavy training, reference and accuracy process has exited,
timing executes one cold and one warm **complete uncached prediction** per
method/group: conditional batch one, or all 16,384 IID histograms with their
initialization, ODE, head and aggregation work. Cold and warm are separately
labelled observations, not a latency-distribution estimate. Baseline batch size
is fixed at 64. Accuracy-only shared caches cannot substitute for deployment
work. Full-study shared preparation and one-model deployment costs remain
separate; preparation components are not added again to full invocation costs.
All accuracy intervals, reference refinement and cost payback remain unresolved
until the declared computation, tests, audits and archive are complete.

`--final-timing` attests that other heavy work has exited. This includes the
entire 90-fit allocation and all reference/accuracy jobs; a software fixture
running alongside those jobs cannot make that attestation. Once the host is
quiet and all accuracy records are complete:

```bash
.venv/bin/python analysis/evaluate_payoff_collocation_study.py benchmark --final-timing --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/evaluate_lna_payoff_correction.py benchmark --final-timing --root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/report_payoff_collocation_study.py --root runs/payoff-collocation-study-2026-10-29
.venv/bin/python analysis/report_lna_payoff_correction.py report --root runs/lna-payoff-correction-2026-10-30
```

The correction report requires all 288 literal uninterrupted cold/warm jobs,
the complete all-seed accuracy inventory and the signed contrasts. CLI process
receipts distinguish whole invocation overhead from nested prediction timers.
Source, model, identifier or saved-array changes fail closed; incomplete cached
arrays are never silently overwritten. The report/compact archive still need
real full-allocation execution; software checks alone do not complete R4–R6.

After both reports and the current-source full-suite test receipt are complete,
create and independently verify each compact evidence archive:

```bash
.venv/bin/python analysis/report_payoff_collocation_study.py --root runs/payoff-collocation-study-2026-10-29 --archive docs/experiments/artifacts/payoff_collocation_2026-10-30
.venv/bin/python analysis/report_payoff_collocation_study.py --verify-archive docs/experiments/artifacts/payoff_collocation_2026-10-30
.venv/bin/python analysis/report_lna_payoff_correction.py archive --root runs/lna-payoff-correction-2026-10-30 --output docs/experiments/artifacts/lna_payoff_correction_2026-10-30
.venv/bin/python analysis/report_lna_payoff_correction.py verify-archive --output docs/experiments/artifacts/lna_payoff_correction_2026-10-30
```

Archive names identify the study allocation; their manifests retain the actual
execution dates, source identities and all optimizer seeds. Complete weights,
trajectory banks and prediction caches remain in the ignored local roots.

### Natural-validation residuals by boundary and time stratum

The supplementary residual audit uses every locked collocation model and the
same ordered natural validation rows used for selection. Its 19 strata depend
only on state and remaining time. It retains row and distinct-parent counts,
empty strata, signed component summaries, and each stratum's contribution to
the canonical natural MSE and pseudo-Huber loss. It separates native solver
units, physical payoff units and dimensionless train-scale units. Covariance
losses count all 16 matrix entries, including both off-diagonal entries.

An input-only check finds that five of the six validation pools contain only
the farthest distance bin. The strongest case has 11 rows in distance bin 1,
294 in bin 2 and 3,791 in bin 3; bin 0 is empty. Every case has 4,096 rows from
256 parent trajectories, and none has initially absorbed or zero-time rows.
Reweighting a natural pool cannot populate an absent boundary layer. The report
must retain that coverage limit, and it must not treat repeated rows from a
parent as independent physical samples.

The decomposition checks the canonical mean and covariance residuals without
changing weights. A separately labelled counterfactual substitutes the
same-seed uniform/pseudo-Huber model's covariance source by subtraction only;
it is not a fit with a common frozen mean. These are descriptive diagnostics
on selection data, with no confidence intervals, independent accuracy claim
or additional Monte Carlo families. Production execution requires the full
72-fit lock and freezes the audit/report sources before reading residuals.

After that lock is complete, use a separate diagnostic root:

```bash
.venv/bin/python analysis/audit_payoff_collocation_residuals.py prepare --study-root runs/payoff-collocation-study-2026-10-29 --root runs/payoff-collocation-residuals-2026-10-30
.venv/bin/python analysis/audit_payoff_collocation_residuals.py run --root runs/payoff-collocation-residuals-2026-10-30
.venv/bin/python analysis/audit_payoff_collocation_residuals.py verify --root runs/payoff-collocation-residuals-2026-10-30
.venv/bin/python analysis/report_payoff_collocation_residuals.py report --root runs/payoff-collocation-residuals-2026-10-30
```

Run the complete project Ruff and pytest checks against the final maintained
sources and retain their source-bound receipt and exact logs before archiving.
The supplement reuses that receipt from the correction study's `audit` folder:

```bash
.venv/bin/python analysis/report_payoff_collocation_residuals.py archive --root runs/payoff-collocation-residuals-2026-10-30 --output docs/experiments/artifacts/payoff_collocation_residuals_2026-10-30 --test-root runs/lna-payoff-correction-2026-10-30
.venv/bin/python analysis/report_payoff_collocation_residuals.py verify-archive --output docs/experiments/artifacts/payoff_collocation_residuals_2026-10-30
```

The compact supplement includes the component/stratum summaries, figures,
checksums, source and test proof. Raw per-row arrays, state banks and complete
weights remain local. Archive verification checks the copied bytes without
requiring those large local inputs.

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
new target. The sections below document this and other experiment tracks.

## Path and new-payoff generalization tools

The [broader generalization study](../docs/experiments/path_task_advantages_2026-10-27.md)
is complete. The independent reference contains 98,304 families; all six groups,
174 main evaluations, twelve Gaussian marginal audits and both final timing
experiments are complete. These scripts provide its frozen execution and
reporting workflow. They require the original full-history checkout and local
historical inputs under `runs/interacting-followup-2026-10-25/`: path and moment
checkpoints with metadata, configurations, and train/validation HDF5 banks. The
compact public archives do not contain these full inputs.

For that fixed study, prepare the shared payoff pools, finish all three optimizer
seeds, and lock all twelve fresh heads before generating held-out references:

```bash
uv run python analysis/run_payoff_generalization.py prepare --root runs/path-generalization-2026-10-26
uv run python analysis/run_payoff_generalization.py fit --seed 0 --root runs/path-generalization-2026-10-26
uv run python analysis/run_payoff_generalization.py fit --seed 1 --root runs/path-generalization-2026-10-26
uv run python analysis/run_payoff_generalization.py fit --seed 2 --root runs/path-generalization-2026-10-26
uv run python analysis/run_payoff_generalization.py lock --root runs/path-generalization-2026-10-26
uv run python analysis/run_generalization_evaluation.py --prepare-only --root runs/path-generalization-2026-10-26
uv run python analysis/benchmark_payoff_generalization.py --prepare-only --root runs/path-generalization-2026-10-26
uv run python analysis/benchmark_payoff_generalization.py --root runs/path-generalization-2026-10-26 --final-timing
uv run python analysis/audit_payoff_gaussian.py --prepare-only --root runs/path-generalization-2026-10-26
```

The last command preregisters the literal three-query Gaussian cost/quality
bridge before any `reference/reference.h5` outcomes exist. It freezes source,
configuration, helper, fit-lock, completed payoff timing, timing-plan and
reference-plan hashes; rerunning with a changed identity fails. Finish fitting
and all other CPU work before the payoff timing command. Keep this order:
preregistering the Gaussian audit before the payoff timing exists would freeze
a different identity.

After preparation, generate the full reference bank and complete every evaluation
group. Only after all fitting and other CPU work have stopped, run the final
path timing. `--final-timing` records the caller's confirmation of this condition;
it does not detect external workers. Preserve the already completed payoff
timing frozen into the Gaussian audit.

```bash
uv run python analysis/path_generalization_reference.py \
  --directory runs/path-generalization-2026-10-26/reference \
  --fit-lock runs/path-generalization-2026-10-26/payoffs/fit_lock.json
uv run python analysis/run_generalization_evaluation.py --root runs/path-generalization-2026-10-26
uv run python analysis/audit_payoff_gaussian.py --root runs/path-generalization-2026-10-26
uv run python analysis/benchmark_path_generalization.py \
  --output runs/path-generalization-2026-10-26/timing/path_workloads.json --include-long --final-timing
uv run python analysis/report_path_generalization.py --root runs/path-generalization-2026-10-26 \
  --path-timing runs/path-generalization-2026-10-26/timing/path_workloads.json
```

`audit_payoff_gaussian.py` evaluates same-time marginal moments at the exact three
queries and requested RK4 steps used by the payoff timing comparison. It retains
both step sizes in all six groups and uses whole-trajectory-family bootstrap
counts shared with the broader evaluator. Its unstopped linear-Gaussian law is
an approximation: linearization, stopping and killed-integral bias remain
unresolved, and it supplies no joint-time covariance claim.

`report_path_generalization.py` fits no model and simulates no trajectories. It
requires a complete evaluation index and all declared group/method records,
reference-refinement audits, twelve locked fits, three complete worker records,
both final timing experiments, and all twelve Gaussian audit records. By default
it verifies linked checkpoints, reference, reduced/bootstrap arrays and timing
artifacts against their SHA-256 hashes. Keep artifact verification enabled for
scientific reports. It writes tables, figures, `report.md` and a hashed output
manifest under the study's `report/`; `--output` selects another directory.

The report keeps optimizer seeds, same-time and joint-time tasks, paired error
uncertainty, standalone candidate Monte Carlo uncertainty, initial-law integration
uncertainty and reference refinement separate. Full paths and statistical heads
have different capacities and training data. The completed comparison concerns
the declared finite-grid tasks and costs, with continuous-reference and hitting
bias unresolved.

The report includes fitting and historical pretraining, shared data preparation,
restoration, kernel preparation, input generation and online costs. Repeated-call
totals are known-component accounting, not accuracy-qualified break-even claims.
No neural cost advantage passes the declared accuracy and uncertainty gates.
The new integral-head failure is documented by a read-only exact-heat and
small-time covariance audit; the failed fits are retained without post-test
refitting.

`archive_path_generalization.py` verifies the completed report's input/output
hashes and creates a compact lossless evidence snapshot, leaving binary weights,
physical-noise arrays and full HDF5 banks in the indexed local run. The committed
[archive](../docs/experiments/artifacts/path_task_advantages_2026-10-27/README.md)
also retains all individual result JSON, checkpoint metadata, analysis-code
provenance and diagnostic replays. Historical immutable paths inside those
records are preserved; they are not portable substitutes for the original raw
inputs.

## Repairing the terminal and integral moment formulation

The [completed repair study](../docs/experiments/payoff_repair_2026-10-27.md)
uses source freeze `10a898d` and an independent test seed. Its known free-heat
mean and covariance anchors preserve the integral small-time orders; separate
networks train the mean first and then covariance with the selected mean frozen.
Training-only output/pair scales include the different integral time powers.
The covariance has no physical floor. Exact face traces introduce no auxiliary
noise estimator, and physical and optimizer RNG streams remain separate.

The fixed study retains ten fits: both banks with raw and balanced v1 loss at
seed zero, and repaired models at all three seeds. The repaired variant uses
two networks and two complete optimizer stages, so the full repair comparison
changes formulation, capacity and training allocation together. The v1 loss
ablation isolates only balancing. Untrained repaired models measure the value
of the known physical anchor separately from fitting.

With the original full historical inputs available, reproduce in this order:

```bash
uv run python analysis/run_payoff_repair.py prepare --root runs/payoff-repair-2026-10-27
uv run python analysis/evaluate_payoff_repair.py prepare --root runs/payoff-repair-2026-10-27
uv run python analysis/run_payoff_repair.py fit --seed 0 --root runs/payoff-repair-2026-10-27
uv run python analysis/run_payoff_repair.py fit --seed 1 --root runs/payoff-repair-2026-10-27
uv run python analysis/run_payoff_repair.py fit --seed 2 --root runs/payoff-repair-2026-10-27
uv run python analysis/run_payoff_repair.py lock --root runs/payoff-repair-2026-10-27
uv run python analysis/run_payoff_repair.py reference --root runs/payoff-repair-2026-10-27
uv run python analysis/evaluate_payoff_repair.py evaluate --root runs/payoff-repair-2026-10-27
uv run python analysis/evaluate_payoff_repair.py benchmark --final-timing --root runs/payoff-repair-2026-10-27
uv run python analysis/report_payoff_repair.py --root runs/payoff-repair-2026-10-27
```

Run the final benchmark after all fitting, accuracy, audit and test processes
have exited. The study used three seed workers; the commands above are a simpler
sequential reproduction. Atomic checkpoints preserve optimizer and RNG state.
Both preregistrations precede fitting; all ten checkpoints must restore before
the first fresh test anchor or future-noise outcome can be created.

Reporting also requires the independent pretest, postfit-validation, reference
and selected-accuracy audits preserved in the
[compact archive](../docs/experiments/artifacts/payoff_repair_2026-10-27/README.md).
Their replay scripts are included there. The reference and metric replays verify
the actual saved arrays; the postfit diagnostic reconstructs the covariance
phase's starting target using the selected mean, rather than the obsolete
initial mean's quadratic variation. All six repaired fits and four v1 controls
remain in this audit.

Accuracy compares the same three horizons and same-time 4×4 covariance bank as
the timing calls. Initial-ensemble NN and path costs use actual 16,384-sample
integrations. B1 first-query costs, later cached-batch costs, warm latency,
restoration and kernel preparation are distinct. Repeated-call totals assume
new initial-state queries with fixed physics, bank and horizons. They retain
all known fitting and setup costs; unresolved accuracy gates cannot produce a
certified crossover. The original source-strict historical loaders execute in
their exact Git snapshots; no guard or saved checkpoint is rewritten.

`archive_payoff_repair.py` verifies the complete report and creates a lossless
compact snapshot. Full raw trajectories, Brownian arrays and restorable weights
remain in the checksummed local inventory; the public compact archive does not
replace those inputs or a full Git history.

## Longer horizons and stronger interactions

`run_payoff_regimes.py` freezes six fixed problems: T=0.25/1 and κ=0.75/3/6,
with G=32, N=2048, D=1 and a₂=0.1. Each problem fits all three repaired
statistics seeds to the mixed terminal-mode-3/time-integral-mode-1 bank. The
mean and covariance stages each receive the complete declared optimizer
allocation; selection uses validation only. Sixteen deterministic quadrature
nodes keep the known heat anchor accurate through T=1.

```bash
.venv/bin/python analysis/run_payoff_regimes.py prepare --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/evaluate_payoff_regimes.py prepare --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_regimes.py pretest --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/evaluate_payoff_regimes_deployment.py prepare --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_regimes_deployment.py --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/run_payoff_regimes.py fit --root runs/payoff-regimes-2026-10-27 --seed 0
.venv/bin/python analysis/run_payoff_regimes.py fit --root runs/payoff-regimes-2026-10-27 --seed 1
.venv/bin/python analysis/run_payoff_regimes.py fit --root runs/payoff-regimes-2026-10-27 --seed 2
.venv/bin/python analysis/run_payoff_regimes.py lock --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_regimes_validation.py --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_boundary_corner.py --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/run_payoff_regimes.py reference --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/evaluate_payoff_regimes.py evaluate --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/evaluate_payoff_regimes_deployment.py evaluate --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_regimes_native.py --root runs/payoff-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_regimes_metrics.py --root runs/payoff-regimes-2026-10-27
```

The three seed workers may run concurrently in separate processes; their
checkpoint ownership locks are disjoint. Wait for every worker to exit before
locking. Native references remain unavailable until all 18 complete checkpoints
pass strict restoration. References can run concurrently by supplying distinct
`--case` values; each case owns its HDF5 and generation lock. The frozen plan
records every source/helper hash. Exit 75 denotes a resumable budget boundary;
repeat the same invocation without changing its plan or source.

Each problem retains two supplied histograms per initial law and one IID-start
ensemble per law. The independent native deployment has fresh forcing in every
group and fresh initial histograms for ensemble groups. Conditional methods
share the declared supplied histogram. Whole-family bootstrap intervals compare
NN and Gaussian errors using a shared reference resample. Standalone initial-law
and native-Monte-Carlo sampling uncertainty remain separate.

The independent-deployment supplement preregisters 12 additional initial-law
integration pools before any reference draw. Its 48 NN/control records use
initial histograms and bootstrap counts independent of the reference. They
replace the 48 shared-initial NN ensemble records in primary deployment tables;
all 300 records remain retained, with the original paired estimates available
as approximation-bias diagnostics. Fresh ensemble timing already uses this
same prediction algorithm. This evaluation control does not change any fit.

After **all fitting, reference, accuracy and audit processes have exited**, run:

```bash
.venv/bin/python analysis/evaluate_payoff_regimes.py benchmark --root runs/payoff-regimes-2026-10-27 --final-timing
.venv/bin/python analysis/report_payoff_regimes.py report --run runs/payoff-regimes-2026-10-27 \
  --documentation docs/experiments/payoff_regimes_2026-10-27.md
```

The report preserves both literal Gaussian steps, all NN seeds, observed
stopping/near-boundary frequencies, physical payoff bounds and the exact
supported four means/same-time covariance bank. It charges one model fit, its
training/validation pool, restoration, first-query/cache costs and warm calls.
Native deployment actually simulates 16,384 paths per call; no large-ensemble
cost is extrapolated. Resolved pointwise accuracy contrasts, the absolute 5%
diagnostic, and accuracy-qualified payback are distinct outputs. All stopped
paths are retained; the unstopped Gaussian law and continuous hitting bias are
explicit limitations.

After final tests have been saved in `audit/final_tests.json`, preserve compact
evidence with `analysis/report_payoff_regimes.py archive --run RUNROOT`; use
`verify-archive --archive ARCHIVE` to verify stored and restored checksums.
Raw HDF5, arrays and weights remain local with a full inventory. Prior studies
and checkpoint formats are preserved.

## Boundary and residual loss repair

`run_payoff_boundary_regimes.py` runs the distinct v3 statistics formulation
described in the [statistics guide](../docs/BSDE_STATISTICS.md#boundary-consistent-generator-formulation).
It keeps the earlier six horizon/interaction cases, all three optimizer seeds,
and the earlier v2 checkpoints as controls. Full historical local inputs from
`runs/payoff-regimes-2026-10-27/` and the checksum-indexed public evidence archive
are required to restore those controls. The legacy loader verifies the archived
source bytes against the corresponding current historical modules and checks
the local weights, metadata, and fit lock. The compact archive indexes binary
weights and numerical banks that remain local.

Use a fresh directory and keep package source and all preregistered helper
files unchanged during execution. The study freezes fresh training, validation,
reference, native-MC comparison, and neural initial-integration streams before
fitting. Preparation and identifier-only independence checks precede any final
outcomes:

```bash
.venv/bin/python analysis/run_payoff_boundary_regimes.py prepare --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/evaluate_payoff_boundary_regimes.py prepare --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/evaluate_payoff_boundary_deployment.py prepare --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_boundary_regimes.py pretest --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_boundary_deployment.py --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_boundary_formulation.py --output runs/payoff-boundary-regimes-2026-10-27/audit/formulation_control.json
```

Finish all eighteen fits before generating final reference paths. The ordinary
sequential commands are sufficient; separate optimizer seeds may run in three
processes with one owner per seed. No reference job may overlap fitting. A
budget exit of 75 checkpoints the unfinished allocation; resume the identical
seed command before locking. Never discard a seed or refit after viewing final
outcomes.

```bash
.venv/bin/python analysis/run_payoff_boundary_regimes.py fit --seed 0 --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/run_payoff_boundary_regimes.py fit --seed 1 --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/run_payoff_boundary_regimes.py fit --seed 2 --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/run_payoff_boundary_regimes.py lock --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_boundary_validation.py --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/review_payoff_boundary_fit.py --root runs/payoff-boundary-regimes-2026-10-27
```

For each of `T0p25_k0p75`, `T0p25_k3`, `T0p25_k6`, `T1_k0p75`, `T1_k3`,
and `T1_k6`, run the reference command below with that `--case`. Native reference
refinements share Brownian families within a group. Reference, comparison-MC,
and neural initial-integration families are independent. Every stopped path
contributes to the statistics.

```bash
.venv/bin/python analysis/run_payoff_boundary_regimes.py reference --case T0p25_k0p75 --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/evaluate_payoff_boundary_regimes.py evaluate --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/evaluate_payoff_boundary_deployment.py evaluate --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_boundary_native.py --root runs/payoff-boundary-regimes-2026-10-27
.venv/bin/python analysis/audit_payoff_boundary_metrics.py --root runs/payoff-boundary-regimes-2026-10-27
```

The evaluation commands require all six references. They retain 360 original
method records and 84 independent-initial neural records. The latter replace
the shared-initial ensemble predictions in the 360-record primary comparison;
the original paired diagnostics remain available. Matched-seed v3/v2 contrasts
and v3/Gaussian contrasts use whole-family bootstrap uncertainty, with
independent reference and initial-integration resampling. Neither consistency
across seeds nor pointwise intervals constitute a simultaneous grid-wide claim.

After **every** fitting, reference, review, audit, and other heavy CPU process
has exited, measure final timing in isolation:

```bash
.venv/bin/python analysis/evaluate_payoff_boundary_regimes.py benchmark --root runs/payoff-boundary-regimes-2026-10-27 --final-timing
.venv/bin/python analysis/report_payoff_boundary_regimes.py report --run runs/payoff-boundary-regimes-2026-10-27 --documentation docs/experiments/payoff_boundary_regimes_2026-10-27.md
.venv/bin/python analysis/audit_payoff_boundary_report.py --root runs/payoff-boundary-regimes-2026-10-27
```

Conditional timings accept supplied histograms; independent-ensemble timings
include drawing and integrating 16,384 fresh histograms. Both v2 and v3 neural
costs include restoration, cold prediction, literal warm calls, and their own
one-model training and shared state-generation costs. Historical v2 fitting is
disclosed separately from work incurred by this rerun. Summed concurrent fit
durations are not elapsed batch time; worker process CPU and monotonic wall
intervals remain separate. Arithmetic amortization is qualified by accuracy
and uncertainty only when all corresponding gates pass. The finite-grid stopped
Euler reference and its refinements do not certify a continuum DK limit.

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
