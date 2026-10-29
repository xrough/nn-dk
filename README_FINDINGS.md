# NN-DK current findings

**Status as of 30 October 2026.** The completed accuracy comparisons below run
through 27 October. New validation diagnostics and the ongoing independent
comparison are recorded in section 6; their final accuracy and costs are pending.

The project has working neural density-path and neural statistics solvers.
The path network closely reproduces its numerical training target, but the
current implementation is slower than compiled native Euler. Statistics
networks can answer conditional queries quickly, and a restricted noninteracting
moment problem has passed a deterministic accuracy certificate. That success
does not extend to the tested longer, strongly interacting problems. **No
accuracy-qualified neural training-cost advantage has been established.**

## 1. What each solver computes

| Track | Learned object and output | Current interpretation |
| --- | --- | --- |
| Autonomous coefficient solver | Conservative drift and face-noise covariance, followed by numerical time integration | A separate coefficient-learning experiment; it is not the density-path operator. |
| Neural density-path solver | Initial cell masses and physical Brownian increments mapped to complete cell-mass paths | Generates stochastic realizations from supplied noise; retains known interaction and mobility. |
| Neural statistics solvers | Conditional means and same-time covariances for a declared observable bank | Answer statistical queries directly; do not sample density paths or supply arbitrary new observables. |
| Classical controls | Native stopped Euler/Monte Carlo, deterministic moments, linear-Gaussian approximation | Establish numerical references and task-specific accuracy/cost baselines. |

The original trajectory-BSDE objective and the later known-generator residual
objectives are distinct formulations. In the path formulation, the physical
forcing also enters its stochastic residual. In trajectory-BSDE training, the
forward state and martingale use the same Brownian increments. Optimization
randomness is separate; the objective does not add independent physical forcing.

The main interacting results concern a **stopped finite-volume DK system** on
the periodic interval `[0, 2π)`, usually with `G=32`, `N=2048`, and `D=1`.
Initial masses are IID particle histograms. Euler paths freeze at the first
crossing of density `0.01/(2π)`, using piecewise-linear crossing interpolation.
Every stopped path remains in the statistics; states are not clipped or
renormalized. These field experiments do not establish agreement with a
microscopic particle law or an unregularized continuum DK equation.

## 2. Density paths: accurate Euler approximation, no current speed advantage

The earlier autonomous-coefficient CNN fitted the uniform, noninteracting
problem with held-out drift and covariance errors below 0.5%. Its 40 comparison
records did not resolve the full accuracy gates because of reference and
sampling limitations. It established neither learning convergence nor a neural
speed advantage. Its historical integration backend also included unused
native-coefficient work. See the [coefficient study](docs/experiments/a_first_batch_2026-10-24.md).

The interacting refinement study used `T=1/16`,
`V(x)=-0.75 cos(x)+0.1 cos(2x)`, and uniform/nonuniform initial laws. Its
full-field strong errors against a coupled fine Euler reference were:

| Separately fitted step | NN / fine Euler strong RMS |
| --- | ---: |
| `2^-8` | 16.51–16.73% |
| `2^-9` | 7.719–7.734% |
| `2^-10` | 3.597–3.647% |

All 18 same-step Euler gates passed; the largest NN/same-step error was 0.105%.
Native Euler's fine-reference errors nearly coincide with the neural errors.
Time discretization therefore dominates this comparison. These percentages use
fixed uniform-IID field-fluctuation energy, not relative density error.
Each resolution has a separate fit: this is not a temporal-order result for one
frozen network. See the [interacting refinement report](docs/experiments/interacting_followup_2026-10-25.md).

The separate first `T=1` batch completed 15 fits with maximum same-step strong
error 0.1104%, while stochastic fine-reference errors remained about 44–45% at
the tested coarse step `2^-7`. Good same-step agreement alone is insufficient.
See the [first path-batch report](docs/experiments/artifacts/path_first_batch_2026-10-25/path_results.md).

In the broader independent comparison, prepared single-path inference at
`Δ=1/1024` was **32.7–34 times slower** than compiled native Euler. Native JVPs
and adjoints were also faster. A small full-state strong error did not establish
5% accuracy for every high-mode or nonlinear payoff. These are measured CPU
implementation results, not intrinsic limits on neural architectures.
See the [broader task comparison](docs/experiments/path_task_advantages_2026-10-27.md).

## 3. Statistics: restricted successes and limits of transfer

The noninteracting structured-generator study passed **all 15 declared
comparisons**, including deterministic reference and stopping bounds. Its worst
normalized mean bound was **0.13723%**, worst full-covariance bound **0.17964%**
(OU control), and worst DK covariance bound **0.12574%**. The largest reference
contribution was **0.06277%**. This is a positive continuous-time, fixed-grid
stopped-DK result for the tested low Fourier moments and initial states.
The interior moment equations close at degree two; interaction breaks that
closure. No matched classical speed advantage was demonstrated.
See the [certified moment study](docs/experiments/generator_accuracy_2026-10-25.md).

For the earlier short interacting fixed bank, conditional mean/covariance point
errors were **1.22–2.42% / 0.73–2.58%**, and warm conditional queries took
**0.36–0.76 ms**. Pointwise bounds lay below 5% against the finite reference;
the stricter uncertainty and reference-bias requirements remained unresolved.
New observable banks, particularly time integrals, exposed failures.
The initial [formulation repair](docs/experiments/payoff_repair_2026-10-27.md)
improved its short-case mean/covariance discrepancies to roughly **2.4% / 3.3%**,
but longer and stronger interactions remained problematic.

These results support carefully validated fixed-bank statistical maps, not a
general statistics solver. The earlier [BSDE coverage study](docs/experiments/bsde_anchor_study_2026-10-24.md)
also found that more independent initial anchors and a structured head helped
at a fixed total training-path budget; it did not establish learning convergence.

## 4. Latest boundary and residual-loss repair

The latest study retained **18 fresh v3 fits**, **18 frozen v2 controls**, three
optimizer seeds, and **589,824 fresh native reference trajectory families**.
It tested `T∈{0.25,1}`, `κ∈{0.75,3,6}`, and `a₂=0.1`. The bank contains terminal
sine/cosine mode 3 and killed sine/cosine mode-1 time integrals divided by `T`,
queried at `T/4`, `T/2`, and `T`.

V3 introduced a moving boundary gate, exact native short-time mean slopes,
time-ordered positive covariance, and fixed full-horizon residual units with
pseudo-Huber loss. It removed inverse-time loss weights. The consistency,
conservation-related generator, absorption, and derivative checks passed.
**Overall held-out statistical accuracy did not improve.**

The following errors are maxima over query times, both initial laws,
conditional/ensemble groups, and observable entries; ranges retain all seeds.
Mean percentages divide by reference SD; covariance percentages divide by
products of reference SDs. Thus **100% mean error means one reference SD**, not
100% relative error of a possibly zero mean. These units differ from the
field-path normalization above and the fixed scales in older studies.

| T | κ | v3 mean (% SD) | v3 covariance (% SD products) | Gaussian mean (% SD) | Gaussian covariance (% SD products) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.75 | 1.61–1.64 | 3.08–3.09 | 1.47 | 3.02 |
| 0.25 | 3 | 4.19–5.42 | 3.38–3.41 | 1.40 | 2.94 |
| 0.25 | 6 | 10.0–12.9 | 3.72–4.57 | 1.42 | 3.79 |
| 1 | 0.75 | 8.51–16.3 | 4.78–5.98 | 2.08 | 3.58 |
| 1 | 3 | 28.1–33.6 | 14.1–26.6 | 1.68 | 2.92 |
| 1 | 6 | 247–305 | 304–326 | 26.0 | 17.4 |

At `T=1, κ=6`, v3 improves the worst mean error from v2's **5.75–5.86 SD** to
**2.47–3.05 SD**. However, terminal covariance error worsens to **304–326%**,
while integral–integral covariance error remains **94.3–94.6%**. Across 216
group/time/statistic comparisons, all-seed resolved pointwise contrasts favor
v3 over v2 in **21**, and v2 in **49**. Against fine Gaussian, **0** favor v3
and **85** favor Gaussian. Remaining comparisons are unresolved or depend on
optimizer seed. These counts are not simultaneous grid-wide inference.

The residual diagnosis remains severe: only **12/4,096 collocation rows** near
the stopping boundary contribute **98.2–98.9%** of squared covariance residual
and **62.7–69.4%** of robust loss. On a matched covariance-stage diagnostic
subset with the mean fixed, robust loss decreases **4.3–7.4%** while normalized
non-robust residual RMS increases **2.0–4.2%**. Bulk-state fitting errors remain
as well.
Correct boundary slopes, positive covariance, and a smaller robust objective
therefore do not establish accurate finite-time fluctuations.

V2 and v3 use different learner pools. Their common fresh tests compare fitted
methods, but do not uniquely attribute differences to the boundary or loss
change. See the [complete latest report](docs/experiments/payoff_boundary_regimes_2026-10-27.md)
and [residual formulation and diagnosis](docs/BSDE_STATISTICS.md#boundary-consistent-generator-formulation).

## 5. Training and inference costs

Latest CPU timings return the full three-time bank. Conditional queries assume
a supplied histogram. Neural IID estimates draw and integrate 16,384 fresh
histograms; Gaussian integrates initial sampling covariance analytically.
Native MC simulates 16,384 independent physical continuations.

| Method | Training per model | Warm conditional query | Warm IID-law estimate |
| --- | ---: | ---: | ---: |
| Repaired v3 NN | 513–635 s | 0.701–0.738 ms | 312–349 ms |
| Frozen v2 NN | 481–743 s, historical | 0.619–0.649 ms | 249–260 ms |
| Cheaper Gaussian step | None | 7.8–31.4 ms | 7.9–31.8 ms |
| Native MC | None | 1.9–7.7 s | 2.0–7.8 s |

These ranges span the six cases. The cheaper Gaussian step has essentially the
same accuracy as its finer step here. Neural conditional latency is low, but
training, loading and initialization matter. Neural IID integration is slower
than Gaussian. Total costs include each chosen model's fit and data pool,
restoration, cold prediction, cache preparation, and subsequent warm calls.
The report separates historical costs from this batch's new work.

**No accuracy-qualified training-cost crossover is established.** Large
strong-case errors are model failures; some mild-case gates remain unresolved
because of sampling/reference uncertainty. Arithmetic break-even curves alone
do not resolve either issue. Repeated answers for an identical initial law
can also be cached.

## 6. Practical conclusions and next work

| Task | Current evidence supports |
| --- | --- |
| Full paths, new observables, sensitivities | Compiled native Euler with task-appropriate refinement checks. |
| Repeated fixed-bank conditional moments | Gaussian as the inexpensive approximate control; use a neural map only within its independently validated contract. |
| IID-law moments | Gaussian when its approximation is adequate; native MC for the stopped nonlinear law. |
| Strong-interaction fluctuation accuracy | Further NN formulation work; current v3 covariance predictions are unreliable. Gaussian also misses the 5% target in the hardest case. |
| Rare stopping events or a continuum DK limit | No general accuracy/convergence claim from these experiments. |

The next priorities are to investigate mean-gradient accuracy in the covariance
source, boundary-layer sampling, and alignment between the training objective
and fluctuation accuracy. A nonlinear mean-field/Gaussian baseline with learned
corrections is a candidate to test, not an established improvement. Any new
formulation needs fresh final data after validation choices are frozen.

The [29 October diagnostic study](docs/experiments/payoff_diagnostics_2026-10-29.md)
completed the mean-gradient/source and residual-transport investigations on
frozen validation anchors. Remaining derivative-interchange, stopping, time and
quadrature errors prevent treating the diagnostic residual as an error bound.

The matched proposal/loss study has now completed all **72 fits**. Its
[residual supplement](docs/experiments/payoff_collocation_residuals_2026-10-30.md)
retains all nineteen boundary/time strata in common units. Five regimes occupy
only four strata; the strongest regime occupies eleven. The closest boundary
bin is empty in every regime. Occupancy reweighting cannot fill that gap, and
these model-selection data are not independent accuracy evidence.

All **18 learned nonlinear-LNA correction fits** and their label-prefix replays
also completed. Both studies' fits, sources, criteria and disjoint identifier
plans were jointly locked before the new **589,824-family** reference bank was
generated. Its primary step is `2^-12`, with paired-prefix checks at `2^-13` and
`2^-14`. Independent accuracy evaluations are running. Contrasts, isolated
inference timings and the overall research completion record remain pending;
the [task ledger](docs/README_TASKS.md) records those obligations.

The interacting path studies do not establish a temporal convergence order,
learning-convergence rate, particle-law equivalence, or continuum SPDE limit.
The restricted noninteracting certificate above remains a separate valid
result. Its assumptions do not transfer to the strong-interaction experiments.

## 7. Evidence and reproduction

The completed boundary-study revision records **1,181 passing tests**, passing lint,
independent native/statistical replay, **408 isolated timing workloads**, and
verified cost accounting. Software acceptance is separate from scientific
accuracy. The current research implementation separately passes **1,502 tests**
and Ruff, including historical-source loading tests. The clean full-history
clone, package builds and smoke command also pass; see the
[verification record](docs/verification.md). Software acceptance does not
complete the independent scientific comparison.

- [Latest report](docs/experiments/payoff_boundary_regimes_2026-10-27.md)
- [Completion checks](docs/experiments/payoff_boundary_regimes_2026-10-27_completion.json)
- [Evidence archive](docs/experiments/artifacts/payoff_boundary_regimes_2026-10-27/README.md): 765 verified copied artifacts and checksums indexing 1,211 local raw files
- [Implementation and execution guide](analysis/README.md#boundary-and-residual-loss-repair)
- [Project installation and workflow overview](README.md)

The completed boundary study's source freeze is `0d38243`; its findings/archive
commit is `d4bd378`. The new comparison freezes its own source and fit locks.
Source/configuration hashes, trajectory-family splits, optimizer
seeds and failures are retained. Large binary checkpoints and numerical banks
remain in local ignored `runs/` directories. Historical studies must be
reproduced with their recorded source and inputs; timings from different
revisions or workloads are not directly interchangeable.
