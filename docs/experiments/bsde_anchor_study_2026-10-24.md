# Initial-state coverage and structured BSDE study — 24 October 2026

**Status: COMPLETE.** All **36 fits, 36 evaluations, 234 comparison rows and 75
paired contrasts** completed in **150.3082 minutes**. The strict completed-run
audit checked checkpoint bytes, the fit lock, declared allocations, reference
families and numerical provenance. Source and configuration stayed frozen.

Broader initial-state coverage substantially reduced DK conditional errors at
the fixed 2048-path training budget. The structured value head improved them
further. At 2048 anchors its conditional mean RMSE is **4.47–4.75%**, while
variance RMSE is **5.91–7.06%**. These comparisons include appreciable reference
Monte Carlo noise: **5% accuracy for the improved DK fits remains unresolved**.
Every primary pointwise gate remains
unresolved. The next limitation is reference precision alongside the separate
OU fitting defect described below.

This study follows the [first BSDE batch](bsde_first_batch_2026-10-24.md), where
large conditional errors persisted despite smaller ensemble-average errors.
It tests two possible remedies: broader coverage of independent initial states
and a value head with learned modewise structure. Every arm is fitted again and
uses a fresh common test bank. Earlier inspected test results are contextual
evidence, not the new architecture-selection bank.
Fresh initial states are IID histograms from the same two prescribed profile
laws; this does not test generalization to new profiles or physical parameters.
The outputs are conditional means and same-time 4×4 covariances, not sampled
density paths or a cross-time joint law.

## Fixed protocol

| Setting | Prespecified allocation |
|---|---|
| Problems | Four-coordinate OU calibration and noninteracting native stopped DK |
| DK target | Periodic [0, 2π), G=32, N=2048, D=1, T=1 |
| DK initial laws | IID uniform particles; IID density proportional to exp(0.2 cos x), histogrammed into cells |
| Training anchors × children | 128×16; 512×4; 2048×1 |
| Training paths per fit | **2048 in every arm** |
| Shared validation | 256 independent anchors × 4 independent continuations = 1024 paths |
| Value architectures | Existing MLP; structured modewise backbone with full-state DK residual |
| Optimizer seeds | 0, 1, 2, kept separate |
| Number of fits | 2 problems × 3 allocations × 2 architectures × 3 seeds = **36** |
| Forward / main reference step | 2^-7 / 2^-13 |
| Reference refinement | 32 coupled families per profile at 2^-13, 2^-14, 2^-15 |
| Queries | Exact cell-integrated normalized sine/cosine modes 1 and 2 at 1/4, 1/2, 1 |
| Fresh test master seed | **20261025**; the earlier inspected batch used 20261024 |
| OU analytic test | 1024 independent initial states; exact Euler and continuous-time moments |
| Conditional DK test | 32 fixed initial anchors per profile × 512 independent continuations |
| Ensemble DK test | 8192 independent initial-histogram/forcing families per profile |
| Optimization | AdamW lr 1e-3, weight decay 1e-5, batch 32, at most 100 epochs, patience 10 |
| Integrand network | Three hidden layers of width 64, SiLU; unchanged between value architectures |
| Execution | CPU, float64, one PyTorch thread; eight-hour resumable invocation budget |
| Uncertainty | 200 paired whole-trajectory bootstrap resamples; pointwise 95% intervals |

This is a **coverage-versus-replication study at fixed training-path budget**.
Increasing the number of anchors reduces the number of continuations per anchor;
it does not increase the total number of training trajectories. Its curves
cannot establish a training-sample-size convergence rate. The shared training
union contains only 5120 paths per problem, rather than a dense 2048×16 pool.
Overlapping views share exact initial states and Brownian increments. Each DK
training view contains 1024 trajectories from each profile.

Train, validation, calibration and test streams have separate namespaces. For a
matched optimizer seed, the martingale network starts with bitwise-identical
parameters across value architectures. Feature normalization uses each selected
view's unique training anchors. The integrand is fitted jointly with the value
head, so its fitted weights can differ between arms. Test outcomes cannot select an epoch or alter a
checkpoint: the runner requires completion and a byte-hash lock of all 36 fits
before evaluating any final test arm.

## Structure and stochastic convention

The structured OU value head has **four learned positive parameters**: two decay
rates and two variance scales, shared within sine/cosine mode pairs. Rates and
scales start at one and are learned from the same BSDE trajectory objective as
the MLP. Analytic rates and analytic mean forecasts are not supplied to the
predictor. This family contains the stable Euler OU conditional moment map.

Structured DK uses this learned backbone with a **0.1-scaled full-state residual**
for the mean and Cholesky covariance factor. The residual retains every
mass-zero initial-state coordinate and starts at zero. Stopping generally breaks
exact low-mode closure, so this is an inductive bias rather than an exact DK
formula. No additive covariance floor is introduced. Both architectures return
the exact deterministic values at time zero and for initially stopped states.

The physical state is the native finite-volume mass vector. A path freezes at
the first crossing of density 0.01/(2π) by its piecewise-linear Euler
interpolation, and every stopped path remains in statistics. No clipping or
renormalization is applied. The BSDE martingale uses the **same full Brownian
increments** as the forward process with a predictable active-at-start mask,
including the entire crossing-step increment. Oracle pre-stopping proposals
remain separate from learner features.

## Completed numerical and data checks

Numerical calibration passed for Brownian, OU and both native DK profiles. The
Brownian quadratic-payoff residual has expected MSE **0.015625** and measured MSE
**0.01583111**; the discrete pathwise identity agrees within **1.27e-14**. A nonzero
terminal residual is expected from the first-order stochastic integral and is
not itself a value-head prediction error.

| DK profile | Coupled calibration families | Max saved mass error | Minimum saved mass | Finest-refinement max absolute CI endpoint |
|---|---:|---:|---:|---:|
| Uniform | 32 | 5.33e-15 | 0.018592 | 0.17019% |
| Nonuniform | 32 | 5.11e-15 | 0.016130 | 0.16190% |

The last column describes the finite 2^-14-versus-2^-15 contrast in fixed
fluctuation units. It is a pointwise Monte Carlo diagnostic, not a bound on
remaining continuous-time bias. Reference bias remains **uncertified**.

A separate read-only audit of HDF5 metadata and identifier arrays passed:

- Each OU/DK pool has **5120 unique training families and 1024 validation
  families**, with 2048 training anchors and 256 validation anchors.
- Anchor and continuation IDs are disjoint between training and validation and
  between forward problems. Every DK allocation balances the two profiles.
- Every anchor commit is complete, arrays are chunked, and uncensored proposals
  are stored only under the separate oracle group.

The audit read no full state arrays or held-out outcomes. Its saved evidence is
`runs/bsde-anchor-study/report/study_data_audit.json`. Software acceptance before
launch comprised **226 passing tests** in 29.45 seconds, a passing Ruff check,
and the saved CLI smoke completing all 12 fits and 12 evaluations in **3.9397
seconds**. The smoke checks execution, not scientific accuracy.

## DK results: coverage and structure both help

The table uses the **same-step Euler target**, separating neural approximation
from forward-time bias. Entries are minimum–maximum RMSE over seeds 0, 1 and 2,
not pooled estimates or confidence intervals. All individual seeds, fine-total
comparisons, covariances, pointwise gates and paired intervals remain in the
[234-row table](artifacts/bsde_anchor_study_2026-10-24/study_table.csv) and
[audited summary](artifacts/bsde_anchor_study_2026-10-24/study_summary.json).
Percentages are errors in fixed fluctuation units, not relative density errors.

| Profile | Value head | Anchors × children | Conditional mean (%) | Conditional variance (%) | Ensemble mean (%) | Ensemble variance (%) |
|---|---|---:|---:|---:|---:|---:|
| Uniform | MLP | 128×16 | 14.60–17.84 | 20.86–26.31 | 2.33–2.86 | 7.77–10.26 |
| Uniform | MLP | 512×4 | 5.88–6.18 | 7.27–8.26 | 1.23–2.03 | 1.79–2.39 |
| Uniform | MLP | 2048×1 | 5.09–5.63 | 7.11–7.40 | 1.23–2.02 | 1.78–1.93 |
| Uniform | Structured | 128×16 | 6.66–7.34 | 7.51–8.11 | 1.19–1.70 | 1.10–1.73 |
| Uniform | Structured | 512×4 | 4.50–4.69 | 6.79–6.91 | 1.13–1.24 | 1.19–1.60 |
| Uniform | Structured | 2048×1 | 4.57–4.75 | 6.70–7.06 | 0.92–1.11 | 1.61–1.86 |
| Nonuniform | MLP | 128×16 | 14.56–16.91 | 14.91–20.62 | 1.78–2.50 | 4.17–6.01 |
| Nonuniform | MLP | 512×4 | 5.91–6.17 | 6.50–7.78 | 1.19–2.23 | 2.08–3.52 |
| Nonuniform | MLP | 2048×1 | 5.19–5.33 | 6.63–7.46 | 1.28–1.95 | 1.88–4.70 |
| Nonuniform | Structured | 128×16 | 5.51–6.27 | 6.65–7.92 | 1.22–1.37 | 1.51–1.69 |
| Nonuniform | Structured | 512×4 | 4.55–4.84 | 5.68–5.91 | 1.30–1.60 | 1.27–1.69 |
| Nonuniform | Structured | 2048×1 | 4.47–4.67 | 5.91–6.12 | 1.28–1.43 | 0.85–1.15 |

The large MLP improvement occurs mainly between 128 and 512 anchors. The
structured head is already substantially better at 128 anchors. Changes beyond
512 anchors are smaller and not uniformly favorable in every metric; this is
not evidence of a formal rate or a proven approximation plateau.

Prespecified paired pointwise 95% intervals are negative for all **12 of 12**
DK conditional mean contrasts comparing 2048 versus 128 anchors, and **11 of
12** variance contrasts. Comparing structured versus MLP at fixed allocation,
they are negative for **17 of 18** conditional mean contrasts and **16 of 18**
variance contrasts; the remaining intervals cross zero. These counts preserve
each profile and optimizer seed and are not multiplicity-adjusted. Ensemble
improvements are less consistently resolved and cannot replace conditional
accuracy checks. The [complete paired contrasts](artifacts/bsde_anchor_study_2026-10-24/paired_contrasts.json)
retain every interval and unresolved comparison.

![Same-step mean and variance learning curves](artifacts/bsde_anchor_study_2026-10-24/study_learning_curves.png)

## Precision limits and OU optimization control

At 512 reference children per anchor, [saved-reference plug-in noise diagnostics](artifacts/bsde_anchor_study_2026-10-24/reference_noise_diagnostic.json)
put conditional mean RMSE around **3.94–3.95%** and variance RMSE around
**5.11–5.12%**, even for a predictor equal to the true conditional moments.
These are estimated Monte Carlo contributions, not certified bounds. Thus the
structured model's observed 4.47–4.75% mean and 5.91–7.06% variance discrepancies
mix neural error and reference sampling noise. One cannot attribute the whole
variance discrepancy to neural bias or subtract these RMSEs linearly.
The variance diagnostic substitutes empirical fourth moments in the sampling
variance formula; it is not a confidence interval or a certified error floor.

The maximum conditional sampling halfwidth is **15.76–16.69%**, and ensemble
halfwidths are **3.12–3.21%**, above the 1.25% budget. All **90 primary gates**
remain unresolved. The conditional bootstrap resamples whole children within
each of the 32 fixed anchors per profile; it does not quantify generalization
uncertainty over newly drawn conditional anchors. Ensemble bootstrap families
include initial-histogram randomness. Remaining fine-reference DK bias is also
uncertified. Same-step, fine-total and time-contrast RMSEs are not additive.

OU gives a separate noiseless-reference warning. Across all nine matched fits,
structure improves variance RMSE but worsens mean RMSE relative to the MLP:

| Value head | Anchors | Mean RMSE (%) | Variance RMSE (%) | Maximum mean error (%) |
|---|---:|---:|---:|---:|
| MLP | 128 | 1.39–1.58 | 2.34–2.54 | 22.10–35.53 |
| MLP | 512 | 1.09–1.48 | 2.81–3.13 | 12.13–17.68 |
| MLP | 2048 | 1.14–1.33 | 2.41–2.74 | 12.43–16.90 |
| Structured | 128 | 1.94–1.95 | 0.93–0.97 | 11.91–12.01 |
| Structured | 512 | 1.77–2.21 | 0.83–1.08 | 10.94–13.58 |
| Structured | 2048 | 1.70–1.95 | 0.88–1.00 | 10.49–12.03 |

All nine structured OU fits reached the 100-epoch cap. Their selected mode-2
rates are **3.584–3.677**, compared with the representable Euler rate **4.011**.
The architecture contains the exact Euler map, yet fitting has not recovered
it. This supports a separate estimation/optimization investigation, not a claim
that simply running longer will solve it. The completed
[training-only probe](artifacts/bsde_anchor_study_2026-10-24/training_probe.json)
retains selected/final parameters and loss diagnostics. DK backbone rates are
not identifiable physical estimates because its residual can compensate them.

## Stopping, cost and next experiment

All **49,152 distinct DK reference continuation families** across the four banks
were retained; conditional children share their 32 fixed initial histograms per
profile. None stopped. These observations leave boundary-sensitive accuracy
empirically untested. For one fixed anchor, 512 zero events give a one-sided
95% upper stopping-probability bound of **0.5834%**; for an 8192-family ensemble
bank the bound is **0.03656%**. Distinct anchors, profiles, resolutions or fitted
models must not be pooled to tighten a claim about one conditional target. The
moment predictor does not itself generate stopping times or density paths.

The 36 distinct fits used **7629.55 seconds** of recorded training time.
Shared pool generation took **5.70 seconds for OU** and **26.74 seconds for DK**.
The four shared coupled-reference banks took approximately **601.16 seconds**
of recorded simulation time, charged once across all fits. Warm single-state
DK value inference took **0.309–0.542 ms** for the MLP and **0.369–0.405 ms**
for the structured head, excluding initialization and checkpoint loading.
The [cost extract](artifacts/bsde_anchor_study_2026-10-24/costs.json) retains all
seeds, recorded latency/throughput and initialization costs. No matched-accuracy
speed advantage is established.

The next useful experiment is to **freeze these fits and prespecify a larger,
independent conditional reference bank**, sufficient to resolve errors near
5%, while running a separate OU optimization control. No model should be
selected or refitted using the enlarged final reference outcomes. Additional
training data, temporal rates, new profiles/physics and a pathwise SPDE solver
remain separate questions; this fixed-grid moment study establishes none of
those convergence claims.

## Provenance and reproduction

| Record | Value |
|---|---|
| Source commit at launch | `7c7b516` |
| Frozen numerical source hash | `033b1f54597f4fd5adc9b90e82c39b29b73c9aa99857424928dfc3cfac89aa47` |
| Study configuration hash | `5891960f24e26323cd941fe9e7ef59207a5e7a193c93778b11d8b9f927f6eae9` |
| Base configuration hash | `67f48de45807604ba295a7c1d1fee05cda5c8f5b1c9335973a218cdb7502c0f0` |

```bash
MPLCONFIGDIR=/private/tmp/nn-dk-mpl-cache .venv/bin/nn-dk bsde-study --preset bsde-anchor-study --output runs/bsde-anchor-study
```

Repeating the command resumes saved work only when source and configuration
match. The [implementation guide](../BSDE_ANCHOR_STUDY.md) describes stages,
model structure and interpretation. Strict post-run extraction used
`analysis/summarize_bsde_study.py` and `analysis/diagnose_bsde_reference_noise.py`;
**16 post-run analysis checks** passed (11 artifact guards and five reference-noise
tests), separately from the 226 prelaunch tests. Recorded lock/evaluation ordering supports the design
but is not cryptographic attestation.

The [archive manifest](artifacts/bsde_anchor_study_2026-10-24/archive_manifest.json)
hashes compact evidence, including the complete derived summary, all 234 table
rows, all 75 paired intervals, calibration, completed training probe, data audit,
fixed design, final manifest, figure and cost extract. Byte copies are marked;
derived extracts declare their original SHA-256 and omitted fields. Raw
anchor/horizon/mode error and interval arrays, full epoch histories, HDF5 paths,
reference observations and checkpoint tensors remain under the ignored run
directory. No optimizer seed or paired-contrast interval is omitted.
