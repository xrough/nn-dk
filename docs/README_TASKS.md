# README task and evidence ledger

Audit date: **29 October 2026**, starting from commit `d4bd378` and the local
[`README_FINDINGS.md`](../README_FINDINGS.md). This ledger tracks the active
request to implement the README plans. It does not mark that request complete.

## Historical work already supported by evidence

The commands in the [analysis README](../analysis/README.md) reproduce existing
studies. Their frozen source, inputs, optimizer allocations and integrity guards
must be preserved; rerunning an old command against changed source is not a way
to complete the new research tasks.

| Completed study | Declared execution retained | Completion record | Copied compact artifacts reverified |
| --- | --- | --- | ---: |
| Broader task comparison | 12 fresh statistics heads, 98,304 independent reference families, 174 main evaluations and 12 Gaussian checks | [Task-comparison completion](experiments/path_task_advantages_2026-10-27_completion.json) | 300 |
| Heat-anchored formulation repair | 10 fitted models/controls and 98,304 fresh reference families | [Repair completion](experiments/payoff_repair_2026-10-27_completion.json) | 403 |
| Longer/stronger regimes | 18 fitted statistics models and 589,824 fresh reference families | [Regime completion](experiments/payoff_regimes_2026-10-27_completion.json) | 594 |
| Boundary/residual repair | 18 fresh v3 fits, 18 frozen v2 controls, 589,824 fresh reference families, 444 retained accuracy records and 408 timing workloads | [Boundary-repair completion](experiments/payoff_boundary_regimes_2026-10-27_completion.json) | 765 |

This audit verified all **2,062 copied compact artifacts**, both their stored
SHA-256 values and restored original bytes for gzip files, plus the four
generated archive READMEs. No missing or mismatched copied artifacts were found.
At the audit start, the latest completion record's hashes matched the main
README, analysis README, statistics guide and latest experiment report at the
recorded historical revision. Later research documentation is tracked separately. The latest
local fit lock records all 18 models as complete without seed pooling or
selection from final outcomes.

These checks validate the compact evidence and its recorded provenance. They
do not constitute a new full-suite test run, a replay of every large local
HDF5/checkpoint bank, or a new scientific experiment. The recorded 1,181-test
result and full numerical replays belong to the completed boundary study.
Earlier archive READMEs are evidence/reproduction notes, not additional pending
training allocations.

## Forward requirements still to finish

The explicit new priorities are in
[findings, section 6](../README_FINDINGS.md#6-practical-conclusions-and-next-work).
None is satisfied merely by the historical consistency checks or the negative
v3 accuracy result.

| ID | Requirement | Status at this audit | Evidence needed for completion |
| --- | --- | --- | --- |
| R1 | Investigate mean-gradient accuracy in the covariance source | **Finite-Euler diagnostic complete.** The all-regime audit completed 768 chunks, 12,288 fresh families, 24 validation anchors and 72 model comparisons, with signed componentwise and paired-refinement evidence. Remaining derivative-interchange/stopping/time bias is explicit. | [29 October diagnostic report](experiments/payoff_diagnostics_2026-10-29.md) and its checksummed gradient/source evidence. This is a validation-anchor investigation, not a continuum or all-state certificate. |
| R2 | Investigate boundary-layer sampling | **Controlled study running.** The source-locked 72-fit natural-pool comparison started on 30 October, with one live worker per optimizer seed. Importance-corrected proposals and parent-grouped stress utilities are tested; this trial uses natural pools without synthetic augmentation. | A declared boundary/interior sampling design, correct treatment of the intended loss measure, diagnostics by stratum, and a controlled comparison that retains the original baseline and all declared seeds. |
| R3 | Investigate training-objective alignment with fluctuation accuracy | **Transport diagnostic complete; controlled study running.** All 192 diagnostic chunks, 1,536 fresh families and 72 model comparisons completed. The matched proposal/loss cross now runs from its locked design. Strong-case quadrature/time/stopping defects remain unresolved; the residual is not an error certificate. | Controlled, common-state comparisons of the chosen objectives or source treatment, keeping frozen-mean covariance stages distinct; report raw/common-unit residuals and independently measured mean/covariance errors, including failures. |
| R4 | Test the nonlinear mean-field/Gaussian baseline with learned corrections | **All 18 fits completed and locked; final evaluation pending.** All 307,200 prescribed learner continuations and 12 role banks completed, with 24 independent package-simulator prefix replays passing. Fit lock `3fdbc913…` preserves all seeds, model bytes and measured preparation/fit cost records. Raw and gated zero-correction controls remain in the design; final accuracy is unmeasured. | Implement and verify the stated nonlinear baseline and correction, retain its untrained baseline and the existing classical/neural controls, and evaluate its declared capabilities and limitations. A documented negative result is still a result; an implementation-only smoke test is insufficient. |
| R5 | Freeze validation choices before fresh final data | **Training plans and inputs frozen; joint final-data gate pending.** Actual final references remain unopened. The joint protocol requires all 72 collocation and 18 correction fits, exact evaluation/report sources, criteria and disjoint future identifier plans before shared final outcomes. | Prespecified design/source/configuration identities and completed, locked fits before generating new final reference outcomes; demonstrate family/stream separation, retain every declared seed and stopped path, and report reference refinement and sampling uncertainty. |
| R6 | Finish comparison, cost accounting and documentation | **Pending for the new work.** Historical reports cannot establish a new method's performance. | Complete every newly declared fit/evaluation/audit, measure matched inference after heavy jobs exit, account for training/preparation/initialization/inference, preserve checksummed evidence, and update the READMEs from actual outcomes. |

The existing [`native_gaussian_baseline.py`](../analysis/native_gaussian_baseline.py)
already evolves a nonlinear deterministic mean and linear-noise covariance
along it, using its state-dependent drift Jacobian and noise covariance. R4
therefore calls for auditing/reusing that baseline and testing the new learned
correction formulation, not relabeling its existing nonlinear mean evolution
as a new result.

R2/R3 also require fitted residual diagnostics by boundary/time stratum. Sampler
occupancy counts and global validation losses alone do not satisfy that check.
The additive residual audit/report uses the unchanged natural validation rows,
all fitted variants and seeds, and the original 19 state/time strata. It reports
native solver units, physical payoff units and train-scale dimensionless units,
including empty strata and distinct parent counts. These descriptive validation
diagnostics are separate from the independent final accuracy comparison; the
validation rows were already used for model selection.

The supplementary software passes the full **1,497-test** suite and Ruff. A
separate 12-fit fixture also passed the actual CLI, report, compact archive and
offline verification round trip. Those fixture results do not complete the
scientific diagnostic: execution on the full 72-model allocation remains gated
on its completed fit lock.

## Execution and acceptance

1. Establish the R1–R3 diagnostics on frozen validation data before choosing a
   new formulation. Keep diagnostic results separate from fresh final accuracy.
2. Implement and test R4, then use validation only to fix the candidate, sampling
   and objective choices. Prespecify the regimes, banks, seeds, allocations and
   baselines rather than selecting favorable final cases afterward.
3. Complete R5 and execute the entire declared comparison. Partial pilot runs,
   successful software tests or a smaller residual cannot replace the final
   independent accuracy experiment.
4. Complete R6 and audit each row above against its actual outputs. Leave any
   missing, unresolved or unverified requirement open.

All comparisons retain the stopped finite-grid target, observable-bank limits,
whole-family uncertainty and optimizer-seed distinctions. No guaranteed positive
NN accuracy/speed advantage is specified by these investigation tasks. The
README's absent continuum, rare-event and convergence theorems are scientific
limitations, not instructions to assert or prove those results as part of this
implementation. Numerical gates that remain unresolved must remain explicit.
