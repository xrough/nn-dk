# README task and evidence ledger

Audit date: **29 October 2026**, starting from commit `d4bd378` and the local
[`README_FINDINGS.md`](../README_FINDINGS.md). This ledger tracks the active
request to implement the README plans. It does not mark that request complete.

Current status was updated on **30 October 2026** after the external collector
verified all ten terminal commands, strict allocations, reports and compact
archives. The bounded R1–R5 investigations and the computational part of R6 are
complete. **Final R6 documentation/publication signoff remains pending**: the
reviewed public `main`, its exact-commit CI and a fresh full-history clone still
need verification. The overall request is not yet marked complete.

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
recorded historical revision. Later research documentation is tracked separately.
The latest local fit lock at that historical audit records all 18 models as
complete without seed pooling or selection from final outcomes.

These checks validate the compact evidence and its recorded provenance. They
do not constitute a new full-suite test run, a replay of every large local
HDF5/checkpoint bank, or a new scientific experiment. The recorded 1,181-test
result and full numerical replays belong to the completed boundary study.
Earlier archive READMEs are evidence/reproduction notes, not additional pending
training allocations.

## Bounded investigation status and final signoff

The priorities came from
[findings, section 6](../README_FINDINGS.md#6-practical-conclusions-and-next-work).
They have now been tested at their declared allocations. Completion of an
investigation is distinct from passing its accuracy or cost gates; no positive
neural advantage was required.

| ID | Requirement | Verified bounded result | Evidence and remaining limits |
| --- | --- | --- | --- |
| R1 | Investigate mean-gradient accuracy in the covariance source | **Diagnostic complete.** All 768 chunks, 12,288 fresh families, 24 validation anchors and 72 model comparisons completed, including signed components and paired refinement. | [Gradient/transport report](experiments/payoff_diagnostics_2026-10-29.md) and its [archive](experiments/artifacts/payoff_diagnostics_2026-10-29/README.md). Derivative interchange and remaining stopping/time bias are unresolved; this is not an all-state or continuum certificate. |
| R2 | Investigate boundary-layer sampling | **Controlled study complete.** All 72 fits, four proposal/loss variants, three seeds, independent accuracy, 696 timing workloads and 156 restorations are retained. The same natural validation measure, initial weights and train-only scales are used within each matched case/seed. | [Collocation report](experiments/payoff_collocation_2026-10-30.md), [19-stratum supplement](experiments/payoff_collocation_residuals_2026-10-30.md) and both compact archives. Natural-pool reweighting does not create absent boundary states. No general accuracy improvement or qualified payback was established. |
| R3 | Investigate objective alignment with fluctuation accuracy | **Transport and matched comparison complete.** The transport audit retains 192 chunks, 1,536 fresh families and 72 model comparisons. All 72 collocation models retain their own selected means during covariance fitting. In T1_k6, stratified/squared covariance discrepancy is 3.54–6.97 reference-SD products despite lower validation residual MSE. | [Transport evidence](experiments/payoff_diagnostics_2026-10-29.md), [decomposition](experiments/payoff_collocation_residuals_2026-10-30.md) and [independent accuracy](experiments/payoff_collocation_2026-10-30.md). Common-Γ substitution is diagnostic only; the other covariance models were not trained with a common mean. A residual is not an error certificate. |
| R4 | Test learned corrections to the existing nonlinear-LNA baseline | **Full test complete with a retained zero-correction result.** All 18 fits executed 1,000 Adam updates and 30 L-BFGS outer steps in each stage, 2,060 updates/model. Natural validation selected step 0 in both stages for all seeds. All 216 new method/group predictions are available; their trained point moments equal `gated_zero`. | [Correction report](experiments/lna_payoff_correction_2026-10-30.md) and [archive](experiments/artifacts/lna_payoff_correction_2026-10-30/README.md). All 307,200 label families, 12 role banks, 24 independent package prefix replays and full paid optimization remain recorded. Baseline gains over collocation or analytic IID Gaussian controls are not learned-head gains. |
| R5 | Freeze validation choices before fresh final data | **Protocol and final comparison verified.** Both full fit locks and the exact source/criteria/identifier plans preceded final materialization. All six reference banks contain 589,824 unique families at primary step `2^-12`; the 36,864 coupled-prefix families are reused at `2^-13` and `2^-14`. Independent integration pools and all stopped paths remain. | [Computational collector](experiments/artifacts/readme_research_2026-10-30/final_computational/final_computational_evidence.json), [terminal receipts](experiments/artifacts/readme_research_2026-10-30/final_computational/actual_final_stages.json) and final reports. Sampling and refinement diagnostics do not certify continuous hitting, a particle law or a continuum limit. |
| R6 | Finish comparison, cost accounting, documentation and publication | **Computation verified; final publication signoff pending.** The reports contain 792 joint primary / 948 retained predictions, 1,080 contrast packets / 6,480 signed statistic rows, and 288 correction cold/warm jobs. Collocation has zero qualified payback rows out of 2,160; correction has zero out of 432. | The [public proof collection](experiments/artifacts/readme_research_2026-10-30/README.md) preserves the terminal evidence. Final documentation links/hashes, the [composite record](experiments/readme_research_2026-10-30_completion.json) and exact public `main`/CI/fresh-clone verification still require owner signoff. Historical CI is not proof for a later publication commit. |

The existing [`native_gaussian_baseline.py`](../analysis/native_gaussian_baseline.py)
already evolves a nonlinear deterministic mean and linear-noise covariance along
it. The correction experiment reused that baseline, rather than relabeling its
nonlinear evolution as a new learned result. Completed optimization and retained
checkpoints are separate facts: the 18 fits completed all 37,080 outer updates
in total, while validation retained their initialized zero corrections. This
negative learned-head result keeps every seed and all paid labels/fit costs.

The residual supplement uses the unchanged natural validation rows, all fitted
variants and seeds, and the original 19 boundary/time strata. Each case has
4,096 rows from 256 parent families; snapshots and remaining-time replicas share
parents. Its 570,240 component-summary and 2,880 stratum-loss rows retain empty
bins as undefined, all four mean components and all 16 covariance entries. These
selection-pool diagnostics have no independent accuracy or significance claim.
Physical residuals use bank scale/√N for means and the scale outer product/N for
covariance; the common training units remain those frozen before fitting.

The independent comparison covers each profile's conditional anchors 0 and 1
and its ensemble query 0, at T/4, T/2 and T. Ensemble 0 is not another conditional
anchor. Primary IID neural moments integrate fresh histograms independent of
reference, with whole-family uncertainty; paired-initial collocation outputs remain
retained diagnostics. The correction exposes only the fixed four-payoff bank and
same-time covariance, not arbitrary observables, density paths, complete derivatives through the
supplied baseline ODE or a generator interface.

The [collocation report](experiments/payoff_collocation_2026-10-30.md) and
[correction report](experiments/lna_payoff_correction_2026-10-30.md) preserve failed
and unresolved 5%/1.25% gates, not just point errors. No A groups are unavailable
and physical moment-bound checks pass, but neither framework has a qualified
payback row. C qualification uses coarse-versus-finest reference refinement; the
correction uses all matching refinement rows. Signed intervals remain pointwise,
and seed consistency is not simultaneous inference across the grid.

Costs include preparation, full fit invocations, initialization and literal
inference. Shared components are not added again to their parent totals. Every
correction timing evolves its per-histogram ODE uncached, including all 16,384
histograms for an IID call; the accuracy-only shared cache is not free deployment
work. Its one cold and one warm observation do not estimate a latency
distribution. The correction payback table compares only gated zero and raw
conditional LNA7/10; accuracy contrasts also retain collocation and classical controls. That payback
table does not report a timing crossover against collocation.

## Computational evidence and publication boundary

The [collector](experiments/artifacts/readme_research_2026-10-30/final_computational/final_computational_evidence.json)
(SHA-256 `afde909b15905e075c39f6756c90078f7ce1931cbb198c10a44b0a5d9ffaaf4c`)
verified all ten canonical commands, report/source identities, allocations,
matching 1,502-test/287-input receipts and the archives below. The numerical
package source is `0d9bda04e5da1cc00cc3ec5917d23a4352e6ceb4585dd1a066779b364cf79576`;
analysis helpers have their own frozen hashes and exact snapshots. Older studies
retain their recorded source/test identities. The software/full-clone proof at
`011277684b540860379eb726b2440a3fd0385226` is not reassigned to a later commit.

| Archive | Copied artifacts | Indexed local raw files | Manifest SHA-256 prefix |
| --- | ---: | ---: | --- |
| [Collocation](experiments/artifacts/payoff_collocation_2026-10-30/archive_manifest.json) | 1,169 | 1,937 | `9cff4ebdf90919e0` |
| [Correction and comparative context](experiments/artifacts/lna_payoff_correction_2026-10-30/archive_manifest.json) | 18,236 | 35,418 | `afa571a9322967e2` |
| [Residual supplement](experiments/artifacts/payoff_collocation_residuals_2026-10-30/archive_manifest.json) | 2,865 | 5,048 | `597188eb34323e45` |
| [Earlier gradient/transport diagnostics](experiments/artifacts/payoff_diagnostics_2026-10-29/archive_manifest.json) | 1,122 | 1,955, plus 46 external inputs | `4855fa60333b6885` |

These checks verify copied bytes, restored compressed bytes and the local raw
inventory; full checkpoints, trajectory banks, labels and IID arrays remain
local. They are not new simulations. The
[public cross-stage collection](experiments/artifacts/readme_research_2026-10-30/README.md)
preserves the earlier payloads and appended terminal evidence without rewriting
historical scientific records.

**Pending R6 signoff:** the owner must finish the composite documentation record
and verify the reviewed public `main`, Actions at that exact commit and a fresh
full-history clone. Record that later proof in
[verification](verification.md#final-research-publication-signoff) and the
[composite completion record](experiments/readme_research_2026-10-30_completion.json)
before marking the overall README request complete.

## Execution and acceptance

The completed computation followed these rules; final publication signoff above
is still open. They do not prescribe another training run after a negative
result.

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
