# Frozen-v3 gradient and residual-transport diagnostics

Date: **29 October 2026**. These are diagnostics at validation-state anchors, not a new final accuracy/cost comparison. No model was fitted or selected, and no historical final reference outcomes were reopened.

## Frozen allocation and controls

The mean-gradient diagnostic retains all six horizon/interaction regimes and all three optimizer seeds: 24 state-only validation anchors, 512 fresh continuation families per anchor (12,288 total), 32 physical noise-face directions, and three remaining horizons. All 768 chunks and 72 model/anchor comparisons completed. A coupled half-step reference uses 64 families per anchor. Whole-family bootstrap uses 128 resamples.

The independent residual-transport diagnostic uses 64 fresh families per anchor (1,536 total), 192 complete chunks and the same 72 model/anchor comparisons. It pairs Euler steps 2^-10 and 2^-11 and residual quadrature strides 16/8 on the coarse grid, using the same physical quadrature nodes on the refined grid. All families, including stopped paths, remain in every estimate.

Native JVPs differentiate both interpolated terminal states and the killed running-payoff exposure. The maximum recorded common-noise finite-difference discrepancy is 1.08e-11. Exact zero-time face-gradient checks have maximum error 4.44e-16; initially absorbed face gradients are exactly zero. These implementation checks do not prove finite-horizon learned-gradient accuracy.

## Mean gradients and the covariance source

The model mean is normalized as sqrt(N) times the physical payoff mean divided by the fixed bank scales. Its face derivatives therefore use the physical noise amplitude divided by sqrt(N). The table shows ranges of entrywise gradient RMS discrepancies and the largest covariance-source Frobenius discrepancy across query horizons, anchors and seeds. These units are not the reference-SD percentages in earlier accuracy reports.

| T | κ | Gradient RMS discrepancy range | Largest source Frobenius discrepancy |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.75 | 0.000904449–0.00368627 | 0.0423237 |
| 0.25 | 3 | 0.00110107–0.0157593 | 0.147601 |
| 0.25 | 6 | 0.00363446–0.116066 | 1.76447 |
| 1 | 0.75 | 0.00273069–0.0119727 | 0.103559 |
| 1 | 3 | 0.0114566–0.110618 | 1.66427 |
| 1 | 6 | 0.0501692–19.2993 | 47580.4 |

The largest discrepancy occurs at a nonuniform T=1, κ=6 validation anchor with minimum mass 3.87 times the stopping mass. All 512 diagnostic continuations stop by the last query. Its largest learned-gradient RMS discrepancy is 19.30; the corresponding reference bootstrap sampling RMS is about 0.00691. This is descriptive finite-Euler evidence of a severe derivative/source problem, not a continuum or all-state claim.

The independent-half symmetric cross-Gram reference avoids the positive quadratic plug-in Monte Carlo bias and can be indefinite. Both estimates are retained. Signed componentwise intervals, raw arrays and paired refinement diagnostics are in the [gradient report](artifacts/payoff_diagnostics_2026-10-29/mean_gradient/report/report.md). Componentwise exclusion counts are correlated and pointwise; positive Frobenius-distance ranges are never used as tests of nonzero bias. Derivative interchange and remaining stopping/time bias remain unresolved.

## Does the residual explain prediction error?

For the normalized mean u, covariance C and accumulated running payoff A, the raw-second transport residual is r_C + (u+A)r_mu^T + r_mu(u+A)^T. The covariance residual r_C alone omits the mean-source coupling. Raw second moments and sample covariance are reported separately.

The table uses maximum absolute entries across all validation anchors, queries and seeds on the finer Euler grid with half-stride quadrature. Each column has its own maximum; columns need not attain it at the same model or component.

| T | κ | Mean discrepancy | Mean transport defect | Raw-second discrepancy | Raw-second transport defect |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.75 | 0.175214 | 0.178827 | 0.88587 | 0.87478 |
| 0.25 | 3 | 0.226788 | 0.234364 | 2.64203 | 2.4878 |
| 0.25 | 6 | 0.256021 | 0.293217 | 4.12132 | 3.19351 |
| 1 | 0.75 | 0.248204 | 0.249569 | 1.35383 | 1.2907 |
| 1 | 3 | 0.483301 | 0.267061 | 7.67477 | 3.14649 |
| 1 | 6 | 18.6528 | 17.6735 | 379.661 | 491.07 |

The hardest regime does not resolve the transport remainder: halving the residual quadrature stride changes some mean/raw-second defect RMS values by as much as 5.52/102.27, while Euler refinement changes them by up to 1.13/64.26. Sparse quadrature, finite-Euler generator mismatch and stopping effects therefore prevent an error certificate or unique causal attribution. Many mild-case intervals remain unresolved; some exclude zero pointwise, so those defects cannot all be attributed solely to sampling.

The [transport report](artifacts/payoff_diagnostics_2026-10-29/residual_transport/report/report.md) retains both levels, both quadratures, signed paired bootstrap contrasts, stopped-family counts and all seeds. A smaller collocation loss or a correct initial slope does not establish accurate finite-time covariance.

## What remains

The boundary/time sampler and grouped stress-state utilities are implemented, including exact importance weights applied after the per-row robust loss. The distinct staged trainer, locked evaluator and reporter for the common-pool cross of uniform/stratified proposals and pseudo-Huber/squared losses pass software tests. Its controlled scientific fits, independent final accuracy and isolated cost comparison are not complete.

The existing Gaussian baseline already evolves a nonlinear deterministic mean. A learned correction to that baseline remains an untested candidate. It must be assessed with declared training costs, validation choices frozen before fresh final outcomes, and its unstopped approximation distinguished from the stopped native target.

See the [task ledger](../README_TASKS.md) for the full remaining scope. The [compact evidence archive](artifacts/payoff_diagnostics_2026-10-29/README.md) preserves exact frozen code and checksummed diagnostic metadata; large raw arrays and weights remain local. None of these diagnostic results marks the active README implementation goal complete.

The archive's `code/` directory is a partial source snapshot. Its preserved
licensing note refers to documentation outside that snapshot; use the canonical
[DK guide](../DK_BACKGROUND.md), [provenance note](../provenance.md) and
[software citation](../../CITATION.cff) in this repository for those references.

## Follow-up — 30 October 2026

The work listed above as remaining on 29 October has since completed its
declared allocation. The [matched collocation study](payoff_collocation_2026-10-30.md)
retains all 72 fits, independent accuracy and isolated costs. The
[conditional-LNA correction study](lna_payoff_correction_2026-10-30.md) retains
all 18 completed optimizations; natural validation selected the initial zero
correction in both stages for every fit. Neither study establishes a general
learned accuracy advantage or accuracy-qualified payback. The
[composite research record](readme_research_2026-10-30_completion.json) and
[task ledger](../README_TASKS.md) record the later evidence and publication
signoff. The dated diagnostic results and their immutable archive remain
unchanged; the continuum and stopping limitations above still apply.
