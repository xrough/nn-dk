# Matched collocation residual supplement

All 72 declared models remain, with all three optimizer seeds and all four proposal/loss variants. The component table retains all four means and all sixteen covariance entries in solver-raw, physical-raw and stored training-normalized units.

These models were selected on this natural validation pool. These are descriptive selection-pool diagnostics, not independent accuracy measurements, field bias estimates, 95% significance tests or an error/continuum certificate. No new physical continuations, labels or final Monte Carlo data are used.

All nineteen boundary/time strata appear. Unoccupied bins have count zero and undefined metrics, not evidence of zero residual. Stratification only rebalances occupied states; it cannot create absent boundary, corner, absorbed or zero-time states. Empty strata therefore supply no evidence for those states. Distinct parent counts describe grouping; snapshots and remaining-time replicas are not independent families, and parent counts must not be summed across bins.

Natural-population MSE and pseudo-Huber contributions use row occupancy and all four/sixteen entries. Their sums reconstruct both canonical global validation losses; an unweighted average of occupied-bin losses would change the measure. Both common-unit MSE and pseudo-Huber loss are retained for every variant.

The physical-raw mean residual is the solver residual times bank scale/√N; physical covariance residual uses bank-scale outer product/N. Training-normalized terms use T times the solver term divided by its original stored stage scale; no new fit, inverse-time weighting, clipping or normalization is introduced.

Stored loss units are common across the crossed variants within each case/stage; magnitudes across different physical regimes are not independently measured fluctuation errors. Preparation and residual-extraction attempt costs are separate diagnostic overhead in audit_costs.csv.

Covariance source Γ(u), operator L C and time derivative are shown separately. Their RMS magnitudes do not add to the residual loss. Γ_common and rC_common substitute the matched-seed uniform/pseudo-Huber mean only as a counterfactual source diagnostic; other variants were not trained with that common mean.

The ten unique covariance entries are a plot convenience with diagonal multiplicity one and off-diagonal multiplicity two. All sixteen signed entries remain in the compressed CSV. `component_statistics.csv.gz` is streamed losslessly; raw per-row arrays, state banks and checkpoints remain local and checksum-indexed.

| Case | Validation rows | Distinct parents | Occupied / 19 |
| --- | ---: | ---: | ---: |
| T0p25_k0p75 | 4096 | 256 | 4 / 19 |
| T0p25_k3 | 4096 | 256 | 4 / 19 |
| T0p25_k6 | 4096 | 256 | 4 / 19 |
| T1_k0p75 | 4096 | 256 | 4 / 19 |
| T1_k3 | 4096 | 256 | 4 / 19 |
| T1_k6 | 4096 | 256 | 11 / 19 |
