# Independent payoff formulation repair evidence

The [report](report/report.md) records all declared repaired, ablation, historical and classical models, independent accuracy and uncertainty, training and inference costs. No optimizer seed or formulation was selected or refitted using final reference outcomes.

All report tables/figures, current JSON records, training-stage histories, audits, exact code and historical checkpoint sidecars are retained. Large JSON is deterministically gzipped; stored and restored original checksums are both verified. Full local trajectory banks, Brownian increments, weights, numerical arrays and compiled libraries are indexed without duplication. Immutable metadata paths and bytes are preserved.

This fixed-grid, short-horizon comparison makes no continuum-DK or continuous-stopping certificate. Paired error uncertainty, standalone Monte Carlo variability, reference refinement and approximate-law bias remain distinct. See amortization.csv for unresolved gates.

Code uses the project MIT license; authored prose, tables and figures use CC BY 4.0.

Verify with `.venv/bin/python analysis/archive_payoff_repair.py --verify-archive docs/experiments/artifacts/payoff_repair_2026-10-27`.
