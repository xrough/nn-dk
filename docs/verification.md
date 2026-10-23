# Verification workflow

```bash
uv sync --locked --group dev
uv run ruff check .
uv run pytest
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
