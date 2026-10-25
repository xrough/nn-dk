# Density-path refinement smoke archive — 25 October 2026

Fifteen actual fitted path operators completed a short T=1/16 workflow across
five cases and three separately fitted steps, 2^-7, 2^-8 and 2^-9. All passed
the declared **same-step Euler** accuracy gates. Each case has 32 training,
16 validation and 64 final independent trajectory families. Every optimizer
seed is retained; this smoke uses only seed zero.

The maximum neural/same-step strong RMS was 0.3161%. The maximum finite
Euler-reference strong differences decreased from 36.53% to 15.15% to 5.702%
across the steps. The reference is only 2^-10, and the stored time grid also
changes. These are execution/refinement diagnostics, not an estimated
continuous-time convergence rate. The finest stochastic comparisons still
exceed the 5% target.

See the [method](../../../PATH_SOLVER.md),
[mathematical audit](../../../PATH_OPERATOR_AUDIT.md), and every individual
comparison in [path_table.csv](path_table.csv).

`path_results.json.gz` preserves the lossless complete report. The Markdown,
CSV and PNGs provide compact views. Configuration, preregistration, calibration,
manifest, checkpoint metadata, environment and archive checksum files preserve
the design and provenance. `checkpoint_training_history.json.gz` retains all
optimizer histories; `checkpoints.json` provides readable summaries.
Full checkpoints and full input/output HDF5 banks
remain in `runs/path-refinement-smoke-2026-10-25/`; their hashes are retained.

Source freeze: `bd256de`; package source hash
`a37ebd4fa82c8c590922417004d47150b1ea71d75be635a316b670506053223a`.
Reproduce with the committed Python 3.12 lock and
`nn-dk path-solver --config configs/path_refinement_smoke.toml --output runs/path-refinement-smoke-reproduction`.
