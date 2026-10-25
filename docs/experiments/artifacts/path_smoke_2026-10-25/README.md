# Density-path smoke archive — 25 October 2026

Five actual fitted path operators completed the G32, N2048, T=1/16 software
check. All five passed the declared comparisons against **same-step Euler**.
The largest full-state normalized strong RMS was 0.2692%; the largest
NN/fine-Euler strong difference was 35.95%. These statements concern different
targets. No continuous-time DK accuracy or convergence rate follows.

The [method](../../../PATH_SOLVER.md),
[mathematical audit](../../../PATH_OPERATOR_AUDIT.md), and
[complete table](path_table.csv) explain the target and normalization.

- `findings.json`: compact measurements, with optimizer seeds retained.
- `path_results.json.gz`: lossless full report, including signed mean/covariance
  contrasts, pointwise uncertainty, stopping diagnostics and provenance.
- `path_results.md`, `path_table.csv`, and PNGs: human-readable report and actual
  held-out full-grid density realizations.
- `config.json`, `preregistration.json`, `calibration.json`, `manifest.json`:
  the exact design and final locks.
- `checkpoints.json`: checkpoint paths, byte hashes and training summaries;
  `checkpoint_training_history.json.gz` preserves all training histories.
- `environment.json`, `archive_manifest.json`: environment and checksums.

Full restorable checkpoints, physical input banks, raw proposals and frozen
paths remain in `runs/path-smoke-2026-10-25-v2/` in the original checkout.
They are not duplicated in this compact archive. The manifest retains their
hashes. The selected plotted realization is trajectory zero, seed zero; it
was selected by index rather than accuracy.

Source freeze: `bd256de`; package source hash
`a37ebd4fa82c8c590922417004d47150b1ea71d75be635a316b670506053223a`.
Reproduce with the committed Python 3.12 lock and
`nn-dk path-solver --preset path-smoke --output runs/path-smoke-reproduction`.
The same configured smoke inputs were used in the earlier pre-calibration-fix
execution; this repeated smoke is not a fresh model-selection experiment.
