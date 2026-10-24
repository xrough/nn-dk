# Analysis of completed Project A runs

Run these commands from the repository root, using its locked Python environment:

```bash
.venv/bin/python analysis/summarize_first_batch.py runs/a-first-batch
.venv/bin/python analysis/probe_uniform_generator.py runs/a-first-batch --plot
.venv/bin/python analysis/time_native_solver.py runs/a-first-batch --include-neural
```

The summary reads saved JSON only. It checks completion and the exact declared allocation, preserves individual optimizer seeds, and writes `report/findings_summary.json` and `report/findings_table.csv`. Whole-ensemble uncertainty, paired contrasts, reference bias, and overall accuracy gates remain distinct. It never fits or resimulates a model.

The uniform-state probe loads completed coefficient checkpoints and differentiates their generators. Its Fourier eigenvalues and frozen-noise covariance describe a local interior linearization. They do not certify nonlinear rollouts, stopped dynamics, or continuum convergence. Source, configuration, and checkpoint hashes identify the calculation.

Standalone timing requires a completed primary run and matching numerical source. Run it after other CPU work has finished. It measures three repeats at each declared native timestep and, optionally, every complete neural checkpoint at the finest step. Costs include histogram initialization, internal random-number generation, and the current shared simulation backend. Neural rollouts also incur that backend's native-coefficient calculation before the neural coefficients override it. Model loading is recorded separately. These are specified-step timings, without an accuracy-matched speedup claim.

Numerical source and configuration must stay unchanged while an experiment runs. Derived analysis files do not change the package source hash. The saved experiment note links the compact archived findings; full trajectory caches and checkpoints remain under the ignored `runs/` directory.
