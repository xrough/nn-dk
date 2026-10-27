# Broader interacting DK task comparison evidence

The completed [report](report/report.md), 25 report outputs, all current
scientific JSON, exact analysis-code snapshots and independent audit scripts
are preserved here. Historical path-checkpoint metadata is in
`historical_inputs/`; its original absolute path and checksum remain recorded
in `archive_manifest.json`. Immutable original metadata has not been rewritten.

The final snapshot includes the optional T=1 model's original training-metadata
provenance in addition to the eighteen short-horizon path checkpoints. An
earlier review snapshot was preserved locally and replaced at this canonical
documentation location after that cost-provenance gap was corrected. No
primary numerical experiment or fitted model changed in this correction.

The comparison found no computational advantage for the learned SPDE path
solver over the compiled classical controls. Full-path reuse supports new
observables without retraining, and classical paths have the same capability.
The new statistics heads have cheap conditional inference but fail the stated
accuracy targets, especially for the mixed terminal/integral bank. The
no-training Gaussian approximation is the inexpensive statistics control in
this short, weak-noise setting, subject to its law and uncertainty limits.
These results describe the fitted configurations and measured tasks, not an
intrinsic ranking of neural and classical methods.

Copied Python/C++ source uses the project's MIT code license. Authored
experiment prose, tables and figures use CC BY 4.0, as specified in the
repository's `LICENSE-DOCS` and licensing guide.

Large JSON uses deterministic lossless gzip. Every stored checksum and the
checksum of every restored original are verified. Full HDF5 paths, Brownian
increments, model weights, NumPy arrays and compiled libraries remain in the
local run directories. Their complete checksummed inventory is retained here;
they are not duplicated into this compact archive. The incomplete superseded
timing attempt is excluded from copied evidence and explicitly identified in
the inventory. Old execution progress records are provenance, not current job
status.

All optimizer seeds remain separate. Whole-family sampling uncertainty,
standalone Monte Carlo variability, reference refinement and approximate-law
bias are distinct. Zero observed stops does not certify rare-event accuracy.
This fixed-grid study makes no continuum or continuous-stopping convergence
claim and establishes no accuracy-qualified training amortization crossover.

Verify the snapshot with:

```sh
.venv/bin/python analysis/archive_path_generalization.py --verify-archive docs/experiments/artifacts/path_task_advantages_2026-10-27
```
