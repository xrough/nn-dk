# Structured generator accuracy archive — 25 October 2026

All six fits and 15 final comparisons passed the prespecified moment accuracy gates. Read the [experiment report](../../generator_accuracy_2026-10-25.md) and [main README](../../../../README.md) for interpretation and limitations.

- `findings.json`: concise measurements, initial/replay cost and completion audit.
- `generator_table.csv`, `generator_bounds.png`: every final comparison and bounds.
- `generator_results.json.gz`: losslessly compressed complete generated JSON report, including all signed errors, reference enclosures, seeds and manifest. Decompress with Python `gzip` or `gzip -dk`.
- `config.toml`, `config.json`, `preregistration.json`, `manifest.json`, `calibration.json`: exact experiment design and provenance.
- `inputs/`: development/final initial states and content hashes. These contain initial states, not simulated future paths.
- `reference/`: all four independent final DK reference packets, including continuous stopping and arithmetic certificates.
- `checkpoints.json`: six local checkpoint paths, byte hashes and training summaries. Complete checkpoints and optimizer state remain in `runs/generator-accuracy-2026-10-25/checkpoints/` in the original checkout; binary models are not duplicated in this compact archive.
- `environment.json`, `archive_manifest.json`: environment and file checksums.

Source freeze: commit `49a6528`. Reproduce with Python 3.12 and the committed `uv.lock`, then run `nn-dk generator --config configs/generator_accuracy.toml --output runs/generator-accuracy-reproduction`. Resume an identical command; use a new directory after changing source/configuration. All package source remains unchanged in the report commit.

The reference enclosure assumes the documented IEEE binary64/Decimal arithmetic model and audited fixed-grid scope. A pass is not a continuum DK theorem, density-path result, or population-generalization confidence statement.
