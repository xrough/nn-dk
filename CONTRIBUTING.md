# Contributing

Use Python 3.12 and the committed lockfile from a full-history Git clone:

```bash
git clone https://github.com/xrough/nn-dk.git
cd nn-dk
uv sync --locked --group dev --python 3.12
uv run ruff check .
uv run pytest
```

Historical-source loading and its tests execute package snapshots from recorded
Git commits. Avoid shallow clones; repair an existing one with
`git fetch --unshallow origin`. A source distribution or installed wheel alone
does not include that Git history or the full experimental inputs.

Keep changes focused and explain the numerical or statistical behavior they
affect. Run Ruff and pytest before submitting a pull request. Add targeted tests
for changed numerical behavior, integrity checks or statistical guarantees;
documentation-only changes need link and example checks. Do not rewrite history
containing the source revisions used by frozen checkpoints.

## Reproducible experiments

- Keep numerical source, configuration and helper files fixed while a run is
  active. Their hashes identify the experiment. Use a separate directory under
  `runs/` for a changed design; never edit locks or hashes to resume incompatible
  artifacts.
- Preserve checkpoint bytes, metadata and recorded source commits. Use the
  explicit historical loader when deriving a supported current runtime from an
  older checkpoint; do not bypass source or physical-model validation.
- Declare physical parameters, grids, path counts, whole-family splits, random
  streams and optimizer seeds before inspecting final outcomes. Keep optimizer
  seeds separate; select models on validation data and retain stopping paths.
- State the target's stopping rule and readout convention, including frozen
  density paths versus killed running payoffs. Report failures and conservation
  diagnostics without dropping or repairing paths to improve an estimate.
- Distinguish conditional and IID-ensemble statistics, same-time and joint-time
  covariance, paired errors and standalone estimator uncertainty. Resample whole
  trajectory families; state whether intervals are pointwise or simultaneous.
  Finite reference refinement does not certify continuum or stopping-time bias.
- Time final inference without competing training/evaluation jobs. Record warmup,
  repetitions, backend and supplied inputs. Separate preparation, loading,
  initialization, fitting, inference and aggregation costs; use matched tasks and
  accuracy requirements for speedup claims.

Commit small synthetic fixtures or compact, checksummed evidence with provenance
and a clear findings note. Keep full checkpoints, trajectory banks, environments,
caches and generated runs out of Git; `runs/` and these build/runtime outputs are
ignored. Do not mark a study complete until its declared fits, evaluations and
independent audits are complete. See [analysis instructions](analysis/README.md)
and the [path solver guide](docs/PATH_SOLVER.md) for workflow-specific contracts.

## Licensing and attribution

Contribute code under the project's MIT license and project-authored documentation
under CC BY 4.0. Preserve existing notices and credit papers, thesis material and
coauthors where their work is discussed. Check third-party terms before adding
material; cite and adapt only what is needed for the project.
