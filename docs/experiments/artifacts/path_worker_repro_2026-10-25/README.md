# Path-worker software reproducibility audit

**Software validation only.** This is not a DK accuracy, convergence, stability,
or speed benchmark. The independent test used `G=8`, `N=2048`, `T=1/16`,
`dt=1/128`, width eight, depth two, 12 training paths, and eight validation
paths. Both directories used identical physical input arrays, all five cases,
optimizer seeds zero and one, 40 AdamW steps and 20 LBFGS calls with at most
five inner iterations. BLAS and Torch used one thread per process.

The controls were actually fitted and passed their validation gates before
starting the unchanged worker helper in two separate subprocesses. Worker zero
resumed an incomplete DK checkpoint after **seven AdamW updates**.

All **10 sequential/worker model pairs** matched bitwise in current/best
parameters and hashes, optimizer states, Torch/shuffle RNG states, selected
validation state, provenance, and complete numerical training/validation
histories. Timing fields were excluded. Each observed seed-lock PID matched
its actual child-process handle, both subprocesses exited zero, and existing
common files were unchanged. **No final bank, fit lock, or scientific final
inputs were generated or read.** The configured test allocation was unused.

Files:

- [result.json](result.json): actual comparisons, identities, environment,
  process/lock observations, and the seven-update partial checkpoint.
- [config.json](config.json): complete shared software configuration.
- [worker_0.log](worker_0.log), [worker_1.log](worker_1.log): actual worker logs.
- [checksums.json](checksums.json): archive checksums and external source hashes;
  the checksum file excludes itself.

Reproduce from the repository root, using a new output directory:

```bash
.venv/bin/python analysis/audit_parallel_path_fitting.py \
  --output runs/path-worker-repro-replay
```

The audit script SHA-256 is
`5f3a2a5fc922495726ff652c4166a6fdea165a477193052dd641a23122619ebd`;
the unchanged worker helper SHA-256 is
`35f436f156729e1c3f040824d52945e098e704784a9e45e4b603777bb5facd47`.
The package source identity is
`a37ebd4fa82c8c590922417004d47150b1ea71d75be635a316b670506053223a`.

Raw software checkpoints, banks, and source snapshots remain in
`runs/path-worker-repro-2026-10-25-v2` and are not duplicated here. The earlier
local audit directory remains unchanged. See the
[operator audit](../../../PATH_OPERATOR_AUDIT.md) for mathematical scope,
ownership preconditions, and wall-time accounting qualifications. This small
test does not establish bitwise reproducibility on arbitrary hardware.
