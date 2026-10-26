# Project A: full density-path pilot

Evidence prepared from `/Users/xiaohao/Projects/Finance/Covariance/NN-DK/runs/path-first-batch-2026-10-25-v2`; 15 fixed-weight fits and 15/15 same-step Euler accuracy gates passed. Continuous-time stopped-DK and continuum-SPDE accuracy remain uncertified. The archive is complete only when its `archive_manifest.json` and listed files are present.

- [Machine findings and cost scopes](findings.json)
- [Complete report](path_results.md) and [all optimizer seeds](path_table.csv)
- [Warm validation timing diagnostics](path_timings.json)
- [Checkpoint metadata](checkpoints.json) and [checksums](archive_manifest.json)

Per-fit training wall times overlap across three one-thread seed workers. CLI elapsed excludes those standalone intervals; monotonic active parallel span, worker process-CPU time and UTC wall span are reported separately. UTC clock gaps do not by themselves establish suspension or count against active execution. Workflow completion excludes later plotting, timing and archival. Inference timings exclude training, model loading, initialization, generation and bootstrap; restoration/initialization are separate. Validation strong point errors are not matched continuous-time or marginal-law certification.
