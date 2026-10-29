# Cross-stage research provenance

This directory preserves compact operational receipts and logs that connect the research stages. The numerical reports and their separate archives retain the scientific estimates, uncertainty, source snapshots and local raw-file inventories. Full HDF5/NPZ data, checkpoints and trajectory banks remain local.

This collection does **not** mark R1–R6 or the README research goal complete. At creation, final accuracy and cost reporting remained pending. The parent completion record must add its terminal final-stage receipt and final publication evidence later.

- `actual/` records the 90-fit joint freeze before final values, all six terminal reference workers and their integrity receipt, the full 72-model validation residual stage, and unchanged-byte verification of the original 15-fit/26-file path archive.
- `tiny_v2/` records a separate reduced software fixture and its complete fit/lock/reference/accuracy/timing/report/archive workflow. Its earlier accuracy-only receipt remains incomplete for timing by design; `joint_end_to_end_acceptance.json` records the later completed software acceptance. Neither receipt supplies the full scientific allocation or a scientific speed claim.
- `clean_clone_0112776/` records the clean full-history clone, locked installation, wheel/source build, documented smoke run and wheel import/CLI checks at implementation commit `0112776`. Build products remain local and checksum-described in the receipt. This is software verification, not package-registry publication.

The original `actual_joint_freeze.json` deliberately retains `complete: false`: its prepare and freeze commands both exited zero, but the outer postcheck logger looked in the wrong location for the collocation evaluation lock. `actual_joint_freeze_verified.json` corrects only that logger postcheck using the authoritative `comparison/evaluation_lock.json`; it binds the original receipt and verifies the unchanged joint lock before any final values. No work command or physical draw was repeated for this correction.

The production residual receipt preserves the original generated narrative SHA. The maintained narrative may later append a separately labeled interpretation; that does not change the archived generated report or this historical receipt. The final composite record must hash the current maintained note and identify that appended interpretation separately.

`manifest.json` lists every copied payload's original absolute path, byte count and SHA-256. Copied and original SHA-256 values are identical. Generated README bytes have a separate digest. Absolute paths in preserved receipts are provenance records, not portable input locations.

The collection excludes process snapshots, credentials, large numerical files, dynamic worker journals and unfinished `actual_final_stages.json`. A parent may later append stable terminal receipts by adding payloads and manifest entries, preserving all existing copied bytes and source identities. Update the generated-file digest if this wrapper changes, then verify the exact complete file inventory again.

See the [task ledger](../../../README_TASKS.md), [validation residual archive](../payoff_collocation_residuals_2026-10-30/README.md) and [gradient/transport archive](../payoff_diagnostics_2026-10-29/README.md). The final collocation and correction archives are separately allocated as `payoff_collocation_2026-10-30` and `lna_payoff_correction_2026-10-30`.

Project-authored prose, figures and tables are CC BY 4.0; code is MIT. See [license coverage](../../../../LICENSING.md). This wrapper and preserved receipts add operational provenance without changing earlier numerical archives.
