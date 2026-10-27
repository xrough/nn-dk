"""Archive a complete, independently checked broader DK task comparison.

The compact snapshot preserves every current scientific JSON record, report
output, exact analysis-code audit and historical checkpoint metadata. Large
JSON is compressed without changing a byte. Arrays, weights and libraries are
indexed with checksums and retained in their original local run directories.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis.report_path_generalization import load_complete  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402

KIND = "broad-dk-task-comparison-evidence-snapshot-v1"
SUPERSEDED_PREFIX = "timing/path_workloads_superseded_readout"
COMPRESS_THRESHOLD = 100_000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _inside(path, root):
    return Path(path).is_relative_to(root)


def verify_ready(run, *, report=None, require_completion=True):
    """Check actual scientific evidence and exact report inputs before copying.

    This performs no training, trajectory generation or report regeneration.
    A preliminary read-only check may omit the final software-test record.
    The overall goal-completion audit is written after snapshot verification,
    avoiding a circular archive checksum dependency.
    """
    run = Path(run).resolve()
    report = Path(report or run / "report").resolve()
    if not _inside(report, run):
        raise ValueError("Keep the report inside its authoritative run directory")
    manifest_path = report / "manifest.json"
    manifest = read(manifest_path)
    if (manifest.get("kind") != "broad-DK-task-comparison-report-v1"
            or manifest.get("complete") is not True
            or manifest.get("source_hash") != source_hash()
            or manifest.get("continuous_DK_claim") is not False
            or manifest.get("optimizer_seeds_pooled") is not False):
        raise ValueError("Only a complete source-matching finite-grid report may be archived")
    generator = REPO / "analysis/report_path_generalization.py"
    if sha(generator) != manifest["generator_sha256"]:
        raise ValueError("The current report generator differs from the completed report")
    expected_outputs = manifest["outputs"]
    actual_outputs = {path.name for path in report.iterdir()
                      if path.is_file() and path.name != "manifest.json"}
    if len(expected_outputs) != 25 or actual_outputs != set(expected_outputs):
        raise ValueError("All 25 declared report outputs, with no stale extra outputs, are required")
    for name, digest in expected_outputs.items():
        if Path(name).name != name or sha(report / name) != digest:
            raise ValueError("A report output changed or escaped its report directory")
    # The original loader verifies the 12 fitted models, six reference groups,
    # 174 main results, both independent timings and 12 Gaussian quality audits,
    # including every referenced raw artifact and the exact analysis-code audit.
    evidence = load_complete(run, run / "timing/path_workloads.json",
                             run / "timing/payoff_benchmark.json", verify_artifacts=True)
    if (evidence["input_files"] != manifest["inputs"]
            or manifest["verified_artifact_count"] != evidence["verified_artifact_count"]
            or len(evidence["groups"]) != 6 or len(evidence["records"]) != 186
            or manifest["evaluation_methods"] != 186
            or set(manifest["evaluation_groups"]) != set(evidence["index"]["declared_groups"])):
        raise ValueError("The completed report no longer matches the complete experiment")
    for name, digest in manifest["inputs"].items():
        if sha(name) != digest:
            raise ValueError(f"A report input changed: {name}")
    historical = {Path(name).resolve(): digest for name, digest in manifest["inputs"].items()
                  if not _inside(Path(name).resolve(), run)}
    checkpoint_metadata = {path for path in historical if path.parent.name == "checkpoints"}
    historical_manifests = set(historical) - checkpoint_metadata
    if (len(checkpoint_metadata) != 19
            or any(path.suffix != ".json" or not _inside(path, REPO / "runs")
                   for path in historical)
            or any(path.name != "manifest.json" for path in historical_manifests)):
        raise ValueError("Nineteen historical checkpoint metadata inputs and declared run manifests are required")
    tests = run / "audit/final_tests_2026-10-27.json"
    if require_completion:
        if not tests.is_file():
            raise ValueError("Save the final software-test audit before writing the snapshot")
        tested = read(tests)
        if tested.get("complete") is not True or tested.get("source_hash") != source_hash():
            raise ValueError("The complete final test evidence must match the package freeze")
    return {"run": run, "report": report, "manifest": manifest,
            "report_manifest_sha256": sha(manifest_path), "historical": historical,
            "historical_checkpoint_metadata_count": len(checkpoint_metadata),
            "verified_artifact_count": evidence["verified_artifact_count"],
            "completion_required": require_completion}


def _copy_exact(original, destination):
    """Copy bytes or deterministic gzip and verify the restored original hash."""
    original, destination = Path(original), Path(destination)
    before = sha(original)
    compressed = original.suffix == ".json" and original.stat().st_size > COMPRESS_THRESHOLD
    if compressed:
        destination = destination.with_name(destination.name + ".gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        with original.open("rb") as source, destination.open("wb") as sink:
            with gzip.GzipFile(fileobj=sink, mode="wb", filename="", mtime=0) as zipped:
                shutil.copyfileobj(source, zipped)
        digest = hashlib.sha256()
        with gzip.open(destination, "rb") as restored:
            for block in iter(lambda: restored.read(1024 * 1024), b""):
                digest.update(block)
        restored_sha = digest.hexdigest()
    else:
        shutil.copyfile(original, destination)
        restored_sha = sha(destination)
    if restored_sha != before or sha(original) != before:
        raise ValueError(f"Copy did not retain the immutable original: {original}")
    return destination, {"sha256": sha(destination), "original_sha256": before,
                         "original_path": str(original.resolve()), "compressed": compressed,
                         "original_bytes": original.stat().st_size,
                         "archive_bytes": destination.stat().st_size,
                         "restored_original_verified": True}


def _selected(run, report):
    # Preserve historic progress packets verbatim. Their recorded running states
    # are superseded by the complete evaluation index, never treated as liveness.
    chosen = {path for path in run.rglob("*.json")
              if not str(path.relative_to(run)).startswith(SUPERSEDED_PREFIX)}
    chosen.update(path for path in report.iterdir() if path.is_file())
    chosen.update(path for path in (run / "audit").rglob("*")
                  if path.is_file() and path.suffix in (".py", ".cpp", ".md"))
    return sorted(chosen)


def _readme():
    return """# Broader interacting DK task comparison evidence

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
"""


def archive(run, output, *, report=None):
    """Create a fresh compact snapshot only after all readiness checks pass."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Preserve existing archives; choose a fresh destination")
    ready = verify_ready(run, report=report, require_completion=True)
    run, report = ready["run"], ready["report"]
    if _inside(output, run):
        raise ValueError("The archive must not become part of its own source inventory")
    selected = _selected(run, report)
    inventory = {str(path.relative_to(run)): {"sha256": sha(path), "bytes": path.stat().st_size,
                 "copied": path in selected,
                 "superseded_scientific_evidence": str(path.relative_to(run)).startswith(SUPERSEDED_PREFIX)}
                 for path in sorted(run.rglob("*")) if path.is_file()}
    for original, digest in ready["manifest"]["inputs"].items():
        path = Path(original).resolve()
        if _inside(path, run) and inventory[str(path.relative_to(run))]["sha256"] != digest:
            raise ValueError("An input changed while inventorying the completed run")
    for name, digest in ready["manifest"]["outputs"].items():
        if inventory[str((report / name).relative_to(run))]["sha256"] != digest:
            raise ValueError("An output changed while inventorying the completed run")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".path-task-archive-", dir=output.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        copied, historical = {}, {}
        for original in selected:
            relative = original.relative_to(run)
            target, packet = _copy_exact(original, staging / relative)
            copied[str(target.relative_to(staging))] = {**packet, "source_role": "current-run",
                                                        "original_relative_path": str(relative)}
        for original, expected in sorted(ready["historical"].items()):
            if sha(original) != expected:
                raise ValueError("Historical report metadata changed before copying")
            relative = original.relative_to(REPO / "runs")
            target, packet = _copy_exact(original, staging / "historical_inputs" / relative)
            name = str(target.relative_to(staging))
            copied[name] = {**packet, "source_role": "historical-report-input"}
            historical[str(original)] = {"archived_relative_path": name,
                                          "original_sha256": expected}
        for name in ("report_path_generalization.py", "archive_path_generalization.py"):
            target, packet = _copy_exact(REPO / "analysis" / name, staging / "code" / name)
            copied[str(target.relative_to(staging))] = {**packet, "source_role": "archive-or-report-code"}
        (staging / "README.md").write_text(_readme())
        packet = {"kind": KIND, "complete": True, "snapshot_utc": datetime.now(timezone.utc).isoformat(),
                  "source_hash": source_hash(), "run_directory": str(run),
                  "report_manifest_sha256": ready["report_manifest_sha256"],
                  "scientific_artifacts_verified_before_copy": ready["verified_artifact_count"],
                  "report_outputs": len(ready["manifest"]["outputs"]),
                  "copied_artifacts": copied, "historical_input_mapping": historical,
                  "historical_checkpoint_metadata_count": ready["historical_checkpoint_metadata_count"],
                  "full_run_inventory": inventory,
                  "full_run_bytes": sum(item["bytes"] for item in inventory.values()),
                  "superseded_evidence_excluded_from_copy": [name for name, item in inventory.items()
                      if item["superseded_scientific_evidence"]],
                  "generated_files": {"README.md": sha(staging / "README.md")},
                  "lossless_gzip": {"threshold_bytes": COMPRESS_THRESHOLD, "mtime": 0,
                                    "filename_header": "", "restored_original_checksum_verified": True},
                  "large_raw_artifacts_retained_locally": True,
                  "immutable_absolute_metadata_rewritten": False,
                  "continuous_DK_claim": False, "accuracy_qualified_NN_advantage": False}
        _write_json(staging / "archive_manifest.json", packet)
        verify_archive(staging)
        if (sha(report / "manifest.json") != ready["report_manifest_sha256"]
                or source_hash() != ready["manifest"]["source_hash"]):
            raise ValueError("The completed report or package changed while archiving")
        staging.rename(output)
    return verify_archive(output)


def verify_archive(output):
    """Verify copied bytes and losslessly restored originals without local raw data."""
    output = Path(output).resolve()
    manifest = read(output / "archive_manifest.json")
    if manifest.get("kind") != KIND or manifest.get("complete") is not True:
        raise ValueError("Expected a complete broader-task evidence archive")
    expected = set(manifest["copied_artifacts"]) | set(manifest["generated_files"]) | {"archive_manifest.json"}
    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Archive file inventory differs from its manifest")
    restored = 0
    for name, packet in manifest["copied_artifacts"].items():
        path = output / name
        if not _inside(path.resolve(), output) or sha(path) != packet["sha256"]:
            raise ValueError(f"Archive checksum or path validation failed: {name}")
        if packet["compressed"]:
            digest = hashlib.sha256()
            with gzip.open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            restored_sha = digest.hexdigest()
        else:
            restored_sha = packet["sha256"]
        if restored_sha != packet["original_sha256"]:
            raise ValueError(f"Restored original checksum failed: {name}")
        restored += 1
    for name, digest in manifest["generated_files"].items():
        if sha(output / name) != digest:
            raise ValueError("Generated archive documentation changed")
    return {"kind": KIND, "complete": True, "archive": str(output),
            "copied_and_restored_artifacts_verified": restored,
            "historical_inputs": len(manifest["historical_input_mapping"]),
            "historical_checkpoint_metadata": manifest.get("historical_checkpoint_metadata_count"),
            "report_outputs": manifest["report_outputs"],
            "indexed_run_files": len(manifest["full_run_inventory"]),
            "indexed_run_bytes": manifest["full_run_bytes"],
            "archive_manifest_sha256": sha(output / "archive_manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--preliminary", action="store_true",
                        help="Read-only readiness check before the final software-test record")
    parser.add_argument("--verify-archive", type=Path)
    args = parser.parse_args()
    if args.verify_archive is not None:
        result = verify_archive(args.verify_archive)
    elif args.run is None:
        parser.error("--run is required to verify or create an archive")
    elif args.verify_only:
        ready = verify_ready(args.run, report=args.report, require_completion=not args.preliminary)
        result = {"complete": True, "report_outputs": len(ready["manifest"]["outputs"]),
                  "historical_inputs": len(ready["historical"]),
                  "historical_checkpoint_metadata": ready["historical_checkpoint_metadata_count"],
                  "verified_artifacts": ready["verified_artifact_count"],
                  "final_completion_required": ready["completion_required"]}
    elif args.preliminary or args.output is None:
        parser.error("Writing an archive requires --output and the final completion records")
    else:
        result = archive(args.run, args.output, report=args.report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
