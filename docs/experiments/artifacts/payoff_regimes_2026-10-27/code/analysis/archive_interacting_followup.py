"""Archive completed follow-up evidence with exact compressed JSON snapshots."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from nn_dk.data import source_hash


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Preserve existing archives; choose a fresh destination")
    complete = json.loads((run/"execution/completed.json").read_text())
    if (not complete.get("complete") or complete["identity"]["source_hash"] != source_hash()
            or not (run/"comparison/task_timing.json").is_file()
            or not (run/"report/summary.json").is_file()):
        raise ValueError("Archive only a completed source-matching scientific batch, comparison and report")
    for child in ("paths", "balanced"):
        manifest = json.loads((run/child/"manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError("A child study is incomplete")
        for name, digest in manifest["artifact_lock"].items():
            if sha(run/child/name) != digest:
                raise ValueError("A locked scientific artifact changed")
    selected = {run/"study_config.json"}
    selected.update((run/"execution").glob("*.json"))
    selected.update((run/"execution").glob("*.log"))
    selected.update(p for p in (run/"report").rglob("*") if p.is_file())
    for child in ("paths", "unscaled", "balanced"):
        selected.update((run/child).glob("*.json"))
        selected.update((run/child/"checkpoints").glob("*.json"))
        selected.update(p for p in (run/child/"report").glob("*") if p.is_file())
    selected.update((run/"comparison").glob("*.json"))
    selected.update((run/"comparison").glob("*.csv"))
    # These duplicate large full joint-time/cell tensors already retained in
    # the original run. Their exact hashes stay in the complete inventory;
    # tables, checkpoint metadata and common-grid evidence remain here.
    omitted = {run/"paths/evaluations.json", run/"paths/report/path_results.json"}
    selected.difference_update(omitted)
    output.mkdir(parents=True)
    copied = {}
    for original in sorted(selected):
        relative = original.relative_to(run)
        compress = original.suffix == ".json" and original.stat().st_size > 100_000
        target = output/(str(relative)+(".gz" if compress else ""))
        target.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with original.open("rb") as source, target.open("wb") as sink:
                with gzip.GzipFile(fileobj=sink, mode="wb", filename="", mtime=0) as zipped:
                    shutil.copyfileobj(source, zipped)
            digest = hashlib.sha256()
            with gzip.open(target, "rb") as restored:
                for block in iter(lambda: restored.read(1024*1024), b""):
                    digest.update(block)
            if digest.hexdigest() != sha(original):
                raise ValueError("Compressed evidence does not restore exactly")
        else:
            shutil.copyfile(original, target)
        copied[str(target.relative_to(output))] = {
            "sha256": sha(target), "original_sha256": sha(original),
            "original_relative_path": str(relative), "compressed": compress}
    inventory = {str(path.relative_to(run)): {"sha256": sha(path), "bytes": path.stat().st_size}
                 for path in sorted(run.rglob("*")) if path.is_file()}
    value = {"kind": "interacting-followup-evidence-snapshot-v1", "source_hash": source_hash(),
             "source_freeze_commit": "3aa3b8b", "comparison_helper_commit": "058f991",
             "snapshot_utc": datetime.now(timezone.utc).isoformat(), "run_directory": str(run),
             "copied_artifacts": copied, "full_run_inventory": inventory,
             "large_joint_tensor_files_indexed_without_duplicate_copy":
                 [str(path.relative_to(run)) for path in sorted(omitted)],
             "full_checkpoints_and_HDF5_retained_locally": True,
             "continuous_time_accuracy_certified": False}
    (output/"archive_manifest.json").write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    (output/"README.md").write_text(
        "# Interacting refinement and task comparison evidence\n\n"
        "The report, tables, figures, fixed study design, training metadata, comparison "
        "results and warm timings are retained here. Large JSON files use lossless gzip; "
        "their original and archive SHA-256 checksums are recorded. Full restorable "
        "checkpoints, physical-noise banks and reference/predicted trajectories remain in "
        "the local run directory indexed by `archive_manifest.json`.\n\n"
        "All optimizer seeds are retained separately. Pointwise finite-reference intervals "
        "do not certify continuous stopped-DK convergence.\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(archive(args.run, args.output))
