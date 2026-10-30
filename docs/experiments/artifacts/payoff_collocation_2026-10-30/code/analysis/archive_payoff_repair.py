"""Preserve the complete payoff-repair report and compact verified evidence.

Original JSON bytes are copied exactly or losslessly gzipped. Full trajectory
banks, weights, arrays and libraries stay local, with checksummed inventories.
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
sys.path.insert(0, str(REPO / "src"))

from analysis.report_payoff_repair import load_complete, read, sha  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402

KIND = "independent-payoff-repair-evidence-archive-v1"
COMPRESS_THRESHOLD = 100_000


def _json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def verify_ready(run, *, preliminary=False):
    run = Path(run).resolve()
    manifest_path = run / "report/manifest.json"
    manifest = read(manifest_path)
    if (not manifest.get("complete") or manifest.get("kind") != "independent-payoff-repair-report-v1"
            or manifest.get("source_hash") != source_hash()
            or manifest.get("generator_sha256") != sha(REPO / "analysis/report_payoff_repair.py")
            or manifest.get("seeds_pooled") is not False or manifest.get("post_test_refitting") is not False
            or manifest.get("continuous_DK_claim") is not False):
        raise ValueError("A complete matching finite-grid report is required")
    actual = {p.name for p in (run / "report").iterdir() if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["outputs"]):
        raise ValueError("Report output inventory changed")
    for name, expected in manifest["outputs"].items():
        if Path(name).name != name or sha(run / "report" / name) != expected:
            raise ValueError("Report output changed or escaped the report directory")
    for name, expected in manifest["inputs"].items():
        if sha(name) != expected:
            raise ValueError(f"Report input changed: {name}")
    evidence = load_complete(run)
    if len(evidence["records"]) != 198 or len(evidence["timing"]["records"]) != 232:
        raise ValueError("Complete scientific comparison required")
    if not preliminary:
        test_path = run / "audit/final_tests_2026-10-27.json"
        tested = read(test_path)
        if tested.get("complete") is not True or tested.get("source_hash") != source_hash():
            raise ValueError("Save complete final software-test evidence for this source freeze")
    return {"run": run, "manifest": manifest, "manifest_sha256": sha(manifest_path),
            "verified_scientific_inputs": len(evidence["inputs"]), "preliminary": preliminary}


def _copy_exact(source, destination):
    source, destination = Path(source), Path(destination)
    original_sha = sha(source)
    compressed = source.suffix == ".json" and source.stat().st_size > COMPRESS_THRESHOLD
    if compressed:
        destination = destination.with_name(destination.name + ".gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        with source.open("rb") as stream, destination.open("wb") as sink:
            with gzip.GzipFile(fileobj=sink, mode="wb", filename="", mtime=0) as zipped:
                shutil.copyfileobj(stream, zipped)
    else:
        shutil.copyfile(source, destination)
    packet = {"sha256": sha(destination), "original_sha256": original_sha,
              "original_path": str(source.resolve()), "compressed": compressed,
              "original_bytes": source.stat().st_size, "archive_bytes": destination.stat().st_size}
    if _restored_sha(destination, compressed) != original_sha or sha(source) != original_sha:
        raise ValueError("Copied evidence bytes changed")
    return destination, packet


def _restored_sha(path, compressed):
    if not compressed:
        return sha(path)
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive(run, output):
    ready = verify_ready(run)
    run, output = ready["run"], Path(output).resolve()
    if output.exists() or output.is_relative_to(run):
        raise ValueError("Use a fresh archive outside the authoritative run directory")
    selected = {p for p in run.rglob("*.json")}
    selected.update(p for p in (run / "report").iterdir() if p.is_file())
    selected.update(p for p in (run / "audit").rglob("*") if p.is_file() and p.suffix in (".py", ".md"))
    inventory = {str(p.relative_to(run)): {"sha256": sha(p), "bytes": p.stat().st_size,
                 "copied": p in selected} for p in sorted(run.rglob("*")) if p.is_file()}
    historical = {Path(p).resolve(): expected for p, expected in ready["manifest"]["inputs"].items()
                  if not Path(p).resolve().is_relative_to(run) and Path(p).suffix == ".json"}
    code = {p for p in (REPO / "analysis").glob("*.py")}
    code.update((REPO / "analysis").glob("*.cpp"))
    code.update((REPO / "src/nn_dk").rglob("*.py"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".payoff-repair-archive-", dir=output.parent) as temporary:
        staging = Path(temporary) / "archive"
        staging.mkdir()
        copied = {}
        for source in sorted(selected):
            target, packet = _copy_exact(source, staging / source.relative_to(run))
            copied[str(target.relative_to(staging))] = {**packet, "role": "current-run"}
        for source, expected in sorted(historical.items()):
            if sha(source) != expected:
                raise ValueError("Historical evidence changed")
            target, packet = _copy_exact(source, staging / "historical_inputs" / source.relative_to(REPO))
            copied[str(target.relative_to(staging))] = {**packet, "role": "historical-report-input"}
        for source in sorted(code):
            target, packet = _copy_exact(source, staging / "code" / source.relative_to(REPO))
            copied[str(target.relative_to(staging))] = {**packet, "role": "exact-source-snapshot"}
        (staging / "README.md").write_text(
            "# Independent payoff formulation repair evidence\n\n"
            "The [report](report/report.md) records all declared repaired, ablation, historical and classical "
            "models, independent accuracy and uncertainty, training and inference costs. No optimizer seed or "
            "formulation was selected or refitted using final reference outcomes.\n\n"
            "All report tables/figures, current JSON records, training-stage histories, audits, exact code and "
            "historical checkpoint sidecars are retained. Large JSON is deterministically gzipped; stored and "
            "restored original checksums are both verified. Full local trajectory banks, Brownian increments, "
            "weights, numerical arrays and compiled libraries are indexed without duplication. Immutable "
            "metadata paths and bytes are preserved.\n\n"
            "This fixed-grid, short-horizon comparison makes no continuum-DK or continuous-stopping "
            "certificate. Paired error uncertainty, standalone Monte Carlo variability, reference refinement "
            "and approximate-law bias remain distinct. See amortization.csv for unresolved gates.\n\n"
            "Code uses the project MIT license; authored prose, tables and figures use CC BY 4.0.\n\n"
            "Verify with `.venv/bin/python analysis/archive_payoff_repair.py --verify-archive "
            "docs/experiments/artifacts/payoff_repair_2026-10-27`.\n")
        manifest = {"kind": KIND, "complete": True, "source_hash": source_hash(),
                    "created_utc": datetime.now(timezone.utc).isoformat(), "run_directory": str(run),
                    "report_manifest_sha256": ready["manifest_sha256"],
                    "copied_artifacts": copied, "generated_files": {"README.md": sha(staging / "README.md")},
                    "full_run_inventory": inventory, "full_run_bytes": sum(v["bytes"] for v in inventory.values()),
                    "raw_artifacts_retained_locally": True, "metadata_rewritten": False,
                    "continuous_DK_claim": False, "post_test_refitting": False}
        _json(staging / "archive_manifest.json", manifest)
        verify_archive(staging)
        if sha(run / "report/manifest.json") != ready["manifest_sha256"] or source_hash() != manifest["source_hash"]:
            raise ValueError("Report or package changed while archiving")
        staging.rename(output)
    return verify_archive(output)


def verify_archive(output):
    output = Path(output).resolve()
    manifest = read(output / "archive_manifest.json")
    if manifest.get("kind") != KIND or not manifest.get("complete"):
        raise ValueError("A complete payoff repair evidence archive is required")
    expected = set(manifest["copied_artifacts"]) | set(manifest["generated_files"]) | {"archive_manifest.json"}
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if expected != actual:
        raise ValueError("Archive inventory changed")
    for name, packet in manifest["copied_artifacts"].items():
        path = output / name
        if (not path.resolve().is_relative_to(output) or sha(path) != packet["sha256"]
                or _restored_sha(path, packet["compressed"]) != packet["original_sha256"]):
            raise ValueError(f"Archived or restored evidence changed: {name}")
    for name, expected_sha in manifest["generated_files"].items():
        if sha(output / name) != expected_sha:
            raise ValueError("Archive documentation changed")
    return {"complete": True, "archive": str(output), "copied_artifacts_verified": len(manifest["copied_artifacts"]),
            "indexed_raw_files": len(manifest["full_run_inventory"]),
            "manifest_sha256": sha(output / "archive_manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=REPO / "runs/payoff-repair-2026-10-27")
    parser.add_argument("--output", type=Path,
                        default=REPO / "docs/experiments/artifacts/payoff_repair_2026-10-27")
    parser.add_argument("--verify-archive", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--preliminary", action="store_true")
    args = parser.parse_args()
    if args.verify_archive:
        result = verify_archive(args.verify_archive)
    elif args.verify_only:
        result = verify_ready(args.run, preliminary=args.preliminary)
        result = {key: value for key, value in result.items() if key != "manifest"}
    else:
        result = archive(args.run, args.output)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
