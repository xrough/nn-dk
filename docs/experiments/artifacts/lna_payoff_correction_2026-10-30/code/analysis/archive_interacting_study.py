"""Archive compact evidence while indexing every retained full run artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from nn_dk.data import source_hash


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive(moments, paths, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Use a fresh archive destination; preserve previous evidence")
    inventories = {}
    copied = {}
    for name, directory in (("statistics", moments), ("paths", paths)):
        directory = Path(directory).resolve()
        manifest = json.loads((directory/"manifest.json").read_text())
        if manifest.get("status") != "complete" or manifest.get("source_hash") != source_hash():
            raise ValueError("Only complete source-matching runs may be archived")
        for relative, expected in manifest["artifact_lock"].items():
            if sha(directory/relative) != expected:
                raise ValueError("A locked run artifact changed")
        inventories[name] = {"directory": str(directory), "files": {
            str(p.relative_to(directory)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file()}}
        selected = ([p for p in directory.glob("*.json")]+list(directory.glob("*.csv"))
                    +list(directory.glob("*.png"))+list(directory.glob("*.md"))
                    if name == "statistics" else
                    [directory/f for f in ("config.json", "manifest.json", "preregistration.json", "calibration.json")]
                    +list((directory/"report").glob("*.csv"))
                    +list((directory/"report").glob("*.png"))
                    +list((directory/"report").glob("*.md")))
        selected += list((directory/"checkpoints").glob("*.json"))
        for original in selected:
            target = destination/name/original.relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            copied[str(target.relative_to(destination))] = sha(target)
    evidence = {"source_hash": source_hash(), "source_freeze_commit": "69d5685",
                "full_run_inventories": inventories, "copied_artifacts": copied,
                "full_checkpoints_and_HDF5_retained_in_original_run_directories": True}
    (destination/"archive_manifest.json").write_text(json.dumps(evidence, indent=2, sort_keys=True)+"\n")
    (destination/"README.md").write_text(
        "# Interacting solver smoke evidence\n\n"
        "Compact reports, full statistics comparison packets, configuration, checkpoint metadata "
        "and plots are retained here. `archive_manifest.json` hashes every copied artifact and "
        "indexes all full run files, including restorable checkpoints, raw physical noise, "
        "references and density predictions retained in the recorded local run directories.\n\n"
        "Package source freeze: `69d5685`. These short experiments do not certify continuous-DK convergence.\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moments", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(archive(args.moments, args.paths, args.output))
