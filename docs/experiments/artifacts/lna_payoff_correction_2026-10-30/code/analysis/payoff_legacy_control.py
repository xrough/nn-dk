"""Read-only replay of exactly archived v2 checkpoints in the successor study.

The old package snapshot and every corresponding current source file must be
byte-identical. New files may exist but cannot replace any historical module.
The v2 verifier runs in its own package module with the verified historical
source hash; the canonical verifier and model code are never modified.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from nn_dk.types import stable_hash

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO/"docs/experiments/artifacts/payoff_regimes_2026-10-27"
HISTORICAL = REPO/"runs/payoff-regimes-2026-10-27"
SOURCE_HASH = "b7876b03f8f4fc6b5b030a3e631ed7404b35529f6ac3b391d4c4c1cdc90a8b15"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source():
    archived = ARCHIVE/"code/src/nn_dk"
    values = {str(path.relative_to(archived)): path.read_text()
              for path in sorted(archived.rglob("*.py"))}
    if not values or stable_hash(values) != SOURCE_HASH:
        raise ValueError("Historical package snapshot does not match its frozen source hash")
    for relative in values:
        if (archived/relative).read_bytes() != (REPO/"src/nn_dk"/relative).read_bytes():
            raise ValueError("A historical package dependency changed; legacy replay is forbidden")
    return {"source_hash": SOURCE_HASH, "historical_files_verified": len(values),
            "archive_manifest_sha256": sha(ARCHIVE/"archive_manifest.json"),
            "every_historical_current_source_byte_identical": True}


def _archived_json(relative):
    manifest = json.loads((ARCHIVE/"archive_manifest.json").read_text())
    matches = [(key, value) for key, value in manifest["copied"].items()
               if key == relative or key == relative+".gz"]
    if len(matches) != 1:
        raise ValueError("Historical archive inventory is missing or ambiguous")
    name, record = matches[0]
    path = ARCHIVE/name
    if sha(path) != record["sha256"]:
        raise ValueError("Historical archived record changed")
    data = gzip.decompress(path.read_bytes()) if record["compressed"] else path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record["original_sha256"]:
        raise ValueError("Historical archive decompressed bytes changed")
    return json.loads(data)


def inventory(cases, seeds):
    proof = verify_source()
    lock = _archived_json("payoffs/fit_lock.json")
    archive_manifest = json.loads((ARCHIVE/"archive_manifest.json").read_text())
    expected_lock_sha = next(value["original_sha256"]
        for key, value in archive_manifest["copied"].items()
        if key in ("payoffs/fit_lock.json", "payoffs/fit_lock.json.gz"))
    if sha(HISTORICAL/"payoffs/fit_lock.json") != expected_lock_sha:
        raise ValueError("Current historical lock bytes differ from its archived original")
    if not lock.get("complete") or lock.get("source_hash") != SOURCE_HASH:
        raise ValueError("Historical all-model lock is incomplete or has another source")
    selected = []
    for case in cases:
        rows = [row for row in lock["models"] if row["case"] == case]
        if {row["seed"] for row in rows} != set(seeds) or len(rows) != len(seeds):
            raise ValueError("Every declared historical optimizer seed is required")
        for row in rows:
            if (sha(row["checkpoint_path"]) != row["checkpoint_sha256"]
                    or sha(row["metadata_path"]) != row["metadata_sha256"]):
                raise ValueError("Historical checkpoint or selected metadata changed")
            metadata = json.loads(Path(row["metadata_path"]).read_text())
            if (metadata["source_hash"] != SOURCE_HASH or not metadata["complete"]
                    or metadata["target_id"] != row["target_id"]):
                raise ValueError("Historical metadata provenance differs from its archived lock")
            selected.append({**row, "historical_name": row["name"],
                "name": f"{case}_legacy_v2_seed{row['seed']}", "family": "legacy-v2",
                "historical_training_seconds": row["training_seconds"]})
    return {"complete": True, "kind": "archived-v2-all-seed-control-inventory-v1",
            "source_verification": proof, "models": selected,
            "historical_fit_lock_sha256": sha(HISTORICAL/"payoffs/fit_lock.json"),
            "fresh_final_outcomes_read": False, "historical_refitting": False}


def verify_targets(controls, runner, study):
    from nn_dk.payoff_repair import REPAIRED_PAYOFF_KIND

    for item in controls["models"]:
        configuration = runner.configuration(item["case"], study)
        bank = runner.payoff_bank(item["case"], study)
        target = stable_hash({"native_law": configuration.target_id,
                              "kind": REPAIRED_PAYOFF_KIND, "bank": bank.to_dict()})
        if target != item["target_id"] or bank.bank_hash != item["bank_hash"]:
            raise ValueError("Historical control and successor physical target/payoff bank differ")


def load_model(item):
    verify_source()
    path = Path(item["checkpoint_path"])
    if item.get("family") != "legacy-v2" or sha(path) != item["checkpoint_sha256"]:
        raise ValueError("Expected an unchanged declared historical control checkpoint")
    namespace = "nn_dk._archived_v2_payoff_replay"
    module = sys.modules.get(namespace)
    if module is None:
        location = REPO/"src/nn_dk/payoff_repair.py"
        specification = importlib.util.spec_from_file_location(namespace, location)
        module = importlib.util.module_from_spec(specification)
        sys.modules[namespace] = module
        specification.loader.exec_module(module)
        module.source_hash = lambda: SOURCE_HASH
    model = module.load_repaired_payoff_model(path)
    if (model.provenance["source_hash"] != SOURCE_HASH
            or model.provenance["seed"] != item["seed"]
            or model.target_id != item["target_id"]):
        raise ValueError("Strict historical model replay has different provenance")
    return model
