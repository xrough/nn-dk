"""Historical source/checkpoint control acceptance does not waive provenance."""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import payoff_legacy_control as legacy  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402


def test_historical_source_requires_all_original_bytes_but_allows_new_modules(tmp_path, monkeypatch):
    archive, repo = tmp_path/"archive", tmp_path/"repo"
    saved = archive/"code/src/nn_dk"
    current = repo/"src/nn_dk"
    saved.mkdir(parents=True)
    current.mkdir(parents=True)
    (saved/"a.py").write_text("old = 1\n")
    (current/"a.py").write_text("old = 1\n")
    (current/"new.py").write_text("new = 2\n")
    (archive/"archive_manifest.json").write_text("{}")
    monkeypatch.setattr(legacy, "ARCHIVE", archive)
    monkeypatch.setattr(legacy, "REPO", repo)
    monkeypatch.setattr(legacy, "SOURCE_HASH", stable_hash({"a.py": "old = 1\n"}))
    assert legacy.verify_source()["historical_files_verified"] == 1
    (current/"a.py").write_text("old = 3\n")
    with pytest.raises(ValueError, match="dependency changed"):
        legacy.verify_source()
    (current/"a.py").write_text("old = 1\n")
    (saved/"a.py").write_text("old = 3\n")
    with pytest.raises(ValueError, match="source hash"):
        legacy.verify_source()


def test_archived_lock_restores_exact_decompressed_bytes_and_rejects_edits(tmp_path, monkeypatch):
    raw = b'{"complete": true}\n'
    compressed = gzip.compress(raw, mtime=0)
    path = tmp_path/"payoffs/fit_lock.json.gz"
    path.parent.mkdir()
    path.write_bytes(compressed)
    manifest = {"copied": {"payoffs/fit_lock.json.gz": {"compressed": True,
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "original_sha256": hashlib.sha256(raw).hexdigest()}}}
    (tmp_path/"archive_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(legacy, "ARCHIVE", tmp_path)
    assert legacy._archived_json("payoffs/fit_lock.json") == {"complete": True}
    path.write_bytes(gzip.compress(b'{"complete": false}\n', mtime=0))
    with pytest.raises(ValueError, match="record changed"):
        legacy._archived_json("payoffs/fit_lock.json")


def test_legacy_controls_reject_different_physics_even_when_case_label_matches():
    from nn_dk.payoff_repair import REPAIRED_PAYOFF_KIND

    bank = SimpleNamespace(to_dict=lambda: {"fixed": "bank"}, bank_hash="bank-hash")
    runner = SimpleNamespace(configuration=lambda case, study: SimpleNamespace(target_id="native-target"),
                             payoff_bank=lambda case, study: bank)
    control = {"models": [{"case": "T1_k6", "bank_hash": "bank-hash",
        "target_id": stable_hash({"native_law": "native-target", "kind": REPAIRED_PAYOFF_KIND,
                                  "bank": bank.to_dict()})}]}
    legacy.verify_targets(control, runner, object())
    runner.configuration = lambda case, study: SimpleNamespace(target_id="changed-particle-count")
    with pytest.raises(ValueError, match="physical target"):
        legacy.verify_targets(control, runner, object())
