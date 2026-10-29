"""Independent saved-family replay; no new scientific allocation is generated."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit_lna_supervised_labels as audit  # noqa: E402
from analysis import run_lna_payoff_correction_study as producer  # noqa: E402
from analysis.compiled_native import compiler_available  # noqa: E402


@pytest.fixture
def learner(tmp_path):
    if not compiler_available():
        pytest.skip("Optional C++ label producer unavailable")
    torch.set_num_threads(1)
    plan = producer.StudyPlan(horizons=(1/64,), kappas=(.75,), n_cells=8, n_particles=2048,
        train_natural_per_profile=2, train_stress_per_profile=1, validation_per_profile=1,
        train_continuations=4, validation_continuations=4, label_chunk=2, label_step=1/1024,
        width=4, depth=1, adam_steps=1, lbfgs_steps=0, batch_size=4, validation_every=1)
    root = tmp_path/"learner"
    producer.prepare(root, plan)
    return root, plan


def test_scientific_default_is_exactly_the_predetermined_twenty_four_families():
    plan = producer.StudyPlan()
    allocated = audit.allocation(plan)
    assert len(allocated) == 24
    assert {row["anchor"] for row in allocated} == {0, 40, 60, 100}
    assert {row["continuation"] for row in allocated} == {0}
    assert {row["role"] for row in allocated} == {"train"}
    assert {row["case"] for row in allocated} == set(producer.cases(plan))
    assert len(audit.allocation(producer.StudyPlan(horizons=(.25,)))) == 12


def test_partial_saved_labels_resume_via_independent_numpy_without_redraw(learner, monkeypatch):
    root, plan = learner
    producer.generate_data(root, max_chunks=1)
    partial = audit.run(root)
    assert not partial["complete"] and len(partial["records"]) == 1
    assert partial["records"][0]["anchor"] == 0
    assert partial["records"][0]["compiled_backend_called"] is False
    with pytest.raises(ValueError, match="predetermined learner-label"):
        audit.verify(root)
    assert audit.verify(root, require_complete=False) == partial
    producer.generate_data(root, role="train")
    replay = audit._replay
    seen = []

    def remaining(*args):
        item = args[2]
        assert item["anchor"] != 0
        seen.append(item["anchor"])
        return replay(*args)

    monkeypatch.setattr(audit, "_replay", remaining)
    result = audit.run(root)
    assert result["complete"] and seen == [2, 3, 5]
    assert audit.verify(root) == result
    assert result["new_scientific_families"] == 0
    assert result["final_outcomes_or_models_accessed"] is False
    assert all(row["verified"] and max(row["errors"].values()) < 3e-12 for row in result["records"])
    assert all(attempt["finished"] and attempt["wall_seconds"] > 0 for attempt in result["attempts"])
    assert not list(root.rglob("*.pt")) and not list(root.rglob("*.h5"))

    def forbidden(*args, **kwargs):
        pytest.fail("A verified learner family was replayed again")

    monkeypatch.setattr(audit, "_replay", forbidden)
    assert audit.run(root)["complete"]
    assert result["identity"]["expected_rows"] == 4  # Explicit reduced software allocation.
    assert len(producer.cases(plan)) == 1


def test_changed_raw_chunk_or_saved_entropy_is_rejected(learner):
    root, plan = learner
    producer.generate_data(root, role="train")
    result = audit.run(root)
    path = root/"data"/producer.cases(plan)[0]/"train/labels/anchor0/chunk0/arrays.npz"
    original = path.read_bytes()
    path.write_bytes(original+b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        audit.verify(root)
    path.write_bytes(original)
    changed = copy.deepcopy(result)
    changed["records"][0]["forcing_entropy_sha256"] = "0"*64
    audit._write(root, changed)
    with pytest.raises(ValueError, match="address, forcing"):
        audit.verify(root)


@pytest.mark.parametrize("alteration", ["source", "allocation", "missing_error", "cost"])
def test_source_or_incomplete_verification_cannot_be_relabelled_complete(learner, alteration):
    root, _ = learner
    producer.generate_data(root, role="train")
    changed = copy.deepcopy(audit.run(root))
    if alteration == "source":
        changed["identity"]["script_sha256"] = "0"*64
    elif alteration == "allocation":
        changed["records"].pop()
    elif alteration == "missing_error":
        changed["records"][0]["errors"].pop("stopping_time_error")
    else:
        changed["attempts"][0].pop("cpu_seconds")
    audit._write(root, changed)
    with pytest.raises(ValueError):
        audit.verify(root)


def test_uncommitted_producer_chunk_is_pending_and_never_repaired(learner):
    root, plan = learner
    case = producer.cases(plan)[0]
    path = root/"data"/case/"train/labels/anchor0/chunk0"
    path.mkdir(parents=True)
    (path/"arrays.npz.tmp").write_bytes(b"producer is still committing")
    result = audit.run(root)
    assert not result["complete"] and not result["records"]
    assert not (path/"record.json").exists()
    with pytest.raises(ValueError, match="predetermined learner-label"):
        audit.verify(root)


def test_prior_audit_costs_and_exact_source_bytes_remain_separate(learner):
    root, _ = learner
    producer.generate_data(root, role="train")
    previous = audit.run(root)
    source = b"# exact previous auditor source\n"
    previous["identity"]["script_sha256"] = hashlib.sha256(source).hexdigest()
    audit._write(root, previous)
    receipt = (root/audit.RECEIPT).read_bytes()
    folder = root/"execution/superseded_replay_audit_v1"
    folder.mkdir()
    (folder/"audit_lna_supervised_labels.py").write_bytes(source)
    (folder/"native_label_replay_audit.json").write_bytes(receipt)
    manifest = {"kind": "preserved-native-label-replay-audit-v1", "files": {
        "audit_lna_supervised_labels.py": hashlib.sha256(source).hexdigest(),
        "native_label_replay_audit.json": hashlib.sha256(receipt).hexdigest()},
        "verified_rows": 4, "complete": True,
        "paid_wall_seconds": sum(v["wall_seconds"] for v in previous["attempts"]),
        "paid_cpu_seconds": sum(v["cpu_seconds"] for v in previous["attempts"])}
    (folder/"manifest.json").write_text(json.dumps(manifest))
    (root/audit.RECEIPT).unlink()
    current = audit.run(root)
    assert current["complete"] and len(current["records"]) == 4
    assert len(current["prior_audit_costs"]) == 1
    assert current["prior_audit_costs"][0]["wall_seconds"] == manifest["paid_wall_seconds"]
    assert (folder/"native_label_replay_audit.json").read_bytes() == receipt
    (folder/"audit_lna_supervised_labels.py").write_bytes(source+b"changed")
    with pytest.raises(ValueError, match="prior audit source"):
        audit.verify(root)
