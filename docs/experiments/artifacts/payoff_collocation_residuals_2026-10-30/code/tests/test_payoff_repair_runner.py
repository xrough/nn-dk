"""Declared repair inventory, literal-pool binding and reference unlock checks."""

from __future__ import annotations

import copy
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "payoff_repair_runner", Path(__file__).parents[1] / "analysis" / "run_payoff_repair.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

from analysis import path_generalization_reference as original_reference  # noqa: E402


def test_all_ten_slots_are_fixed_and_variants_share_physical_data_identity():
    slots = runner.slots()
    assert len(slots) == runner.EXPECTED_MODELS == 10
    assert len({row["name"] for row in slots}) == 10
    assert {row["seed"] for row in slots if row["variant"] != "repaired"} == {0}
    assert sum(row["seed"] == 0 for row in slots) == 6
    for bank in runner.payoff_banks():
        assert {row["seed"] for row in slots if row["bank"] == bank.name
                and row["variant"] == "repaired"} == {0, 1, 2}
    ids = {runner.collocation_data_id(runner.configuration(variant)) for variant in runner.VARIANTS}
    assert len(ids) == 1
    assert runner.configuration("v1-balanced").loss_balancing == "training-initial-rms"
    assert runner.reference_configuration().seed != runner.configuration().seed
    assert runner.reference_configuration().expected_models == 10
    with pytest.raises(ValueError, match="Unknown"):
        runner.configuration("post-test-best")


def test_optimizer_variant_rebinding_keeps_literal_arrays_and_rejects_physics_changes(monkeypatch):
    monkeypatch.setattr(runner, "source_hash", lambda: "source")
    cfg = runner.configuration()
    states, horizons = np.ones((2, 32)) / 32, np.array([.01, .02])
    bank = {"states": states, "horizons": horizons,
            "config_hash": cfg.config_hash, "source_hash": "source", "content_hash": "rows"}
    balanced = runner.configuration("v1-balanced")
    bound = runner._rebound(bank, balanced)
    assert bound["states"] is states and bound["horizons"] is horizons
    assert bound["content_hash"] == "rows"
    assert bound["config_hash"] == balanced.config_hash
    assert bound["shared_pool_original_config_hash"] == cfg.config_hash
    assert bank["config_hash"] == cfg.config_hash
    with pytest.raises(ValueError, match="identical physical"):
        runner._rebound(bank, replace(balanced, n_particles=4096))
    with pytest.raises(ValueError, match="identical physical"):
        runner._rebound({**bank, "source_hash": "changed"}, balanced)


def test_reference_adapter_uses_its_own_namespace_and_does_not_patch_historical_backend():
    original_loader = original_reference.load_payoff_model
    cloned = runner._reference_backend()
    assert cloned is not original_reference
    assert cloned._fit_lock.__globals__ is not original_reference._fit_lock.__globals__
    cloned.load_payoff_model = lambda path: "local-only"
    assert original_reference.load_payoff_model is original_loader


def test_fit_lock_verification_requires_exact_declared_inventory(tmp_path, monkeypatch):
    directory = tmp_path / "payoffs"
    runner._json(directory / "preregistration.json", {"frozen": True})
    runner._json(directory / "data_lock.json", {"banks": "frozen"})
    slots = runner.slots()
    records = [{**slot, "checkpoint_sha256": f"sha-{index}"} for index, slot in enumerate(slots)]
    monkeypatch.setattr(runner, "source_hash", lambda: "source")
    monkeypatch.setattr(runner, "_prepared", lambda root: (directory, {"slots": slots}, {}))
    monkeypatch.setattr(runner, "_model_record", lambda directory, slot: next(
        record for record in records if record["name"] == slot["name"]))
    saved = {"complete": True, "source_hash": "source", "expected_models": 10, "models": records,
             "data_lock_sha256": runner._sha(directory / "data_lock.json"),
             "preregistration_sha256": runner._sha(directory / "preregistration.json")}
    runner._json(directory / "fit_lock.json", saved)
    assert runner.verify_fit_lock(tmp_path) == saved
    incomplete = copy.deepcopy(saved)
    incomplete["models"].pop()
    runner._json(directory / "fit_lock.json", incomplete)
    with pytest.raises(ValueError, match="fit lock"):
        runner.verify_fit_lock(tmp_path)


def test_reference_generation_unlocks_only_strictly_declared_paths(tmp_path, monkeypatch):
    path = tmp_path / "declared.pt"
    path.write_bytes(b"frozen")
    slot = {"variant": "repaired", "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": runner._sha(path)}
    lock = {"expected_models": 10, "models": [slot]}
    monkeypatch.setattr(runner, "verify_fit_lock", lambda root: lock)
    model = SimpleNamespace(complete=True)
    monkeypatch.setattr(runner, "load_declared_model", lambda item: model)
    backend = SimpleNamespace()

    def generate(directory, cfg, *, fit_lock, max_chunks, deadline):
        assert fit_lock is lock and cfg.expected_models == 10
        assert max_chunks == 0 and deadline == 123.
        assert backend.load_payoff_model(path) is model
        with pytest.raises(ValueError, match="undeclared"):
            backend.load_payoff_model(tmp_path / "unlocked.pt")
        return {"status": "partial", "committed_chunks": 0}

    backend.generate_reference = generate
    monkeypatch.setattr(runner, "_reference_backend", lambda: backend)
    result = runner.generate_reference(tmp_path, deadline=123., max_chunks=0)
    assert result == {"status": "partial", "committed_chunks": 0}


def test_current_declared_loader_rejects_changed_checkpoint_before_restoration(tmp_path):
    path = tmp_path / "changed.pt"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.load_declared_model({"variant": "repaired", "checkpoint_path": str(path),
                                    "checkpoint_sha256": "original"})


def test_worker_locks_are_released_and_live_owner_is_rejected(tmp_path):
    path = tmp_path / ".worker.lock"
    with runner._exclusive(path):
        with pytest.raises(RuntimeError, match="Another process"):
            with runner._exclusive(path):
                raise AssertionError("An active lock must not be acquired")
    with runner._exclusive(path):
        assert path.read_text().strip().isdigit()
