"""Independent locked reference generation, stopped payoffs, coupling and safe resume."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from nn_dk.payoff_learning import PayoffBank

_spec = importlib.util.spec_from_file_location(
    "path_generalization_reference", Path(__file__).parents[1]/"analysis"/"path_generalization_reference.py")
reference = importlib.util.module_from_spec(_spec)
# Dataclasses consult the defining module while processing postponed annotations.
sys.modules[_spec.name] = reference
_spec.loader.exec_module(reference)


def fixture(tmp_path, monkeypatch, *, n_particles=2048):
    cfg = reference.ReferenceConfig(n_cells=16, n_particles=n_particles, cases=("uniform", "nonuniform"),
                                    conditional_anchors=1, paths_per_group=4, refinement_paths=2,
                                    chunk_paths=2, expected_models=2, reference_step=2**-8,
                                    common_step=2**-6, model_step=2**-7)
    monkeypatch.setattr(reference, "source_hash", lambda: "reference-test-frozen-source")
    models = []
    for index in range(cfg.expected_models):
        path = tmp_path/f"checkpoint{index}.pt"
        path.write_bytes(f"complete-test-checkpoint{index}".encode())
        models.append({"checkpoint_path": str(path), "checkpoint_sha256": hashlib.sha256(
            path.read_bytes()).hexdigest(), "complete": True, "source_hash": reference.source_hash()})
    monkeypatch.setattr(reference, "load_payoff_model", lambda path: SimpleNamespace(
        complete=True, provenance={"source_hash": reference.source_hash()}))
    return cfg, {"expected_models": cfg.expected_models, "models": models}


def test_plan_does_not_reveal_anchors_or_outcomes_and_rejects_incomplete_fit_inventory(tmp_path, monkeypatch):
    cfg, lock = fixture(tmp_path, monkeypatch)
    directory = tmp_path/"ref"
    plan = reference.prepare_reference(directory, cfg)
    assert plan["status"] == "waiting-for-complete-fit-lock"
    assert sorted(path.name for path in directory.iterdir()) == ["manifest.json"]
    incomplete = copy.deepcopy(lock)
    incomplete["models"][0]["complete"] = False
    with pytest.raises(ValueError, match="Incomplete"):
        reference.generate_reference(directory, cfg, fit_lock=incomplete)
    assert not (directory/"reference.h5").exists()
    with pytest.raises(ValueError, match="allocation"):
        reference.generate_reference(directory, cfg, fit_lock={**lock, "models": lock["models"][:1]})


def test_chunk_resume_matches_fresh_data_and_all_families_are_disjoint(tmp_path, monkeypatch):
    cfg, lock = fixture(tmp_path, monkeypatch)
    partial = reference.generate_reference(tmp_path/"resume", cfg, fit_lock=lock, max_chunks=1)
    assert partial["status"] == "partial" and partial["committed_chunks"] == 1
    with pytest.raises(ValueError, match="complete"):
        reference.load_reference(tmp_path/"resume", cfg)
    complete = reference.generate_reference(tmp_path/"resume", cfg, fit_lock=lock)
    assert complete["status"] == "complete"
    reference.generate_reference(tmp_path/"fresh", cfg, fit_lock=lock)
    resumed, fresh = (reference.load_reference(tmp_path/directory, cfg) for directory in ("resume", "fresh"))
    all_ids = []
    for key in cfg.group_keys:
        for name in reference._shapes(cfg):
            np.testing.assert_array_equal(resumed["groups"][key][name], fresh["groups"][key][name])
        group = resumed["groups"][key]
        all_ids.extend(group["family_ids"])
        np.testing.assert_allclose(group["common_paths"].sum(-1), 1, atol=4e-16)
        assert group["many_observables"].shape == (4, 4*14)
        assert group["model_increments"].shape == (4, 8, 16)
    assert len(set(all_ids)) == len(all_ids)
    assert complete["failed_paths"] == 0


def test_reference_payoffs_use_complete_polygon_and_noise_is_literally_coupled(tmp_path, monkeypatch):
    cfg, lock = fixture(tmp_path, monkeypatch)
    reference.generate_reference(tmp_path/"ref", cfg, fit_lock=lock)
    key = cfg.group_keys[0]
    data = reference.load_reference(tmp_path/"ref", cfg, group_key=key)
    initial = json.loads(data["metadata"]["conditional_initial_json"])
    masses, finest, factor, families = reference._draw(cfg, key, 0, 2, initial)
    base = reference._aggregate(finest, factor)
    model = reference._aggregate(base, round(cfg.model_step/cfg.reference_step))
    np.testing.assert_array_equal(model, data["model_increments"][:2])
    np.testing.assert_array_equal(families, data["family_ids"][:2])
    np.testing.assert_array_equal(data["base_brownian_sha256"][:2],
                                  np.asarray([reference._array_sha(row) for row in base], dtype="S64"))
    fine_packet = reference._probe.numpy_euler(masses, finest, cfg.context("uniform"),
                                               cfg.spec, cfg.reference_step/factor)
    times = np.arange(finest.shape[1]+1)*cfg.reference_step/factor
    bank = PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon)
    complete = bank.evaluate_paths(fine_packet["states"], times, fine_packet["stopped_at"])
    indices = np.rint(np.asarray(cfg.query_times)/(cfg.reference_step/factor)).astype(int)
    np.testing.assert_array_equal(data["audit_bank_c"][:2, -1], complete[:, indices])
    np.testing.assert_array_equal(data["audit_common_paths"][:2, 0], data["common_paths"][:2])


def test_stopped_paths_remain_in_all_banks_and_killed_integrals_freeze(tmp_path, monkeypatch):
    cfg, lock = fixture(tmp_path, monkeypatch, n_particles=4)
    reference.generate_reference(tmp_path/"ref", cfg, fit_lock=lock)
    data = reference.load_reference(tmp_path/"ref", cfg, group_key=cfg.group_keys[0])
    assert len(data["family_ids"]) == cfg.paths_per_group
    assert (data["stopped_at"] == 0).all()
    np.testing.assert_array_equal(data["bank_c"][:, :, 2:], 0.)
    np.testing.assert_array_equal(data["common_paths"], np.repeat(
        data["initial"][:, None], len(cfg.common_times), axis=1))


def test_fractional_hit_kills_running_payoff_while_common_workload_uses_frozen_path(tmp_path, monkeypatch):
    cfg, _ = fixture(tmp_path, monkeypatch, n_particles=32)
    initial = np.full((2, cfg.n_cells), 1/cfg.n_cells)
    increments = np.zeros((2, round(cfg.horizon/cfg.reference_step), cfg.n_cells))
    increments[0, 2, 0] = 5
    packet = reference._probe.numpy_euler(initial, increments, cfg.context("uniform"),
                                          cfg.spec, cfg.reference_step)
    assert 2*cfg.reference_step < packet["stopped_at"][0] < 3*cfg.reference_step
    reduced = reference._reduce(packet, cfg, cfg.reference_step)
    np.testing.assert_array_equal(reduced["bank_c"][0, :, 2:], np.repeat(
        reduced["bank_c"][0, :1, 2:], len(cfg.query_times), axis=0))
    assert reduced["workloads"].shape == (2, 3)
    invalid = dict(packet)
    invalid["states"] = packet["states"].copy()
    invalid["states"][0, -1, 0] = np.nan
    with pytest.raises(ValueError, match="survivor"):
        reference._reduce(invalid, cfg, cfg.reference_step)


def test_mutated_committed_chunk_source_lock_and_grid_are_rejected(tmp_path, monkeypatch):
    cfg, lock = fixture(tmp_path, monkeypatch)
    directory = tmp_path/"ref"
    reference.generate_reference(directory, cfg, fit_lock=lock, max_chunks=1)
    with h5py.File(directory/"reference.h5", "r+") as out:
        out[cfg.group_keys[0]]["bank_b"][0, 0, 0] += .01
    with pytest.raises(ValueError, match="checksum"):
        reference.generate_reference(directory, cfg, fit_lock=lock)
    other = tmp_path/"grid"
    reference.generate_reference(other, cfg, fit_lock=lock, max_chunks=0)
    with h5py.File(other/"reference.h5", "r+") as out:
        out["common_times"][1] += .01
    with pytest.raises(ValueError, match="grids"):
        reference.generate_reference(other, cfg, fit_lock=lock)
    bad = copy.deepcopy(lock)
    bad["models"][0]["checkpoint_sha256"] = "0"*64
    with pytest.raises(ValueError, match="mutated"):
        reference.generate_reference(tmp_path/"badlock", cfg, fit_lock=bad)
    monkeypatch.setattr(reference, "source_hash", lambda: "new-source")
    with pytest.raises(ValueError, match="identity"):
        reference.prepare_reference(directory, cfg)


def test_uncommitted_chunk_is_overwritten_and_failures_cannot_be_averaged(tmp_path, monkeypatch):
    cfg, lock = fixture(tmp_path, monkeypatch)
    directory = tmp_path/"ref"
    reference.generate_reference(directory, cfg, fit_lock=lock, max_chunks=0)
    with h5py.File(directory/"reference.h5", "r+") as out:
        out[cfg.group_keys[0]]["bank_b"][0] = 666
    reference.generate_reference(directory, cfg, fit_lock=lock)
    data = reference.load_reference(directory, cfg, group_key=cfg.group_keys[0])
    assert abs(data["bank_b"]).max() < 2
    original = reference._simulate_chunk

    def fail(*args, **kwargs):
        raise RuntimeError("test numerical failure")

    monkeypatch.setattr(reference, "_simulate_chunk", fail)
    failed = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="numerical failure"):
        reference.generate_reference(failed, cfg, fit_lock=lock)
    manifest = json.loads((failed/"manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"]["survivor_conditioned_estimates_allowed"] is False
    with pytest.raises(ValueError, match="complete"):
        reference.load_reference(failed, cfg)
    monkeypatch.setattr(reference, "_simulate_chunk", original)


def test_probability_and_fixed_allocation_audits():
    cfg = reference.ReferenceConfig()
    assert len(cfg.group_keys) == 6
    assert cfg.paths_per_group == 16384 and cfg.refinement_paths == 1024
    assert len(cfg.common_times) == 17
    for case in cfg.cases:
        probabilities, audit = reference.profile_probabilities(cfg, case)
        assert probabilities.min() > 0
        assert audit["normalization_error"] < 1e-14
        assert audit["maximum_probability_refinement_difference"] < 1e-14
    with pytest.raises(ValueError, match="nested"):
        reference.ReferenceConfig(reference_factors=(1, 3, 4))
