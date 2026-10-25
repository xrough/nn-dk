"""Independent input families, nested forcing, locked references and interrupted storage."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

import nn_dk.path_data as data
from nn_dk.path_config import PathConfig
from nn_dk.path_data import ensure_path_bank, load_path_bank
from nn_dk.path_numerics import aggregate_brownian, torch_euler


@pytest.fixture(autouse=True)
def fixed_test_source(monkeypatch):
    monkeypatch.setattr(data, "source_hash", lambda: "path-bank-test-source")


def _config():
    return replace(PathConfig(), n_cells=8, cases=("heat", "dk_uniform"),
                   step_sizes=(1/64, 1/128), reference_step=1/256,
                   train_paths=12, validation_paths=8, test_paths=10)


def _lock(directory, config):
    """A mock completed inventory isolates the bank's lock checks from training."""
    directory.mkdir(parents=True, exist_ok=True)
    inventory = []
    for case in config.cases:
        for dt in config.step_sizes:
            for seed in config.optimizer_seeds:
                checkpoint = directory/f"{case}_{dt}_{seed}.pt"
                metadata = checkpoint.with_suffix(".json")
                torch.save({"complete": True, "provenance": {
                    "config_hash": config.config_hash, "source_hash": data.source_hash(),
                    "case": case, "dt": dt, "seed": seed,
                    "target_id": config.target_id(case, dt)}}, checkpoint)
                metadata.write_text(json.dumps({"complete": True}))
                inventory.append({"case": case, "dt": dt, "seed": seed,
                                  "checkpoint_path": str(checkpoint.resolve()),
                                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                  "metadata_path": str(metadata.resolve()),
                                  "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest()})
    manifest = {"source_hash": data.source_hash(), "config_hash": config.config_hash,
                "test_unlocked_unix": 1., "fit_lock": {"expected_models": config.expected_models,
                                                        "models": inventory}}
    (directory/"manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_inputs_are_iid_histograms_independent_families_and_nested_noise(tmp_path):
    config = _config()
    ensure_path_bank(tmp_path, config, "train")
    ensure_path_bank(tmp_path, config, "validation")
    fine = load_path_bank(tmp_path, config, "train", "heat", 1/128)
    coarse = load_path_bank(tmp_path, config, "train", "heat", 1/64)
    validation = load_path_bank(tmp_path, config, "validation", "heat", 1/128)
    np.testing.assert_array_equal(coarse["brownian"], aggregate_brownian(fine["brownian"], 2))
    np.testing.assert_array_equal(coarse["initial"], fine["initial"])
    np.testing.assert_array_equal(coarse["family_ids"], fine["family_ids"])
    assert not set(fine["family_ids"]) & set(validation["family_ids"])
    np.testing.assert_array_equal(fine["initial"]*config.n_particles,
                                  np.rint(fine["initial"]*config.n_particles))
    assert np.count_nonzero(fine["brownian"]) == fine["brownian"].size
    assert "reference_states" not in fine
    with h5py.File(tmp_path/"path_train.h5", "r") as store:
        assert store["heat"]["brownian"].compression == "gzip"
        assert store["heat"]["brownian"].chunks is not None
        assert not any("reference" in key for key in store["heat"])


def test_chunk_interruption_resumes_same_inputs_and_content_hash(tmp_path, monkeypatch):
    config = _config()
    full, resumed = tmp_path/"full", tmp_path/"resumed"
    complete = ensure_path_bank(full, config, "train")
    clock = iter(range(1, 100))
    with monkeypatch.context() as interruption:
        interruption.setattr(data, "time", SimpleNamespace(
            monotonic=lambda: next(clock), perf_counter=data.time.perf_counter))
        partial = ensure_path_bank(resumed, config, "train", deadline=2)
    assert not partial["complete"] and partial["completed_paths"] == 8
    with pytest.raises(ValueError, match="incomplete"):
        load_path_bank(resumed, config, "train", "heat", 1/64)
    continued = ensure_path_bank(resumed, config, "train")
    assert continued["complete"] and continued["content_hash"] == complete["content_hash"]
    for case in config.cases:
        first = load_path_bank(full, config, "train", case, 1/64)
        second = load_path_bank(resumed, config, "train", case, 1/64)
        for key in ("initial", "brownian", "family_ids"):
            np.testing.assert_array_equal(first[key], second[key])


def test_completed_bank_replay_preserves_file_bytes_and_generation_cost(tmp_path):
    config = _config()
    metadata = ensure_path_bank(tmp_path, config, "train")
    path = tmp_path/"path_train.h5"
    before, timestamp = path.read_bytes(), path.stat().st_mtime_ns
    replay = ensure_path_bank(tmp_path, config, "train")
    assert before == path.read_bytes() and timestamp == path.stat().st_mtime_ns
    assert replay["generation_seconds"] == metadata["generation_seconds"]
    assert replay["generated_this_call"] == 0


def test_source_and_committed_content_mutation_are_rejected(tmp_path, monkeypatch):
    config = _config()
    ensure_path_bank(tmp_path, config, "train")
    with monkeypatch.context() as changed_source:
        changed_source.setattr(data, "source_hash", lambda: "changed-source")
        with pytest.raises(ValueError, match="source"):
            load_path_bank(tmp_path, config, "train", "heat", 1/64)
    with h5py.File(tmp_path/"path_train.h5", "r+") as store:
        store["heat"]["brownian"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="committed path bank row changed"):
        ensure_path_bank(tmp_path, config, "train")


def test_test_inputs_cannot_be_generated_or_read_before_complete_immutable_fit_lock(tmp_path):
    config = _config()
    with pytest.raises(ValueError, match="fit lock"):
        ensure_path_bank(tmp_path, config, "test")
    assert not (tmp_path/"path_test.h5").exists()
    manifest = _lock(tmp_path, config)
    bad = dict(manifest)
    bad["fit_lock"] = {"expected_models": config.expected_models, "models": manifest["fit_lock"]["models"][:-1]}
    (tmp_path/"manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="complete fit lock"):
        ensure_path_bank(tmp_path, config, "test")
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    ensure_path_bank(tmp_path, config, "test")
    metadata = manifest["fit_lock"]["models"][0]["metadata_path"]
    with open(metadata, "a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="changed"):
        load_path_bank(tmp_path, config, "test", "heat", 1/64)


def test_final_reference_uses_same_fine_physical_noise_then_strides_every_fit_step(tmp_path):
    config = _config()
    _lock(tmp_path, config)
    ensure_path_bank(tmp_path, config, "test")
    for case in config.cases:
        bank = load_path_bank(tmp_path, config, "test", case, 1/128)
        inputs = [data._one_input(config, "test", case, i) for i in range(config.test_paths)]
        initial = np.stack([item[0] for item in inputs])
        fine_dw = np.stack([item[1] for item in inputs])
        factor = round((1/128)/config.reference_step)
        with torch.no_grad():
            expected = torch_euler(initial, fine_dw, config.context(case), config.spec,
                                   config.reference_step, config.problem(case))
        np.testing.assert_array_equal(bank["brownian"], aggregate_brownian(fine_dw, factor))
        np.testing.assert_array_equal(bank["reference_states"], expected["states"][:, ::factor])
        np.testing.assert_array_equal(bank["reference_stopped_at"], expected["stopped_at"])
        coarse = load_path_bank(tmp_path, config, "test", case, 1/64)
        np.testing.assert_array_equal(coarse["reference_states"], bank["reference_states"][:, ::2])
        assert len(bank["family_ids"]) == config.test_paths


def test_expired_generation_budget_does_not_commit_partial_rows(tmp_path):
    config = _config()
    metadata = ensure_path_bank(tmp_path, config, "validation", deadline=0.)
    assert not metadata["complete"] and metadata["completed_paths"] == 0
    with h5py.File(tmp_path/"path_validation.h5", "r") as store:
        assert not np.any(store["heat"]["complete"][:])
    assert ensure_path_bank(tmp_path, config, "validation")["complete"]
