"""Family-level provenance, uncensored proposals, and resumable native pools."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from nn_dk.convergence import _aggregate_increments, _coupling
from nn_dk.convergence_config import ConvergenceConfig, convergence_preset
from nn_dk.convergence_data import (
    generate_native_pool,
    load_native_pool,
    prepare_convergence_run,
)
from nn_dk.native import initialize_masses, native_coefficients, simulate_native
from nn_dk.observations import incidence_matrix
from nn_dk.types import seeded_rng


def small_config(**changes):
    return replace(convergence_preset("a-smoke"), cases=("uniform",), n_particles=64,
                   train_path_counts=(4, 8), query_times=(0.125,), proposal_dt=0.125,
                   trajectory_batch_size=2, diagnostic_paths=4, evaluation_batch_size=2,
                   **changes)


@pytest.mark.parametrize("change", [
    {"cases": ("unknown",)}, {"cases": ("uniform", "uniform")},
    {"training_modes": ("survivor-increments",)}, {"n_cells": 4},
    {"n_particles": 1}, {"states_per_path": 0}, {"diagnostic_paths": 0},
    {"train_path_counts": (8, 4)}, {"train_path_counts": (4, 4)},
    {"step_sizes": (1 / 128, 1 / 64)}, {"step_sizes": ()},
    {"reference_dt": 0}, {"proposal_dt": 0}, {"query_times": (0.03,)},
    {"query_times": (0.25,)}, {"uncertainty_budget": 0.05},
    {"tolerance": 0.005}, {"budget_seconds": 0},
    {"widths": (8,), "base_width": 16}, {"evaluation_batch_size": 1_000_000},
    {"train_path_counts": (4, 8.5)}, {"widths": (8, 16.5), "base_width": 8},
    {"optimizer_seeds": (0, 0)}, {"optimizer_seeds": (0.5,)},
    {"particle_numbers": (0,)}, {"budget_seconds": float("nan")},
    {"query_times": (0.03125, 0.0625)},
])
def test_invalid_study_settings_are_rejected(change):
    with pytest.raises(ValueError):
        replace(convergence_preset("a-smoke"), **change)


def test_fit_axes_separate_nested_data_capacity_and_increment_estimation():
    config = convergence_preset("a-full")
    fits = config.fit_specs()
    assert len({f["key"] for f in fits}) == len(fits) == 21
    for seed in config.optimizer_seeds:
        oracle = [f for f in fits if f["seed"] == seed and f["mode"] == "oracle"]
        data_curve = [f for f in oracle if f["width"] == config.base_width]
        assert [f["paths"] for f in data_curve] == [64, 256, 1024, 4096]
        assert {f["width"] for f in oracle if f["paths"] == 4096} == {16, 32, 64}
        assert all(f["width"] == config.base_width for f in oracle if f["paths"] < 4096)
        increments = [f for f in fits if f["seed"] == seed and f["mode"] == "increments"]
        assert [(f["paths"], f["width"]) for f in increments] == [(4096, 32)]
    assert config.context("uniform").kappa == 0
    assert config.context("nonuniform").cos_coeffs == (0.2, 0, 0)
    assert config.context("interacting").kappa == 0.75
    assert config.context("interacting").a2 == 0.1


def test_shipped_toml_snapshots_match_named_presets():
    root = Path(__file__).resolve().parents[1]
    for name in ("a-smoke", "a-first-batch", "a-full"):
        loaded = ConvergenceConfig.load(root / "configs" / f"{name.replace('-', '_')}.toml")
        preset = convergence_preset(name)
        assert loaded == preset and loaded.config_hash == preset.config_hash


def test_native_pools_are_iid_histogram_histories_with_nested_family_prefixes(tmp_path):
    config = small_config()
    assert generate_native_pool(tmp_path, config, "uniform")["complete"]
    small = load_native_pool(tmp_path, config, "uniform", "train", 4)
    large = load_native_pool(tmp_path, config, "uniform", "train", 8)
    validation = load_native_pool(tmp_path, config, "uniform", "validation")
    test = load_native_pool(tmp_path, config, "uniform", "test")
    assert small.independent_paths == 4 and large.independent_paths == 8
    np.testing.assert_array_equal(small.states, large.states[:len(small.states)])
    assert small.family_ids == large.family_ids[:len(small.family_ids)]
    assert small.sample_ids == large.sample_ids[:len(small.sample_ids)]
    assert len(set(large.sample_ids)) == 8 * config.states_per_path
    for first, second in ((large, validation), (large, test), (validation, test)):
        assert not set(first.family_ids) & set(second.family_ids)
        assert not set(first.sample_ids) & set(second.sample_ids)
    seed = int(seeded_rng(config.seed, "native-pool", "uniform", "train", 0).integers(0, 2 ** 63))
    replay = simulate_native(config.context("uniform"), config.spec, config.sample_times,
                             config.reference_dt, seed, n_paths=2,
                             family_ids=("uniform/train/0", "uniform/train/1"))
    np.testing.assert_array_equal(large.states[:2 * config.states_per_path],
                                   replay.states.reshape(-1, config.n_cells))
    initial = large.states.reshape(8, config.states_per_path, config.n_cells)[:, 0]
    np.testing.assert_allclose(initial * config.n_particles,
                               np.round(initial * config.n_particles), atol=1e-12)
    np.testing.assert_allclose(large.states.sum(-1), 1, atol=1e-12)
    assert not np.array_equal(initial[0], initial[1])


def test_hdf5_keeps_raw_crossing_proposals_and_separates_oracle_metadata(tmp_path):
    config = small_config()
    generate_native_pool(tmp_path, config, "uniform")
    context = config.context("uniform")
    with h5py.File(tmp_path / "pool_uniform.h5", "r") as store:
        group = store["validation"]
        assert {"drift", "face_amplitudes"} == set(group["oracle"])
        assert "oracle_drift" not in group and "oracle_face_amplitudes" not in group
        states, raw = group["states"][:], group["proposals"][:]
        active = group["active"][:]
        crossings = group["proposal_crosses"][:]
        expected_crossings = (raw.min(-1) <= config.spec.mass_stop) & active
        np.testing.assert_array_equal(crossings, expected_crossings)
        assert crossings.any() and raw[crossings].min() < 0
        assert np.all(active[crossings])
        # The exact uncensored Euler law is replayed, including its negative tail.
        origins = states[:2].reshape(-1, config.n_cells)
        coefficients = native_coefficients(origins, context, config.spec)
        z = seeded_rng(config.seed, "native-proposals", "uniform", "validation", 0).normal(size=origins.shape)
        expected = origins + config.proposal_dt * coefficients.drift
        expected += np.sqrt(config.proposal_dt / context.n_particles) * (
            z * coefficients.face_amplitudes) @ incidence_matrix(config.n_cells).T
        expected[~active[:2].reshape(-1)] = origins[~active[:2].reshape(-1)]
        np.testing.assert_array_equal(raw[:2].reshape(-1, config.n_cells), expected)
        np.testing.assert_allclose(raw.sum(-1), 1, atol=1e-12)
        assert group["proposals"].chunks is not None
    loaded = load_native_pool(tmp_path, config, "uniform", "validation")
    assert np.any(loaded.proposals[loaded.active].min(-1) < 0)
    assert len(loaded.proposals) == len(loaded.states)  # No survivor filtering.


def test_interrupted_generation_resumes_exactly_without_regenerating_completed_chunks(tmp_path, monkeypatch):
    config = small_config()
    resumed, fresh = tmp_path / "resumed", tmp_path / "fresh"
    calls = iter((0.0, 2.0))
    with monkeypatch.context() as patch:
        patch.setattr("nn_dk.convergence_data.time.monotonic", lambda: next(calls))
        interrupted = generate_native_pool(resumed, config, "uniform", deadline=1.0)
    assert not interrupted["complete"]
    with h5py.File(resumed / "pool_uniform.h5", "r") as store:
        assert store["train/completed"][:].tolist() == [True, True, False, False, False, False, False, False]
        completed_seconds = store["train/seconds"][:2]
    with pytest.raises(ValueError, match="incomplete"):
        load_native_pool(resumed, config, "uniform", "train", 4)
    assert generate_native_pool(resumed, config, "uniform")["complete"]
    assert generate_native_pool(fresh, config, "uniform")["complete"]
    for split in ("train", "validation", "test"):
        actual = load_native_pool(resumed, config, "uniform", split)
        expected = load_native_pool(fresh, config, "uniform", split)
        assert actual.family_ids == expected.family_ids and actual.sample_ids == expected.sample_ids
        for key in ("states", "proposals", "active"):
            np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    with h5py.File(resumed / "pool_uniform.h5", "r") as store:
        np.testing.assert_array_equal(store["train/seconds"][:2], completed_seconds)
    def reject_regeneration(*args, **kwargs):
        raise AssertionError("Completed native paths must be reused")
    monkeypatch.setattr("nn_dk.native.simulate_native", reject_regeneration)
    assert generate_native_pool(resumed, config, "uniform")["complete"]
    with pytest.raises(ValueError, match="mismatch"):
        prepare_convergence_run(resumed, replace(config, seed=config.seed + 1))


def test_evaluation_coupling_is_deterministic_disjoint_and_aggregates_actual_increments(tmp_path):
    config = small_config()
    initial, dw, ids, seed, _ = _coupling(config, "uniform", 0, 2)
    repeated = _coupling(config, "uniform", 0, 2)
    np.testing.assert_array_equal(initial, repeated[0])
    np.testing.assert_array_equal(dw, repeated[1])
    assert ids == repeated[2] and seed == repeated[3]
    np.testing.assert_array_equal(initial, initialize_masses(config.context("uniform"), config.spec, 2, seed))
    assert ids == tuple(f"{config.config_hash}/uniform/evaluation/{i}" for i in range(2))
    assert not set(ids) & {"uniform/train/0", "uniform/validation/0", "uniform/test/0"}
    coarse = _aggregate_increments(dw, 1 / 64, config.reference_dt)
    np.testing.assert_array_equal(coarse, dw.reshape(2, -1, 16, config.n_cells).sum(2))
    half = _aggregate_increments(dw, 1 / 128, config.reference_dt)
    np.testing.assert_allclose(_aggregate_increments(half, 1 / 64, 1 / 128), coarse, atol=1e-15)
    np.testing.assert_allclose(coarse.sum(1), dw.sum(1), atol=1e-15)
    with pytest.raises(ValueError, match="nest"):
        _aggregate_increments(dw, 0.01, config.reference_dt)
