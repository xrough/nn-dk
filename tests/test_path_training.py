"""Pathwise learning, independent families, checkpoint integrity and exact resume."""

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import nn_dk.path_training as training
from nn_dk.path_config import PathConfig
from nn_dk.path_model import CausalPathOperator
from nn_dk.path_numerics import torch_euler
from nn_dk.path_training import fit_path, load_path_model, path_objective


@pytest.fixture(autouse=True)
def fixed_test_source(monkeypatch):
    # Other agents may edit unrelated source during these focused tests. The
    # explicit mismatch test below verifies that this guard remains strict.
    monkeypatch.setattr(training, "source_hash", lambda: "path-training-test-source")


def _config(**changes):
    return replace(PathConfig(), n_cells=8, cases=("heat", "additive"),
                   width=8, depth=2, train_paths=12, validation_paths=8, test_paths=8,
                   adam_steps=2, lbfgs_steps=1, lbfgs_max_iter=2, batch_size=5,
                   **changes)


def _banks(config, dt):
    rng = np.random.default_rng(511)
    probabilities = np.arange(1, config.n_cells+1, dtype=float)
    probabilities /= probabilities.sum()

    def bank(size, label):
        return {"initial": rng.multinomial(config.n_particles, probabilities, size=size)
                           /config.n_particles,
                "brownian": rng.normal(0, np.sqrt(dt),
                                        (size, round(config.horizon/dt), config.n_cells)),
                "family_ids": np.asarray([f"{label}-{i}" for i in range(size)])}

    return bank(config.train_paths, "training"), bank(config.validation_paths, "validation")


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_training_checkpoint_exact_resume_including_lbfgs_and_shuffle_rng(tmp_path, monkeypatch):
    config = _config()
    dt = config.step_sizes[0]
    train_bank, validation_bank = _banks(config, dt)
    full, resumed = tmp_path/"full.pt", tmp_path/"resumed.pt"
    complete = fit_path(config, "additive", dt, 0, full, train_bank, validation_bank)
    assert complete["complete"]
    clock = iter(range(1, 100))
    with monkeypatch.context() as interruption:
        interruption.setattr(training, "time", SimpleNamespace(
            monotonic=lambda: next(clock), perf_counter=time.perf_counter))
        partial = fit_path(config, "additive", dt, 0, resumed, train_bank, validation_bank,
                           deadline=5)
    assert not partial["complete"]
    assert partial["adam_steps"] == 1 and partial["lbfgs_steps"] == 0
    with pytest.raises(ValueError, match="incomplete"):
        load_path_model(resumed)
    continued = fit_path(config, "additive", dt, 0, resumed, train_bank, validation_bank)
    assert continued["complete"]
    first, second = torch.load(full, weights_only=True), torch.load(resumed, weights_only=True)
    for key in ("model_state", "best_state", "adam_state", "lbfgs_state", "torch_rng_state",
                "shuffle_rng_state", "best_validation", "best_step"):
        _assert_nested_equal(first[key], second[key])
    loaded = load_path_model(resumed)
    assert loaded.target_id == config.target_id("additive", dt)
    assert loaded.path_config.config_hash == config.config_hash
    assert loaded.case == "additive" and loaded.dt == dt


def test_completed_replay_preserves_checkpoint_bytes_time_and_selection(tmp_path):
    config = _config()
    dt = config.step_sizes[0]
    train_bank, validation_bank = _banks(config, dt)
    path = tmp_path/"operator.pt"
    metadata = fit_path(config, "heat", dt, 0, path, train_bank, validation_bank)
    before, timestamp = path.read_bytes(), path.stat().st_mtime_ns
    replay = fit_path(config, "heat", dt, 0, path, train_bank, validation_bank)
    assert path.read_bytes() == before and path.stat().st_mtime_ns == timestamp
    assert replay["training_seconds"] == metadata["training_seconds"]
    assert replay["best_validation"] == metadata["best_validation"]
    saved = torch.load(path, weights_only=True)
    objectives = [saved["initial_validation"]["objective"],
                  *(record["validation"]["objective"] for record in saved["history"])]
    assert saved["best_validation"]["objective"] == min(objectives)


def test_input_fingerprints_source_and_parameter_checksums_reject_mutation(tmp_path, monkeypatch):
    config = _config()
    dt = config.step_sizes[0]
    train_bank, validation_bank = _banks(config, dt)
    path = tmp_path/"operator.pt"
    fit_path(config, "heat", dt, 0, path, train_bank, validation_bank)
    changed = {**train_bank, "brownian": train_bank["brownian"].copy()}
    changed["brownian"][0, 0, 0] += .01
    with pytest.raises(ValueError, match="input data mismatch"):
        fit_path(config, "heat", dt, 0, path, changed, validation_bank)
    with monkeypatch.context() as wrong_source:
        wrong_source.setattr(training, "source_hash", lambda: "different-source")
        with pytest.raises(ValueError, match="source hash"):
            load_path_model(path)
    saved = torch.load(path, weights_only=True)
    weight = next(key for key in saved["best_state"] if "weight" in key)
    saved["best_state"][weight].flatten()[0] += .01
    torch.save(saved, path)
    with pytest.raises(ValueError, match="checksum"):
        load_path_model(path)


def test_family_split_rejects_coupled_or_derived_path_leakage(tmp_path):
    config = _config()
    dt = config.step_sizes[0]
    train_bank, validation_bank = _banks(config, dt)
    validation_bank["family_ids"][0] = train_bank["family_ids"][0]
    with pytest.raises(ValueError, match="disjoint"):
        fit_path(config, "heat", dt, 0, tmp_path/"leaked.pt", train_bank, validation_bank)


def test_expired_budget_saves_incomplete_without_optimizer_updates(tmp_path):
    config = _config()
    dt = config.step_sizes[0]
    train_bank, validation_bank = _banks(config, dt)
    result = fit_path(config, "heat", dt, 0, tmp_path/"budget.pt", train_bank, validation_bank,
                      deadline=time.monotonic()-1)
    assert not result["complete"]
    assert result["adam_steps"] == result["lbfgs_steps"] == 0
    assert result["best_step"] == 0


def test_exact_euler_solution_has_zero_path_residual_under_same_physical_forcing():
    config = _config()
    dt = config.step_sizes[0]
    train_bank, _ = _banks(config, dt)
    initial = torch.tensor(train_bank["initial"])
    brownian = torch.tensor(train_bank["brownian"])

    def oracle(initial, increments, dt, context):
        packet = torch_euler(initial, increments, context, config.spec, dt, "additive")
        return torch.cat((initial[:, None], packet["raw_proposals"]), 1)

    loss, packet = path_objective(oracle, initial, brownian, config.context("additive"),
                                  config.spec, dt, "additive")
    assert loss < 1e-24
    unrelated = brownian.roll(1, 0)
    wrong_loss, _ = path_objective(lambda *args: oracle(initial, brownian, dt,
                                                      config.context("additive")),
                                   initial, unrelated, config.context("additive"),
                                   config.spec, dt, "additive")
    assert wrong_loss > .1
    assert packet["states"].shape[0] == config.train_paths


def test_short_actual_control_fit_reduces_independent_full_field_residual(tmp_path):
    config = _config(horizon=.25, query_times=(.125, .25), step_sizes=(1/32,),
                     reference_step=1/128, learning_rate=.01)
    config = replace(config, adam_steps=50, lbfgs_steps=5, lbfgs_max_iter=5)
    dt = config.step_sizes[0]
    train_bank, validation_bank = _banks(config, dt)
    torch.manual_seed(0)
    initial = CausalPathOperator(config.spec, "additive", config.width, config.depth)
    before, _ = path_objective(initial, torch.tensor(validation_bank["initial"]),
                               torch.tensor(validation_bank["brownian"]),
                               config.context("additive"), config.spec, dt, "additive")
    path = tmp_path/"development.pt"
    fit_path(config, "additive", dt, 0, path, train_bank, validation_bank)
    fitted = load_path_model(path)
    after, _ = path_objective(fitted, torch.tensor(validation_bank["initial"]),
                              torch.tensor(validation_bank["brownian"]),
                              config.context("additive"), config.spec, dt, "additive")
    assert after.item() < .05*before.item()
    assert all(parameter.grad is None for parameter in fitted.parameters())
