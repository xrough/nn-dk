"""Independent generator residual fitting, integrity checks and exact resume."""

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import nn_dk.interacting_training as training
from nn_dk.interacting_config import InteractingStudyConfig
from nn_dk.interacting_model import InteractingMomentModel
from nn_dk.interacting_training import (
    fit_interacting,
    interacting_objective,
    load_interacting_model,
)
from nn_dk.types import stable_hash


@pytest.fixture(autouse=True)
def fixed_test_source(monkeypatch):
    # Other isolated modules may still be edited concurrently. Source mismatch
    # rejection is tested separately; actual scientific fits require a freeze.
    monkeypatch.setattr(training, "source_hash", lambda: "interacting-training-test-source")


def _config(**changes):
    defaults = dict(n_cells=8, width=8, train_paths=2, validation_paths=2,
                    horizon_samples=2, adam_steps=3, lbfgs_steps=2,
                    lbfgs_max_iter=2, batch_size=8, validation_every=2)
    return replace(InteractingStudyConfig(), **(defaults | changes))


def _banks(cfg):
    generator = np.random.default_rng(512)

    def bank(role):
        paths = cfg.train_paths if role == "train" else cfg.validation_paths
        families = paths * len(cfg.cases)
        repeats = len(cfg.state_times) * cfg.horizon_samples
        raw = .2 * generator.normal(size=(families, cfg.n_cells))
        raw -= raw.max(-1, keepdims=True)
        states = np.exp(raw)
        states /= states.sum(-1, keepdims=True)
        return {"states": np.repeat(states, repeats, axis=0),
                "horizons": generator.uniform(0, cfg.horizon, families * repeats),
                "family_ids": np.repeat([f"{role}-{j}" for j in range(families)], repeats),
                "content_hash": f"committed-{role}", "source_hash": training.source_hash(),
                "config_hash": cfg.config_hash, "target_id": cfg.target_id, "role": role,
                "complete": True, "future_payoff_labels": False}

    return bank("train"), bank("validation")


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
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


@pytest.mark.parametrize("deadline,expected_steps", [(5, (1, 0)), (11, (3, 1))])
@pytest.mark.parametrize("balancing", ["none", "training-initial-rms"])
def test_exact_checkpoint_continuation_including_lbfgs_and_shuffle_rng(
        tmp_path, monkeypatch, deadline, expected_steps, balancing):
    cfg = _config(loss_balancing=balancing)
    train, validation = _banks(cfg)
    full_path, resumed_path = tmp_path / "full.pt", tmp_path / "resumed.pt"
    full = fit_interacting(cfg, 0, full_path, train, validation)
    assert full["complete"]
    clock = iter(range(1, 100))
    with monkeypatch.context() as interrupt:
        interrupt.setattr(training, "time", SimpleNamespace(
            monotonic=lambda: next(clock), perf_counter=time.perf_counter))
        partial = fit_interacting(cfg, 0, resumed_path, train, validation, deadline=deadline)
    assert not partial["complete"]
    assert (partial["adam_steps"], partial["lbfgs_steps"]) == expected_steps
    with pytest.raises(ValueError, match="incomplete"):
        load_interacting_model(resumed_path)
    resumed = fit_interacting(cfg, 0, resumed_path, train, validation)
    assert resumed["complete"]
    first = torch.load(full_path, weights_only=True)
    second = torch.load(resumed_path, weights_only=True)
    for key in ("model_state", "best_state", "adam_state", "lbfgs_state",
                "torch_rng_state", "shuffle_rng_state", "best_validation", "best_step", "provenance"):
        _assert_nested_equal(first[key], second[key])
    before_rng = torch.get_rng_state().clone()
    model = load_interacting_model(resumed_path)
    assert model.complete and model.config.config_hash == cfg.config_hash
    assert model.target_id == cfg.target_id and model.provenance["seed"] == 0
    assert model.loss_scaling == second["provenance"]["loss_scaling"]
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_completed_fit_is_immutable_and_selects_only_declared_validation_checks(tmp_path):
    cfg = _config()
    train, validation = _banks(cfg)
    path = tmp_path / "complete.pt"
    metadata = fit_interacting(cfg, 0, path, train, validation)
    before, modified = path.read_bytes(), path.stat().st_mtime_ns
    replay = fit_interacting(cfg, 0, path, train, validation)
    assert path.read_bytes() == before and path.stat().st_mtime_ns == modified
    assert replay["training_seconds"] == metadata["training_seconds"]
    saved = torch.load(path, weights_only=True)
    checks = [item for item in saved["history"] if item["validation"] is not None]
    assert [item["step"] for item in checks] == [2, 3, 4, 5]
    objectives = [saved["initial_validation"]["objective"],
                  *(item["validation"]["objective"] for item in checks)]
    assert saved["best_validation"]["objective"] == min(objectives)
    initial = metadata["initial_validation"]
    assert initial["objective"] == pytest.approx(cfg.horizon**2 * (
        initial["mean_raw_rms"]**2 + initial["covariance_raw_rms"]**2))
    assert metadata["architecture"]["boundary_width"] == .05 / cfg.n_cells


def test_fingerprints_source_and_parameter_checksums_reject_mutations(tmp_path, monkeypatch):
    cfg = _config()
    train, validation = _banks(cfg)
    path = tmp_path / "integrity.pt"
    fit_interacting(cfg, 0, path, train, validation)
    changed = {**train, "states": train["states"].copy()}
    changed["states"][0, 0] += .001
    changed["states"][0, 1] -= .001
    with pytest.raises(ValueError, match="input data mismatch"):
        fit_interacting(cfg, 0, path, changed, validation)
    with monkeypatch.context() as wrong_source:
        wrong_source.setattr(training, "source_hash", lambda: "different-source")
        with pytest.raises(ValueError, match="source hash"):
            load_interacting_model(path)
    saved = torch.load(path, weights_only=True)
    saved["best_state"]["network.0.weight"][0, 0] += .01
    torch.save(saved, path)
    with pytest.raises(ValueError, match="checksum"):
        load_interacting_model(path)


def test_family_leakage_future_labels_and_incomplete_pools_are_rejected(tmp_path):
    cfg = _config()
    train, validation = _banks(cfg)
    leaked = {**validation, "family_ids": validation["family_ids"].copy()}
    leaked["family_ids"][leaked["family_ids"] == leaked["family_ids"][0]] = train["family_ids"][0]
    with pytest.raises(ValueError, match="disjoint"):
        fit_interacting(cfg, 0, tmp_path / "leaked.pt", train, leaked)
    with pytest.raises(ValueError, match="bank source"):
        fit_interacting(cfg, 0, tmp_path / "future.pt", {**train, "future_payoff_labels": True}, validation)
    with pytest.raises(ValueError, match="bank source"):
        fit_interacting(cfg, 0, tmp_path / "incomplete.pt", {**train, "complete": False}, validation)
    with pytest.raises(ValueError, match="configured unit-mass"):
        fit_interacting(cfg, 0, tmp_path / "omitted.pt",
                         {**train, "states": train["states"][:-1]}, validation)


def test_expired_budget_preserves_initial_checkpoint_without_updates(tmp_path):
    cfg = _config()
    train, validation = _banks(cfg)
    metadata = fit_interacting(cfg, 0, tmp_path / "budget.pt", train, validation,
                               deadline=time.monotonic() - 1)
    assert not metadata["complete"]
    assert metadata["adam_steps"] == metadata["lbfgs_steps"] == metadata["best_step"] == 0


def test_short_actual_fit_reduces_independent_generator_residual(tmp_path):
    # A state-dependent map needs independent initial-state coverage. Four
    # training anchors suffice for resume tests but allow complete overfitting.
    cfg = _config(width=24, train_paths=16, validation_paths=8, adam_steps=20,
                   lbfgs_steps=5, lbfgs_max_iter=10, batch_size=128, validation_every=5)
    train, validation = _banks(cfg)
    metadata = fit_interacting(cfg, 0, tmp_path / "learning.pt", train, validation)
    assert metadata["complete"]
    assert metadata["best_validation"]["objective"] < .9 * metadata["initial_validation"]["objective"]
    fitted = load_interacting_model(tmp_path / "learning.pt")
    assert all(parameter.grad is None for parameter in fitted.parameters())


def test_default_objective_matches_original_unscaled_sum_exactly():
    mean = torch.arange(8, dtype=torch.float64).reshape(2, 4)
    covariance = torch.arange(32, dtype=torch.float64).reshape(2, 4, 4)
    model = SimpleNamespace(horizon=.0625, residual=lambda states, horizons: (mean, covariance))
    loss, values = interacting_objective(model, None, None)
    expected = model.horizon**2*(mean.square().mean()+covariance.square().mean())
    assert torch.equal(loss, expected)
    assert values == (mean, covariance)
    scaled, _ = interacting_objective(model, None, None,
                                      {"mean_scale": 2., "covariance_scale": 4.})
    assert torch.equal(scaled, model.horizon**2*(
        mean.square().mean()/4+covariance.square().mean()/16))


def test_initial_rms_uses_training_only_and_freezes_each_component(tmp_path):
    cfg = _config(loss_balancing="training-initial-rms")
    train, validation = _banks(cfg)
    altered = {**validation, "states": np.full_like(validation["states"], 1/cfg.n_cells),
               "horizons": np.zeros_like(validation["horizons"]),
               "content_hash": "different-independent-validation"}
    first = fit_interacting(cfg, 0, tmp_path/"first.pt", train, validation, deadline=-1.)
    second = fit_interacting(cfg, 0, tmp_path/"second.pt", train, altered, deadline=-1.)
    assert first["loss_scaling"] == second["loss_scaling"]
    assert first["initial_validation"] != second["initial_validation"]
    training_bank, fingerprint = training._bank(train, cfg, "train")
    torch.manual_seed(0)
    model = InteractingMomentModel(cfg.spec, cfg.physics, cfg.horizon, cfg.width)
    scales = training._initial_loss_scaling(model, training_bank, cfg, fingerprint)
    diagnostics = training._validation(model, training_bank, cfg, scales)
    assert scales == first["loss_scaling"]
    assert not scales["mean_floor_applied"] and not scales["covariance_floor_applied"]
    assert diagnostics["mean_normalized_mse"] == pytest.approx(1.)
    assert diagnostics["covariance_normalized_mse"] == pytest.approx(1.)
    assert diagnostics["objective"] == pytest.approx(2.)
    for component in ("mean", "covariance"):
        assert diagnostics[f"{component}_raw_rms"] > 0
        assert diagnostics[f"{component}_normalized_rms"] == pytest.approx(1.)
    assert scales["source_role"] == "train" and scales["frozen_before_optimizer"]
    assert "no-physical-covariance-floor" in scales["floor_interpretation"]


def test_zero_initial_residual_has_finite_declared_loss_scale_safeguard():
    cfg = _config(loss_balancing="training-initial-rms", loss_scale_floor=1e-5)
    states = torch.full((3, cfg.n_cells), 1/cfg.n_cells, dtype=torch.float64)
    model = SimpleNamespace(eval=lambda: None, state_dict=lambda: {}, horizon=cfg.horizon,
                            residual=lambda states, horizons: (
                                torch.zeros((len(states), 4), dtype=torch.float64),
                                torch.zeros((len(states), 4, 4), dtype=torch.float64)))
    bank = {"states": states, "horizons": torch.zeros(3), "family_ids": np.array(["a", "b", "c"])}
    scales = training._initial_loss_scaling(model, bank, cfg, {"role": "train"})
    assert scales["mean_scale"] == scales["covariance_scale"] == cfg.loss_scale_floor
    assert scales["mean_floor_applied"] and scales["covariance_floor_applied"]
    objective, _ = interacting_objective(model, states, bank["horizons"], scales)
    assert objective == 0. and torch.isfinite(objective)


def test_invalid_saved_scale_rejected_even_with_recomputed_identity(tmp_path):
    cfg = _config(loss_balancing="training-initial-rms")
    train, validation = _banks(cfg)
    path = tmp_path/"invalid-scale.pt"
    fit_interacting(cfg, 0, path, train, validation)
    saved = torch.load(path, weights_only=True)
    saved["provenance"]["loss_scaling"]["mean_scale"] *= 2
    saved["provenance"]["identity_hash"] = stable_hash({
        key: value for key, value in saved["provenance"].items() if key != "identity_hash"})
    torch.save(saved, path)
    with pytest.raises(ValueError, match="frozen residual scales"):
        load_interacting_model(path)
