"""New-payoff physics, killed integrals and resumable independent learning."""

import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import nn_dk.interacting_training as legacy_training
import nn_dk.payoff_learning as learning
from nn_dk.interacting_config import InteractingStudyConfig
from nn_dk.interacting_model import InteractingMomentModel
from nn_dk.native import native_coefficients
from nn_dk.native_types import NativeDKSpec
from nn_dk.payoff_learning import PayoffBank, PayoffMomentModel, fit_payoff, load_payoff_model
from nn_dk.types import Context, stable_hash


@pytest.fixture(autouse=True)
def fixed_source(monkeypatch):
    # Other owned new modules may still be in development. Scientific fitting
    # is separately blocked until the complete source is frozen.
    monkeypatch.setattr(learning, "source_hash", lambda: "payoff-test-frozen-source")
    monkeypatch.setattr(legacy_training, "source_hash", lambda: "payoff-test-frozen-source")


def _model(mixed=False):
    spec, context = NativeDKSpec(16), Context(kappa=.75, a2=.1, n_particles=2048)
    bank = (PayoffBank.mixed_terminal_integral if mixed else PayoffBank.terminal_high_modes)(spec, 1/16)
    torch.manual_seed(7)
    return PayoffMomentModel(spec, context, 1/16, bank, 8)


def _states(g=16, count=2):
    raw = np.random.default_rng(8).uniform(.8, 1.2, (count, g))
    return torch.tensor(raw/raw.sum(-1, keepdims=True))


def _config(**changes):
    return replace(InteractingStudyConfig(), **({"n_cells": 16, "width": 8,
        "train_paths": 1, "validation_paths": 1, "horizon_samples": 1,
        "state_fractions": (0., 1.), "adam_steps": 2, "lbfgs_steps": 1,
        "lbfgs_max_iter": 1, "batch_size": 4, "validation_every": 1} | changes))


def _banks(cfg):
    generator = np.random.default_rng(19)

    def make(role):
        count = (cfg.train_paths if role == "train" else cfg.validation_paths)*len(cfg.cases)
        repeats = len(cfg.state_times)*cfg.horizon_samples
        states = generator.uniform(.8, 1.2, (count, cfg.n_cells))
        states /= states.sum(-1, keepdims=True)
        return {"states": np.repeat(states, repeats, axis=0),
                "horizons": generator.uniform(0, cfg.horizon, count*repeats),
                "family_ids": np.repeat([f"{role}-{i}" for i in range(count)], repeats),
                "content_hash": f"frozen-{role}", "source_hash": learning.source_hash(),
                "config_hash": cfg.config_hash, "role": role}

    return make("train"), make("validation")


def test_payoff_weights_exact_cell_integrals_hash_and_roundtrip():
    spec = NativeDKSpec(32)
    bank = PayoffBank.terminal_high_modes(spec, 1/16)
    edges = np.arange(33)*spec.h
    for row, (k, kind) in enumerate(((3, "sin"), (3, "cos"), (4, "sin"), (4, "cos"))):
        exact = ((np.cos(k*edges[:-1])-np.cos(k*edges[1:])) if kind == "sin" else
                 (np.sin(k*edges[1:])-np.sin(k*edges[:-1])))/(k*spec.h)
        np.testing.assert_allclose(bank.terminal_weights[row], exact, atol=2e-15)
    restored = PayoffBank.from_dict(bank.to_dict())
    assert restored.bank_hash == bank.bank_hash
    assert bank.bank_hash != PayoffBank.mixed_terminal_integral(spec, 1/16).bank_hash
    assert not bank.terminal_weights.flags.writeable
    bad = bank.to_dict() | {"stopping": "integrate-after-stop"}
    with pytest.raises(ValueError, match="convention"):
        PayoffBank.from_dict(bad)


def test_killed_integral_reduction_uses_partial_segment_and_retains_every_path():
    model = _model(True)
    initial = _states(count=3).numpy()
    final = np.roll(initial, 1, axis=-1)
    times = np.array([0., model.horizon/2, model.horizon])
    # A boundary hit at half of the first segment, an initially killed path,
    # and one full active trajectory. Frozen mass states are not integrated
    # after their respective stopping times.
    states = np.stack([initial, final, final], axis=1)
    states[1] = initial[1]
    stops = np.array([model.horizon/4, 0., np.inf])
    values = model.bank.evaluate_paths(states, times, stops)
    np.testing.assert_allclose(values[:, :, :2], (states@model.bank.terminal_weights.T)[:, :, :2])
    running = states@model.bank.running_weights.T
    np.testing.assert_allclose(values[0, 1:, 2:],
        np.broadcast_to(.5*stops[0]*(running[0, 0, 2:]+running[0, 1, 2:]), (2, 2)))
    np.testing.assert_array_equal(values[1, :, 2:], 0.)
    expected = model.horizon/4*(running[2, 0, 2:]+3*running[2, 1, 2:])
    np.testing.assert_allclose(values[2, -1, 2:], expected)


@pytest.mark.parametrize("mixed", [False, True])
def test_initial_boundary_and_covariance_positive(mixed):
    model, states = _model(mixed), _states()
    mean, covariance = model.moment_values(states, torch.zeros(2))
    torch.testing.assert_close(mean, model.scaled_observables(states))
    assert torch.count_nonzero(covariance) == 0
    boundary = states.clone()
    boundary[:, 0] = model.spec.mass_stop
    boundary[:, 1:] *= (1-model.spec.mass_stop)/boundary[:, 1:].sum(-1, keepdim=True)
    mean, covariance = model.moment_values(boundary, torch.full((2,), model.horizon))
    torch.testing.assert_close(mean, model.scaled_observables(boundary))
    assert torch.count_nonzero(covariance) == 0
    mr, cr = model.residual(boundary, torch.full((2,), model.horizon))
    assert torch.count_nonzero(mr) == torch.count_nonzero(cr) == 0
    result = model.predict_conditional(states.numpy(), [0., 1/32, 1/16])
    assert np.linalg.eigvalsh(result.scaled_covariance).min() >= -1e-12
    assert result.target_id != InteractingStudyConfig().target_id


def test_integral_small_time_variance_orders_and_known_initial_qv():
    model = _model(True)
    states = torch.full((2, 16), 1/16, dtype=torch.float64)
    times = torch.tensor([1e-5, 2e-5], dtype=torch.float64)
    _, covariance = model.moment_values(states, times)
    ratios = covariance[1].diagonal()/covariance[0].diagonal()
    torch.testing.assert_close(ratios, torch.tensor([2., 2., 8., 8.], dtype=torch.float64),
                               rtol=1e-13, atol=1e-13)
    # The initialized factor is derived solely from instantaneous native QV.
    qv = native_coefficients(np.full(16, 1/16), model.context, model.spec).covariance
    phi, q = model.bank.terminal_weights, model.bank.running_weights
    expected = phi @ qv @ phi.T + model.horizon**2/3*q @ qv @ q.T
    scale = model.bank.scales
    factor = model._factor(model.network[-1].bias[4:])
    np.testing.assert_allclose((factor@factor.T).detach().numpy(), expected/scale[:, None]/scale[None, :], atol=2e-13)


def test_identical_terminal_integral_modes_have_correct_cross_leading_factor():
    model = _model(True)
    phi, q = model.bank.terminal_weights.copy(), model.bank.running_weights.copy()
    q[2:] = phi[:2]/model.horizon
    bank = PayoffBank("terminal-and-integral-same-mode3", phi, q, model.horizon)
    model = PayoffMomentModel(model.spec, model.context, model.horizon, bank, 8)
    qv = native_coefficients(np.full(16, 1/16), model.context, model.spec).covariance
    expected = (phi@qv@phi.T + model.horizon/2*(phi@qv@q.T+q@qv@phi.T)
                + model.horizon**2/3*q@qv@q.T)
    factor = model._factor(model.network[-1].bias[4:])
    np.testing.assert_allclose((factor@factor.T).detach().numpy(),
                              expected/bank.scales[:, None]/bank.scales[None, :], atol=2e-13)


@pytest.mark.parametrize("mixed", [False, True])
def test_generator_jet_residual_matches_dense_hessians_including_running_source(mixed):
    model = _model(mixed)
    # Exercise nonlinear network terms and tau-dependent integral covariance.
    with torch.no_grad():
        model.network[-1].weight.normal_(0, .003)
    initial = _states(count=1)[0].requires_grad_(True)
    horizon = torch.tensor(.027, dtype=torch.float64, requires_grad=True)
    coefficients = native_coefficients(initial.detach().numpy(), model.context, model.spec)
    drift = torch.tensor(coefficients.drift)
    covariance = torch.tensor(coefficients.covariance/model.context.n_particles)
    mean, cov = model.moment_values(initial, horizon)
    expected_mean, expected_cov = [], []

    def generator(function):
        gradient = torch.autograd.functional.jacobian(function, initial)
        hessian = torch.autograd.functional.hessian(function, initial)
        return gradient@drift + .5*(hessian*covariance).sum()

    for i in range(4):
        tderiv, = torch.autograd.grad(mean[i], horizon, retain_graph=True)
        source = math_sqrt(model.context.n_particles)*(initial@model.normalized_running_weights[i])
        expected_mean.append(tderiv-generator(lambda m: model.moment_values(m, horizon)[0][i])-source)
    gradients = torch.autograd.functional.jacobian(lambda m: model.moment_values(m, horizon)[0], initial)
    quadratic = gradients@covariance@gradients.T
    for i in range(4):
        row = []
        for j in range(4):
            tderiv, = torch.autograd.grad(cov[i, j], horizon, retain_graph=True)
            row.append(tderiv-generator(lambda m: model.moment_values(m, horizon)[1][i, j])-quadratic[i, j])
        expected_cov.append(torch.stack(row))
    residuals = model.residual(initial[None], horizon[None])
    torch.testing.assert_close(residuals[0][0], torch.stack(expected_mean), atol=2e-10, rtol=1e-10)
    torch.testing.assert_close(residuals[1][0], torch.stack(expected_cov), atol=2e-10, rtol=1e-10)
    loss = sum(r.square().mean() for r in residuals)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def math_sqrt(value):
    return float(np.sqrt(value))


def test_transfer_only_hidden_layers_and_reset_new_payoff_head():
    model = _model()
    old = InteractingMomentModel(model.spec, model.context, model.horizon, 8)
    with torch.no_grad():
        for parameter in old.parameters():
            parameter.add_(.1)
    fresh = PayoffMomentModel(model.spec, model.context, model.horizon, model.bank, 8,
                             transfer_state=old.state_dict())
    for index in (0, 2, 4):
        torch.testing.assert_close(fresh.network[index].weight, old.network[index].weight)
    assert torch.count_nonzero(fresh.network[-1].weight) == 0
    torch.testing.assert_close(fresh.network[-1].bias, model.network[-1].bias)
    torch.testing.assert_close(fresh.observation_weights, model.observation_weights)
    assert fresh.transfer_hash
    with pytest.raises(ValueError, match="shape"):
        PayoffMomentModel(model.spec, model.context, model.horizon, model.bank, 8, {})


def _nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            _nested_equal(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("balancing", ["none", "training-initial-rms"])
@pytest.mark.parametrize("deadline", [5, 9])
def test_fit_resume_replay_and_strict_checkpoint_guards(tmp_path, monkeypatch, balancing, deadline):
    cfg = _config(loss_balancing=balancing, lbfgs_steps=2)
    train, val = _banks(cfg)
    bank = PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon)
    full, partial = tmp_path/"full.pt", tmp_path/"resume.pt"
    metadata = fit_payoff(cfg, bank, 0, full, train, val)
    assert metadata["complete"]
    clock = iter(range(1, 100))
    with monkeypatch.context() as interrupt:
        interrupt.setattr(learning, "time", SimpleNamespace(
            monotonic=lambda: next(clock), perf_counter=time.perf_counter))
        result = fit_payoff(cfg, bank, 0, partial, train, val, deadline=deadline)
    assert not result["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        load_payoff_model(partial)
    fit_payoff(cfg, bank, 0, partial, train, val)
    a, b = torch.load(full, weights_only=True), torch.load(partial, weights_only=True)
    for key in ("model_state", "best_state", "initial_state", "adam_state", "lbfgs_state",
                "torch_rng_state", "shuffle_rng_state", "best_validation", "best_step", "provenance"):
        _nested_equal(a[key], b[key])
    model = load_payoff_model(partial)
    assert model.bank.bank_hash == bank.bank_hash
    before = partial.read_bytes()
    fit_payoff(cfg, bank, 0, partial, train, val)
    assert partial.read_bytes() == before
    with pytest.raises(ValueError, match="resume"):
        fit_payoff(cfg, PayoffBank.terminal_high_modes(cfg.spec, cfg.horizon), 0, partial, train, val)
    corrupted = tmp_path/"corrupted.pt"
    a["best_state"]["network.0.weight"][0, 0] += 1
    torch.save(a, corrupted)
    with pytest.raises(ValueError, match="checksum"):
        load_payoff_model(corrupted)
    monkeypatch.setattr(learning, "source_hash", lambda: "changed-source")
    with pytest.raises(ValueError, match="source hash"):
        load_payoff_model(partial)


def test_transferred_fit_loader_preserves_initial_state_and_bank_integrity(tmp_path):
    cfg = _config(adam_steps=1, lbfgs_steps=0)
    train, val = _banks(cfg)
    bank = PayoffBank.terminal_high_modes(cfg.spec, cfg.horizon)
    torch.manual_seed(34)
    original = InteractingMomentModel(cfg.spec, cfg.physics, cfg.horizon, cfg.width).state_dict()
    path = tmp_path/"transfer.pt"
    fit_payoff(cfg, bank, 0, path, train, val, transfer_state=original)
    state = torch.load(path, weights_only=True)
    loaded = load_payoff_model(path)
    assert loaded.provenance["transfer_trunk_hash"]
    torch.testing.assert_close(state["initial_state"]["network.0.weight"], original["network.0.weight"])
    assert torch.count_nonzero(state["initial_state"]["network.6.weight"]) == 0
    # Even recomputed state/provenance hashes cannot relabel geometry or the
    # declared observation buffers into another target.
    state["best_state"] = {key: value.clone() for key, value in state["best_state"].items()}
    state["best_state"]["observation_weights"][0, 0] += .1
    state["best_state_hash"] = learning._state_hash(state["best_state"])
    bad = tmp_path/"wrong-bank.pt"
    torch.save(state, bad)
    with pytest.raises(ValueError, match="observable or geometry"):
        load_payoff_model(bad)
    state["kind"] = "continuous-interacting-native-dk-generator-moments-v1"
    torch.save(state, bad)
    with pytest.raises(ValueError, match="terminal/integral"):
        load_payoff_model(bad)


def test_no_future_labels_or_overlapping_families(tmp_path):
    cfg = _config()
    train, val = _banks(cfg)
    bank = PayoffBank.terminal_high_modes(cfg.spec, cfg.horizon)
    train["future_payoff_labels"] = True
    with pytest.raises(ValueError, match="source, configuration"):
        fit_payoff(cfg, bank, 0, tmp_path/"a.pt", train, val)
    train["future_payoff_labels"] = False
    val["family_ids"] = train["family_ids"].copy()
    with pytest.raises(ValueError, match="disjoint"):
        fit_payoff(cfg, bank, 0, tmp_path/"a.pt", train, val)


def test_bank_rejects_combined_rows_and_changed_stopping():
    bank = _model().bank
    with pytest.raises(ValueError, match="either terminal"):
        PayoffBank("bad", bank.terminal_weights, bank.terminal_weights, bank.horizon)
    assert stable_hash(bank.to_dict()) == bank.bank_hash
