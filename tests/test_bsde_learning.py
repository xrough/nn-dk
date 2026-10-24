"""Mathematical and reproducibility checks for the BSDE learning construction."""
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

from nn_dk.bsde_config import bsde_preset
from nn_dk.bsde_data import simulate_forward_trace
from nn_dk.bsde_model import BSDEMomentModel, moment_payoff, svec
from nn_dk.bsde_training import (
    _prepare,
    fit_bsde,
    load_bsde_model,
    terminal_objective,
    terminal_residual,
)
from nn_dk.native import initialize_masses
from nn_dk.types import Context


def _config(epochs=2):
    config = bsde_preset("bsde-smoke")
    return replace(config, width=8, training={**config.training, "epochs": epochs,
                                           "batch_size": 4, "patience": 5})


def _ou_trace(config, split):
    rng = np.random.default_rng(1 if split == "train" else 2)
    initial = np.repeat(rng.normal(size=(4, 4)), 2, axis=0)
    return simulate_forward_trace("ou", config, initial, 4 if split == "train" else 5,
                                  anchor_ids=tuple(f"{split}-a{i//2}" for i in range(8)),
                                  family_ids=tuple(f"{split}-f{i}" for i in range(8)))


def test_svec_is_frobenius_isometry_and_uses_centered_payoff():
    raw = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    symmetric = raw + raw.T
    assert torch.sum(svec(symmetric)**2).item() == pytest.approx(torch.sum(symmetric**2).item(), rel=1e-14)
    y = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64)
    payoff = moment_payoff(y, y)
    torch.testing.assert_close(payoff[:, :4], y)
    assert torch.count_nonzero(payoff[:, 4:]) == 0


def test_native_exact_time_zero_absorption_and_positive_covariance():
    config = _config()
    model = BSDEMomentModel(config, "dk")
    initial = initialize_masses(config.context("uniform"), config.spec, 3, 8)
    initial[0] = 1 / config.n_cells
    initial[0, 0] = config.spec.mass_stop
    initial[0, 1] += 1 / config.n_cells - config.spec.mass_stop
    prediction = model.predict_conditional(initial, (0., *config.query_times))
    truth = initial @ model.observation_weights.numpy().T
    np.testing.assert_allclose(prediction.mean[:, 0], truth, atol=1e-15)
    np.testing.assert_allclose(prediction.mean[0], np.broadcast_to(truth[0], prediction.mean[0].shape), atol=1e-15)
    assert np.max(np.abs(prediction.scaled_covariance[:, 0])) == 0
    assert np.max(np.abs(prediction.scaled_covariance[0])) == 0
    assert np.linalg.eigvalsh(prediction.scaled_covariance[1:, 1:]).min() > 0
    with pytest.raises(ValueError, match="declared"):
        model.predict_conditional(initial, (0.123,))


def test_initial_normalization_is_anchor_weight_invariant():
    model = BSDEMomentModel(_config(), "ou")
    initial = torch.tensor([[0., 1., 2., 3.], [0., 1., 2., 3.], [2., 3., 4., 5.]], dtype=torch.float64)
    model.fit_normalization(initial, ("a", "a", "b"))
    torch.testing.assert_close(model.feature_mean, torch.tensor([1., 2., 3., 4.], dtype=torch.float64))
    torch.testing.assert_close(model.feature_scale, torch.ones(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="different initial"):
        model.fit_normalization(initial, ("a", "a", "a"))


def test_terminal_residual_ignores_proposals_and_uses_full_active_increment():
    config = _config()
    trace = _ou_trace(config, "train")
    data = _prepare(trace, config, "ou")
    model = BSDEMomentModel(config, "ou")
    model.fit_normalization(data["states"][:, 0], trace.anchor_ids)
    with torch.no_grad():
        model.integrand_network[-1].bias.fill_(1.)
    ids = torch.arange(trace.n_paths)
    horizon = config.query_times[0]
    end = data["steps"][horizon]
    initial = data["states"][:, 0]
    times = torch.full((trace.n_paths,), horizon, dtype=torch.float64)
    center = model.deterministic_center(initial, times)
    target = moment_payoff(model.scaled_observables(data["states"][:, end]), center)
    value = model.initial_payoff(initial, times, center)
    expected = target - value - data["increments"][:, :end].sum((1, 2))[:, None]
    torch.testing.assert_close(terminal_residual(model, data, ids, horizon), expected)
    # Exiting proposals are available for numerical diagnostics but absent from all model inputs.
    original = terminal_objective(model, data, ids).detach()
    trace.proposals[:] = 1e10
    torch.testing.assert_close(terminal_objective(model, _prepare(trace, config, "ou"), ids), original)
    # Disable steps after the first: the crossing step itself must keep its FULL dW.
    data["active"][:, 1:end] = False
    expected = target - value - data["increments"][:, 0].sum(-1)[:, None]
    torch.testing.assert_close(terminal_residual(model, data, ids, horizon), expected)


def test_value_predictions_do_not_evaluate_integrand_or_center(monkeypatch):
    model = BSDEMomentModel(_config(), "ou")
    def forbidden(*args, **kwargs):
        raise AssertionError("Training-only network or center evaluated at inference")
    monkeypatch.setattr(model, "deterministic_center", forbidden)
    monkeypatch.setattr(model.integrand_network, "forward", forbidden)
    prediction = model.predict_conditional(np.zeros((2, 4)), model.config.query_times)
    assert prediction.mean.shape == (2, 3, 4)


def test_resume_reproduces_uninterrupted_fit_and_rejects_contamination(tmp_path):
    config = _config()
    training, validation = _ou_trace(config, "train"), _ou_trace(config, "validation")
    full = tmp_path / "full.pt"
    resumed = tmp_path / "resumed.pt"
    first = fit_bsde(training, validation, config, "ou", 0, full)
    interrupted = fit_bsde(training, validation, config, "ou", 0, resumed, deadline=time.monotonic()-1)
    assert not interrupted["complete"] and interrupted["epochs"] == 0
    with pytest.raises(ValueError, match="incomplete"):
        load_bsde_model(resumed)
    second = fit_bsde(training, validation, config, "ou", 0, resumed)
    assert first["complete"] and second["complete"]
    model1, model2 = load_bsde_model(full), load_bsde_model(resumed)
    for key, value in model1.state_dict().items():
        torch.testing.assert_close(value, model2.state_dict()[key], rtol=0, atol=0)
    assert first["first_epoch_seconds"] > 0
    complete_bytes = full.read_bytes()
    replay = fit_bsde(training, validation, config, "ou", 0, full)
    assert full.read_bytes() == complete_bytes
    assert replay["training_seconds"] == first["training_seconds"]
    with pytest.raises(ValueError, match="anchor"):
        fit_bsde(training, training, config, "ou", 0, tmp_path / "bad.pt")
    training.increments[0, 0, 0] += 1e-4
    with pytest.raises(ValueError, match="mismatch"):
        fit_bsde(training, validation, config, "ou", 0, resumed)


def test_ensemble_prediction_includes_between_initial_covariance():
    config = _config()
    model = BSDEMomentModel(config, "dk")
    context = config.context("uniform")
    prediction = model.predict_ensemble(context, config.queries, 8, 123)
    initial = initialize_masses(context, config.spec, 8, 123)
    conditional = model.predict_conditional(initial, config.query_times)
    expected = conditional.scaled_covariance.mean(0) + config.n_particles * np.stack(
        [np.cov(conditional.mean[:, i].T) for i in range(len(config.query_times))])
    np.testing.assert_allclose(prediction.scaled_covariance, expected, atol=1e-12)
    assert prediction.diagnostics["initial_sampling_covariance_included"]
    assert prediction.target_id == config.target_id("dk")
    assert not prediction.diagnostics["initial_profile_ood"]
    unknown = Context(cos_coeffs=(0.1, 0., 0.), n_particles=config.n_particles,
                      diffusion=config.diffusion)
    ood = model.predict_ensemble(unknown, config.queries, 8, 123)
    assert ood.diagnostics["initial_profile_ood"]
    assert not ood.diagnostics["ood_generalization_certified"]


def test_checkpoint_load_rejects_changed_source_and_data_identity(tmp_path, monkeypatch):
    import nn_dk.bsde_training as training_module

    config = _config(epochs=1)
    training, validation = _ou_trace(config, "train"), _ou_trace(config, "validation")
    path = tmp_path / "fit.pt"
    fit_bsde(training, validation, config, "ou", 0, path)
    with monkeypatch.context() as changed_source:
        changed_source.setattr(training_module, "source_hash", lambda: "different-source")
        with pytest.raises(ValueError, match="source hash"):
            load_bsde_model(path)
    saved = torch.load(path, weights_only=False)
    saved["provenance"]["training_fingerprint"] = "different-training-data"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="data identity"):
        load_bsde_model(path)
