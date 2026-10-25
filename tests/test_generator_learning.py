"""Physics, units and reproducibility of continuous neural generator moments."""
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nn_dk.bsde_config import bsde_preset
from nn_dk.generator_model import GeneratorMomentModel
from nn_dk.generator_training import (
    fit_generator,
    generator_objective,
    load_generator_model,
    validation_residuals,
)
from nn_dk.native import initialize_masses


def _base():
    return bsde_preset("bsde-smoke")


def _settings():
    return {"width": 8, "collocation_nodes": 9, "validation_nodes": 17,
            "adam_steps": 2, "lbfgs_steps": 1, "lbfgs_max_iter": 2,
            "residual_tolerance": 1e-12}


@pytest.mark.parametrize("problem", ("dk", "ou"))
def test_generator_time_zero_psd_and_initial_freeze_with_finite_gradients(problem):
    base = _base()
    model = GeneratorMomentModel(base, problem, width=8)
    if problem == "dk":
        initial = initialize_masses(base.context("uniform"), base.spec, 3, 8)
        initial[0] = 1/base.n_cells
        initial[0, 0] = base.spec.mass_stop
        initial[0, 1] += 1/base.n_cells-base.spec.mass_stop
    else:
        initial = np.random.default_rng(2).normal(size=(3, 4))
    prediction = model.predict_conditional(initial, (0., *base.query_times))
    truth = initial @ model.observation_weights.numpy().T
    np.testing.assert_allclose(prediction.mean[:, 0], truth, atol=1e-15, rtol=0)
    assert np.count_nonzero(prediction.scaled_covariance[:, 0]) == 0
    assert np.linalg.eigvalsh(prediction.scaled_covariance).min() >= -1e-12
    if problem == "dk":
        np.testing.assert_allclose(prediction.mean[0], np.broadcast_to(truth[0], prediction.mean[0].shape),
                                   atol=1e-15, rtol=0)
        assert np.count_nonzero(prediction.scaled_covariance[0]) == 0
    times = torch.tensor([0., .02, base.horizon], dtype=torch.float64, requires_grad=True)
    mean_residual, covariance_residual = model.pde_residual(times)
    (mean_residual.square().mean()+covariance_residual.square().mean()).backward()
    assert torch.isfinite(times.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    assert not hasattr(model, "integrand_network")


def test_generator_covariance_uses_correct_face_noise_units_and_rotations():
    base = _base()
    model = GeneratorMomentModel(base, "dk", width=8)
    normalized = model.normalized_weights
    identity = torch.eye(base.n_cells, dtype=torch.float64)
    face_zero, face_last = identity[1]-identity[0], identity[0]-identity[-1]
    covariance = base.diffusion/base.spec.h**2*(torch.outer(face_zero, face_zero)
                                              + torch.outer(face_last, face_last))
    torch.testing.assert_close(model.noise_kernel, normalized @ covariance @ normalized.T)
    torch.testing.assert_close(model.initial_noise_factor @ model.initial_noise_factor.T,
                               model.noise_kernel)
    (_, _), (_, derivative_at_zero) = torch.func.jvp(
        model.time_coefficients, (torch.zeros(1, dtype=torch.float64),),
        (torch.ones(1, dtype=torch.float64),))
    torch.testing.assert_close(derivative_at_zero[0], model.noise_kernel)
    # The extra Gram directions are not fixed floors: their nonzero
    # initialization prevents the zero-gradient trap and they remain trainable.
    assert torch.count_nonzero(model.time_network[-1].bias[10:]) > 0
    generator_objective(model, torch.tensor([.01, .07], dtype=torch.float64)).backward()
    assert model.time_network[-1].bias.grad[10:].abs().sum() > 0
    initial = torch.tensor(initialize_masses(base.context("uniform"), base.spec, 2, 4))
    times = torch.full((2,), .08, dtype=torch.float64)
    mean, covariance = model.moment_values(initial, times)
    shifted_mean, shifted_covariance = model.moment_values(torch.roll(initial, 1, -1), times)
    torch.testing.assert_close(shifted_mean, mean @ model.rotation_plus.T)
    torch.testing.assert_close(shifted_covariance,
                               model.rotation_plus @ covariance @ model.rotation_plus.T)
    # Covariance is affine in the complete initial histogram, rather than a
    # constant noise estimate or an accidentally N-scaled forcing term.
    midpoint = .3*initial[:1]+.7*initial[1:]
    _, middle_covariance = model.moment_values(midpoint, times[:1])
    torch.testing.assert_close(middle_covariance[0], .3*covariance[0]+.7*covariance[1])


def test_independent_matrix_exponential_kernel_satisfies_native_generator(monkeypatch):
    base = _base()
    model = GeneratorMomentModel(base, "dk", width=8)
    rates = torch.tensor(base.mode_rates, dtype=torch.float64)
    identity = torch.eye(16, dtype=torch.float64)
    rotation_plus, rotation_minus = model.rotation_plus.contiguous(), model.rotation_minus.contiguous()
    transport = base.diffusion/base.spec.h**2*(torch.kron(rotation_plus, rotation_plus)
                    + torch.kron(rotation_minus, rotation_minus)-2*identity)
    augmented = torch.zeros((32, 32), dtype=torch.float64)
    augmented[:16, :16] = transport
    augmented[:16, 16:] = identity
    augmented[16:, 16:] = -torch.diag((rates[:, None]+rates[None, :]).reshape(-1))
    source = torch.cat((torch.zeros(16, dtype=torch.float64), model.noise_kernel.reshape(-1)))

    def exact_coefficients(times):
        decays = torch.exp(-times[..., None]*rates)
        values = torch.matrix_exp(times[..., None, None]*augmented) @ source
        return decays, values[..., :16].reshape(*times.shape, 4, 4)

    # This is an independent test solution, never part of default predictions
    # or fitting labels. It checks the PDE transport/source and time JVP.
    monkeypatch.setattr(model, "time_coefficients", exact_coefficients)
    mean, covariance = model.pde_residual(torch.tensor([0., .02, .09], dtype=torch.float64))
    assert mean.abs().max() < 1e-12
    assert covariance.abs().max() < 1e-11


def test_predictions_do_not_evaluate_known_generator_coefficients(monkeypatch):
    model = GeneratorMomentModel(_base(), "dk", width=8)
    initial = np.full((2, model.config.n_cells), 1/model.config.n_cells)
    first = model.predict_conditional(initial, model.config.query_times)

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference evaluated the physics training operator")

    monkeypatch.setattr(model, "pde_residual", forbidden)
    with torch.no_grad():
        model.mode_rates.fill_(123.)
        model.noise_kernel.fill_(456.)
    second = model.predict_conditional(initial, model.config.query_times)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.scaled_covariance, second.scaled_covariance)


def test_analytic_multinomial_initialization_preserves_neural_total_covariance():
    base = replace(_base(), n_cells=8, n_particles=2)
    model = GeneratorMomentModel(base, "dk", width=8)
    probabilities = np.arange(1, 9, dtype=float)
    probabilities /= probabilities.sum()
    prediction = model.predict_ensemble(probabilities, base.query_times)
    # Enumerate all two-particle initial outcomes. These are the interior
    # polynomial map, before any stopping correction; the public ensemble
    # prediction states explicitly that this correction needs its own bound.
    initial, weights = [], []
    for first in range(base.n_cells):
        for second in range(base.n_cells):
            masses = np.zeros(base.n_cells)
            masses[first] += .5
            masses[second] += .5
            initial.append(masses)
            weights.append(probabilities[first]*probabilities[second])
    initial, weights = torch.tensor(np.asarray(initial)), torch.tensor(weights)
    with torch.no_grad():
        decays, kernel = model.time_coefficients(torch.tensor(base.query_times, dtype=torch.float64))
        conditional_mean = model.scaled_observables(initial)[:, None]*decays
        conditional_covariance = torch.einsum("bg,hgij->bhij", initial, model._rotated_kernels(kernel))
        mean = torch.einsum("b,bhi->hi", weights, conditional_mean)
        centered = conditional_mean-mean
        covariance = (torch.einsum("b,bhij->hij", weights, conditional_covariance)
                      + torch.einsum("b,bhi,bhj->hij", weights, centered, centered))
        mean, covariance = model._physical_units(mean, covariance)
    np.testing.assert_allclose(prediction.mean, mean.numpy(), atol=1e-14)
    np.testing.assert_allclose(prediction.scaled_covariance, covariance.numpy(), atol=1e-14)


def test_generator_checkpoint_resume_replay_and_source_settings_guards(tmp_path, monkeypatch):
    import nn_dk.generator_training as training_module

    base, settings = _base(), _settings()
    full, resumed = tmp_path/"full.pt", tmp_path/"resumed.pt"
    result = fit_generator(base, "ou", 0, full, settings)
    assert result["complete"] and not result["optimized"]
    clock_calls = iter(range(1, 100))
    with monkeypatch.context() as interrupted_clock:
        interrupted_clock.setattr(training_module, "time", SimpleNamespace(
            monotonic=lambda: next(clock_calls), perf_counter=time.perf_counter))
        paused = fit_generator(base, "ou", 0, resumed, settings, deadline=5)
    assert not paused["complete"]
    assert paused["adam_steps"] == 1 and paused["lbfgs_steps"] == 0
    with pytest.raises(ValueError, match="incomplete"):
        load_generator_model(resumed)
    fit_generator(base, "ou", 0, resumed, settings)
    first, second = load_generator_model(full), load_generator_model(resumed)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    immutable = full.read_bytes()
    replay = fit_generator(base, "ou", 0, full, settings)
    assert full.read_bytes() == immutable
    assert replay["training_seconds"] == result["training_seconds"]
    with pytest.raises(ValueError, match="mismatch"):
        fit_generator(base, "ou", 0, full, {**settings, "adam_steps": 3})
    saved = torch.load(full, weights_only=True)
    assert saved["provenance"]["collocation_hash"] != saved["provenance"]["validation_collocation_hash"]
    monkeypatch.setattr("nn_dk.generator_training.source_hash", lambda: "changed-source")
    with pytest.raises(ValueError, match="source hash"):
        load_generator_model(full)


def test_short_development_fit_reduces_independent_physics_residual(tmp_path):
    base = _base()
    torch.manual_seed(3)
    model = GeneratorMomentModel(base, "ou", width=8)
    times = torch.linspace(.001, base.horizon-.001, 41, dtype=torch.float64)
    before = validation_residuals(model, times)["objective"]
    settings = {**_settings(), "adam_steps": 30, "lbfgs_steps": 8, "lbfgs_max_iter": 8}
    path = tmp_path/"development.pt"
    fit_generator(base, "ou", 3, path, settings)
    fitted = load_generator_model(path)
    after = validation_residuals(fitted, times)["objective"]
    assert after < before*.05
    assert generator_objective(fitted, times).isfinite()
