"""Full-state interacting conditional moment invariants and exact Itô residuals."""

import numpy as np
import pytest
import torch

from nn_dk.interacting_model import InteractingMomentModel
from nn_dk.native import native_coefficients, physical_observable_weights
from nn_dk.native_types import NativeDKSpec
from nn_dk.path_numerics import path_coefficients
from nn_dk.types import Context, QueryBank, seeded_rng


def _model(n_cells=32):
    torch.manual_seed(12)
    return InteractingMomentModel(NativeDKSpec(n_cells),
                                  Context(kappa=.75, a2=.1, n_particles=2048), 1., width=8)


def _states(model, count=3):
    generator = torch.Generator().manual_seed(42)
    raw = .15 * torch.randn(count, model.spec.n_cells, generator=generator, dtype=torch.float64)
    return torch.softmax(raw, -1)


def _nontrivial_network(model):
    generator = torch.Generator().manual_seed(21)
    with torch.no_grad():
        model.network[-1].weight.copy_(.03 * torch.randn(model.network[-1].weight.shape,
                                                        generator=generator, dtype=torch.float64))
        model.network[-1].bias[:4].copy_(torch.tensor([.1, -.05, .02, .03]))


def test_exact_initial_values_absorbing_boundary_and_psd():
    model = _model()
    _nontrivial_network(model)
    initial = _states(model)
    mean, covariance = model.moment_values(initial, torch.zeros(3))
    torch.testing.assert_close(mean, model.scaled_observables(initial), atol=0, rtol=0)
    assert torch.equal(covariance, torch.zeros_like(covariance))
    boundary = initial.clone()
    removed = boundary[:, 0] - model.spec.mass_stop
    boundary[:, 0] = model.spec.mass_stop
    boundary[:, 1] += removed
    mean, covariance = model.moment_values(boundary, torch.tensor([.1, .5, 1.]))
    torch.testing.assert_close(mean, model.scaled_observables(boundary), atol=0, rtol=0)
    assert torch.equal(covariance, torch.zeros_like(covariance))
    mean_residual, covariance_residual = model.residual(boundary, torch.tensor([.1, .5, 1.]))
    assert torch.equal(mean_residual, torch.zeros_like(mean_residual))
    assert torch.equal(covariance_residual, torch.zeros_like(covariance_residual))
    _, covariance = model.moment_values(initial, torch.tensor([.1, .5, 1.]))
    assert bool((torch.linalg.eigvalsh(covariance) >= -1e-12).all())


def test_boundary_strip_is_bounded_and_flat_at_typical_interior_states():
    model = _model()
    states = _states(model, 16)
    barrier = model.barrier(states)
    assert model.boundary_width == .05 / 32
    assert bool(((barrier > 0) & (barrier <= 1)).all())
    assert float(barrier.min()) > 1 - 1e-8
    states[0, 0] = model.spec.mass_stop
    assert model.barrier(states)[0] == 0


def test_native_noise_conservation_and_scaled_coordinate_qv():
    model = _model()
    states = _states(model)
    drift, amplitudes = path_coefficients(states, model.context, model.spec)
    face = model.incidence * amplitudes.unsqueeze(-2) / np.sqrt(model.context.n_particles)
    torch.testing.assert_close(drift.sum(-1), torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(face.sum(-2), torch.zeros(3, 32, dtype=torch.float64), atol=0, rtol=0)
    normalized_face = np.sqrt(model.context.n_particles * model.spec.n_cells) * (
        model.basis.T @ face)
    for row in range(len(states)):
        exact = native_coefficients(states[row].numpy(), model.context, model.spec)
        expected = model.spec.n_cells * model.basis.T.numpy() @ exact.covariance @ model.basis.numpy()
        torch.testing.assert_close(normalized_face[row] @ normalized_face[row].T,
                                   torch.tensor(expected), atol=5e-12, rtol=2e-14)


def test_full_model_jets_and_covariance_pde_against_dense_hessians():
    model = _model(8)
    _nontrivial_network(model)
    state = _states(model, 1)[0].clone()
    # Exercise the nonconstant boundary strip as well as neural derivatives.
    new_first = model.spec.mass_stop + .7 * model.boundary_width
    state[1] += state[0] - new_first
    state[0] = new_first
    time = torch.tensor(.31, dtype=torch.float64)
    point = torch.cat((state, time[None]))

    def function(z):
        mean, covariance = model._values_unchecked(z[:-1], z[-1])
        return torch.cat((mean, covariance.reshape(-1)))

    jacobian = torch.func.jacrev(function)(point)
    hessian = torch.func.jacrev(torch.func.jacrev(function))(point)
    drift, amplitudes = path_coefficients(state, model.context, model.spec)
    faces = model.incidence * amplitudes[None] / np.sqrt(model.context.n_particles)
    action = jacobian[:, :-1] @ drift + .5 * torch.einsum(
        "oij,if,jf->o", hessian[:, :-1, :-1], faces, faces)
    mean_jet, covariance_jet = model._moment_jets(state[None], time[None])
    for actual, expected in (
        (mean_jet.value[0], function(point)[:4]),
        (mean_jet.generator[0], action[:4]),
        (mean_jet.time_derivative[0], jacobian[:4, -1]),
        (mean_jet.face_derivatives[0], jacobian[:4, :-1] @ faces),
        (covariance_jet.generator[0], action[4:].reshape(4, 4)),
        (covariance_jet.time_derivative[0], jacobian[4:, -1].reshape(4, 4)),
    ):
        torch.testing.assert_close(actual, expected, atol=5e-10, rtol=5e-11)
    mean_residual, covariance_residual = model.residual(state[None], time[None])
    source = (jacobian[:4, :-1] @ faces) @ (jacobian[:4, :-1] @ faces).T
    torch.testing.assert_close(mean_residual[0], jacobian[:4, -1] - action[:4],
                               atol=5e-10, rtol=5e-11)
    torch.testing.assert_close(covariance_residual[0],
                               (jacobian[4:, -1] - action[4:]).reshape(4, 4) - source,
                               atol=5e-10, rtol=5e-11)


def test_interaction_cannot_close_on_the_four_observed_modes():
    model = _model()
    x = (np.arange(32) + .5) * model.spec.h
    first = (1 + .2 * np.cos(x) + .1 * np.cos(3 * x)) / 32
    second = (1 + .2 * np.cos(x) - .1 * np.cos(3 * x)) / 32
    weights = physical_observable_weights(model.spec, QueryBank())
    np.testing.assert_allclose(weights @ first, weights @ second, atol=1e-15, rtol=0)
    drift_difference = weights @ (native_coefficients(first, model.context, model.spec).drift
                                  - native_coefficients(second, model.context, model.spec).drift)
    assert drift_difference[2] == pytest.approx(-.0100040286, abs=5e-11)
    assert model.feature_dimension == 31


def test_parameter_gradients_and_inactive_safe_origins_are_finite():
    model = _model()
    _nontrivial_network(model)
    states = _states(model)
    states[0, 1] += states[0, 0]
    states[0, 0] = 0
    mean, covariance = model.residual(states, torch.tensor([0., .25, .75]))
    loss = mean.square().mean() + covariance.square().mean()
    loss.backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())


def test_reject_incompatible_inputs_without_repair():
    model = _model()
    state = _states(model, 1)
    for bad in (-state, state * .9, state * float("nan")):
        with pytest.raises(ValueError):
            model.moment_values(bad, .1)
    with pytest.raises(ValueError):
        model.moment_values(state, 1.1)
    with pytest.raises(ValueError):
        model.require_physics(Context(kappa=.74, a2=.1, n_particles=2048))
    model.require_physics(Context(cos_coeffs=(.2, 0., 0.), kappa=.75, a2=.1, n_particles=2048))


def test_public_prediction_units_and_profile_independent_target():
    model = _model()
    states = _states(model, 2).numpy()
    prediction = model.predict_conditional(states, [0., .25, 1.])
    expected = states @ model.observation_weights.numpy().T
    np.testing.assert_allclose(prediction.mean[:, 0], expected, atol=1e-15, rtol=0)
    assert prediction.scaled_covariance.shape == (2, 3, 4, 4)
    assert not prediction.diagnostics["cross_time_covariance_supported"]
    other = InteractingMomentModel(model.spec,
                                   Context(cos_coeffs=(.2, 0., 0.), kappa=.75, a2=.1,
                                           n_particles=2048), 1., width=8)
    assert other.target_id == model.target_id


def test_ensemble_initial_sampling_total_covariance_and_replay():
    model = _model()
    probabilities = np.full(32, 1 / 32)
    samples, seed = 512, 71
    prediction = model.predict_ensemble(probabilities, [0., .25], n_samples=samples, seed=seed)
    rng = seeded_rng(seed, "interacting-initial-mc", model.target_id)
    initial = rng.multinomial(model.context.n_particles, probabilities,
                              size=samples) / model.context.n_particles
    observed = initial @ model.observation_weights.numpy().T
    np.testing.assert_allclose(prediction.mean[0], observed.mean(0), atol=1e-15, rtol=0)
    np.testing.assert_allclose(prediction.scaled_covariance[0],
                               model.context.n_particles * np.cov(observed, rowvar=False, ddof=1),
                               atol=5e-15, rtol=5e-14)
    assert prediction.diagnostics["initial_samples"] == samples
    assert not prediction.diagnostics["initial_sampling_uncertainty_certified"]
    replay = model.predict_ensemble(probabilities, [0., .25], n_samples=samples, seed=seed)
    np.testing.assert_array_equal(replay.mean, prediction.mean)
    np.testing.assert_array_equal(replay.scaled_covariance, prediction.scaled_covariance)


def test_ensemble_includes_absorbed_initial_histograms_without_survivors():
    model = _model()
    probabilities = np.zeros(32)
    probabilities[0] = 1
    prediction = model.predict_ensemble(probabilities, [0., .25, 1.], n_samples=4, seed=12)
    expected = model.observation_weights[:, 0].numpy()
    np.testing.assert_allclose(prediction.mean, np.broadcast_to(expected, (3, 4)), atol=1e-15, rtol=0)
    np.testing.assert_array_equal(prediction.scaled_covariance, np.zeros((3, 4, 4)))
