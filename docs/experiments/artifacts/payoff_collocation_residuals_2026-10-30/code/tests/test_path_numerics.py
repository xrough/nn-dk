"""Physical path equations, stopping, and independent linear calibrations."""

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from nn_dk.native import boundary_step, native_coefficients, simulate_native
from nn_dk.native_types import NativeDKSpec
from nn_dk.observations import incidence_matrix
from nn_dk.path_numerics import (
    aggregate_brownian,
    divergence,
    ito_residual,
    linear_moments,
    path_coefficients,
    torch_euler,
    torch_stop_paths,
)
from nn_dk.types import Context


@pytest.mark.parametrize("interaction", [(0., 0.), (.75, .1)])
def test_differentiable_coefficients_match_native_force_and_covariance(interaction):
    spec = NativeDKSpec(n_cells=16)
    context = Context(kappa=interaction[0], a2=interaction[1], n_particles=2048)
    initial = np.random.default_rng(82).dirichlet(np.ones(16), size=5)
    state = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    drift, amplitude = path_coefficients(state, context, spec)
    exact = native_coefficients(initial, context, spec)
    np.testing.assert_allclose(drift.detach(), exact.drift, atol=3e-15, rtol=2e-14)
    np.testing.assert_allclose(amplitude.detach(), exact.face_amplitudes, atol=3e-16)
    b = incidence_matrix(spec.n_cells)
    covariance = np.einsum("ik,bk,jk->bij", b, amplitude.detach().numpy()**2, b)
    np.testing.assert_allclose(covariance, exact.covariance, atol=2e-15)
    (drift.square().sum() + amplitude.square().sum()).backward()
    assert torch.isfinite(state.grad).all()
    assert torch.max(torch.abs(drift.sum(-1))) < 3e-15


@pytest.mark.parametrize("interaction", [(0., 0.), (.75, .1)])
def test_torch_euler_matches_native_supplied_noise_including_absorption(interaction):
    spec = NativeDKSpec(n_cells=8)
    context = Context(kappa=interaction[0], a2=interaction[1], n_particles=32)
    dt, paths, steps = 1 / 128, 12, 8
    initial = np.full((paths, spec.n_cells), 1 / spec.n_cells)
    initial[0, :2] = 0, .25  # Initially stopped paths must stay in the batch.
    increments = np.random.default_rng(431).normal(0, np.sqrt(dt), (paths, steps, 8))
    increments[1, 0, 0] = 5  # Unambiguous exit with a strongly negative raw proposal.
    reference_trace = {}
    reference = simulate_native(context, spec, tuple(np.arange(steps + 1) * dt), dt,
                                seed=3, n_paths=paths, initial_masses=initial,
                                brownian_increments=increments, trace_capture=reference_trace)
    actual = torch_euler(initial, increments, context, spec, dt)
    np.testing.assert_allclose(actual["states"], reference.states, atol=5e-16, rtol=2e-14)
    np.testing.assert_allclose(actual["raw_proposals"], reference_trace["proposals"],
                               atol=8e-16, rtol=2e-14)
    np.testing.assert_array_equal(actual["active"], reference_trace["active"])
    np.testing.assert_allclose(actual["stopped_at"], reference.stopped_at, atol=2e-16)
    assert actual["raw_proposals"][1, 0].min() < 0
    assert actual["stopped_at"][1] < dt
    residual = ito_residual(initial, actual["raw_proposals"], increments,
                            context, spec, dt)
    assert residual["local"].abs().max() < 2e-16
    assert actual["states"].shape[0] == paths


def test_first_boundary_interpolation_preserves_raw_proposals_and_gradients():
    spec = NativeDKSpec(n_cells=8)
    initial = torch.full((1, 8), .125, dtype=torch.float64)
    raw = initial.repeat(1, 3, 1)
    raw[0, 0, 0], raw[0, 0, 1], raw[0, 0, 2] = -.2, -.1, .675
    raw[0, 1:] = initial
    raw.requires_grad_()
    packet = torch_stop_paths(initial, raw, spec, .2)
    expected, _, tau = boundary_step(initial.numpy()[0], raw.detach().numpy()[0, 0],
                                     0, .2, spec)
    np.testing.assert_allclose(packet["states"][0, 1].detach(), expected, atol=2e-16)
    np.testing.assert_array_equal(packet["active"], [[True, False, False]])
    assert packet["stopped_at"].item() == pytest.approx(tau)
    assert packet["states"][0, 1, 0].item() == pytest.approx(spec.mass_stop)
    np.testing.assert_allclose(packet["states"].detach().sum(-1), 1, atol=2e-16)
    np.testing.assert_array_equal(packet["raw_proposals"].detach(), raw.detach())
    packet["states"][0, -1, 2].backward()
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[0, 0, 2] > 0
    assert raw.grad[0, 1:].abs().sum() == 0


def test_crossing_proposal_is_not_discarded_from_ito_training_residual():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    initial = torch.full((2, 8), .125, dtype=torch.float64)
    initial[1, :2] = torch.tensor([0., .25])
    raw = initial[:, None].repeat(1, 3, 1)
    raw[0, 0, 0], raw[0, 0, 1] = -.2, .45
    raw.requires_grad_()
    packet = ito_residual(initial, raw, torch.zeros_like(raw), context, spec, .01)
    assert packet["local"][0, 0].abs().max() > .3
    assert packet["local"][0, 1:].abs().sum() == 0
    assert packet["local"][1].abs().sum() == 0
    loss = packet["cumulative"].square().mean()
    loss.backward()
    assert raw.grad[0, 0].abs().sum() > 0
    assert torch.isfinite(raw.grad).all()
    assert packet["states"].shape[0] == 2


def test_initially_absorbed_zero_faces_have_finite_backward_without_density_clipping():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    initial = torch.tensor([[0., 0., .25, .25, .125, .125, .125, .125]],
                           dtype=torch.float64, requires_grad=True)
    increments = torch.zeros((1, 2, 8), dtype=torch.float64, requires_grad=True)
    packet = torch_euler(initial, increments, context, spec, .01)
    packet["states"].square().sum().backward()
    assert torch.isfinite(initial.grad).all()
    assert torch.isfinite(increments.grad).all()
    np.testing.assert_array_equal(packet["states"].detach()[0],
                                  initial.detach().repeat(3, 1))


def test_linear_controls_do_not_use_stopping_or_multiplicative_mobility():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=1)
    initial = np.full((1, 8), .125)
    increments = np.zeros((1, 3, 8))
    increments[0, 0, 0] = 10
    heat = torch_euler(initial, increments, context, spec, .01, "heat")
    additive = torch_euler(initial, increments, context, spec, .01, "additive")
    np.testing.assert_array_equal(heat["states"], np.tile(initial[:, None], (1, 4, 1)))
    assert additive["states"].min() < 0
    assert additive["active"].all()
    assert torch.isinf(additive["stopped_at"]).all()
    _, amplitude = path_coefficients(additive["states"], context, spec, "additive")
    assert torch.unique(amplitude).numel() == 1
    assert ito_residual(initial, additive["raw_proposals"], increments,
                        context, spec, .01, "additive")["local"].abs().max() < 1e-15


def test_left_point_amplitudes_are_predictable_and_future_noise_does_not_change_past():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    initial = torch.full((1, 8), .125, dtype=torch.float64)
    increments = torch.zeros((1, 4, 8), dtype=torch.float64, requires_grad=True)
    packet = torch_euler(initial, increments, context, spec, .01)
    residual = ito_residual(initial, packet["raw_proposals"], increments,
                            context, spec, .01)
    derivative = torch.autograd.grad(residual["predictable_amplitudes"][0, 1, 0],
                                     increments, retain_graph=True)[0]
    assert derivative[0, 0].abs().sum() > 0
    assert derivative[0, 1:].abs().sum() == 0
    perturbed = increments.detach().clone()
    perturbed[:, 2:] = .1
    changed = torch_euler(initial, perturbed, context, spec, .01)
    torch.testing.assert_close(changed["states"][:, :3], packet["states"][:, :3])


def test_brownian_coupling_aggregates_without_new_randomness_or_scaling():
    rng = np.random.default_rng(733)
    fine = rng.normal(0, np.sqrt(.001), (20_000, 8, 8))
    coarse = aggregate_brownian(fine, 4)
    np.testing.assert_array_equal(coarse[:, 0], fine[:, :4].sum(1))
    assert np.var(coarse) == pytest.approx(.004, rel=.01)
    fine_t = torch.tensor(fine[:2], dtype=torch.float64, requires_grad=True)
    aggregate_brownian(fine_t, 4).sum().backward()
    torch.testing.assert_close(fine_t.grad, torch.ones_like(fine_t))
    with pytest.raises(ValueError, match="complete blocks"):
        aggregate_brownian(fine, 3)


def test_additive_one_step_quadratic_variation_matches_full_native_uniform_covariance():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    initial = np.full((40_000, 8), .125)
    dt = .001
    increments = np.random.default_rng(845).normal(0, np.sqrt(dt), (len(initial), 1, 8))
    packet = torch_euler(initial, increments, context, spec, dt, "additive")
    samples = packet["states"][:, 1].numpy() - initial
    observed = np.cov(samples, rowvar=False) / dt
    exact = native_coefficients(initial[0], context, spec).covariance / context.n_particles
    assert np.linalg.norm(observed - exact) / np.linalg.norm(exact) < .02
    np.testing.assert_allclose(observed.sum(-1), 0, atol=5e-18)
    assert np.linalg.eigvalsh(exact)[1] > 0


def test_linear_full_covariance_reference_matches_independent_block_matrix_exponential():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    p = np.arange(1, 9, dtype=float)
    p /= p.sum()
    c0 = (np.diag(p) - np.outer(p, p)) / context.n_particles
    times = (0., .1, .2)
    result = linear_moments(p, c0, times, context, spec)
    h, q = result["drift_matrix"], result["noise_covariance"]
    # Independent Van Loan formula integrates Q without spectral decomposition.
    block = np.block([[h, q], [np.zeros_like(h), -h.T]])
    for i, t in enumerate(times):
        evolution = expm(t * block)
        transition = expm(t * h)
        expected = transition @ c0 @ transition.T + evolution[:8, 8:] @ transition.T
        np.testing.assert_allclose(result["covariance"][i], expected, atol=2e-18)
        np.testing.assert_allclose(result["mean"][i], transition @ p, atol=5e-16)
    np.testing.assert_allclose(result["joint_covariance"][1, :, 2, :],
                               result["covariance"][1] @ expm(.1 * h.T), atol=2e-18)
    np.testing.assert_allclose(result["covariance"].sum(-1), 0, atol=3e-18)


def test_exact_euler_linear_covariance_and_continuous_refinement():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    p = np.full(8, .125)
    c0 = np.zeros((8, 8))
    times = (.125, .25)
    exact = linear_moments(p, c0, times, context, spec)
    errors = []
    for dt in (1 / 32, 1 / 64, 1 / 128):
        result = linear_moments(p, c0, times, context, spec, scheme="euler", dt=dt)
        h, q = result["drift_matrix"], result["noise_covariance"]
        transition = np.eye(8) + dt * h
        covariance = c0.copy()
        for _ in range(round(times[-1] / dt)):
            covariance = transition @ covariance @ transition.T + dt * q
        np.testing.assert_allclose(result["covariance"][-1], covariance, atol=3e-18)
        errors.append(np.linalg.norm(covariance - exact["covariance"][-1]))
    assert errors[0] / errors[1] > 1.9
    assert errors[1] / errors[2] > 1.9


def test_control_and_input_contracts_reject_wrong_law_without_silent_repair():
    spec = NativeDKSpec(n_cells=8)
    initial = np.full((1, 8), .125)
    with pytest.raises(ValueError, match="zero interaction"):
        path_coefficients(initial, Context(kappa=.1), spec, "additive")
    with pytest.raises(ValueError, match="nonnegative"):
        path_coefficients(-initial, Context(), spec, "dk")
    with pytest.raises(ValueError, match="CFL"):
        torch_euler(initial, np.zeros((1, 2, 8)), Context(), spec, 10, "heat")
    with pytest.raises(ValueError, match="integer multiples"):
        linear_moments(initial[0], np.zeros((8, 8)), (.1,), Context(), spec,
                       scheme="euler", dt=.03)
    face = torch.randn((3, 8), dtype=torch.float64)
    torch.testing.assert_close(divergence(face).sum(-1), torch.zeros(3, dtype=torch.float64),
                               atol=1e-15, rtol=0)
