"""Fourier classical baselines and pathwise sensitivities are physical comparators."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from nn_dk.native import native_coefficients
from nn_dk.native_types import NativeDKSpec
from nn_dk.path_numerics import torch_euler
from nn_dk.types import Context

_spec = importlib.util.spec_from_file_location(
    "probe_path_workloads", Path(__file__).parents[1]/"analysis"/"probe_path_workloads.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


@pytest.mark.parametrize("interaction", [(0., 0.), (.75, .1), (-.4, -.2)])
def test_fourier_force_drift_and_amplitude_match_native(interaction):
    spec = NativeDKSpec(n_cells=16)
    context = Context(kappa=interaction[0], a2=interaction[1], n_particles=2048)
    initial = np.random.default_rng(21).dirichlet(np.ones(16), size=7)
    drift, noise = probe.fourier_coefficients(initial, context, spec)
    reference = native_coefficients(initial, context, spec)
    np.testing.assert_allclose(drift, reference.drift, atol=3e-15, rtol=2e-14)
    np.testing.assert_allclose(noise, reference.face_amplitudes, atol=5e-16)


@pytest.mark.parametrize("crossing", [False, True])
def test_full_numpy_and_torch_paths_match_native_reference_including_stops(crossing):
    spec, context = NativeDKSpec(n_cells=8), Context(kappa=.75, a2=.1, n_particles=64)
    initial = np.full((4, 8), 1/8)
    increments = np.random.default_rng(4).normal(0, np.sqrt(.01), (4, 8, 8))
    if crossing:
        initial[0, :2] = 0, .25
        increments[1, 0, 0] = 8
    expected = torch_euler(initial, increments, context, spec, .01)
    for actual in (probe.numpy_euler(initial, increments, context, spec, .01),
                   probe.torch_fourier_euler(initial, increments, context, spec, .01)):
        np.testing.assert_allclose(actual["states"], expected["states"], atol=6e-16, rtol=2e-14)
        np.testing.assert_allclose(actual["stopped_at"], expected["stopped_at"], atol=2e-16)
        np.testing.assert_allclose(actual["states"].sum(-1), 1, atol=4e-16)
        assert actual["states"].shape[0] == 4
    if crossing:
        assert torch.isfinite(expected["stopped_at"][:2]).all()


@pytest.mark.parametrize("crossing", [False, True])
@pytest.mark.parametrize("parameter", ["initial", "diffusion", "kappa", "a2"])
def test_exact_tangent_matches_finite_difference_and_reverse_mode(parameter, crossing):
    spec, context = NativeDKSpec(n_cells=8), Context(kappa=.75, a2=.1, n_particles=64)
    initial = np.full((3, 8), 1/8)
    increments = np.random.default_rng(42).normal(0, np.sqrt(.01), (3, 6, 8))
    if crossing:
        increments[0, 0, 0] = 4
    centers = (np.arange(8)+.5)*spec.h
    direction = np.cos(3*centers)/8
    pdir = np.zeros(3)
    if parameter != "initial":
        pdir[("diffusion", "kappa", "a2").index(parameter)] = 1
        direction = np.zeros(8)
    actual = probe.numpy_euler(initial, increments, context, spec, .01,
                               initial_tangent=direction, physics_tangent=pdir)
    epsilon = 1e-6
    base = np.array((context.diffusion, context.kappa, context.a2))

    def shifted(sign):
        p = base+sign*epsilon*pdir
        ctx = Context(diffusion=p[0], kappa=p[1], a2=p[2], n_particles=context.n_particles)
        return probe.numpy_euler(initial+sign*epsilon*direction, increments, ctx, spec, .01)

    plus, minus = shifted(1), shifted(-1)
    fd = (plus["states"]-minus["states"])/(2*epsilon)
    np.testing.assert_allclose(actual["tangent"], fd, atol=5e-10, rtol=1e-6)
    np.testing.assert_allclose(actual["tangent"].sum(-1), 0, atol=3e-15)
    if crossing:
        stop_fd = (plus["stopped_at"][0]-minus["stopped_at"][0])/(2*epsilon)
        assert actual["stopped_at_tangent"][0] == pytest.approx(stop_fd, abs=2e-11, rel=1e-5)
    tinitial = torch.tensor(initial, requires_grad=True)
    physics = torch.tensor(base, requires_grad=True)
    torch_states = probe.torch_fourier_euler(
        tinitial, increments, context, spec, .01, physics=physics)["states"]
    values = probe.workload_values(torch_states, spec, .01)
    gradient, pgradient = torch.autograd.grad(values.sum(), (tinitial, physics))
    derivative = probe.workload_jvp(actual["states"], actual["tangent"], spec, .01).sum()
    adjoint_derivative = (gradient.detach().numpy()*direction).sum()+pgradient.detach().numpy() @ pdir
    assert derivative == pytest.approx(adjoint_derivative, abs=2e-12, rel=2e-11)
    discrete_adjoint = probe.numpy_adjoint(initial, increments, context, spec, .01)
    np.testing.assert_allclose(discrete_adjoint["initial_gradient"], gradient.detach().numpy(),
                               atol=3e-14, rtol=2e-13)
    np.testing.assert_allclose(discrete_adjoint["physics_gradient"].sum(0),
                               pgradient.detach().numpy(), atol=3e-14, rtol=2e-13)


def test_initially_stopped_paths_have_finite_jvp_and_adjoint_without_repair():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=64)
    initial = np.array([[0, 0, .25, .25, .125, .125, .125, .125]])
    increments = np.zeros((1, 4, 8))
    direction = np.array([0, 0, .1, -.1, 0, 0, 0, 0])
    actual = probe.numpy_euler(initial, increments, context, spec, .01, initial_tangent=direction)
    np.testing.assert_array_equal(actual["states"], np.repeat(initial[:, None], 5, axis=1))
    np.testing.assert_array_equal(actual["tangent"], np.tile(direction, (1, 5, 1)))
    state = torch.tensor(initial, requires_grad=True)
    output = probe.torch_fourier_euler(state, increments, context, spec, .01)["states"]
    gradient = torch.autograd.grad(output.square().sum(), state)[0]
    assert torch.isfinite(gradient).all()
    adjoint = probe.numpy_adjoint(initial, increments, context, spec, .01,
                                  path_gradient=2*actual["states"])
    np.testing.assert_allclose(adjoint["initial_gradient"], gradient.detach(), atol=3e-16)
    np.testing.assert_array_equal(adjoint["physics_gradient"], 0.)


def test_workloads_numpy_torch_match_and_many_observables_have_expected_shape():
    spec = NativeDKSpec(n_cells=32)
    states = np.random.default_rng(1).dirichlet(np.ones(32), size=(3, 20))
    expected = probe.workload_values(states, spec, .01)
    np.testing.assert_allclose(probe.workload_values(torch.tensor(states), spec, .01),
                               expected, atol=3e-16)
    observables = probe.many_observables(states, spec, n_times=16)
    assert observables.shape == (3, 16*30)
    np.testing.assert_allclose(probe.many_observables(torch.tensor(states), spec, n_times=16),
                               observables, atol=3e-16)


@pytest.mark.parametrize("crossing", [False, True])
def test_arbitrary_full_path_adjoint_cotangent_matches_autograd(crossing):
    spec, context = NativeDKSpec(n_cells=8), Context(kappa=.75, a2=.1, n_particles=64)
    rng = np.random.default_rng(123)
    initial = np.tile(np.array([.1, .11, .12, .13, .13, .12, .14, .15]), (4, 1))
    noise = rng.normal(0, .1, (4, 12, 8))
    if crossing:
        noise[0, 3, 2] = -7
    initial_tensor = torch.tensor(initial, requires_grad=True)
    physics = torch.tensor([1., .75, .1], dtype=torch.float64, requires_grad=True)
    states = probe.torch_fourier_euler(initial_tensor, noise, context, spec, .01,
                                       physics=physics)["states"]
    cotangent = rng.normal(size=states.shape)
    expected_initial, expected_physics = torch.autograd.grad(
        (states*torch.tensor(cotangent)).sum(), (initial_tensor, physics))
    actual = probe.numpy_adjoint(initial, noise, context, spec, .01, path_gradient=cotangent)
    np.testing.assert_allclose(actual["initial_gradient"], expected_initial.detach(),
                               atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(actual["physics_gradient"].sum(0), expected_physics.detach(),
                               atol=3e-14, rtol=3e-13)


def test_probe_records_complete_declared_comparison_scope():
    result = probe.probe(steps=(4,), batches=(1,), horizon=1/64, repeats=1, warmup=0)
    assert len(result["records"]) == 7
    assert all(record["median_seconds"] > 0 for record in result["records"])
    assert all(record["conservation_error"] < 1e-14 for record in result["records"])
    assert "no NN" in result["claim"]
    assert result["cpu_threads"] == 1


def test_invalid_inputs_and_nonconservative_tangent_are_rejected():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=64)
    with pytest.raises(ValueError, match="stability"):
        probe.numpy_euler(np.full((1, 8), 1/8), np.zeros((1, 3, 8)), context, spec, 1.)
    with pytest.raises(ValueError, match="conservative"):
        probe.numpy_euler(np.full((1, 8), 1/8), np.zeros((1, 3, 8)), context, spec, .01,
                           initial_tangent=np.ones(8))
