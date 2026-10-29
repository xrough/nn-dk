"""Numerical equivalence and strict causality of optional path acceleration."""

import numpy as np
import pytest
import torch

from nn_dk.native_types import NativeDKSpec
from nn_dk.path_acceleration import (
    RESPONSE_BACKENDS,
    accelerated_response,
    prepare_response,
    vectorized_stop_paths,
)
from nn_dk.path_model import CausalPathOperator
from nn_dk.path_numerics import torch_stop_paths
from nn_dk.types import Context


@pytest.mark.parametrize("steps", [1, 7, 16, 33])
@pytest.mark.parametrize("mode", tuple(RESPONSE_BACKENDS))
def test_response_matches_dense_with_full_grid_and_gradients(steps, mode):
    torch.manual_seed(400 + steps)
    model = CausalPathOperator(NativeDKSpec(n_cells=8), "additive", width=4)
    initial = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    forcing = torch.randn(3, steps, 8, dtype=torch.float64, requires_grad=True)
    kernel = torch.randn(steps + 1, 5, dtype=torch.float64, requires_grad=True)
    expected = model._response(initial, forcing, kernel)
    actual = accelerated_response(initial, forcing, kernel, mode=mode)
    torch.testing.assert_close(actual, expected, atol=3e-14, rtol=3e-14)
    coefficients = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad((expected * coefficients).sum(),
                                           (initial, forcing, kernel), retain_graph=True)
    actual_gradients = torch.autograd.grad((actual * coefficients).sum(),
                                         (initial, forcing, kernel))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-13, rtol=2e-13)
    assert torch.equal(actual[:, 0], initial)
    torch.testing.assert_close(actual.sum(-1), initial.sum(-1)[:, None].expand(-1, steps + 1),
                               atol=2e-14, rtol=0)


@pytest.mark.parametrize("mode", tuple(RESPONSE_BACKENDS))
@pytest.mark.parametrize("cut", [1, 3, 8, 12, 19])
def test_response_exactly_rejects_future_forcing(mode, cut):
    torch.manual_seed(51)
    initial = torch.full((2, 8), 1 / 8, dtype=torch.float64)
    forcing = torch.randn(2, 21, 8, dtype=torch.float64, requires_grad=True)
    kernel = torch.randn(22, 5, dtype=torch.float64)
    plan = prepare_response(kernel, 8, mode=mode)
    actual = plan.apply(initial, forcing)
    changed = forcing.detach().clone()
    changed[:, cut:] = torch.randn_like(changed[:, cut:]) * 1e8
    alternate = plan.apply(initial, changed)
    assert torch.equal(actual[:, :cut + 1], alternate[:, :cut + 1])
    gradient, = torch.autograd.grad(actual[:, :cut + 1].square().sum(), forcing)
    assert torch.count_nonzero(gradient[:, cut:]) == 0


@pytest.mark.parametrize("mode", tuple(RESPONSE_BACKENDS))
def test_response_agrees_with_actual_learned_kernel(mode):
    torch.manual_seed(91)
    model = CausalPathOperator(NativeDKSpec(), "dk", width=4)
    context = Context(n_particles=2048, kappa=.75, a2=.1)
    initial = torch.full((2, 32), 1 / 32, dtype=torch.float64)
    forcing = torch.randn(2, 32, 32, dtype=torch.float64) / 1000
    kernel = model.kernel_values(32, 2 ** -9, context)
    expected = model._response(initial, forcing, kernel)
    actual = accelerated_response(initial, forcing, kernel, mode=mode)
    torch.testing.assert_close(actual, expected, atol=1e-16, rtol=1e-14)


def _stop_inputs():
    spec = NativeDKSpec(n_cells=8)
    rng = np.random.default_rng(891)
    initial = np.full((5, 8), .125)
    raw = rng.normal(0, .001, (5, 11, 8))
    raw -= raw.mean(-1, keepdims=True)
    raw += initial[:, None]
    # First path never exits, second exits on its first segment, third later;
    # fourth starts absorbed, fifth has an exact two-cell crossing tie.
    raw[1, 0, 0] = -.01
    raw[1, 0, 1] += .135
    raw[2, 5, 3] = -.01
    raw[2, 5, 4] += .135
    initial[3, 0] = 0
    initial[3, 1] = .25
    raw[4] = initial[4]
    raw[4, 7, :2] = -.01
    raw[4, 7, 2:4] += .135
    return spec, torch.tensor(initial), torch.tensor(raw)


@pytest.mark.parametrize("stopping", [True, False])
def test_vectorized_stopping_preserves_every_path_and_proposal(stopping):
    spec, initial, raw = _stop_inputs()
    expected = torch_stop_paths(initial, raw, spec, .01, stopping=stopping)
    actual = vectorized_stop_paths(initial, raw, spec, .01, stopping=stopping)
    for key in expected:
        if expected[key].dtype == torch.bool:
            assert torch.equal(actual[key], expected[key])
        else:
            torch.testing.assert_close(actual[key], expected[key], atol=1e-15, rtol=1e-14)
    assert actual["raw_proposals"].data_ptr() == raw.data_ptr()
    assert actual["states"].shape == (5, 12, 8)
    assert torch.equal(actual["states"][:, 0], initial)
    if stopping:
        assert torch.isinf(actual["stopped_at"][0])
        assert actual["stopped_at"][3] == 0
        torch.testing.assert_close(actual["states"][3], initial[3].expand(12, -1),
                                   atol=0, rtol=0)


def test_vectorized_stopping_gradients_and_prefix_causality():
    spec, initial, raw = _stop_inputs()
    # Include the tie and match the legacy chosen first-cell subgradient.
    raw = raw.clone().requires_grad_()
    initial = initial.clone().requires_grad_()
    expected = torch_stop_paths(initial, raw, spec, .01)
    actual = vectorized_stop_paths(initial, raw, spec, .01)
    weights = torch.arange(actual["states"].numel(), dtype=torch.float64).reshape_as(
        actual["states"]) / 10
    losses = [(packet["states"] * weights).sum()
              + packet["crossing_fraction"].sum() for packet in (expected, actual)]
    old = torch.autograd.grad(losses[0], (initial, raw), retain_graph=True)
    new = torch.autograd.grad(losses[1], (initial, raw), retain_graph=True)
    for observed, reference in zip(new, old):
        torch.testing.assert_close(observed, reference, atol=1e-11, rtol=1e-12)
    changed = raw.detach().clone()
    changed[:, 4:] -= 100
    alternate = vectorized_stop_paths(initial, changed, spec, .01)
    assert torch.equal(actual["states"][:, :5], alternate["states"][:, :5])
    gradient, = torch.autograd.grad(actual["states"][:, :5].sum(), raw)
    assert torch.count_nonzero(gradient[:, 4:]) == 0


@pytest.mark.parametrize("bad_dt", [0, -1, float("nan")])
def test_stopping_rejects_invalid_steps(bad_dt):
    spec, initial, raw = _stop_inputs()
    with pytest.raises(ValueError):
        vectorized_stop_paths(initial, raw, spec, bad_dt)


def test_response_plan_validation_and_explicit_backend_ids():
    initial = torch.full((2, 8), .125, dtype=torch.float64)
    forcing = torch.zeros(2, 5, 8, dtype=torch.float64)
    kernel = torch.ones(6, 5, dtype=torch.float64)
    for mode in RESPONSE_BACKENDS:
        plan = prepare_response(kernel, 8, mode=mode)
        assert plan.backend_id == RESPONSE_BACKENDS[mode]
        torch.testing.assert_close(plan.apply(initial, forcing), initial[:, None].expand(-1, 6, -1))
    for bad in (torch.ones(6, 5), torch.ones(6, 4, dtype=torch.float64),
                torch.full((6, 5), torch.nan, dtype=torch.float64)):
        with pytest.raises(ValueError):
            prepare_response(bad, 8)
    with pytest.raises(ValueError):
        prepare_response(kernel, 8, mode="full-horizon-fft")
    with pytest.raises(ValueError):
        accelerated_response(initial, forcing, kernel, plan=prepare_response(kernel, 8))
    with pytest.raises(ValueError):
        accelerated_response(initial, forcing[:, :3], plan=prepare_response(kernel, 8))
