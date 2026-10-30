"""Exact generator chain/product rules checked independently by dense Hessians."""

import pytest
import torch
from torch import nn

from nn_dk.generator_jets import (
    GeneratorJet,
    jet_gram,
    jet_sequential,
    jet_softplus,
    jet_tanh,
)


def _dense_jet(function, point, time_direction, drift, faces):
    jacobian = torch.func.jacrev(function)(point)
    hessian = torch.func.jacrev(torch.func.jacrev(function))(point)
    return GeneratorJet(function(point), jacobian @ time_direction,
                        jacobian @ drift + .5 * torch.einsum("...ij,if,jf->...", hessian,
                                                              faces, faces),
                        jacobian @ faces)


def _compare(actual, expected, tolerance=2e-12):
    for name in ("value", "time_derivative", "generator", "face_derivatives"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name),
                                   atol=tolerance, rtol=tolerance)


def test_exact_linear_silu_generator_against_dense_hessian():
    torch.manual_seed(71)
    network = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 6), nn.SiLU(),
                            nn.Linear(6, 4)).double()
    point = torch.randn(5, dtype=torch.float64)
    drift = torch.randn(5, dtype=torch.float64)
    faces = torch.randn(5, 8, dtype=torch.float64)
    time_direction = torch.randn(5, dtype=torch.float64)
    initial = GeneratorJet(point, time_direction, drift, faces)
    _compare(jet_sequential(initial, network),
             _dense_jet(network, point, time_direction, drift, faces))


@pytest.mark.parametrize("activation", ["tanh", "softplus"])
def test_smooth_activation_generator_against_dense_hessian(activation):
    torch.manual_seed(18)
    point = torch.tensor([-40., -3., 0., 2., 40.], dtype=torch.float64)
    drift = torch.randn(5, dtype=torch.float64)
    faces = torch.randn(5, 8, dtype=torch.float64)
    time_direction = torch.randn(5, dtype=torch.float64)
    initial = GeneratorJet(point, time_direction, drift, faces)
    if activation == "tanh":
        actual, function = jet_tanh(initial), torch.tanh
    else:
        actual = jet_softplus(initial)

        def function(x):
            return torch.logaddexp(torch.zeros_like(x), x)
    _compare(actual, _dense_jet(function, point, time_direction, drift, faces))


def test_gram_product_rule_includes_all_cross_variations():
    torch.manual_seed(74)
    network = nn.Sequential(nn.Linear(5, 8), nn.SiLU(), nn.Linear(8, 6)).double()
    point = torch.randn(5, dtype=torch.float64)
    drift = torch.randn(5, dtype=torch.float64)
    faces = torch.randn(5, 8, dtype=torch.float64)
    time_direction = torch.randn(5, dtype=torch.float64)
    raw = jet_sequential(GeneratorJet(point, time_direction, drift, faces), network)
    factor = GeneratorJet(raw.value.reshape(2, 3), raw.time_derivative.reshape(2, 3),
                          raw.generator.reshape(2, 3), raw.face_derivatives.reshape(2, 3, 8))

    def function(x):
        value = network(x).reshape(2, 3)
        return value @ value.T

    _compare(jet_gram(factor), _dense_jet(function, point, time_direction, drift, faces))


def test_exact_trace_retains_every_face_and_parameter_gradients():
    torch.manual_seed(27)
    network = nn.Sequential(nn.Linear(3, 5), nn.SiLU(), nn.Linear(5, 2)).double()
    value = torch.randn(4, 3, dtype=torch.float64)
    faces = torch.randn(4, 3, 32, dtype=torch.float64)
    inputs = GeneratorJet(value, torch.zeros_like(value), torch.ones_like(value), faces)
    result = jet_sequential(inputs, network)
    loss = (result.generator.square().mean() + result.face_derivatives.square().mean()
            + .01 * result.value.square().mean())
    loss.backward()
    assert result.face_derivatives.shape == (4, 2, 32)
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in network.parameters())
