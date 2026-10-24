"""Structure, stopping and isolation checks for the BSDE value ablation."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from nn_dk.bsde_config import bsde_preset
from nn_dk.bsde_data import ou_exact_moments
from nn_dk.bsde_model import BSDEMomentModel
from nn_dk.bsde_structured import (
    STRUCTURED_DK_BSDE_MODEL_KIND,
    STRUCTURED_OU_BSDE_MODEL_KIND,
    StructuredBSDEMomentModel,
)
from nn_dk.native import initialize_masses


def _config():
    return replace(bsde_preset("bsde-smoke"), width=8)


def _inverse_softplus(value):
    return value + torch.log(-torch.expm1(-value))


def test_ou_value_has_four_positive_learned_parameters_and_no_oracle_initialization():
    model = StructuredBSDEMomentModel(_config(), "ou")
    assert model.architecture == "structured"
    assert model.kind == STRUCTURED_OU_BSDE_MODEL_KIND
    assert sum(p.numel() for p in model.value_network.parameters()) == 4
    torch.testing.assert_close(model.value_network.rates, torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(model.value_network.scales, torch.ones(2, dtype=torch.float64))
    initial = torch.tensor([[2., 3., -4., 5.], [8., -3., 6., 2.]], dtype=torch.float64)
    horizons = torch.full((2,), .25, dtype=torch.float64)
    mean, covariance = model.moment_values(initial, horizons)
    torch.testing.assert_close(mean / initial, torch.full_like(initial, np.exp(-.25)))
    torch.testing.assert_close(covariance[0], covariance[1])
    assert torch.count_nonzero(covariance[0] - torch.diag(torch.diagonal(covariance[0]))) == 0
    assert covariance[0, 0, 0] == covariance[0, 1, 1]
    assert covariance[0, 2, 2] == covariance[0, 3, 3]


def test_four_parameter_ou_class_contains_the_analytic_euler_family():
    config = _config()
    model = StructuredBSDEMomentModel(config, "ou")
    rates = torch.tensor(config.mode_rates[::2], dtype=torch.float64)
    transition = 1 - config.dt * rates
    learned_rates = -torch.log(transition) / config.dt
    learned_scales = 2 * rates * config.dt / (1 - transition.square())
    # Oracle coefficients are installed exclusively in this class-containment
    # test.  Training and default predictions never read them.
    with torch.no_grad():
        model.value_network.raw_rates.copy_(_inverse_softplus(learned_rates))
        model.value_network.raw_scales.copy_(_inverse_softplus(learned_scales))
    initial = np.random.default_rng(17).normal(size=(7, 4))
    prediction = model.predict_conditional(initial, (0., *config.query_times))
    mean, covariance = ou_exact_moments(initial, prediction.horizons, config, dt=config.dt)
    np.testing.assert_allclose(prediction.mean, mean, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(prediction.scaled_covariance, covariance, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("problem", ("dk", "ou"))
def test_integrand_initialization_and_parameters_are_identical_across_value_architectures(problem):
    config = _config()
    torch.manual_seed(42)
    baseline = BSDEMomentModel(config, problem)
    torch.manual_seed(42)
    structured = StructuredBSDEMomentModel(config, problem)
    for key, value in baseline.integrand_network.state_dict().items():
        torch.testing.assert_close(structured.integrand_network.state_dict()[key], value,
                                   rtol=0, atol=0)
    assert structured.kind != baseline.kind


def test_native_head_starts_at_learned_backbone_and_retains_full_state_residual():
    config = _config()
    model = StructuredBSDEMomentModel(config, "dk")
    assert model.kind == STRUCTURED_DK_BSDE_MODEL_KIND
    initial = torch.tensor(initialize_masses(config.context("uniform"), config.spec, 4, 19))
    network = model.value_network.residual_network
    assert network[0].in_features == config.n_cells
    assert torch.count_nonzero(network[-1].weight) == 0
    assert torch.count_nonzero(network[-1].bias) == 0
    time = torch.full((len(initial),), .25, dtype=torch.float64)
    mean, covariance = model.moment_values(initial, time)
    y0 = model.scaled_observables(initial)
    torch.testing.assert_close(mean, np.exp(-.25) * y0)
    expected = torch.diag(torch.full((4,), -np.expm1(-.5), dtype=torch.float64))
    torch.testing.assert_close(covariance, expected.expand(4, -1, -1))

    # High grid mode 7 is invisible to the four queried initial observations.
    # Its effect can still reach the residual value/covariance network.
    cells = torch.arange(config.n_cells, dtype=torch.float64)
    direction = torch.cos(2 * np.pi * 7 * (cells + .5) / config.n_cells)
    direction = direction / direction.norm()
    raw_direction = model.raw_features(direction[None] + 1 / config.n_cells)[0]
    raw_direction = raw_direction / raw_direction.norm()
    with torch.no_grad():
        for layer in network:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        network[0].weight[0, :-1] = raw_direction
        network[2].weight[0, 0] = 1.
        network[4].weight[0, 0] = 1.
        network[6].weight[0, 0] = 1.
        network[6].weight[4, 0] = 1.
    origin = torch.full((1, config.n_cells), 1 / config.n_cells,
                        dtype=torch.float64, requires_grad=True)
    mean, covariance = model.moment_values(origin, torch.tensor([.25], dtype=torch.float64))
    mean_gradient = torch.autograd.grad(mean[0, 0], origin, retain_graph=True)[0]
    variance_gradient = torch.autograd.grad(covariance[0, 0, 0], origin)[0]
    assert abs((mean_gradient * direction).sum().item()) > 1e-5
    assert abs((variance_gradient * direction).sum().item()) > 1e-5
    assert np.linalg.eigvalsh(covariance.detach().numpy()).min() > 0


@pytest.mark.parametrize("problem", ("dk", "ou"))
def test_zero_horizon_is_exact_with_finite_backward_pass(problem):
    config = _config()
    model = StructuredBSDEMomentModel(config, problem)
    if problem == "dk":
        initial = torch.full((2, config.n_cells), 1 / config.n_cells, dtype=torch.float64)
    else:
        initial = torch.tensor([[1., -2., 3., 4.], [-4., 2., 1., 0.]], dtype=torch.float64)
    initial.requires_grad_()
    horizons = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    mean, covariance = model.moment_values(initial, horizons)
    torch.testing.assert_close(mean, model.scaled_observables(initial), atol=0, rtol=0)
    assert torch.count_nonzero(covariance) == 0
    (mean.sum() + covariance.sum()).backward()
    assert torch.isfinite(initial.grad).all()
    assert torch.isfinite(horizons.grad).all()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_initial_boundary_freezes_mean_and_covariance_at_all_horizons():
    config = _config()
    model = StructuredBSDEMomentModel(config, "dk")
    initial = np.full((1, config.n_cells), 1 / config.n_cells)
    initial[0, 0] = config.spec.mass_stop
    initial[0, 1] += 1 / config.n_cells - config.spec.mass_stop
    prediction = model.predict_conditional(initial, (0., *config.query_times))
    expected = initial @ model.observation_weights.numpy().T
    np.testing.assert_allclose(prediction.mean, np.broadcast_to(expected[:, None], prediction.mean.shape),
                               atol=1e-15, rtol=0)
    assert np.count_nonzero(prediction.scaled_covariance) == 0


@pytest.mark.parametrize("problem", ("dk", "ou"))
def test_value_inference_never_calls_oracle_center_or_integrand(problem, monkeypatch):
    import nn_dk.bsde_data as data_module

    config = _config()
    model = StructuredBSDEMomentModel(config, problem)

    def forbidden(*args, **kwargs):
        raise AssertionError("Value inference evaluated a training-only or oracle function")

    monkeypatch.setattr(model, "deterministic_center", forbidden)
    monkeypatch.setattr(model.integrand_network, "forward", forbidden)
    initial = (np.full((2, config.n_cells), 1 / config.n_cells) if problem == "dk"
               else np.ones((2, 4)))
    first = model.predict_conditional(initial, config.query_times)
    monkeypatch.setattr(data_module, "ou_rates", forbidden)
    second = model.predict_conditional(initial, config.query_times)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.scaled_covariance, second.scaled_covariance)
