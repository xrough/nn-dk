"""Full-field, causality, and actual-learning checks for the path operator."""

import math

import numpy as np
import pytest
import torch

from nn_dk.native_types import NativeDKSpec
from nn_dk.path_model import CausalPathOperator
from nn_dk.path_numerics import (
    divergence,
    ito_residual,
    path_coefficients,
    torch_euler,
    torch_stop_paths,
)
from nn_dk.types import Context


def inputs(cells=8, paths=3, steps=12, dt=1 / 32, seed=10):
    generator = torch.Generator().manual_seed(seed)
    initial = .5 + torch.rand(paths, cells, dtype=torch.float64, generator=generator)
    initial = initial / initial.sum(-1, keepdim=True)
    increments = torch.randn(paths, steps, cells, dtype=torch.float64,
                             generator=generator) * math.sqrt(dt)
    return initial, increments


@pytest.mark.parametrize("problem", ["heat", "additive", "dk"])
@pytest.mark.parametrize("cells", [8, 32])
def test_full_state_initialization_and_conservation(problem, cells):
    dt = 1 / 128
    initial, increments = inputs(cells=cells, dt=dt)
    context = Context(n_particles=2048, kappa=.75 if problem == "dk" else 0,
                      a2=.1 if problem == "dk" else 0)
    model = CausalPathOperator(NativeDKSpec(n_cells=cells), problem, width=8, depth=2)
    result = model(initial, increments, dt, context)
    assert result.shape == (3, 13, cells)
    assert result.dtype == torch.float64
    assert torch.equal(result[:, 0], initial)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.sum(-1), torch.ones_like(result[..., 0]),
                               atol=5e-15, rtol=0)


@pytest.mark.parametrize("problem", ["additive", "dk"])
def test_future_noise_has_no_computational_dependency(problem):
    dt = 1 / 32
    initial, increments = inputs(dt=dt)
    model = CausalPathOperator(NativeDKSpec(n_cells=8), problem, width=8, depth=2)
    context = Context(n_particles=2048)
    changed = increments.clone()
    changed[:, 5:] += 100
    result = model(initial, increments, dt, context)
    alternate = model(initial, changed, dt, context)
    assert torch.equal(result[:, :6], alternate[:, :6])
    assert not torch.equal(result[:, 6:], alternate[:, 6:])
    increments.requires_grad_()
    gradient = torch.autograd.grad(model(initial, increments, dt, context)[:, 5, 0].sum(),
                                   increments)[0]
    assert torch.count_nonzero(gradient[:, 5:]) == 0
    assert torch.count_nonzero(gradient[:, :5]) > 0


def test_full_thirty_one_dimensional_noise_not_four_observable_shortcut():
    spec = NativeDKSpec()
    initial = torch.full((1, 32), 1 / 32, dtype=torch.float64)
    model = CausalPathOperator(spec, "additive", width=8)
    context = Context(n_particles=2048)
    zero = torch.zeros(32, dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(
        lambda noise: model(initial, noise.reshape(1, 1, 32), 1 / 128, context)[0, 1], zero)
    assert torch.linalg.matrix_rank(jacobian, atol=1e-12, rtol=0) == 31
    torch.testing.assert_close(jacobian.sum(0), torch.zeros(32, dtype=torch.float64),
                               atol=1e-15, rtol=0)


def test_initial_kernel_is_half_generator_not_future_oracle():
    spec = NativeDKSpec()
    model = CausalPathOperator(spec, width=8)
    dt = 1 / 128
    context = Context(n_particles=2048)
    kernel = model.kernel_values(12, dt, context)
    exact_first = 1 - .5 * dt * context.diffusion * model.laplacian_rates
    torch.testing.assert_close(kernel[1], exact_first, atol=1e-15, rtol=0)
    assert torch.equal(kernel[0], torch.ones_like(kernel[0]))
    assert torch.equal(kernel[:, 0], torch.ones_like(kernel[:, 0]))
    assert kernel.shape == (13, 17)


@pytest.mark.parametrize("problem", ["heat", "additive"])
def test_green_ansatz_contains_full_euler_control(problem):
    spec = NativeDKSpec()
    context = Context(n_particles=2048)
    dt = 1 / 128
    initial, increments = inputs(cells=32, steps=16, dt=dt)
    model = CausalPathOperator(spec, problem, width=8)
    pole = 1 - dt * context.diffusion * model.laplacian_rates
    exact_kernel = pole[None] ** torch.arange(17, dtype=torch.float64)[:, None]
    _, amplitude = path_coefficients(initial[:, None], context, spec, problem)
    forcing = divergence(amplitude * increments / math.sqrt(context.n_particles))
    response = model._response(initial, forcing, exact_kernel)
    reference = torch_euler(initial, increments, context, spec, dt, problem)["states"]
    torch.testing.assert_close(response, reference, atol=3e-16, rtol=2e-14)


def test_weights_change_full_path_predictions():
    initial, increments = inputs()
    model = CausalPathOperator(NativeDKSpec(n_cells=8), "additive", width=8)
    before = model(initial, increments, 1 / 32, Context(n_particles=2048)).detach()
    with torch.no_grad():
        model.kernel_network[-1].bias[0].add_(.5)
    after = model(initial, increments, 1 / 32, Context(n_particles=2048))
    assert (before - after).abs().max() > 1e-5
    assert torch.equal(before[:, 0], after[:, 0])


@pytest.mark.parametrize("problem", ["heat", "additive"])
def test_integrated_residual_training_improves_independent_full_paths(problem):
    torch.manual_seed(1)
    spec = NativeDKSpec(n_cells=8)
    context = Context(n_particles=2048)
    dt = 1 / 32
    initial, increments = inputs(steps=16, dt=dt)
    held_initial, held_noise = inputs(steps=16, dt=dt, seed=11)
    reference = torch_euler(held_initial, held_noise, context, spec, dt, problem)["states"]
    model = CausalPathOperator(spec, problem, width=16)

    def held_error():
        return (model(held_initial, held_noise, dt, context) - reference).square().mean()

    before = held_error().detach()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=120,
                                  tolerance_grad=1e-12, tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        proposal = model(initial, increments, dt, context)[:, 1:]
        residual = ito_residual(initial, proposal, increments, context, spec, dt, problem)
        objective = residual["cumulative"].square().mean() + 10 * residual["local"].square().mean()
        objective.backward()
        return objective

    optimizer.step(closure)
    after = held_error().detach()
    assert after < before / 100
    assert after < 1e-7


def test_crossing_proposals_retained_and_intermediate_mobility_safe():
    spec = NativeDKSpec(n_cells=8)
    initial = torch.full((1, 8), 1 / 8, dtype=torch.float64)
    increments = torch.zeros(1, 6, 8, dtype=torch.float64)
    increments[:, 0, 0] = 100
    model = CausalPathOperator(spec, "dk", width=8, depth=3)
    raw = model(initial, increments, 1 / 32, Context(n_particles=2048))
    assert torch.isfinite(raw).all()
    assert (raw < spec.mass_stop).any()  # Not hidden clipping or absorption.
    raw.square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_initially_absorbed_zero_histogram_has_finite_backward():
    initial = torch.tensor([[0., 0., .25, .25, .25, .25, 0., 0.]], dtype=torch.float64)
    increments = torch.zeros(1, 6, 8, dtype=torch.float64)
    model = CausalPathOperator(NativeDKSpec(n_cells=8), "dk", width=8)
    model(initial, increments, 1 / 32, Context(n_particles=2048)).square().sum().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("kwargs", [{"problem": "moments"}, {"width": 0}, {"depth": 0},
                                    {"width": True}])
def test_invalid_constructors(kwargs):
    with pytest.raises((ValueError, TypeError)):
        CausalPathOperator(NativeDKSpec(), **kwargs)


def test_invalid_inputs_and_unsupported_resolution():
    initial, increments = inputs()
    model = CausalPathOperator(NativeDKSpec(n_cells=8), "dk", width=8)
    context = Context(n_particles=2048)
    with pytest.raises(ValueError, match="sum to one"):
        model(initial * 2, increments, 1 / 32, context)
    with pytest.raises(ValueError, match="increments"):
        model(initial, increments[..., :-1], 1 / 32, context)
    with pytest.raises(ValueError, match="finite"):
        model(initial, increments * torch.nan, 1 / 32, context)
    with pytest.raises(ValueError, match="positive-pole"):
        model(initial, increments, 1., context)
    with pytest.raises(TypeError, match="Context"):
        model(initial, increments, 1 / 32, {"diffusion": 1})


def test_public_sampler_uses_supplied_physical_noise_and_preserves_exit_proposals():
    initial, increments = inputs(steps=6)
    increments[:, 0, 0] = 100
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    model = CausalPathOperator(spec, "dk", width=8)
    with torch.no_grad():
        raw = model(initial, increments, 1/32, context)
        expected = torch_stop_paths(initial, raw[:, 1:], spec, 1/32)
    result = model.sample(context, np.arange(7)/32, 3, 123, step_size=1/32,
                          initial_masses=initial.numpy(), brownian_increments=increments.numpy())
    np.testing.assert_array_equal(result["states"], expected["states"].numpy())
    np.testing.assert_array_equal(result["raw_proposals"], raw[:, 1:].numpy())
    assert (result["raw_proposals"] < spec.mass_stop).any()
    assert result["diagnostics"]["supplied_physical_noise"]
    assert result["diagnostics"]["fixed_weights_during_simulation"]


def test_public_sampler_reproduces_initial_and_noise_streams_without_optimizer_rng():
    model = CausalPathOperator(NativeDKSpec(n_cells=8), "additive", width=8)
    context = Context(n_particles=2048)
    result = model.sample(context, [0., .0625], 4, 123, step_size=1/32)
    torch.manual_seed(901)  # A separate optimization stream cannot change inference.
    replay = model.sample(context, [0., .0625], 4, 123, step_size=1/32)
    different = model.sample(context, [0., .0625], 4, 124, step_size=1/32)
    np.testing.assert_array_equal(result["states"], replay["states"])
    assert not np.array_equal(result["states"], different["states"])
    assert len(set(result["family_ids"])) == 4
    assert result["diagnostics"]["maximum_mass_error"] < 1e-14


def test_public_sampler_rejects_checkpoint_resolution_and_context_changes():
    from nn_dk.path_config import PathConfig

    config = PathConfig(n_cells=8, cases=("dk_uniform",))
    model = CausalPathOperator(config.spec, "dk", width=8)
    model.path_config, model.case, model.dt = config, "dk_uniform", config.step_sizes[0]
    with pytest.raises(ValueError, match="timestep"):
        model.sample(config.context(model.case), [.03125], step_size=model.dt/2)
    with pytest.raises(ValueError, match="horizon"):
        model.sample(config.context(model.case), [config.horizon+model.dt])
    with pytest.raises(ValueError, match="context"):
        model.sample(Context(n_particles=512), [.03125])
    initial_only = model.sample(config.context(model.case), [0.])
    assert initial_only["states"].shape == (1, 1, 8)
    assert initial_only["raw_proposals"].shape == (1, 0, 8)
