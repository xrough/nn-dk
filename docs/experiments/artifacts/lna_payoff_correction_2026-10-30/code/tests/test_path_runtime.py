"""Prepared path equivalence, sensitivities, and explicit frozen-source loading."""

import copy
import hashlib
import importlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from nn_dk.frozen_models import current_context, frozen_package, load_frozen_model
from nn_dk.native_types import NativeDKSpec
from nn_dk.path_model import CausalPathOperator
from nn_dk.path_numerics import path_coefficients, torch_stop_paths
from nn_dk.path_runtime import PreparedPathRuntime
from nn_dk.types import Context


def _inputs(cells=8, paths=2, steps=7, dt=2 ** -7):
    generator = torch.Generator().manual_seed(7251)
    initial = .5 + torch.rand(paths, cells, generator=generator, dtype=torch.float64)
    initial = initial / initial.sum(-1, keepdim=True)
    increments = torch.randn(paths, steps, cells, generator=generator,
                             dtype=torch.float64) * math.sqrt(dt)
    return initial, increments


def _model(problem="dk", cells=8):
    torch.manual_seed(951)
    model = CausalPathOperator(NativeDKSpec(n_cells=cells), problem, width=6, depth=2)
    # Exercise learned dependencies on every physics feature, beyond the zero
    # initialized output layer of an unfitted model.
    with torch.no_grad():
        model.kernel_network[-1].weight.normal_(0, .1)
        model.kernel_network[-1].bias.normal_(0, .1)
    context = Context(n_particles=2048, diffusion=1.113,
                      kappa=.731 if problem == "dk" else 0.,
                      a2=.123 if problem == "dk" else 0.)
    return model, context


@pytest.mark.parametrize("problem", ["heat", "additive", "dk"])
@pytest.mark.parametrize("mode", ["dense-cached", "causal-block-fft"])
@pytest.mark.parametrize("learned", [True, False])
def test_prepared_path_matches_legacy_fixed_kernel(problem, mode, learned):
    model, context = _model(problem)
    initial, increments = _inputs()
    dt = 2 ** -7
    if not learned:
        model.kernel_values = lambda count, step, physics: (
            (1 - step * physics.diffusion * model.laplacian_rates)[None]
            ** torch.arange(count + 1, dtype=torch.float64)[:, None])
    expected_raw = model(initial, increments, dt, context)
    expected = torch_stop_paths(initial, expected_raw[:, 1:], model.spec, dt,
                                stopping=problem == "dk")
    runtime = PreparedPathRuntime(model, context, dt, 7 * dt, mode=mode, learned=learned)
    packet, actual_raw = runtime(initial, increments, return_raw=True)
    torch.testing.assert_close(actual_raw, expected_raw, atol=3e-16, rtol=3e-14)
    for key in expected:
        if expected[key].dtype == torch.bool:
            assert torch.equal(packet[key], expected[key])
        else:
            torch.testing.assert_close(packet[key], expected[key], atol=3e-16, rtol=3e-14)
    torch.testing.assert_close(packet["states"].sum(-1),
                               torch.ones_like(packet["states"][..., 0]), atol=2e-15, rtol=0)
    assert torch.equal(packet["states"][:, 0], initial)
    assert runtime.prepare(7) is runtime.prepare(7)


@pytest.mark.parametrize("problem", ["heat", "additive", "dk"])
@pytest.mark.parametrize("mode", ["dense-cached", "causal-block-fft"])
def test_prepared_path_initial_and_noise_gradients_match_legacy(problem, mode):
    model, context = _model(problem)
    initial, increments = (value.requires_grad_() for value in _inputs())
    dt = 2 ** -7
    expected_raw = model(initial, increments, dt, context)
    expected = torch_stop_paths(initial, expected_raw[:, 1:], model.spec, dt,
                                stopping=problem == "dk")["states"]
    runtime = PreparedPathRuntime(model, context, dt, 7 * dt, mode=mode)
    actual = runtime(initial, increments)
    weights = torch.linspace(-1, 1, actual.numel(), dtype=torch.float64).reshape_as(actual)
    old = torch.autograd.grad((expected * weights).square().sum(), (initial, increments),
                             allow_unused=True, retain_graph=True)
    new = torch.autograd.grad((actual * weights).square().sum(), (initial, increments),
                             allow_unused=True)
    for reference, observed in zip(old, new):
        if reference is None:
            assert observed is None
        else:
            torch.testing.assert_close(observed, reference, atol=5e-14, rtol=2e-12)


@pytest.mark.parametrize("mode", ["dense-cached", "causal-block-fft"])
def test_runtime_strict_future_causality_and_initial_sensitivity(mode):
    model, context = _model()
    initial, increments = _inputs(steps=13)
    runtime = PreparedPathRuntime(model, context, 2 ** -7, 13 * 2 ** -7, mode=mode)
    initial.requires_grad_()
    increments.requires_grad_()
    actual = runtime(initial, increments)
    alternate_noise = increments.detach().clone()
    alternate_noise[:, 5:] += 100
    alternate = runtime(initial, alternate_noise)
    assert torch.equal(actual[:, :6], alternate[:, :6])
    weights = torch.arange(8, dtype=torch.float64)
    loss = (actual[:, :6] * weights).square().sum()
    initial_gradient, noise_gradient = torch.autograd.grad(loss, (initial, increments))
    assert torch.count_nonzero(noise_gradient[:, 5:]) == 0
    assert torch.count_nonzero(noise_gradient[:, :5]) > 0
    assert torch.isfinite(initial_gradient).all()
    assert torch.count_nonzero(initial_gradient) > 0


@pytest.mark.parametrize("learned", [True, False])
def test_explicit_physics_kernel_matches_declared_context(learned):
    model, context = _model()
    runtime = PreparedPathRuntime(model, context, 2 ** -7, .125, learned=learned)
    physics = torch.tensor([context.diffusion, context.kappa, context.a2], dtype=torch.float64)
    torch.testing.assert_close(runtime.kernel(13, physics), runtime.kernel(13), atol=1e-15, rtol=1e-14)


@pytest.mark.parametrize("cells", [8, 32])
def test_fast_fourier_moment_coefficients_match_native_full_force(cells):
    model, context = _model(cells=cells)
    runtime = PreparedPathRuntime(model, context, 2 ** -10, .125)
    initial, _ = _inputs(cells=cells, paths=6)
    states = initial.reshape(2, 3, cells)
    expected = path_coefficients(states, context, model.spec, "dk")
    actual = runtime.coefficients(states)
    physics = torch.tensor([context.diffusion, context.kappa, context.a2], dtype=torch.float64)
    explicit = runtime.coefficients(states, physics)
    for reference, observed, specified in zip(expected, actual, explicit):
        torch.testing.assert_close(observed, reference, atol=3e-15, rtol=2e-13)
        torch.testing.assert_close(specified, reference, atol=3e-15, rtol=2e-13)
    torch.testing.assert_close(actual[0].sum(-1), torch.zeros((2, 3), dtype=torch.float64),
                               atol=2e-15, rtol=0)


@pytest.mark.parametrize("learned", [True, False])
@pytest.mark.parametrize("mode", ["dense-cached", "causal-block-fft"])
def test_runtime_physics_and_initial_directional_derivatives(learned, mode):
    model, context = _model()
    runtime = PreparedPathRuntime(model, context, 2 ** -7, 7 * 2 ** -7,
                                  mode=mode, learned=learned)
    initial, increments = _inputs()
    initial.requires_grad_()
    physics = torch.tensor([context.diffusion, context.kappa, context.a2], dtype=torch.float64,
                            requires_grad=True)
    weights = torch.arange(8, dtype=torch.float64).sin()

    def objective(origin, parameters):
        path = runtime(origin, increments, physics=parameters)
        return (path[:, -1] * weights).sum()

    gradient_initial, gradient_physics = torch.autograd.grad(objective(initial, physics),
                                                          (initial, physics))
    epsilon = 1e-5
    for index in range(3):
        direction = torch.zeros_like(physics)
        direction[index] = 1
        finite = (objective(initial.detach(), physics.detach() + epsilon * direction)
                  - objective(initial.detach(), physics.detach() - epsilon * direction)) / (2 * epsilon)
        torch.testing.assert_close(gradient_physics[index], finite, atol=2e-10, rtol=2e-6)
    direction = torch.arange(8, dtype=torch.float64).cos().expand_as(initial)
    direction = direction - direction.mean(-1, keepdim=True)
    finite = (objective(initial.detach() + epsilon * direction, physics.detach())
              - objective(initial.detach() - epsilon * direction, physics.detach())) / (2 * epsilon)
    torch.testing.assert_close((gradient_initial * direction).sum(), finite, atol=2e-9, rtol=2e-7)


def test_runtime_stopping_keeps_crossing_and_initially_absorbed_paths():
    model, context = _model()
    initial, increments = _inputs()
    initial[1] = torch.tensor([0., 0., .25, .25, .25, .25, 0., 0.], dtype=torch.float64)
    increments[0, 0, 0] = 100
    expected_raw = model(initial, increments, 2 ** -7, context)
    expected = torch_stop_paths(initial, expected_raw[:, 1:], model.spec, 2 ** -7)
    runtime = PreparedPathRuntime(model, context, 2 ** -7, 7 * 2 ** -7)
    actual, raw = runtime(initial, increments, return_raw=True)
    torch.testing.assert_close(raw, expected_raw, atol=1e-15, rtol=1e-13)
    torch.testing.assert_close(actual["states"], expected["states"], atol=1e-15, rtol=1e-13)
    torch.testing.assert_close(actual["stopped_at"], expected["stopped_at"], atol=1e-15, rtol=1e-13)
    assert actual["stopped_at"][1] == 0
    assert (raw[0] < model.spec.mass_stop).any()


def test_runtime_stopping_has_finite_initial_and_noise_gradients():
    model, context = _model()
    initial, increments = _inputs()
    initial[1] = torch.tensor([0., 0., .25, .25, .25, .25, 0., 0.], dtype=torch.float64)
    increments[0, 0, 0] = 100
    initial.requires_grad_()
    increments.requires_grad_()
    runtime = PreparedPathRuntime(model, context, 2 ** -7, 7 * 2 ** -7)
    actual = runtime(initial, increments)
    gradients = torch.autograd.grad(actual.square().sum(), (initial, increments))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.count_nonzero(gradients[1][1]) == 0


@pytest.fixture(scope="module")
def historical_checkpoint(tmp_path_factory):
    """Construct a tiny valid checkpoint using its committed original trainer."""
    alias, revision, source, root, _ = frozen_package("26ac405")
    configuration = importlib.import_module(f"{alias}.path_config")
    training = importlib.import_module(f"{alias}.path_training")
    config = configuration.PathConfig(
        name="frozen-loader-test", n_cells=8, cases=("dk_interacting",), width=4, depth=2,
        horizon=1 / 64, step_sizes=(1 / 128,), reference_step=1 / 256,
        query_times=(1 / 64,), train_paths=2, validation_paths=2, test_paths=2,
        adam_steps=1, lbfgs_steps=0, batch_size=2, n_bootstrap=2)
    destination = tmp_path_factory.mktemp("historical-path") / "checkpoint.pt"
    initial, noise = _inputs(steps=2)

    def bank(role):
        return {"initial": initial.numpy(), "brownian": noise.numpy(),
                "family_ids": np.asarray([f"{role}-0", f"{role}-1"])}

    training.fit_path(config, "dk_interacting", 1 / 128, 0, destination,
                      bank("train"), bank("validation"))
    return destination, revision, source, root


def test_historical_checkpoint_strict_load_and_runtime_reproduction(historical_checkpoint):
    path, revision, source, _ = historical_checkpoint
    original, provenance = load_frozen_model(path, revision)
    assert provenance["historical_commit"] == revision
    assert provenance["historical_source_hash"] == source
    assert provenance["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["original_target_id"] == original.target_id
    context = current_context(original.path_config.context(original.case))
    assert isinstance(context, Context)
    initial, noise = _inputs(steps=2)
    reference_raw = original(initial, noise, original.dt,
                             original.path_config.context(original.case))
    reference = torch_stop_paths(initial, reference_raw[:, 1:], NativeDKSpec(n_cells=8),
                                  original.dt)["states"]
    runtime = PreparedPathRuntime.from_checkpoint(path, revision)
    torch.testing.assert_close(runtime(initial, noise), reference, atol=2e-16, rtol=2e-14)
    assert runtime.provenance["checkpoint_sha256"] == provenance["checkpoint_sha256"]


def test_historical_source_cache_uses_canonical_revision(historical_checkpoint):
    _, revision, _, root = historical_checkpoint
    assert frozen_package("26ac405")[3] == root
    assert frozen_package(revision)[3] == root


def test_historical_loader_rejects_wrong_commit(historical_checkpoint):
    path, _, _, _ = historical_checkpoint
    with pytest.raises(ValueError, match="source hash"):
        load_frozen_model(path, "bd256de")


def test_historical_loader_rejects_changed_weights(historical_checkpoint, tmp_path):
    path, revision, _, _ = historical_checkpoint
    saved = torch.load(path, weights_only=True)
    changed = copy.deepcopy(saved)
    key = next(key for key in changed["best_state"] if "weight" in key)
    changed["best_state"][key].reshape(-1)[0] += .1
    destination = tmp_path / "changed.pt"
    torch.save(changed, destination)
    with pytest.raises(ValueError, match="checksum"):
        load_frozen_model(destination, revision)


def test_historical_loader_rechecks_materialized_source(historical_checkpoint):
    path, revision, _, root = historical_checkpoint
    target = Path(root) / "path_model.py"
    original = target.read_text()
    try:
        target.write_text(original + "\n# accidental mutation\n")
        with pytest.raises(ValueError, match="snapshot changed"):
            load_frozen_model(path, revision)
    finally:
        target.write_text(original)


@pytest.mark.parametrize("physics", [[0, .75, .1], [1, float("nan"), 0], [1, 0]])
def test_runtime_rejects_invalid_physics(physics):
    model, context = _model()
    runtime = PreparedPathRuntime(model, context, 2 ** -7, .125)
    initial, increments = _inputs()
    with pytest.raises(ValueError):
        runtime(initial, increments, physics=physics)


@pytest.mark.parametrize("problem", ["heat", "additive"])
def test_controls_reject_explicit_interaction(problem):
    model, context = _model(problem)
    runtime = PreparedPathRuntime(model, context, 2 ** -7, .125)
    initial, increments = _inputs()
    with pytest.raises(ValueError, match="zero interaction"):
        runtime(initial, increments, physics=[1., .75, .1])


@pytest.mark.parametrize("kwargs", [{"dt": 0}, {"horizon": 0}, {"depth": 0}])
def test_runtime_rejects_invalid_constructor_settings(kwargs):
    model, context = _model()
    arguments = {"dt": 2 ** -7, "horizon": .125, **kwargs}
    with pytest.raises(ValueError):
        PreparedPathRuntime(model, context, **arguments)
