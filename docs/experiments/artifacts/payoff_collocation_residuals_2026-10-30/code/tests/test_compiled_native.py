"""Optional serial compiled classical control matches native Euler and its JVP."""

import importlib.util
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from nn_dk.native_types import NativeDKSpec
from nn_dk.types import Context

_spec = importlib.util.spec_from_file_location(
    "compiled_native_test", Path(__file__).parents[1] / "analysis/compiled_native.py")
control = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(control)
_probe_spec = importlib.util.spec_from_file_location(
    "compiled_native_reference", Path(__file__).parents[1] / "analysis/probe_path_workloads.py")
probe = importlib.util.module_from_spec(_probe_spec)
_probe_spec.loader.exec_module(probe)


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    if not control.compiler_available():
        pytest.skip("Optional C++ compiler unavailable")
    return control.compile_native(tmp_path_factory.mktemp("compiled-native"))


@pytest.mark.parametrize("interaction", [(0., 0.), (.75, .1), (-.43, -.17)])
@pytest.mark.parametrize("crossing", [False, True])
def test_forward_and_jvp_match_numpy_for_all_paths(compiled, interaction, crossing):
    spec = NativeDKSpec(n_cells=16)
    context = Context(kappa=interaction[0], a2=interaction[1], n_particles=2048)
    initial = np.random.default_rng(82).uniform(.8, 1.2, (4, 16))
    initial /= initial.sum(-1, keepdims=True)
    noise = np.random.default_rng(67).normal(0, np.sqrt(.003), (4, 24, 16))
    if crossing:
        initial[0, :2] = 0., initial[0, :2].sum()
        noise[1, 5, 4] = 20
    direction = np.cos(3 * (np.arange(16) + .5) * spec.h) / 16
    direction -= direction.mean()
    parameters = np.array([.15, -.2, .3])
    for tangent in (False, True):
        kwargs = {"initial_tangent": direction, "physics_tangent": parameters} if tangent else {}
        reference = probe.numpy_euler(initial, noise, context, spec, .003, **kwargs)
        actual = compiled(initial, noise, context, spec, .003, **kwargs)
        for key in reference:
            np.testing.assert_allclose(actual[key], reference[key], atol=2e-14, rtol=2e-12)
        np.testing.assert_array_equal(actual["states"][:, 0], initial)
        np.testing.assert_allclose(actual["states"].sum(-1), 1., atol=3e-15)
        if tangent:
            np.testing.assert_allclose(actual["tangent"].sum(-1), 0., atol=3e-14)
        if crossing:
            assert actual["stopped_at"][0] == 0
            assert np.isfinite(actual["stopped_at"][1])


@pytest.mark.parametrize("parameter", ["initial", "diffusion", "kappa", "a2"])
@pytest.mark.parametrize("crossing", [False, True])
def test_compiled_initial_and_physics_jvp_matches_finite_difference(compiled, parameter, crossing):
    spec = NativeDKSpec(n_cells=8)
    context = Context(kappa=.731, a2=.123, n_particles=64)
    initial = np.full((2, 8), .125)
    noise = np.random.default_rng(51).normal(0, .1, (2, 12, 8))
    if crossing:
        noise[0, 3, 2] = -4
    direction = np.cos(3 * (np.arange(8) + .5) * spec.h) / 8
    direction -= direction.mean()
    physics_direction = np.zeros(3)
    if parameter != "initial":
        direction[:] = 0
        physics_direction[("diffusion", "kappa", "a2").index(parameter)] = 1
    actual = compiled(initial, noise, context, spec, .01, initial_tangent=direction,
                      physics_tangent=physics_direction)
    epsilon = 1e-6

    def perturbed(sign):
        base = np.array([context.diffusion, context.kappa, context.a2]) + sign * epsilon * physics_direction
        changed = replace(context, diffusion=base[0], kappa=base[1], a2=base[2])
        return compiled(initial + sign * epsilon * direction, noise, changed, spec, .01)

    plus, minus = perturbed(1), perturbed(-1)
    np.testing.assert_allclose(actual["tangent"], (plus["states"] - minus["states"]) / (2 * epsilon),
                               atol=3e-10, rtol=2e-6)
    if crossing:
        assert actual["stopped_at_tangent"][0] == pytest.approx(
            (plus["stopped_at"][0] - minus["stopped_at"][0]) / (2 * epsilon), abs=3e-11)


def test_compiler_fingerprint_cache_and_invalid_inputs(compiled):
    metadata = compiled.metadata
    assert metadata["cpp_sha256"]
    assert metadata["library_sha256"]
    assert metadata["compile_seconds"] > 0
    assert not metadata["fast_math"]
    assert metadata["cpu_threads"] == 1
    repeated = control.compile_native(Path(metadata["library_path"]).parent)
    assert repeated.metadata["cache_hit"]
    assert repeated.metadata["compile_seconds_this_process"] == 0
    assert repeated.metadata["fingerprint"] == metadata["fingerprint"]
    initial, noise = np.full((1, 8), .125), np.zeros((1, 2, 8))
    spec, context = NativeDKSpec(n_cells=8), Context()
    with pytest.raises(ValueError, match="stability"):
        compiled(initial, noise, context, spec, 1.)
    with pytest.raises(ValueError, match="conservative"):
        compiled(initial, noise, context, spec, .01, initial_tangent=np.ones(8))
    noise[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite"):
        compiled(initial, noise, context, spec, .01)


def test_unavailable_compiler_is_optional(tmp_path):
    assert not control.compiler_available("nn_dk_no_such_compiler")
    with pytest.raises(RuntimeError, match="Optional"):
        control.compile_native(tmp_path, compiler="nn_dk_no_such_compiler")


def test_compiler_driver_symlink_preserves_cpp_linking(tmp_path):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Optional C++ compiler unavailable")
    driver = tmp_path / "clang++"
    driver.symlink_to(Path(compiler).resolve())
    compiled = control.compile_native(tmp_path / "cache", compiler=str(driver))
    assert compiled.library.nn_dk_control_abi() == 2
    assert compiled.metadata["compiler"] == str(driver.absolute())
    assert compiled.metadata["command"][0] == str(driver.absolute())


@pytest.mark.parametrize("interaction", [(0., 0.), (.731, .123), (-.4, -.2)])
@pytest.mark.parametrize("crossing", [False, True])
def test_compiled_adjoint_matches_numpy_and_torch_for_every_path(compiled, interaction, crossing):
    spec = NativeDKSpec(n_cells=16)
    context = Context(kappa=interaction[0], a2=interaction[1], n_particles=64)
    initial = np.random.default_rng(381).uniform(.8, 1.2, (3, 16))
    initial /= initial.sum(-1, keepdims=True)
    noise = np.random.default_rng(397).normal(0, np.sqrt(.003), (3, 12, 16))
    if crossing:
        initial[0, :2] = 0., initial[0, :2].sum()
        noise[1, 3, 4] = -4
    actual = compiled.adjoint(initial, noise, context, spec, .003)
    reference = probe.numpy_adjoint(initial, noise, context, spec, .003)
    np.testing.assert_allclose(actual["initial_gradient"], reference["initial_gradient"], atol=4e-13, rtol=4e-12)
    np.testing.assert_allclose(actual["physics_gradient"], reference["physics_gradient"], atol=4e-14, rtol=4e-12)
    np.testing.assert_array_equal(actual["states"][:, 0], initial)
    origin = torch.tensor(initial, requires_grad=True)
    physics = torch.tensor([context.diffusion, context.kappa, context.a2], dtype=torch.float64,
                            requires_grad=True)
    states = probe.torch_fourier_euler(origin, noise, context, spec, .003, physics=physics)["states"]
    gradient, physical = torch.autograd.grad(probe.workload_values(states, spec, .003).sum(), (origin, physics))
    np.testing.assert_allclose(actual["initial_gradient"], gradient.detach(), atol=5e-13, rtol=5e-12)
    np.testing.assert_allclose(actual["physics_gradient"].sum(0), physical.detach(), atol=5e-14, rtol=5e-12)
    if crossing:
        np.testing.assert_allclose(actual["initial_gradient"][0],
            probe.workload_state_gradient(actual["states"], spec, .003)[0].sum(0), atol=3e-15)
        np.testing.assert_array_equal(actual["physics_gradient"][0], 0)


@pytest.mark.parametrize("parameter", ["initial", "diffusion", "kappa", "a2"])
@pytest.mark.parametrize("crossing", [False, True])
def test_compiled_adjoint_directions_match_finite_differences(compiled, parameter, crossing):
    spec = NativeDKSpec(n_cells=8)
    context = Context(diffusion=1.113, kappa=.731, a2=.123, n_particles=64)
    initial = np.full((2, 8), .125)
    noise = np.random.default_rng(51).normal(0, .1, (2, 12, 8))
    if crossing:
        noise[0, 3, 2] = -4
    direction = np.cos(3 * (np.arange(8) + .5) * spec.h) / 8
    direction -= direction.mean()
    physics_direction = np.zeros(3)
    if parameter != "initial":
        direction[:] = 0
        physics_direction[("diffusion", "kappa", "a2").index(parameter)] = 1
    result = compiled.adjoint(initial, noise, context, spec, .01)
    derivative = ((result["initial_gradient"] * direction).sum()
                  + result["physics_gradient"].sum(0) @ physics_direction)
    epsilon = 1e-6

    def value(sign):
        parameters = np.array([context.diffusion, context.kappa, context.a2]) + sign * epsilon * physics_direction
        changed = replace(context, diffusion=parameters[0], kappa=parameters[1], a2=parameters[2])
        path = compiled(initial + sign * epsilon * direction, noise, changed, spec, .01)["states"]
        return probe.workload_values(path, spec, .01).sum()

    assert derivative == pytest.approx((value(1) - value(-1)) / (2 * epsilon), abs=5e-10, rel=3e-6)


def test_compiled_adjoint_arbitrary_cotangent_and_tied_first_cell(compiled):
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=64)
    initial = np.full((2, 8), .125)
    noise = np.zeros((2, 5, 8))
    noise[0, 0, ::2] = 3  # Four identical first crossings; select the first cell.
    path = compiled(initial, noise, context, spec, .01)
    cotangent = np.random.default_rng(109).normal(size=path["states"].shape)
    actual = compiled.adjoint(initial, noise, context, spec, .01, path_gradient=cotangent)
    reference = probe.numpy_adjoint(initial, noise, context, spec, .01, path_gradient=cotangent)
    np.testing.assert_allclose(actual["initial_gradient"], reference["initial_gradient"], atol=5e-14, rtol=3e-12)
    np.testing.assert_allclose(actual["physics_gradient"], reference["physics_gradient"], atol=5e-14, rtol=3e-12)
    origin = torch.tensor(initial, requires_grad=True)
    physics = torch.tensor([context.diffusion, context.kappa, context.a2], dtype=torch.float64,
                            requires_grad=True)
    torch_path = probe.torch_fourier_euler(origin, noise, context, spec, .01, physics=physics)["states"]
    gradient, physical = torch.autograd.grad((torch_path * torch.tensor(cotangent)).sum(), (origin, physics))
    np.testing.assert_allclose(actual["initial_gradient"], gradient.detach(), atol=6e-14, rtol=4e-12)
    np.testing.assert_allclose(actual["physics_gradient"].sum(0), physical.detach(), atol=6e-14, rtol=4e-12)
    assert np.isfinite(actual["stopped_at"][0])
    with pytest.raises(ValueError, match="cotangent"):
        compiled.adjoint(initial, noise, context, spec, .01, path_gradient=np.zeros((2, 8)))
