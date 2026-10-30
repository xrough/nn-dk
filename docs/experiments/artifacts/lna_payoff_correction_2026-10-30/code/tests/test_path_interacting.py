"""Interacting physics, native-state fidelity and causal path-operator checks."""

from dataclasses import replace
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from nn_dk.native import initialize_masses, native_coefficients
from nn_dk.path_config import PathConfig, path_preset
from nn_dk.path_model import CausalPathOperator
from nn_dk.path_numerics import path_coefficients, torch_euler


@pytest.mark.parametrize("preset", ["path-interacting-smoke", "path-interacting-pilot"])
def test_interacting_presets_match_committed_configs(preset):
    config = path_preset(preset)
    path = Path(__file__).resolve().parents[1] / "configs" / (preset.replace("-", "_") + ".toml")
    assert PathConfig.load(path) == config
    assert config.cases == ("heat", "additive", "dk_interacting_uniform", "dk_interacting")
    assert config.step_sizes == (2**-8,)
    assert config.seed != path_preset("path-smoke").seed
    assert config.context("dk_interacting_uniform").cos_coeffs == (0., 0., 0.)
    assert config.context("dk_interacting").cos_coeffs == (.2, 0., 0.)
    assert config.expected_models == (4 if preset.endswith("smoke") else 12)


def test_interacting_parameters_change_only_declared_physics_and_target():
    config = path_preset("path-interacting-smoke")
    changed = replace(config, interaction_kappa=-.4, interaction_a2=-.08)
    assert changed.config_hash != config.config_hash
    for case in ("heat", "additive", "dk_uniform", "dk_nonuniform"):
        assert changed.context(case) == config.context(case)
    for case in ("dk_interacting_uniform", "dk_interacting"):
        assert changed.context(case).kappa == -.4
        assert changed.context(case).a2 == -.08
        assert changed.target_id(case, 2**-8) != config.target_id(case, 2**-8)
    assert changed.target_id("heat", 2**-8) == config.target_id("heat", 2**-8)
    assert config.target_id("dk_interacting_uniform", 2**-8) != config.target_id("dk_interacting", 2**-8)


@pytest.mark.parametrize("parameter", ["interaction_kappa", "interaction_a2"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.75"])
def test_nonfinite_or_nonreal_interaction_parameters_rejected(parameter, value):
    with pytest.raises(ValueError, match="Finite real interaction"):
        PathConfig(**{parameter: value})


def test_centered_interacting_transport_obeys_native_peclet_limit():
    with pytest.raises(ValueError, match="Peclet"):
        PathConfig(cases=("dk_interacting_uniform",), interaction_kappa=11.)
    # An unused parameter cannot introduce interaction into a heat control.
    PathConfig(cases=("heat", "additive"), interaction_kappa=11.)


@pytest.mark.parametrize("kappa,a2", [(.75, .1), (-.4, -.08), (0., .2)])
def test_native_and_torch_interaction_match_independent_fourier_moments(kappa, a2):
    config = replace(path_preset("path-interacting-smoke"),
                     interaction_kappa=kappa, interaction_a2=a2)
    spec, context = config.spec, config.context("dk_interacting")
    centers = (np.arange(spec.n_cells) + .5) * spec.h
    faces = (np.arange(spec.n_cells) + 1) * spec.h
    masses = (1 + .2*np.cos(centers) + .15*np.sin(2*centers)) / spec.n_cells
    # Compute force from four Fourier moments, independently of the dense
    # cell-to-face quadrature matrix used by the native implementation.
    velocity = np.zeros(spec.n_cells)
    for mode, coefficient in ((1, -kappa), (2, 2*a2)):
        cosine = masses @ np.cos(mode*centers)
        sine = masses @ np.sin(mode*centers)
        velocity += coefficient*np.sinc(mode/spec.n_cells) * (
            np.sin(mode*faces)*cosine - np.cos(mode*faces)*sine)
    following = np.roll(masses, -1)
    face_density = (masses+following)/(2*spec.h)
    expected_flux = (face_density*velocity
                     - context.diffusion/spec.h**2*(following-masses))
    expected_drift = np.roll(expected_flux, 1)-expected_flux
    native = native_coefficients(masses, context, spec)
    actual_drift, actual_amplitudes = path_coefficients(masses, context, spec, "dk")
    np.testing.assert_allclose(native.drift, expected_drift, atol=2e-16, rtol=2e-13)
    np.testing.assert_allclose(actual_drift.numpy(), expected_drift, atol=2e-16, rtol=2e-13)
    np.testing.assert_allclose(actual_amplitudes.numpy(), native.face_amplitudes,
                               atol=2e-16, rtol=2e-14)
    no_interaction, _ = path_coefficients(masses, replace(context, kappa=0., a2=0.), spec, "dk")
    assert np.max(np.abs(actual_drift.numpy()-no_interaction.numpy())) > 1e-5
    assert abs(expected_drift.sum()) < 1e-16


@pytest.mark.parametrize("case", ["dk_interacting_uniform", "dk_interacting"])
def test_interacting_full_path_is_conservative_and_exactly_causal(case):
    config = path_preset("path-interacting-smoke")
    spec, context, dt = config.spec, config.context(case), config.step_sizes[0]
    initial = torch.from_numpy(initialize_masses(context, spec, 3, 441))
    generator = torch.Generator().manual_seed(442)
    increments = torch.randn(3, 16, spec.n_cells, dtype=torch.float64,
                             generator=generator)*np.sqrt(dt)
    model = CausalPathOperator(spec, "dk", width=8, depth=3)
    changed = increments.clone()
    changed[:, 7:] += .5
    prediction = model(initial, increments, dt, context)
    alternate = model(initial, changed, dt, context)
    assert prediction.shape == (3, 17, spec.n_cells)
    assert torch.equal(prediction[:, :8], alternate[:, :8])
    assert not torch.equal(prediction[:, 8:], alternate[:, 8:])
    no_interaction = model(initial, increments, dt, replace(context, kappa=0., a2=0.))
    assert float((prediction-no_interaction).detach().abs().max()) > 1e-5
    torch.testing.assert_close(prediction.sum(-1), torch.ones((3, 17), dtype=torch.float64),
                               rtol=0, atol=1e-14)
    increments.requires_grad_()
    gradient = torch.autograd.grad(model(initial, increments, dt, context)[:, 7, 0].sum(),
                                   increments)[0]
    assert torch.count_nonzero(gradient[:, 7:]) == 0
    assert torch.count_nonzero(gradient[:, :7]) > 0


@pytest.mark.parametrize("case", ["dk_interacting_uniform", "dk_interacting"])
def test_exact_linear_kernel_picard_iteration_approaches_interacting_euler(case):
    config = path_preset("path-interacting-smoke")
    spec, context, dt = config.spec, config.context(case), config.step_sizes[0]
    initial = torch.from_numpy(initialize_masses(context, spec, 3, 445))
    generator = torch.Generator().manual_seed(446)
    increments = torch.randn(3, 16, spec.n_cells, dtype=torch.float64,
                             generator=generator)*np.sqrt(dt)
    reference = torch_euler(initial, increments, context, spec, dt, "dk")
    assert torch.isinf(reference["stopped_at"]).all()
    errors = []
    for depth in (1, 4, 8):
        model = CausalPathOperator(spec, "dk", width=8, depth=depth)

        def exact_kernel(self, steps, step_size, physical_context):
            pole = 1-step_size*physical_context.diffusion*self.laplacian_rates
            return pole[None]**torch.arange(steps+1, dtype=torch.float64)[:, None]

        # This is an architecture-only oracle-kernel diagnostic, not a fit or
        # held-out learned-solver accuracy result.
        model.kernel_values = MethodType(exact_kernel, model)
        with torch.no_grad():
            prediction = model(initial, increments, dt, context)
        errors.append(float((prediction-reference["states"]).abs().max()))
    assert errors[1] < errors[0]/1000
    assert errors[2] < errors[1]/1000
    assert errors[2] < 1e-11
