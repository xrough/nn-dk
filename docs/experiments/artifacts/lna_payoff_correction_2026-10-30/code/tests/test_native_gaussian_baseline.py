"""Known-generator Gaussian baseline: exact Jacobian, moments, integrals and law limits."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from nn_dk.native import native_coefficients
from nn_dk.native_types import NativeDKSpec
from nn_dk.path_numerics import path_coefficients
from nn_dk.types import Context

_spec = importlib.util.spec_from_file_location(
    "native_gaussian_baseline", Path(__file__).parents[1]/"analysis"/"native_gaussian_baseline.py")
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


@pytest.mark.parametrize("interaction", [(0., 0.), (.75, .1), (-.4, -.2)])
def test_drift_covariance_and_exact_jacobian_match_native_autograd_and_fd(interaction):
    spec = NativeDKSpec(n_cells=16)
    context = Context(kappa=interaction[0], a2=interaction[1], n_particles=2048)
    m = np.random.default_rng(123).dirichlet(np.full(16, 20.))
    drift, jacobian, covariance = baseline.drift_jacobian_covariance(m, context, spec)
    exact = native_coefficients(m, context, spec)
    np.testing.assert_allclose(drift, exact.drift, atol=3e-15, rtol=2e-14)
    np.testing.assert_allclose(covariance, exact.covariance, atol=3e-15)
    automatic = torch.autograd.functional.jacobian(
        lambda state: path_coefficients(state, context, spec)[0], torch.tensor(m))
    np.testing.assert_allclose(jacobian, automatic, atol=5e-15, rtol=3e-14)
    direction = np.random.default_rng(321).normal(size=16)
    direction -= direction.mean()
    epsilon = 1e-7
    plus = baseline.drift_jacobian_covariance(m+epsilon*direction, context, spec)[0]
    minus = baseline.drift_jacobian_covariance(m-epsilon*direction, context, spec)[0]
    np.testing.assert_allclose(jacobian @ direction, (plus-minus)/(2*epsilon), atol=1e-9)
    np.testing.assert_allclose(jacobian.sum(0), 0, atol=4e-15)


@pytest.mark.parametrize("iid", [False, True])
def test_uniform_free_particle_moments_match_analytic_matrix_exponential(iid):
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    initial = np.full(8, 1/8)
    times = np.array([0., .125, .25, .5])
    stationary = baseline.iid_scaled_covariance(initial)
    initial_cov = stationary if iid else np.zeros((8, 8))
    result = baseline.linear_gaussian(initial, context, spec, times, 1/1024,
                                      initial_scaled_covariance=initial_cov if iid else None)
    a = baseline.drift_jacobian_covariance(initial, context, spec)[1]
    np.testing.assert_allclose(result["mean"], np.tile(initial, (4, 1)), atol=3e-16)
    expected = []
    for t in times:
        f = expm(a*t)
        expected.append(stationary+f @ (initial_cov-stationary) @ f.T)
    np.testing.assert_allclose(result["scaled_covariance"], expected, atol=2e-11, rtol=2e-10)
    for i in range(len(times)):
        for j in range(i, len(times)):
            cross = expected[i] @ expm(a.T*(times[j]-times[i]))
            np.testing.assert_allclose(result["joint_scaled_covariance"][i, :, j, :], cross,
                                       atol=2e-11, rtol=2e-10)
    assert result["diagnostics"]["mean_mass_error"] < 1e-14
    assert result["diagnostics"]["covariance_mass_error"] < 1e-14


def test_augmented_linear_integrals_and_joint_moments_match_van_loan():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    p = np.full(8, 1/8)
    centers = (np.arange(8)+.5)*spec.h
    terminal = np.stack((np.cos(centers), np.sin(centers)))
    running = np.cos(2*centers)[None]
    times = np.array([.125, .25, .5])
    result = baseline.linear_gaussian(p, context, spec, times, 1/2048,
                                      terminal_weights=terminal, running_weights=running)
    _, j, q = baseline.drift_jacobian_covariance(p, context, spec)
    a, forcing = np.zeros((9, 9)), np.zeros((9, 9))
    a[:8, :8], a[8:, :8], forcing[:8, :8] = j, running, q
    output = np.zeros((3, 9))
    output[:2, :8], output[2, 8] = terminal, 1
    covariances = []
    for t in times:
        van_loan = expm(np.block([[a, forcing], [np.zeros((9, 9)), -a.T]])*t)
        f = van_loan[:9, :9]
        covariance = van_loan[:9, 9:] @ f.T
        covariances.append(covariance)
    for i in range(3):
        np.testing.assert_allclose(result["scaled_covariance_augmented"][i], covariances[i],
                                   atol=2e-12, rtol=2e-10)
        for k in range(i, 3):
            expected = output @ covariances[i] @ expm(a.T*(times[k]-times[i])) @ output.T
            np.testing.assert_allclose(result["joint_scaled_covariance"][i, :, k, :], expected,
                                       atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(result["mean"], 0, atol=3e-16)


def test_refinement_reduces_rk4_covariance_error_without_clipping_or_psd_repair():
    spec, context = NativeDKSpec(n_cells=16), Context(kappa=.75, a2=.1, n_particles=2048)
    centers = (np.arange(16)+.5)*spec.h
    p = np.exp(.2*np.cos(centers))
    p /= p.sum()
    reference = baseline.linear_gaussian(p, context, spec, [.25], 1/2048)
    errors = []
    for step in (1/64, 1/128, 1/256):
        result = baseline.linear_gaussian(p, context, spec, [.25], step)
        errors.append(np.linalg.norm(result["scaled_covariance"]-reference["scaled_covariance"]))
        assert result["diagnostics"]["no_covariance_repair"]
        assert not result["diagnostics"]["stopped_dk_equivalence"]
        assert result["diagnostics"]["stopping_effects"] == "unresolved"
    assert errors[1] < errors[0]/10
    assert errors[2] < errors[1]/10


def test_mean_boundary_encounter_is_explicitly_flagged_and_not_repaired():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    p = np.full(8, 1/8)
    p[0], p[1] = 0, .25
    result = baseline.linear_gaussian(p, context, spec, [0., .01], .001)
    np.testing.assert_array_equal(result["mean_states"][0], p)
    assert result["diagnostics"]["deterministic_boundary_encountered"]
    assert not result["diagnostics"]["stopped_dk_equivalence"]
    # This unstopped approximation is deliberately not absorbed at the boundary.
    assert result["mean_states"][1, 0] > 0


def test_linearized_euler_matches_exact_discrete_fourier_moments_and_running_sum():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    p = np.full(8, 1/8)
    step = 1/64
    times = np.array([0., .125, .25])
    result = baseline.linear_gaussian(p, context, spec, times, step,
                                      running_weights=np.ones((1, 8)), scheme="linearized-euler")
    rates = 4*context.diffusion/spec.h**2*np.sin(np.pi*np.arange(8)/8)**2
    poles = 1-step*rates
    for index, t in enumerate(times):
        count = round(t/step)
        variances = np.zeros(8)
        variances[1:] = (2*rates[1:]/8)*step*(1-poles[1:]**(2*count))/(1-poles[1:]**2)
        kernel = np.fft.ifft(variances).real
        covariance = kernel[(np.arange(8)[:, None]-np.arange(8)) % 8]
        np.testing.assert_allclose(result["scaled_covariance_states"][index], covariance, atol=4e-16)
        assert result["mean"][index, -1] == pytest.approx(t, abs=3e-16)
    assert result["law"] == baseline.EULER_LAW
    assert result["diagnostics"]["running_reward"] == "unstopped-left-point-sum"
    assert not result["diagnostics"]["killed_or_stopped_reward_equivalence"]
    with pytest.raises(ValueError, match="fixed step"):
        baseline.linear_gaussian(p, context, spec, [.1], step, scheme="linearized-euler")


def test_joint_gaussian_sampler_has_correct_moments_and_does_not_apply_stopping():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=8)
    p = np.full(8, 1/8)
    result = baseline.linear_gaussian(p, context, spec, [.125, .25], 1/512)
    sampled = baseline.sample_gaussian_process(result, context.n_particles, 16384, seed=8)
    samples = sampled["samples_augmented"]
    assert samples.shape == (16384, 2, 8)
    np.testing.assert_allclose(samples.mean(0), result["mean_augmented"], atol=.003)
    scaled_covariance = context.n_particles*np.cov(samples.reshape(len(samples), -1), rowvar=False)
    np.testing.assert_allclose(scaled_covariance,
                               result["joint_scaled_covariance_augmented"].reshape(16, 16), atol=.004)
    assert samples.min() < 0  # Gaussian tails are retained, without clipping.
    assert not sampled["diagnostics"]["stopping_applied"]


def test_illposed_initial_covariance_and_unstable_steps_are_rejected():
    spec, context = NativeDKSpec(n_cells=8), Context(n_particles=2048)
    p = np.full(8, 1/8)
    with pytest.raises(ValueError, match="mass-zero"):
        baseline.linear_gaussian(p, context, spec, [.25], .001,
                                 initial_scaled_covariance=np.eye(8))
    with pytest.raises(ValueError, match="stability"):
        baseline.linear_gaussian(p, context, spec, [1.], 1.)
    with pytest.raises(ValueError, match="increasing"):
        baseline.linear_gaussian(p, context, spec, [.25, .125], .001)
    result = baseline.linear_gaussian(p, context, spec, [.01], .001, return_joint=False)
    assert result["joint_scaled_covariance"] is None
    with pytest.raises(ValueError, match="Joint"):
        baseline.sample_gaussian_process(result, 2048, 5)
