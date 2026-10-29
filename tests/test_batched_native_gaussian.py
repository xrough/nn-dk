"""Independent dense/sparse LNA equivalence and aligned failure contracts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.batched_native_gaussian import _rhs, conditional_gaussian_batch  # noqa: E402
from analysis.native_gaussian_baseline import (  # noqa: E402
    drift_jacobian_covariance,
    linear_gaussian,
)
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.types import Context  # noqa: E402


def _initials(spec):
    x = (np.arange(spec.n_cells)+.5)*spec.h
    a = np.full(spec.n_cells, 1/spec.n_cells)
    b = np.exp(.2*np.cos(x))
    b /= b.sum()
    return np.stack((a, b))


def test_sparse_covariance_rhs_matches_dense_jacobian_for_nonsymmetric_input():
    spec, context = NativeDKSpec(8), Context(kappa=6., a2=.1)
    bank = PayoffBank.mixed_terminal_integral(spec, 1.)
    m = _initials(spec)
    means = np.concatenate((m, np.zeros((2, 4))), axis=1)
    cov = np.random.default_rng(81).normal(size=(2, 12, 12))
    observed_mean, observed_cov = _rhs(means, cov, context, spec, bank.running_weights)
    for i in range(2):
        drift, jac, noise = drift_jacobian_covariance(m[i], context, spec)
        a = np.zeros((12, 12))
        a[:8, :8] = jac
        a[8:, :8] = bank.running_weights
        q = np.zeros((12, 12))
        q[:8, :8] = noise
        np.testing.assert_allclose(observed_mean[i], np.r_[drift, bank.running_weights @ m[i]],
                                   atol=3e-15, rtol=2e-14)
        np.testing.assert_allclose(observed_cov[i], a @ cov[i]+cov[i] @ a.T+q,
                                   atol=2e-13, rtol=2e-14)


@pytest.mark.parametrize("horizon", [.25, 1.])
@pytest.mark.parametrize("kappa", [.75, 3., 6.])
@pytest.mark.parametrize("step", [2**-7, 2**-10])
def test_all_regimes_match_historical_dense_rk4_at_every_query(horizon, kappa, step):
    spec, context = NativeDKSpec(32), Context(kappa=kappa, a2=.1, n_particles=2048)
    bank = PayoffBank.mixed_terminal_integral(spec, horizon)
    times = np.array([0., horizon/4, horizon/2, horizon])
    m = _initials(spec)
    result = conditional_gaussian_batch(m, context, spec, times, step, bank.terminal_weights,
                                       bank.running_weights, batch_size=2, return_native=True)
    assert result["diagnostics"]["valid"].all()
    assert result["diagnostics"]["no_covariance_repair"]
    for i in range(2):
        exact = linear_gaussian(m[i], context, spec, times, step,
                                terminal_weights=bank.terminal_weights,
                                running_weights=bank.running_weights, return_joint=False)
        reduction = np.concatenate((np.eye(4), np.eye(4)), axis=1)
        expected_mean = exact["mean"] @ reduction.T
        expected_cov = np.einsum("ai,tij,bj->tab", reduction, exact["scaled_covariance"], reduction)
        np.testing.assert_allclose(result["mean"][i], expected_mean, atol=5e-13, rtol=3e-12)
        np.testing.assert_allclose(result["scaled_covariance"][i], expected_cov, atol=5e-12, rtol=3e-11)
        np.testing.assert_allclose(result["mean_augmented"][i], exact["mean_augmented"],
                                   atol=5e-13, rtol=3e-12)
        np.testing.assert_allclose(result["scaled_covariance_augmented"][i],
                                   exact["scaled_covariance_augmented"], atol=5e-12, rtol=3e-11)
        assert result["diagnostics"]["integration_steps"] == exact["diagnostics"]["integration_steps"]
        assert result["diagnostics"]["minimum_deterministic_mass"][i] == pytest.approx(
            exact["diagnostics"]["minimum_deterministic_mass"], abs=5e-14)


def test_batches_and_tail_rows_preserve_each_histogram_and_conditional_zero_time():
    spec, context = NativeDKSpec(8), Context(kappa=3., a2=.1)
    bank = PayoffBank.mixed_terminal_integral(spec, .25)
    m = np.concatenate((_initials(spec), _initials(spec)[:1]), axis=0)
    times = [0., .03125, .25]
    batched = conditional_gaussian_batch(m, context, spec, times, 2**-7, bank.terminal_weights,
                                        bank.running_weights, batch_size=2)
    singles = conditional_gaussian_batch(m, context, spec, times, 2**-7, bank.terminal_weights,
                                        bank.running_weights, batch_size=1)
    assert batched["diagnostics"]["valid"].all()
    np.testing.assert_allclose(batched["mean"], singles["mean"], atol=3e-14, rtol=2e-13)
    np.testing.assert_allclose(batched["scaled_covariance"], singles["scaled_covariance"],
                               atol=3e-13, rtol=2e-12)
    np.testing.assert_array_equal(batched["scaled_covariance"][:, 0], np.zeros((3, 4, 4)))
    np.testing.assert_allclose(batched["mean"][:, 0], m @ bank.terminal_weights.T,
                               atol=2e-16, rtol=0)


def test_nonzero_terminal_and_running_rows_describe_the_same_payoffs():
    spec, context = NativeDKSpec(8), Context(kappa=3., a2=.1)
    x = (np.arange(8)+.5)*spec.h
    terminal = np.stack((np.sin(x), np.cos(2*x), np.sin(3*x)))
    running = np.stack((np.cos(x), np.sin(2*x), np.cos(3*x)))
    m = _initials(spec)
    observed = conditional_gaussian_batch(m, context, spec, [.125, .25], 2**-7,
                                         terminal, running, return_native=True)
    assert observed["diagnostics"]["valid"].all()
    for i in range(2):
        exact = linear_gaussian(m[i], context, spec, [.125, .25], 2**-7,
                                terminal_weights=terminal, running_weights=running, return_joint=False)
        reduction = np.concatenate((np.eye(3), np.eye(3)), axis=1)
        np.testing.assert_allclose(observed["mean"][i], exact["mean"] @ reduction.T,
                                   atol=3e-14, rtol=2e-13)
        expected = np.einsum("ai,tij,bj->tab", reduction, exact["scaled_covariance"], reduction)
        np.testing.assert_allclose(observed["scaled_covariance"][i], expected, atol=3e-13, rtol=2e-12)
    mass_cross_error = np.abs(observed["scaled_covariance_augmented"][:, :, :8, :].sum(axis=2))
    np.testing.assert_allclose(observed["diagnostics"]["covariance_mass_error"],
                               mass_cross_error.max(axis=(1, 2)), atol=0, rtol=0)


def test_invalid_histogram_remains_aligned_and_never_becomes_a_survivor_average():
    spec, context = NativeDKSpec(8), Context(kappa=.75, a2=.1)
    bank = PayoffBank.mixed_terminal_integral(spec, .25)
    m = np.concatenate((_initials(spec), _initials(spec)[:1]), axis=0)
    m[1, 0] = -.1
    m[1, 1] += 1-m[1].sum()
    result = conditional_gaussian_batch(m, context, spec, [.25], 2**-7,
                                       bank.terminal_weights, bank.running_weights, batch_size=3)
    np.testing.assert_array_equal(result["diagnostics"]["valid"], [True, False, True])
    assert result["mean"].shape == (3, 1, 4)
    assert np.isnan(result["mean"][1]).all()
    assert np.isnan(result["scaled_covariance"][1]).all()
    assert "nonnegative" in result["diagnostics"]["errors"][1]
    assert not result["diagnostics"]["boundary_diagnostic_valid"][1]
    assert result["diagnostics"]["no_rows_dropped"]
    np.testing.assert_allclose(result["mean"][0], result["mean"][2], atol=3e-15, rtol=0)


def test_zero_observable_retains_exact_singular_covariance_without_a_floor():
    spec, context = NativeDKSpec(8), Context(diffusion=1., kappa=0., a2=0.)
    bank = PayoffBank.mixed_terminal_integral(spec, .25)
    terminal, running = bank.terminal_weights.copy(), bank.running_weights.copy()
    terminal[0] = 0.
    running[0] = 0.
    result = conditional_gaussian_batch(_initials(spec), context, spec, [0., .25], 2**-7,
                                       terminal, running)
    assert result["diagnostics"]["valid"].all()
    np.testing.assert_array_equal(result["scaled_covariance"][:, :, 0, :], np.zeros((2, 2, 4)))
    np.testing.assert_array_equal(result["scaled_covariance"][:, :, :, 0], np.zeros((2, 2, 4)))


def test_structural_and_stability_errors_fail_before_any_anchor_evolution():
    spec, context = NativeDKSpec(32), Context(kappa=6., a2=.1)
    bank = PayoffBank.mixed_terminal_integral(spec, 1.)
    with pytest.raises(ValueError, match="stability"):
        conditional_gaussian_batch(_initials(spec), context, spec, [1.], .1,
                                   bank.terminal_weights, bank.running_weights)
    with pytest.raises(ValueError, match="increasing"):
        conditional_gaussian_batch(_initials(spec), context, spec, [.25, .25], 2**-7,
                                   bank.terminal_weights, bank.running_weights)
