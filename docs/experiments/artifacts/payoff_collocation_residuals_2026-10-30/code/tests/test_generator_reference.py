"""Independent moment, boundary and numerical-enclosure checks for references."""

from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from nn_dk.bsde_study_config import bsde_study_preset
from nn_dk.generator_reference import (
    _exp_upper,
    _formal_moments,
    _geometry_parameters,
    _negative_exp_interval,
    _pi,
    _riccati_grid,
    _riccati_jacobian_action,
    _riccati_rhs,
    _weights,
    profile_probabilities,
    reference_conditional,
    reference_ensemble,
    reference_ou,
)
from nn_dk.native import native_coefficients, physical_observable_weights
from nn_dk.observations import incidence_matrix
from nn_dk.types import Context


def _base():
    return bsde_study_preset("bsde-anchor-study").base_config


def test_decimal_geometry_and_observations_agree_with_native_contract():
    base = _base()
    alpha, error, stop, stop_error = _geometry_parameters(base)
    with localcontext() as ctx:
        ctx.prec = 80
        exact_alpha = Decimal(1)/(2*_pi()/32)**2
        assert abs(Decimal.from_float(alpha)-exact_alpha) <= Decimal.from_float(error)
        exact_stop = Decimal.from_float(base.rho_stop)*2*_pi()/32
        assert abs(Decimal.from_float(stop)-exact_stop) <= Decimal.from_float(stop_error)
    weights, radius = _weights(32)
    native = physical_observable_weights(base.spec, base.queries)
    assert np.max(np.abs(weights-native)) < 4e-15
    assert radius.max() < 3e-16


def test_profile_simpson_bounds_cover_independent_adaptive_quadrature():
    base = _base()
    context = base.context("nonuniform")
    p, bound = profile_probabilities(base, context)
    normalizer = quad(lambda x: np.exp(.2*np.cos(x)), 0, 2*np.pi,
                      epsabs=1e-13, epsrel=1e-13)[0]
    exact = np.asarray([quad(lambda x: np.exp(.2*np.cos(x)), i*base.spec.h,
                            (i+1)*base.spec.h, epsabs=1e-14, epsrel=1e-13)[0]/normalizer
                        for i in range(32)])
    assert abs(p.sum()-1) < 3e-16
    assert np.all(np.abs(p-exact) <= bound+2e-17)
    assert bound.max() < 2e-14
    uniform, error = profile_probabilities(base, base.context("uniform"))
    np.testing.assert_array_equal(uniform, np.full(32, 1/32))
    np.testing.assert_array_equal(error, 0)


def test_poisson_backward_moments_match_full_forward_lyapunov_ode():
    base = replace(_base(), n_cells=8)
    initial = np.random.default_rng(4).dirichlet(np.full(8, 3.))
    time = 1/32
    mean, covariance, mean_error, covariance_error, _ = _formal_moments(
        base, initial[None], (time,))
    incidence = incidence_matrix(8)
    alpha = base.diffusion/base.spec.h**2
    matrix = -alpha*incidence@incidence.T
    def rhs(t, state):
        mu, cov = state[:8], state[8:].reshape(8, 8)
        source = alpha*incidence@np.diag(mu+np.roll(mu, -1))@incidence.T/base.n_particles
        return np.concatenate((matrix@mu, (matrix@cov+cov@matrix.T+source).ravel()))
    initial_state = np.concatenate((initial, np.zeros(64)))
    solved = solve_ivp(rhs, (0, time), initial_state, method="DOP853", atol=1e-14, rtol=3e-13)
    weights, _ = _weights(8)
    expected_mean = weights@solved.y[:8, -1]
    expected_covariance = base.n_particles*weights@solved.y[8:, -1].reshape(8, 8)@weights.T
    assert np.all(np.abs(mean[0, 0]-expected_mean) < mean_error[0, 0]+2e-13)
    assert np.all(np.abs(covariance[0, 0]-expected_covariance) < covariance_error[0, 0]+2e-11)


def test_backward_source_normalization_matches_native_coefficients():
    base = replace(_base(), n_cells=8)
    state = np.random.default_rng(8).dirichlet(np.full(8, 5.))
    coefficients = native_coefficients(state, base.context("uniform"), base.spec)
    weights, _ = _weights(8)
    difference = np.roll(weights, -1, axis=1)-weights
    alpha = base.diffusion/base.spec.h**2
    for i in range(4):
        for j in range(4):
            c = alpha*(difference[i]*difference[j]+np.roll(difference[i]*difference[j], 1))
            assert c@state == pytest.approx(weights[i]@coefficients.covariance@weights[j], abs=2e-14)


def test_uniform_conditional_and_iid_stationary_formulas_are_enclosed():
    base = _base()
    packet = reference_conditional(base, np.full((1, 32), 1/32), base.query_times)
    assert packet["certificate"]["valid"] is True
    expected_variance = base.observable_scale**2*(1-np.exp(-2*np.asarray(base.query_times)[:, None]*base.mode_rates))
    expected = np.zeros((3, 4, 4))
    expected[:, np.arange(4), np.arange(4)] = expected_variance
    assert np.all(np.abs(packet["mean"]) <= packet["mean_error_bound"])
    assert np.all(np.abs(packet["scaled_covariance"][0]-expected) <= packet["scaled_covariance_error_bound"][0])
    assert packet["scaled_covariance_error_bound"].max() < .0125
    assert packet["certificate"]["stopping"]["probability_bound"][0][-1] < 1e-9
    stationary = reference_ensemble(base, base.context("uniform"), base.query_times)
    target = np.broadcast_to(np.diag(base.observable_scale**2), (3, 4, 4))
    assert np.all(np.abs(stationary["scaled_covariance"]-target) <= stationary["scaled_covariance_error_bound"])
    assert stationary["certificate"]["valid"] is True


def test_initial_absorption_returns_frozen_state_and_zero_covariance():
    base = _base()
    initial = np.full((2, 32), 1/32)
    initial[0, 0], initial[0, 1] = 0., 2/32
    packet = reference_conditional(base, initial, base.query_times)
    weights, _ = _weights(32)
    expected = initial[0]@weights.T
    np.testing.assert_allclose(packet["mean"][0], np.broadcast_to(expected, (3, 4)), atol=1e-15)
    np.testing.assert_array_equal(packet["scaled_covariance"][0], 0.)
    assert packet["scaled_covariance_error_bound"][0].max() < 1e-300
    assert packet["certificate"]["initially_stopped_conditional_states"] == 1


def test_unresolved_ode_error_never_gets_valid_certificate():
    base = _base()
    packet = reference_conditional(base, np.full((1, 32), 1/32), base.query_times,
                                   settings={"certificate_tolerance": 1e-12})
    assert packet["certificate"]["valid"] is False
    assert packet["certificate"]["stopping"]["maximum_u_error"] > 1e-12


def test_riccati_enclosure_covers_an_independent_high_order_solver():
    base = replace(_base(), n_cells=8)
    alpha, alpha_error, _, _ = _geometry_parameters(base)
    horizon = 1/32
    values, error, _ = _riccati_grid(8, alpha, horizon, 2**-13, 2**-13, 1., alpha_error)
    initial = np.zeros(8)
    initial[0] = 1.
    result = solve_ivp(lambda t, u: _riccati_rhs(u, alpha), (0, horizon), initial,
                       method="DOP853", rtol=3e-13, atol=1e-14)
    assert np.max(np.abs(values[-1]-result.y[:, -1])) <= error[-1]
    assert np.min(values) >= 0 and np.max(values) <= 1
    u = np.random.default_rng(7).uniform(size=8)
    jacobian = np.stack([_riccati_jacobian_action(u, row, alpha) for row in np.eye(8)], axis=1)
    off_diagonal = jacobian.copy()
    np.fill_diagonal(off_diagonal, 0)
    assert off_diagonal.min() >= 0
    np.testing.assert_allclose(jacobian.sum(1), 0, atol=2e-15)


def test_positive_exponential_enclosures_cover_decimal_values():
    with localcontext() as ctx:
        ctx.prec = 80
        for x in (-700., -20., -1., 0., 1., 3.):
            assert Decimal.from_float(float(_exp_upper(np.asarray(x)))) >= Decimal.from_float(x).exp()
        lower, upper = _negative_exp_interval(np.asarray((0., .5, 8.)), np.asarray((0., .5, 8.)))
        for x, lo, hi in zip((0., .5, 8.), lower, upper, strict=True):
            exact = (-Decimal.from_float(x)).exp()
            assert Decimal.from_float(float(lo)) <= exact <= Decimal.from_float(float(hi))


def test_ou_reference_has_joint_positive_covariance_and_tight_bounds():
    base = _base()
    initial = np.asarray(((1., -2., 3., -4.), (0., 0., 0., 0.)))
    packet = reference_ou(base, initial, base.query_times)
    expected = initial[:, None]*np.exp(-np.asarray(base.query_times)[:, None]*base.mode_rates)
    assert np.all(np.abs(packet["mean"]-expected) <= packet["mean_error_bound"]+2e-15)
    assert packet["certificate"]["valid"] is True
    assert np.linalg.eigvalsh(packet["scaled_covariance"]).min() > 0
    assert packet["mean_error_bound"].max() < 1e-12
    assert packet["scaled_covariance_error_bound"].max() < 1e-12


@pytest.mark.parametrize("horizons", [(), (0.,), (np.nan,), (.5, .25)])
def test_invalid_reference_queries_are_rejected(horizons):
    with pytest.raises(ValueError, match="Positive increasing"):
        reference_conditional(_base(), np.full(32, 1/32), horizons)


def test_unsupported_physics_or_certificate_assumptions_are_rejected():
    base = _base()
    with pytest.raises(ValueError, match="noninteracting"):
        reference_ensemble(base, Context(kappa=.5, n_particles=2048), base.query_times)
    with pytest.raises(ValueError, match="tilt"):
        reference_conditional(base, np.full(32, 1/32), base.query_times, settings={"tilt": 2.})
    with pytest.raises(ValueError, match="nonnegative normalized"):
        reference_conditional(base, np.full(32, 1/64), base.query_times)
