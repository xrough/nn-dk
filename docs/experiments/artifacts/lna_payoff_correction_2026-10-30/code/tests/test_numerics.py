"""Numerical contracts: observation, forcing, conservation and fluctuation laws."""
import numpy as np
import pytest

from nn_dk.classical import _drift_and_jacobian, _geometry, discrete_dk, meanfield_and_fluctuations
from nn_dk.observations import (
    incidence_matrix,
    observation_matrix,
    observe_field,
    observe_particles,
    spline_weights,
    tangent_basis,
    uniform_quadratic_variation,
)
from nn_dk.particles import (
    coupled_refinement,
    free_particle_statistics,
    initial_profile,
    pairwise_force,
    particle_force,
    sample_initial,
    simulate_particles,
)
from nn_dk.types import LENGTH, Context, QueryBank, TargetSpec


def test_spline_partition_derivatives_and_periodicity():
    target = TargetSpec()
    x = np.linspace(-1, 2 * LENGTH, 257)
    w = spline_weights(x, target)
    np.testing.assert_allclose(w.sum(axis=1), 1, atol=2e-15)
    assert w.min() >= 0
    np.testing.assert_allclose(w, spline_weights(x + LENGTH, target), atol=1e-14)
    for order in (1, 2):
        np.testing.assert_allclose(spline_weights(x, target, order).sum(axis=1), 0, atol=5e-13)
    eps = 1e-6
    first_difference = (spline_weights(x + eps, target) - spline_weights(x - eps, target)) / (2 * eps)
    second_difference = (spline_weights(x + eps, target, 1) - spline_weights(x - eps, target, 1)) / (2 * eps)
    np.testing.assert_allclose(first_difference, spline_weights(x, target, 1), atol=5e-9)
    np.testing.assert_allclose(second_difference, spline_weights(x, target, 2), atol=3e-4)


def test_particle_and_piecewise_constant_field_observations_match():
    target = TargetSpec(n_cells=16)
    rng = np.random.default_rng(91)
    positions = rng.uniform(-LENGTH, 2 * LENGTH, (3, 127))
    np.testing.assert_allclose(observe_particles(positions, target), spline_weights(positions, target).mean(axis=1), atol=2e-15)
    for cells in (16, 24, 32):
        a = observation_matrix(target, cells)
        np.testing.assert_allclose(a.sum(axis=0), 1, atol=1e-14)
        np.testing.assert_allclose(observe_field(np.full(cells, 1 / cells), target), 1 / target.n_cells, atol=2e-15)
        # Deterministic cell-midpoint quadrature converges to the exact spline integral.
        h = LENGTH / cells
        fine = (np.arange(cells)[:, None] + (np.arange(1024) + .5) / 1024) * h
        empirical = spline_weights(fine, target).mean(axis=1).T
        np.testing.assert_allclose(a, empirical, atol=2e-7)


def test_force_sign_fast_formula_and_conservation():
    context = Context(kappa=1.7, a2=.3, n_particles=23)
    x = np.random.default_rng(81).uniform(0, LENGTH, (4, 23))
    np.testing.assert_allclose(particle_force(x, context), pairwise_force(x, context), atol=2e-15)
    np.testing.assert_allclose(particle_force(x, context).sum(axis=-1), 0, atol=1e-14)
    attraction = particle_force(np.array([0, .2]), Context(kappa=1, n_particles=2))
    assert attraction[0] > 0 and attraction[1] < 0
    repulsion = particle_force(np.array([0, .2]), Context(a2=1, n_particles=2))
    assert repulsion[0] < 0 and repulsion[1] > 0


def test_exact_initial_sampler_profile_moments():
    context = Context(cos_coeffs=(.4, -.2, .1), sin_coeffs=(.1, .2, -.1), n_particles=4096)
    positions = sample_initial(context, 8, np.random.default_rng(941))
    grid = np.arange(16384) * LENGTH / 16384
    rho = initial_profile(grid, context)
    np.testing.assert_allclose(rho.mean() * LENGTH, 1, atol=1e-14)
    for k in (1, 2, 3):
        expected = (rho * np.cos(k * grid)).mean() * LENGTH
        assert abs(np.cos(k * positions).mean() - expected) < .022


def test_tangent_geometry_and_exact_uniform_quadratic_variation():
    target = TargetSpec(n_cells=16)
    basis = tangent_basis(16)
    np.testing.assert_allclose(basis.T @ basis, np.eye(15), atol=5e-16)
    np.testing.assert_allclose(basis.sum(axis=0), 0, atol=5e-16)
    b = incidence_matrix(16)
    np.testing.assert_array_equal(b.sum(axis=0), 0)
    q = uniform_quadratic_variation(target, diffusion=.7)
    np.testing.assert_allclose(q.sum(axis=0), 0, atol=1e-14)
    assert np.linalg.eigvalsh(basis.T @ q @ basis).min() > 0
    x = np.arange(32768) * LENGTH / 32768
    d = spline_weights(x, target, derivative=1)
    np.testing.assert_allclose(q, 1.4 * d.T @ d / len(x), rtol=2e-12, atol=2e-12)


def test_particle_wrapping_brownian_variance_and_coupled_free_identity():
    target = TargetSpec(n_cells=16)
    context = Context(n_particles=32)
    dt = 1 / 256
    # Supplied increments give a fully checkable wrapped trajectory.
    increments = np.full((2, 2, 32), .2)
    x0 = np.full((2, 32), LENGTH - .01)
    batch = simulate_particles(context, target, (0, dt, 2 * dt), dt, 2, 2,
                               x0, increments, retain_particles=True)
    np.testing.assert_allclose(batch.particle_states[:, -1], (x0 + np.sqrt(2) * .4) % LENGTH)
    assert batch.diagnostics["mass_error"] < 1e-14
    paired = coupled_refinement(context, target, (0, 1 / 32), 1 / 128, 71, 3)
    np.testing.assert_allclose(paired["coarse"].states, paired["quarter"].states, atol=3e-15)
    # Very short time avoids wrapping: EM's microscopic variance is 2D dt.
    many = Context(n_particles=4096)
    microscopic = simulate_particles(many, target, (0, 1e-4), 1e-4, 15, 2,
                                     np.full((2, 4096), np.pi), retain_particles=True)
    delta = microscopic.particle_states[:, 1] - microscopic.particle_states[:, 0]
    assert abs(delta.var() / (2e-4) - 1) < .04


def test_heat_identity_filtered_modes_and_joint_covariance():
    target = TargetSpec(n_cells=16)
    query = QueryBank(times=(.1, .4))
    reference = free_particle_statistics(Context(), target, query, quadrature=8192)
    np.testing.assert_allclose(reference.mean, 0, atol=5e-15)
    assert np.linalg.eigvalsh(reference.scaled_covariance).min() > 0
    # Cubic filtering attenuates mode k by sinc(k/M)^4. Aliases at |k+nM|
    # have decayed below roundoff at the nonzero two-time lag used here.
    for k in (1, 2):
        attenuation = np.sinc(k / target.n_cells) ** 4
        expected = attenuation**2 * np.exp(-k**2 * .3)
        offset = (k - 1) * 2
        np.testing.assert_allclose(reference.scaled_covariance[offset, offset + 4], expected, rtol=1e-9)
    with pytest.raises(ValueError, match="noninteracting"):
        free_particle_statistics(Context(kappa=1), target, query)


def test_fv_jacobian_initial_covariance_and_heat_refinement():
    context = Context(kappa=.4, a2=.1)
    target = TargetSpec(n_cells=16)
    n = 24
    geometry = _geometry(context, target, n, 1 / 2048)
    mass = np.random.default_rng(8).dirichlet(np.full(n, 5))
    # The velocity integrates the declared piecewise constant reconstruction,
    # including each cell's sinc attenuation, rather than a point-mass rule.
    h, force, _, _ = geometry
    y = (np.arange(n)[:, None] + (np.arange(2048) + .5) / 2048) * h
    faces = (np.arange(n) + 1) * h
    difference = faces[:, None, None] - y[None]
    exact_velocity = ((-context.kappa * np.sin(difference)
                       + 2 * context.a2 * np.sin(2 * difference)).mean(axis=2) @ mass)
    np.testing.assert_allclose(force @ mass, exact_velocity, atol=2e-10)
    drift, jacobian, density = _drift_and_jacobian(mass, context, geometry)
    direction = np.random.default_rng(9).normal(size=n)
    eps = 1e-7
    numerical = (_drift_and_jacobian(mass + eps * direction, context, geometry)[0]
                 - _drift_and_jacobian(mass - eps * direction, context, geometry)[0]) / (2 * eps)
    np.testing.assert_allclose(jacobian @ direction, numerical, atol=2e-8)
    assert abs(drift.sum()) < 1e-14
    b = geometry[2]
    q = (b * (2 * context.diffusion / geometry[0] * density)[None, :]) @ b.T
    np.testing.assert_allclose(q.sum(axis=0), 0, atol=3e-14)
    query = QueryBank(times=(1 / 16, 1 / 8))
    free = Context()
    _, coarse = meanfield_and_fluctuations(free, target, query, 1 / 2048, n_grid=16)
    deterministic, fine = meanfield_and_fluctuations(free, target, query, 1 / 8192, n_grid=64)
    truth = free_particle_statistics(free, target, query)
    assert np.linalg.norm(fine.scaled_covariance - truth.scaled_covariance) < np.linalg.norm(coarse.scaled_covariance - truth.scaled_covariance)
    np.testing.assert_allclose(deterministic.scaled_covariance, 0)
    np.testing.assert_allclose(fine.diagnostics["initial_scaled_covariance_trace"], 1 - 1 / 64, atol=1e-14)
    assert fine.diagnostics["mass_error"] < 1e-14


def test_dk_matched_initial_projection_failure_and_cfl():
    target = TargetSpec(n_cells=16)
    context = Context(n_particles=128)
    x0 = sample_initial(context, 3, np.random.default_rng(83))
    batch = discrete_dk(context, target, (0, 1 / 128), 1 / 4096, 3, 3, 16, x0)
    h = LENGTH / 16
    cell = np.asarray([np.bincount((row / h).astype(int), minlength=16) / 128 for row in x0])
    np.testing.assert_allclose(batch.states[:, 0], observe_field(cell, target), atol=2e-15)
    assert batch.diagnostics["initial_projection_rms"] > 0
    assert batch.diagnostics["mass_error"] < 1e-13
    # A single particle yields empty cells; conservative noise exposes invalidity.
    failing = discrete_dk(Context(n_particles=1), target, (0, 1 / 32), 1 / 4096, 4, 8, 16)
    assert not failing.valid.all()
    assert np.all(np.isfinite(failing.failure_times[~failing.valid]))
    assert np.isnan(failing.states[~failing.valid, -1]).all()
    with pytest.raises(ValueError, match="CFL"):
        discrete_dk(context, target, (0, 1), 1, 1)
    with pytest.raises(ValueError, match="time grid"):
        simulate_particles(context, target, (0, .1), 1 / 32, 1)
