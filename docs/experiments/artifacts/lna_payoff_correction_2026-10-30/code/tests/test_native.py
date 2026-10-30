"""Native DK calibration, absorption, and coupled-reference contracts."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from nn_dk.native import (
    boundary_step,
    coupled_native_refinement,
    extract_observables,
    initialize_masses,
    linear_gaussian_statistics,
    native_coefficients,
    physical_observable_weights,
    simulate_native,
)
from nn_dk.native_types import DKCoefficients, NativeDKSpec
from nn_dk.observations import incidence_matrix
from nn_dk.types import Context, QueryBank


def test_native_coefficients_conservation_noise_scaling_and_cell_integrated_force():
    spec = NativeDKSpec(n_cells=16)
    context = Context(kappa=0.75, a2=0.1, n_particles=2048)
    masses = np.arange(1, 17, dtype=float)
    masses /= masses.sum()
    coefficients = native_coefficients(masses, context, spec)
    b = incidence_matrix(spec.n_cells)
    following = np.roll(masses, -1)
    rho_face = (masses + following) / (2 * spec.h)
    np.testing.assert_allclose(coefficients.face_amplitudes**2,
                               2 * context.diffusion / spec.h * rho_face)
    np.testing.assert_allclose(coefficients.covariance,
                               (b * coefficients.face_amplitudes**2) @ b.T)
    np.testing.assert_allclose(coefficients.covariance.sum(0), 0, atol=1e-14)
    assert abs(coefficients.drift.sum()) < 1e-14

    # Independent high-order cell integration verifies both signs and sinc factors.
    nodes, weights = leggauss(16)
    positions = (np.arange(spec.n_cells)[:, None] + (nodes + 1) / 2) * spec.h
    faces = (np.arange(spec.n_cells) + 1) * spec.h
    differences = faces[:, None, None] - positions[None, :, :]
    force = -context.kappa * np.sin(differences) + 2 * context.a2 * np.sin(2 * differences)
    velocity = (force @ weights / 2) @ masses
    flux = rho_face * velocity - context.diffusion / spec.h**2 * (following - masses)
    np.testing.assert_allclose(coefficients.drift, b @ flux, atol=1e-14)
    doubled_n = native_coefficients(masses, Context(kappa=.75, a2=.1, n_particles=4096), spec)
    # Particle count is applied by the integrator, never inside generator labels.
    np.testing.assert_array_equal(coefficients.covariance, doubled_n.covariance)


def test_native_initialization_is_raw_histogram_and_reproducible():
    spec = NativeDKSpec(n_cells=16)
    context = Context(n_particles=2048, cos_coeffs=(.2, 0, 0))
    states = initialize_masses(context, spec, 200, 712)
    np.testing.assert_array_equal(states, initialize_masses(context, spec, 200, 712))
    np.testing.assert_array_equal(states * context.n_particles,
                                  np.rint(states * context.n_particles))
    np.testing.assert_allclose(states.sum(-1), 1, atol=1e-14)
    assert np.mean(states[:, :4]) > np.mean(states[:, 4:8])


def test_boundary_step_first_crossing_preserves_mass_without_clipping():
    spec = NativeDKSpec(n_cells=8)
    initial = np.full(8, 1 / 8)
    proposal = initial.copy()
    proposal[0] = -.2
    proposal[1] = -.1
    proposal[2] += .55
    stopped, crossed, tau = boundary_step(initial, proposal, .1, .2, spec)
    fraction = (initial[0] - spec.mass_stop) / (initial[0] - proposal[0])
    assert crossed
    assert tau == pytest.approx(.1 + .2 * fraction)
    np.testing.assert_allclose(stopped, initial + fraction * (proposal - initial))
    assert stopped[0] == pytest.approx(spec.mass_stop)
    assert stopped[1] > spec.mass_stop
    assert stopped.sum() == pytest.approx(1)
    assert np.linalg.norm(stopped - np.maximum(proposal, 0) / np.maximum(proposal, 0).sum()) > .01


def test_native_stopping_is_frozen_and_every_path_remains_in_statistics():
    spec = NativeDKSpec(n_cells=8)
    context = Context(n_particles=2048)
    origins = np.full((2, 8), 1 / 8)
    origins[1, 0], origins[1, 1] = 0, 1 / 4

    def outward(masses, physics):
        drift = np.zeros_like(masses)
        drift[:, 0], drift[:, 1] = -64, 64
        return DKCoefficients(drift, np.zeros((*masses.shape, 8)), np.zeros_like(masses))

    batch = simulate_native(context, spec, (0, 1 / 256, 2 / 256), 1 / 256, 45, 2,
                            initial_masses=origins, coefficients=outward)
    assert 0 < batch.stopped_at[0] < 1 / 256
    assert batch.stopped_at[1] == 0
    np.testing.assert_array_equal(batch.states[:, 1], batch.states[:, 2])
    np.testing.assert_array_equal(batch.states[1], np.tile(origins[1], (3, 1)))
    assert batch.states.shape[0] == 2
    assert batch.diagnostics["stopping_probability"] == 1
    assert batch.diagnostics["minimum_uncensored_proposal_mass"] < 0
    assert np.all(np.isfinite(batch.states))
    assert batch.diagnostics["mass_error"] < 1e-14


def test_native_streamed_noise_matches_supplied_increments_and_ablation():
    spec = NativeDKSpec(n_cells=8)
    context = Context(n_particles=100_000)
    initial = np.full((3, 8), 1 / 8)
    dw = np.random.default_rng(2).normal(0, np.sqrt(1 / 128), (3, 4, 8))

    def constant_noise(masses, physics):
        return DKCoefficients(np.zeros_like(masses), np.zeros((*masses.shape, 8)),
                              np.ones_like(masses))

    batch = simulate_native(context, spec, (0, 1 / 32), 1 / 128, 1, 3,
                            initial_masses=initial, brownian_increments=dw,
                            coefficients=constant_noise)
    total = dw.sum(1) / np.sqrt(context.n_particles)
    np.testing.assert_allclose(batch.states[:, -1],
                               initial + np.roll(total, 1, axis=-1) - total, atol=1e-15)
    assert np.all(np.isinf(batch.stopped_at))
    coarse = simulate_native(context, spec, (0, 1 / 32), 1 / 64, 1, 3,
                             initial_masses=initial,
                             brownian_increments=dw.reshape(3, 2, 2, 8).sum(2),
                             coefficients=constant_noise)
    np.testing.assert_allclose(batch.states[:, -1], coarse.states[:, -1], atol=1e-15)
    oracle = simulate_native(context, spec, (0, 1 / 32), 1 / 128, 1, 3,
                             initial_masses=initial, brownian_increments=dw,
                             coefficients=constant_noise, ablation="oracle")
    assert np.linalg.norm(batch.states[:, -1] - oracle.states[:, -1]) > 1e-5


def test_native_physical_observables_are_exact_fixed_functions_on_each_grid():
    queries = QueryBank(times=(.25,))
    # The same reconstructed piecewise-constant field is represented on two grids.
    coarse_spec, fine_spec = NativeDKSpec(n_cells=8), NativeDKSpec(n_cells=32)
    mass = np.arange(1, 9, dtype=float)
    mass /= mass.sum()
    fine_mass = np.repeat(mass / 4, 4)
    coarse = physical_observable_weights(coarse_spec, queries) @ mass
    fine = physical_observable_weights(fine_spec, queries) @ fine_mass
    np.testing.assert_allclose(coarse, fine, atol=1e-15)
    nodes, weights = leggauss(16)
    positions = (np.arange(8)[:, None] + (nodes + 1) / 2) * coarse_spec.h
    exact = np.asarray([np.sum(mass * ((np.sqrt(2) * function(k * positions)) @ weights / 2))
                        for k in queries.modes for function in (np.cos, np.sin)])
    np.testing.assert_allclose(coarse, exact, atol=1e-15)
    batch = simulate_native(Context(n_particles=2048), coarse_spec, (0, .25), 1 / 128, 44, 2)
    np.testing.assert_allclose(extract_observables(batch, queries),
                               batch.states[:, -1] @ physical_observable_weights(coarse_spec, queries).T)


def test_linear_uniform_heat_moments_and_euler_refinement():
    spec = NativeDKSpec(n_cells=8)
    context = Context(n_particles=2048)
    queries = QueryBank(times=(.25, .5, 1.0))
    continuous = linear_gaussian_statistics(context, spec, queries)
    weights = physical_observable_weights(spec, queries)
    p = np.full(8, 1 / 8)
    expected = weights @ (np.diag(p) - np.outer(p, p)) @ weights.T
    np.testing.assert_allclose(continuous["scaled_covariance"][:4, :4], expected, atol=1e-14)
    np.testing.assert_allclose(continuous["scaled_covariance"][-4:, -4:], expected, atol=1e-14)
    assert not continuous["metadata"]["stopped"]
    assert not continuous["metadata"]["initial_law_assumed_gaussian"]
    errors = []
    for dt in (1 / 32, 1 / 64, 1 / 128):
        euler = linear_gaussian_statistics(context, spec, queries, step_size=dt)
        errors.append(np.linalg.norm(euler["scaled_covariance"] - continuous["scaled_covariance"]))
    assert .4 < errors[1] / errors[0] < .55
    assert .4 < errors[2] / errors[1] < .55
    with pytest.raises(ValueError, match="uniform"):
        linear_gaussian_statistics(Context(cos_coeffs=(.2, 0, 0)), spec, queries)


def test_native_coupled_refinement_keeps_families_initialization_and_all_paths():
    spec = NativeDKSpec(n_cells=8)
    context = Context(n_particles=2048)
    coupled = coupled_native_refinement(context, spec, (0, .0625), 1 / 128, 99,
                                        n_paths=3, family_ids=("a", "b", "c"))
    assert set(coupled) == {"coarse", "half", "quarter"}
    for level, batch in coupled.items():
        assert batch.family_ids == ("a", "b", "c")
        assert batch.states.shape == (3, 2, 8)
        np.testing.assert_array_equal(batch.states[:, 0], coupled["coarse"].states[:, 0])
        assert batch.diagnostics["mass_error"] < 1e-14
        assert batch.diagnostics["all_paths_retained"]
    assert np.linalg.norm(coupled["coarse"].states[:, -1]
                          - coupled["quarter"].states[:, -1]) < .003


def test_native_rejects_bad_cfl_nonconservative_inputs_and_duplicate_families():
    spec = NativeDKSpec(n_cells=32)
    context = Context(n_particles=2048)
    with pytest.raises(ValueError, match="CFL"):
        simulate_native(context, spec, (0, 1), 1, 1)
    with pytest.raises(ValueError, match="sum to one"):
        native_coefficients(np.ones(32), context, spec)
    with pytest.raises(ValueError, match="unique"):
        simulate_native(context, spec, (0, 1 / 128), 1 / 128, 1, 2,
                        family_ids=("same", "same"))
