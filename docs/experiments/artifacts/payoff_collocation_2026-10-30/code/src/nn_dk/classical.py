"""Classical finite-volume comparators with explicit conservation and failures."""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss

from .observations import incidence_matrix, observation_matrix, observe_particles, spline_weights
from .particles import _time_grid, initial_profile, sample_initial
from .types import (
    Context,
    QueryBank,
    StatisticalPrediction,
    TargetSpec,
    TrajectoryBatch,
    seeded_rng,
    stable_hash,
)


def _geometry(context: Context, target: TargetSpec, n_grid: int, dt: float):
    if n_grid < 4:
        raise ValueError("At least four finite-volume cells required")
    h = target.length / n_grid
    velocity_bound = abs(context.kappa) + 2 * abs(context.a2)
    if dt * (2 * context.diffusion / h**2 + velocity_bound / h) > 1 + 1e-12:
        raise ValueError("Explicit finite-volume diffusion/advection CFL condition violated")
    if velocity_bound * h > 2 * context.diffusion + 1e-12:
        raise ValueError("Centered advection requires a finer grid for the cell Peclet condition")
    centers = (np.arange(n_grid) + 0.5) * h
    faces = (np.arange(n_grid) + 1) * h
    delta = faces[:, None] - centers[None, :]
    force = (-context.kappa * np.sinc(h / (2 * np.pi)) * np.sin(delta)
             + 2 * context.a2 * np.sinc(h / np.pi) * np.sin(2 * delta))
    b = incidence_matrix(n_grid)
    next_cell = np.roll(np.eye(n_grid), -1, axis=0)
    return h, force, b, next_cell


def _drift_and_jacobian(mass: np.ndarray, context: Context, geometry):
    h, force, b, shift = geometry
    face_density = (mass + np.roll(mass, -1)) / (2 * h)
    velocity = force @ mass
    flux = face_density * velocity - context.diffusion / h**2 * (np.roll(mass, -1) - mass)
    flux_jacobian = (velocity[:, None] * (np.eye(len(mass)) + shift) / (2 * h)
                     + face_density[:, None] * force
                     - context.diffusion / h**2 * (shift - np.eye(len(mass))))
    return b @ flux, b @ flux_jacobian, face_density


def _initial_cell_probabilities(context: Context, target: TargetSpec, n_grid: int) -> np.ndarray:
    nodes, weights = leggauss(16)
    h = target.length / n_grid
    x = (np.arange(n_grid)[:, None] + (nodes[None, :] + 1) / 2) * h
    probabilities = initial_profile(x, context) @ weights * h / 2
    return probabilities / probabilities.sum()


def discrete_dk(context: Context, target: TargetSpec, times: Sequence[float], dt: float,
                seed: int, n_paths: int = 1, n_grid: int = 32,
                initial_particles: np.ndarray | None = None) -> TrajectoryBatch:
    """Centered finite-volume DK; shared face noise conserves total mass.

    Face flux noise has amplitude sqrt(2D max(rho_face,0)/(N h)). A raw cell
    mass below -1e-12 invalidates the whole path immediately. Future snapshots
    are NaN; states are never clipped or renormalized.
    """
    started = time.perf_counter()
    times, save_steps, n_steps = _time_grid(times, dt)
    geometry = _geometry(context, target, n_grid, dt)
    h, force, b, _ = geometry
    projection = observation_matrix(target, n_grid)
    x = (sample_initial(context, n_paths, seeded_rng(seed, "initial", context.context_id))
         if initial_particles is None else np.asarray(initial_particles, dtype=float))
    if x.shape != (n_paths, context.n_particles) or not np.all(np.isfinite(x)):
        raise ValueError("Initial particles must have shape (paths, particles) and be finite")
    mass = np.asarray([np.bincount(np.floor((row % target.length) / h).astype(int),
                                   minlength=n_grid) / context.n_particles for row in x])
    initial_observed = mass @ projection.T
    exact_observed = observe_particles(x, target)
    initialization_seconds = time.perf_counter() - started
    initial_noise_trace = float(np.mean(np.sum((2 * context.diffusion / h)
                                              * (mass + np.roll(mass, -1, axis=1)) / (2 * h), axis=1) * 2))
    states = np.full((n_paths, len(times), target.n_cells), np.nan)
    valid = np.ones(n_paths, dtype=bool)
    failure_times = np.full(n_paths, np.nan)
    save = dict(zip(save_steps.tolist(), range(len(times)), strict=True))
    rng = seeded_rng(seed, "dk-face-noise", context.context_id)
    minimum_mass = float(mass.min())
    mass_error = 0.0
    for step in range(n_steps + 1):
        if step in save:
            states[valid, save[step]] = mass[valid] @ projection.T
        if step == n_steps:
            break
        active = np.flatnonzero(valid)
        if not len(active):
            break
        current = mass[active]
        face_density = (current + np.roll(current, -1, axis=1)) / (2 * h)
        velocity = current @ force.T
        flux = face_density * velocity - context.diffusion / h**2 * (np.roll(current, -1, axis=1) - current)
        noise_amplitude = np.sqrt(2 * context.diffusion / (context.n_particles * h)
                                  * np.maximum(face_density, 0))
        integrated_flux = dt * flux + np.sqrt(dt) * noise_amplitude * rng.normal(size=flux.shape)
        current = current + integrated_flux @ b.T
        mass[active] = current
        minimum_mass = min(minimum_mass, float(current.min()))
        mass_error = max(mass_error, float(np.max(np.abs(current.sum(axis=1) - 1))))
        failed = np.any(current < -1e-12, axis=1) | np.any(~np.isfinite(current), axis=1)
        valid[active[failed]] = False
        failure_times[active[failed]] = (step + 1) * dt
    return TrajectoryBatch(times, states, target, valid, failure_times, diagnostics={
        "context_id": context.context_id, "method": "discrete-dk", "seed": int(seed),
        "trajectory_ids": [stable_hash([context.context_id, int(seed), i])[:24] for i in range(n_paths)],
        "dt": dt, "n_grid": n_grid, "n_particles": context.n_particles,
        "initialization_seconds": initialization_seconds,
        "simulation_seconds": time.perf_counter() - started - initialization_seconds,
        "total_seconds": time.perf_counter() - started,
        "mass_error": mass_error, "minimum_mass": minimum_mass,
        "invalid_path_frequency": float(1 - valid.mean()),
        "initial_projection_rms": float(np.sqrt(np.mean((initial_observed - exact_observed)**2))),
        "initial_projection_max": float(np.max(np.abs(initial_observed - exact_observed))),
        "initial_scaled_noise_trace": initial_noise_trace,
        "noise_convention": "shared-face amplitude sqrt(2D*max(rho_face,0)/(N*h))",
    })


def meanfield_and_fluctuations(context: Context, target: TargetSpec, query_bank: QueryBank,
                               dt: float, n_grid: int = 32
                               ) -> tuple[StatisticalPrediction, StatisticalPrediction]:
    """Finite-volume mean field and linearized Gaussian fluctuation predictions.

    The scaled initial covariance is diag(p)-p p^T for IID particles. Both drift
    Jacobian and shared-face noise are evaluated on the evolving mean field.
    Euler covariance propagation uses T C T^T + dt Q, preserving positivity.
    Saved cross covariances propagate causally to construct the joint query law.
    """
    started = time.perf_counter()
    times, save_steps, n_steps = _time_grid(query_bank.times, dt)
    geometry = _geometry(context, target, n_grid, dt)
    h, _, b, _ = geometry
    projection = observation_matrix(target, n_grid)
    observable = query_bank.weights(target.n_cells) @ projection
    mass = _initial_cell_probabilities(context, target, n_grid)
    initial_mass = mass.copy()
    covariance = np.diag(mass) - np.outer(mass, mass)
    initial_covariance = covariance.copy()
    # Direct continuous-profile observation is the initialization projection oracle.
    x = np.arange(8192) * target.length / 8192
    direct_initial = (initial_profile(x, context)[:, None] * spline_weights(x, target)).mean(axis=0) * target.length
    initialization_error = float(np.linalg.norm(projection @ mass - direct_initial))
    initialization_seconds = time.perf_counter() - started
    d = query_bank.dimension
    width = observable.shape[0]
    means = np.empty(d)
    joint_covariance = np.zeros((d, d))
    cross_covariances: list[np.ndarray] = []
    save = dict(zip(save_steps.tolist(), range(len(times)), strict=True))
    minimum_mass = float(mass.min())
    initial_noise_trace = None
    for step in range(n_steps + 1):
        if step in save:
            index = save[step]
            current_slice = slice(index * width, (index + 1) * width)
            means[current_slice] = observable @ mass
            joint_covariance[current_slice, current_slice] = observable @ covariance @ observable.T
            for previous, cross in enumerate(cross_covariances):
                previous_slice = slice(previous * width, (previous + 1) * width)
                block = observable @ cross @ observable.T
                joint_covariance[previous_slice, current_slice] = block
                joint_covariance[current_slice, previous_slice] = block.T
            cross_covariances.append(covariance.copy())
        if step == n_steps:
            break
        drift, jacobian, face_density = _drift_and_jacobian(mass, context, geometry)
        noise = (b * (2 * context.diffusion / h * face_density)[None, :]) @ b.T
        if initial_noise_trace is None:
            initial_noise_trace = float(np.trace(noise))
        transition = np.eye(n_grid) + dt * jacobian
        covariance = transition @ covariance @ transition.T + dt * noise
        covariance = (covariance + covariance.T) / 2
        cross_covariances = [cross @ transition.T for cross in cross_covariances]
        mass = mass + dt * drift
        minimum_mass = min(minimum_mass, float(mass.min()))
        if mass.min() < -1e-12 or not np.all(np.isfinite(mass)):
            raise RuntimeError("Mean-field path invalid; refine grid/time step")
    diagnostics = {
        "n_grid": n_grid, "dt": dt, "initialization_seconds": initialization_seconds,
        "total_seconds": time.perf_counter() - started,
        "mass_error": float(abs(mass.sum() - 1)), "minimum_mass": minimum_mass,
        "initial_projection_l2": initialization_error,
        "initial_scaled_covariance_trace": float(np.trace(initial_covariance)),
        "initial_scaled_covariance_mass_error": float(np.max(np.abs(initial_covariance.sum(axis=0)))),
        "initial_scaled_noise_trace": initial_noise_trace,
        "initial_cell_probabilities": initial_mass.tolist(),
    }
    deterministic = StatisticalPrediction(means, np.zeros_like(joint_covariance), target, query_bank,
                                          context.context_id, {**diagnostics, "method": "mean-field"})
    linearized = StatisticalPrediction(means.copy(), joint_covariance, target, query_bank,
                                       context.context_id, {**diagnostics, "method": "linearized-fluctuations"})
    return deterministic, linearized
