"""Native finite-volume DK coefficients and stopped Euler trajectories.

The state is a vector of cell probability masses, never a spline projection.
Every trajectory is retained: at its first density boundary crossing the
piecewise-linear Euler polygon is stopped and its boundary state is frozen.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from functools import lru_cache

import numpy as np

from .native_types import DKCoefficients, NativeBatch, NativeDKSpec
from .observations import incidence_matrix
from .particles import _time_grid, sample_initial
from .types import Context, QueryBank, seeded_rng, stable_hash


@lru_cache(maxsize=64)
def _geometry(n_cells: int, length: float, kappa: float, a2: float):
    h = length / n_cells
    centers = (np.arange(n_cells) + 0.5) * h
    faces = (np.arange(n_cells) + 1) * h
    difference = faces[:, None] - centers[None, :]
    # Cell integrals of sin(k (face-x)) for a piecewise-constant density.
    force = (-kappa * np.sinc(h / (2 * np.pi)) * np.sin(difference)
             + 2 * a2 * np.sinc(h / np.pi) * np.sin(2 * difference))
    return force, incidence_matrix(n_cells)


def _check_masses(masses: np.ndarray, spec: NativeDKSpec) -> np.ndarray:
    masses = np.asarray(masses, dtype=np.float64)
    if masses.ndim < 1 or masses.shape[-1] != spec.n_cells:
        raise ValueError("Native masses must have a final finite-volume cell axis")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0):
        raise ValueError("Native coefficient origins must be finite and nonnegative")
    if not np.allclose(masses.sum(-1), 1, atol=1e-10, rtol=0):
        raise ValueError("Native masses must sum to one; no renormalization is performed")
    return masses


def _native_parts(masses: np.ndarray, context: Context, spec: NativeDKSpec):
    """The solver does not materialize a (paths,G,G) covariance each step."""
    force, _ = _geometry(spec.n_cells, spec.length, context.kappa, context.a2)
    following = np.roll(masses, -1, axis=-1)
    face_density = (masses + following) / (2 * spec.h)
    velocity = masses @ force.T
    flux = (face_density * velocity
            - context.diffusion / spec.h**2 * (following - masses))
    drift = np.roll(flux, 1, axis=-1) - flux
    amplitudes = np.sqrt((2 * context.diffusion / spec.h) * face_density)
    return drift, amplitudes, flux


def native_coefficients(masses: np.ndarray, context: Context,
                        spec: NativeDKSpec) -> DKCoefficients:
    """Return native drift and ``S S^T`` BEFORE division by particle count.

    The positive shared-face factor is
    ``sigma_i = sqrt(2 D rho_face_i / h)``, and ``S = B diag(sigma)``.
    Leading mass dimensions are preserved, including a single unbatched state.
    """
    masses = _check_masses(masses, spec)
    drift, amplitudes, flux = _native_parts(masses, context, spec)
    _, incidence = _geometry(spec.n_cells, spec.length, context.kappa, context.a2)
    covariance = np.einsum("ik,...k,jk->...ij", incidence, amplitudes**2, incidence)
    return DKCoefficients(drift, covariance, amplitudes, flux)


def initialize_masses(context: Context, spec: NativeDKSpec,
                      n_paths: int, seed: int) -> np.ndarray:
    """Exact IID profile samples followed by raw finite-volume histograms."""
    positions = sample_initial(context, n_paths,
                               seeded_rng(seed, "native-initial", context.context_id,
                                          spec.target_id))
    cells = np.floor((positions % spec.length) / spec.h).astype(np.int64)
    result = np.zeros((n_paths, spec.n_cells), dtype=np.float64)
    np.add.at(result, (np.arange(n_paths)[:, None], cells), 1 / context.n_particles)
    return result


def boundary_step(current: np.ndarray, proposal: np.ndarray, t: float,
                  dt: float, spec: NativeDKSpec):
    """Stop the Euler polygon at its FIRST cell-density boundary crossing.

    Return ``(state, crossed, stopping_time)`` with preserved leading axes.
    No state is clipped or renormalized. Already-boundary origins stop at t;
    noncrossing proposals have stopping time infinity. The integrator separately
    carries the absorbed flag so floating-point boundary rounding cannot revive
    a previously stopped path.
    """
    current, proposal = np.asarray(current, dtype=float), np.asarray(proposal, dtype=float)
    if (current.shape != proposal.shape or current.ndim < 1
            or current.shape[-1] != spec.n_cells
            or not np.all(np.isfinite(current)) or not np.all(np.isfinite(proposal))):
        raise ValueError("Finite matching cell-mass origins and proposals required")
    if not np.isfinite(t) or t < 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Finite nonnegative time and positive timestep required")
    already = np.any(current <= spec.mass_stop, axis=-1)
    exiting = proposal <= spec.mass_stop
    crossed = already | np.any(exiting, axis=-1)
    fraction = np.ones(current.shape, dtype=float)
    descending = exiting & (proposal < current)
    np.divide(current - spec.mass_stop, current - proposal,
              out=fraction, where=descending)
    first = np.min(np.where(descending, fraction, 1.0), axis=-1)
    first = np.where(already, 0.0, np.clip(first, 0.0, 1.0))
    result = current + first[..., None] * (proposal - current)
    stopped_at = np.where(crossed, t + dt * first, np.inf)
    return result, crossed, stopped_at


def _check_cfl(context: Context, spec: NativeDKSpec, step_size: float):
    velocity_bound = abs(context.kappa) + 2 * abs(context.a2)
    if (step_size * (2 * context.diffusion / spec.h**2 + velocity_bound / spec.h)
            > 1 + 1e-12):
        raise ValueError("Explicit native DK diffusion/advection CFL condition violated")
    if velocity_bound * spec.h > 2 * context.diffusion + 1e-12:
        raise ValueError("Centered native DK advection violates the cell Peclet condition")


def _families(context: Context, spec: NativeDKSpec, seed: int, n_paths: int,
              family_ids: Sequence[str] | None):
    families = (tuple(stable_hash(["native-family", context.context_id, spec.target_id,
                                  int(seed), i])[:24] for i in range(n_paths))
                if family_ids is None else tuple(family_ids))
    if len(families) != n_paths or len(set(families)) != n_paths:
        raise ValueError("One unique trajectory-family identifier per path is required")
    return families


def _learned_parts(coefficients: Callable, masses: np.ndarray, context: Context):
    result = coefficients(masses, context)
    if not isinstance(result, DKCoefficients):
        raise TypeError("Coefficient callable must return DKCoefficients")
    drift, amplitude = np.asarray(result.drift, dtype=float), np.asarray(result.face_amplitudes,
                                                                        dtype=float)
    if (drift.shape != masses.shape or amplitude.shape != masses.shape
            or not np.all(np.isfinite(drift)) or not np.all(np.isfinite(amplitude))
            or np.any(amplitude < 0)):
        raise ValueError("Finite conservative drift and nonnegative face amplitudes required")
    if not np.allclose(drift.sum(-1), 0, atol=1e-10, rtol=0):
        raise ValueError("Native learned drift must conserve mass")
    return drift, amplitude


def simulate_native(context: Context, spec: NativeDKSpec, times: Sequence[float],
                    step_size: float, seed: int, n_paths: int = 1,
                    initial_masses: np.ndarray | None = None,
                    brownian_increments: np.ndarray | None = None,
                    coefficients: Callable | None = None, ablation: str = "both",
                    family_ids: Sequence[str] | None = None) -> NativeBatch:
    """Simulate stopped native DK, retaining only requested snapshots.

    ``coefficients(masses, context)`` supplies learned native coefficients.
    Ablations select ``oracle``, learned ``drift``, learned ``noise``, or ``both``.
    With no coefficient callable, the native oracle is used. Supplied Brownian
    increments have shape ``(paths, steps, faces)`` and variance ``step_size``;
    otherwise face noise is generated one step at a time.
    """
    started = time.perf_counter()
    times, save_steps, n_steps = _time_grid(times, step_size)
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError("A positive integer path count is required")
    if ablation not in ("oracle", "drift", "noise", "both"):
        raise ValueError("Native ablation must be oracle, drift, noise, or both")
    if coefficients is None and ablation in ("drift", "noise"):
        raise ValueError("Learned coefficient ablations require a coefficient callable")
    _check_cfl(context, spec, step_size)
    masses = (initialize_masses(context, spec, n_paths, seed) if initial_masses is None
              else _check_masses(initial_masses, spec).copy())
    if masses.shape != (n_paths, spec.n_cells):
        raise ValueError("Initial native masses must have shape (paths, cells)")
    families = _families(context, spec, seed, n_paths, family_ids)
    if brownian_increments is not None:
        brownian_increments = np.asarray(brownian_increments, dtype=float)
        if (brownian_increments.shape != (n_paths, n_steps, spec.n_cells)
                or not np.all(np.isfinite(brownian_increments))):
            raise ValueError("Face Brownian increments must have shape (paths, steps, cells)")
    initialized = time.perf_counter() - started
    states = np.empty((n_paths, len(times), spec.n_cells), dtype=float)
    stopped_at = np.where(np.any(masses <= spec.mass_stop, axis=-1), 0.0, np.inf)
    active = np.isinf(stopped_at)
    rng = seeded_rng(seed, "native-face-noise", context.context_id, spec.target_id)
    save = dict(zip(save_steps.tolist(), range(len(times)), strict=True))
    minimum_proposal = float(masses.min())
    for step in range(n_steps + 1):
        if step in save:
            states[:, save[step]] = masses
        if step == n_steps:
            break
        if not np.any(active):
            # Include every frozen path in every remaining observation.
            for remaining_step, index in save.items():
                if remaining_step > step:
                    states[:, index] = masses
            break
        # Draw all paths, including stopped paths, to keep each stream unaffected
        # by another trajectory's stopping event.
        dw = (rng.normal(0, np.sqrt(step_size), masses.shape)
              if brownian_increments is None else brownian_increments[:, step])
        origins = masses[active]
        drift, amplitudes, _ = _native_parts(origins, context, spec)
        if coefficients is not None and ablation != "oracle":
            learned_drift, learned_noise = _learned_parts(coefficients, origins, context)
            if ablation in ("drift", "both"):
                drift = learned_drift
            if ablation in ("noise", "both"):
                amplitudes = learned_noise
        noise_flux = amplitudes * dw[active] / np.sqrt(context.n_particles)
        proposal = (origins + step_size * drift
                    + np.roll(noise_flux, 1, axis=-1) - noise_flux)
        minimum_proposal = min(minimum_proposal, float(proposal.min()))
        stopped_state, crossed, crossing_time = boundary_step(
            origins, proposal, step * step_size, step_size, spec)
        indices = np.flatnonzero(active)
        masses[indices] = stopped_state
        stopped_at[indices[crossed]] = crossing_time[crossed]
        active[indices[crossed]] = False
    elapsed = time.perf_counter() - started
    return NativeBatch(times, states, spec, context, stopped_at, families, {
        "method": "native-stopped-dk", "context_id": context.context_id,
        "target_id": spec.target_id, "seed": int(seed), "dt": float(step_size),
        "n_steps": n_steps, "ablation": "oracle" if coefficients is None else ablation,
        "stopping_probability": float(np.mean(np.isfinite(stopped_at))),
        "initial_stopping_probability": float(np.mean(stopped_at == 0)),
        "mass_error": float(np.max(np.abs(states.sum(-1) - 1))),
        "minimum_mass": float(states.min()), "minimum_uncensored_proposal_mass": minimum_proposal,
        "initialization_seconds": initialized, "simulation_seconds": elapsed - initialized,
        "total_seconds": elapsed, "noise_covariance_scaling": "SS^T before division by N",
        "all_paths_retained": True,
    })


def physical_observable_weights(spec: NativeDKSpec, queries: QueryBank) -> np.ndarray:
    """Cell averages of fixed continuous ``sqrt(2) cos/sin(k x)`` functions."""
    if max(queries.modes) >= spec.n_cells / 2:
        raise ValueError("Physical observable modes must be below the grid Nyquist frequency")
    centers = (np.arange(spec.n_cells) + 0.5) * spec.h
    return np.sqrt(2.0) * np.asarray([
        np.sinc(k * spec.h / (2 * np.pi)) * function(k * centers)
        for k in queries.modes for function in (np.cos, np.sin)])


def extract_observables(batch: NativeBatch, queries: QueryBank) -> np.ndarray:
    indices = []
    for t in queries.times:
        hits = np.flatnonzero(np.isclose(batch.times, t, atol=1e-10, rtol=0))
        if len(hits) != 1:
            raise ValueError(f"Query time {t} is absent from the native observation grid")
        indices.append(hits[0])
    values = batch.states[:, indices] @ physical_observable_weights(batch.spec, queries).T
    return values.reshape(len(batch.states), queries.dimension)


def coupled_native_refinement(context: Context, spec: NativeDKSpec, times: Sequence[float],
                              coarse_dt: float, seed: int, n_paths: int = 1,
                              family_ids: Sequence[str] | None = None) -> dict[str, NativeBatch]:
    """Coupled coarse/half/quarter paths with identical initial histograms.

    Fine increments are summed in pairs/fours. Fine Brownian storage is bounded
    by one trajectory, independently of ensemble size. Family IDs agree across
    levels, and every path remains in each result after stopping.
    """
    _time_grid(times, coarse_dt)
    _, _, fine_steps = _time_grid(times, coarse_dt / 4)
    _check_cfl(context, spec, coarse_dt)
    families = _families(context, spec, seed, n_paths, family_ids)
    outputs: dict[str, list[NativeBatch]] = {key: [] for key in ("coarse", "half", "quarter")}
    for path in range(n_paths):
        path_seed = int(seeded_rng(seed, "native-refinement", path).integers(0, 2**63 - 1))
        initial = initialize_masses(context, spec, 1, path_seed)
        increments = seeded_rng(path_seed, "native-coupled-increments").normal(
            0, np.sqrt(coarse_dt / 4), (1, fine_steps, spec.n_cells))
        for key, group in (("coarse", 4), ("half", 2), ("quarter", 1)):
            dw = increments.reshape(1, fine_steps // group, group, spec.n_cells).sum(axis=2)
            outputs[key].append(simulate_native(
                context, spec, times, coarse_dt * group / 4, path_seed, 1,
                initial_masses=initial, brownian_increments=dw, family_ids=(families[path],)))
    result = {}
    for key, batches in outputs.items():
        states = np.concatenate([batch.states for batch in batches])
        stopped = np.concatenate([batch.stopped_at for batch in batches])
        result[key] = NativeBatch(np.asarray(times), states, spec, context, stopped, families, {
            "method": "coupled-native-refinement", "dt": batches[0].diagnostics["dt"],
            "seed": int(seed), "all_paths_retained": True,
            "stopping_probability": float(np.mean(np.isfinite(stopped))),
            "mass_error": float(np.max(np.abs(states.sum(-1) - 1))),
            "minimum_mass": float(states.min()),
            "total_seconds": sum(batch.diagnostics["total_seconds"] for batch in batches),
        })
    return result


def linear_gaussian_statistics(context: Context, spec: NativeDKSpec, queries: QueryBank,
                               step_size: float | None = None) -> dict:
    """Exact first/two-time moments of an UNSTOPPED frozen-noise heat system.

    Its drift is the discrete heat operator; noise is frozen at uniform density.
    Initial covariance is the exact IID uniform histogram covariance divided by
    N. Initial histograms need not be Gaussian: these linear moment identities
    do not assume a Gaussian initial law. With ``step_size``, return the exact
    moments of the Euler recursion instead of the continuous linear SDE.
    This calibration law is distinct from stopped native DK.
    """
    if context.kappa != 0 or context.a2 != 0:
        raise ValueError("The frozen-noise heat calibration requires zero interaction")
    if np.any(context.cos_coeffs) or np.any(context.sin_coeffs):
        raise ValueError("The frozen-noise calibration uses IID uniform initial histograms")
    if step_size is not None:
        _time_grid(queries.times, step_size)
        _check_cfl(context, spec, step_size)
    p = np.full(spec.n_cells, 1 / spec.n_cells)
    _, incidence = _geometry(spec.n_cells, spec.length, 0.0, 0.0)
    operator = -context.diffusion / spec.h**2 * incidence @ incidence.T
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    zero = np.abs(eigenvalues) < 1e-12 * max(1.0, np.max(np.abs(eigenvalues)))
    eigenvalues[zero] = 0.0
    c0 = np.where(zero, 0.0, 1 / spec.n_cells)
    q = -2 * eigenvalues / spec.n_cells
    weights = physical_observable_weights(spec, queries)
    spectral_weights = weights @ eigenvectors

    def covariance_modes(t):
        if step_size is None:
            decay = np.exp(2 * eigenvalues * t)
            integral = np.full_like(eigenvalues, t)
            np.divide(np.expm1(2 * eigenvalues * t), 2 * eigenvalues,
                      out=integral, where=~zero)
            return decay * c0 + q * integral
        steps = int(round(t / step_size))
        transition = 1 + step_size * eigenvalues
        decay = transition ** (2 * steps)
        denominator = 1 - transition**2
        integral = np.full_like(eigenvalues, steps * step_size)
        np.divide(step_size * (1 - decay), denominator, out=integral,
                  where=np.abs(denominator) > 1e-14)
        return decay * c0 + q * integral

    width = len(weights)
    covariance = np.empty((queries.dimension, queries.dimension))
    for i, ti in enumerate(queries.times):
        c = covariance_modes(ti)
        for j in range(i, len(queries.times)):
            lag = queries.times[j] - ti
            propagation = (np.exp(eigenvalues * lag) if step_size is None else
                           (1 + step_size * eigenvalues) ** int(round(lag / step_size)))
            block = (spectral_weights * (c * propagation)[None, :]) @ spectral_weights.T
            si, sj = slice(i * width, (i + 1) * width), slice(j * width, (j + 1) * width)
            covariance[si, sj] = block
            covariance[sj, si] = block.T
    return {
        "mean": np.tile(weights @ p, len(queries.times)),
        "scaled_covariance": covariance,
        "metadata": {
            "law": "unstopped-linear-frozen-uniform-noise-heat-v1", "stopped": False,
            "initialization": "iid-uniform-histogram-moments",
            "initial_law_assumed_gaussian": False, "dt": step_size,
            "n_cells": spec.n_cells, "n_particles": context.n_particles,
            "scaled_initial_covariance_trace": float(1 - 1 / spec.n_cells),
        },
    }
