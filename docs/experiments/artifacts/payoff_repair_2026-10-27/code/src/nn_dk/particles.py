"""Interacting particle references and analytic free-particle identities."""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .observations import observe_particles, spline_weights
from .types import (
    LENGTH,
    Context,
    QueryBank,
    StatisticalPrediction,
    TargetSpec,
    TrajectoryBatch,
    seeded_rng,
    stable_hash,
)


def _log_profile(x: np.ndarray, context: Context) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    modes = np.arange(1, len(context.cos_coeffs) + 1)
    return (np.cos(x[..., None] * modes) @ np.asarray(context.cos_coeffs)
            + np.sin(x[..., None] * modes) @ np.asarray(context.sin_coeffs))


def initial_profile(x: np.ndarray, context: Context) -> np.ndarray:
    """Normalized periodic exponential profile; spectral trapezoidal normalization."""
    grid = np.arange(8192) * (LENGTH / 8192)
    envelope = np.sum(np.hypot(context.cos_coeffs, context.sin_coeffs))
    normalization = LENGTH * np.exp(_log_profile(grid, context) - envelope).mean()
    return np.exp(_log_profile(np.asarray(x), context) - envelope) / normalization


def sample_initial(context: Context, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Exact IID rejection samples, avoiding an inverse-CDF discretization bias."""
    if n_paths < 1:
        raise ValueError("A positive path count is required")
    count = int(n_paths) * context.n_particles
    values = np.empty(count)
    envelope = float(np.sum(np.hypot(context.cos_coeffs, context.sin_coeffs)))
    filled = 0
    while filled < count:
        proposals = rng.uniform(0, LENGTH, max(32, 2 * (count - filled)))
        accept = np.log(rng.uniform(size=len(proposals))) <= _log_profile(proposals, context) - envelope
        selected = proposals[accept][:count - filled]
        values[filled:filled + len(selected)] = selected
        filled += len(selected)
    return values.reshape(n_paths, context.n_particles)


def particle_force(x: np.ndarray, context: Context) -> np.ndarray:
    """-V'*empirical measure, including the zero self force, in O(N K_int)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for k, coefficient in ((1, -context.kappa), (2, 2 * context.a2)):
        if coefficient:
            c, s = np.cos(k * x), np.sin(k * x)
            out += coefficient * (s * c.mean(axis=-1, keepdims=True)
                                  - c * s.mean(axis=-1, keepdims=True))
    return out


def pairwise_force(x: np.ndarray, context: Context) -> np.ndarray:
    """Quadratic correctness oracle for small particle systems."""
    d = np.asarray(x)[..., :, None] - np.asarray(x)[..., None, :]
    return (-context.kappa * np.sin(d) + 2 * context.a2 * np.sin(2 * d)).mean(axis=-1)


def _time_grid(times: Sequence[float], dt: float) -> tuple[np.ndarray, np.ndarray, int]:
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not len(t) or not np.all(np.isfinite(t)) or t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("Finite, nonnegative, increasing observation times required")
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("A positive finite time step is required")
    steps = np.rint(t / dt).astype(int)
    if not np.allclose(steps * dt, t, atol=1e-10, rtol=0):
        raise ValueError("Observation times must lie on the numerical time grid")
    return t, steps, int(steps[-1])


def simulate_particles(context: Context, target: TargetSpec, times: Sequence[float],
                       dt: float, seed: int, n_paths: int = 1,
                       initial_particles: np.ndarray | None = None,
                       brownian_increments: np.ndarray | None = None,
                       retain_particles: bool = False) -> TrajectoryBatch:
    """Euler--Maruyama on the torus; supplied increments are dW with variance dt.

    Only requested observation snapshots are retained. With retain_particles=True,
    the corresponding microscopic snapshots are returned in particle_states.
    """
    started = time.perf_counter()
    times, save_steps, n_steps = _time_grid(times, dt)
    x = (sample_initial(context, n_paths, seeded_rng(seed, "initial", context.context_id))
         if initial_particles is None else np.asarray(initial_particles, dtype=float).copy())
    if x.shape != (n_paths, context.n_particles) or not np.all(np.isfinite(x)):
        raise ValueError("Initial particles must have shape (n_paths, n_particles) and be finite")
    x %= target.length
    if brownian_increments is not None:
        brownian_increments = np.asarray(brownian_increments, dtype=float)
        if brownian_increments.shape != (n_paths, n_steps, context.n_particles) or not np.all(np.isfinite(brownian_increments)):
            raise ValueError("Brownian increments must have shape (paths, steps, particles) and be finite")
    initial_seconds = time.perf_counter() - started
    states = np.empty((n_paths, len(times), target.n_cells))
    particle_states = np.empty((n_paths, len(times), context.n_particles)) if retain_particles else None
    rng = seeded_rng(seed, "brownian", context.context_id)
    save = dict(zip(save_steps.tolist(), range(len(times)), strict=True))
    for step in range(n_steps + 1):
        if step in save:
            index = save[step]
            states[:, index] = observe_particles(x, target)
            if particle_states is not None:
                particle_states[:, index] = x
        if step == n_steps:
            break
        dw = (rng.normal(0, np.sqrt(dt), x.shape) if brownian_increments is None
              else brownian_increments[:, step])
        x = (x + dt * particle_force(x, context) + np.sqrt(2 * context.diffusion) * dw) % target.length
    return TrajectoryBatch(times, states, target, particle_states=particle_states, diagnostics={
        "context_id": context.context_id, "method": "particle-monte-carlo", "seed": int(seed),
        "trajectory_ids": [stable_hash([context.context_id, int(seed), i])[:24] for i in range(n_paths)],
        "dt": dt, "n_particles": context.n_particles, "initialization_seconds": initial_seconds,
        "simulation_seconds": time.perf_counter() - started - initial_seconds,
        "total_seconds": time.perf_counter() - started,
        "mass_error": float(np.max(np.abs(states.sum(axis=-1) - 1))),
        "minimum_mass": float(states.min()), "invalid_path_frequency": 0.0,
    })


def coupled_refinement(context: Context, target: TargetSpec, times: Sequence[float],
                       coarse_dt: float, seed: int, n_paths: int = 1) -> dict[str, TrajectoryBatch]:
    """Return coarse/half/quarter references with coupled initial states and dW.

    Fine increments are summed in groups of two/four. Generation is per path to
    bound memory independently of n_paths; no path is selected by its outcome.
    """
    _, _, fine_steps = _time_grid(times, coarse_dt / 4)
    outputs: dict[str, list[TrajectoryBatch]] = {key: [] for key in ("coarse", "half", "quarter")}
    for path in range(n_paths):
        path_seed = int(seeded_rng(seed, "refinement", path).integers(0, 2**63 - 1))
        x0 = sample_initial(context, 1, seeded_rng(path_seed, "initial", context.context_id))
        increments = seeded_rng(path_seed, "coupled-increments").normal(
            0, np.sqrt(coarse_dt / 4), (1, fine_steps, context.n_particles))
        for key, group in (("coarse", 4), ("half", 2), ("quarter", 1)):
            dw = increments.reshape(1, fine_steps // group, group, context.n_particles).sum(axis=2)
            outputs[key].append(simulate_particles(context, target, times, coarse_dt * group / 4,
                                                   path_seed, 1, x0, dw))
    result = {}
    for key, batches in outputs.items():
        diagnostics = {"context_id": context.context_id, "method": "coupled-refinement", "seed": seed,
                       "dt": batches[0].diagnostics["dt"],
                       "trajectory_ids": [b.diagnostics["trajectory_ids"][0] for b in batches],
                       "total_seconds": sum(b.diagnostics["total_seconds"] for b in batches)}
        result[key] = TrajectoryBatch(np.asarray(times), np.concatenate([b.states for b in batches]),
                                      target, diagnostics=diagnostics)
    return result


def free_particle_statistics(context: Context, target: TargetSpec, query_bank: QueryBank,
                             quadrature: int = 16384) -> StatisticalPrediction:
    """Heat-semigroup means/covariances for the actual filtered observables.

    Spectral quadrature resolves spline aliases rather than replacing the spline
    interpolation by a single Fourier mode. Covariances are scaled by N.
    """
    if context.kappa != 0 or context.a2 != 0:
        raise ValueError("The heat-semigroup oracle requires noninteracting particles")
    if quadrature < 4 * target.n_cells:
        raise ValueError("Heat quadrature must resolve the observation functions")
    x = np.arange(quadrature) * target.length / quadrature
    f = spline_weights(x, target) @ query_bank.weights(target.n_cells).T
    wave = 2 * np.pi * np.fft.fftfreq(quadrature, d=target.length / quadrature)
    rho_hat = np.fft.fft(initial_profile(x, context))
    f_hat = np.fft.fft(f, axis=0)
    dx = target.length / quadrature
    means, densities = [], []
    for t in query_bank.times:
        rho = np.fft.ifft(rho_hat * np.exp(-context.diffusion * wave**2 * t)).real
        densities.append(rho)
        means.append(dx * (rho @ f))
    mean = np.concatenate(means)
    width = f.shape[-1]
    covariance = np.empty((len(mean), len(mean)))
    for i, ti in enumerate(query_bank.times):
        for j in range(i, len(query_bank.times)):
            lag = query_bank.times[j] - ti
            heat_f = np.fft.ifft(f_hat * np.exp(-context.diffusion * wave[:, None] ** 2 * lag), axis=0).real
            block = dx * (f.T @ (densities[i][:, None] * heat_f)) - np.outer(means[i], means[j])
            covariance[i * width:(i + 1) * width, j * width:(j + 1) * width] = block
            covariance[j * width:(j + 1) * width, i * width:(i + 1) * width] = block.T
    return StatisticalPrediction(mean, covariance, target, query_bank, context.context_id,
                                 {"method": "free-particle-heat-identity", "quadrature": quadrature})
