"""Differentiable finite-volume equations for noise-conditioned density paths.

Physical Brownian increments are supplied by the caller.  There is no hidden
random draw, density clipping, renormalization, or replacement of an exiting
proposal by an absorbed Gaussian increment.  Heat and additive-noise controls
are linear, unstopped equations; only ``dk`` uses the declared boundary rule.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .native_types import NativeDKSpec
from .types import Context

PATH_PROBLEMS = ("heat", "additive", "dk")


def _double(value, *, like: torch.Tensor | None = None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64,
                           device=None if like is None else like.device)


def _check_problem(problem: str, context: Context):
    if problem not in PATH_PROBLEMS:
        raise ValueError(f"Unknown path problem {problem!r}; use {PATH_PROBLEMS}")
    if problem != "dk" and (context.kappa != 0 or context.a2 != 0):
        raise ValueError("Heat/additive calibration problems require zero interaction")


def _check_step(dt: float):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("A finite positive timestep is required")


def divergence(face_values: torch.Tensor) -> torch.Tensor:
    """Periodic face-to-cell divergence ``B f = roll(f,1) - f``."""
    return torch.roll(face_values, 1, dims=-1) - face_values


def path_coefficients(states, context: Context, spec: NativeDKSpec,
                      problem: str = "dk") -> tuple[torch.Tensor, torch.Tensor]:
    """Return native drift and face amplitudes BEFORE division by ``sqrt(N)``.

    Arbitrary leading dimensions are preserved.  DK origins must be nonnegative;
    the unstopped additive control is allowed to have negative components.
    """
    _check_problem(problem, context)
    states = _double(states)
    if states.ndim < 1 or states.shape[-1] != spec.n_cells:
        raise ValueError("States must have a final native finite-volume cell axis")
    if not bool(torch.isfinite(states).all()):
        raise ValueError("Finite coefficient origins are required")
    if problem == "dk" and bool((states < 0).any()):
        raise ValueError("DK coefficients require nonnegative origins, without clipping")
    following = torch.roll(states, -1, dims=-1)
    face_density = (states + following) / (2 * spec.h)
    flux = -context.diffusion / spec.h**2 * (following - states)
    if problem == "dk" and (context.kappa != 0 or context.a2 != 0):
        cells = torch.arange(spec.n_cells, dtype=torch.float64, device=states.device)
        difference = ((cells + 1)[:, None] - (cells + .5)[None, :]) * spec.h
        force = (-context.kappa * np.sinc(spec.h / (2 * np.pi)) * torch.sin(difference)
                 + 2 * context.a2 * np.sinc(spec.h / np.pi) * torch.sin(2 * difference))
        flux = flux + face_density * (states @ force.T)
    drift = divergence(flux)
    if problem == "heat":
        amplitudes = torch.zeros_like(states)
    elif problem == "additive":
        amplitude = math.sqrt(2 * context.diffusion / (spec.n_cells * spec.h**2))
        amplitudes = torch.full_like(states, amplitude)
    else:
        amplitudes = torch.sqrt((2 * context.diffusion / spec.h) * face_density)
    return drift, amplitudes


def _boundary(current: torch.Tensor, proposal: torch.Tensor, spec: NativeDKSpec):
    """Piecewise differentiable first segment intersection; densities are not clipped."""
    already = (current <= spec.mass_stop).any(-1)
    exiting = proposal <= spec.mass_stop
    descending = exiting & (proposal < current)
    # A safe denominator avoids a masked 0/0 whose backward pass could be NaN.
    denominator = torch.where(descending, current - proposal, torch.ones_like(current))
    fractions = torch.where(descending, (current - spec.mass_stop) / denominator,
                            torch.ones_like(current))
    first = fractions.min(-1).values.clamp(0, 1)
    first = torch.where(already, torch.zeros_like(first), first)
    crossed = already | exiting.any(-1)
    result = current + first[..., None] * (proposal - current)
    return result, crossed, first


def _path_inputs(initial, proposals, spec: NativeDKSpec, dt: float):
    _check_step(dt)
    initial = _double(initial)
    proposals = _double(proposals, like=initial)
    if (initial.ndim != 2 or initial.shape[-1] != spec.n_cells
            or proposals.ndim != 3 or proposals.shape[0] != initial.shape[0]
            or proposals.shape[-1] != spec.n_cells or proposals.shape[1] < 1
            or initial.shape[0] < 1):
        raise ValueError("Initial states [paths,cells] and proposals [paths,steps,cells] required")
    if not bool(torch.isfinite(initial).all() and torch.isfinite(proposals).all()):
        raise ValueError("Initial states and proposals must be finite")
    return initial, proposals


def torch_stop_paths(initial, raw_paths, spec: NativeDKSpec, dt: float, *,
                     stopping: bool = True) -> dict[str, torch.Tensor]:
    """Freeze absolute neural proposals at the first crossing of their Euler polygon.

    ``raw_paths`` excludes the initial state.  Every original proposal is retained,
    including the crossing proposal and outputs after absorption. ``active`` is
    the PRE-step mask. Stopping-time gradients are piecewise defined away from
    ties and changes in which cell first reaches the boundary.
    """
    initial, raw_paths = _path_inputs(initial, raw_paths, spec, dt)
    if stopping and bool((initial < 0).any()):
        raise ValueError("Stopped DK initial states must be nonnegative")
    current = initial
    stopped_at = torch.full_like(initial[:, 0], torch.inf)
    if stopping:
        stopped_at = torch.where((initial <= spec.mass_stop).any(-1),
                                 torch.zeros_like(stopped_at), stopped_at)
    active = torch.isinf(stopped_at)
    states, masks, fractions = [initial], [], []
    for step in range(raw_paths.shape[1]):
        masks.append(active)
        proposal = raw_paths[:, step]
        if stopping:
            boundary, crossed, fraction = _boundary(current, proposal, spec)
            current = torch.where(active[:, None], boundary, current)
            newly = active & crossed
            stopped_at = torch.where(newly, (step + fraction) * dt, stopped_at)
            fractions.append(torch.where(active, fraction, torch.zeros_like(fraction)))
            active = active & ~crossed
        else:
            current = proposal
            fractions.append(torch.ones_like(stopped_at))
        states.append(current)
    return {"states": torch.stack(states, 1), "raw_proposals": raw_paths,
            "proposals": raw_paths,
            "active": torch.stack(masks, 1), "stopped_at": stopped_at,
            "crossing_fraction": torch.stack(fractions, 1)}


def torch_euler(initial, increments, context: Context, spec: NativeDKSpec,
                dt: float, problem: str = "dk") -> dict[str, torch.Tensor]:
    """Differentiable native Euler reference using caller-supplied physical noise."""
    _check_problem(problem, context)
    initial, increments = _path_inputs(initial, increments, spec, dt)
    velocity_bound = abs(context.kappa) + 2 * abs(context.a2)
    if dt * (2 * context.diffusion / spec.h**2 + velocity_bound / spec.h) > 1 + 1e-12:
        raise ValueError("Explicit finite-volume diffusion/advection CFL condition violated")
    if problem == "dk" and velocity_bound * spec.h > 2 * context.diffusion + 1e-12:
        raise ValueError("Centered DK advection violates the cell Peclet condition")
    if problem == "dk" and bool((initial < 0).any()):
        raise ValueError("Stopped DK initial states must be nonnegative")
    current = initial
    stopped_at = torch.full_like(initial[:, 0], torch.inf)
    if problem == "dk":
        stopped_at = torch.where((initial <= spec.mass_stop).any(-1),
                                 torch.zeros_like(stopped_at), stopped_at)
    active = torch.isinf(stopped_at)
    states, proposals, masks, fractions = [initial], [], [], []
    for step in range(increments.shape[1]):
        masks.append(active)
        # Coefficients of absorbed paths are not used. A safe evaluation origin
        # prevents a masked sqrt'(0) from introducing NaNs in their backward pass.
        origins = torch.where(active[:, None], current,
                              torch.full_like(current, 1 / spec.n_cells))
        drift, amplitudes = path_coefficients(origins, context, spec, problem)
        proposal = current + dt * drift + divergence(
            amplitudes * increments[:, step] / math.sqrt(context.n_particles))
        proposal = torch.where(active[:, None], proposal, current)
        proposals.append(proposal)
        if problem == "dk":
            boundary, crossed, fraction = _boundary(current, proposal, spec)
            current = torch.where(active[:, None], boundary, current)
            stopped_at = torch.where(active & crossed, (step + fraction) * dt, stopped_at)
            fractions.append(torch.where(active, fraction, torch.zeros_like(fraction)))
            active = active & ~crossed
        else:
            current = proposal
            fractions.append(torch.ones_like(stopped_at))
        states.append(current)
    raw = torch.stack(proposals, 1)
    return {"states": torch.stack(states, 1), "raw_proposals": raw, "proposals": raw,
            "active": torch.stack(masks, 1), "stopped_at": stopped_at,
            "crossing_fraction": torch.stack(fractions, 1)}


def ito_residual(initial, proposals, increments, context: Context, spec: NativeDKSpec,
                 dt: float, problem: str = "dk") -> dict[str, torch.Tensor]:
    """Left-point Itô residual on ALL active-start proposals, including exits.

    The terminal segment uses its full uncensored Euler proposal. It must not be
    replaced by the interpolated stopped state or a fractionally scaled Brownian
    increment. Subsequent absorbed outputs have zero residual and zero weight;
    every path remains in the outer path average. ``cumulative`` is the integrated
    proposal residual, and ``local`` permits diagnosis of temporal cancellation.
    """
    _check_problem(problem, context)
    packet = torch_stop_paths(initial, proposals, spec, dt, stopping=problem == "dk")
    states = packet["states"]
    increments = _double(increments, like=states)
    if increments.shape != packet["raw_proposals"].shape or not bool(
            torch.isfinite(increments).all()):
        raise ValueError("Finite Brownian increments must match all uncensored proposals")
    origins = states[:, :-1]
    evaluation_origins = torch.where(packet["active"][..., None], origins,
                                     torch.full_like(origins, 1 / spec.n_cells))
    drift, amplitudes = path_coefficients(evaluation_origins, context, spec, problem)
    drift = torch.where(packet["active"][..., None], drift, torch.zeros_like(drift))
    amplitudes = torch.where(packet["active"][..., None], amplitudes,
                             torch.zeros_like(amplitudes))
    stochastic = divergence(amplitudes * increments / math.sqrt(context.n_particles))
    local = packet["raw_proposals"] - origins - dt * drift - stochastic
    local = torch.where(packet["active"][..., None], local, torch.zeros_like(local))
    return {**packet, "local": local, "cumulative": torch.cumsum(local, dim=1),
            "predictable_drift": drift, "predictable_amplitudes": amplitudes}


def aggregate_brownian(increments, factor: int):
    """Sum fine physical increments into coupled coarse increments, without resampling."""
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError("A positive integer Brownian aggregation factor is required")
    if increments.ndim != 3 or increments.shape[1] % factor:
        raise ValueError("Brownian increments [paths,steps,faces] must divide into complete blocks")
    shape = (increments.shape[0], increments.shape[1] // factor, factor, increments.shape[2])
    blocks = increments.reshape(shape)
    return blocks.sum(dim=2) if isinstance(increments, torch.Tensor) else blocks.sum(axis=2)


def linear_moments(initial_mean, initial_covariance, times, context: Context,
                   spec: NativeDKSpec, problem: str = "additive", *,
                   scheme: str = "continuous", dt: float | None = None) -> dict[str, np.ndarray]:
    """Full-state mean and joint covariance of the linear heat/additive controls.

    Covariances have physical scaling, including ``1/N`` for physical forcing.
    ``joint_covariance[i,:,j,:]`` includes propagated random initial-state
    covariance. The Gaussian control is unstopped, so this is not a DK reference.
    """
    _check_problem(problem, context)
    if problem == "dk" or scheme not in ("continuous", "euler"):
        raise ValueError("Linear heat/additive moments require continuous or Euler scheme")
    mean = np.asarray(initial_mean, dtype=np.float64)
    covariance = np.asarray(initial_covariance, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    g = spec.n_cells
    if (mean.shape != (g,) or covariance.shape != (g, g) or times.ndim != 1
            or not len(times) or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or not all(np.all(np.isfinite(x)) for x in (mean, covariance, times))
            or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12)):
        raise ValueError("Finite mean [cells], symmetric covariance [cells,cells], increasing times required")
    if np.linalg.eigvalsh(covariance).min() < -1e-12:
        raise ValueError("Initial covariance must be positive semidefinite")
    alpha = context.diffusion / spec.h**2
    identity = np.eye(g)
    generator = alpha * (np.roll(identity, 1, axis=0) + np.roll(identity, -1, axis=0)
                         - 2 * identity)
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    # The constant mode is analytically zero; avoid numerical division near zero.
    eigenvalues[-1] = 0
    noise = (-2 * generator / (g * context.n_particles)
             if problem == "additive" else np.zeros_like(generator))
    noise_modes = eigenvectors.T @ noise @ eigenvectors
    if scheme == "euler":
        if dt is None:
            raise ValueError("Euler moments require a timestep")
        _check_step(dt)
        if 2 * alpha * dt > 1 + 1e-12:
            raise ValueError("Explicit heat CFL condition violated")
        steps = np.rint(times / dt).astype(int)
        if not np.allclose(steps * dt, times, rtol=0, atol=1e-12):
            raise ValueError("Euler moment times must be integer multiples of dt")
        multipliers = 1 + dt * eigenvalues

        def propagator(t):
            return (eigenvectors * multipliers**int(round(t / dt))) @ eigenvectors.T

        def integrated_noise(t):
            count = int(round(t / dt))
            products = multipliers[:, None] * multipliers[None, :]
            total = np.zeros_like(products)
            power = np.ones_like(products)
            for _ in range(count):
                total += power
                power *= products
            return eigenvectors @ (dt * total * noise_modes) @ eigenvectors.T
    else:
        def propagator(t):
            return (eigenvectors * np.exp(t * eigenvalues)) @ eigenvectors.T

        def integrated_noise(t):
            rates = eigenvalues[:, None] + eigenvalues[None, :]
            integral = np.full_like(rates, t)
            np.divide(np.expm1(t * rates), rates, out=integral, where=rates != 0)
            return eigenvectors @ (integral * noise_modes) @ eigenvectors.T

    means, covariances = [], []
    for t in times:
        transition = propagator(t)
        means.append(transition @ mean)
        covariances.append(transition @ covariance @ transition.T + integrated_noise(t))
    joint = np.empty((len(times), g, len(times), g))
    for i, ti in enumerate(times):
        for j in range(i, len(times)):
            block = covariances[i] @ propagator(times[j] - ti).T
            joint[i, :, j, :] = block
            joint[j, :, i, :] = block.T
    return {"times": times, "mean": np.asarray(means), "covariance": np.asarray(covariances),
            "joint_covariance": joint, "drift_matrix": generator, "noise_covariance": noise}
