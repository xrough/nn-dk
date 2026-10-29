"""Conditional nonlinear LNA marginals with batched sparse/rank-four RK4.

Every supplied histogram has its own nonlinear mean and conditional covariance
ODE. Batching does not replace their integration by analytic IID propagation.
Only same-time linear-payoff marginals are returned; this is the unstopped LNA,
not the stopped native law. The historical dense implementation stays unchanged.
"""
from __future__ import annotations

import os
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402

from analysis.native_gaussian_baseline import LAW, _geometry  # noqa: E402

IMPLEMENTATION = "conditional-batched-sparse-rank4-rk4-v1"


def _rhs(mean, covariance, context, spec, running):
    """Augmented mean/covariance derivative; no dense Jacobian is required."""
    g = spec.n_cells
    masses = mean[:, :g]
    if not np.isfinite(masses).all() or np.any(masses < 0):
        raise ValueError("Nonfinite or negative deterministic mean; no clipping")
    moments, reconstruction, scales, _, _, _ = _geometry(
        g, spec.length, context.kappa, context.a2)
    velocity = ((masses @ moments)*scales) @ reconstruction
    following = np.roll(masses, -1, axis=1)
    face = (masses+following)/(2*spec.h)
    flux = face*velocity-context.diffusion/spec.h**2*(following-masses)
    drift = np.concatenate((np.roll(flux, 1, axis=1)-flux, masses @ running.T), axis=1)
    diagonal = velocity/(2*spec.h)+context.diffusion/spec.h**2
    neighbor = velocity/(2*spec.h)-context.diffusion/spec.h**2
    force_left = reconstruction.T*scales

    def action(value):
        native = value[:, :g]
        force_action = force_left @ (moments.T @ native)
        flux_action = (diagonal[..., None]*native
                       +neighbor[..., None]*np.roll(native, -1, axis=1)
                       +face[..., None]*force_action)
        return np.concatenate((np.roll(flux_action, 1, axis=1)-flux_action,
                               running @ native), axis=1)

    derivative = action(covariance)+action(covariance.swapaxes(-1, -2)).swapaxes(-1, -2)
    amplitude2 = context.diffusion/spec.h**2*(masses+following)
    cells = np.arange(g)
    derivative[:, cells, cells] += amplitude2+np.roll(amplitude2, 1, axis=1)
    derivative[:, cells, (cells+1) % g] -= amplitude2
    derivative[:, (cells+1) % g, cells] -= amplitude2
    if not np.isfinite(drift).all() or not np.isfinite(derivative).all():
        raise ValueError("Nonfinite nonlinear Gaussian derivative; no result may be averaged")
    return drift, derivative


def _integrate(initials, context, spec, times, counts, steps, terminal, running):
    count, g = len(initials), spec.n_cells
    k, d = len(running), g+len(running)
    if (not np.isfinite(initials).all() or np.any(initials < 0)
            or not np.allclose(initials.sum(-1), 1., atol=1e-10, rtol=0)):
        raise ValueError("Finite nonnegative unit-mass histograms required; no normalization")
    mean = np.concatenate((initials.copy(), np.zeros((count, k))), axis=1)
    covariance = np.zeros((count, d, d))
    minimum = initials.min(-1).copy()
    means, covariances = [], []

    def rhs(mu, cov):
        np.minimum(minimum, mu[:, :g].min(-1), out=minimum)
        return _rhs(mu, cov, context, spec, running)

    for n_steps, dt in zip(counts, steps, strict=True):
        for _ in range(n_steps):
            first = rhs(mean, covariance)
            second = rhs(mean+dt/2*first[0], covariance+dt/2*first[1])
            third = rhs(mean+dt/2*second[0], covariance+dt/2*second[1])
            fourth = rhs(mean+dt*third[0], covariance+dt*third[1])
            mean = mean+dt/6*(first[0]+2*second[0]+2*third[0]+fourth[0])
            covariance = covariance+dt/6*(first[1]+2*second[1]+2*third[1]+fourth[1])
        np.minimum(minimum, mean[:, :g].min(-1), out=minimum)
        means.append(mean.copy())
        covariances.append(covariance.copy())
    means = np.stack(means, axis=1)
    covariances = np.stack(covariances, axis=1)
    if not np.isfinite(means).all() or not np.isfinite(covariances).all():
        raise ValueError("Nonfinite Gaussian approximation; no result may be averaged")
    output = np.concatenate((terminal, np.eye(k)), axis=1)
    projected_mean = means @ output.T
    projected_covariance = np.einsum("ai,btij,cj->btac", output, covariances, output,
                                     optimize=True)
    eigenvalues = np.linalg.eigvalsh(projected_covariance)
    minimum_eigenvalue = eigenvalues.min(axis=(1, 2))
    symmetry_error = np.abs(covariances-covariances.swapaxes(-1, -2)).max(axis=(1, 2, 3))
    covariance_scale = np.maximum(1., np.abs(covariances).max(axis=(1, 2, 3)))
    mass_error = np.abs(means[:, :, :g].sum(-1)-1.).max(-1)
    covariance_mass_error = np.abs(covariances[:, :, :g, :].sum(axis=2)).max(axis=(1, 2))
    valid = ((minimum >= 0) & (minimum_eigenvalue >= -1e-10*covariance_scale)
             & (symmetry_error <= 1e-10*covariance_scale)
             & (mass_error <= 1e-10)
             & (covariance_mass_error <= 1e-8*covariance_scale))
    return {"mean": projected_mean, "scaled_covariance": projected_covariance,
            "mean_augmented": means, "scaled_covariance_augmented": covariances,
            "minimum_deterministic_mass": minimum,
            "minimum_covariance_eigenvalue": minimum_eigenvalue,
            "covariance_symmetry_error": symmetry_error, "mean_mass_error": mass_error,
            "covariance_mass_error": covariance_mass_error, "valid": valid}


def conditional_gaussian_batch(initials, context, spec, times, step, terminal_weights,
                               running_weights, batch_size=64, return_native=False):
    """Evolve each histogram's conditional nonlinear LNA with zero initial Cov.

    Corresponding terminal/running rows describe the same K payoffs. Outputs
    have shapes [histogram,time,K] and [histogram,time,K,K]; covariance is scaled
    by N. Invalid rows stay aligned, have NaN outputs, and are explicitly flagged.
    Callers must reject an entire statistical estimate when any row is invalid;
    survivor averaging is forbidden. No density or covariance repair occurs.
    """
    started = time.perf_counter()
    initial = np.asarray(initials, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    terminal = np.asarray(terminal_weights, dtype=np.float64)
    running = np.asarray(running_weights, dtype=np.float64)
    g = spec.n_cells
    if (initial.ndim != 2 or initial.shape[1] != g or not len(initial)
            or times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or terminal.ndim != 2 or terminal.shape[1] != g or not len(terminal)
            or running.shape != terminal.shape or not np.isfinite(terminal).all()
            or not np.isfinite(running).all() or not np.isfinite(step) or step <= 0
            or not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1):
        raise ValueError("Histogram matrix, increasing nonnegative times, matched payoff rows and positive step/batch required")
    intervals = np.diff(np.concatenate(([0.], times)))
    counts = np.ceil(intervals/step).astype(int)
    steps = np.divide(intervals, counts, out=np.zeros_like(intervals), where=counts > 0)
    velocity_bound = abs(context.kappa)+2*abs(context.a2)
    if np.any(steps*(4*context.diffusion/spec.h**2+2*velocity_bound/spec.h) > 2.5):
        raise ValueError("RK4 covariance stability bound violated; refine step")
    n, q, k = len(initial), len(times), len(terminal)
    result = {"law": LAW, "times": times.copy(), "mean": np.full((n, q, k), np.nan),
              "scaled_covariance": np.full((n, q, k, k), np.nan)}
    if return_native:
        result["mean_augmented"] = np.full((n, q, g+k), np.nan)
        result["scaled_covariance_augmented"] = np.full((n, q, g+k, g+k), np.nan)
    metric_names = ("minimum_deterministic_mass", "minimum_covariance_eigenvalue",
                    "covariance_symmetry_error", "mean_mass_error", "covariance_mass_error")
    metrics = {name: np.full(n, np.nan) for name in metric_names}
    valid = np.zeros(n, dtype=bool)
    errors = [""]*n

    def execute(indices):
        try:
            value = _integrate(initial[indices], context, spec, times, counts, steps, terminal, running)
        except (ValueError, FloatingPointError) as error:
            if len(indices) > 1:
                middle = len(indices)//2
                execute(indices[:middle])
                execute(indices[middle:])
            else:
                errors[int(indices[0])] = str(error)
            return
        for name in metric_names:
            metrics[name][indices] = value[name]
        valid[indices] = value["valid"]
        for position, index in enumerate(indices):
            if not value["valid"][position]:
                errors[int(index)] = "Conditional LNA geometry invalid; no clipping or covariance repair"
        accepted = value["valid"]
        for name in ("mean", "scaled_covariance", *(
                ("mean_augmented", "scaled_covariance_augmented") if return_native else ())):
            result[name][indices[accepted]] = value[name][accepted]

    for start in range(0, n, batch_size):
        execute(np.arange(start, min(start+batch_size, n)))
    if return_native:
        result["mean_states"] = result["mean_augmented"][:, :, :g]
    result["diagnostics"] = {"implementation": IMPLEMENTATION, "valid": valid, "errors": errors,
        **metrics, "deterministic_boundary_encountered": (
            metrics["minimum_deterministic_mass"] <= spec.mass_stop),
        "boundary_diagnostic_valid": np.isfinite(metrics["minimum_deterministic_mass"]),
        "integration_steps": int(counts.sum()), "batch_size": batch_size, "step": float(step),
        "integration_seconds": time.perf_counter()-started, "initialization": "conditional-zero-covariance",
        "covariance_scale": "N-times-physical-covariance", "no_covariance_repair": True,
        "covariance_eigenvalue_scope": "projected-payoff-marginals",
        "covariance_mass_error_scope": "native-mass-sum-against-all-augmented-coordinates",
        "integration_steps_scope": "nominal-per-histogram; recursive retries are included in measured wall time",
        "no_density_repair": True, "no_rows_dropped": True, "stopped_dk_equivalence": False,
        "time_target": "continuous-linear-noise-ODE", "running_reward": "unstopped-integral",
        "joint_time_covariance_returned": False}
    return result
