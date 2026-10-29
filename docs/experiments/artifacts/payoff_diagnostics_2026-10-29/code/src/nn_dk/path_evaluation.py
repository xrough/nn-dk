"""Full-state diagnostics for a causal, physical-noise-conditioned path operator.

This module deliberately does not infer path accuracy from four Fourier modes.
The primary strong metric uses every cell and every stored time under *shared*
initial states and physical Brownian increments. Its fixed normalization is the
uniform IID histogram fluctuation energy, not a held-out error denominator.

All uncertainty intervals are pointwise whole-trajectory bootstrap intervals.
An Euler-polygon comparison is not a certificate for continuous boundary hitting.
No failed or stopped trajectory is silently discarded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from .evaluation import json_safe
from .native import physical_observable_weights
from .native_types import NativeBatch, NativeDKSpec
from .types import Context, QueryBank, seeded_rng


def path_normalization(spec: NativeDKSpec, n_particles: int) -> dict[str, Any]:
    """Analytic fluctuation units for uniform IID finite-volume histograms.

    If ``m`` is an N-particle histogram, then
    ``E sum((m-1/G)**2)/h = (1-1/G)/(N*h)``. Consequently the squared
    normalized full-state density-L2 error is
    ``N*sum((prediction-reference)**2)/(1-1/G)``. The factor h cancels;
    it is retained below to identify the physical norm unambiguously.
    """
    if not isinstance(spec, NativeDKSpec):
        raise ValueError("A native finite-volume target is required")
    if not isinstance(n_particles, (int, np.integer)) or n_particles < 1:
        raise ValueError("A positive integer particle count is required")
    g = spec.n_cells
    trace = 1.0 - 1.0 / g
    cell_std = np.sqrt((1.0 / g) * (1.0 - 1.0 / g))
    return {
        "source": "analytic-uniform-iid-histogram-v1",
        "density_l2_squared": "sum(cell_mass_error**2)/h",
        "uniform_density_fluctuation_energy": trace / (n_particles * spec.h),
        "n_scaled_mass_covariance_trace": trace,
        "n_scaled_cell_standard_deviation": cell_std,
        "n_scaled_cell_covariance_scale": cell_std**2,
        "interpretation": "Fixed analytic units; every finite-volume cell is included",
    }


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _states(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value["states"]
    elif hasattr(value, "states"):
        value = value.states
    return _array(value)


def _finite_scalar(values: np.ndarray, operation: Callable) -> float | None:
    finite = values[np.isfinite(values)]
    return float(operation(finite)) if finite.size else None


def _times(times: Sequence[float], n_times: int) -> np.ndarray:
    result = _array(times)
    if (result.shape != (n_times,) or n_times < 2 or not np.all(np.isfinite(result))
            or result[0] != 0 or np.any(np.diff(result) <= 0)):
        raise ValueError("A finite strictly increasing time grid starting at zero is required")
    return result


def _flags(value: Any, n: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(n, dtype=bool)
    result = np.asarray(value)
    if result.shape != (n,) or result.dtype != np.bool_:
        raise ValueError(f"{name} must contain one Boolean failure flag per allocated path")
    return result


def _stops(value: Any, n: int, name: str) -> np.ndarray:
    result = np.full(n, np.inf) if value is None else _array(value)
    if result.shape != (n,) or np.any(np.isnan(result)) or np.any(result < 0):
        raise ValueError(f"{name} requires nonnegative stopping times or positive infinity")
    return result


def _families(value: Sequence[str] | None, n: int, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    result = tuple(value)
    if (len(result) != n or len(set(result)) != n
            or any(not isinstance(item, str) or not item for item in result)):
        raise ValueError(f"{name} requires a unique nonempty trajectory-family ID per path")
    return result


def _diagnostics(states: np.ndarray, times: np.ndarray, stops: np.ndarray,
                 failures: np.ndarray, spec: NativeDKSpec, require_nonnegative: bool,
                 stopping_declared: bool) -> dict[str, Any]:
    nonfinite = np.any(~np.isfinite(states), axis=(1, 2))
    negative = np.any(states < -1e-12, axis=(1, 2)) if require_nonnegative else np.zeros(
        len(states), dtype=bool)
    mass_errors = np.abs(states.sum(axis=-1) - 1)
    nonconservative = np.any(mass_errors > 1e-10, axis=1)
    not_frozen = np.zeros(len(states), dtype=bool)
    boundary_undershoot = np.zeros(len(states), dtype=bool)
    if stopping_declared:
        for i, stop in enumerate(stops):
            after = np.flatnonzero(times >= stop - 1e-12)
            if len(after) > 1 and np.any(np.abs(states[i, after] - states[i, after[0]]) > 1e-10):
                not_frozen[i] = True
            # An IID initial histogram may already be below the stopping threshold.
            # It must be frozen immediately, rather than repaired to the boundary.
            if stop != 0 and np.any(states[i] < spec.mass_stop - 1e-10):
                boundary_undershoot[i] = True
    invalid = failures | nonfinite | negative | nonconservative | not_frozen | boundary_undershoot
    return {
        "n_allocated_paths": len(states),
        "all_provided_paths_included": True,
        "n_declared_failures": int(failures.sum()),
        "n_nonfinite_paths": int(nonfinite.sum()),
        "n_negative_mass_paths": int(negative.sum()),
        "n_nonconservative_paths": int(nonconservative.sum()),
        "n_not_frozen_after_stop": int(not_frozen.sum()),
        "n_boundary_undershoot_paths": int(boundary_undershoot.sum()),
        "n_invalid_paths": int(invalid.sum()),
        "invalid_path_frequency": float(invalid.mean()),
        "minimum_mass": _finite_scalar(states, np.min),
        "minimum_density": _finite_scalar(states / spec.h, np.min),
        "maximum_mass_error": _finite_scalar(mass_errors, np.max),
        "stopping_declared": stopping_declared,
        "nonnegative_state_required": require_nonnegative,
    }


def _observation_indices(times: np.ndarray, requested: Sequence[float] | None) -> np.ndarray:
    if requested is None:
        # Avoid an implicit T*G by T*G covariance for a fine simulation mesh.
        requested = (float(times[-1]),)
    requested = _array(requested)
    if (requested.ndim != 1 or not len(requested) or not np.all(np.isfinite(requested))
            or requested[0] <= 0 or np.any(np.diff(requested) <= 0)):
        raise ValueError("Observation times must be positive, finite, unique, and increasing")
    indices = []
    for time in requested:
        found = np.flatnonzero(np.isclose(times, time, rtol=0, atol=1e-12))
        if len(found) != 1:
            raise ValueError(f"Observation time {time} is absent from the stored grid")
        indices.append(int(found[0]))
    return np.asarray(indices, dtype=np.int64)


def _moments(values: np.ndarray, n_particles: int) -> tuple[np.ndarray, np.ndarray]:
    means = values.mean(axis=0)
    centered = values - means
    covariance = n_particles * (centered.T @ centered) / (len(values) - 1)
    return means, covariance


def _ci(draws: np.ndarray) -> np.ndarray:
    return np.moveaxis(np.quantile(draws, [0.025, 0.975], axis=0), 0, -1)


def _same_time_blocks(covariance: np.ndarray, n_times: int, dimension: int) -> np.ndarray:
    return np.asarray([covariance[i * dimension:(i + 1) * dimension,
                                  i * dimension:(i + 1) * dimension]
                       for i in range(n_times)])


def _strong(path_energy: np.ndarray, indices: np.ndarray | None = None) -> dict[str, Any]:
    if indices is not None:
        path_energy = path_energy[indices]
    by_time = np.sqrt(path_energy.mean(axis=0))
    return {
        "rms_over_paths_and_stored_times": float(np.sqrt(path_energy[:, 1:].mean())),
        "rms_final_time": float(by_time[-1]),
        "maximum_time_rms": float(by_time.max()),
        "pathwise_supremum_rms": float(np.sqrt(path_energy.max(axis=1).mean())),
        "rms_by_time": by_time,
    }


def _weak(predicted: np.ndarray, reference: np.ndarray, std: np.ndarray,
          n_particles: int, observations: int, dimension: int,
          bootstrap_indices: list[np.ndarray]) -> dict[str, Any]:
    pm, pc = _moments(predicted, n_particles)
    rm, rc = _moments(reference, n_particles)
    mean_signed = np.sqrt(n_particles) * (pm - rm) / std
    covariance_signed = (pc - rc) / np.outer(std, std)
    result = {
        "prediction_mean": pm.reshape(observations, dimension),
        "reference_mean": rm.reshape(observations, dimension),
        "prediction_n_scaled_joint_covariance": pc,
        "reference_n_scaled_joint_covariance": rc,
        "prediction_n_scaled_same_time_covariance": _same_time_blocks(
            pc, observations, dimension),
        "reference_n_scaled_same_time_covariance": _same_time_blocks(
            rc, observations, dimension),
        "normalized_mean_signed_error": mean_signed.reshape(observations, dimension),
        "normalized_joint_covariance_signed_error": covariance_signed,
        "maximum_mean_normalized_error": float(np.max(np.abs(mean_signed))),
        "maximum_joint_covariance_normalized_error": float(np.max(np.abs(covariance_signed))),
        "mean_normalized_rmse": float(np.sqrt(np.mean(mean_signed**2))),
        "joint_covariance_normalized_rmse": float(np.sqrt(np.mean(covariance_signed**2))),
        "two_time_covariances_included": observations > 1,
        "scalar_wasserstein_normalized": np.asarray([
            np.sqrt(n_particles) * wasserstein_distance(predicted[:, i], reference[:, i])
            / std[i] for i in range(predicted.shape[1])
        ]).reshape(observations, dimension),
        "uncertainty": {"available": bool(bootstrap_indices)},
    }
    if bootstrap_indices:
        means, covariances = [], []
        for index in bootstrap_indices:
            bpm, bpc = _moments(predicted[index], n_particles)
            brm, brc = _moments(reference[index], n_particles)
            means.append(np.sqrt(n_particles) * (bpm - brm) / std)
            covariances.append((bpc - brc) / np.outer(std, std))
        mean_ci, covariance_ci = _ci(np.asarray(means)), _ci(np.asarray(covariances))
        result["uncertainty"].update({
            "mean_normalized_ci95": mean_ci.reshape(observations, dimension, 2),
            "joint_covariance_normalized_ci95": covariance_ci,
            "maximum_mean_ci_absolute_endpoint": float(np.max(np.abs(mean_ci))),
            "maximum_joint_covariance_ci_absolute_endpoint": float(
                np.max(np.abs(covariance_ci))),
            "maximum_mean_ci_halfwidth": float(np.max(np.diff(mean_ci, axis=-1) / 2)),
            "maximum_joint_covariance_ci_halfwidth": float(
                np.max(np.diff(covariance_ci, axis=-1) / 2)),
            "method": "paired whole-trajectory-family percentile bootstrap",
            "confidence_level": 0.95,
            "simultaneous_band": False,
            "wasserstein_uncertainty": "Descriptive empirical distance; not bootstrapped",
        })
    return result


def _reference_bound(reference_bias: dict[str, Any] | None, name: str,
                     uncertainty_budget: float) -> tuple[float | None, bool]:
    bound = None if reference_bias is None else reference_bias.get(name)
    valid = bool(reference_bias is not None and reference_bias.get("status") == "resolved"
                 and isinstance(bound, (int, float)) and not isinstance(bound, bool)
                 and np.isfinite(bound) and 0 <= bound <= uncertainty_budget)
    return float(bound) if valid else None, valid


def evaluate_paths(
    prediction: np.ndarray,
    reference: np.ndarray,
    times: Sequence[float],
    context: Context,
    spec: NativeDKSpec,
    *,
    stopped_at: np.ndarray | None = None,
    reference_stopped_at: np.ndarray | None = None,
    failures: np.ndarray | None = None,
    reference_failures: np.ndarray | None = None,
    family_ids: Sequence[str] | None = None,
    reference_family_ids: Sequence[str] | None = None,
    expected_paths: int | None = None,
    coupling_verified: bool = False,
    observation_times: Sequence[float] | None = None,
    n_bootstrap: int = 200,
    seed: int = 0,
    tolerance: float = 0.05,
    uncertainty_budget: float = 0.0125,
    reference_bias: dict[str, Any] | None = None,
    require_nonnegative: bool = True,
    comparison_kind: str = "neural-versus-same-step-euler",
) -> dict[str, Any]:
    """Compare paired complete density paths, never an average of survivors.

    Arrays have shape ``(paths, stored_times, cells)`` and include time zero.
    ``coupling_verified=True`` is a caller assertion that the *same* physical
    Brownian increments were used. Equal seeds alone are insufficient: the
    caller must verify increment arrays or their provenance hashes. This module
    additionally checks initial masses and ordered family IDs, when supplied.

    ``reference_bias`` needs separate ``strong_normalized_bound`` and
    ``weak_normalized_bound`` entries, and ``status='resolved'``. Its absence
    prevents an accuracy-certification claim even for identical arrays. A
    coarse/fine Euler difference is diagnostic evidence, not by itself such a
    bound. The strong 5% threshold is an explicit research target.
    """
    if not isinstance(context, Context) or not isinstance(spec, NativeDKSpec):
        raise ValueError("Native context and finite-volume specification are required")
    prediction, reference = _array(prediction), _array(reference)
    if (prediction.shape != reference.shape or prediction.ndim != 3
            or prediction.shape[-1] != spec.n_cells or prediction.shape[0] < 1):
        raise ValueError("Matching complete (paths, times, native cells) arrays are required")
    n, nt, g = prediction.shape
    grid = _times(times, nt)
    if (not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0
            or not np.isfinite(tolerance) or tolerance <= 0
            or not np.isfinite(uncertainty_budget) or uncertainty_budget <= 0):
        raise ValueError("Nonnegative bootstrap count and positive finite thresholds are required")
    if not isinstance(coupling_verified, (bool, np.bool_)):
        raise ValueError("coupling_verified must be an explicit Boolean")
    if expected_paths is not None and (
        not isinstance(expected_paths, (int, np.integer)) or expected_paths != n
    ):
        raise ValueError("The full allocated path pool is required; survivor-only arrays are forbidden")
    pf = _families(family_ids, n, "family_ids")
    rf = _families(reference_family_ids, n, "reference_family_ids")
    if (pf is None) != (rf is None) or (pf is not None and pf != rf):
        raise ValueError("Paired arrays require the same ordered trajectory-family IDs")
    stops = _stops(stopped_at, n, "stopped_at")
    rstops = _stops(reference_stopped_at, n, "reference_stopped_at")
    if (stopped_at is None) != (reference_stopped_at is None):
        raise ValueError("Prediction and reference must both declare the stopping convention")
    failed = _flags(failures, n, "failures")
    rfailed = _flags(reference_failures, n, "reference_failures")
    diagnostics = {
        "prediction": _diagnostics(prediction, grid, stops, failed, spec,
                                   require_nonnegative, stopped_at is not None),
        "reference": _diagnostics(reference, grid, rstops, rfailed, spec,
                                  require_nonnegative, reference_stopped_at is not None),
        "full_allocation_verified": expected_paths is not None,
        "ordered_family_ids_verified": pf is not None,
    }
    indices = _observation_indices(grid, observation_times)
    norm = path_normalization(spec, context.n_particles)
    result = {
        "target_id": spec.target_id,
        "context_id": context.context_id,
        "comparison_kind": comparison_kind,
        "scope": "Full finite-volume density paths on the declared stored-time grid",
        "reference_scope": (
            "Shared-noise Euler polygon; continuous stopping and continuum DK bias are uncertified"
        ),
        "n_independent_trajectory_families": n,
        "times": grid,
        "observation_times": grid[indices],
        "normalization": norm,
        "diagnostics": diagnostics,
        "coupling": {
            "physical_noise_verified_by_caller": bool(coupling_verified),
            "same_initial_states": bool(np.array_equal(prediction[:, 0], reference[:, 0])),
            "interpretation": "A strong path comparison requires common initial masses and Brownian increments",
        },
        "uncertainty_interpretation": (
            "Pointwise paired whole-family bootstrap intervals; exclude optimizer uncertainty, "
            "reference discretization bias, continuous hitting-time bias, and unobserved between-grid excursions"
        ),
    }
    invalid = diagnostics["prediction"]["n_invalid_paths"] + diagnostics["reference"]["n_invalid_paths"]
    if invalid:
        return json_safe({**result, "status": "invalid", "strong": None,
                          "weak_full_state": None, "weak_observables": None, "stopping": None,
                          "accuracy_gate": {
                              "status": "invalid",
                              "reason": "At least one path failed; no survivor-conditioned estimate is produced",
                          }})
    if coupling_verified and not result["coupling"]["same_initial_states"]:
        raise ValueError("Shared-noise strong comparison requires identical initial masses")
    if not coupling_verified:
        return json_safe({**result, "status": "unavailable", "strong": None,
                          "weak_full_state": None, "weak_observables": None, "stopping": None,
                          "accuracy_gate": {
                              "status": "unresolved", "reason": "Physical-noise coupling is unverified",
                          }})
    rng = seeded_rng(seed, "path-evaluation", spec.target_id, context.context_id)
    draws = [rng.integers(0, n, n) for _ in range(n_bootstrap)] if n >= 2 and n_bootstrap >= 2 else []
    energy = context.n_particles * np.sum((prediction - reference)**2, axis=-1) / (
        norm["n_scaled_mass_covariance_trace"])
    strong = _strong(energy)
    strong["normalization"] = "Uniform IID histogram density-L2 fluctuation units"
    strong["primary_metric"] = "pathwise_supremum_rms"
    strong["supremum_scope"] = "Maximum over stored times; excursions between them are not measured"
    strong["uncertainty"] = {"available": bool(draws)}
    if draws:
        rms_draws = np.asarray([_strong(energy, index)["pathwise_supremum_rms"] for index in draws])
        ci = _ci(rms_draws)
        strong["uncertainty"].update({
            "pathwise_supremum_rms_ci95": ci,
            "ci_halfwidth": float(np.diff(ci)[0] / 2),
            "method": "paired whole-trajectory-family percentile bootstrap",
            "confidence_level": 0.95,
            "simultaneous_band": False,
        })
    result["strong"] = strong
    stop_values = stops[:, None] <= grid[indices][None, :]
    rstop_values = rstops[:, None] <= grid[indices][None, :]
    stop_difference = stop_values.mean(axis=0) - rstop_values.mean(axis=0)
    stopping = {
        "all_stopped_paths_included": True,
        "prediction_cdf": stop_values.mean(axis=0),
        "reference_cdf": rstop_values.mean(axis=0),
        "cdf_difference": stop_difference,
        "prediction_n_stopped_by_horizon": int(np.count_nonzero(stops <= grid[-1])),
        "reference_n_stopped_by_horizon": int(np.count_nonzero(rstops <= grid[-1])),
        "prediction_initially_stopped": int(np.count_nonzero(stops == 0)),
        "reference_initially_stopped": int(np.count_nonzero(rstops == 0)),
        "prediction_restricted_mean_stop_time": float(np.minimum(stops, grid[-1]).mean()),
        "reference_restricted_mean_stop_time": float(np.minimum(rstops, grid[-1]).mean()),
        "restricted_stopping_time_absolute_difference_mean": float(
            np.abs(np.minimum(stops, grid[-1]) - np.minimum(rstops, grid[-1])).mean()),
        "stopping_time_wasserstein": float(wasserstein_distance(
            np.minimum(stops, grid[-1]), np.minimum(rstops, grid[-1]))),
        "time_distribution": "Restricted at the horizon; nonstopped paths are right-censored",
        "uncertainty": {"available": bool(draws)},
    }
    if draws:
        stopping["uncertainty"].update({
            "cdf_difference_ci95": _ci(np.asarray([
                stop_values[index].mean(axis=0) - rstop_values[index].mean(axis=0)
                for index in draws
            ])),
            "method": "paired whole-trajectory-family percentile bootstrap",
            "rare_event_warning": "Zero observed exits does not bound the probability of an unseen exit",
        })
    result["stopping"] = stopping
    if n < 2:
        return json_safe({**result, "status": "available", "weak_full_state": None,
                          "weak_observables": None, "accuracy_gate": {
                              "status": "unresolved", "reason": "At least two independent families are required",
                          }})
    observations = len(indices)
    full_std = np.full(observations * g, norm["n_scaled_cell_standard_deviation"])
    result["weak_full_state"] = _weak(prediction[:, indices].reshape(n, -1),
                                      reference[:, indices].reshape(n, -1), full_std,
                                      context.n_particles, observations, g, draws)
    queries = QueryBank(tuple(grid[indices]), (1, 2))
    weights = physical_observable_weights(spec, queries)
    mode_std = np.tile(np.repeat(np.sinc(np.asarray(queries.modes) / g), 2), observations)
    result["weak_observables"] = _weak(
        (prediction[:, indices] @ weights.T).reshape(n, -1),
        (reference[:, indices] @ weights.T).reshape(n, -1), mode_std,
        context.n_particles, observations, len(weights), draws)
    result["weak_observables"]["physical_test_functions"] = "Cell averages of sqrt(2) cos/sin modes 1 and 2"
    strong_bound, strong_ref = _reference_bound(reference_bias, "strong_normalized_bound",
                                               uncertainty_budget)
    weak_bound, weak_ref = _reference_bound(reference_bias, "weak_normalized_bound",
                                           uncertainty_budget)
    strong_upper = None if not draws else max(
        float(strong["uncertainty"]["pathwise_supremum_rms_ci95"][1]),
        strong["pathwise_supremum_rms"])
    weak_uncertainty = result["weak_full_state"]["uncertainty"]
    mode_uncertainty = result["weak_observables"]["uncertainty"]
    weak_upper = None if not draws else max(
        weak_uncertainty["maximum_mean_ci_absolute_endpoint"],
        weak_uncertainty["maximum_joint_covariance_ci_absolute_endpoint"],
        mode_uncertainty["maximum_mean_ci_absolute_endpoint"],
        mode_uncertainty["maximum_joint_covariance_ci_absolute_endpoint"],
        result["weak_full_state"]["maximum_mean_normalized_error"],
        result["weak_full_state"]["maximum_joint_covariance_normalized_error"],
        result["weak_observables"]["maximum_mean_normalized_error"],
        result["weak_observables"]["maximum_joint_covariance_normalized_error"])
    strong_sampling = bool(draws and strong["uncertainty"]["ci_halfwidth"] <= uncertainty_budget)
    weak_sampling = bool(draws and max(
        weak_uncertainty["maximum_mean_ci_halfwidth"],
        weak_uncertainty["maximum_joint_covariance_ci_halfwidth"],
        mode_uncertainty["maximum_mean_ci_halfwidth"],
        mode_uncertainty["maximum_joint_covariance_ci_halfwidth"]) <= uncertainty_budget)
    strong_inside = bool(strong_upper is not None and strong_upper + (strong_bound or 0) <= tolerance)
    weak_inside = bool(weak_upper is not None and weak_upper + (weak_bound or 0) <= tolerance)
    allocation = expected_paths is not None
    result["accuracy_gate"] = {
        "status": "resolved_within_tolerance" if (
            strong_ref and weak_ref and strong_sampling and weak_sampling
            and strong_inside and weak_inside and allocation
        ) else "unresolved",
        "strong_status": "resolved_within_tolerance" if (
            strong_ref and strong_sampling and strong_inside and allocation
        ) else "unresolved",
        "weak_status": "resolved_within_tolerance" if (
            weak_ref and weak_sampling and weak_inside and allocation
        ) else "unresolved",
        "research_tolerance": tolerance,
        "each_uncertainty_reference_budget": uncertainty_budget,
        "sampling_scope": (
            "Paired prediction-minus-reference contrast; separate marginal-law Monte Carlo "
            "precision is not certified by this gate"
        ),
        "strong_empirical_interval_within_tolerance": strong_inside,
        "weak_full_state_and_modes_empirical_intervals_within_tolerance": weak_inside,
        "strong_sampling_resolved": strong_sampling,
        "weak_sampling_resolved": weak_sampling,
        "strong_reference_bias_resolved": strong_ref,
        "weak_reference_bias_resolved": weak_ref,
        "full_allocation_verified": allocation,
        "strong_upper_plus_reference_bound": None if strong_upper is None or not strong_ref else strong_upper + strong_bound,
        "weak_upper_plus_reference_bound": None if weak_upper is None or not weak_ref else weak_upper + weak_bound,
        "reference_bias": reference_bias,
        "interpretation": (
            "Strong and weak gates use all cells; weak gates also include all declared mode contrasts. "
            "Weak intervals are pointwise paired contrasts. "
            "No continuum or continuous-boundary claim follows from same-step agreement. "
            "Stopping diagnostics remain separate and are not certified by this gate."
        ),
    }
    return json_safe({**result, "status": "available"})


def evaluate_native_paths(prediction: NativeBatch, reference: NativeBatch, **kwargs) -> dict[str, Any]:
    """NativeBatch adapter retaining the full allocated stopped path ensembles."""
    if not isinstance(prediction, NativeBatch) or not isinstance(reference, NativeBatch):
        raise ValueError("Two native path batches are required")
    prediction.spec.require_compatible(reference.spec)
    if prediction.context != reference.context or not np.array_equal(prediction.times, reference.times):
        raise ValueError("Native path comparison requires matching physics and observation times")
    expected_paths = kwargs.pop("expected_paths", len(reference.states))
    return evaluate_paths(
        prediction.states, reference.states, reference.times, reference.context, reference.spec,
        stopped_at=prediction.stopped_at, reference_stopped_at=reference.stopped_at,
        family_ids=prediction.family_ids, reference_family_ids=reference.family_ids,
        expected_paths=expected_paths, **kwargs,
    )


def empirical_quadratic_variation(
    states: np.ndarray,
    times: Sequence[float],
    n_particles: int,
    *,
    active_at_start: np.ndarray | None = None,
    crossed: np.ndarray | None = None,
    predictable_drift: np.ndarray | None = None,
    predictable_covariance: np.ndarray | None = None,
    predictable_face_amplitudes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Finite-mesh realized QV, with predictable *pre-stop* exposure accounting.

    Use uncensored Euler proposal increments and ``active_at_start`` for a
    coefficient/noise-normalization check. A boundary-crossing *stopped polygon*
    has a fractional final increment and must not be compared with a full-step
    covariance as though it were uncensored. ``crossed`` marks such increments:
    they remain in the reported realized QV but disable that coefficient check.
    Selecting only noncrossing proposals is also censored and is unsupported.

    Covariance is SS^T *before* division by N. Native face amplitudes may be
    supplied instead, avoiding a (paths,steps,G,G) covariance allocation.
    The optional predictable drift
    has shape (P,K,G), covariance (P,K,G,G), and both must be evaluated at the
    left endpoint using no future noise. This function cannot prove causality.
    """
    states = _array(states)
    if states.ndim != 3 or states.shape[0] < 1 or not np.all(np.isfinite(states)):
        raise ValueError("Finite complete path states of shape (paths,times,cells) are required")
    p, nt, g = states.shape
    grid = _times(times, nt)
    if not isinstance(n_particles, (int, np.integer)) or n_particles < 1:
        raise ValueError("A positive integer particle count is required")
    shape = (p, nt - 1)
    active = np.ones(shape, dtype=bool) if active_at_start is None else np.asarray(active_at_start)
    crossing = np.zeros(shape, dtype=bool) if crossed is None else np.asarray(crossed)
    if (active.shape != shape or active.dtype != np.bool_
            or crossing.shape != shape or crossing.dtype != np.bool_):
        raise ValueError("Boolean pre-stop activity and crossing flags must match every increment")
    if np.any(crossing & ~active):
        raise ValueError("A crossing increment must be active at its left endpoint")
    if np.any(np.diff(active.astype(int), axis=1) > 0):
        raise ValueError("Stopped activity cannot restart")
    dt = np.diff(grid)
    increments = np.diff(states, axis=1)
    if predictable_drift is not None:
        drift = _array(predictable_drift)
        if drift.shape != (p, nt - 1, g) or not np.all(np.isfinite(drift)):
            raise ValueError("Finite predictable drift must match all pre-step states")
        increments = increments - drift * dt[None, :, None] * active[:, :, None]
    residual = increments * active[:, :, None]
    realized = n_particles * np.einsum("ptg,pth->gh", residual, residual) / p
    exposure = float(np.sum(active * dt[None, :]) / p)
    result = {
        "n_paths": p,
        "n_active_proposals": int(active.sum()),
        "n_crossing_increments": int(crossing.sum()),
        "mean_active_time_exposure": exposure,
        "n_scaled_realized_qv": realized,
        "n_scaled_realized_trace": float(np.trace(realized)),
        "drift_removed": predictable_drift is not None,
        "all_pre_stop_increments_retained": True,
        "interpretation": (
            "Finite-mesh realized quadratic variation; drift, finite-step noise and stopping interpolation "
            "can contribute bias. This is not a continuous-QV convergence certificate."
        ),
        "coefficient_comparison": {"status": "unavailable"},
    }
    if predictable_covariance is not None and predictable_face_amplitudes is not None:
        raise ValueError("Provide covariance or native face amplitudes, not both")
    expected = None
    if predictable_covariance is not None:
        covariance = _array(predictable_covariance)
        if covariance.shape != (p, nt - 1, g, g) or not np.all(np.isfinite(covariance)):
            raise ValueError("Finite predictable SS^T covariance must match all pre-step states")
        expected = np.einsum("ptgh,pt,t->gh", covariance, active, dt) / p
    elif predictable_face_amplitudes is not None:
        amplitude = _array(predictable_face_amplitudes)
        if (amplitude.shape != (p, nt - 1, g) or not np.all(np.isfinite(amplitude))
                or np.any(amplitude < 0)):
            raise ValueError("Finite nonnegative native face amplitudes must match every pre-step state")
        integrated_face_variance = np.einsum("ptg,pt,t->g", amplitude**2, active, dt) / p
        incidence = np.roll(np.eye(g), 1, axis=0) - np.eye(g)
        expected = (incidence * integrated_face_variance[None, :]) @ incidence.T
    if expected is not None:
        result["n_scaled_predictable_integrated_covariance"] = expected
        result["coefficient_comparison"] = {
            "status": "unresolved-stopped-crossing" if np.any(crossing) else "descriptive-uncensored-proposals",
            "n_scaled_difference": realized - expected,
            "relative_frobenius_difference": float(np.linalg.norm(realized - expected)
                                                     / max(np.linalg.norm(expected), 1e-300)),
            "interpretation": (
                "Full-step comparison requires uncensored proposals including crossing proposals; "
                "a fractional stopped increment has a different finite-step second moment"
            ),
        }
    return json_safe(result)


def empirical_proposal_quadratic_variation(
    pre_states: np.ndarray,
    proposals: np.ndarray,
    times: Sequence[float],
    n_particles: int,
    *,
    active_at_start: np.ndarray,
    crossing_proposals: np.ndarray | None = None,
    predictable_drift: np.ndarray | None = None,
    predictable_covariance: np.ndarray | None = None,
    predictable_face_amplitudes: np.ndarray | None = None,
) -> dict[str, Any]:
    """QV of uncensored proposals, retaining exit proposals before absorption.

    ``pre_states`` and ``proposals`` have shape (paths,steps,cells). Crossing
    proposals are ordinary uncensored innovations here, not the fractional
    increments of a stopped Euler polygon. Origins after absorption contribute
    zero exposure, determined solely by ``active_at_start``.
    """
    origins, proposals = _array(pre_states), _array(proposals)
    if (origins.shape != proposals.shape or origins.ndim != 3 or origins.shape[0] < 1
            or not np.all(np.isfinite(origins)) or not np.all(np.isfinite(proposals))):
        raise ValueError("Finite matching pre-states and uncensored proposals are required")
    active = np.asarray(active_at_start)
    if active.shape != origins.shape[:2] or active.dtype != np.bool_:
        raise ValueError("Boolean active-at-start flags must match every proposal")
    crossing = np.zeros_like(active) if crossing_proposals is None else np.asarray(crossing_proposals)
    if crossing.shape != active.shape or crossing.dtype != np.bool_ or np.any(crossing & ~active):
        raise ValueError("Boolean exit-proposal flags must identify active-at-start proposals")
    # A cumulative representation simply lets the common QV routine consume
    # these innovations. It is not a reconstructed physical trajectory.
    innovation = proposals - origins
    cumulative = np.concatenate((np.zeros_like(origins[:, :1]),
                                  np.cumsum(innovation, axis=1)), axis=1)
    result = empirical_quadratic_variation(
        cumulative, times, n_particles, active_at_start=active,
        predictable_drift=predictable_drift,
        predictable_covariance=predictable_covariance,
        predictable_face_amplitudes=predictable_face_amplitudes,
    )
    result.update({
        "input_representation": "uncensored-pre-stop-proposal-increments",
        "n_uncensored_exit_proposals": int(crossing.sum()),
        "crossing_proposals_retained": True,
    })
    return result


def verify_same_noise_replay(operator: Callable, initial_masses: np.ndarray,
                             brownian_increments: np.ndarray, *, atol: float = 1e-12) -> dict[str, Any]:
    """Check that a fixed-weight evaluation callable has no extra inference noise."""
    initial, increments = _array(initial_masses), _array(brownian_increments)
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("A finite nonnegative replay tolerance is required")
    first = _states(operator(initial.copy(), increments.copy()))
    second = _states(operator(initial.copy(), increments.copy()))
    valid = first.shape == second.shape and np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    difference = float(np.max(np.abs(first - second))) if valid and first.size else None
    return {
        "passed": bool(valid and difference is not None and difference <= atol),
        "maximum_same_noise_replay_difference": difference,
        "atol": atol,
        "interpretation": "Same physical noise and initial masses should determine the trajectory at fixed weights",
    }


def verify_causality(operator: Callable, initial_masses: np.ndarray,
                     brownian_increments: np.ndarray, cut_index: int, *,
                     seed: int = 0, atol: float = 1e-12) -> dict[str, Any]:
    """Perturb only future forcing and compare every state through the cut.

    Increments ``[:,k]`` drive state k to k+1. Thus increments starting at
    ``cut_index`` must not affect outputs through state ``cut_index``. Passing
    one perturbation is a regression check, not a proof of architectural causality.
    """
    initial, increments = _array(initial_masses), _array(brownian_increments)
    if (increments.ndim != 3 or initial.shape != (increments.shape[0], increments.shape[2])
            or not np.all(np.isfinite(initial)) or not np.all(np.isfinite(increments))):
        raise ValueError("Finite initial masses and (paths,steps,faces) increments are required")
    if (not isinstance(cut_index, (int, np.integer)) or not 0 <= cut_index < increments.shape[1]
            or not np.isfinite(atol) or atol < 0):
        raise ValueError("A valid future-noise cut and finite nonnegative tolerance are required")
    changed = increments.copy()
    rng = seeded_rng(seed, "future-noise-perturbation", int(cut_index))
    changed[:, cut_index:] += rng.normal(size=changed[:, cut_index:].shape)
    first = _states(operator(initial.copy(), increments.copy()))
    second = _states(operator(initial.copy(), changed))
    expected = (increments.shape[0], increments.shape[1] + 1, initial.shape[1])
    if first.shape != expected or second.shape != expected:
        raise ValueError("Causality check requires all states, including the initial state")
    valid = np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    prefix = float(np.max(np.abs(first[:, :cut_index + 1] - second[:, :cut_index + 1]))) if valid else None
    future = float(np.max(np.abs(first[:, cut_index + 1:] - second[:, cut_index + 1:]))) if valid else None
    return {
        "passed": bool(valid and prefix is not None and prefix <= atol),
        "cut_index": int(cut_index),
        "maximum_past_difference": prefix,
        "maximum_future_difference": future,
        "future_noise_was_perturbed": bool(np.any(changed != increments)),
        "atol": atol,
        "interpretation": "Future-noise regression check; architectural causality and forcing response require separate checks",
    }
