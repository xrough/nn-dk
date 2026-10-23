"""Population-statistic diagnostics with explicit reference and online uncertainty.

All ensemble estimates are unconditional. One failed trajectory makes an entire
ensemble unavailable; silently conditioning on successful rollouts is forbidden.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from .types import (
    Context,
    QueryBank,
    StatisticalPrediction,
    TargetSpec,
    TrajectoryBatch,
    seeded_rng,
)


def json_safe(value: Any) -> Any:
    """Convert scientific values to strict JSON, representing missing values by null."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def query_labels(queries: QueryBank) -> list[dict[str, Any]]:
    return [
        {"index": i, "time": float(t), "mode": int(k), "component": f,
         "label": f"{f}({k}x), t={t:g}"}
        for i, (t, k, f) in enumerate(
            (t, k, f) for t in queries.times for k in queries.modes for f in ("cos", "sin")
        )
    ]


def _check_context(batch: TrajectoryBatch, context: Context) -> str:
    recorded = batch.diagnostics.get("context_id")
    if recorded is not None and recorded != context.context_id:
        raise ValueError("Trajectory context identifier does not match the requested context")
    identifiers = batch.diagnostics.get("trajectory_ids")
    if identifiers is not None and (len(identifiers) != len(batch.states) or len(set(identifiers)) != len(identifiers)):
        raise ValueError("Independent trajectories require unique per-path identifiers")
    return "verified" if recorded is not None else "not_recorded"


def path_diagnostics(batch: TrajectoryBatch) -> dict[str, Any]:
    finite = np.isfinite(batch.states)
    finite_mass = np.where(finite, batch.states, np.nan)
    sums = np.sum(batch.states, axis=-1)
    finite_sums = sums[np.isfinite(sums)]
    any_finite = bool(finite.any())
    n = len(batch.states)
    minimum = float(np.nanmin(finite_mass)) if any_finite else None
    mass_error = float(np.max(np.abs(finite_sums - 1.0))) if finite_sums.size else None
    recorded_minimum = batch.diagnostics.get("minimum_mass")
    recorded_error = batch.diagnostics.get("maximum_mass_error", batch.diagnostics.get("mass_error"))
    if recorded_minimum is not None and np.isfinite(recorded_minimum):
        minimum = float(recorded_minimum) if minimum is None else min(minimum, float(recorded_minimum))
    if recorded_error is not None and np.isfinite(recorded_error):
        mass_error = float(recorded_error) if mass_error is None else max(mass_error, float(recorded_error))
    return json_safe({
        "n_paths": n,
        "n_valid_paths": int(np.count_nonzero(batch.valid)),
        "invalid_path_frequency": float(np.mean(~batch.valid)) if n else None,
        "minimum_mass": minimum,
        "maximum_mass_error": mass_error,
        "nonfinite_entries": int(np.count_nonzero(~finite)),
        "failure_times": batch.failure_times,
    })


def _covariance(y: np.ndarray, n_particles: int) -> np.ndarray:
    centered = y - y.mean(axis=0)
    return n_particles * (centered.T @ centered) / (len(y) - 1)


def _bootstrap(y: np.ndarray, n_particles: int, count: int, rng: np.random.Generator) -> dict[str, Any]:
    if len(y) < 2 or count < 2:
        return {"available": False, "reason": "At least two independent paths and two bootstrap draws are required"}
    means = np.empty((count, y.shape[1]))
    covariances = np.empty((count, y.shape[1], y.shape[1]))
    for b in range(count):
        sample = y[rng.integers(0, len(y), size=len(y))]
        means[b] = sample.mean(axis=0)
        covariances[b] = _covariance(sample, n_particles)
    return {
        "available": True,
        "method": "independent-trajectory percentile bootstrap",
        "confidence_level": 0.95,
        "n_bootstrap": count,
        "mean_ci95": np.quantile(means, [0.025, 0.975], axis=0).T,
        "scaled_covariance_ci95": np.moveaxis(np.quantile(covariances, [0.025, 0.975], axis=0), 0, -1),
        "caveat": "Finite-ensemble Monte Carlo uncertainty; does not include numerical bias or model fitting uncertainty",
    }


def estimate_queries(batch: TrajectoryBatch, queries: QueryBank, context: Context,
                     n_bootstrap: int = 200, seed: int = 0) -> dict[str, Any]:
    """Estimate a fixed query bank using whole independent trajectories as units."""
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be nonnegative")
    context_check = _check_context(batch, context)
    diagnostics = path_diagnostics(batch)
    base = {
        "target_id": batch.target.target_id,
        "context_id": context.context_id,
        "query_id": queries.query_id,
        "query_labels": query_labels(queries),
        "context_compatibility": context_check,
        "n_independent_paths": len(batch.states),
        "diagnostics": diagnostics,
    }
    # Validate the requested times even for a failed ensemble.
    y = queries.extract(batch)
    if len(y) < 2 or not np.all(batch.valid) or not np.all(np.isfinite(y)):
        reason = "Failed trajectories prohibit survivor-conditioned estimates" if not np.all(batch.valid) else "At least two finite independent paths are required"
        return json_safe({**base, "status": "unavailable", "reason": reason,
                          "mean": None, "scaled_covariance": None,
                          "uncertainty": {"available": False, "reason": reason}, "marginals": None})
    mean = y.mean(axis=0)
    covariance = _covariance(y, context.n_particles)
    sd = y.std(axis=0, ddof=1)
    centered = y - mean
    safe_sd = np.where(sd > 1e-15, sd, 1.0)
    marginals = []
    for i, label in enumerate(query_labels(queries)):
        marginals.append({
            **label, "mean": mean[i], "standard_deviation": sd[i],
            "quantile_probabilities": [0.05, 0.25, 0.5, 0.75, 0.95],
            "quantiles": np.quantile(y[:, i], [0.05, 0.25, 0.5, 0.75, 0.95]),
            "skewness": np.mean((centered[:, i] / safe_sd[i]) ** 3) if sd[i] > 1e-15 else None,
            "excess_kurtosis": np.mean((centered[:, i] / safe_sd[i]) ** 4) - 3 if sd[i] > 1e-15 else None,
        })
    return json_safe({
        **base, "status": "available", "mean": mean,
        "scaled_covariance": covariance, "scaled_variances": np.diag(covariance),
        "mean_standard_error": sd / np.sqrt(len(y)),
        "uncertainty": _bootstrap(y, context.n_particles, n_bootstrap,
                                   seeded_rng(seed, "query-bootstrap", context.context_id, queries.query_id)),
        "marginals": marginals,
    })


@lru_cache(maxsize=64)
def _uniform_covariance(target: TargetSpec, queries: QueryBank, diffusion: float) -> np.ndarray:
    from .particles import free_particle_statistics

    return free_particle_statistics(Context(diffusion=diffusion), target, queries).scaled_covariance


def _analytic_scales(context: Context, target: TargetSpec, queries: QueryBank) -> dict[str, Any]:
    """Known uniform-particle fluctuation units; never fitted to held-out outcomes."""
    covariance = _uniform_covariance(target, queries, context.diffusion)
    std = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    return {"source": "analytic-uniform-particle-observation", "mean": std,
            "scaled_covariance": np.outer(std, std)}


def evaluate(prediction: StatisticalPrediction | TrajectoryBatch, reference: TrajectoryBatch,
             context: Context, queries: QueryBank, seed: int = 0,
             n_bootstrap: int = 200, normalization_scales: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compare a prediction to independent reference paths without hiding failures.

    Normalizers must be analytic or derived from training data. Reference test
    outcomes are used for errors and uncertainty, never for the normalizer.
    """
    reference_check = _check_context(reference, context)
    reference_estimate = estimate_queries(reference, queries, context, n_bootstrap, seed)
    if isinstance(prediction, StatisticalPrediction):
        reference.target.require_compatible(prediction.target)
        if prediction.context_id != context.context_id or prediction.query_bank != queries:
            raise ValueError("Prediction context/query identifier mismatch")
        pred_estimate = {
            "status": "available", "mean": prediction.mean.tolist(),
            "scaled_covariance": prediction.scaled_covariance.tolist(),
            "diagnostics": json_safe(prediction.diagnostics),
            "uncertainty": {"available": False, "interpretation": "Direct moment prediction has no online trajectory sampling; fitting uncertainty is not estimated"},
            "n_independent_paths": None,
        }
        prediction_type = "direct_statistics"
    else:
        reference.target.require_compatible(prediction.target)
        _check_context(prediction, context)
        pred_estimate = estimate_queries(prediction, queries, context, n_bootstrap, seed + 1)
        pred_estimate["uncertainty"]["interpretation"] = "Online model Monte Carlo uncertainty, separate from reference Monte Carlo uncertainty"
        prediction_type = "trajectories"
        ref_ids = reference.diagnostics.get("trajectory_ids")
        pred_ids = prediction.diagnostics.get("trajectory_ids")
        if ref_ids is not None and pred_ids is not None and set(ref_ids) & set(pred_ids):
            raise ValueError("Evaluation requires independent reference and prediction trajectories")
    reference_estimate["uncertainty"]["interpretation"] = "Reference Monte Carlo uncertainty; numerical discretization bias is assessed separately in calibration"
    result: dict[str, Any] = {
        "status": "available", "target_id": reference.target.target_id,
        "context_id": context.context_id, "query_id": queries.query_id,
        "context": asdict(context), "queries": asdict(queries),
        "reference_context_compatibility": reference_check,
        "prediction_type": prediction_type,
        "reference": reference_estimate, "prediction": pred_estimate,
        "uncertainty_policy": "Reference intervals and online sampling intervals are distinct; neither certifies numerical bias or learned-model generalization",
    }
    if reference_estimate["status"] != "available" or pred_estimate["status"] != "available":
        result.update(status="unavailable", errors=None,
                      reason="An unconditional estimate is unavailable for at least one ensemble")
        return json_safe(result)
    scales = _analytic_scales(context, reference.target, queries) if normalization_scales is None else normalization_scales
    if scales.get("source") not in ("analytic-uniform-particle-observation", "training"):
        raise ValueError("Normalizers must declare an analytic or training source")
    mean_scale = np.asarray(scales["mean"], dtype=float)
    covariance_scale = np.asarray(scales["scaled_covariance"], dtype=float)
    d = queries.dimension
    if mean_scale.shape != (d,) or covariance_scale.shape != (d, d) or not np.all(np.isfinite(mean_scale)) or not np.all(np.isfinite(covariance_scale)) or np.any(mean_scale <= 0) or np.any(covariance_scale <= 0):
        raise ValueError("Positive finite mean/covariance normalization scales with query dimensions required")
    pm = np.asarray(pred_estimate["mean"])
    pc = np.asarray(pred_estimate["scaled_covariance"])
    rm = np.asarray(reference_estimate["mean"])
    rc = np.asarray(reference_estimate["scaled_covariance"])
    dm, dc = pm - rm, pc - rc
    symmetry = float(np.max(np.abs(pc - pc.T)))
    minimum_eigenvalue = float(np.linalg.eigvalsh((pc + pc.T) / 2).min())
    result["covariance_diagnostics"] = {"symmetry_error": symmetry,
        "minimum_eigenvalue": minimum_eigenvalue,
        "positive_semidefinite": minimum_eigenvalue >= -1e-10 and symmetry <= 1e-10}
    if not result["covariance_diagnostics"]["positive_semidefinite"]:
        result["status"] = "invalid_prediction"
    result["normalization"] = scales
    ref_ci = reference_estimate["uncertainty"]
    scalar_distances = None
    if isinstance(prediction, TrajectoryBatch):
        py, ry = queries.extract(prediction), queries.extract(reference)
        scalar_distances = np.asarray([wasserstein_distance(py[:, i], ry[:, i]) for i in range(d)])
    per_query = []
    for i, label in enumerate(query_labels(queries)):
        entry = {
            **label, "prediction_mean": pm[i], "reference_mean": rm[i],
            "mean_error": dm[i], "mean_absolute_error": abs(dm[i]),
            "mean_normalized_error": abs(dm[i]) / mean_scale[i],
            "prediction_scaled_variance": pc[i, i], "reference_scaled_variance": rc[i, i],
            "scaled_variance_absolute_error": abs(dc[i, i]),
            "scaled_variance_normalized_error": abs(dc[i, i]) / covariance_scale[i, i],
            "wasserstein_1": scalar_distances[i] if scalar_distances is not None else None,
        }
        if ref_ci["available"]:
            entry["reference_mean_ci95"] = ref_ci["mean_ci95"][i]
            entry["reference_scaled_variance_ci95"] = ref_ci["scaled_covariance_ci95"][i][i]
        per_query.append(entry)
    labels = query_labels(queries)
    pairs = [
        {"i": i, "j": j, "first": labels[i]["label"], "second": labels[j]["label"],
         "two_time": labels[i]["time"] != labels[j]["time"],
         "reference": rc[i, j], "prediction": pc[i, j],
         "absolute_error": abs(dc[i, j]), "normalized_error": abs(dc[i, j]) / covariance_scale[i, j],
         "reference_ci95": ref_ci["scaled_covariance_ci95"][i][j] if ref_ci["available"] else None}
        for i in range(d) for j in range(i, d)
    ]
    result["errors"] = {
        "mean_absolute": np.abs(dm), "mean_normalized": np.abs(dm) / mean_scale,
        "scaled_covariance_absolute": np.abs(dc),
        "scaled_covariance_normalized": np.abs(dc) / covariance_scale,
        "mean_normalized_rmse": float(np.sqrt(np.mean((dm / mean_scale) ** 2))),
        "scaled_covariance_normalized_rmse": float(np.sqrt(np.mean((dc / covariance_scale) ** 2))),
        "per_query": per_query, "covariance_pairs": pairs,
    }
    return json_safe(result)
