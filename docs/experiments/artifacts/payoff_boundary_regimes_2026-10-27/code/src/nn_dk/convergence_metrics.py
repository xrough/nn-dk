"""Weak-error diagnostics for the stopped, finite-volume DK target.

Every independent trajectory family is one sampling unit. Absorbed paths are
included, and analytic physical units never depend on held-out observations.
The confidence intervals are pointwise Monte Carlo intervals, not simultaneous
bands or confidence intervals over fitted neural-network seeds.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from .evaluation import json_safe, query_labels
from .native_types import NativeBatch, NativeDKSpec
from .types import QueryBank, seeded_rng


def native_normalization(spec: NativeDKSpec, queries: QueryBank) -> dict[str, Any]:
    """Exact uniform-IID histogram fluctuation units for cell-integrated modes.

    For phi=sqrt(2) cos(kx) (or sine), its cell average has uniform variance
    sinc(k/G)^2. Off-diagonal covariances are divided by the product of the
    corresponding standard deviations, including when the true covariance is 0.
    """
    if max(queries.modes) >= spec.n_cells / 2:
        raise ValueError("Physical observable modes must be below grid Nyquist")
    std = np.tile(np.repeat(np.sinc(np.asarray(queries.modes) / spec.n_cells), 2),
                  len(queries.times))
    return {"source": "analytic-uniform-iid-cell-integrated-physical-modes",
            "mean": std, "scaled_covariance": np.outer(std, std),
            "interpretation": "Fixed uniform-histogram units, independent of held-out outcomes"}


def _extract(batch: NativeBatch, queries: QueryBank) -> np.ndarray:
    # Import lazily so the numerical backend can import this module in its CLI.
    from .native import extract_observables

    return extract_observables(batch, queries)


def _moments(y: np.ndarray, n_particles: int) -> tuple[np.ndarray, np.ndarray]:
    mean = y.mean(axis=0)
    z = y - mean
    covariance = n_particles * (z.T @ z) / (len(y) - 1)
    return mean, covariance


def _stop_indicators(batch: NativeBatch, times: np.ndarray) -> np.ndarray:
    return batch.stopped_at[:, None] <= times[None, :]


def _draw_stats(y: np.ndarray, stops: np.ndarray, n_particles: int,
                indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, cov = _moments(y[indices], n_particles)
    return mean, cov, stops[indices].mean(axis=0)


def _bootstrap_arrays(y: np.ndarray, stops: np.ndarray, n_particles: int,
                      count: int, rng: np.random.Generator) -> dict[str, np.ndarray] | None:
    if len(y) < 2 or count < 2:
        return None
    means, covs, probabilities = [], [], []
    for _ in range(count):
        mean, cov, prob = _draw_stats(y, stops, n_particles,
                                      rng.integers(0, len(y), len(y)))
        means.append(mean)
        covs.append(cov)
        probabilities.append(prob)
    return {"mean": np.asarray(means), "covariance": np.asarray(covs),
            "stopping": np.asarray(probabilities)}


def _ci(draws: np.ndarray) -> np.ndarray:
    return np.moveaxis(np.quantile(draws, [0.025, 0.975], axis=0), 0, -1)


def _halfwidth(ci: np.ndarray) -> np.ndarray:
    return (ci[..., 1] - ci[..., 0]) / 2


def _estimate(batch: NativeBatch, queries: QueryBank, y: np.ndarray,
              draws: dict[str, np.ndarray] | None) -> dict[str, Any]:
    horizon = queries.times[-1]
    stop_times = np.asarray(queries.times)
    probabilities = _stop_indicators(batch, stop_times).mean(axis=0)
    finite_stops = batch.stopped_at[np.isfinite(batch.stopped_at) & (batch.stopped_at <= horizon)]
    diagnostics = {
        "all_stopped_paths_included": True,
        "n_paths": len(y), "n_stopped": len(finite_stops),
        "initially_stopped": int(np.count_nonzero(batch.stopped_at == 0)),
        "minimum_mass": float(batch.states.min()),
        "maximum_mass_error": float(np.max(np.abs(batch.states.sum(-1) - 1))),
        "minimum_density": float(batch.states.min() / batch.spec.h),
    }
    base = {
        "target_id": batch.spec.target_id, "target": asdict(batch.spec),
        "context_id": batch.context.context_id, "query_id": queries.query_id,
        "query_labels": query_labels(queries), "n_independent_paths": len(y),
        "diagnostics": diagnostics,
        "stopping": {"times": stop_times, "cdf": probabilities,
                     "probability_by_horizon": probabilities[-1],
                     "horizon": horizon,
                     "restricted_mean_stopping_time": float(np.minimum(batch.stopped_at, horizon).mean()),
                     "observed_stopping_times": finite_stops,
                     "right_censored_paths": len(y) - len(finite_stops)},
    }
    if len(y) < 2:
        return {**base, "status": "unavailable", "mean": None,
                "scaled_covariance": None, "scaled_variances": None,
                "uncertainty": {"available": False, "reason": "Two independent paths required"}}
    mean, cov = _moments(y, batch.context.n_particles)
    uncertainty = {"available": draws is not None,
                   "method": "whole-independent-trajectory-family percentile bootstrap",
                   "confidence_level": 0.95,
                   "interpretation": "Pointwise finite-ensemble MC uncertainty; excludes training and reference bias"}
    if draws is not None:
        uncertainty.update(mean_ci95=_ci(draws["mean"]),
                           scaled_covariance_ci95=_ci(draws["covariance"]),
                           stopping_cdf_ci95=_ci(draws["stopping"]),
                           n_bootstrap=len(draws["mean"]))
    return {**base, "status": "available", "mean": mean,
            "scaled_covariance": cov, "scaled_variances": np.diag(cov),
            "mean_standard_error": y.std(axis=0, ddof=1) / np.sqrt(len(y)),
            "uncertainty": uncertainty}


def native_estimates(batch: NativeBatch, queries: QueryBank,
                     n_bootstrap: int = 200, seed: int = 0) -> dict[str, Any]:
    """Estimate the full stopped law; a stop never removes a trajectory."""
    if not isinstance(batch, NativeBatch):
        raise ValueError("A native stopped DK target is required")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be nonnegative")
    y = _extract(batch, queries)
    draws = _bootstrap_arrays(y, _stop_indicators(batch, np.asarray(queries.times)),
                              batch.context.n_particles, n_bootstrap,
                              seeded_rng(seed, "native-estimate", batch.context.context_id))
    return json_safe(_estimate(batch, queries, y, draws))


def compare_native(prediction: NativeBatch, reference: NativeBatch, queries: QueryBank,
                   n_bootstrap: int = 200, seed: int = 0, paired: bool = False,
                   tolerance: float = 0.05, uncertainty_budget: float = 0.0125,
                   reference_bias: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compare matching stopped DK laws, with independent or coupled families.

    Paired comparisons require identical family IDs (order may differ). Unpaired
    comparisons reject any overlapping IDs. ``reference_bias`` can certify a
    separate refinement audit with status=resolved and normalized_bound<=budget;
    Monte Carlo intervals alone never certify the fine reference.
    """
    if not isinstance(reference, NativeBatch) or not isinstance(prediction, NativeBatch):
        raise ValueError("Only native stopped DK targets can be compared here")
    reference.spec.require_compatible(prediction.spec)
    if reference.context != prediction.context:
        raise ValueError("Native comparison requires identical initialization law and physics")
    if n_bootstrap < 0 or tolerance <= 0 or uncertainty_budget <= 0:
        raise ValueError("Nonnegative bootstrap count and positive error budgets required")
    pred_ids, ref_ids = set(prediction.family_ids), set(reference.family_ids)
    if paired and pred_ids != ref_ids:
        raise ValueError("Paired refinements require matching trajectory family IDs")
    if not paired and pred_ids & ref_ids:
        raise ValueError("Independent comparisons forbid overlapping trajectory families")
    py, ry = _extract(prediction, queries), _extract(reference, queries)
    pstop = _stop_indicators(prediction, np.asarray(queries.times))
    rstop = _stop_indicators(reference, np.asarray(queries.times))
    if paired:
        index = {key: i for i, key in enumerate(prediction.family_ids)}
        order = np.asarray([index[key] for key in reference.family_ids])
        py, pstop = py[order], pstop[order]
    rng = seeded_rng(seed, "native-contrast", reference.context.context_id, queries.query_id)
    pdraw, rdraw = None, None
    if min(len(py), len(ry)) >= 2 and n_bootstrap >= 2:
        pvalues, rvalues = [], []
        for _ in range(n_bootstrap):
            ri = rng.integers(0, len(ry), len(ry))
            pi = ri if paired else rng.integers(0, len(py), len(py))
            pvalues.append(_draw_stats(py, pstop, reference.context.n_particles, pi))
            rvalues.append(_draw_stats(ry, rstop, reference.context.n_particles, ri))
        pdraw = {key: np.asarray([v[i] for v in pvalues])
                 for i, key in enumerate(("mean", "covariance", "stopping"))}
        rdraw = {key: np.asarray([v[i] for v in rvalues])
                 for i, key in enumerate(("mean", "covariance", "stopping"))}
    result = {
        "target_id": reference.spec.target_id, "context_id": reference.context.context_id,
        "query_id": queries.query_id, "paired": paired,
        "prediction": _estimate(prediction, queries, py, pdraw),
        "reference": _estimate(reference, queries, ry, rdraw),
        "normalization": native_normalization(reference.spec, queries),
        "scope": "Stopped spatially discrete DK at fixed grid and particle number",
    }
    if min(len(py), len(ry)) < 2:
        return json_safe({**result, "status": "unavailable", "errors": None,
                          "accuracy_gate": {"status": "unresolved", "reason": "Two independent paths per pool required"}})
    n = reference.context.n_particles
    pm, pc = _moments(py, n)
    rm, rc = _moments(ry, n)
    std = np.asarray(result["normalization"]["mean"])
    cscale = np.outer(std, std)
    mean_error = np.sqrt(n) * (pm - rm)
    cov_error = pc - rc
    nm, nc = mean_error / std, cov_error / cscale
    distances = np.sqrt(n) * np.asarray([wasserstein_distance(py[:, i], ry[:, i])
                                        for i in range(queries.dimension)])
    contrast = {"available": pdraw is not None,
                "paired": paired, "confidence_level": 0.95,
                "unit": "whole trajectory family",
                "wasserstein_uncertainty": "Not bootstrapped; descriptive empirical distance only"}
    sampling_resolved = False
    contrast_resolved = False
    if pdraw is not None and rdraw is not None:
        mean_ci = _ci(np.sqrt(n) * (pdraw["mean"] - rdraw["mean"]) / std)
        cov_ci = _ci((pdraw["covariance"] - rdraw["covariance"]) / cscale)
        stop_ci = _ci(pdraw["stopping"] - rdraw["stopping"])
        components = {}
        for name, draws in (("prediction", pdraw), ("reference", rdraw)):
            components[name] = {
                "mean_normalized_ci_halfwidth": _halfwidth(_ci(np.sqrt(n) * draws["mean"] / std)),
                "scaled_covariance_normalized_ci_halfwidth": _halfwidth(_ci(draws["covariance"] / cscale)),
                "stopping_probability_ci_halfwidth": _halfwidth(_ci(draws["stopping"])),
            }
        all_widths = [np.max(v) for component in components.values() for v in component.values()]
        sampling_resolved = bool(max(all_widths) <= uncertainty_budget)
        contrast_resolved = bool(np.max(np.abs(mean_ci)) <= tolerance
                                 and np.max(np.abs(cov_ci)) <= tolerance
                                 and np.max(np.abs(stop_ci)) <= tolerance)
        contrast.update(mean_normalized_ci95=mean_ci, scaled_covariance_normalized_ci95=cov_ci,
                        stopping_probability_difference_ci95=stop_ci,
                        mean_normalized_standard_error=np.std(np.sqrt(n) * (pdraw["mean"] - rdraw["mean"]) / std, axis=0, ddof=1),
                        scaled_covariance_normalized_standard_error=np.std((pdraw["covariance"] - rdraw["covariance"]) / cscale, axis=0, ddof=1),
                        sampling_components=components, n_bootstrap=n_bootstrap)
    labels = query_labels(queries)
    per_query = [{**label, "sqrt_n_mean_error": mean_error[i],
                  "mean_normalized_error": abs(nm[i]),
                  "scaled_variance_error": cov_error[i, i],
                  "scaled_variance_normalized_error": abs(nc[i, i]),
                  "sqrt_n_wasserstein": distances[i],
                  "wasserstein_normalized": distances[i] / std[i],
                  "mean_normalized_contrast_ci95": contrast.get("mean_normalized_ci95", [None] * len(labels))[i]}
                 for i, label in enumerate(labels)]
    pairs = [{"i": i, "j": j, "first": labels[i]["label"], "second": labels[j]["label"],
              "two_time": labels[i]["time"] != labels[j]["time"],
              "prediction": pc[i, j], "reference": rc[i, j],
              "scaled_covariance_error": cov_error[i, j], "normalized_error": abs(nc[i, j]),
              "normalized_contrast_ci95": contrast["scaled_covariance_normalized_ci95"][i, j] if pdraw is not None else None}
             for i in range(len(labels)) for j in range(i, len(labels))]
    bias_bound = None if reference_bias is None else reference_bias.get("normalized_bound")
    reference_resolved = bool(reference_bias is not None and reference_bias.get("status") == "resolved"
                              and isinstance(bias_bound, (int, float)) and np.isfinite(bias_bound)
                              and 0 <= bias_bound <= uncertainty_budget)
    if reference_resolved:
        # The triangle inequality adds reference bias to the sampled contrast.
        contrast_resolved = bool(pdraw is not None
                                 and np.max(np.abs(contrast["mean_normalized_ci95"])) + bias_bound <= tolerance
                                 and np.max(np.abs(contrast["scaled_covariance_normalized_ci95"])) + bias_bound <= tolerance
                                 and np.max(np.abs(contrast["stopping_probability_difference_ci95"])) + bias_bound <= tolerance)
    gate = {
        "status": "resolved_within_tolerance" if sampling_resolved and contrast_resolved and reference_resolved else "unresolved",
        "tolerance": tolerance, "each_sampling_reference_budget": uncertainty_budget,
        "sampling_resolved": sampling_resolved, "contrast_within_tolerance": contrast_resolved,
        "reference_bias_resolved": reference_resolved,
        "reference_bias": reference_bias,
        "interpretation": "Pointwise MC intervals exclude training uncertainty. Full weak-error certification also requires a separate reference bias audit; Wasserstein is descriptive.",
    }
    result.update(status="available", errors={
        "sqrt_n_mean_signed": mean_error, "mean_normalized": np.abs(nm),
        "scaled_covariance_signed": cov_error, "scaled_covariance_normalized": np.abs(nc),
        "mean_normalized_rmse": float(np.sqrt(np.mean(nm ** 2))),
        "scaled_covariance_normalized_rmse": float(np.sqrt(np.mean(nc ** 2))),
        "sqrt_n_wasserstein": distances, "wasserstein_normalized": distances / std,
        "per_query": per_query, "covariance_pairs": pairs,
        "stopping_probability_difference": pstop.mean(axis=0) - rstop.mean(axis=0),
    }, uncertainty=contrast, accuracy_gate=gate)
    return json_safe(result)


def temporal_rate(records: list[dict[str, Any]], metric: str = "mean_normalized_rmse",
                  reference_bias_bound: float | None = None,
                  uncertainty_budget: float = 0.0125) -> dict[str, Any]:
    """Fit an observed log-log slope only when signal and reference are resolved.

    Each row accepts ``step_size``, ``error``/``standard_error`` directly, or a
    comparison from :func:`compare_native`. A comparison's conservative maximum
    pointwise SE is used as the scalar error uncertainty. No convergence order is
    assumed and nonmonotone/noisy plateaus are explicitly unresolved.
    """
    points = []
    identity_keys = ("case", "training_mode", "method", "seed", "width", "train_paths")
    if any(len({str(r[k]) for r in records if k in r}) > 1 for k in identity_keys):
        return {"metric": metric, "points": [], "order": None, "status": "unresolved",
                "reason": "Temporal curves must hold the fitted model, seed, data, and problem fixed"}
    for record in records:
        comparison = record.get("comparison", record)
        dt = record.get("step_size", record.get("dt"))
        error = record.get("error", (comparison.get("errors") or {}).get(metric))
        se = record.get("standard_error")
        if se is None:
            key = "mean_normalized_standard_error" if metric.startswith("mean") else "scaled_covariance_normalized_standard_error"
            uncertainty = comparison.get("uncertainty", {})
            values = uncertainty.get(key)
            se = float(np.max(values)) if values is not None else None
        if all(isinstance(x, (int, float)) and np.isfinite(x) for x in (dt, error, se)) and dt > 0 and error > 0 and se >= 0:
            points.append({"step_size": float(dt), "error": float(error), "standard_error": float(se)})
    base = {"metric": metric, "points": points, "order": None,
            "interpretation": "Observed weak-error slope for a fixed fitted coefficient model; no theoretical order is presumed"}
    if reference_bias_bound is None or not np.isfinite(reference_bias_bound) or not 0 <= reference_bias_bound <= uncertainty_budget:
        return {**base, "status": "unresolved", "reason": "Reference bias is not bounded within its budget"}
    points = sorted(points, key=lambda p: p["step_size"])
    if len({p["step_size"] for p in points}) < 3:
        return {**base, "status": "unresolved", "reason": "At least three distinct finite refinement points required"}
    if any(p["error"] <= 3 * p["standard_error"] + reference_bias_bound for p in points):
        return {**base, "status": "unresolved", "reason": "Errors are not distinguishable from sampling/reference uncertainty"}
    for fine, coarse in zip(points, points[1:], strict=False):
        if coarse["error"] - fine["error"] <= 3 * np.hypot(coarse["standard_error"], fine["standard_error"]) + 2 * reference_bias_bound:
            return {**base, "status": "unresolved", "reason": "Adjacent refinement errors do not resolve a decreasing trend"}
    x, y = np.log([p["step_size"] for p in points]), np.log([p["error"] for p in points])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return {**base, "status": "resolved", "order": float(slope),
            "log_fit_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "reason": "Three or more signal-resolved monotone refinement points with a bounded reference"}
