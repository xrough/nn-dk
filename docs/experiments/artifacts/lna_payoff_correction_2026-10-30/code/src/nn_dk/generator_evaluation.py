"""Pointwise accuracy gates against deterministic continuous-moment enclosures."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .bsde_evaluation import _errors
from .data import json_ready


def _field(value, name):
    try:
        return value[name] if isinstance(value, Mapping) else getattr(value, name)
    except (KeyError, AttributeError) as error:
        raise ValueError(f"Certified moment comparison requires {name}") from error


def _bound(value, shape, name):
    try:
        bound = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Reference {name} must broadcast to its moment array") from error
    if not np.all(np.isfinite(bound)) or np.any(bound < 0):
        raise ValueError(f"Reference {name} must be finite and nonnegative")
    return bound


def compare_certified_moments(prediction, reference_packet, *, n_particles, std,
                               tolerance=.05, uncertainty_budget=.0125):
    """Compare every mean/full-covariance entry with deterministic error bounds.

    Reference means are physical means and covariances are already N-scaled.
    Bound arrays use those same units and may be scalar or broadcastable.  Their
    validity is supplied by the reference certificate; small numbers alone never
    turn an invalid or absent certificate into an accuracy certificate.
    """
    if (not isinstance(n_particles, (int, np.integer)) or isinstance(n_particles, (bool, np.bool_))
            or n_particles < 1 or not np.isfinite(float(n_particles))):
        raise ValueError("Positive finite integer particle count required")
    scale = np.asarray(std, dtype=np.float64)
    if scale.shape != (4,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Four positive finite observable normalization scales required")
    covariance_scale = np.outer(scale, scale)
    if not np.all(np.isfinite(covariance_scale)) or np.any(covariance_scale <= 0):
        raise ValueError("Observable covariance normalization scales overflow or underflow")
    if (not np.isfinite(tolerance) or not np.isfinite(uncertainty_budget)
            or not 0 < uncertainty_budget < tolerance):
        raise ValueError("Finite positive accuracy and reference-error budgets required")
    mean = np.asarray(_field(prediction, "mean"), dtype=np.float64)
    covariance = np.asarray(_field(prediction, "scaled_covariance"), dtype=np.float64)
    reference_mean = np.asarray(_field(reference_packet, "mean"), dtype=np.float64)
    reference_covariance = np.asarray(_field(reference_packet, "scaled_covariance"), dtype=np.float64)
    if (mean.ndim not in (2, 3) or mean.shape[-1] != 4 or any(size < 1 for size in mean.shape)
            or covariance.shape != (*mean.shape, 4)
            or reference_mean.shape != mean.shape or reference_covariance.shape != covariance.shape
            or not all(np.all(np.isfinite(value)) for value in (
                mean, covariance, reference_mean, reference_covariance))):
        raise ValueError("Matching finite horizon-by-four means and full same-time covariances required")
    if (not np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-9, rtol=0)
            or not np.allclose(reference_covariance, reference_covariance.swapaxes(-1, -2),
                               atol=1e-9, rtol=0)
            or np.linalg.eigvalsh(covariance).min() < -1e-9):
        raise ValueError("Predicted covariance must be positive semidefinite; both covariances must be symmetric")
    mean_bound = _bound(_field(reference_packet, "mean_error_bound"), mean.shape, "mean_error_bound")
    covariance_bound = _bound(_field(reference_packet, "scaled_covariance_error_bound"),
                              covariance.shape, "scaled_covariance_error_bound")
    certificate = _field(reference_packet, "certificate")
    if not isinstance(certificate, Mapping):
        raise ValueError("A reference certificate mapping is required")
    declared_valid = certificate.get("valid")
    certificate_valid = isinstance(declared_valid, (bool, np.bool_)) and bool(declared_valid)
    if isinstance(reference_packet, Mapping):
        predicted_target = (prediction.get("target_id") if isinstance(prediction, Mapping)
                            else getattr(prediction, "target_id", None))
        if (predicted_target is not None and reference_packet.get("target_id") is not None
                and predicted_target != reference_packet["target_id"]):
            raise ValueError("Prediction and reference continuous target identifiers differ")
        reference_times = reference_packet.get("horizons", reference_packet.get("query_times"))
        predicted_times = (prediction.get("horizons") if isinstance(prediction, Mapping)
                           else getattr(prediction, "horizons", None))
        if reference_times is not None and predicted_times is not None:
            a, b = np.asarray(reference_times), np.asarray(predicted_times)
            if a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError("Prediction and reference physical query times differ")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        errors = _errors(mean, covariance, reference_mean, reference_covariance, n_particles, scale)
        normalized_mean_bound = np.sqrt(n_particles) * mean_bound / scale
        normalized_covariance_bound = covariance_bound / covariance_scale
    if (not all(np.all(np.isfinite(value)) for value in errors.values())
            or not np.all(np.isfinite(normalized_mean_bound))
            or not np.all(np.isfinite(normalized_covariance_bound))):
        raise ValueError("Nonfinite normalized comparison errors or reference bounds")
    signed_mean = errors["mean_normalized_signed"]
    signed_covariance = errors["scaled_covariance_normalized_signed"]
    mean_interval = np.stack((signed_mean - normalized_mean_bound,
                              signed_mean + normalized_mean_bound), axis=-1)
    covariance_interval = np.stack((signed_covariance - normalized_covariance_bound,
                                    signed_covariance + normalized_covariance_bound), axis=-1)
    max_mean_endpoint = float(np.max(np.abs(mean_interval)))
    max_covariance_endpoint = float(np.max(np.abs(covariance_interval)))
    max_bound = max(float(np.max(normalized_mean_bound)), float(np.max(normalized_covariance_bound)))
    max_endpoint = max(max_mean_endpoint, max_covariance_endpoint)
    diagonal = np.diagonal(signed_covariance, axis1=-2, axis2=-1)
    errors["maximum_scaled_variance_normalized_error"] = float(np.max(np.abs(diagonal)))
    reference_resolved = bool(certificate_valid and max_bound <= uncertainty_budget)
    contrast_resolved = bool(max_endpoint <= tolerance)
    gate = {"status": "resolved_within_tolerance" if reference_resolved and contrast_resolved else "unresolved",
            "sampling_resolved": True, "sampling_error_bound": 0.,
            "reference_bias_resolved": reference_resolved,
            "reference_certificate_valid": certificate_valid,
            "reference_bound_within_budget": bool(max_bound <= uncertainty_budget),
            "contrast_within_tolerance": contrast_resolved,
            "point_estimate_within_tolerance": bool(max(errors["maximum_mean_normalized_error"],
                errors["maximum_scaled_covariance_normalized_error"]) <= tolerance),
            "tolerance": float(tolerance), "each_uncertainty_budget": float(uncertainty_budget),
            "maximum_reference_normalized_bound": max_bound,
            "maximum_contrast_normalized_enclosure_endpoint": max_endpoint,
            "interpretation": "All mean and full same-time covariance entries are checked. Bounds are deterministic enclosures, not bootstrap confidence intervals; optimizer seeds remain separate."}
    return json_ready({"errors": errors, "reference_mean": reference_mean,
                       "reference_scaled_covariance": reference_covariance,
                       "reference_certificate": dict(certificate), "accuracy_gate": gate,
                       "uncertainty": {"available": True, "method": "deterministic reference enclosure",
                           "interval_kind": "deterministic; no confidence level or Monte Carlo interpretation",
                           "maximum_sampling_normalized_halfwidth": 0., "reference_monte_carlo_paths": 0,
                           "mean_normalized_reference_error_bound": normalized_mean_bound,
                           "scaled_covariance_normalized_reference_error_bound": normalized_covariance_bound,
                           "mean_normalized_contrast_enclosure": mean_interval,
                           "scaled_covariance_normalized_contrast_enclosure": covariance_interval,
                           "maximum_reference_normalized_bound": max_bound,
                           "maximum_mean_normalized_enclosure_endpoint": max_mean_endpoint,
                           "maximum_scaled_covariance_normalized_enclosure_endpoint": max_covariance_endpoint,
                           "maximum_contrast_normalized_enclosure_endpoint": max_endpoint},
                       "normalization": {"n_particles": int(n_particles), "observable_std": scale,
                                         "mean": "sqrt(N) times physical difference divided by std",
                                         "covariance": "N-scaled physical covariance difference divided by std products"}})
