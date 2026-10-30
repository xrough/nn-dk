"""Certified pointwise comparisons keep reference validity and all covariances."""

import copy

import numpy as np
import pytest

from nn_dk.generator_evaluation import compare_certified_moments


def moments(batch=False):
    shape = (2, 3, 4) if batch else (3, 4)
    prediction = {"mean": np.zeros(shape),
                  "scaled_covariance": np.broadcast_to(np.eye(4), (*shape, 4)).copy(),
                  "horizons": (.25, .5, 1.), "target_id": "continuous-target"}
    packet = {**copy.deepcopy(prediction), "mean_error_bound": 0.,
              "scaled_covariance_error_bound": 0., "certificate": {"valid": True}}
    return prediction, packet


@pytest.mark.parametrize("batch", (False, True))
def test_exact_valid_reference_passes_without_monte_carlo_or_ci_claims(batch):
    prediction, packet = moments(batch)
    result = compare_certified_moments(prediction, packet, n_particles=2048, std=np.ones(4))
    assert result["accuracy_gate"]["status"] == "resolved_within_tolerance"
    assert result["accuracy_gate"]["sampling_resolved"]
    assert result["accuracy_gate"]["reference_bias_resolved"]
    assert result["errors"]["mean_normalized_rmse"] == 0
    assert result["errors"]["scaled_variance_normalized_rmse"] == 0
    uncertainty = result["uncertainty"]
    assert uncertainty["reference_monte_carlo_paths"] == 0
    assert uncertainty["maximum_sampling_normalized_halfwidth"] == 0
    assert "confidence_level" not in uncertainty
    assert not any("ci95" in key for key in uncertainty)
    assert np.asarray(uncertainty["scaled_covariance_normalized_contrast_enclosure"]).shape == (
        *prediction["scaled_covariance"].shape, 2)


@pytest.mark.parametrize("valid", (False, None, 1, "true"))
def test_invalid_or_untyped_certificate_cannot_pass_zero_error(valid):
    prediction, packet = moments()
    packet["certificate"] = {"valid": valid}
    result = compare_certified_moments(prediction, packet, n_particles=1, std=np.ones(4))
    assert result["errors"]["maximum_mean_normalized_error"] == 0
    assert result["accuracy_gate"]["status"] == "unresolved"
    assert not result["accuracy_gate"]["reference_certificate_valid"]
    assert not result["accuracy_gate"]["reference_bias_resolved"]


def test_normalization_applies_sqrt_n_only_to_physical_means_and_bounds():
    prediction, packet = moments()
    prediction["mean"][:] = .003
    prediction["scaled_covariance"] += .08 * np.eye(4)
    packet["mean_error_bound"] = .001
    packet["scaled_covariance_error_bound"] = .004
    result = compare_certified_moments(prediction, packet, n_particles=100, std=2 * np.ones(4))
    assert result["errors"]["mean_normalized_rmse"] == pytest.approx(.015)
    assert result["errors"]["scaled_variance_normalized_rmse"] == pytest.approx(.02)
    assert result["errors"]["scaled_covariance_normalized_rmse"] == pytest.approx(.01)
    assert result["accuracy_gate"]["maximum_reference_normalized_bound"] == pytest.approx(.005)
    np.testing.assert_allclose(result["uncertainty"]["mean_normalized_contrast_enclosure"],
                               np.broadcast_to((.01, .02), (3, 4, 2)))
    assert result["accuracy_gate"]["status"] == "resolved_within_tolerance"


def test_reference_bound_budget_and_error_plus_bound_are_separate_gates():
    prediction, packet = moments()
    packet["mean_error_bound"] = .02
    result = compare_certified_moments(prediction, packet, n_particles=1, std=np.ones(4))
    gate = result["accuracy_gate"]
    assert gate["contrast_within_tolerance"] and not gate["reference_bound_within_budget"]
    assert gate["status"] == "unresolved"
    prediction["mean"][:] = .048
    packet["mean_error_bound"] = .004
    result = compare_certified_moments(prediction, packet, n_particles=1, std=np.ones(4))
    gate = result["accuracy_gate"]
    assert gate["point_estimate_within_tolerance"] and gate["reference_bound_within_budget"]
    assert not gate["contrast_within_tolerance"]
    assert gate["maximum_contrast_normalized_enclosure_endpoint"] == pytest.approx(.052)
    assert gate["status"] == "unresolved"


def test_offdiagonal_failure_cannot_be_hidden_by_exact_variances_or_small_rmse():
    prediction, packet = moments(batch=True)
    prediction["scaled_covariance"][0, 0, 0, 1] = .08
    prediction["scaled_covariance"][0, 0, 1, 0] = .08
    result = compare_certified_moments(prediction, packet, n_particles=1, std=np.ones(4))
    assert result["errors"]["scaled_variance_normalized_rmse"] == 0
    assert result["errors"]["scaled_covariance_normalized_rmse"] < .05
    assert result["errors"]["maximum_scaled_covariance_normalized_error"] == .08
    assert result["accuracy_gate"]["status"] == "unresolved"


def test_broadcast_bounds_use_each_mode_product_and_preserve_signed_intervals():
    prediction, packet = moments(batch=True)
    prediction["mean"][:] = -.001
    packet["mean_error_bound"] = np.asarray((.0001, .0002, .0003, .0004))
    packet["scaled_covariance_error_bound"] = np.outer((1., 2., 3., 4.), (1., 2., 3., 4.)) * .002
    result = compare_certified_moments(prediction, packet, n_particles=100,
                                       std=np.asarray((1., 2., 3., 4.)))
    np.testing.assert_allclose(result["uncertainty"]["mean_normalized_reference_error_bound"], .001)
    np.testing.assert_allclose(result["uncertainty"]["scaled_covariance_normalized_reference_error_bound"], .002)
    interval = np.asarray(result["uncertainty"]["mean_normalized_contrast_enclosure"])
    assert np.all(interval[..., 1] < 0)


@pytest.mark.parametrize("field,value", [
    ("mean", np.zeros((3, 3))), ("mean", np.zeros((3, 1, 3, 4))),
    ("scaled_covariance", np.zeros((3, 4))), ("mean", np.full((3, 4), np.nan)),
    ("scaled_covariance", np.full((3, 4, 4), np.inf)),
    ("mean_error_bound", -.1), ("mean_error_bound", np.inf),
    ("scaled_covariance_error_bound", np.zeros((5, 5))),
    ("certificate", True),
])
def test_invalid_reference_arrays_and_bounds_rejected(field, value):
    prediction, packet = moments()
    packet[field] = value
    with pytest.raises(ValueError):
        compare_certified_moments(prediction, packet, n_particles=1, std=np.ones(4))


@pytest.mark.parametrize("field,value", [
    ("mean", np.zeros((3, 3))), ("mean", np.full((3, 4), np.nan)),
    ("scaled_covariance", -np.broadcast_to(np.eye(4), (3, 4, 4))),
])
def test_invalid_prediction_arrays_rejected(field, value):
    prediction, packet = moments()
    prediction[field] = value
    with pytest.raises(ValueError):
        compare_certified_moments(prediction, packet, n_particles=1, std=np.ones(4))


def test_incompatible_target_and_horizon_metadata_rejected():
    prediction, packet = moments()
    packet["target_id"] = "Euler-target"
    with pytest.raises(ValueError, match="target identifiers"):
        compare_certified_moments(prediction, packet, n_particles=1, std=np.ones(4))
    packet["target_id"] = prediction["target_id"]
    packet["horizons"] = (.25, .5, 2.)
    with pytest.raises(ValueError, match="query times"):
        compare_certified_moments(prediction, packet, n_particles=1, std=np.ones(4))


@pytest.mark.parametrize("kwargs", [
    {"n_particles": 0}, {"n_particles": 1.5}, {"n_particles": True},
    {"std": np.ones(3)}, {"std": np.zeros(4)}, {"std": np.full(4, np.nan)},
    {"tolerance": .01}, {"uncertainty_budget": 0.},
])
def test_invalid_normalization_and_budget_rejected(kwargs):
    prediction, packet = moments()
    parameters = {"n_particles": 1, "std": np.ones(4), **kwargs}
    with pytest.raises(ValueError):
        compare_certified_moments(prediction, packet, **parameters)
