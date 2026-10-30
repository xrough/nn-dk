"""Whole-family bootstrap and common-observation definitions."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.evaluate_path_generalization import (  # noqa: E402
    bootstrap_counts,
    diagonal_comparison,
    empirical,
    mixture_moments,
    moment_comparison,
    reduce_paths,
)
from nn_dk.native_types import NativeDKSpec  # noqa: E402


def test_weighted_bootstrap_matches_literal_family_resampling():
    y = np.random.default_rng(91).normal(size=(11, 3, 4))
    counts = bootstrap_counts(11, 92, 5)
    means, covs = empirical(y, counts)
    for i, weights in enumerate(counts):
        sample = y[np.repeat(np.arange(11), weights.astype(int))]
        mean, cov = empirical(sample)
        np.testing.assert_allclose(means[i], mean, atol=1e-15)
        np.testing.assert_allclose(covs[i], cov, atol=2e-15)


def test_exact_paired_paths_have_zero_error_and_uncertainty():
    y = np.random.default_rng(93).normal(size=(25, 3, 4))
    result = moment_comparison(y, y.copy(), counts=bootstrap_counts(25, 94, 11))
    for row in result["records"]:
        assert row["max_mean_error_in_reference_std"] == 0
        assert row["max_normalized_covariance_error"] == 0
        assert row["mean_sampling95_simultaneous_radius"] == 0
        assert row["covariance_sampling95_simultaneous_radius"] == 0


def test_fixed_prediction_includes_reference_uncertainty():
    y = np.random.default_rng(95).normal(size=(40, 2, 3))
    mean, cov = empirical(y)
    result = moment_comparison(y, predicted_mean=mean,
                               predicted_scaled_covariance=2048*cov,
                               counts=bootstrap_counts(40, 96, 20))
    assert all(r["mean_error_plus_sampling95"] > .1 for r in result["records"])
    assert all(r["covariance_error_plus_sampling95"] > .1 for r in result["records"])


def test_diagonal_and_small_bank_scores_agree():
    y = np.random.default_rng(97).normal(size=(33, 4))
    z = 1.03*y+.02
    counts = bootstrap_counts(33, 98, 20)
    diag = diagonal_comparison(y, z, counts=counts)
    full = moment_comparison(y, z, counts=counts)["records"][0]
    assert abs(diag["mean_max_std"]-full["max_mean_error_in_reference_std"]) < 1e-15
    np.testing.assert_allclose(diag["variance_errors"],
                               np.diag(full["normalized_covariance_difference"]), atol=1e-15)


def test_common_grid_reduction_and_killed_integral_use_all_path_nodes():
    spec = NativeDKSpec()
    dt, horizon = 2**-9, 1/16
    states = np.full((3, 33, 32), 1/32)
    # Initially stopped paths have zero killed integrals, terminal outputs
    # remain their frozen field; here a uniform field has every payoff zero.
    result = reduce_paths({"states": states, "stopped_at": np.array([0., np.inf, np.inf])},
                          spec, dt, horizon)
    assert result["many"].shape == (3, 480)
    assert result["common_states"].shape == (3, 17, 32)
    for values in result["banks"].values():
        assert values.shape == (3, 3, 4)
        np.testing.assert_allclose(values, 0, atol=1e-15)


def test_mixture_covariance_bootstrap_repeats_initial_families():
    rng = np.random.default_rng(99)
    mean = rng.normal(size=(13, 3, 4))
    factors = rng.normal(size=(13, 3, 4, 4))
    conditional_cov = factors @ factors.swapaxes(-1, -2)
    counts = bootstrap_counts(13, 100, 5)
    bm, bc = mixture_moments(mean, conditional_cov, counts=counts)
    for i, weights in enumerate(counts):
        indices = np.repeat(np.arange(13), weights.astype(int))
        mu, cov = mixture_moments(mean[indices], conditional_cov[indices])
        np.testing.assert_allclose(bm[i], mu, atol=1e-15)
        np.testing.assert_allclose(bc[i], cov, rtol=1e-14)


def test_nonfinite_predictions_cannot_produce_an_estimate():
    y = np.random.default_rng(101).normal(size=(20, 3, 4))
    mean, cov = empirical(y)
    mean[0, 0] = np.nan
    with pytest.raises(ValueError, match="match the declared bank"):
        moment_comparison(y, predicted_mean=mean, predicted_scaled_covariance=2048*cov)


def test_misaligned_path_observation_grid_is_rejected():
    spec = NativeDKSpec()
    states = np.full((3, 9, 32), 1/32)
    with pytest.raises(ValueError, match="exactly nested"):
        reduce_paths({"states": states, "stopped_at": np.full(3, np.inf)},
                     spec, 2**-7, 1/16)
