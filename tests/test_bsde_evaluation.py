"""Independent-statistics checks for the conditional BSDE evaluation contract."""

from dataclasses import replace

import numpy as np
import pytest

from nn_dk.bsde_config import bsde_preset
from nn_dk.bsde_evaluation import (
    _cache_chunk,
    _load_chunk,
    _provenance,
    _stopping_summary,
    aggregate_conditional_moments,
    compare_analytic_moments,
    compare_conditional_moments,
    compare_ensemble_moments,
    sample_moments,
)


def test_total_covariance_retains_histogram_sampling_component():
    means = np.zeros((3, 2, 4))
    means[:, :, 0] = np.asarray((-1., 0., 1.))[:, None]
    conditional = np.broadcast_to(2*np.eye(4), (3, 2, 4, 4)).copy()
    mean, covariance = aggregate_conditional_moments(means, conditional, 7)
    np.testing.assert_allclose(mean, 0)
    np.testing.assert_allclose(covariance[:, 0, 0], 9.)
    np.testing.assert_allclose(covariance[:, 1, 1], 2.)
    with pytest.raises(ValueError, match="At least two"):
        aggregate_conditional_moments(means[:1], conditional[:1], 7)


def test_exact_analytic_comparison_is_resolved_on_fixed_test_bank():
    mean = np.zeros((3, 2, 4))
    covariance = np.broadcast_to(np.eye(4), (3, 2, 4, 4))
    result = compare_analytic_moments(mean, covariance, mean, covariance)
    assert result["errors"]["mean_normalized_rmse"] == 0
    assert result["accuracy_gate"]["status"] == "resolved_within_tolerance"
    assert "fixed test states" in result["uncertainty"]["method"]


def test_conditional_bootstrap_retains_anchor_and_time_structure():
    rng = np.random.default_rng(14)
    samples = rng.normal(size=(2, 30, 2, 4))
    # A large deterministic anchor offset must not become conditional variance.
    samples[1] += 100
    stats = [sample_moments(s, 1) for s in samples]
    mean, covariance = np.asarray([s[0] for s in stats]), np.asarray([s[1] for s in stats])
    result = compare_conditional_moments(mean, covariance, samples, n_particles=1,
                                         std=np.ones(4), n_bootstrap=20, seed=9)
    assert result["errors"]["scaled_covariance_normalized_rmse"] == 0
    assert np.asarray(result["uncertainty"]["scaled_covariance_normalized_contrast_ci95"]).shape == (2, 2, 4, 4, 2)
    assert not result["accuracy_gate"]["reference_bias_resolved"]
    assert not result["uncertainty"]["anchor_generalization_uncertainty_included"]
    replay = compare_conditional_moments(mean, covariance, samples, n_particles=1,
                                         std=np.ones(4), n_bootstrap=20, seed=9)
    assert result == replay


def test_ensemble_bootstrap_keeps_matched_families():
    samples = np.random.default_rng(1).normal(size=(20, 3, 4))
    # Conditional means equal observed values, covariance=0: exact equality in
    # every paired bootstrap resample, including the between-initial term.
    conditional = np.zeros((20, 3, 4, 4))
    result = compare_ensemble_moments(samples, conditional, samples, n_particles=2,
                                      std=np.ones(4), n_bootstrap=20)
    assert result["errors"]["mean_normalized_rmse"] == 0
    assert result["errors"]["scaled_covariance_normalized_rmse"] == 0
    assert np.max(np.abs(result["uncertainty"]["mean_normalized_contrast_ci95"])) == 0
    assert np.max(np.abs(result["uncertainty"]["scaled_covariance_normalized_contrast_ci95"])) == 0
    assert result["accuracy_gate"]["status"] == "unresolved"


def test_all_stopped_paths_retained_and_zero_event_bound():
    result = _stopping_summary(np.asarray((0., .25, np.inf, 2.)), 1.)
    assert result["n_paths"] == 4
    assert result["n_stopped"] == 2
    assert result["probability_by_horizon"] == .5
    assert result["restricted_mean_stopping_time"] == .5625
    assert result["all_stopped_paths_included"]
    zero = _stopping_summary(np.full(8192, np.inf), 1.)
    assert zero["zero_stop_one_sided_95_upper"] == pytest.approx(1-.05**(1/8192))


def test_cache_rejects_changed_source_and_configuration(tmp_path):
    config = bsde_preset("bsde-smoke")
    path = tmp_path / "chunk.npz"
    _cache_chunk(path, {"samples": np.zeros((2, 3, 4))}, {**_provenance(config), "family": "one"})
    arrays, _ = _load_chunk(path, config, {"family": "one"})
    assert arrays["samples"].shape == (2, 3, 4)
    with pytest.raises(ValueError, match="family mismatch"):
        _load_chunk(path, config, {"family": "another"})
    with pytest.raises(ValueError, match="source/configuration mismatch"):
        _load_chunk(path, replace(config, seed=7), {"family": "one"})


def test_analytic_calibration_distinguishes_euler_and_continuous_truth():
    from nn_dk.bsde_calibration import brownian_quadratic_calibration, ou_matrix_calibration

    config = bsde_preset("bsde-smoke")
    brownian = brownian_quadratic_calibration(config)
    assert brownian["passed"]
    assert brownian["expected_terminal_residual_mse"] == 2*config.horizon*config.dt
    assert brownian["pathwise_discrete_identity_max_abs_error"] < 1e-10
    ou = ou_matrix_calibration(config)
    assert ou["passed"]
    assert all(row["euler_vs_continuous_covariance_max_abs"] > 0 for row in ou["rows"])
    assert max(row["continuous_matrix_closed_form_max_abs"] for row in ou["rows"]) < 1e-10


def test_native_reference_resume_reuses_whole_matched_families(tmp_path):
    from nn_dk.bsde_evaluation import _native_reference_bank

    config = replace(bsde_preset("bsde-smoke"), n_cells=8, horizon=1/32,
                     query_times=(1/128, 1/64, 1/32), reference_dt=1/256,
                     conditional_anchors=2, conditional_paths=2,
                     outer_paths=2, reference_batch_size=2)
    reference = _native_reference_bank(tmp_path, config, "nonuniform", "conditional", None)
    assert reference["fine"].shape == (2, 2, 3, 4)
    assert len(set(reference["family_ids"])) == 4
    assert reference["diagnostics"]["maximum_mass_error"] < 1e-12
    assert reference["stopping"]["fine"]["all_stopped_paths_included"]
    before = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*.npz")}
    replay = _native_reference_bank(tmp_path, config, "nonuniform", "conditional", None)
    np.testing.assert_array_equal(reference["fine"], replay["fine"])
    np.testing.assert_array_equal(reference["coarse"], replay["coarse"])
    assert before == {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*.npz")}


def test_reference_expiry_does_not_silently_shrink_allocations(tmp_path):
    from nn_dk.bsde_evaluation import _native_reference_bank
    from nn_dk.experiments import BudgetExhausted

    with pytest.raises(BudgetExhausted):
        _native_reference_bank(tmp_path, bsde_preset("bsde-smoke"), "uniform", "ensemble", 0.)
    assert not list(tmp_path.rglob("*.npz"))
