from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from nn_dk.calibration import _certify_reference, run_calibration
from nn_dk.evaluation import estimate_queries, evaluate
from nn_dk.particles import free_particle_statistics, simulate_particles
from nn_dk.reporting import render_report
from nn_dk.types import Context, QueryBank, StatisticalPrediction, TargetSpec, TrajectoryBatch


def batch(seed=0, n=48, m=8, context=None):
    context = context or Context(n_particles=32)
    target = TargetSpec(n_cells=m)
    states = np.random.default_rng(seed).dirichlet(np.ones(m) * 4, size=(n, 3))
    return TrajectoryBatch(np.array([0.0, 0.125, 0.25]), states, target, diagnostics={
        "context_id": context.context_id,
        "trajectory_ids": [f"{seed}-{i}" for i in range(n)],
    })


def test_estimates_use_independent_trajectory_bootstrap_and_strict_json():
    context, queries = Context(n_particles=32), QueryBank((0.125, 0.25))
    trajectories = batch(context=context)
    estimate = estimate_queries(trajectories, queries, context, n_bootstrap=32, seed=17)
    y = queries.extract(trajectories)
    np.testing.assert_allclose(estimate["mean"], y.mean(axis=0))
    np.testing.assert_allclose(estimate["scaled_covariance"], 32 * np.cov(y, rowvar=False))
    assert np.asarray(estimate["uncertainty"]["mean_ci95"]).shape == (8, 2)
    assert np.asarray(estimate["uncertainty"]["scaled_covariance_ci95"]).shape == (8, 8, 2)
    assert estimate["n_independent_paths"] == len(y)
    assert len(estimate["marginals"]) == queries.dimension
    assert np.linalg.eigvalsh(estimate["scaled_covariance"]).min() > -1e-12
    assert estimate == estimate_queries(trajectories, queries, context, n_bootstrap=32, seed=17)
    json.dumps(estimate, allow_nan=False)


def test_any_failed_path_makes_entire_ensemble_unavailable():
    context, queries = Context(n_particles=32), QueryBank((0.125, 0.25))
    trajectories = batch(context=context)
    trajectories.valid[-1] = False
    trajectories.states[-1, 1:] = np.nan
    trajectories.failure_times[-1] = 0.1
    trajectories.diagnostics.update(minimum_mass=-0.2, maximum_mass_error=1e-8)
    estimate = estimate_queries(trajectories, queries, context, n_bootstrap=8)
    assert estimate["status"] == "unavailable"
    assert estimate["mean"] is None and estimate["scaled_covariance"] is None
    assert estimate["diagnostics"]["invalid_path_frequency"] == pytest.approx(1 / 48)
    assert estimate["diagnostics"]["minimum_mass"] == -0.2
    assert estimate["diagnostics"]["maximum_mass_error"] == 1e-8
    assert not estimate["uncertainty"]["available"]
    result = evaluate(trajectories, batch(seed=1, context=context), context, queries, n_bootstrap=8)
    assert result["status"] == "unavailable" and result["errors"] is None
    json.dumps(result, allow_nan=False)


def test_insufficient_paths_and_missing_metadata_are_explicit():
    context, queries = Context(n_particles=32), QueryBank((0.125,))
    trajectories = batch(n=1, context=context)
    estimate = estimate_queries(trajectories, queries, context)
    assert estimate["status"] == "unavailable"
    trajectories = batch(context=context)
    trajectories.diagnostics.clear()
    estimate = estimate_queries(trajectories, queries, context, n_bootstrap=0)
    assert estimate["context_compatibility"] == "not_recorded"
    assert not estimate["uncertainty"]["available"]


def test_context_target_query_and_independence_checks():
    context, queries = Context(n_particles=32), QueryBank((0.125, 0.25))
    reference = batch(context=context)
    with pytest.raises(ValueError, match="context identifier"):
        estimate_queries(reference, queries, Context(n_particles=64))
    with pytest.raises(ValueError, match="Incompatible"):
        evaluate(batch(seed=1, m=16, context=context), reference, context, queries)
    with pytest.raises(ValueError, match="independent reference"):
        evaluate(reference, reference, context, queries)
    bad_ids = batch(context=context)
    bad_ids.diagnostics["trajectory_ids"][-1] = bad_ids.diagnostics["trajectory_ids"][0]
    with pytest.raises(ValueError, match="unique"):
        estimate_queries(bad_ids, queries, context)
    other_queries = QueryBank((0.125,))
    prediction = StatisticalPrediction(np.zeros(4), np.eye(4), reference.target, other_queries, context.context_id)
    with pytest.raises(ValueError, match="context/query"):
        evaluate(prediction, reference, context, queries)


def test_identical_distributions_have_zero_errors_but_separate_sampling_intervals():
    context, queries = Context(n_particles=32), QueryBank((0.125, 0.25))
    reference = batch(context=context)
    prediction = batch(seed=1, context=context)
    prediction.states[:] = reference.states
    result = evaluate(prediction, reference, context, queries, n_bootstrap=16)
    assert result["errors"]["mean_normalized_rmse"] == 0
    assert result["errors"]["scaled_covariance_normalized_rmse"] == 0
    assert all(row["wasserstein_1"] == 0 for row in result["errors"]["per_query"])
    assert any(row["two_time"] for row in result["errors"]["covariance_pairs"])
    assert result["covariance_diagnostics"]["positive_semidefinite"]
    assert "Online" in result["prediction"]["uncertainty"]["interpretation"]
    assert "Reference" in result["reference"]["uncertainty"]["interpretation"]


def test_direct_statistics_covariance_validity_and_analytic_normalization():
    context, queries = Context(n_particles=32), QueryBank((0.125, 0.25))
    reference = batch(context=context)
    prediction = StatisticalPrediction(np.zeros(8), -np.eye(8), reference.target, queries, context.context_id)
    first = evaluate(prediction, reference, context, queries, n_bootstrap=8)
    assert first["status"] == "invalid_prediction"
    assert not first["covariance_diagnostics"]["positive_semidefinite"]
    second = evaluate(prediction, batch(seed=5, context=context), context, queries, n_bootstrap=8)
    assert first["normalization"] == second["normalization"]
    assert first["normalization"]["source"] == "analytic-uniform-particle-observation"
    assert not first["prediction"]["uncertainty"]["available"]
    with pytest.raises(ValueError, match="analytic or training"):
        evaluate(prediction, reference, context, queries, normalization_scales={"source": "test"})


def test_filtered_heat_truth_is_consistent_with_particle_estimates():
    context, target, queries = Context(n_particles=16), TargetSpec(n_cells=8), QueryBank((0.125, 0.25))
    reference = simulate_particles(context, target, [0, 0.125, 0.25], 1 / 32, seed=22, n_paths=1024)
    truth = free_particle_statistics(context, target, queries, quadrature=4096)
    result = evaluate(truth, reference, context, queries, n_bootstrap=24)
    assert result["errors"]["mean_normalized_rmse"] < 0.04
    assert result["errors"]["scaled_covariance_normalized_rmse"] < 0.1
    # Coarse spline filtering attenuates mode 2 substantially at M=8.
    assert truth.scaled_covariance[2, 2] < 0.5


def test_calibration_separates_implementation_from_scientific_resolution():
    result = run_calibration(TargetSpec(n_cells=8), dt=1 / 64, save_interval=1 / 32,
                             horizon=0.125, seed=8, n_paths=24, n_particles=16,
                             refinement_paths=4, n_bootstrap=16)
    assert result["status"] == "passed"
    assert result["reference_certification"]["status"] == "unresolved"
    assert not result["reference_certification"]["free_reference_uncertainty_resolved"]
    assert set(result["interacting_timestep_refinement"]["comparisons"]) == {"coarse_to_half", "half_to_quarter"}
    errors = [r["relative_uniform_observation_covariance_rate_error"] for r in result["spatial_noise_refinement"]]
    assert errors[-1] < errors[0]
    json.dumps(result, allow_nan=False)


def test_precise_but_biased_oracle_estimate_cannot_resolve_reference_gate():
    truth = np.eye(2)
    biased = truth + 0.2
    intervals = np.stack([biased - 0.001, biased + 0.001], axis=-1)
    refinement_intervals = np.full((2, 2, 2), 0.001)
    result = _certify_reference(True, intervals, truth, np.ones((2, 2)), refinement_intervals, 0.05)
    assert result["free_reference_uncertainty_resolved"]
    assert result["paired_refinement_difference_resolved"]
    assert not result["free_particle_oracle_consistency_resolved"]
    assert result["status"] == "unresolved"


def test_report_uses_immutable_records_and_writes_reviewable_artifacts(tmp_path):
    context, queries = Context(n_particles=32), QueryBank((0.125, 0.25))
    result = evaluate(batch(seed=1, context=context), batch(context=context), context, queries, n_bootstrap=8)
    result.update(method="test model", costs={"inference_seconds": 0.01, "training_seconds": 0.2})
    records = [result]
    before = copy.deepcopy(records)
    paths = render_report(records, {"status": "passed", "reference_certification": {"status": "unresolved"}},
                          {"seed": 4, "configuration_hash": "abc"}, tmp_path)
    assert records == before
    assert set(paths) == {"markdown", "json", "figure"}
    assert (tmp_path / "accuracy_cost.png").stat().st_size > 1000
    assert "unresolved" in (tmp_path / "report.md").read_text()
    assert json.loads((tmp_path / "report.json").read_text())["records"][0]["method"] == "test model"
