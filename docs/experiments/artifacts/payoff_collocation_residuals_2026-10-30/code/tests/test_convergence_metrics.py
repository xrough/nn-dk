from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from nn_dk.convergence_metrics import (
    compare_native,
    native_estimates,
    native_normalization,
    temporal_rate,
)
from nn_dk.convergence_reporting import render_convergence_report
from nn_dk.native import extract_observables, physical_observable_weights
from nn_dk.native_types import NativeBatch, NativeDKSpec
from nn_dk.types import Context, QueryBank, TargetSpec, TrajectoryBatch


def batch(seed=0, n=48, context=None, cells=8):
    context = context or Context(n_particles=32)
    spec = NativeDKSpec(n_cells=cells)
    states = np.random.default_rng(seed).dirichlet(np.ones(cells) * 20, size=(n, 3))
    return NativeBatch(np.array([0.0, 0.25, 0.5]), states, spec, context,
                       np.full(n, np.inf), tuple(f"{seed}:{i}" for i in range(n)))


def test_exact_sample_moments_and_whole_family_bootstrap():
    queries = QueryBank((0.25, 0.5))
    paths = batch()
    result = native_estimates(paths, queries, n_bootstrap=20, seed=18)
    y = extract_observables(paths, queries)
    np.testing.assert_allclose(result["mean"], y.mean(axis=0))
    np.testing.assert_allclose(result["scaled_covariance"], 32 * np.cov(y, rowvar=False))
    assert np.asarray(result["uncertainty"]["mean_ci95"]).shape == (8, 2)
    assert np.asarray(result["uncertainty"]["scaled_covariance_ci95"]).shape == (8, 8, 2)
    assert result["n_independent_paths"] == 48
    assert result == native_estimates(paths, queries, n_bootstrap=20, seed=18)
    json.dumps(result, allow_nan=False)


def test_all_stopped_initial_and_absorbed_paths_remain_in_moments():
    paths = batch(n=4)
    paths.stopped_at[:] = [0, 0.1, 0.4, np.inf]
    paths.states[0, 1:] = paths.states[0, 0]
    paths.states[1, 1:] = paths.states[1, 1]
    queries = QueryBank((0.25, 0.5))
    result = native_estimates(paths, queries, n_bootstrap=10)
    assert result["diagnostics"]["all_stopped_paths_included"]
    assert result["diagnostics"]["initially_stopped"] == 1
    assert result["stopping"]["cdf"] == [0.5, 0.75]
    assert result["stopping"]["restricted_mean_stopping_time"] == pytest.approx(0.25)
    np.testing.assert_allclose(result["mean"], extract_observables(paths, queries).mean(axis=0))
    paths.stopped_at[:] = 0
    paths.states[:, 1:] = paths.states[:, :1]
    assert native_estimates(paths, queries, 8)["stopping"]["probability_by_horizon"] == 1


def test_independent_and_paired_pool_policies_and_reordering():
    queries = QueryBank((0.25, 0.5))
    reference = batch()
    with pytest.raises(ValueError, match="overlapping"):
        compare_native(reference, reference, queries)
    with pytest.raises(ValueError, match="matching"):
        compare_native(batch(1), reference, queries, paired=True)
    prediction = copy.deepcopy(reference)
    prediction.states = prediction.states[::-1].copy()
    prediction.stopped_at = prediction.stopped_at[::-1].copy()
    prediction.family_ids = prediction.family_ids[::-1]
    paired = compare_native(prediction, reference, queries, n_bootstrap=20, paired=True)
    assert paired["errors"]["mean_normalized_rmse"] == 0
    assert paired["errors"]["scaled_covariance_normalized_rmse"] == 0
    np.testing.assert_allclose(paired["uncertainty"]["mean_normalized_ci95"], 0)
    assert paired["accuracy_gate"]["status"] == "unresolved"
    independent = batch(1)
    independent.states[:] = reference.states
    compared = compare_native(independent, reference, queries, n_bootstrap=20)
    assert compared["errors"]["mean_normalized_rmse"] == 0
    assert np.max(np.abs(compared["uncertainty"]["mean_normalized_ci95"])) > 0
    assert any(p["two_time"] for p in compared["errors"]["covariance_pairs"])
    json.dumps(compared, allow_nan=False)


def test_fixed_physical_normalization_equals_uniform_histogram_covariance():
    queries = QueryBank((0.25, 0.5))
    paths = batch()
    fixed = native_normalization(paths.spec, queries)
    weights = physical_observable_weights(paths.spec, queries)
    covariance = weights @ weights.T / paths.spec.n_cells
    np.testing.assert_allclose(np.asarray(fixed["mean"])[:4] ** 2, np.diag(covariance), atol=1e-14)
    np.testing.assert_allclose(np.asarray(fixed["mean"])[:4], np.repeat(np.sinc(np.array([1, 2]) / 8), 2))
    first = compare_native(batch(1), paths, queries, n_bootstrap=8)
    second = compare_native(batch(5), batch(6), queries, n_bootstrap=8)
    assert first["normalization"] == second["normalization"]


def test_context_target_and_query_mismatch_are_rejected():
    queries = QueryBank((0.25,))
    reference = batch()
    with pytest.raises(ValueError, match="initialization law"):
        compare_native(batch(1, context=Context(cos_coeffs=(0.2, 0, 0), n_particles=32)), reference, queries)
    with pytest.raises(ValueError, match="Incompatible"):
        compare_native(batch(1, cells=16), reference, queries)
    legacy = TrajectoryBatch(reference.times, reference.states, TargetSpec(n_cells=8))
    with pytest.raises(ValueError, match="native stopped DK"):
        compare_native(legacy, reference, queries)
    with pytest.raises(ValueError, match="absent"):
        native_estimates(reference, QueryBank((0.125,)))


def test_reference_bias_and_mc_budgets_are_separate():
    queries = QueryBank((0.25,))
    prediction, reference = batch(1), batch()
    prediction.states[:] = reference.states[:] = 1 / 8
    unresolved = compare_native(prediction, reference, queries, n_bootstrap=16)
    assert unresolved["accuracy_gate"]["sampling_resolved"]
    assert unresolved["accuracy_gate"]["status"] == "unresolved"
    certified = compare_native(prediction, reference, queries, n_bootstrap=16,
                               reference_bias={"status": "resolved", "normalized_bound": 0.001})
    assert certified["accuracy_gate"]["status"] == "resolved_within_tolerance"
    unavailable = native_estimates(batch(n=1), queries)
    assert unavailable["status"] == "unavailable"
    assert not native_estimates(reference, queries, n_bootstrap=0)["uncertainty"]["available"]


def test_temporal_rates_need_reference_budget_signal_and_fixed_model():
    records = [{"step_size": dt, "error": 2 * dt, "standard_error": 1e-4}
               for dt in (1 / 8, 1 / 16, 1 / 32)]
    assert temporal_rate(records)["status"] == "unresolved"
    rate = temporal_rate(records, reference_bias_bound=0)
    assert rate["status"] == "resolved"
    assert rate["order"] == pytest.approx(1)
    noisy = [{**r, "standard_error": 0.1} for r in records]
    assert temporal_rate(noisy, reference_bias_bound=0)["status"] == "unresolved"
    plateau = [{**r, "error": 0.1} for r in records]
    assert temporal_rate(plateau, reference_bias_bound=0)["status"] == "unresolved"
    mixed = [{**r, "seed": i} for i, r in enumerate(records)]
    assert temporal_rate(mixed, reference_bias_bound=0)["status"] == "unresolved"


def test_report_is_immutable_strict_json_and_plots_seeds_separately(tmp_path):
    comparison = compare_native(batch(1), batch(), QueryBank((0.25, 0.5)), n_bootstrap=8)
    records = [{"case": "uniform", "stage": "A1", "training_mode": "oracle",
                "width": 32, "train_paths": 64, "seed": seed, "step_size": 1 / 128,
                "method": "both_learned", "comparison": comparison,
                "coefficient_diagnostics": {"covariance_relative_error": 0.1},
                "cost": {"prediction_seconds": 0.1, "training_seconds": 0.2}}
               for seed in (0, 1)]
    before = copy.deepcopy(records)
    manifest = {"seed": 5, "config": {"optimizer_seeds": [1, 0], "base_width": 32,
                                       "train_path_counts": [64], "step_sizes": [1 / 128]}}
    paths = render_convergence_report(records, {"status": "passed"}, manifest, tmp_path)
    assert records == before
    assert set(paths) == {"markdown", "json", "figure"}
    assert (tmp_path / "convergence_curves.png").stat().st_size > 1000
    saved = json.loads((tmp_path / "convergence.json").read_text())
    assert [r["seed"] for r in saved["records"]] == [0, 1]
    assert saved["plot_selection"]["seed"] == 1  # Configuration order, never an average.
    assert (tmp_path / "convergence_uniform.png").stat().st_size > 1000
    markdown = (tmp_path / "convergence.md").read_text()
    assert "unresolved" in markdown and "No positivity repair" in markdown
    assert "continuum" in markdown and "not Monte Carlo confidence intervals" in markdown
