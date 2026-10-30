"""Task-matched costs, sampling units and conservative harder-regime gates."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "payoff_regimes_report", Path(__file__).parents[1]/"analysis/report_payoff_regimes.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def test_new_regime_costs_charge_one_fit_full_data_and_real_cache_probe():
    b1 = dict(slot="b1", regime="r", profile="uniform", name="head", family="repaired",
              role="conditional", batch=1, median_seconds=.001, first_prediction_seconds=.02)
    ensemble = {**b1, "slot": "ensemble", "role": "ensemble", "batch": 16384,
                "median_seconds": .2, "first_prediction_seconds": .3}
    native = {**ensemble, "slot": "native", "name": "native", "family": "native-MC",
              "median_seconds": 2., "first_prediction_seconds": 3.}
    timing = {"records": [b1, ensemble, native], "loading": [
        dict(regime="r", profile="uniform", name="head", seconds=.1)],
        "compiled_control": dict(compile_seconds=2., setup_seconds_this_process=.5,
                                 compile_seconds_this_process=.2)}
    costs, totals = report.cost_tables(timing, {"models": [
        dict(case="r", name="head", training_seconds=10.)]}, {"r": 1.})
    assert costs[0]["known_offline_seconds"] == pytest.approx(11.1)
    assert costs[1]["known_offline_seconds"] == pytest.approx(11.12)
    assert costs[2]["known_offline_seconds"] == pytest.approx(2.3)
    first = next(r for r in totals if r["slot"] == "ensemble" and r["query_calls"] == 1)
    hundred = next(r for r in totals if r["slot"] == "ensemble" and r["query_calls"] == 100)
    assert first["total_known_seconds"] == pytest.approx(11.42)
    assert hundred["total_known_seconds"] == pytest.approx(11.42+99*.2)


def test_sampling_normalization_uses_reference_units_and_removes_N_scaling():
    n = 2048
    std = np.array([[2., 3.]])
    scale = std[..., :, None]*std[..., None, :]
    mean, covariance = np.zeros((1, 2)), np.zeros((1, 2, 2))
    bm = np.array([mean+.02*std, mean-.02*std])
    bc = np.array([covariance+.03*n*scale, covariance-.03*n*scale])
    mr, cr = report.normalized_sampling(mean, covariance, bm, bc, n*np.array([np.diag(std[0]**2)]), n)
    np.testing.assert_allclose(mr, .02)
    np.testing.assert_allclose(cr, .03)


def test_small_point_error_does_not_hide_sampling_or_reference_uncertainty():
    a = dict(max_mean_error_in_reference_std=.01, mean_error_plus_sampling95=.02,
             mean_sampling95_simultaneous_radius=.01)
    ref = dict(a, mean_error_plus_sampling95=.002)
    assert report.statistic_gate([a], [], [ref], "mean") == "finite-reference-diagnostic-pass"
    assert report.statistic_gate([dict(a, mean_sampling95_simultaneous_radius=.02)], [], [ref], "mean") == "sampling-component-above-1.25pct"
    assert report.statistic_gate([a], [dict(mean95_radius=.03)], [ref], "mean") == "standalone-MC-component-unresolved"
    assert report.statistic_gate([a], [], [dict(ref, mean_error_plus_sampling95=.04)], "mean") == "error-plus-uncertainty-unresolved"


def test_arithmetic_payback_is_distinct_from_accuracy_qualification():
    candidate = dict(known_offline_seconds=10., first_prediction_seconds=.2, median_seconds=.125)
    control = dict(known_offline_seconds=0., first_prediction_seconds=.2, median_seconds=.375)
    assert report.payback(candidate, control) == 41
    assert report.payback(control, candidate) is None


def test_physical_bounds_include_zero_killed_integral_and_variance_limit():
    bank = SimpleNamespace(terminal_weights=np.array([[0., 0.], [-1., 1.]]),
                           running_weights=np.array([[2., 4.], [0., 0.]]))
    bounds = report.physical_bounds(bank, .5, 0, 1., .5)
    assert bounds["physical_mean_lower"] == 0
    assert bounds["physical_mean_upper"] == 2
    assert bounds["variance_inside_physical_bounds"]
    assert not report.physical_bounds(bank, .5, 1, 1.1, 0.)["mean_inside_physical_bounds"]
    assert not report.physical_bounds(bank, .5, 1, 0., 2.)["variance_inside_physical_bounds"]


def test_stopping_interval_retains_zero_counts_without_zero_uncertainty():
    low, high = report.binomial_interval(0, 16384)
    assert low == pytest.approx(0., abs=1e-16)
    assert 0 < high < .001
    with pytest.raises(ValueError):
        report.binomial_interval(0, 0)


def test_model_summary_does_not_pool_optimizer_seeds():
    rows = [dict(regime="r", family="repaired", name=f"seed{s}", seed=s, kappa=3., horizon=1.,
                 max_mean_error_in_reference_std=.01+s*.01, max_normalized_covariance_error=.03,
                 mean_error_plus_sampling95=.04, covariance_error_plus_sampling95=.05) for s in (0, 1, 2)]
    result = report.model_summary(rows)
    assert len(result) == 3
    assert {r["seed"] for r in result} == {0, 1, 2}
    assert sorted(r["worst_mean"] for r in result) == [.01, .02, .03]


def test_markdown_table_rows_remain_contiguous():
    table = report._table(["a", "b"], [[1, 2], [3, 4]])
    assert "\n\n" not in table
    assert len(table.splitlines()) == 4


def test_seed_consistency_requires_every_optimizer_and_does_not_choose_a_winner():
    base = dict(regime="r", profile="uniform", role="conditional", group="g", time_index=2,
                time=1., statistic="mean", advantage=.02, lower95=.01, upper95=.03)
    rows = [{**base, "name": f"seed{s}"} for s in (0, 1, 2)]
    assert report.seed_consistency(rows, (0, 1, 2))[0]["all_seeds_NN_closer_pointwise95"]
    rows[-1]["lower95"] = -.001
    assert not report.seed_consistency(rows, (0, 1, 2))[0]["all_seeds_NN_closer_pointwise95"]
    with pytest.raises(ValueError, match="drop optimizer"):
        report.seed_consistency(rows[:-1], (0, 1, 2))


def test_complete_small_workflow_reaches_verified_report_without_dropping_seeds(tmp_path):
    """A real tiny run checks cross-module schemas; its timings are test fixtures."""
    from analysis import evaluate_payoff_regimes as evaluator
    from analysis import run_payoff_regimes as runner
    plan = runner.StudyPlan(horizons=(1/64,), kappas=(.75,), n_cells=16,
        width=2, train_paths=2, validation_paths=2, adam_steps=1, lbfgs_steps=0,
        validation_every=1, paths_per_group=8, refinement_paths=4, chunk_paths=2)
    runner.prepare(tmp_path, plan)
    evaluator.prepare(tmp_path)
    for seed in plan.optimizer_seeds:
        runner.fit_seed(tmp_path, seed)
    runner.lock(tmp_path)
    runner.generate_reference(tmp_path)
    evaluator.evaluate(tmp_path)
    evaluator.benchmark(tmp_path, final_timing=True)
    result = report.generate(tmp_path, independent_initials=False, validation_diagnostics=False)
    assert result["models"] == 3
    assert result["accuracy_records"] == 42
    assert result["timing_records"] == 44
    assert result["all_optimizer_seeds_retained"]
    assert len(list((tmp_path/"report").glob("*.png"))) == 7
    evidence = report.load_complete(tmp_path, independent_initials=False)
    assert len(evidence["records"]) == 42
    # A foreign identity remains invalid even if its caller updates file hashes.
    changed = report.read(tmp_path/"comparison/timing.json")
    changed["identity"]["fit_lock_sha256"] = "foreign"
    report.json_write(tmp_path/"comparison/timing.json", changed)
    with pytest.raises(ValueError, match="All declared"):
        report.load_complete(tmp_path, independent_initials=False)
