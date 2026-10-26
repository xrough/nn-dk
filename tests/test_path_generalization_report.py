"""Postprocessing must distinguish paired bias uncertainty from deployed MC."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "report_path_generalization", Path(__file__).parents[1] / "analysis/report_path_generalization.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def _metric():
    return {"max_mean_error_in_reference_std": .01, "max_normalized_covariance_error": .01,
            "mean_sampling95_simultaneous_radius": .002, "covariance_sampling95_simultaneous_radius": .002,
            "mean_error_plus_sampling95": .012, "covariance_error_plus_sampling95": .012}


def test_partial_evaluation_is_rejected_before_opening_final_costs(tmp_path):
    directory = tmp_path / "evaluation"
    directory.mkdir()
    (directory / "evaluation_index.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="All evaluation groups"):
        report.load_complete(tmp_path, tmp_path / "does-not-exist.json")


def test_gate_and_amortization_never_turn_missing_mc_precision_into_a_win():
    reference = [{"mean_point": .001, "mean_sampling95": .001,
                  "covariance_point": .001, "covariance_sampling95": .001}]
    assert report.accuracy_gate([_metric()], reference) == "standalone-MC-not-assessed"
    assert report.accuracy_gate([_metric()], reference, standalone_mc_radius=.1) == "deployed-MC-unresolved"
    assert report.accuracy_gate([_metric()], reference, standalone_mc_radius=0) == "finite-reference-diagnostic-only"
    unresolved = report.break_even(100, .001, .01, candidate_gate="sampling-unresolved",
                                    baseline_gate="finite-reference-diagnostic-only")
    assert unresolved["queries"] is None
    measured = report.break_even(100, .001, .011, candidate_gate="finite-reference-diagnostic-only",
                                  baseline_gate="finite-reference-diagnostic-only")
    assert measured["queries"] == 10000
    no_saving = report.break_even(100, .02, .01, candidate_gate="finite-reference-diagnostic-only",
                                   baseline_gate="finite-reference-diagnostic-only")
    assert no_saving["queries"] is None


def test_standalone_candidate_mc_is_recomputed_in_physical_reference_units(tmp_path):
    reference_path = tmp_path / "reference.npz"
    np.savez(reference_path, many_reference_variance=np.array([1., 4.]))
    candidate_path = tmp_path / "candidate.npz"
    covariance = np.diag([1., 4.])[None]
    difference = np.array([[[0., 1.], [1., 0.]]])
    content = {bank + "_bootstrap_mean": np.array([[[.5, 0.]], [[-.5, 0.]]]) for bank in report.BANKS}
    content.update({bank + "_bootstrap_scaled_covariance": np.stack((covariance + difference, covariance - difference))
                    for bank in report.BANKS})
    content.update(many_bootstrap_mean=np.array([[1., 0.], [-1., 0.]]), many_mean=np.zeros(2),
                   many_bootstrap_variance=np.array([[1.1, 4.], [.9, 4.]]), many_variance=np.array([1., 4.]))
    np.savez(candidate_path, **content)
    summary = {"records": [{"candidate_mean": [0., 0.], "candidate_scaled_covariance": covariance[0].tolist(),
                            "reference_scaled_covariance": covariance[0].tolist()}]}
    evidence = {"design": {"reference_config": {"n_particles": 4}},
        "groups": [{"identity": {"group": "uniform__conditional__0"},
                    "reference_bootstrap": {"path": str(reference_path)}}],
        "records": [{"group": "uniform__conditional__0", "method": {"name": "native", "method": "native-euler"},
                     "bootstrap_artifact": {"path": str(candidate_path)},
                     "metrics": {bank: {"same_time": summary} for bank in report.BANKS}}]}
    rows = report.deployed_sampling(evidence)
    assert len(rows) == 3
    for row in rows[:2]:
        assert row["mean95_radius"] == pytest.approx(1.)
        assert row["covariance95_normalized_radius"] == pytest.approx(.5)
        assert "not paired" in row["scope"]
    assert rows[2]["mean95_radius"] == pytest.approx(1.)
    assert rows[2]["covariance95_normalized_radius"] == pytest.approx(.1)


def test_literal_path_moment_costs_are_not_extrapolated_from_small_batches():
    common = {"case": "dk_interacting_uniform", "role": "conditional", "bank": report.BANKS[0],
        "method": "compiled_native_euler", "paths": 16384, "chunk_size": 64, "dt": 2 ** -10,
        "horizon": 1 / 16, "median_seconds": .1, "p10_seconds": .09, "p90_seconds": .11,
        "initialization_seconds": .01, "physical_noise_seconds": .02,
        "new_bank_readout_component_seconds": [.02], "simulation_component_seconds": [.07],
        "aggregation_component_seconds": [.01], "cost_scope": "measured complete query"}
    evidence = {"path_timing": {"cases": [], "moment_tasks": [common]}, "payoff_timing": {"records": []}}
    rows = report.flatten_costs(evidence)
    assert rows[0]["case"] == "uniform"
    assert rows[0]["batch"] == 16384
    assert rows[0]["median_seconds"] == .1
    assert rows[0]["bank_readout_median_seconds"] == .02
    row = {"case": "uniform", "role": "conditional", "bank": report.BANKS[0],
           "method": "native-euler", "dt": 2 ** -10}
    assert report._matched_cost(row, rows) == rows[0]


def test_task_summary_keeps_joint_and_same_time_costs_distinct():
    identity = {"group": "uniform__conditional__0", "case": "uniform", "role": "conditional", "anchor": "0",
        "method_name": "path_dt10_seed0", "method": "prepared-learned-path", "optimizer_seed": 0,
        "initialization": None, "dt": 2 ** -10, "depth": 3, "lga_max_step": None,
        "bank": report.BANKS[0]}
    moments = [{**identity, "task": task, "time_index": 0, **_metric()}
               for task in ("same_time", "joint_time_12")]
    costs = [{"benchmark": "path-moments", "case": "uniform", "method": "learned_dense_path",
        "bank": report.BANKS[0], "task": "conditional_same_time_moments", "batch": 16384,
        "median_seconds": .5, "scope": "literal same-time query"}]
    deployed = [{**identity, "time_index": 0, "mean95_radius": .03, "covariance95_normalized_radius": .04}]
    rows = report.task_summary_rows({"moments": moments, "deployed_sampling": deployed}, costs)
    assert rows[0]["literal_query_median_seconds"] == .5
    assert rows[0]["standalone_covariance95_radius"] == .04
    assert rows[1]["literal_query_median_seconds"] is None
    assert rows[1]["standalone_covariance95_radius"] is None
