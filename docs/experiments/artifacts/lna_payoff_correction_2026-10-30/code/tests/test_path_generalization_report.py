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


def test_total_costs_charge_training_data_and_restoration_once_per_deployment(tmp_path):
    pools = {}
    for role, seconds in (("train", 1.), ("validation", 2.)):
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps({"generation_seconds": seconds}))
        pools[role] = {"metadata_path": str(path)}
    evidence = {"fit_lock": {"input_banks": pools}, "path_timing": {"cases": []},
        "payoff_timing": {"cold_restoration": [{"name": "bank_trunk-transfer_seed2", "median_seconds": .5}]}}
    costs = [{"benchmark": "payoff", "case": "uniform", "method": "new-payoff-statistics-head",
        "task": "conditional_same_time_moments", "bank": "bank", "batch": 1, "optimizer_seed": 2,
        "initialization": "trunk-transfer", "fresh_training_seconds": 10., "historical_training_seconds": 20.,
        "median_seconds": .01, "scope": "literal same-time query"}]
    rows = report.total_cost_rows(evidence, costs, query_counts=(1, 100))
    assert rows[0]["fresh_fitting_seconds_once"] == rows[1]["fresh_fitting_seconds_once"] == 10.
    assert rows[0]["shared_current_data_generation_seconds_once"] == 3.
    assert rows[1]["online_seconds_all_calls"] == 1.
    assert rows[1]["existing_model_known_component_seconds"] == 1.5
    assert rows[1]["new_head_known_component_seconds"] == 14.5
    assert rows[1]["including_historical_fit_known_component_seconds"] == 34.5
    assert rows[1]["historical_data_generation_recorded"] is False
    assert rows[1]["complete_end_to_end_cost"] is False
    assert "not a fully costed" in rows[1]["cost_interpretation"]


def test_total_path_moment_cost_includes_recorded_fresh_inputs_per_call(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"generation_seconds": 1.}))
    evidence = {"fit_lock": {"input_banks": {"train": {"metadata_path": str(path)}}},
                "path_timing": {"cases": []}, "payoff_timing": {"cold_restoration": []}}
    costs = [{"benchmark": "path-moments", "case": "uniform", "method": "compiled_native_euler",
        "task": "conditional_same_time_moments", "bank": "bank", "batch": 16384,
        "median_seconds": .1, "initialization_seconds_separate": .02,
        "physical_noise_seconds_separate": .03, "scope": "complete query"}]
    row = report.total_cost_rows(evidence, costs, query_counts=(100,))[0]
    assert row["batch_per_call"] == 16384
    assert row["online_seconds_all_calls"] == 10.
    assert row["online_with_recorded_inputs_seconds_all_calls"] == pytest.approx(15.)
    assert row["fresh_fitting_seconds_once"] == 0.


def test_uncached_compiler_setup_never_charges_the_same_build_twice(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"generation_seconds": 1.}))
    compiler = {"available": True, "complete_setup_seconds": 1.5,
                "compile_seconds_this_process": 1.2, "compile_seconds": 1.2}
    evidence = {"fit_lock": {"input_banks": {"train": {"metadata_path": str(path)}}},
                "path_timing": {"cases": [], "compiled_classical_control": compiler},
                "payoff_timing": {"cold_restoration": []}}
    costs = [{"benchmark": "path-moments", "case": "uniform", "method": "compiled_native_euler",
        "task": "conditional_same_time_moments", "bank": "bank", "batch": 16384,
        "median_seconds": .1, "initialization_seconds_separate": .02,
        "physical_noise_seconds_separate": .03, "scope": "complete query"}]
    row = report.total_cost_rows(evidence, costs, query_counts=(1,))[0]
    assert row["restoration_seconds_once"] == pytest.approx(.3)
    assert row["uncached_compiled_control_build_seconds_once"] == 1.2
    assert row["including_historical_fit_known_component_seconds"] == pytest.approx(1.65)
    compiler["compile_seconds_this_process"] = 2.
    with pytest.raises(ValueError, match="setup is smaller"):
        report.total_cost_rows(evidence, costs, query_counts=(1,))


def test_nested_identity_paths_preserve_literal_python_filenames():
    assert report._nested({"identity": {"helpers": {"file.py": "abc"}}}, "identity.helpers.file.py") == "abc"


def _provenance_fixture(tmp_path):
    directory = tmp_path / "audit"
    directory.mkdir()
    sources = {}
    for name in ("compiled_native.py", "native_euler_control.cpp", "analysis.py"):
        path = directory / name
        path.write_text(name)
        sources[name] = {"path": str(path), "sha256": report.sha(path)}
    library = directory / "library.so"
    library.write_bytes(b"exact original compiled runtime")
    identity_path = directory / "identity.json"
    identity_path.write_text(json.dumps({"helpers": {"analysis.py": sources["analysis.py"]["sha256"]}}))
    packet = {"complete": True, "package_source_hash": "package", "package_source_unchanged": True,
        "archived_sources": sources,
        "experiment_identity_checks": [{"match": True, "source_name": "analysis.py",
            "expected_sha256": sources["analysis.py"]["sha256"],
            "archived_sha256": sources["analysis.py"]["sha256"],
            "identity_path": str(identity_path), "identity_sha256": report.sha(identity_path),
            "field": "helpers.analysis.py"}],
        "compiled_wrapper_bridge": {"historical_timing_rewritten": False,
            "library_checksum_verified": True, "driver_abspath_equals_previous_resolved_path_on_this_machine": True,
            "historical_wrapper_sha256": sources["compiled_native.py"]["sha256"],
            "cpp_sha256": sources["native_euler_control.cpp"]["sha256"],
            "library_path": str(library), "library_sha256": report.sha(library)}}
    (directory / "code_provenance.json").write_text(json.dumps(packet))
    return packet


def test_analysis_code_provenance_requires_exact_archived_source_and_binary(tmp_path):
    packet = _provenance_fixture(tmp_path)
    inputs = {}
    report.verify_code_provenance(tmp_path, "package", inputs, {})
    assert str(tmp_path / "audit/code_provenance.json") in inputs
    Path(packet["archived_sources"]["analysis.py"]["path"]).write_text("changed source")
    with pytest.raises(ValueError, match="Evidence artifact changed"):
        report.verify_code_provenance(tmp_path, "package", {}, {})


def test_analysis_code_provenance_rejects_rewritten_historical_timing(tmp_path):
    packet = _provenance_fixture(tmp_path)
    packet["compiled_wrapper_bridge"]["historical_timing_rewritten"] = True
    (tmp_path / "audit/code_provenance.json").write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="bridge is incomplete"):
        report.verify_code_provenance(tmp_path, "package", {}, {})


def test_cost_overview_separates_gaussian_accuracy_audit_from_cheap_coarse_query():
    costs = [{"benchmark": "payoff", "method": "native-linear-Gaussian-RK4",
              "dt": step, "task": "conditional_same_time_moments", "batch": 1,
              "median_seconds": seconds} for step, seconds in ((2 ** -7, .003), (2 ** -10, .03))]
    _, table = report.cost_overview_tables(costs)
    assert "One four-observable bank at three horizons; same-time covariance only" in table
    assert "maximum step 1/128 | 0 | 3–3" in table
    assert "fine 1/1024 audit | 0 | 30–30" in table
    assert "\n\n|" not in table


def _historical_timing_fixture(tmp_path):
    directory = tmp_path / "historical/checkpoints"
    directory.mkdir(parents=True)
    path = directory / "model.pt"
    path.write_bytes(b"the immutable historical weights")
    metadata = path.with_suffix(".json")
    metadata.write_text(json.dumps({"source_hash": "old-package", "target_id": "old-target",
                                    "training_seconds": 2076.7}))
    manifest_path = directory.parent / "manifest.json"
    manifest_path.write_text(json.dumps({"status": "complete", "source_hash": "old-package",
        "artifact_lock": {"checkpoints/model.pt": report.sha(path),
                          "checkpoints/model.json": report.sha(metadata)}}))
    checkpoint = {"checkpoint_path": str(path), "checkpoint_sha256": report.sha(path),
        "current_source_hash": "current-package", "historical_source_hash": "old-package",
        "original_target_id": "old-target", "kind": "path"}
    return {"cases": [{"checkpoint": checkpoint}]}, metadata, manifest_path


def test_optional_long_timing_metadata_is_locked_and_included_in_report_inputs(tmp_path):
    timing, metadata, manifest = _historical_timing_fixture(tmp_path)
    inputs = {}
    report.verify_timing_checkpoints(timing, {"candidate_plan": []}, "current-package", inputs, {})
    assert inputs[str(metadata)] == report.sha(metadata)
    assert inputs[str(manifest)] == report.sha(manifest)
    metadata.write_text(json.dumps({"source_hash": "old-package", "target_id": "old-target",
                                    "training_seconds": 1.}))
    with pytest.raises(ValueError, match="Evidence artifact changed"):
        report.verify_timing_checkpoints(timing, {"candidate_plan": []}, "current-package", {}, {})


def test_timing_checkpoint_weights_and_historical_source_are_verified(tmp_path):
    timing, _, manifest_path = _historical_timing_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["source_hash"] = "wrong-historical-package"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="historical artifact lock"):
        report.verify_timing_checkpoints(timing, {"candidate_plan": []}, "current-package", {}, {})
    Path(timing["cases"][0]["checkpoint"]["checkpoint_path"]).write_bytes(b"replaced historical weights")
    with pytest.raises(ValueError, match="Evidence artifact changed"):
        report.verify_timing_checkpoints(timing, {"candidate_plan": []}, "current-package", {}, {})
