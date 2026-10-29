"""Scientific normalization and honest deployment-cost accounting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "payoff_repair_report", Path(__file__).parents[1] / "analysis/report_payoff_repair.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def _metric_row():
    return {"time_index": 0, "max_mean_error_in_reference_std": .01,
            "max_normalized_covariance_error": .02,
            "mean_sampling95_simultaneous_radius": .002,
            "covariance_sampling95_simultaneous_radius": .003,
            "mean_error_plus_sampling95": .012, "covariance_error_plus_sampling95": .023,
            "reference_mean": [0., 1., 2., 3.], "candidate_mean": [0., 1., 2., 3.],
            "reference_scaled_covariance": np.diag([2., 3., 4., 5.]).tolist(),
            "candidate_scaled_covariance": np.diag([2., 3., 2., 10.]).tolist(),
            "normalized_mean_difference": [0., 0., 0., 0.],
            "normalized_covariance_difference": np.zeros((4, 4)).tolist()}


def test_variance_collapse_ratios_retain_literal_components_and_each_seed():
    bank = "terminal3-integral1"
    row = _metric_row()
    evidence = {"cfg": SimpleNamespace(query_times=(.25, .5, 1.)), "plan": {"banks": [
        {"name": bank, "labels": ["terminal-sin3", "terminal-cos3", "integral-sin1", "integral-cos1"]}]},
        "groups": [], "records": [{"identity": {"group": "uniform__conditional__0",
            "name": f"head_seed{seed}", "bank": bank},
            "method": {"family": "repaired", "seed": seed},
            "metrics": {"paired": False, "records": [row]}} for seed in (0, 1, 2)]}
    tables = report.accuracy_tables(evidence)
    assert len(tables["accuracy"]) == 3
    integral_rows = [r for r in tables["components"] if "integral" in r["observable"]]
    assert {r["variance_ratio"] for r in integral_rows} == {.5, 2.}
    assert {r["seed"] for r in integral_rows} == {0, 1, 2}
    assert len(tables["covariances"]) == 3 * 16


def test_sampling_radii_use_physical_mean_and_N_scaled_covariance_consistently():
    n = 2048
    std = np.array([[2., 4.]])
    products = std[..., :, None] * std[..., None, :]
    mean = np.zeros((1, 2))
    covariance = np.zeros((1, 2, 2))
    bootstrap_mean = np.array([mean + .02 * std, mean - .02 * std])
    bootstrap_covariance = np.array([covariance + .03 * n * products,
                                     covariance - .03 * n * products])
    mr, cr = report._sampling_radii(mean, covariance, bootstrap_mean, bootstrap_covariance,
                                    std, products, n)
    np.testing.assert_allclose(mr, .02)
    np.testing.assert_allclose(cr, .03)


def test_covariance_curve_starts_against_selected_mean_not_obsolete_Gamma():
    metadata = {"initial_validation": {"mean": {"objective": .4, "raw_rms": .2},
                "covariance": {"objective": 1e-12, "raw_rms": 1e-8}}}
    diagnostic = {"validation": {"selected_mean_covariance_initial": {"covariance": {
        "normalized_objective": .3, "raw_global_rms": .014}}}}
    start = report._stage_initial_validation(metadata, diagnostic, "covariance")
    assert start["validation_raw_rms"] == .014
    assert start["validation_objective"] == .3
    assert start["global_step"] == 1030
    assert "selected frozen mean" in start["scope"]
    assert report._stage_initial_validation(metadata, diagnostic, "mean")["validation_raw_rms"] == .2


def test_final_postfit_diagnostic_rejects_missing_sixth_fit(monkeypatch):
    monkeypatch.setattr(report, "source_hash", lambda: "source")
    lock = {"models": [{"variant": "repaired", "bank": bank, "seed": seed,
                        "checkpoint_sha256": f"{bank}-{seed}"}
                       for bank in report.BANKS for seed in (0, 1, 2)]}
    records = [{"bank": r["bank"], "seed": r["seed"], "checkpoint_sha256": r["checkpoint_sha256"],
                "strict_restore": True, "selected_mean_unchanged": True,
                "selected_mean_hashes": {"best_mean_state": "same", "best_state": "same"},
                "validation": {"selected_mean_covariance_initial": {"covariance": {}}}}
               for r in lock["models"]]
    controls = [{"variant": variant, "bank": bank, "seed": 0, "checkpoint_sha256": f"{bank}-{variant}",
                 "strict_restore": True} for bank in report.BANKS for variant in ("v1-none", "v1-balanced")]
    lock["models"] += controls
    complete = {"complete": True, "source_hash": "source", "records": records, "pending": [],
                "controls": controls, "pending_controls": []}
    report.verify_postfit_diagnostic(complete, lock)
    with pytest.raises(ValueError, match="All six"):
        report.verify_postfit_diagnostic({**complete, "records": records[:-1]}, lock)


def test_low_paired_error_does_not_pass_deployed_MC_gate():
    metric = _metric_row()
    metric.update(mean_error_plus_sampling95=.012, covariance_error_plus_sampling95=.023)
    reference = [{**metric, "mean_error_plus_sampling95": .001,
                  "covariance_error_plus_sampling95": .002}]
    assert report.accuracy_gate([metric], [{"mean95_radius": .001, "covariance95_radius": .03}],
                                reference) == "standalone-MC-unresolved"
    assert report.accuracy_gate([metric], [{"mean95_radius": .001, "covariance95_radius": .002}],
                                reference) == "finite-reference-diagnostic-only"
    too_inaccurate = {**metric, "max_normalized_covariance_error": .06}
    assert report.accuracy_gate([too_inaccurate], [], reference) == "point-error-above-target"


def test_total_cost_counts_fitting_data_restore_first_cache_and_later_warm_calls():
    item = {"slot": "uniform/head/conditional/1", "case": "uniform", "name": "head",
            "family": "repaired", "bank": "bank", "role": "conditional", "batch": 1,
            "median_seconds": .1, "first_prediction_seconds": .5, "samples_seconds": [.1]}
    evidence = {"timing": {"records": [item], "loading": [
        {"case": "uniform", "name": "head", "seconds": .2}]}}
    training = [{"case": "both", "name": "head", "training_seconds": 20.,
                 "prior_trunk_training_seconds": 0., "historical": False}]
    rows, totals = report.cost_tables(evidence, training, [{"generation_seconds": 1.}])
    assert rows[0]["known_offline_seconds"] == 21.
    first = next(r for r in totals if r["query_calls"] == 1)
    hundred = next(r for r in totals if r["query_calls"] == 100)
    assert first["total_known_seconds"] == pytest.approx(21.7)
    assert hundred["total_known_seconds"] == pytest.approx(21.7 + 99 * .1)
    assert first["subsequent_warm_calls_seconds"] == 0


def test_native_cost_charges_compile_and_cache_setup_without_double_counting_build():
    item = {"slot": "uniform/native/conditional", "case": "uniform", "name": "native-Euler",
            "family": "native-Euler", "bank": "bank", "role": "conditional", "batch": 16384,
            "median_seconds": .1, "first_prediction_seconds": .5, "samples_seconds": [.1]}
    evidence = {"timing": {"records": [item], "loading": [], "compiled_control": [
        {"case": "uniform", "compile_seconds": 2., "compile_seconds_this_process": .1,
         "setup_seconds_this_process": .3}]}}
    costs, totals = report.cost_tables(evidence, [], [])
    assert costs[0]["cold_compilation_seconds"] == 2.
    assert costs[0]["compiled_cache_setup_seconds"] == pytest.approx(.2)
    assert next(r for r in totals if r["query_calls"] == 1)["total_known_seconds"] == pytest.approx(2.7)


def test_ensemble_total_conservatively_retains_earlier_B1_cache_creation_probe():
    b1 = {"slot": "uniform/head/conditional/1", "case": "uniform", "name": "head",
          "family": "untrained-repaired", "bank": "bank", "role": "conditional", "batch": 1,
          "median_seconds": .01, "first_prediction_seconds": .5, "samples_seconds": [.01]}
    ensemble = {**b1, "slot": "uniform/head/ensemble", "role": "ensemble", "batch": 16384,
                "median_seconds": .1, "first_prediction_seconds": 2.}
    evidence = {"timing": {"records": [b1, ensemble], "loading": [
        {"case": "uniform", "name": "head", "seconds": .2}]}}
    rows, totals = report.cost_tables(evidence, [], [])
    assert rows[0]["cache_creation_B1_probe_seconds"] == 0.
    assert rows[1]["cache_creation_B1_probe_seconds"] == .5
    value = next(r for r in totals if r["role"] == "ensemble" and r["query_calls"] == 1)
    assert value["total_known_seconds"] == pytest.approx(.2 + .5 + 2.)


def test_no_crossover_is_reported_when_accuracy_or_uncertainty_fails():
    candidate = {"family": "repaired", "case": "uniform", "bank": "bank", "role": "conditional",
                 "name": "head", "batch": 1, "median_seconds": .001}
    control = {"family": "LGA", "case": "uniform", "bank": "bank", "role": "conditional",
               "name": "LGA_dt7", "batch": 1, "median_seconds": .005}
    rows = report.amortization_tables([], [], [], [candidate, control])
    assert len(rows) == 1 and rows[0]["queries"] is None
    assert rows[0]["candidate_gate"] == "unsupported"


def test_markdown_table_rows_are_contiguous():
    text = report._table(["Method", "Cost"], [["A", 1], ["B", 2]])
    assert "\n\n" not in text
    assert len(text.splitlines()) == 4


def test_report_refuses_incomplete_comparison_before_reading_final_arrays(tmp_path):
    directory = tmp_path / "comparison"
    directory.mkdir()
    (directory / "preregistration.json").write_text(json.dumps({"source_hash": "other"}))
    (directory / "accuracy_index.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="complete matching"):
        report.load_complete(tmp_path)


def test_compact_archive_compression_is_deterministic_and_restores_exact_bytes(tmp_path):
    from analysis import archive_payoff_repair as archive
    source = tmp_path / "large.json"
    source.write_text(json.dumps({"history": ["literal source bytes"] * 10_000}) + "\n")
    a, pa = archive._copy_exact(source, tmp_path / "first" / "large.json")
    b, pb = archive._copy_exact(source, tmp_path / "second" / "large.json")
    assert pa["compressed"] and pb["compressed"]
    assert pa["sha256"] == pb["sha256"]
    assert archive._restored_sha(a, True) == archive._restored_sha(b, True) == report.sha(source)


def test_archive_verification_rejects_mutated_stored_evidence(tmp_path):
    from analysis import archive_payoff_repair as archive
    source = tmp_path / "source.json"
    source.write_text('{"literal": true}\n')
    output = tmp_path / "archive"
    target, packet = archive._copy_exact(source, output / "evidence.json")
    (output / "README.md").write_text("Archive\n")
    manifest = {"kind": archive.KIND, "complete": True,
                "copied_artifacts": {"evidence.json": packet},
                "generated_files": {"README.md": report.sha(output / "README.md")},
                "full_run_inventory": {}}
    archive._json(output / "archive_manifest.json", manifest)
    assert archive.verify_archive(output)["copied_artifacts_verified"] == 1
    target.write_text("changed\n")
    with pytest.raises(ValueError, match="evidence changed"):
        archive.verify_archive(output)
