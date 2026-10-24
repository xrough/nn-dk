"""Artifact-only BSDE analysis must reject partial/mismatched scientific runs."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "analysis" / "summarize_bsde_batch.py"
_SPEC = importlib.util.spec_from_file_location("bsde_findings_analysis", _PATH)
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _zeros(shape):
    return [_zeros(shape[1:]) for _ in range(shape[0])] if shape else 0.


def _comparison(config, anchors=None):
    prefix = () if anchors is None else (anchors,)
    return {"errors": {"mean_normalized_signed": _zeros((*prefix, 2, 4)),
        "scaled_covariance_normalized_signed": _zeros((*prefix, 2, 4, 4)),
        "mean_normalized_rmse": 0., "scaled_covariance_normalized_rmse": 0.,
        "maximum_mean_normalized_error": 0., "maximum_scaled_covariance_normalized_error": 0.,
        "scaled_variance_normalized_rmse": 0.},
        "uncertainty": {"available": True, "maximum_sampling_normalized_halfwidth": .1,
                        "maximum_contrast_normalized_ci_endpoint": .1},
        "accuracy_gate": {"status": "unresolved", "reference_bias_resolved": False},
        "minimum_predicted_conditional_covariance_eigenvalue": .1}


def _stop(config, count):
    return {"n_paths": count, "n_stopped": 0, "probability_by_horizon": 0.,
            "times": config["query_times"], "stopping_cdf": [0., 0.],
            "all_stopped_paths_included": True}


def _completed_fixture(directory, seeds=(0,)):
    config = {"cases": ["uniform"], "optimizer_seeds": list(seeds), "query_times": [.25, .5],
        "dt": 1/128, "reference_dt": 1/1024, "train_anchors": 2, "train_continuations": 3,
        "validation_anchors": 2, "validation_continuations": 2,
        "conditional_anchors": 2, "conditional_paths": 3, "outer_paths": 4,
        "ou_test_anchors": 3, "refinement_paths": 2, "bootstrap_samples": 8,
        "training": {"epochs": 2}}
    manifest = {"workflow": "conditional-bsde-moments-v1", "status": "complete", "config": config,
        "source_hash": analysis._source_hash(), "config_hash": analysis._stable_hash(config)}
    provenance = {key: manifest[key] for key in ("source_hash", "config_hash")}
    audit = {"complete": True, "passed": True, "n_independent_families": 2,
        "reference_steps": [1/1024, 1/2048, 1/4096], "maximum_mass_error": 1e-15,
        "minimum_saved_mass": .01, "reference_bias": {"status": "uncertified"}, "seconds": .2}
    calibration = {**provenance, "complete": True, "passed": True, "status": "calibrated",
        "brownian": {"complete": True, "passed": True}, "ou": {"complete": True, "passed": True}, "native_refinement": {"uniform": audit}}
    records = []
    for problem in ("ou", "dk"):
        for seed in seeds:
            path = directory/"checkpoints"/problem/f"seed_{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"opaque checkpoint {problem} seed {seed}".encode())
            metadata = {"complete": True, "problem": problem, "seed": seed,
                "kind": "native-dk-bsde-moments-v1" if problem == "dk" else "linear-ou-bsde-moments-v1",
                "training_anchors": 2, "validation_anchors": 2,
                "training_paths": 6, "validation_paths": 4,
                "epochs": 2, "history": [{"epoch": 1}, {"epoch": 2}]}
            _write(path.with_suffix(".json"), metadata)
            identity = {"problem": problem, "seed": seed, "checkpoint_sha256": analysis._hash_file(path)}
            record = {**provenance, "complete": True, "problem": problem, "seed": seed,
                "times": config["query_times"], "step_size": config["dt"], "model": identity,
                "training": metadata}
            if problem == "ou":
                record.update(n_test_anchors=3, **{key: _comparison(config, 3)
                    for key in ("total_error", "neural_error", "time_error")})
            else:
                value = {key: _comparison(config, 2 if key.startswith("conditional") else None)
                    for key in ("conditional_total_error", "conditional_neural_error",
                                "ensemble_total_error", "ensemble_neural_error", "ensemble_time_error")}
                for kind, count in (("conditional", 6), ("ensemble", 4)):
                    value[f"{kind}_stopping"] = {level: _stop(config, count) for level in ("coarse", "fine")}
                    value[f"{kind}_reference_diagnostics"] = {"all_stopped_paths_included": True,
                        "maximum_mass_error": 1e-15, "minimum_saved_mass": .01}
                value["cost"] = {"reference_paths": 4, "n_initial_anchors": 4,
                    "conditional_reference": {"reference_paths": 6, "n_initial_anchors": 2}}
                value["reference_bias"] = {"status": "uncertified"}
                record["cases"] = {"uniform": value}
            records.append(record)
    _write(directory/"manifest.json", manifest)
    _write(directory/"config.json", config)
    _write(directory/"calibration.json", calibration)
    _save_records(directory, records)
    return records


def _save_records(directory, records):
    _write(directory/"evaluations.json", records)
    for record in records:
        identity = analysis._stable_hash(record["model"])[:20]
        path = directory/"evaluation"/f"{record['problem']}_seed{record['seed']}_{identity}.json"
        _write(path, record)


def test_partial_bsde_run_rejected_before_reading_outputs(tmp_path):
    _write(tmp_path/"manifest.json", {"status": "budget-exhausted"})
    with pytest.raises(ValueError, match="status='complete'"):
        analysis.summarize(tmp_path)
    assert not (tmp_path/"report").exists()


def test_complete_summary_preserves_optimizer_seeds_and_stopping_bounds(tmp_path):
    _completed_fixture(tmp_path, seeds=(3, 7))
    paths = analysis.summarize(tmp_path)
    result = json.loads(Path(paths["summary"]).read_text())
    assert result["model_count"] == 4
    assert result["comparison_count"] == 16
    assert {model["seed"] for model in result["models"]} == {3, 7}
    model = next(m for m in result["models"] if m["problem"] == "dk")
    stop = model["cases"]["uniform"]["ensemble_stopping"]["fine"]
    assert stop["zero_event_one_sided95_probability_upper"] > 0
    assert model["cases"]["uniform"]["decomposition"]["maximum_absolute_signed_identity_residual"] == 0
    assert "not additive" in result["interpretation"]["error_decomposition"]
    assert Path(paths["table"]).read_text().count("\n") == 17
    assert b"\r\n" not in Path(paths["table"]).read_bytes()


def test_summary_rejects_modified_checkpoint_bytes(tmp_path):
    _completed_fixture(tmp_path)
    (tmp_path/"checkpoints"/"dk"/"seed_0.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        analysis.summarize(tmp_path)


def test_summary_rejects_silently_reduced_test_allocation(tmp_path):
    records = _completed_fixture(tmp_path)
    records[1]["cases"]["uniform"]["ensemble_stopping"]["fine"]["n_paths"] = 3
    _save_records(tmp_path, records)
    with pytest.raises(ValueError, match="Reference path allocation"):
        analysis.summarize(tmp_path)


def test_summary_rejects_missing_optimizer_seed(tmp_path):
    records = _completed_fixture(tmp_path, seeds=(0, 1))
    _save_records(tmp_path, records[:-1])
    with pytest.raises(ValueError, match="problems/optimizer seeds"):
        analysis.summarize(tmp_path)


def test_signed_decomposition_is_checked_without_adding_rmse():
    config = {"query_times": [.25, .5]}
    neural = _comparison(config)
    time_error = copy.deepcopy(neural)
    total = copy.deepcopy(neural)
    neural["errors"]["mean_normalized_signed"][0][0] = 2.
    time_error["errors"]["mean_normalized_signed"][0][0] = -1.
    total["errors"]["mean_normalized_signed"][0][0] = 1.
    assert analysis._decomposition(total, neural, time_error)["maximum_absolute_signed_identity_residual"] == 0
    total["errors"]["mean_normalized_signed"][0][0] = 3.
    with pytest.raises(ValueError, match="Total signed contrast"):
        analysis._decomposition(total, neural, time_error)


def test_variance_and_same_time_cross_mode_errors_remain_separate():
    config = {"query_times": [.25, .5]}
    value = _comparison(config)
    value["errors"]["scaled_covariance_normalized_signed"][0][0][0] = 2.
    value["errors"]["scaled_covariance_normalized_signed"][0][0][1] = 3.
    value["uncertainty"]["mean_normalized_contrast_ci95"] = [[[-.1, .2]]*4]*2
    result = analysis.component_summary(value, config["query_times"])
    assert result["by_time"]["0.25"]["variance_rmse"] == 1.
    assert result["by_time"]["0.25"]["same_time_cross_mode_covariance_rmse"] == pytest.approx(3/12**.5)
    assert result["mean_ci95_max_abs_endpoint"] == .2
    assert result["accuracy_gate"]["status"] == "unresolved"


def test_summary_rejects_changed_numerical_source_hash(tmp_path):
    _completed_fixture(tmp_path)
    manifest = json.loads((tmp_path/"manifest.json").read_text())
    manifest["source_hash"] = "a different numerical implementation"
    _write(tmp_path/"manifest.json", manifest)
    with pytest.raises(ValueError, match="Current numerical package source"):
        analysis.summarize(tmp_path)


def test_summary_rejects_negative_recorded_covariance(tmp_path):
    records = _completed_fixture(tmp_path)
    records[1]["cases"]["uniform"]["conditional_total_error"]["minimum_predicted_conditional_covariance_eigenvalue"] = -.1
    _save_records(tmp_path, records)
    with pytest.raises(ValueError, match="Negative predicted conditional covariance"):
        analysis.summarize(tmp_path)
