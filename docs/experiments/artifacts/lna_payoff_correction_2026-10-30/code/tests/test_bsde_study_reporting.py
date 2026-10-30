"""Complete factorial reports preserve uncertainty and reject partial provenance."""

import copy
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from nn_dk.bsde_study_config import bsde_study_preset
from nn_dk.bsde_study_evaluation import StudyModelDescriptor, test_bank_identity
from nn_dk.bsde_study_reporting import write_study_report
from nn_dk.types import stable_hash


def comparison(error=.02):
    return {"errors": {"mean_normalized_rmse": error,
                       "scaled_variance_normalized_rmse": error * 2,
                       "scaled_covariance_normalized_rmse": error * 3,
                       "maximum_mean_normalized_error": .2,
                       "maximum_scaled_covariance_normalized_error": .3,
                       "scaled_covariance_normalized_signed": np.full((3, 4, 4), .3).tolist()},
            "uncertainty": {"maximum_sampling_normalized_halfwidth": .1,
                            "maximum_contrast_normalized_ci_endpoint": .4},
            "accuracy_gate": {"status": "unresolved", "point_estimate_within_tolerance": False}}


def complete_records(directory, study):
    base, records = study.base_config, []
    for problem, architecture, allocation, seed in product(
            ("ou", "dk"), study.architectures, study.allocations, base.optimizer_seeds):
        anchors, children = allocation
        descriptor = StudyModelDescriptor(architecture, anchors, children, seed)
        checkpoint = directory / "checkpoints" / f"{problem}_{architecture}_{anchors}_{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"{problem}_{architecture}_{anchors}_{seed}".encode())
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        bank = test_bank_identity(base, problem)
        factors = descriptor.to_dict()
        identity = stable_hash({"bank": bank, "factors": factors, "checkpoint_sha256": sha})
        evaluation = {"problem": problem, "seed": seed, "complete": True,
                      "source_hash": "report-test-source", "config_hash": base.config_hash,
                      "model": {"checkpoint_sha256": sha}, "step_size": base.dt,
                      "times": list(base.query_times),
                      "training": {"complete": True, "training_anchors": anchors,
                                   "training_paths": study.training_paths,
                                   "validation_anchors": study.validation[0],
                                   "validation_paths": study.validation[0] * study.validation[1],
                                   "training_seconds": 3., "epochs": 2,
                                   "best_validation_loss": .1}}
        if problem == "ou":
            evaluation.update({key: comparison() for key in ("neural_error", "total_error", "time_error")})
        else:
            evaluation["cases"] = {}
            for case in base.cases:
                result = {key: comparison() for key in ("conditional_neural_error", "conditional_total_error",
                    "ensemble_neural_error", "ensemble_total_error", "ensemble_time_error")}
                stop = {"n_paths": base.outer_paths, "n_stopped": 1, "probability_by_horizon": 1 / base.outer_paths,
                        "zero_stop_one_sided_95_upper": None, "all_stopped_paths_included": True}
                diagnostic = {"maximum_mass_error": 1e-15, "minimum_saved_mass": .02,
                              "all_stopped_paths_included": True}
                for scope in ("conditional", "ensemble"):
                    result[f"{scope}_stopping"] = {"coarse": stop, "fine": stop}
                    result[f"{scope}_reference_diagnostics"] = diagnostic
                result["cost"] = {"reference_generation_seconds": 20.,
                                  "conditional_reference": {"reference_generation_seconds": 10.}}
                evaluation["cases"][case] = result
        records.append({"study_model_id": identity, "test_bank_id": bank, "factors": factors,
                        "study_config_hash": study.config_hash, "evaluation_config_hash": base.config_hash,
                        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha,
                        "training_view_fingerprint": f"{problem}/{anchors}/{children}",
                        "validation_fingerprint": f"{problem}/fixed-validation", "evaluation": evaluation})
    return records


def calibration(study):
    return {"complete": True, "passed": True, "config_hash": study.base_config.config_hash,
            "source_hash": "report-test-source", "status": "calibrated"}


def test_complete_report_includes_variance_uncertainty_and_separate_scopes(tmp_path, monkeypatch):
    monkeypatch.setattr("nn_dk.bsde_study_reporting.source_hash", lambda: "report-test-source")
    study = bsde_study_preset("bsde-study-smoke")
    records = complete_records(tmp_path, study)
    paths = write_study_report(tmp_path, study, calibration(study), records, [])
    assert set(paths) == {"markdown", "json", "csv", "figure"}
    payload = json.loads(Path(paths["json"]).read_text())
    assert len(payload["evaluations"]) == 12
    assert len(payload["comparison_rows"]) == 78
    assert payload["training_seconds_all_distinct_fits"] == 36
    assert len(payload["shared_reference_rows"]) == 8
    assert all(row["all_stopped_paths_included"] for row in payload["shared_reference_rows"])
    assert all(row["accuracy_gate"] == "unresolved" for row in payload["comparison_rows"])
    assert all(row["maximum_variance_error"] == .3 for row in payload["comparison_rows"])
    csv_bytes = Path(paths["csv"]).read_bytes()
    assert b"\r\n" not in csv_bytes
    with Path(paths["csv"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 78
    assert {row["scope"] for row in rows} == {"conditional", "ensemble"}
    assert {row["reference"] for row in rows} == {"training-step Euler", "continuous OU", "fine stopped DK"}
    text = Path(paths["markdown"]).read_text()
    assert "16 training trajectories" in text
    assert "4 initial states × 4 continuations" in text
    assert "**not accuracy certification**" in text
    assert "Max variance" in text and "Max contrast CI endpoint" in text
    assert "must not be summed" in text and "four fitted positive parameters" in text
    assert Path(paths["figure"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("change", ["missing", "duplicate", "source", "test-bank", "allocation",
                                      "training-view", "validation-view", "checkpoint", "identity", "calibration"])
def test_report_rejects_partial_and_incompatible_results(tmp_path, monkeypatch, change):
    monkeypatch.setattr("nn_dk.bsde_study_reporting.source_hash", lambda: "report-test-source")
    study = bsde_study_preset("bsde-study-smoke")
    records, calibrated = complete_records(tmp_path, study), calibration(study)
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1] = records[0]
    elif change == "source":
        records[0]["evaluation"]["source_hash"] = "changed"
    elif change == "test-bank":
        records[0]["test_bank_id"] = "other"
    elif change == "allocation":
        records[0]["evaluation"]["training"]["training_anchors"] = 500
    elif change == "training-view":
        records[3]["training_view_fingerprint"] = "other"
    elif change == "validation-view":
        records[1]["validation_fingerprint"] = "other"
    elif change == "checkpoint":
        Path(records[0]["checkpoint_path"]).write_bytes(b"changed checkpoint")
    elif change == "identity":
        records[0]["study_model_id"] = "other"
    elif change == "calibration":
        calibrated["complete"] = False
    with pytest.raises(ValueError):
        write_study_report(tmp_path, study, calibrated, records, [])
    assert not (tmp_path / "report").exists()


def test_paired_contrast_intervals_render_without_pooling_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr("nn_dk.bsde_study_reporting.source_hash", lambda: "report-test-source")
    study = bsde_study_preset("bsde-study-smoke")
    records = complete_records(tmp_path, study)
    baseline, candidate = records[0], records[3]
    contrast = {"baseline_model_id": baseline["study_model_id"], "candidate_model_id": candidate["study_model_id"],
                "test_bank_id": baseline["test_bank_id"], "optimizer_seed": 0, "problem": "ou",
                "case": "analytic", "scope": "conditional", "factor": "architecture",
                "baseline_architecture": "mlp", "candidate_architecture": "structured",
                "baseline_train_anchors": 4, "candidate_train_anchors": 4,
                "candidate_minus_baseline": {"mean_rmse": -.02, "variance_rmse": -.03},
                "paired_uncertainty": {"candidate_minus_baseline_ci95": {
                    "mean_rmse": [-.03, -.01], "variance_rmse": [-.05, -.01]}}}
    paths = write_study_report(tmp_path, study, calibration(study), records, {"contrasts": [contrast]})
    assert "[-0.03, -0.01]" in Path(paths["markdown"]).read_text()
    changed = copy.deepcopy(contrast)
    changed["optimizer_seed"] = 1
    with pytest.raises(ValueError, match="optimizer seed"):
        write_study_report(tmp_path, study, calibration(study), records, [changed])
