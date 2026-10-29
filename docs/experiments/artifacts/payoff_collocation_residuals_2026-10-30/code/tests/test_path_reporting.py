"""Every final path fit, numerical target and unresolved reference stays visible."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from nn_dk.path_config import PathConfig
from nn_dk.path_evaluation import evaluate_paths
from nn_dk.path_reporting import write_path_report


def fixture(tmp_path, monkeypatch):
    config = PathConfig(cases=("heat",), n_cells=8, optimizer_seeds=(0, 1),
                        train_paths=4, validation_paths=4, test_paths=4,
                        adam_steps=1, lbfgs_steps=0, n_bootstrap=8)
    monkeypatch.setattr("nn_dk.path_reporting.source_hash", lambda: "path-report-test-source")
    provenance, records, inventory = {}, [], []
    for case, dt, seed in product(config.cases, config.step_sizes, config.optimizer_seeds):
        checkpoint = tmp_path / f"{case}_{dt}_{seed}.pt"
        checkpoint.write_bytes(f"path checkpoint {case}, {dt}, seed {seed}".encode())
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        data = {"case": case, "dt": dt, "seed": seed, "problem": config.problem(case),
                "target_id": config.target_id(case, dt), "config_hash": config.config_hash,
                "source_hash": "path-report-test-source"}
        provenance[str(checkpoint.resolve())] = data
        inventory.append({"checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha})
        times = np.arange(round(config.horizon / dt) + 1) * dt
        states = np.full((config.test_paths, len(times), config.n_cells), 1 / config.n_cells)
        comparisons = {}
        for name, kind in (("same_step", "neural-versus-same-step-euler"),
                           ("reference_difference", "euler-versus-fine-euler"),
                           ("total", "neural-versus-fine-euler")):
            comparisons[name] = evaluate_paths(
                states, states, times, config.context(case), config.spec,
                observation_times=config.query_times, n_bootstrap=config.n_bootstrap,
                coupling_verified=True, expected_paths=config.test_paths,
                comparison_kind=kind, reference_bias=(
                    {"status": "resolved", "strong_normalized_bound": 0., "weak_normalized_bound": 0.}
                    if name == "same_step" else None))
        prediction_path = tmp_path/f"{case}_{dt}_{seed}_paths.h5"
        with h5py.File(prediction_path, "w") as saved:
            saved.create_dataset("states", data=states)
        records.append({**data, "role": "locked-final", "checkpoint_path": str(checkpoint.resolve()),
                        "checkpoint_sha256": sha, "training_seconds": 2., "evaluation_seconds": .1,
                        "prediction_path": str(prediction_path.resolve()),
                        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                        "prediction_diagnostics": {"causality": True, "replay": True},
                        **comparisons})
    monkeypatch.setattr("nn_dk.path_reporting.load_path_model", lambda path: SimpleNamespace(
        provenance=provenance[str(Path(path).resolve())]))
    manifest = {"config_hash": config.config_hash, "source_hash": "path-report-test-source",
                "status": "reporting", "test_unlocked_unix": 1.,
                "fit_lock": {"expected_models": config.expected_models, "models": inventory}}
    return config, records, manifest


def test_complete_final_report_separates_neural_numerical_and_continuous_claims(tmp_path, monkeypatch):
    config, records, manifest = fixture(tmp_path, monkeypatch)
    before = copy.deepcopy(records)
    paths = write_path_report(tmp_path, config, records, manifest)
    assert records == before
    payload = json.loads(Path(paths["json"]).read_text())
    assert payload["all_configured_models_reported"]
    assert payload["all_same_step_comparisons_passed"]
    assert not payload["continuous_time_convergence_certified"]
    assert payload["training_seconds_distinct_models"] == 4
    assert len(payload["comparison_rows"]) == config.expected_models
    assert {row["optimizer_seed"] for row in payload["comparison_rows"]} == {0, 1}
    assert all(row["continuous_reference_accuracy_gate"] == "unresolved" for row in payload["comparison_rows"])
    text = Path(paths["markdown"]).read_text()
    assert "not a moments-only predictor" in text
    assert "including Nyquist" in text
    assert "no best seed is selected" in text
    assert "same-time and two-time" in text
    assert "continuous-time" in text.lower() and "remain uncertified" in text
    assert "not simultaneous" in text
    assert "Zero observed stops does not certify" in text
    with Path(paths["csv"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == config.expected_models
    assert b"\r\n" not in Path(paths["csv"]).read_bytes()
    for key in ("figure", "weak_figure"):
        assert Path(paths[key]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_unresolved_final_comparison_is_not_renamed_a_pass(tmp_path, monkeypatch):
    config, records, manifest = fixture(tmp_path, monkeypatch)
    records[0]["same_step"]["accuracy_gate"]["status"] = "unresolved"
    paths = write_path_report(tmp_path, config, records, manifest)
    payload = json.loads(Path(paths["json"]).read_text())
    assert not payload["all_same_step_comparisons_passed"]
    assert payload["same_step_comparisons_passed"] == 1
    assert payload["comparison_rows"][0]["same_step_accuracy_gate"] == "unresolved"


@pytest.mark.parametrize("change", [
    "missing", "duplicate", "source", "config", "target", "development", "checkpoint",
    "lock", "unlock", "causality", "replay", "false-pass", "missing-full-state",
    "missing-time", "missing-cross-time", "negative-cost", "allocation", "fake-gate-bound",
    "false-weak-errors", "locked-checkpoint", "fine-pass", "missing-prediction", "changed-prediction",
])
def test_incomplete_or_misleading_reports_rejected_before_export(tmp_path, monkeypatch, change):
    config, records, manifest = fixture(tmp_path, monkeypatch)
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1] = copy.deepcopy(records[0])
    elif change in ("source", "config"):
        manifest[f"{change}_hash"] = "old"
    elif change == "target":
        records[0]["target_id"] = "continuous-dk"
    elif change == "development":
        records[0]["role"] = "development"
    elif change == "checkpoint":
        Path(records[0]["checkpoint_path"]).write_bytes(b"changed after lock")
    elif change == "lock":
        manifest["fit_lock"]["expected_models"] = 1
    elif change == "unlock":
        manifest.pop("test_unlocked_unix")
    elif change in ("causality", "replay"):
        records[0]["prediction_diagnostics"][change] = False
    elif change == "false-pass":
        records[0]["same_step"]["accuracy_gate"]["strong_reference_bias_resolved"] = False
    elif change == "missing-full-state":
        records[0]["same_step"]["weak_full_state"] = {}
    elif change == "missing-time":
        records[0]["same_step"]["times"] = [0, config.horizon]
    elif change == "missing-cross-time":
        records[0]["same_step"]["weak_observables"]["normalized_joint_covariance_signed_error"] = np.zeros((4, 4)).tolist()
    elif change == "negative-cost":
        records[0]["training_seconds"] = -1
    elif change == "allocation":
        records[0]["same_step"]["n_independent_trajectory_families"] = 3
    elif change == "fake-gate-bound":
        records[0]["same_step"]["accuracy_gate"]["strong_upper_plus_reference_bound"] = .01
    elif change == "false-weak-errors":
        records[0]["same_step"]["weak_full_state"]["normalized_joint_covariance_signed_error"][0][0] = .1
    elif change == "locked-checkpoint":
        manifest["fit_lock"]["models"][0]["checkpoint_sha256"] = "different"
    elif change == "fine-pass":
        records[0]["total"]["accuracy_gate"]["status"] = "resolved_within_tolerance"
    elif change == "missing-prediction":
        Path(records[0]["prediction_path"]).unlink()
    elif change == "changed-prediction":
        with Path(records[0]["prediction_path"]).open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        write_path_report(tmp_path, config, records, manifest)
    assert not (tmp_path / "report").exists()
