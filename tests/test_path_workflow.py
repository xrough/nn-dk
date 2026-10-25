"""Workflow lock ordering using real banks/evaluation and a mock exact path fitter.

The mock returns the native Euler path. These tests certify orchestration and
provenance only; actual learned-operator training is tested separately.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import nn_dk.path_data as path_data
import nn_dk.path_reporting as path_reporting
import nn_dk.path_study as study
import nn_dk.path_training as path_training
from nn_dk.experiments import BudgetExhausted
from nn_dk.path_config import PathConfig, path_preset
from nn_dk.path_model import PATH_MODEL_KIND
from nn_dk.path_numerics import torch_euler


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    fingerprint = "path-workflow-test-source"
    for module in (study, path_data, path_reporting, path_training):
        monkeypatch.setattr(module, "source_hash", lambda: fingerprint)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path/"matplotlib"))
    config = replace(PathConfig(), n_cells=8, cases=("heat", "additive", "dk_uniform"),
                     width=4, depth=1, train_paths=4, validation_paths=4, test_paths=4,
                     adam_steps=1, lbfgs_steps=0, n_bootstrap=8)
    events, bad_controls = [], set()

    class ExactModel:
        kind = PATH_MODEL_KIND

        def __init__(self, provenance, checkpoint_path):
            self.provenance = provenance
            self.checkpoint_path = str(checkpoint_path)
            self.target_id = provenance["target_id"]
            self.case = provenance["case"]
            self.dt = provenance["dt"]

        def __call__(self, initial, increments, dt, context):
            case = self.provenance["case"]
            actual = torch_euler(initial, increments, context, config.spec, dt,
                                  config.problem(case))
            raw = torch.cat((torch.as_tensor(initial, dtype=torch.float64)[:, None],
                             actual["raw_proposals"]), 1)
            if case in bad_controls:
                correction = torch.zeros_like(raw)
                correction[:, 1:, 0], correction[:, 1:, 1] = .01, -.01
                raw = raw+correction
            return raw

    def fit(config, case, dt, seed, path, train, validation, deadline=None):
        if not Path(path).exists():
            assert not (Path(path).parents[1]/"path_test.h5").exists()
        events.append(("fit", case, dt, seed))
        path = Path(path)
        provenance = {"kind": PATH_MODEL_KIND, "case": case, "problem": config.problem(case),
                      "dt": dt, "seed": seed, "source_hash": fingerprint,
                      "config_hash": config.config_hash, "target_id": config.target_id(case, dt)}
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"kind": PATH_MODEL_KIND, "complete": True, "provenance": provenance}, path)
        return {"path": str(path), "complete": True, "training_seconds": .01,
                "source_hash": fingerprint, "config_hash": config.config_hash,
                "target_id": config.target_id(case, dt), "case": case, "dt": dt, "seed": seed}

    def load(path):
        checkpoint = torch.load(path, weights_only=True)
        if not checkpoint.get("complete"):
            raise ValueError("Cannot load an incomplete mock checkpoint")
        if checkpoint.get("provenance", {}).get("source_hash") != fingerprint:
            raise ValueError("Mock checkpoint source mismatch")
        return ExactModel(checkpoint["provenance"], path)

    monkeypatch.setattr(path_training, "fit_path", fit)
    monkeypatch.setattr(path_training, "load_path_model", load)
    monkeypatch.setattr(path_reporting, "load_path_model", load)
    return config, events, bad_controls


def test_workflow_locks_every_fit_before_final_inputs_and_reports_all_states(tmp_path, workflow,
                                                                           monkeypatch):
    config, events, _ = workflow
    original = path_data.ensure_path_bank

    def ensure(directory, current, role, deadline=None):
        if role == "test":
            manifest = json.loads((Path(directory)/"manifest.json").read_text())
            assert len(manifest["fit_lock"]["models"]) == config.expected_models
            assert sum(event[0] == "fit" for event in events) == config.expected_models
            assert manifest["test_unlocked_unix"] > 0
        events.append(("bank", role))
        return original(directory, current, role, deadline)

    monkeypatch.setattr(path_data, "ensure_path_bank", ensure)
    result = study.run_path_study(tmp_path/"run", config)
    assert result["status"] == "complete"
    manifest = json.loads((tmp_path/"run"/"manifest.json").read_text())
    assert len(manifest["artifact_lock"]) > config.expected_models
    records = json.loads((tmp_path/"run"/"evaluations.json").read_text())
    assert len(records) == config.expected_models
    assert {record["case"] for record in records} == set(config.cases)
    assert all(record["prediction_diagnostics"]["causality"] for record in records)
    assert all(record["prediction_diagnostics"]["replay"] for record in records)
    assert all(record["total"]["accuracy_gate"]["status"] == "unresolved" for record in records)
    assert not manifest["continuous_time_accuracy_certified"]


def test_failed_control_prevents_dk_fits_and_final_unlock_even_when_cases_reordered(tmp_path, workflow):
    config, events, bad_controls = workflow
    bad_controls.add("heat")
    config = replace(config, cases=("dk_uniform", "additive", "heat"))
    result = study.run_path_study(tmp_path, config)
    assert result["status"] == "control-validation-unresolved"
    assert result["final_inputs_unlocked"] is False
    assert not any(event[0] == "fit" and event[1].startswith("dk") for event in events)
    assert not (tmp_path/"path_test.h5").exists()
    assert "fit_lock" not in json.loads((tmp_path/"manifest.json").read_text())


def test_evaluate_cannot_bypass_corrupt_or_missing_control_validation(tmp_path, workflow):
    config, _, _ = workflow
    study.run_path_study(tmp_path, config, stage="fit")
    control = next((tmp_path/"development").glob("heat_*.json"))
    content = json.loads(control.read_text())
    content["record"]["within_tolerance"] = False
    control.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="control|development|content"):
        study.run_path_study(tmp_path, config, stage="evaluate")
    assert not (tmp_path/"path_test.h5").exists()


def test_actual_incomplete_checkpoint_cannot_unlock_final_inputs_via_complete_metadata(tmp_path,
                                                                                     workflow):
    config, _, _ = workflow
    study.run_path_study(tmp_path, config, stage="fit")
    checkpoint = study._checkpoint(tmp_path, "dk_uniform", config.step_sizes[0], 0)
    payload = torch.load(checkpoint, weights_only=True)
    payload["complete"] = False
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="incomplete"):
        study.run_path_study(tmp_path, config, stage="evaluate")
    assert not (tmp_path/"path_test.h5").exists()


@pytest.mark.parametrize("mutation", ("source", "config", "preregistration"))
def test_preparation_rejects_changed_source_config_or_design(tmp_path, workflow, monkeypatch, mutation):
    config, _, _ = workflow
    study.prepare_path_run(tmp_path, config)
    if mutation == "source":
        monkeypatch.setattr(study, "source_hash", lambda: "different-source")
    elif mutation == "config":
        config = replace(config, seed=config.seed+1)
    else:
        path = tmp_path/"preregistration.json"
        design = json.loads(path.read_text())
        design["output"] = "only four moments"
        path.write_text(json.dumps(design))
    with pytest.raises(ValueError, match="changed"):
        study.prepare_path_run(tmp_path, config)


def test_incomplete_bank_returns_budget_exit_then_resumes_without_reading_partial_inputs(tmp_path,
                                                                                      workflow,
                                                                                      monkeypatch):
    config, events, _ = workflow
    original = path_data.ensure_path_bank
    with monkeypatch.context() as interrupted:
        interrupted.setattr(path_data, "ensure_path_bank", lambda *args, **kwargs: {"complete": False})
        with pytest.raises(BudgetExhausted):
            study.run_path_study(tmp_path, config)
    assert not events
    assert json.loads((tmp_path/"manifest.json").read_text())["status"] == "budget-exhausted"
    assert path_data.ensure_path_bank is original
    assert study.run_path_study(tmp_path, config)["status"] == "complete"


def test_completed_replay_preserves_locked_checkpoints_and_input_bytes(tmp_path, workflow):
    config, _, _ = workflow
    study.run_path_study(tmp_path, config)
    paths = [*tmp_path.glob("checkpoints/*"), *tmp_path.glob("path_*.h5")]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    study.run_path_study(tmp_path, config)
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == old for path, old in before.items())


@pytest.mark.parametrize("artifact", ("checkpoint", "test-bank", "evaluation", "prediction"))
def test_completed_artifact_mutation_is_rejected(tmp_path, workflow, artifact):
    config, _, _ = workflow
    study.run_path_study(tmp_path, config)
    path = {"checkpoint": next(tmp_path.glob("checkpoints/*.pt")),
            "test-bank": tmp_path/"path_test.h5",
            "evaluation": next(tmp_path.glob("evaluation/*.json")),
            "prediction": next(tmp_path.glob("predictions/*.h5"))}[artifact]
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        study.run_path_study(tmp_path, config, stage="report")


def test_partial_evaluation_cannot_bless_a_changed_cached_prediction(tmp_path, workflow, monkeypatch):
    config, _, _ = workflow
    original = study._evaluate

    def partial(directory, current, lock, deadline):
        original(directory, current, {"models": lock["models"][:1]}, deadline)
        raise BudgetExhausted("Interrupted after one committed final evaluation")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(study, "_evaluate", partial)
        with pytest.raises(BudgetExhausted):
            study.run_path_study(tmp_path, config)
    assert not json.loads((tmp_path/"manifest.json").read_text()).get("artifact_lock")
    prediction = next(tmp_path.glob("predictions/*.h5"))
    with prediction.open("ab") as stream:
        stream.write(b"corrupted after evaluation checkpoint")
    with pytest.raises(ValueError, match="prediction|content|changed"):
        study.run_path_study(tmp_path, config, stage="evaluate")


def test_expired_workflow_deadline_does_not_start_generation(tmp_path, workflow):
    config, events, _ = workflow
    with pytest.raises(BudgetExhausted):
        study.run_path_study(tmp_path, config, deadline=time.monotonic()-1)
    assert not events
    assert not list(tmp_path.glob("path_*.h5"))


def test_full_horizon_covariance_calibration_avoids_growing_block_exponential(tmp_path):
    config = path_preset("path-first-batch")
    study.prepare_path_run(tmp_path, config)
    result = study.calibrate_paths(tmp_path, config)
    assert result["passed"]
    assert result["full_covariance_lyapunov_ode_error"] < 1e-12
    assert result["physical_face_covariance_error"] < 1e-14
