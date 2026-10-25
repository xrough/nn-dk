"""Light orchestration checks: numerical reference/training work is mocked."""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nn_dk.bsde_types import MomentPrediction
from nn_dk.data import write_json
from nn_dk.experiments import BudgetExhausted
from nn_dk.generator_config import generator_preset, generator_target_id


@pytest.fixture
def numerical_stubs(monkeypatch):
    import nn_dk.generator_reference as reference
    import nn_dk.generator_reporting as reporting
    import nn_dk.generator_study as study
    import nn_dk.generator_training as training

    state = {"incomplete": False, "development_bad": False, "invalid_certificate": False,
             "prediction_calls": 0, "final_input_events": [], "fit_events": [],
             "original_reference": study._reference,
             "original_calibration": study.calibrate_generator}
    monkeypatch.setattr(study, "source_hash", lambda: "frozen-workflow-source")
    monkeypatch.setattr(reporting, "source_hash", lambda: "frozen-workflow-source")

    def calibration(directory, config, deadline=None):
        path = Path(directory) / "calibration.json"
        if path.exists():
            return json.loads(path.read_text())
        value = {"passed": True, "source_hash": "frozen-workflow-source", "config_hash": config.config_hash}
        write_json(path, value)
        return value

    def initial_masses(context, spec, count, seed):
        rng = np.random.default_rng(seed)
        masses = 1 + .05 * rng.random((count, spec.n_cells))
        return masses / masses.sum(-1, keepdims=True)

    def fit(base, problem, seed, path, settings, deadline=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(f"immutable fake {problem} checkpoint seed {seed}".encode())
        state["fit_events"].append((problem, seed))
        return {"complete": not state["incomplete"], "problem": problem, "seed": seed,
                "training_seconds": 1., "optimized": False,
                "target_id": generator_target_id(base, problem)}

    def reference_packet(directory, config, role, case, scope, initial=None, deadline=None):
        shape = (len(initial), 3, 4) if initial is not None else (3, 4)
        mean = np.zeros(shape)
        if role == "development" and state["development_bad"]:
            mean[:] = .01
        return {"mean": mean, "scaled_covariance": np.broadcast_to(np.eye(4), (*shape, 4)).copy(),
                "mean_error_bound": 0., "scaled_covariance_error_bound": 0.,
                "certificate": {"valid": not state["invalid_certificate"],
                                "method": "mock workflow unit-test reference"}}

    class Model:
        def __init__(self, base, problem):
            self.base, self.problem = base, problem
            self.target_id = generator_target_id(base, problem)

        def predict_conditional(self, initial, horizons):
            state["prediction_calls"] += 1
            if self.problem == "ou":
                packet = study._ou_reference(self.base, initial, horizons)
            else:
                shape = (len(initial), len(horizons), 4)
                packet = {"mean": np.zeros(shape),
                          "scaled_covariance": np.broadcast_to(np.eye(4), (*shape, 4)).copy()}
            return MomentPrediction(packet["mean"], packet["scaled_covariance"],
                                    horizons, self.problem, self.target_id, {"unit_test_stub": True})

        def predict_ensemble(self, probabilities, horizons):
            state["prediction_calls"] += 1
            return MomentPrediction(np.zeros((len(horizons), 4)),
                                    np.broadcast_to(np.eye(4), (len(horizons), 4, 4)).copy(),
                                    horizons, "dk", self.target_id, {"unit_test_stub": True})

    def load(path):
        path = Path(path)
        directory = path.parent.parent
        config = __import__("nn_dk.generator_config", fromlist=["GeneratorStudyConfig"]).GeneratorStudyConfig.load(
            directory / "config.json")
        return Model(config.base_config, path.stem.split("_seed")[0])

    original_input_bank = study._input_bank

    def inputs(directory, config, role, *, unlocked=False):
        if role == "locked-final":
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            inventory = study._inventory(directory, config)
            assert len(inventory) == config.expected_models
            assert manifest["fit_lock"] == inventory and manifest["test_unlocked_unix"] > 0
            state["final_input_events"].append({"models": len(inventory), "unlocked": unlocked})
        return original_input_bank(directory, config, role, unlocked=unlocked)

    monkeypatch.setattr(study, "calibrate_generator", calibration)
    monkeypatch.setattr(study, "initialize_masses", initial_masses)
    monkeypatch.setattr(study, "_reference", reference_packet)
    monkeypatch.setattr(study, "_input_bank", inputs)
    monkeypatch.setattr(training, "fit_generator", fit)
    monkeypatch.setattr(training, "load_generator_model", load)
    monkeypatch.setattr(reference, "profile_probabilities", lambda base, context: (
        np.full(base.n_cells, 1 / base.n_cells), np.zeros(base.n_cells)))
    return state


@pytest.mark.parametrize("preset,count", (("generator-smoke", 2), ("generator-accuracy", 6)))
def test_all_fits_lock_before_final_inputs_and_completed_replay_is_immutable(
        tmp_path, numerical_stubs, preset, count):
    from nn_dk.generator_study import run_generator

    # Keep the scientific fit/seed count while making mocked JSON banks tiny;
    # the unmodified scientific allocations are checked in config tests.
    config = replace(generator_preset(preset), validation_anchors=4,
                     conditional_anchors=2, ou_test_anchors=8)
    result = run_generator(tmp_path, config)
    assert result["status"] == "complete" and result["accuracy_threshold_met"]
    assert numerical_stubs["final_input_events"] == [{"models": count, "unlocked": True}]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["fit_lock"]) == count
    assert manifest["completed_evaluations"] == 5 * len(config.optimizer_seeds)
    assert (tmp_path / "inputs/development.npz").exists()
    assert (tmp_path / "inputs/locked-final.npz").exists()
    report = json.loads((tmp_path / "report/generator_results.json").read_text())
    assert report["manifest"]["status"] == "complete"
    assert report["all_declared_comparisons_passed"]
    artifacts = [*tmp_path.glob("checkpoints/*.pt"), *tmp_path.glob("checkpoints/*.json"),
                 *tmp_path.glob("evaluation/*/*.json"), *tmp_path.glob("inputs/*.npz")]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}
    numerical_stubs["prediction_calls"] = 0
    repeated = run_generator(tmp_path, config)
    assert repeated == result
    assert numerical_stubs["prediction_calls"] == 0
    assert hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}


@pytest.mark.parametrize("reason", ("inaccurate", "invalid-reference"))
def test_development_failure_keeps_final_bank_absent(tmp_path, numerical_stubs, reason):
    from nn_dk.generator_study import run_generator

    numerical_stubs["development_bad" if reason == "inaccurate" else "invalid_certificate"] = True
    result = run_generator(tmp_path, generator_preset("generator-smoke"))
    assert result == {"status": "development-not-accurate", "final_bank_unlocked": False,
                      "development_results": str(tmp_path / "development_results.json")}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "development-not-accurate"
    assert not manifest.get("test_unlocked_unix") and not manifest.get("fit_lock")
    assert not (tmp_path / "inputs/locked-final.npz").exists()
    assert not (tmp_path / "evaluations.json").exists()
    assert numerical_stubs["final_input_events"] == []


def test_incomplete_fit_budget_exit_preserves_checkpoint_and_resumes(tmp_path, numerical_stubs):
    from nn_dk.generator_study import run_generator

    config = generator_preset("generator-smoke")
    numerical_stubs["incomplete"] = True
    with pytest.raises(BudgetExhausted, match="checkpointed"):
        run_generator(tmp_path, config)
    path = tmp_path / "checkpoints/ou_seed0.pt"
    original = path.read_bytes()
    assert not json.loads(path.with_suffix(".json").read_text())["complete"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "budget-exhausted"
    assert not (tmp_path / "inputs/locked-final.npz").exists()
    numerical_stubs["incomplete"] = False
    assert run_generator(tmp_path, config)["status"] == "complete"
    assert path.read_bytes() == original


@pytest.mark.parametrize("change", ("config", "source", "preregistration"))
def test_source_config_preregistration_guards(tmp_path, numerical_stubs, monkeypatch, change):
    from nn_dk.generator_study import prepare_generator_run

    config = generator_preset("generator-smoke")
    prepare_generator_run(tmp_path, config)
    if change == "config":
        config = replace(config, seed=config.seed + 1)
    elif change == "source":
        monkeypatch.setattr("nn_dk.generator_study.source_hash", lambda: "changed-source")
    else:
        path = tmp_path / "preregistration.json"
        value = json.loads(path.read_text())
        value["all_models_locked_before_final_inputs"] = False
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed"):
        prepare_generator_run(tmp_path, config)


@pytest.mark.parametrize("mutation", ("checkpoint", "metadata"))
def test_postlock_artifact_mutation_rejected(tmp_path, numerical_stubs, mutation):
    from nn_dk.generator_study import report_generator, run_generator

    config = generator_preset("generator-smoke")
    run_generator(tmp_path, config)
    checkpoint = tmp_path / "checkpoints/dk_seed0.pt"
    if mutation == "checkpoint":
        checkpoint.write_bytes(checkpoint.read_bytes() + b" changed")
    else:
        metadata = checkpoint.with_suffix(".json")
        value = json.loads(metadata.read_text())
        value["training_seconds"] += 1
        metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unchanged locked|content lock"):
        report_generator(tmp_path)


def test_initial_array_content_is_checked_before_any_fit_lock(tmp_path, numerical_stubs):
    from nn_dk.generator_study import _input_bank, prepare_generator_run

    config = generator_preset("generator-smoke")
    prepare_generator_run(tmp_path, config)
    _input_bank(tmp_path, config, "development")
    path = tmp_path / "inputs/development.npz"
    with np.load(path, allow_pickle=False) as saved:
        values = {name: saved[name] for name in saved.files}
    values["ou"][0, 0] += 1.
    with path.open("wb") as stream:
        np.savez_compressed(stream, **values)
    with pytest.raises(ValueError, match="array content"):
        _input_bank(tmp_path, config, "development")


def test_cached_reference_body_hash_and_input_binding(tmp_path, numerical_stubs, monkeypatch):
    import nn_dk.generator_reference as reference
    from nn_dk.generator_study import _input_bank, prepare_generator_run

    config = generator_preset("generator-smoke")
    prepare_generator_run(tmp_path, config)
    bank = _input_bank(tmp_path, config, "development")
    initial = bank["uniform"]
    shape = (len(initial), 3, 4)
    packet = {"mean": np.zeros(shape), "scaled_covariance": np.zeros((*shape, 4)),
              "mean_error_bound": 0., "scaled_covariance_error_bound": 0., "certificate": {"valid": True}}
    monkeypatch.setattr(reference, "reference_conditional", lambda *args, **kwargs: packet)
    original_reference = numerical_stubs["original_reference"]
    original_reference(tmp_path, config, "development", "uniform", "conditional", initial)
    path = tmp_path / "reference/development/uniform_conditional.json"
    value = json.loads(path.read_text())
    assert value["identity"]["input_bank_sha256"] == hashlib.sha256(
        (tmp_path / "inputs/development.npz").read_bytes()).hexdigest()
    value["reference"]["mean"][0][0][0] = .5
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="cached body content"):
        original_reference(tmp_path, config, "development", "uniform", "conditional", initial)


@pytest.mark.parametrize("change", ("body", "transplanted-record"))
def test_cached_development_record_is_bound_to_body_and_identity(tmp_path, numerical_stubs, change):
    from nn_dk.generator_study import run_generator

    config = generator_preset("generator-smoke")
    assert run_generator(tmp_path, config, stage="fit")["status"] == "fit-complete"
    path = tmp_path / "evaluation/development/ou_analytic_conditional_seed0.json"
    value = json.loads(path.read_text())
    if change == "body":
        value["record"]["comparison"]["errors"]["mean_normalized_rmse"] = .1
    else:
        other = json.loads((tmp_path / "evaluation/development/dk_uniform_conditional_seed0.json").read_text())
        value["record"], value["record_sha256"] = other["record"], other["record_sha256"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="body content|body differs"):
        run_generator(tmp_path, config, stage="evaluate")
    assert not (tmp_path / "inputs/locked-final.npz").exists()


@pytest.mark.parametrize("relative", ("evaluations.json", "development_results.json",
                                       "inputs/locked-final.npz"))
def test_completed_artifact_inventory_prevents_stale_replay(tmp_path, numerical_stubs, relative):
    from nn_dk.generator_study import report_generator, run_generator

    config = generator_preset("generator-smoke")
    run_generator(tmp_path, config)
    path = tmp_path / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="final content lock"):
        report_generator(tmp_path)


def test_calibration_cached_body_is_bound_even_before_final_lock(tmp_path, numerical_stubs):
    from nn_dk.generator_study import _body_hash, prepare_generator_run

    config = generator_preset("generator-smoke")
    prepare_generator_run(tmp_path, config)
    value = {"passed": True, "source_hash": "frozen-workflow-source", "config_hash": config.config_hash,
             "mean_error": 0., "covariance_error": 0.}
    value["content_sha256"] = _body_hash(value)
    path = tmp_path / "calibration.json"
    write_json(path, value)
    assert numerical_stubs["original_calibration"](tmp_path, config) == value
    value["covariance_error"] = .1
    write_json(path, value)
    with pytest.raises(ValueError, match="calibration cached body"):
        numerical_stubs["original_calibration"](tmp_path, config)


def test_stage_fit_does_not_unlock_final_and_stage_evaluate_requires_inventory(tmp_path, numerical_stubs):
    from nn_dk.generator_study import run_generator

    config = generator_preset("generator-smoke")
    with pytest.raises(ValueError, match="Every generator fit"):
        run_generator(tmp_path / "empty", config, stage="evaluate")
    assert not (tmp_path / "empty/inputs/locked-final.npz").exists()
    result = run_generator(tmp_path / "fit", config, stage="fit")
    assert result["status"] == "fit-complete"
    assert not (tmp_path / "fit/inputs/locked-final.npz").exists()
    assert run_generator(tmp_path / "fit", config, stage="evaluate")["status"] == "complete"


def test_expired_budget_preserves_prepared_work_without_unlock(tmp_path, numerical_stubs):
    from nn_dk.generator_study import run_generator

    with pytest.raises(BudgetExhausted):
        run_generator(tmp_path, generator_preset("generator-smoke"), deadline=time.monotonic() - 1)
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "budget-exhausted"
    assert not list(tmp_path.glob("checkpoints/*.pt"))
    assert not (tmp_path / "inputs/locked-final.npz").exists()


def test_cli_presets_stages_saved_config_dispatch_and_budget_exit(tmp_path, monkeypatch):
    from nn_dk.cli import main

    calls = []

    def run(directory, config, deadline, stage):
        calls.append((directory, config, deadline, stage))
        return {"status": "unit-test-dispatch"}

    monkeypatch.setattr("nn_dk.generator_study.run_generator", run)
    assert main(["generator", "--preset", "generator-accuracy", "--stage", "fit",
                 "--output", str(tmp_path)]) == 0
    assert calls[-1][1] == generator_preset("generator-accuracy")
    assert calls[-1][3] == "fit"
    saved = generator_preset("generator-smoke")
    write_json(tmp_path / "config.json", saved.to_dict())
    assert main(["generator", "--stage", "report", "--output", str(tmp_path)]) == 0
    assert calls[-1][1] == saved and calls[-1][3] == "report"
    assert main(["generator", "--config", "configs/generator_accuracy.toml", "--stage", "generate",
                 "--output", str(tmp_path / "configured")]) == 0
    assert calls[-1][1] == generator_preset("generator-accuracy")
    for arguments in (["run", "--preset", "generator-smoke"],
                      ["generator", "--preset", "bsde-smoke"],
                      ["generator", "--budget-seconds", "0"],
                      ["generator", "--budget-seconds", "nan"]):
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == 2

    def expired(*args, **kwargs):
        raise BudgetExhausted("saved for resume")

    monkeypatch.setattr("nn_dk.generator_study.run_generator", expired)
    assert main(["generator", "--preset", "generator-smoke", "--output", str(tmp_path / "expired")]) == 75
