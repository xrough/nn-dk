"""Small complete BSDE workflows validate execution, not research accuracy."""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from nn_dk.bsde import prepare_bsde_run, report_bsde, run_bsde
from nn_dk.bsde_config import BSDEConfig, bsde_preset
from nn_dk.cli import main
from nn_dk.experiments import BudgetExhausted


def tiny_config():
    return replace(bsde_preset("bsde-smoke"), cases=("uniform",), width=8,
                   horizon=1/32, query_times=(1/128, 1/64, 1/32),
                   train_anchors=4, train_continuations=2,
                   validation_anchors=2, validation_continuations=2,
                   ou_test_anchors=4, conditional_anchors=2, conditional_paths=4,
                   outer_paths=4, refinement_paths=2, reference_batch_size=2,
                   bootstrap_samples=4, training={"epochs": 1, "patience": 1, "batch_size": 4})


def test_bsde_complete_replay_and_incomplete_report_rejected(tmp_path):
    config = tiny_config()
    result = run_bsde(tmp_path, config)
    assert all(Path(path).is_file() for path in result.values())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    records = json.loads((tmp_path / "evaluations.json").read_text())
    assert {(r["problem"], r["seed"]) for r in records} == {("ou", 0), ("dk", 0)}
    checkpoints = sorted((tmp_path / "checkpoints").rglob("*.pt"))
    assert len(checkpoints) == 2
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints]
    assert run_bsde(tmp_path, config) == result
    assert records == json.loads((tmp_path / "evaluations.json").read_text())
    assert hashes == [hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints]
    payload = json.loads(Path(result["json"]).read_text())
    assert payload["manifest"]["status"] == "complete"
    assert not list(tmp_path.rglob("*model_b*"))
    (tmp_path / "evaluations.json").write_text(json.dumps(records[:-1]))
    with pytest.raises(ValueError, match="Incomplete"):
        report_bsde(tmp_path, config)


def test_bsde_budget_and_configuration_guards(tmp_path):
    config = tiny_config()
    with pytest.raises(BudgetExhausted):
        run_bsde(tmp_path, config, time.monotonic()-1)
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "budget-exhausted"
    with pytest.raises(ValueError, match="mismatch"):
        prepare_bsde_run(tmp_path, replace(config, seed=config.seed+1))


def test_bsde_cli_preset_and_stage_guards(tmp_path):
    assert main(["bsde", "--preset", "bsde-smoke", "--output", str(tmp_path),
                 "--budget-seconds", "0.000001"]) == 75
    for command in (["bsde", "--preset", "a-smoke"],
                    ["run", "--preset", "bsde-smoke"],
                    ["convergence", "--stage", "fit"]):
        with pytest.raises(SystemExit):
            main(command)


def test_bsde_numeric_calibration_failure_blocks_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr("nn_dk.bsde_calibration.calibrate_bsde", lambda *a: {"complete": True, "passed": False})
    with pytest.raises(ValueError, match="calibration failed"):
        run_bsde(tmp_path, tiny_config())
    assert not list(tmp_path.glob("*_pool.h5"))
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("name", ["bsde-smoke", "bsde-first-batch", "bsde-refinement"])
def test_saved_bsde_configurations_match_presets(name):
    path = Path(__file__).parents[1] / "configs" / (name.replace("-", "_") + ".toml")
    assert BSDEConfig.load(path).to_dict() == bsde_preset(name).to_dict()
