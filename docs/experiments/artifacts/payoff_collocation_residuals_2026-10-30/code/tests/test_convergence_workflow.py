"""Small complete A-only studies test orchestration, not scientific accuracy."""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nn_dk.cli import main
from nn_dk.convergence import _simulation_cache, report_convergence, run_convergence
from nn_dk.convergence_config import convergence_preset
from nn_dk.convergence_data import prepare_convergence_run
from nn_dk.dk_training import load_coefficient_model
from nn_dk.experiments import BudgetExhausted


def tiny_config():
    return replace(convergence_preset("a-smoke"), cases=("uniform",), horizon=1/32,
                   query_times=(1/128, 1/64, 1/32), step_sizes=(1/128, 1/256, 1/512),
                   proposal_dt=1/128, train_path_counts=(4,), widths=(4,), base_width=4,
                   validation_paths=2, coefficient_test_paths=2, diagnostic_paths=4,
                   refinement_paths=2, bootstrap_samples=4,
                   training={"epochs": 1, "patience": 1, "batch_size": 8, "threads": 1})


def test_a_only_complete_run_replay_and_strict_report(tmp_path):
    config = tiny_config()
    output = run_convergence(tmp_path, config)
    assert all(Path(path).is_file() for path in output.values())
    first = json.loads((tmp_path / "records.json").read_text())
    assert len(first) == 21  # 3 oracle steps, 2 fits x 3 ablations x 3 steps
    assert len(list((tmp_path / "records").glob("*.json"))) == len(first)
    assert {r.get("training_mode") for r in first} == {None, "oracle", "increments"}
    assert not list(tmp_path.rglob("*model_b*"))
    for record in first:
        comparison = record["comparison"]
        assert comparison["prediction"]["diagnostics"]["all_stopped_paths_included"]
        assert comparison["accuracy_gate"]["status"] == "unresolved"
        if "neural_error" in record:
            np.testing.assert_allclose(comparison["errors"]["sqrt_n_mean_signed"],
                np.asarray(record["neural_error"]["errors"]["sqrt_n_mean_signed"])
                + record["timestep_error"]["errors"]["sqrt_n_mean_signed"], atol=1e-14)
    checkpoint = next((tmp_path / "checkpoints").rglob("*.pt"))
    model = load_coefficient_model(checkpoint)
    assert model.kind == "native-dk-autonomous-diagonal-face-v1"
    output2 = run_convergence(tmp_path, config)
    assert output2 == output
    assert json.loads((tmp_path / "records.json").read_text()) == first
    (tmp_path / "records.json").write_text(json.dumps(first[:-1]))
    with pytest.raises(ValueError, match="Incomplete"):
        report_convergence(tmp_path, config)


def test_budget_checkpoint_and_provenance_rejection(tmp_path):
    config = tiny_config()
    with pytest.raises(BudgetExhausted):
        run_convergence(tmp_path, config, deadline=time.monotonic() - 1)
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "budget-exhausted"
    with pytest.raises(ValueError, match="mismatch"):
        prepare_convergence_run(tmp_path, replace(config, seed=config.seed + 1))
    with pytest.raises((ValueError, FileNotFoundError)):
        report_convergence(tmp_path, config)


def test_cli_a_only_budget_and_argument_validation(tmp_path):
    assert main(["convergence", "--preset", "a-smoke", "--output", str(tmp_path),
                 "--budget-seconds", "0.000001"]) == 75
    with pytest.raises(SystemExit):
        main(["convergence", "--preset", "smoke"])
    with pytest.raises(SystemExit):
        main(["run", "--preset", "a-smoke"])


def test_failed_a0_blocks_generation_and_training(tmp_path, monkeypatch):
    monkeypatch.setattr("nn_dk.dk_calibration.calibrate_native", lambda *args: {
        "complete": True, "linear_gaussian": {"status": "failed"}})
    with pytest.raises(ValueError, match="calibration failed"):
        run_convergence(tmp_path, tiny_config())
    assert not list(tmp_path.glob("pool_*.h5"))
    assert not (tmp_path / "checkpoints").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "failed"


def test_evaluation_resume_preserves_coupled_chunks(tmp_path, monkeypatch):
    config = replace(tiny_config(), evaluation_batch_size=2)
    resumed, fresh = tmp_path / "resumed", tmp_path / "fresh"
    ticks = iter((0., 2.))
    with monkeypatch.context() as patch:
        patch.setattr("nn_dk.convergence.time.monotonic", lambda: next(ticks))
        with pytest.raises(BudgetExhausted):
            _simulation_cache(resumed, config, "uniform", min(config.step_sizes), "oracle", deadline=1.)
    completed = list(resumed.rglob("*.npz"))
    assert len(completed) == 1
    before = completed[0].read_bytes()
    actual = _simulation_cache(resumed, config, "uniform", min(config.step_sizes), "oracle")
    expected = _simulation_cache(fresh, config, "uniform", min(config.step_sizes), "oracle")
    assert completed[0].read_bytes() == before
    np.testing.assert_array_equal(actual.states, expected.states)
    np.testing.assert_array_equal(actual.stopped_at, expected.stopped_at)
    assert actual.family_ids == expected.family_ids
