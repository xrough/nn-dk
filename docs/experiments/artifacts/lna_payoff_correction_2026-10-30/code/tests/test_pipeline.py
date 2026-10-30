"""Reproducibility checks exercise independent trajectories, resumption, and the CLI."""

import json
import time
from dataclasses import replace

import h5py
import numpy as np
import pytest

from nn_dk.cli import main
from nn_dk.config import ExperimentConfig
from nn_dk.data import generate_dataset, generation_cost, load_split, prepare_run


def tiny_config():
    return ExperimentConfig(name="integration", n_cells=8, n_particles=64, n_grid=8,
        dt=1 / 512, save_interval=1 / 64, lag=1 / 64, horizon=1 / 16,
        query_times=(1 / 64, 1 / 32, 1 / 16), train_contexts=2, train_replicates=4,
        validation_contexts=1, validation_replicates=4, test_contexts=1, test_replicates=8,
        model_paths=8, calibration_paths=16, optimizer_seeds=(0,), bootstrap_samples=20,
        training={"epochs": 1, "patience": 2, "batch_size": 32, "learning_rate": 1e-3,
                  "weight_decay": 1e-5, "threads": 1})


def test_dataset_independence_resumption_and_provenance(tmp_path):
    config = tiny_config()
    result = generate_dataset(tmp_path, config, deadline=time.monotonic() - 1)
    assert not result["complete"]
    with pytest.raises(ValueError, match="Incomplete"):
        load_split(tmp_path, "train")
    assert generate_dataset(tmp_path, config)["complete"]
    before = generation_cost(tmp_path)
    assert generate_dataset(tmp_path, config)["generated_this_call"] == 0
    assert generation_cost(tmp_path) == before
    groups = [load_split(tmp_path, split) for split in ("train", "validation", "test")]
    ids = [c.context_id for group in groups for c, _ in group]
    assert len(ids) == len(set(ids))
    path_ids = [i for group in groups for _, b in group for i in b.diagnostics["trajectory_ids"]]
    assert len(path_ids) == len(set(path_ids)) == 20
    states = [batch.states for _, batch in groups[0]]
    other = tmp_path / "repeat"
    generate_dataset(other, config)
    for expected, (_, observed) in zip(states, load_split(other, "train"), strict=True):
        np.testing.assert_array_equal(expected, observed.states)
    with h5py.File(tmp_path / "trajectories.h5", "r") as store:
        assert "oracle" in store and "learner" in store
        assert store.attrs["complete"]
    with pytest.raises(ValueError, match="different configuration"):
        prepare_run(tmp_path, replace(config, seed=config.seed + 1))


@pytest.mark.integration
def test_cli_complete_comparison_and_replay(tmp_path):
    config = tiny_config()
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config.to_dict()))
    directory = tmp_path / "run"
    args = ["run", "--config", str(path), "--output", str(directory)]
    assert main([*args, "--budget-seconds", "0.0000001"]) == 75
    assert main(args) == 0
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert (directory / "report" / "report.md").exists()
    assert (directory / "report" / "accuracy_cost.png").stat().st_size > 1000
    evaluations = sorted((directory / "evaluations").glob("*.json"))
    assert len(evaluations) == 9
    records = [json.loads(p.read_text()) for p in evaluations]
    assert {r["method"] for r in records} == {
        "particle_mc", "mean_field", "linear_fluctuations", "discrete_dk",
        "autoregression", "polynomial_ridge", "kernel_ridge", "neural_a_seed0", "neural_b_seed0"}
    assert all(r["config_hash"] == config.config_hash for r in records)
    for r in records:
        assert r["cost"]["standalone_total_seconds"] >= r["cost"]["prediction_seconds"]
        if r["status"] == "unavailable":
            assert r["errors"] is None
    mtimes = [p.stat().st_mtime_ns for p in evaluations]
    assert main(args) == 0
    assert [p.stat().st_mtime_ns for p in evaluations] == mtimes
    assert main(["report", "--output", str(directory)]) == 0
    report_payload = json.loads((directory / "report" / "report.json").read_text())
    assert report_payload["manifest"]["status"] == "complete"
    withheld = evaluations[0].with_suffix(".withheld")
    evaluations[0].rename(withheld)
    with pytest.raises(ValueError, match="Incomplete"):
        main(["report", "--output", str(directory)])
    withheld.rename(evaluations[0])
    assert main(["report", "--output", str(directory)]) == 0
    report_payload = json.loads((directory / "report" / "report.json").read_text())
    assert "error" not in report_payload["manifest"]
