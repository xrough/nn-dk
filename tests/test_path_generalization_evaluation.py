"""Resumable broad-task evidence and comparison integration on synthetic inputs."""

import importlib.util
import json
import time
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from nn_dk.payoff_learning import PayoffMomentModel

_SPEC = importlib.util.spec_from_file_location(
    "broad_task_evaluation", Path(__file__).parents[1]/"analysis"/"run_generalization_evaluation.py")
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)
reference = import_module("analysis.path_generalization_reference")


@pytest.fixture(scope="module")
def synthetic():
    # The complete scientific target is unchanged; only software-test path and
    # bootstrap counts shrink. These draws never enter an experiment directory.
    cfg = replace(reference.ReferenceConfig(), paths_per_group=80, refinement_paths=8, chunk_paths=8)
    key = "nonuniform__ensemble__0"
    data = reference._simulate_chunk(cfg, key, 0, cfg.paths_per_group, None)
    data["probabilities"] = reference.profile_probabilities(cfg, "nonuniform")[0]
    data["metadata"] = {"case": "nonuniform", "role": "ensemble"}
    counts = evaluation.bootstrap_counts(len(data["initial"]), 31, 8)
    summary = evaluation._many_reference(data["many_observables"], counts)
    method = {"name": "native_euler_dt10", "method": "native-euler", "case": "nonuniform", "dt": 2**-10}
    return cfg, key, data, counts, summary, method


def test_transactional_480_bootstrap_resume_matches_literal_raw_comparison(tmp_path, monkeypatch, synthetic):
    cfg, key, data, counts, summary, method = synthetic
    cache = tmp_path/"partial.h5"
    identity = {"group": key, "software_test_only": True}
    clock = iter((0, 2))
    with monkeypatch.context() as interrupt:
        interrupt.setattr(evaluation, "time", SimpleNamespace(
            monotonic=lambda: next(clock), perf_counter=time.perf_counter))
        with pytest.raises(evaluation.BudgetExhausted):
            evaluation.path_candidate(cache, method, cfg, data, counts, summary, identity, deadline=1)
    with h5py.File(cache) as store:
        assert int(store["progress"][()]) == 1
        assert "many_observables" not in store
    result = evaluation.path_candidate(cache, method, cfg, data, counts, summary, identity)
    packet = evaluation.numpy_euler(data["initial"], data["model_increments"],
        cfg.context("nonuniform"), cfg.spec, method["dt"])
    reduced = evaluation.reduce_paths(packet, cfg.spec, method["dt"], cfg.horizon, cfg.common_step)
    expected = evaluation.diagonal_comparison(data["many_observables"], reduced["many"],
                                               counts=counts, n_particles=cfg.n_particles)
    actual, _ = evaluation._many_from_accumulator(data["many_observables"], summary, counts,
        result["accumulator"], len(data["initial"]), cfg.n_particles)
    for field in ("mean_max_std", "variance_max_relative", "mean_rms_std", "variance_rms_relative",
                  "mean_sampling95_simultaneous_radius", "variance_sampling95_simultaneous_radius"):
        np.testing.assert_allclose(actual[field], expected[field], rtol=1e-11, atol=1e-12)
    # Tampering with committed reduced outputs must not be accepted on replay.
    with h5py.File(cache, "r+") as store:
        store["bank_b"][0, 0, 0] += 1.
    with pytest.raises(ValueError, match="committed reduced candidate"):
        evaluation.path_candidate(cache, method, cfg, data, counts, summary, identity)


def test_path_joint_distributions_reference_audits_and_integrity_replay(tmp_path, synthetic):
    cfg, key, data, counts, summary, method = synthetic
    identity = {"group": key, "software_test_only": True}
    record = evaluation.evaluate_path(tmp_path, method, cfg, data, counts, summary, identity)
    assert record["complete"]
    assert len(record["metrics"]["terminal3-integral1"]["joint_time_12"]["records"]) == 1
    assert record["strong_common_grid_sup"]["all_cells_and_common_physical_times"]
    assert record["stopping"]["zero_observed_stops_is_not_zero_population_probability"]
    assert record["continuous_reference_bias"] == "unresolved"
    assert evaluation.evaluate_path(tmp_path, method, cfg, data, counts, summary, identity) == record
    audit = evaluation.reference_audit(tmp_path, cfg, data, {"group": key, "software_test": "audit"})
    assert audit["complete"] and len(audit["records"]) == 3
    with Path(record["bootstrap_artifact"]["path"]).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="evidence artifact changed"):
        evaluation.evaluate_path(tmp_path, method, cfg, data, counts, summary, identity)


def test_statistics_mixture_and_gaussian_comparators_preserve_distinct_laws(tmp_path, monkeypatch, synthetic):
    cfg, key, data, counts, _, _ = synthetic
    bank = evaluation._banks(cfg)[1]
    with torch.random.fork_rng():
        torch.manual_seed(5)
        model = PayoffMomentModel(cfg.spec, cfg.context("nonuniform"), cfg.horizon, bank, 8)
    metadata = tmp_path/"software-metadata.json"
    metadata.write_text(json.dumps({"training_seconds": 1.}))
    item = {"bank": bank.name, "init": "cold", "seed": 0,
            "checkpoint_path": "software-placeholder", "metadata_path": str(metadata)}
    # Mock restoration only. Actual full-state neural inference and all mixture
    # bootstrap calculations run on a fresh model and synthetic initial states.
    monkeypatch.setattr(evaluation, "load_payoff_model", lambda _: model)
    stats = evaluation.evaluate_statistics(tmp_path, item, cfg, data, counts,
                                           {"group": key, "software_test": "statistics"})
    assert stats["complete"] and stats["integration_uncertainty"]["initial_histograms"] == 80
    assert "paired" in stats["integration_uncertainty"]["comparison_coupling"]
    assert "joint-time covariance" in stats["unsupported"]
    gaussian = [evaluation.evaluate_lga(tmp_path, cfg, data, counts,
        {"group": key, "software_test": f"Gaussian-{step}"}, step) for step in (2**-7, 2**-10)]
    assert all(record["complete"] for record in gaussian)
    assert len(evaluation._lga_refinement(gaussian, cfg)["banks"]) == 2
    for record in gaussian:
        diagnostics = record["gaussian_sampling_diagnostics"]
        assert diagnostics["stopping_applied"] is False
        assert diagnostics["density_repair_applied"] is False
        assert "minimum_native_mass" in diagnostics
        assert "below_stopping_threshold_fraction" in diagnostics
        assert "same-Brownian strong density-path error" in record["unsupported"]
