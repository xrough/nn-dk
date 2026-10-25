"""Fair finite-reference comparisons and whole-family uncertainty accounting."""
from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nn_dk.interacting_config import InteractingStudyConfig

_SPEC = importlib.util.spec_from_file_location(
    "interacting_solver_comparison", Path(__file__).parents[1]/"analysis"/"compare_interacting_solvers.py")
comparison = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparison)


@pytest.fixture
def cfg():
    return replace(InteractingStudyConfig(), n_cells=8, n_particles=128)


def test_full_same_time_covariance_and_total_covariance(cfg):
    values = np.random.default_rng(3).normal(size=(7, 3, 4))
    mean, covariance = comparison.moments(values, cfg.n_particles)
    np.testing.assert_allclose(mean, values.mean(0))
    for time in range(3):
        np.testing.assert_allclose(covariance[time], cfg.n_particles*np.cov(values[:, time].T))
    conditional = np.tile(np.eye(4), (7, 3, 1, 1))
    combined = comparison.total_moments(values, conditional, cfg.n_particles)
    np.testing.assert_allclose(combined[0], mean)
    np.testing.assert_allclose(combined[1], covariance+np.eye(4))


def test_failures_block_estimates_without_survivor_averaging(cfg):
    values = np.zeros((4, 3, 4))
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="survivor"):
        comparison.moments(values, cfg.n_particles)
    with pytest.raises(ValueError, match="two complete"):
        comparison.moments(np.zeros((1, 3, 4)), cfg.n_particles)
    with pytest.raises(ValueError, match="Complete full"):
        comparison.total_moments(np.zeros((4, 3, 4)), np.zeros((4, 3, 4)), cfg.n_particles)


def test_candidate_and_reference_uncertainty_are_separate(cfg):
    rng = np.random.default_rng(4)
    reference = rng.normal(0, .01, (300, 3, 4))
    candidate = rng.normal(0, .01, (32, 3, 4))
    result = comparison.compare_prediction(comparison.moments(candidate, cfg.n_particles), reference,
        cfg, seed=1, n_bootstrap=40, candidate=candidate)
    assert result["candidate_uncertainty"]["mean"]["maximum_halfwidth"] > 0
    assert result["reference_uncertainty"]["mean"]["maximum_halfwidth"] > 0
    deterministic = comparison.compare_prediction(comparison.moments(candidate, cfg.n_particles), reference,
        cfg, seed=1, n_bootstrap=40)
    assert deterministic["candidate_uncertainty"]["mean"]["maximum_halfwidth"] == 0
    assert not result["accuracy_certificate"]
    assert result["continuous_reference_bias"] == "unresolved"
    assert np.asarray(result["covariance_error"]["point"]).shape == (3, 4, 4)
    invalid = (np.zeros((3, 4)), np.broadcast_to(-np.eye(4), (3, 4, 4)))
    with pytest.raises(ValueError, match="positive semidefinite"):
        comparison.compare_prediction(invalid, reference, cfg, seed=1, n_bootstrap=20)


def test_paired_bootstrap_keeps_complete_time_and_mode_families(cfg):
    values = np.random.default_rng(8).normal(size=(20, 3, 4))
    result = comparison.paired_difference(values, values.copy(), cfg, seed=3, n_bootstrap=30)
    assert result["mean_difference"]["maximum_absolute_interval_endpoint"] == 0
    assert result["covariance_difference"]["maximum_absolute_interval_endpoint"] == 0
    with pytest.raises(ValueError, match="same complete"):
        comparison.paired_difference(values, values[:10], cfg, seed=3, n_bootstrap=30)


def test_nested_fresh_candidate_streams_and_literal_noise_refinement(cfg):
    anchor = np.full(cfg.n_cells, 1/cfg.n_cells)
    small = comparison.candidate_inputs(cfg, "uniform", "conditional", anchor, 3, 2**-10, 98)
    large = comparison.candidate_inputs(cfg, "uniform", "conditional", anchor, 5, 2**-10, 98)
    for first, second in zip(small[:2], large[:2], strict=True):
        np.testing.assert_array_equal(first, second[:3])
    assert small[2] == large[2][:3] and len(set(large[2])) == 5
    np.testing.assert_array_equal(small[0], np.broadcast_to(anchor, (3, cfg.n_cells)))
    coarse = comparison.aggregate_noise(large[1], 2**-10, 2**-8)
    np.testing.assert_array_equal(coarse, large[1].reshape(5, -1, 4, cfg.n_cells).sum(2))
    with pytest.raises(ValueError, match="integer refinements"):
        comparison.aggregate_noise(large[1], 2**-10, .003)
    ensemble = comparison.candidate_inputs(cfg, "uniform", "ensemble", None, 3, 2**-10, 98)
    assert set(ensemble[2]).isdisjoint(small[2])
    np.testing.assert_allclose(ensemble[0].sum(-1), 1)


def test_comparison_candidate_artifacts_replay_and_reject_tampering(tmp_path):
    path = tmp_path/"candidate.npz"
    identity = {"physical_inputs": "fixed"}
    assert comparison._load_arrays(path, identity) is None
    comparison._save_arrays(path, identity, values=np.arange(10))
    np.testing.assert_array_equal(comparison._load_arrays(path, identity)["values"], np.arange(10))
    with pytest.raises(ValueError, match="provenance"):
        comparison._load_arrays(path, {"physical_inputs": "changed"})
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="provenance"):
        comparison._load_arrays(path, identity)


def test_statistics_candidate_can_be_fits_locked_with_one_shared_reference(tmp_path, cfg, monkeypatch):
    run = tmp_path/"candidate"
    run.mkdir()
    (run/"config.json").write_text(json.dumps(cfg.to_dict()))
    monkeypatch.setattr(comparison, "verify_interacting_fit_lock", lambda run, config:
                        {"status": "fits-locked", "fit_lock": {"models": [{"seed": 0}]}})
    monkeypatch.setattr(comparison, "_verify_item", lambda *args: None)
    assert comparison._moment_inventory(run, cfg)[0]["seed"] == 0
    with pytest.raises(ValueError, match="finish and lock"):
        comparison._moment_inventory(run, cfg, require_reference_complete=True)
    with pytest.raises(ValueError, match="identical physical"):
        comparison._moment_inventory(run, replace(cfg, kappa=.6))


def test_matched_output_timing_does_not_sample_statistics_for_conditional_task(cfg, monkeypatch):
    initial = np.full((3, cfg.n_cells), 1/cfg.n_cells)
    noise = np.zeros((3, 16, cfg.n_cells))
    families = ("a", "b", "c")
    calls = []

    class PathModel:
        def sample(self, context, times, count, seed, **kwargs):
            return {"states": np.broadcast_to(kwargs["initial_masses"][:, None], (count, 3, cfg.n_cells))}

    class MomentModel:
        def predict_conditional(self, states, times):
            calls.append(len(states))
            return SimpleNamespace(mean=np.zeros((len(states), 3, 4)),
                                   scaled_covariance=np.zeros((len(states), 3, 4, 4)))

    monkeypatch.setattr(comparison, "simulate_native", lambda *args, **kwargs:
                        SimpleNamespace(states=np.broadcast_to(kwargs["initial_masses"][:, None], (3, 3, cfg.n_cells))))
    result = comparison.benchmark_tasks(PathModel(), MomentModel(), initial, noise, families,
        cfg.context("uniform"), cfg, 2**-8, warmup=1, repeats=3, role="conditional", batch_size=8)
    assert calls == [1, 1, 1, 1]
    assert result["conditional_query_batch"] == 1
    assert result["simulation_or_initial_mc_paths"] == 3
    assert set(result["timings"]) == {"density_path_nn", "native_euler", "statistics_nn"}


def test_all_stopped_paths_are_reported():
    result = comparison._stopping(np.array([0., .02, np.inf]), (.01, .03, .0625))
    assert result["n_paths"] == 3 and result["stopped_paths"] == 2
    np.testing.assert_allclose(result["probability_at_queries"], (1/3, 2/3, 2/3))
    assert result["all_paths_retained"]


def test_small_end_to_end_comparison_replays_and_rejects_changed_outputs(tmp_path, cfg, monkeypatch):
    from nn_dk.path_model import CausalPathOperator

    cfg = replace(cfg, conditional_paths=4, ensemble_paths=6, refinement_paths=2)
    run = tmp_path/"run"
    run.mkdir()
    (run/"config.json").write_text(json.dumps(cfg.to_dict()))
    metadata = run/"fit.json"
    metadata.write_text(json.dumps({"training_seconds": .01}))
    checkpoint = run/"fit.pt"
    checkpoint.write_text("development stub loader")
    item = dict(seed=0, checkpoint_path=str(checkpoint), metadata_path=str(metadata),
                checkpoint_sha256=comparison.file_sha256(checkpoint))
    path_items = [dict(item, run=str(run), case=case, dt=2**-8) for case in cfg.cases]
    moment_items = [dict(item, run=str(run), candidate=cfg.name)]
    monkeypatch.setattr(comparison, "_inventory", lambda *args: (path_items, moment_items))

    def model_loader(path):
        model = CausalPathOperator(cfg.spec, "dk", width=2, depth=1)
        model.dt = 2**-8
        return model

    class MomentModel:
        def predict_conditional(self, states, times):
            weights = comparison.physical_observable_weights(cfg.spec, cfg.queries)
            mean = np.repeat((states@weights.T)[:, None], len(times), axis=1)
            return SimpleNamespace(mean=mean, scaled_covariance=np.tile(np.eye(4)*.02, (len(states), 3, 1, 1)))

    monkeypatch.setattr(comparison, "load_path_model", model_loader)
    monkeypatch.setattr(comparison, "load_interacting_model", lambda path: MomentModel())

    def reference_loader(directory, config, case):
        values = np.random.default_rng(32).normal(0, .001, (6, 3, 4))
        anchors = np.full((cfg.conditional_anchors, cfg.n_cells), 1/cfg.n_cells)
        anchors[1, 0] += .01
        anchors[1, 1] -= .01
        return dict(anchors=anchors,
            conditional=dict(values=np.tile(values[:4], (cfg.conditional_anchors, 1, 1)),
                anchor_ids=np.repeat(np.arange(cfg.conditional_anchors), 4), failed=np.zeros(8, dtype=bool),
                family_ids=np.asarray([f"reference-{case}-conditional-{i}" for i in range(8)])),
            ensemble=dict(values=values, failed=np.zeros(6, dtype=bool),
                family_ids=np.asarray([f"reference-{case}-ensemble-{i}" for i in range(6)]),
                refinement_values=np.repeat(values[:2, None], 3, axis=1),
                refinement_stopped_at=np.full((2, 3), np.inf)))

    monkeypatch.setattr(comparison, "load_interacting_reference", reference_loader)
    (run/"interacting_reference.h5").write_text("development reference stub loader")
    (run/"interacting_reference.json").write_text("development reference sidecar stub")
    output = tmp_path/"comparison"
    result = comparison.compare((run,), run, output, counts=(2, 3), n_bootstrap=20)
    assert len(result["records"]) == 32
    assert (output/"comparison.csv").exists()
    assert comparison.compare((run,), run, output, counts=(2, 3), n_bootstrap=20) == result
    (run/"interacting_reference.h5").write_text("changed reference")
    with pytest.raises(ValueError, match="preregistration changed"):
        comparison.compare((run,), run, output, counts=(2, 3), n_bootstrap=20)
    (run/"interacting_reference.h5").write_text("development reference stub loader")
    with (output/"comparison.csv").open("a") as stream:
        stream.write("tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        comparison.compare((run,), run, output, counts=(2, 3), n_bootstrap=20)
