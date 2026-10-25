"""Independent-family data locks, coupled Euler references and full moment checks."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest

from nn_dk.data import source_hash
from nn_dk.interacting_config import InteractingStudyConfig
from nn_dk.interacting_data import (
    ensure_interacting_bank,
    ensure_interacting_reference,
    file_sha256,
    load_interacting_bank,
    load_interacting_reference,
    verify_interacting_fit_lock,
)
from nn_dk.interacting_evaluation import (
    compare_monte_carlo_moments,
    integrate_conditional_moments,
    refinement_diagnostics,
)
from nn_dk.native import initialize_masses, physical_observable_weights, simulate_native
from nn_dk.types import seeded_rng


@pytest.fixture
def cfg():
    return replace(InteractingStudyConfig(), train_paths=3, validation_paths=2,
                   conditional_anchors=2, conditional_paths=3, ensemble_paths=4,
                   ensemble_initial_paths=4, refinement_paths=2, n_bootstrap=20,
                   adam_steps=1, lbfgs_steps=0, reference_batch_size=2)


def _lock(directory, cfg, monkeypatch):
    module = ModuleType("nn_dk.interacting_training")
    expected = dict(source_hash=source_hash(), config_hash=cfg.config_hash,
                    target_id=cfg.target_id, seed=cfg.optimizer_seeds[0])
    model = SimpleNamespace(provenance=expected, complete=True)
    module.load_interacting_model = lambda path: model
    monkeypatch.setitem(sys.modules, "nn_dk.interacting_training", module)
    checkpoint = directory/"checkpoint.pt"
    checkpoint.write_text("test checkpoint loader is replaced only in this isolated lock unit test")
    metadata = directory/"checkpoint.json"
    metadata.write_text(json.dumps(dict(complete=True, **expected)))
    manifest = dict(source_hash=source_hash(), config_hash=cfg.config_hash,
                    test_unlocked_unix=time.time(), fit_lock=dict(expected_models=1, models=[dict(
                        seed=cfg.optimizer_seeds[0], checkpoint_path=str(checkpoint.resolve()),
                        checkpoint_sha256=file_sha256(checkpoint), metadata_path=str(metadata.resolve()),
                        metadata_sha256=file_sha256(metadata))]))
    (directory/"manifest.json").write_text(json.dumps(manifest))
    return model


def test_collocation_has_no_payoff_labels_and_grouped_disjoint_families(tmp_path, cfg):
    for role in ("train", "validation"):
        packet = ensure_interacting_bank(tmp_path, cfg, role)
        assert packet["complete"]
        before = file_sha256(packet["path"])
        assert ensure_interacting_bank(tmp_path, cfg, role) == packet
        assert file_sha256(packet["path"]) == before
    train = load_interacting_bank(tmp_path, cfg, "train")
    validation = load_interacting_bank(tmp_path, cfg, "validation")
    assert train["states"].shape == (2*3*4*4, 32)
    assert not train["future_payoff_labels"]
    assert set(train["family_ids"]).isdisjoint(validation["family_ids"])
    counts = np.unique(train["family_ids"], return_counts=True)[1]
    np.testing.assert_array_equal(counts, 16)
    assert np.all((train["horizons"] >= 0) & (train["horizons"] <= cfg.horizon))
    np.testing.assert_allclose(train["states"].sum(-1), 1, atol=5e-15)
    with h5py.File(tmp_path/"interacting_train.h5", "r") as store:
        assert set(store["uniform"]) == {"states", "horizons", "stopped_at", "family_ids", "complete", "row_hash"}


def test_atomic_partial_resume_and_exact_native_snapshots(tmp_path, cfg):
    first = ensure_interacting_bank(tmp_path, cfg, "train", deadline=time.monotonic()-1)
    assert not first["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        load_interacting_bank(tmp_path, cfg, "train")
    assert ensure_interacting_bank(tmp_path, cfg, "train")["complete"]
    import nn_dk.interacting_data as data
    seed = data._seed(cfg, "train", "uniform", 0)
    family = data._family(cfg, "train", "uniform", 0)
    initial = initialize_masses(cfg.context("uniform"), cfg.spec, 1, seed)
    exact = simulate_native(cfg.context("uniform"), cfg.spec, cfg.state_times, cfg.state_step,
                            seed, 1, initial_masses=initial, family_ids=(family,))
    with h5py.File(tmp_path/"interacting_train.h5", "r") as store:
        np.testing.assert_array_equal(store["uniform"]["states"][0], exact.states[0])


def test_tamper_config_shape_and_file_hash_are_rejected(tmp_path, cfg):
    ensure_interacting_bank(tmp_path, cfg, "train")
    with pytest.raises(ValueError, match="configuration"):
        ensure_interacting_bank(tmp_path, replace(cfg, seed=cfg.seed+1), "train")
    with h5py.File(tmp_path/"interacting_train.h5", "r+") as store:
        store["uniform"]["states"][0, 0, 0] += 1e-5
    with pytest.raises(ValueError, match="row changed"):
        load_interacting_bank(tmp_path, cfg, "train")


def test_test_stream_cannot_open_before_all_declared_fits(tmp_path, cfg, monkeypatch):
    with pytest.raises(ValueError, match="fit lock"):
        ensure_interacting_reference(tmp_path, cfg)
    assert not (tmp_path/"interacting_reference.h5").exists()
    model = _lock(tmp_path, cfg, monkeypatch)
    verify_interacting_fit_lock(tmp_path, cfg)
    model.complete = False
    with pytest.raises(ValueError, match="incomplete"):
        ensure_interacting_reference(tmp_path, cfg)
    assert not (tmp_path/"interacting_reference.h5").exists()


def test_reference_literals_coupling_allocation_independence_and_replay(tmp_path, cfg, monkeypatch):
    _lock(tmp_path, cfg, monkeypatch)
    packet = ensure_interacting_reference(tmp_path, cfg)
    assert packet["complete"] and packet["reference_bias_status"] == "unresolved"
    assert ensure_interacting_reference(tmp_path, cfg) == packet
    reference = load_interacting_reference(tmp_path, cfg, "nonuniform")
    assert reference["conditional"]["values"].shape == (6, 3, 4)
    assert reference["ensemble"]["values"].shape == (4, 3, 4)
    a = set(reference["conditional"]["family_ids"].astype(str))
    b = set(reference["ensemble"]["family_ids"].astype(str))
    assert a.isdisjoint(b) and a.isdisjoint(reference["integration_family_ids"])
    import nn_dk.interacting_data as data
    seed = data._seed(cfg, "reference", "ensemble", "nonuniform", 0)
    fine_dt = cfg.reference_step/max(cfg.reference_factors)
    fine = seeded_rng(seed, "physical-reference-face-noise").normal(
        0, np.sqrt(fine_dt), (1, round(cfg.horizon/fine_dt), cfg.n_cells))
    initial = reference["ensemble"]["initial"][:1]
    weights = physical_observable_weights(cfg.spec, cfg.queries)
    for level, factor in enumerate(cfg.reference_factors):
        aggregate = max(cfg.reference_factors)//factor
        dw = fine.reshape(1, -1, aggregate, cfg.n_cells).sum(axis=2)
        exact = simulate_native(cfg.context("nonuniform"), cfg.spec, cfg.query_times,
                                cfg.reference_step/factor, seed, 1, initial_masses=initial,
                                brownian_increments=dw,
                                family_ids=(reference["ensemble"]["family_ids"][0].decode(),))
        np.testing.assert_allclose(reference["ensemble"]["refinement_values"][0, level],
                                   exact.states[0]@weights.T, atol=3e-16, rtol=0)
    # Non-refinement paths have no fabricated fine observations.
    assert np.isnan(reference["ensemble"]["refinement_values"][2, 1:]).all()


def test_full_total_covariance_includes_initial_sampling():
    mean = np.zeros((4, 3, 4))
    mean[:, :, 0] = np.arange(4)[:, None]/100
    covariance = np.broadcast_to(np.eye(4), (4, 3, 4, 4)).copy()
    out_mean, out_cov = integrate_conditional_moments(mean, covariance, 2048)
    np.testing.assert_allclose(out_mean[:, 0], .015)
    np.testing.assert_allclose(out_cov[:, 0, 0], 1+2048*np.var(np.arange(4)/100, ddof=1))
    np.testing.assert_allclose(out_cov[:, 1, 1], 1)


def test_all_covariance_entries_and_continuous_uncertainty_remain_visible(cfg):
    rng = np.random.default_rng(42)
    values = rng.normal(size=(100, 3, 4))/np.sqrt(cfg.n_particles)
    means = values.mean(0)
    cov = np.asarray([np.cov(values[:, t].T)*cfg.n_particles for t in range(3)])
    result = compare_monte_carlo_moments(means, cov, values, cfg, seed=1)
    assert result["accuracy_gate"]["status"] == "unresolved"
    assert not result["accuracy_gate"]["continuous_time_certificate"]
    assert np.shape(result["covariance_error"]["point"]) == (3, 4, 4)
    perturbed = cov.copy()
    perturbed[:, 0, 2] += .1
    perturbed[:, 2, 0] += .1
    bad = compare_monte_carlo_moments(means, perturbed, values, cfg, seed=1)
    assert bad["covariance_error"]["maximum_absolute_point"] > .1
    assert result["uncertainty"]["maximum_reference_sampling_halfwidth"] > cfg.uncertainty_budget
    values[0] = np.nan
    with pytest.raises(ValueError, match="survivor"):
        compare_monte_carlo_moments(means, cov, values, cfg, seed=1)


def test_coupled_refinement_never_certifies_continuous_bias(cfg):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(5, 1, 3, 4))*np.ones((1, 3, 1, 1))
    result = refinement_diagnostics(values, tuple(map(str, range(5))), cfg, 1)
    assert result["continuous_bias_status"] == "unresolved"
    assert result["estimated_temporal_rate"] is None
    np.testing.assert_allclose(result["levels"][0]["mean_difference"]["point"], 0)
    with pytest.raises(ValueError, match="uniquely"):
        refinement_diagnostics(values, ("x",)*5, cfg, 1)


def test_paired_stopping_bias_reported_without_population_claim(cfg):
    values = np.zeros((5, 3, 3, 4))
    stops = np.full((5, 3), np.inf)
    stops[0, 0] = cfg.horizon/2
    result = refinement_diagnostics(values, tuple(map(str, range(5))), cfg, 1, stopped_at=stops)
    stopping = result["levels"][0]["stopping"]
    assert stopping["final_probability_difference"] == .2
    np.testing.assert_allclose(stopping["cdf_difference"]["point"], [0, .2, .2])
    assert stopping["fine"]["no_events_is_not_zero_population_probability"]


def test_all_stopped_paths_retained_in_collocation(tmp_path, cfg):
    small = replace(cfg, n_particles=8)
    ensure_interacting_bank(tmp_path, small, "train")
    with h5py.File(tmp_path/"interacting_train.h5", "r") as store:
        for case in small.cases:
            assert np.all(store[case]["stopped_at"][:] == 0)
            states = store[case]["states"][:]
            np.testing.assert_array_equal(states, np.repeat(states[:, :1], len(small.state_times), axis=1))
    flat = load_interacting_bank(tmp_path, small, "train")
    assert len(flat["states"]) == 2*3*4*4


def test_evaluation_all_anchors_full_predictions_and_immutable_cache(tmp_path, cfg, monkeypatch):
    from nn_dk.interacting_evaluation import evaluate_interacting_model
    model = _lock(tmp_path, cfg, monkeypatch)
    weights = physical_observable_weights(cfg.spec, cfg.queries)

    def predict(initial, horizons):
        mean = np.repeat((initial@weights.T)[:, None], len(horizons), axis=1)
        covariance = np.broadcast_to(np.eye(4), (*mean.shape, 4)).copy()*.1
        return SimpleNamespace(mean=mean, scaled_covariance=covariance,
                               horizons=horizons, target_id=cfg.target_id)

    model.predict_conditional = predict
    ensure_interacting_reference(tmp_path, cfg)
    result = evaluate_interacting_model(tmp_path, cfg, model, 0)
    assert len(result["records"]) == 2*(cfg.conditional_anchors+1)
    assert result["continuous_bias_status"] == "unresolved"
    assert result["evaluation_signature"]
    path = tmp_path/"interacting_evaluation_seed0.json"
    before = path.read_bytes()
    assert evaluate_interacting_model(tmp_path, cfg, model, 0) == result
    assert path.read_bytes() == before
    with h5py.File(result["prediction_path"], "r+") as store:
        assert store["uniform"]["integration_mean"].shape == (cfg.ensemble_initial_paths, 3, 4)
        store["uniform"]["integration_mean"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="artifact changed"):
        evaluate_interacting_model(tmp_path, cfg, model, 0)
