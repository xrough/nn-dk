"""Controlled allocation/architecture comparisons on one fresh held-out bank."""

from dataclasses import replace

import numpy as np
import pytest

from nn_dk.bsde_config import bsde_preset
from nn_dk.bsde_study_evaluation import (
    StudyModelDescriptor,
    _normalized_sample_moments,
    _read_reference_samples,
    _weighted_moments,
    evaluate_study_model,
    paired_metric_contrasts,
    paired_study_contrasts,
    study_metric_rows,
    test_bank_identity,
)


class _Model:
    problem = "dk"

    def __init__(self, config, checkpoint, seed=0):
        self.config = config
        self.checkpoint_path = checkpoint
        self.optimizer_seed = seed
        self.study_allocation = (config.train_anchors, config.train_continuations)
        self.calls = []

    @property
    def target_id(self):
        return self.config.target_id(self.problem)

    def predict_conditional(self, initial, horizons):
        self.calls.append((len(initial), tuple(horizons)))
        mean = np.zeros((len(initial), len(horizons), 4))
        covariance = np.broadcast_to(np.eye(4), (*mean.shape, 4)).copy()
        return {"mean": mean, "scaled_covariance": covariance}


def _small_config():
    return replace(bsde_preset("bsde-smoke"), n_cells=8, horizon=1/32,
                   query_times=(1/128, 1/64, 1/32), reference_dt=1/256,
                   conditional_anchors=2, conditional_paths=2,
                   outer_paths=2, reference_batch_size=2, bootstrap_samples=2)


def test_allocation_changes_reuse_the_same_reference_chunks(tmp_path, monkeypatch):
    import nn_dk.bsde_evaluation as evaluator

    monkeypatch.setattr(evaluator, "source_hash", lambda: "fixed-test-source")
    common = _small_config()
    first_config = replace(common, train_anchors=4, train_continuations=8)
    second_config = replace(common, train_anchors=8, train_continuations=4)
    a, b = tmp_path / "first.pt", tmp_path / "second.pt"
    a.write_bytes(b"first saved model")
    b.write_bytes(b"second saved model")
    first_model, second_model = _Model(first_config, a), _Model(second_config, b)
    first = evaluate_study_model(tmp_path, common, first_model,
                                 StudyModelDescriptor("mlp", 4, 8, 0))
    chunks = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("chunk_*.npz")}
    assert chunks
    second = evaluate_study_model(tmp_path, common, second_model,
                                  StudyModelDescriptor("mlp", 8, 4, 0))
    assert chunks == {p: p.stat().st_mtime_ns for p in tmp_path.rglob("chunk_*.npz")}
    assert first["test_bank_id"] == second["test_bank_id"]
    assert first["study_model_id"] != second["study_model_id"]
    assert first["fit_config_hash"] != second["fit_config_hash"]
    assert first["evaluation_config_hash"] == second["evaluation_config_hash"]
    assert len(list((tmp_path / "evaluation").glob("*.json"))) == 2
    assert first["evaluation"]["complete"] and second["evaluation"]["complete"]
    assert first_model.calls and second_model.calls
    # A saved replay does not call either network or rewrite any reference.
    second_model.calls.clear()
    assert second == evaluate_study_model(tmp_path, common, second_model,
                                          StudyModelDescriptor("mlp", 8, 4, 0))
    assert not second_model.calls
    # Identical fitted value maps have exactly zero paired improvement and zero
    # interval width, even when their shared references have Monte Carlo error.
    monkeypatch.setattr(evaluator, "simulate_native", lambda *args, **kwargs: pytest.fail(
        "Paired contrasts must never generate new reference paths"))
    contrasts = paired_study_contrasts([first, second], tmp_path, common, n_bootstrap=8)
    assert len(contrasts) == 4
    assert {row["scope"] for row in contrasts} == {"conditional", "ensemble"}
    for contrast in contrasts:
        assert all(value == 0 for value in contrast["candidate_minus_baseline"].values())
        assert all(interval == [0., 0.] for interval in
                   contrast["paired_uncertainty"]["candidate_minus_baseline_ci95"].values())
        conditional = contrast["scope"] == "conditional"
        assert contrast["paired_uncertainty"]["anchor_generalization_uncertainty_included"] != conditional


def test_study_rejects_incompatible_targets_and_false_factor_labels(tmp_path):
    config = _small_config()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test")
    model = _Model(config, checkpoint)
    label = StudyModelDescriptor("mlp", config.train_anchors, config.train_continuations, 0)
    with pytest.raises(ValueError, match="incompatible physical targets"):
        evaluate_study_model(tmp_path, replace(config, n_particles=512), model, label)
    with pytest.raises(ValueError, match="allocation"):
        evaluate_study_model(tmp_path, config, model, StudyModelDescriptor("mlp", 4, 8, 0))
    with pytest.raises(ValueError, match="architecture"):
        evaluate_study_model(tmp_path, config, model,
                             StudyModelDescriptor("structured", config.train_anchors,
                                                  config.train_continuations, 0))
    with pytest.raises(ValueError, match="optimizer seed"):
        evaluate_study_model(tmp_path, config, model,
                             StudyModelDescriptor("mlp", config.train_anchors,
                                                  config.train_continuations, 1))
    model.checkpoint_path = tmp_path / "absent.pt"
    with pytest.raises(ValueError, match="saved study checkpoint"):
        evaluate_study_model(tmp_path, config, model, label)


def test_bank_identity_is_separate_from_training_allocation_and_fresh_for_seed():
    config = _small_config()
    identity = test_bank_identity(config, "dk")
    assert identity == test_bank_identity(replace(config, train_anchors=32), "dk")
    assert identity != test_bank_identity(replace(config, seed=config.seed+1), "dk")
    assert identity != test_bank_identity(replace(config, conditional_anchors=4), "dk")
    assert identity != test_bank_identity(replace(config, conditional_paths=4), "dk")
    assert identity != test_bank_identity(config, "ou")
    with pytest.raises(ValueError, match="only OU and native DK"):
        test_bank_identity(config, "brownian")


@pytest.mark.parametrize("args", [("", 1, 1, 0), ("mlp", 0, 1, 0),
                                  ("mlp", 1, 0, 0), ("mlp", 1, 1, -1)])
def test_descriptor_rejects_invalid_independence_counts(args):
    with pytest.raises(ValueError):
        StudyModelDescriptor(*args)


def _record(architecture, anchors, continuations, seed, error, bank="shared"):
    descriptor = StudyModelDescriptor(architecture, anchors, continuations, seed)
    errors = {"mean_normalized_rmse": error,
              "scaled_covariance_normalized_rmse": error*2,
              "scaled_variance_normalized_rmse": error*3,
              "maximum_mean_normalized_error": error*4,
              "maximum_scaled_covariance_normalized_error": error*5}
    return {"study_model_id": f"{architecture}/{anchors}/{seed}/{bank}", "test_bank_id": bank,
            "factors": descriptor.to_dict(),
            "evaluation": {"problem": "ou", "complete": True,
                           "neural_error": {"errors": errors,
                                            "accuracy_gate": {"status": "unresolved"}}}}


def test_factorial_contrasts_keep_seeds_and_banks_separate():
    records = [_record(architecture, anchors, 32//anchors, seed,
                       .4/(anchors**.5) * (1 if architecture == "mlp" else .5))
               for seed in (0, 1) for architecture in ("mlp", "structured")
               for anchors in (2, 4, 8)]
    # A model on a different held-out bank cannot create any matched contrasts.
    records.append(_record("structured", 8, 4, 0, .01, bank="different"))
    rows = study_metric_rows(records)
    assert len(rows) == 13
    contrasts = paired_metric_contrasts(records)
    assert len(contrasts) == 10  # Per seed: two endpoint allocation and three architecture contrasts.
    assert all(row["test_bank_id"] == "shared" for row in contrasts)
    assert all(row["candidate_minus_baseline"]["mean_rmse"] < 0 for row in contrasts)
    assert all(row["fixed_train_paths"] == 32 for row in contrasts)
    for row in contrasts:
        if row["factor"] == "independent_initial_states":
            assert row["candidate_train_anchors"] > row["baseline_train_anchors"]
        else:
            assert row["baseline_architecture"] == "mlp"
            assert row["candidate_architecture"] == "structured"
        assert "point estimate only" in row["uncertainty"]


def test_incomplete_and_duplicate_evaluations_cannot_enter_table():
    record = _record("mlp", 2, 8, 0, .2)
    with pytest.raises(ValueError, match="Duplicate"):
        study_metric_rows([record, record])
    record["evaluation"]["complete"] = False
    with pytest.raises(ValueError, match="Incomplete"):
        study_metric_rows([record])


def test_weighted_bootstrap_moments_equal_whole_family_resampling():
    samples = np.random.default_rng(9).normal(size=(7, 3, 4))
    indices = np.asarray(((0, 0, 1, 2, 2, 4, 6), (1, 1, 1, 3, 4, 5, 6)))
    weights = np.asarray([np.bincount(row, minlength=len(samples)) for row in indices], dtype=float)
    weighted_mean, weighted_covariance = _weighted_moments(weights, samples)
    mean, covariance = _normalized_sample_moments(samples[indices])
    np.testing.assert_allclose(weighted_mean, mean, atol=1e-14)
    np.testing.assert_allclose(weighted_covariance, covariance, atol=1e-14)


def test_paired_analytic_intervals_exclude_reference_monte_carlo_error():
    config = _small_config()
    bank = test_bank_identity(config, "ou")
    records = [_record("mlp", 2, 8, 0, .2, bank), _record("structured", 2, 8, 0, .1, bank)]
    for record in records:
        record["evaluation_config_hash"] = config.config_hash
    result = paired_study_contrasts(records, "unused-no-reference-paths-needed", config)
    assert len(result) == 1
    assert result[0]["candidate_minus_baseline"]["mean_rmse"] == pytest.approx(-.1)
    assert result[0]["paired_uncertainty"]["reference_sampling_error"] == 0
    assert result[0]["paired_uncertainty"]["candidate_minus_baseline_ci95"]["mean_rmse"] == [-.1, -.1]
    with pytest.raises(ValueError, match="At least two"):
        paired_study_contrasts(records, "unused", config, n_bootstrap=1)


def test_missing_reference_bank_cannot_be_replaced_by_new_forward_paths(tmp_path):
    with pytest.raises(ValueError, match="complete saved common reference bank"):
        _read_reference_samples(tmp_path, _small_config(), "uniform", "conditional")
    assert not list(tmp_path.rglob("*.npz"))
