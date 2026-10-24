"""Execution and fairness checks for the fixed-path-budget BSDE follow-up."""
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from nn_dk.bsde_config import bsde_preset
from nn_dk.bsde_data import simulate_forward_trace
from nn_dk.bsde_model import BSDEMomentModel
from nn_dk.bsde_structured import StructuredBSDEMomentModel
from nn_dk.bsde_study_config import BSDEStudyConfig, bsde_study_preset
from nn_dk.bsde_training import fit_bsde, load_bsde_model
from nn_dk.native import initialize_masses


def _trace(config, problem, split):
    seed = 31 if split == "train" else 47
    initial = (initialize_masses(config.context("uniform"), config.spec, 4, seed)
               if problem == "dk" else np.random.default_rng(seed).normal(size=(4, 4)))
    return simulate_forward_trace(
        problem, config, np.repeat(initial, 2, axis=0), seed,
        anchor_ids=tuple(f"{split}-anchor-{i//2}" for i in range(8)),
        family_ids=tuple(f"{split}-family-{i}" for i in range(8)))


def test_study_presets_hold_paths_fixed_and_vary_independent_initial_coverage():
    scientific = bsde_study_preset("bsde-anchor-study")
    assert scientific.training_paths == 2048
    assert scientific.allocations == ((128, 16), (512, 4), (2048, 1))
    assert scientific.expected_models == 36
    assert scientific.validation == (256, 4)
    assert scientific.base_config.optimizer_seeds == (0, 1, 2)
    assert BSDEStudyConfig(**scientific.to_dict()).config_hash == scientific.config_hash
    smoke = bsde_study_preset("bsde-study-smoke")
    assert smoke.expected_models == 12
    assert smoke.training_paths == 16
    assert smoke.allocations == ((4, 4), (8, 2), (16, 1))
    with pytest.raises(ValueError, match="total training-path budget"):
        replace(scientific, allocations=((128, 16), (512, 16)))
    with pytest.raises(ValueError, match="increase"):
        replace(scientific, allocations=((512, 4), (128, 16)))


@pytest.mark.parametrize("problem", ("ou", "dk"))
def test_structured_factory_checkpoint_resume_and_paired_training_normalization(tmp_path, problem):
    config = replace(bsde_preset("bsde-smoke"), width=8,
                     training={"epochs": 1, "patience": 1, "batch_size": 4})
    training, validation = _trace(config, problem, "train"), _trace(config, problem, "validation")
    assert not set(training.anchor_ids) & set(validation.anchor_ids)
    assert not set(training.family_ids) & set(validation.family_ids)
    baseline_path, structured_path = tmp_path / "mlp.pt", tmp_path / "structured.pt"
    fit_bsde(training, validation, config, problem, 0, baseline_path,
             model_factory=BSDEMomentModel)
    result = fit_bsde(training, validation, config, problem, 0, structured_path,
                      model_factory=StructuredBSDEMomentModel)
    assert result["complete"]
    baseline, structured = load_bsde_model(baseline_path), load_bsde_model(structured_path)
    assert isinstance(structured, StructuredBSDEMomentModel)
    assert not isinstance(baseline, StructuredBSDEMomentModel)
    assert baseline.provenance["training_fingerprint"] == structured.provenance["training_fingerprint"]
    assert baseline.provenance["validation_fingerprint"] == structured.provenance["validation_fingerprint"]
    torch.testing.assert_close(structured.feature_mean, baseline.feature_mean, atol=0, rtol=0)
    torch.testing.assert_close(structured.feature_scale, baseline.feature_scale, atol=0, rtol=0)
    expected_features = structured.raw_features(torch.tensor(training.initial[::2]))
    torch.testing.assert_close(structured.feature_mean, expected_features.mean(0))
    assert result["training_anchors"] == 4
    assert result["training_paths"] == 8

    immutable = structured_path.read_bytes()
    fit_bsde(training, validation, config, problem, 0, structured_path,
             model_factory=StructuredBSDEMomentModel)
    assert structured_path.read_bytes() == immutable
    resumed_path = tmp_path / "resumed.pt"
    interrupted = fit_bsde(training, validation, config, problem, 0, resumed_path,
                           deadline=time.monotonic()-1,
                           model_factory=StructuredBSDEMomentModel)
    assert not interrupted["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        load_bsde_model(resumed_path)
    fit_bsde(training, validation, config, problem, 0, resumed_path,
             model_factory=StructuredBSDEMomentModel)
    resumed = load_bsde_model(resumed_path)
    for key, value in structured.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)


def test_study_complete_replay_and_partial_report_rejection(tmp_path):
    from nn_dk.bsde_study import report_study, run_bsde_study

    study = bsde_study_preset("bsde-study-smoke")
    result = run_bsde_study(tmp_path, study)
    assert all(Path(path).is_file() for path in result.values())
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "complete"
    assert json.loads(Path(result["json"]).read_text())["manifest"]["status"] == "complete"
    checkpoints = sorted(tmp_path.rglob("*.pt"))
    assert len(checkpoints) == study.expected_models
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints}
    models = [load_bsde_model(path) for path in checkpoints]
    assert sum(isinstance(model, StructuredBSDEMomentModel) for model in models) == 6
    assert {model.problem for model in models} == {"ou", "dk"}
    paired = {}
    validation_fingerprints = {}
    for path, model in zip(checkpoints, models, strict=True):
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["training_paths"] == study.training_paths
        assert metadata["training_anchors"] in {4, 8, 16}
        assert metadata["validation_anchors"] == study.validation[0]
        pair = paired.setdefault((model.problem, path.parent.name), [])
        pair.append(model)
        validation_fingerprints.setdefault(model.problem, set()).add(
            model.provenance["validation_fingerprint"])
    assert all(len(values) == 1 for values in validation_fingerprints.values())
    assert all(len(pair) == 2 for pair in paired.values())
    for first, second in paired.values():
        assert first.provenance["training_fingerprint"] == second.provenance["training_fingerprint"]
        torch.testing.assert_close(first.feature_mean, second.feature_mean, atol=0, rtol=0)
        torch.testing.assert_close(first.feature_scale, second.feature_scale, atol=0, rtol=0)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["workflow"] == "conditional-bsde-anchor-study-v1"
    assert manifest["test_unlocked_unix"] > 0
    assert json.loads((tmp_path / "config.json").read_text()) == study.to_dict()
    assert (tmp_path / "preregistration.json").is_file()
    assert run_bsde_study(tmp_path, study) == result
    assert hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints}
    checkpoints[-1].unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        report_study(tmp_path, study)


def test_study_preregistration_and_source_guards_and_no_evaluation_before_fits(tmp_path, monkeypatch):
    from nn_dk.bsde_study import prepare_study_run, report_study, run_bsde_study

    study = bsde_study_preset("bsde-study-smoke")
    prepare_study_run(tmp_path, study)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert not manifest.get("test_unlocked_unix")
    with pytest.raises(ValueError):
        run_bsde_study(tmp_path, study, stage="evaluate")
    assert not list(tmp_path.rglob("*.pt"))
    with pytest.raises((ValueError, FileNotFoundError)):
        report_study(tmp_path, study)
    preregistration_path = tmp_path / "preregistration.json"
    original_preregistration = preregistration_path.read_text()
    preregistration = json.loads(original_preregistration)
    preregistration["config_hash"] = "tampered-design"
    preregistration_path.write_text(json.dumps(preregistration))
    with pytest.raises(ValueError, match="preregistration|design|mismatch|changed"):
        prepare_study_run(tmp_path, study)
    preregistration_path.write_text(original_preregistration)
    changed_base = replace(study.base_config, seed=study.base_config.seed+1)
    with pytest.raises(ValueError, match="mismatch|changed"):
        prepare_study_run(tmp_path, replace(study, base=changed_base.to_dict()))
    monkeypatch.setattr("nn_dk.bsde_study.source_hash", lambda: "different-source")
    with pytest.raises(ValueError, match="source|mismatch|changed"):
        prepare_study_run(tmp_path, study)
