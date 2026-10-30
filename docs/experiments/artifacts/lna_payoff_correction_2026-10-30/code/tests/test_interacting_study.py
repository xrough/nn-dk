"""Target calibration and final-reference lock ordering."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from nn_dk.data import source_hash, write_json
from nn_dk.interacting_config import InteractingStudyConfig
from nn_dk.interacting_study import (
    _lock_fits,
    calibrate_interacting,
    prepare_interacting_run,
)


def test_calibration_uses_native_interacting_and_free_fourier_operators(tmp_path):
    cfg = InteractingStudyConfig()
    prepare_interacting_run(tmp_path, cfg)
    result = calibrate_interacting(tmp_path, cfg)
    assert result["passed"]
    assert max(result["errors"].values()) < 1e-10
    assert calibrate_interacting(tmp_path, cfg) == result


def test_preparation_rejects_configuration_and_design_changes(tmp_path):
    cfg = InteractingStudyConfig()
    saved = prepare_interacting_run(tmp_path, cfg)
    assert "test_unlocked_unix" not in saved
    with pytest.raises(ValueError, match="changed"):
        prepare_interacting_run(tmp_path, replace(cfg, width=16))
    design = json.loads((tmp_path/"preregistration.json").read_text())
    design["trace"] = "incorrect random trace"
    write_json(tmp_path/"preregistration.json", design)
    with pytest.raises(ValueError, match="changed"):
        prepare_interacting_run(tmp_path, cfg)


def test_all_seed_checkpoints_lock_before_final_reference_unlock(tmp_path, monkeypatch):
    import nn_dk.interacting_training as training

    cfg = replace(InteractingStudyConfig(), optimizer_seeds=(0, 1))
    prepare_interacting_run(tmp_path, cfg)
    with pytest.raises(ValueError, match="Every declared fit"):
        _lock_fits(tmp_path, cfg)
    assert "test_unlocked_unix" not in json.loads((tmp_path/"manifest.json").read_text())

    def load(path):
        seed = int(path.stem.removeprefix("moments_seed"))
        return SimpleNamespace(provenance={"seed": seed})

    monkeypatch.setattr(training, "load_interacting_model", load)
    (tmp_path/"checkpoints").mkdir()
    for seed in cfg.optimizer_seeds:
        path = tmp_path/"checkpoints"/f"moments_seed{seed}.pt"
        path.write_bytes(f"fake checkpoint seed {seed}".encode())
        write_json(path.with_suffix(".json"), {"complete": True, "seed": seed,
                   "config_hash": cfg.config_hash, "source_hash": source_hash(),
                   "target_id": cfg.target_id})
    lock = _lock_fits(tmp_path, cfg)
    assert len(lock["models"]) == 2
    manifest = json.loads((tmp_path/"manifest.json").read_text())
    assert manifest["test_unlocked_unix"] > 0
    assert _lock_fits(tmp_path, cfg) == lock
    (tmp_path/lock["models"][0]["checkpoint_path"]).write_bytes(b"changed fit")
    with pytest.raises(ValueError, match="changed after"):
        _lock_fits(tmp_path, cfg)


def test_cached_failed_calibration_cannot_unlock_training_on_resume(tmp_path):
    cfg = InteractingStudyConfig()
    prepare_interacting_run(tmp_path, cfg)
    write_json(tmp_path/"calibration.json", {
        "source_hash": source_hash(), "config_hash": cfg.config_hash,
        "passed": False, "errors": {"independent_coefficient_check": 1.},
    })
    with pytest.raises(ValueError, match="calibration|Calibration"):
        calibrate_interacting(tmp_path, cfg)
    manifest = json.loads((tmp_path/"manifest.json").read_text())
    assert "fit_lock" not in manifest
    assert "test_unlocked_unix" not in manifest
    assert not (tmp_path/"checkpoints").exists()
