"""Independent accuracy records and cost calls retain literal task identity."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "evaluate_payoff_repair", Path(__file__).parents[1] / "analysis/evaluate_payoff_repair.py")
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)

from analysis.path_generalization_reference import ReferenceConfig  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402


def test_completed_prediction_cache_rejects_changed_numeric_evidence(tmp_path):
    path = tmp_path / "method.json"
    identity = {"reference": "fixed", "model": "fixed"}
    study._save_result(path, identity, {}, {"mean": np.zeros((3, 4))}, {})
    assert study._replay(path, identity)["complete"]
    with pytest.raises(ValueError, match="changed"):
        study._replay(path, {**identity, "model": "replaced"})
    path.with_suffix(".npz").write_bytes(b"replaced outcomes")
    with pytest.raises(ValueError, match="changed"):
        study._replay(path, identity)


def test_mc_timing_does_not_compute_an_unrequested_payoff_bank(monkeypatch):
    cfg = ReferenceConfig(n_cells=16, paths_per_group=2, refinement_paths=2)
    initial = np.full((2, cfg.n_cells), 1 / cfg.n_cells)
    noise = np.zeros((2, round(cfg.horizon / cfg.model_step), cfg.n_cells))
    calls = []
    original = PayoffBank.evaluate_paths

    def tracked(self, *args):
        calls.append(self.name)
        return original(self, *args)

    monkeypatch.setattr(PayoffBank, "evaluate_paths", tracked)

    def compiled(m, dw, context, spec, dt):
        return {"states": np.repeat(m[:, None], dw.shape[1] + 1, axis=1),
                "stopped_at": np.full(len(m), np.inf)}

    values, _ = study._path_values(None, compiled, initial, noise, cfg, "uniform", "terminal3-integral1")
    assert calls == ["terminal3-integral1"]
    assert set(values) == {"terminal3-integral1"}
    assert values["terminal3-integral1"].shape == (2, 3, 4)


def test_any_invalid_path_blocks_mc_estimate_even_with_other_valid_paths():
    cfg = ReferenceConfig(n_cells=16, paths_per_group=2, refinement_paths=2)
    initial = np.full((2, cfg.n_cells), 1 / cfg.n_cells)
    noise = np.zeros((2, round(cfg.horizon / cfg.model_step), cfg.n_cells))

    def compiled(m, dw, context, spec, dt):
        states = np.repeat(m[:, None], dw.shape[1] + 1, axis=1)
        states[0, -1, 0] = -1e-3
        return {"states": states, "stopped_at": np.full(len(m), np.inf)}

    with pytest.raises(ValueError, match="survivor-conditioned"):
        study._path_values(None, compiled, initial, noise, cfg, "uniform")


def test_every_declared_fit_and_untrained_baseline_is_retained():
    models = [{"name": f"new{i}", "variant": "repaired", "bank": "b", "seed": i} for i in range(10)]
    old = [{"bank": "b", "init": "cold", "seed": i, "checkpoint_path": str(Path("old") / f"{i}.pt")}
           for i in range(12)]
    result = study._model_entries({"models": models}, {"old_heads": old, "banks": [{"name": "b"}, {"name": "c"}]})
    assert len(result) == 24
    assert len({item["name"] for item in result}) == 24
    assert sum(item.get("untrained", False) for item in result) == 2
