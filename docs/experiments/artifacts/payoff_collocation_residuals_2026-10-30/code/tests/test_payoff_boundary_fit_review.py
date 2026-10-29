"""Report-only residual comparisons keep units, stages and empty strata clear."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.review_payoff_boundary_fit import (  # noqa: E402
    compare_saved_strata,
    residual_statistics,
    stratified_statistics,
    verify_selected_mean,
)
from nn_dk.interacting_training import _state_hash  # noqa: E402


def test_concentrations_distinguish_raw_square_fixed_square_and_robust_loss():
    values = np.array([[1., 2., 3., 4.], [0., 0., 0., 0.], [100., 200., 300., 400.]])
    scales = np.array([1., 2., 3., 4.])
    factors = np.array([.1, .2, .3, .4])
    masks = {"all": np.ones(3, dtype=bool), "rare": np.array([False, False, True]),
             "zero": np.array([False, True, False]), "absent": np.zeros(3, dtype=bool)}
    result = stratified_statistics(values, scales, .25, factors, 3., masks)
    assert result["absent"] == {"rows": 0, "empty": True, "row_fraction": 0.}
    assert result["zero"]["raw_square_fraction"] == 0.
    assert result["rare"]["row_fraction"] == pytest.approx(1/3)
    assert result["rare"]["fixed_unit_square_fraction"] == pytest.approx(10000/10001)
    assert result["rare"]["robust_objective_fraction"] < result["rare"]["fixed_unit_square_fraction"]
    direct = 2*3.**2*(np.sqrt(1+(.25*values/scales/3.)**2)-1)
    assert result["all"]["common_pseudo_huber_objective"] == pytest.approx(direct.mean())


def test_zero_denominators_are_unresolved_and_empty_strata_not_perfect_fit():
    values = np.zeros((3, 4, 4))
    masks = {"all": np.ones(3, dtype=bool), "empty": np.zeros(3, dtype=bool)}
    result = stratified_statistics(values, np.ones((4, 4)), 1., np.ones((4, 4)), 3., masks)
    assert result["all"]["fixed_unit_square_fraction"] is None
    assert result["all"]["robust_objective_fraction"] is None
    assert result["empty"]["empty"]
    assert "fixed_unit_rms" not in result["empty"]


def test_same_residual_invariant_to_subdivision_and_uses_physical_conversion():
    values = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    scale, factors = np.array([.5, 1., 2., 4.]), np.array([.01, .02, .03, .04])
    result = residual_statistics(values, scale, .5, factors, 3.)
    repeat = residual_statistics(np.repeat(values, 5, axis=0), scale, .5, factors, 3.)
    assert result["fixed_unit_rms"] == pytest.approx(repeat["fixed_unit_rms"])
    assert result["common_pseudo_huber_objective"] == pytest.approx(repeat["common_pseudo_huber_objective"])
    assert result["raw_physical_rms"] == pytest.approx(np.sqrt(np.mean((values*factors)**2)))


def test_saved_strata_replay_cannot_confuse_squared_and_robust_objectives():
    value = residual_statistics(np.full((2, 4), 20.), np.ones(4), 1., np.ones(4), 3.)
    saved = {"rows": 2, "raw_internal_rms": value["raw_internal_rms"],
             "raw_internal_maximum": value["raw_internal_maximum"],
             "normalized_rms": value["fixed_unit_rms"],
             "normalized_squared_objective": value["fixed_unit_squared_rms"],
             "normalized_objective": value["common_pseudo_huber_objective"],
             "raw_physical_rms_by_output_or_pair": value["raw_physical_rms_by_output_or_pair"]}
    assert compare_saved_strata({"all": value}, {"all": saved}) == 1
    with pytest.raises(AssertionError):
        compare_saved_strata({"all": value}, {"all": {**saved,
            "normalized_objective": saved["normalized_squared_objective"]}})


def test_selected_mean_and_literal_initial_covariance_phase_are_verified():
    state = {"mean_network.0.weight": torch.tensor([[2.]]),
             "covariance_network.0.weight": torch.tensor([[3.]])}
    saved = {name: {key: value.clone() for key, value in state.items()}
             for name in ("initial_state", "best_mean_state", "model_state", "best_state")}
    hashes = {name: _state_hash({"mean_network.0.weight": state["mean_network.0.weight"]})
              for name in ("best_mean_state", "model_state", "best_state")}
    assert verify_selected_mean(saved, hashes)["selected_mean_unchanged"]
    saved["model_state"]["mean_network.0.weight"] += 1
    with pytest.raises(ValueError, match="Selected mean"):
        verify_selected_mean(saved, hashes)


def test_residual_review_rejects_missing_all_mask_and_incompatible_units():
    with pytest.raises(ValueError, match="all-rows"):
        stratified_statistics(np.ones((2, 4)), np.ones(4), 1., np.ones(4), 3., {})
    with pytest.raises(ValueError, match="Compatible"):
        residual_statistics(np.ones((2, 4)), np.ones(3), 1., np.ones(3), 3.)


def test_real_checkpoint_and_two_model_same_pool_review_resume(tmp_path, monkeypatch):
    """A tiny software fixture exercises actual jets, phase arrays and I/O."""
    import h5py

    from analysis import audit_payoff_boundary_validation as validation
    from analysis import audit_payoff_regimes_validation as historical_validation
    from analysis import payoff_legacy_control as legacy
    from analysis import review_payoff_boundary_fit as review
    from analysis import run_payoff_boundary_regimes as runner
    from nn_dk.data import source_hash
    from nn_dk.interacting_config import InteractingStudyConfig
    from nn_dk.interacting_training import _bank
    from nn_dk.payoff_boundary import (
        BoundaryRepairConfig,
        fit_boundary_payoff,
        load_boundary_payoff_model,
    )
    from nn_dk.payoff_learning import PayoffBank
    from nn_dk.payoff_repair import RepairPayoffModel

    torch.set_num_threads(1)
    cfg = InteractingStudyConfig(n_cells=8, width=4, train_paths=2, validation_paths=2,
        state_fractions=(0., 1.), horizon_samples=1, adam_steps=1, lbfgs_steps=0, batch_size=4)
    repair = BoundaryRepairConfig(cfg)
    bank_spec = PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon)
    raw = np.random.default_rng(7547).uniform(.8, 1.2, (8, 8))
    states = raw/raw.sum(-1, keepdims=True)
    horizons = np.linspace(.002, cfg.horizon, 8)

    def bank(role):
        return {"states": states, "horizons": horizons,
                "family_ids": np.array([f"{role}-{i//2}" for i in range(8)]),
                "content_hash": f"literal-{role}", "source_hash": source_hash(),
                "config_hash": cfg.config_hash, "role": role,
                "case_ids": np.array(["uniform"]*4+["nonuniform"]*4),
                "snapshot_times": np.tile([0., cfg.horizon], 4)}

    checkpoint = tmp_path/"selected.pt"
    fit_boundary_payoff(repair, bank_spec, 0, checkpoint, bank("train"), bank("validation"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = load_boundary_payoff_model(checkpoint)
    actual_bank = bank("validation")
    h5_path = tmp_path/"validation.h5"
    with h5py.File(h5_path, "w") as stream:
        for profile, families in (("uniform", ["validation-0", "validation-1"]),
                                  ("nonuniform", ["validation-2", "validation-3"])):
            group = stream.create_group(profile)
            group.create_dataset("family_ids", data=np.asarray(families, dtype="S32"))
            group.create_dataset("stopped_at", data=np.full(2, np.inf))
    actual_bank["path"] = str(h5_path)
    _, fingerprint = _bank(actual_bank, cfg, "validation")
    metadata, masks = review._metadata(actual_bank, cfg.spec, cfg.horizon)
    values = review.evaluate_residuals(model, states, horizons, batch_size=3)
    arrays = {f"full_final_{stage}_residual": value for stage, value in values.items()}
    for phase, state_name in (("initial", "initial_state"),
                               ("selected_mean_covariance_initial", "best_mean_state"),
                               ("final_selected", "best_state")):
        control = type(model)(cfg.spec, cfg.physics, cfg.horizon, bank_spec, width=4)
        control.load_state_dict(saved[state_name])
        phase_values = review.evaluate_residuals(control, states, horizons)
        arrays.update({f"subset_{phase}_{stage}_residual": value for stage, value in phase_values.items()})
    numeric = historical_validation._npz(tmp_path/"audit/validation_numeric/selected.npz", arrays)
    sample = historical_validation._npz(tmp_path/"audit/validation_samples/case.npz", {"indices": np.arange(8)})
    full_summary = {name: {stage: validation.residual_summary(values[stage][mask], horizons[mask],
        saved["provenance"]["loss_scaling"], model, stage) for stage in review.STAGES}
        for name, mask in masks.items()}
    current_record = {"name": "case_boundary_seed0", "case": "case", "seed": 0,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": review.sha(checkpoint),
        "numeric_evidence": numeric, "selected_mean_hashes": historical_validation.selected_mean_hashes(saved),
        "full_selected_stratified_residuals": full_summary,
        "full_selected_active_counts": {"physical_active": 8, "physical_absorbed": 0,
            "model_threshold_active": 8, "rounded_boundary_active_disagreement": 0},
        "selected_steps": saved["best_step"], "allocation": saved["steps_done"],
        "training_seconds": saved["training_seconds"], "full_selected_payoff_consistency": {}}
    review._json(tmp_path/"payoffs/fit_lock.json", {"complete": True})
    review._json(tmp_path/"payoffs/data_lock.json", {"complete": True})
    audit = {"complete": True, "identity": {"source_hash": source_hash(),
        "fit_lock_sha256": review.sha(tmp_path/"payoffs/fit_lock.json")},
        "no_selection_refit_or_future_outcome_access": True,
        "records": [current_record], "samples": {"case": {"full_validation_fingerprint": fingerprint,
            "sample_indices": list(range(8)), "sample_artifact": sample}}}
    review._json(tmp_path/"audit/postfit_validation_diagnostic.json", audit)
    review._json(tmp_path/"comparison/legacy_controls.json", {"models": [{"case": "case", "seed": 0,
        "name": "case_legacy_v2_seed0", "checkpoint_sha256": "fixture-historical-sha"}]})
    monkeypatch.setattr(runner, "_assert_frozen", lambda root: (object(), None))
    monkeypatch.setattr(runner, "verify_fit_lock", lambda root: {"models": [current_record]})
    monkeypatch.setattr(runner, "configuration", lambda case, plan: cfg)
    monkeypatch.setattr(runner, "load_state_bank", lambda root, case, role: actual_bank)
    monkeypatch.setattr(runner, "payoff_bank", lambda case, plan: bank_spec)
    historical = RepairPayoffModel(cfg.spec, cfg.physics, cfg.horizon, bank_spec, width=4, quadrature_nodes=16)
    calls = []
    monkeypatch.setattr(legacy, "load_model", lambda item: (calls.append(item["name"]) or historical))
    result = review.run(tmp_path)
    assert result["complete"] and len(result["records"]) == 1
    record = result["records"][0]
    assert record["selected_mean_verification"]["selected_mean_unchanged"]
    assert set(record["current_phase_strata"]) == set(review.PHASES)
    assert record["legacy_on_new_validation_full_strata"]["all"]["mean"]["rows"] == 8
    assert not result["identity"]["future_outcomes_opened"]
    again = review.run(tmp_path)
    assert again["records"] == result["records"]
    assert len(calls) == 1
    assert json.loads((tmp_path/"audit/fit_interpretation_review.json").read_text())["complete"]
    assert not (tmp_path/"reference").exists()
