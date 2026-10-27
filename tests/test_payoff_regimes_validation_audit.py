"""Independent phase reconstruction, scaling and physical residual diagnostics."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit_payoff_regimes_validation as audit  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.interacting_config import InteractingStudyConfig  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.payoff_repair import (  # noqa: E402
    RepairConfig,
    RepairPayoffModel,
    fit_repaired_payoff,
    load_repaired_payoff_model,
)
from nn_dk.types import Context  # noqa: E402


def _model():
    spec, horizon = NativeDKSpec(8), 1/16
    context = Context(n_particles=2048, kappa=.75, a2=.1)
    return RepairPayoffModel(spec, context, horizon,
                             PayoffBank.mixed_terminal_integral(spec, horizon), width=4, quadrature_nodes=16)


def _states():
    raw = np.random.default_rng(774).uniform(.8, 1.2, (8, 8))
    return raw/raw.sum(-1, keepdims=True)


def test_manual_single_pass_equations_equal_model_residual_and_time_scaling():
    torch.set_num_threads(1)
    model, states = _model(), _states()
    with torch.no_grad():
        model.mean_network[-1].bias.add_(.01)
        model.covariance_network[-1].bias.add_(.03)
    times = np.linspace(.001, model.horizon, len(states))
    pool = audit.evaluate_pool(model, states, times, batch_size=3)
    with torch.no_grad():
        mean_residual, covariance_residual = model.residual(torch.tensor(states), torch.tensor(times))
    np.testing.assert_allclose(pool["mean_residual"], mean_residual, atol=2e-13, rtol=1e-12)
    np.testing.assert_allclose(pool["covariance_residual"], covariance_residual, atol=2e-13, rtol=1e-12)
    scaling = {"small_time_powers": [0, 0, 1, 1], "scale_time_floor": .001,
               "mean_scale": [1., 2., 3., 4.], "covariance_scale": np.outer([1., 2., 3., 4.], [1., 2., 3., 4.])}
    mr = audit.normalized_residual(pool["mean_residual"], times, scaling, model.horizon, "mean")
    expected = model.horizon*pool["mean_residual"]/np.array(scaling["mean_scale"])
    expected[:, 2:] /= (times/model.horizon)[:, None]
    np.testing.assert_allclose(mr, expected, atol=1e-14)
    cr = audit.normalized_residual(pool["covariance_residual"], times, scaling, model.horizon, "covariance")
    expected = model.horizon*pool["covariance_residual"]/scaling["covariance_scale"]
    power = np.array([0, 0, 1, 1])
    expected /= (times/model.horizon)[:, None, None]**(power[:, None]+power[None, :])
    np.testing.assert_allclose(cr, expected, atol=1e-14)


def test_covariance_stage_baseline_uses_selected_mean_and_literal_initial_covariance():
    model = _model()
    initial = copy.deepcopy(model.state_dict())
    selected = copy.deepcopy(initial)
    selected["mean_network.6.bias"].add_(.25)
    current = copy.deepcopy(selected)
    current["covariance_network.6.bias"].add_(.4)
    saved = {"initial_state": initial, "best_mean_state": selected,
             "model_state": current, "best_state": copy.deepcopy(current)}
    hashes = audit.selected_mean_hashes(saved)
    assert len(set(hashes.values())) == 1
    baseline = audit.covariance_phase_baseline(model, saved)
    for key, value in baseline.state_dict().items():
        expected = selected[key] if key.startswith("mean_network.") else initial[key]
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
    tampered = copy.deepcopy(saved)
    tampered["model_state"]["mean_network.6.bias"].add_(1)
    with pytest.raises(ValueError, match="Selected mean"):
        audit.selected_mean_hashes(tampered)
    tampered = copy.deepcopy(saved)
    tampered["best_mean_state"]["covariance_network.6.bias"].add_(1)
    with pytest.raises(ValueError, match="before the mean"):
        audit.selected_mean_hashes(tampered)


def test_deterministic_stratified_subset_covers_profiles_snapshots_clocks_and_low_mass():
    states = np.tile(np.full(8, 1/8), (2*4*4*16, 1))
    profiles, snapshots, horizons = [], [], []
    row = 0
    for profile in ("uniform", "nonuniform"):
        for snapshot in (0., .25, .5, 1.):
            for quartile in range(4):
                for rank in range(16):
                    states[row, 0] = .001+.004*rank
                    states[row, 1:] *= (1-states[row, 0])/states[row, 1:].sum()
                    profiles.append(profile)
                    snapshots.append(snapshot)
                    horizons.append((quartile+.5)/4)
                    row += 1
    args = (states, np.array(horizons), np.array(profiles), np.array(snapshots), 1.)
    selected = audit.deterministic_subset(*args)
    np.testing.assert_array_equal(selected, audit.deterministic_subset(*args))
    assert len(selected) == len(set(selected)) == 256
    assert set(np.array(profiles)[selected]) == {"uniform", "nonuniform"}
    for group in range(32):
        local = selected[(selected >= group*16) & (selected < (group+1)*16)]-group*16
        assert len(local) == 8 and local[0] == 0 and local[-1] == 15


def test_physical_absorption_stratum_is_distinct_from_strict_rounded_mass_gate():
    model, states = _model(), _states()[:2]
    states[0, 0] = model.spec.mass_stop+1e-18
    states[0, 1:] *= (1-states[0, 0])/states[0, 1:].sum()
    metadata = {"physical_active": np.array([False, True]), "profiles": np.array(["uniform", "nonuniform"]),
                "snapshot_times": np.array([model.horizon, 0.])}
    masks = audit.stratum_masks(states, np.array([.01, .03]), metadata, model.spec, model.horizon)
    assert masks["physically_absorbed"][0]
    assert masks["model_threshold_active"][0]
    assert masks["minimum_mass/at_or_below_boundary"][0]
    assert masks["physical_active_only"][1]


def test_bounded_payoff_checks_detect_unphysical_mean_and_variance_without_psd_repair():
    model, states = _model(), _states()
    times = np.full(len(states), model.horizon)
    pool = audit.evaluate_pool(model, states, times)
    normal = audit.bounded_payoff_diagnostics(model, pool, times)
    assert sum(normal["mean_bound_violations_by_output"]) == 0
    assert sum(normal["bounded_variance_violations_by_output"]) == 0
    broken = copy.deepcopy(pool)
    broken["mean"][:, 0] = 1e6
    broken["covariance"][:, 2, 2] = 1e8
    result = audit.bounded_payoff_diagnostics(model, broken, times)
    assert result["mean_bound_violations_by_output"][0] == len(states)
    assert result["bounded_variance_violations_by_output"][2] == len(states)


def test_full_selection_objective_replay_matches_a_real_two_stage_checkpoint(tmp_path):
    torch.set_num_threads(1)
    cfg = InteractingStudyConfig(n_cells=8, width=4, train_paths=2, validation_paths=2,
                                 state_fractions=(0., 1.), horizon_samples=1,
                                 adam_steps=1, lbfgs_steps=0, batch_size=4)
    repair = RepairConfig(cfg, quadrature_nodes=16)
    states, times = _states(), np.linspace(.002, cfg.horizon, 8)

    def bank(role):
        return {"states": states, "horizons": times,
                "family_ids": tuple(f"{role}-{i//2}" for i in range(8)),
                "content_hash": f"literal-{role}", "source_hash": source_hash(),
                "config_hash": cfg.config_hash, "role": role}

    path = tmp_path/"selected.pt"
    fit_repaired_payoff(repair, PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon),
                        0, path, bank("train"), bank("validation"))
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = load_repaired_payoff_model(path)
    audit.selected_mean_hashes(saved)
    pool = audit.evaluate_pool(model, states, times, batch_size=3)
    for stage in ("mean", "covariance"):
        summary = audit.residual_summary(pool[f"{stage}_residual"], times,
                                         saved["provenance"]["loss_scaling"], model, stage)
        np.testing.assert_allclose(summary["normalized_objective"],
                                   saved["best_validation"][stage]["objective"], rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(summary["raw_internal_rms"],
                                   saved["best_validation"][stage]["raw_rms"], rtol=2e-12, atol=1e-14)
