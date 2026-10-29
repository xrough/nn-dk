"""Crossed proposals/losses keep natural validation, parents and resume intact."""
import copy
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import fit_payoff_collocation_ablation as fit  # noqa: E402
from nn_dk.interacting_config import InteractingStudyConfig  # noqa: E402
from nn_dk.payoff_boundary import BoundaryRepairConfig  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402


def problem(proposal="stratified", loss="pseudo_huber", *, lbfgs=1):
    study = replace(InteractingStudyConfig(), n_cells=8, width=2, optimizer_seeds=(0, 1, 2),
        train_paths=1, validation_paths=1, state_fractions=(0., 1.), horizon_samples=1,
        adam_steps=2, lbfgs_steps=lbfgs, lbfgs_max_iter=1, batch_size=3, validation_every=1)
    cfg = fit.CollocationAblationConfig(BoundaryRepairConfig(study, quadrature_nodes=2),
                                       proposal=proposal, loss=loss)
    bank = PayoffBank.mixed_terminal_integral(study.spec, study.horizon)
    rng = np.random.default_rng(1557)

    def make(role):
        states = rng.uniform(.8, 1.2, (5, 8))
        states /= states.sum(-1, keepdims=True)
        return fit.make_flat_bank(states, rng.uniform(.005, study.horizon, 5),
            [f"{role}-a", f"{role}-a", f"{role}-b", f"{role}-c", f"{role}-c"],
            role=role, config=cfg, provenance={"stream_namespace": f"software-fixture/{role}",
                "parent_pool_sha256": fit._digest(states), "stress_augmentation": False})
    return cfg, bank, make("train"), make("validation")


def read(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def write_resigned(path, payload):
    payload = copy.deepcopy(payload)
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = fit._digest(payload)
    torch.save(payload, path)


def test_real_staged_optimization_resumes_bitwise_mid_mean_and_covariance(tmp_path):
    torch.set_num_threads(1)
    cfg, bank, train, val = problem()
    full, resumed = tmp_path/"full.pt", tmp_path/"resumed.pt"
    caller_rng = torch.get_rng_state().clone()
    result = fit.fit_collocation_ablation(cfg, bank, 0, full, train, val)
    assert result["complete"] and result["kind"] == fit.KIND
    partial = fit.fit_collocation_ablation(cfg, bank, 0, resumed, train, val, max_updates=1)
    assert not partial["complete"] and partial["stage"] == "mean"
    with pytest.raises(ValueError, match="incomplete"):
        fit.load_collocation_ablation_model(resumed)
    partial = fit.fit_collocation_ablation(cfg, bank, 0, resumed, train, val, max_updates=3)
    assert not partial["complete"] and partial["stage"] == "covariance"
    intermediate = read(resumed)
    for name in ("model_state", "best_state"):
        assert fit._digest(fit._selected_mean(intermediate[name])) == fit._digest(
            fit._selected_mean(intermediate["best_mean_state"]))
    fit.fit_collocation_ablation(cfg, bank, 0, resumed, train, val)
    first, second = read(full), read(resumed)
    for field in ("model_state", "best_state", "best_mean_state", "optimizer_states",
                  "sampler_state", "torch_rng_state", "steps_done", "best_validation", "best_step"):
        assert fit._digest(first[field]) == fit._digest(second[field]), field
    assert first["steps_done"] == {"mean_adam": 2, "mean_lbfgs": 1,
                                   "covariance_adam": 2, "covariance_lbfgs": 1}
    assert first["sampler_state"]["draws"] == 12
    torch.testing.assert_close(torch.get_rng_state(), caller_rng, atol=0, rtol=0)
    loaded = fit.load_collocation_ablation_model(resumed)
    assert loaded.complete and loaded.method_kind == fit.KIND
    assert loaded.config.config_hash == cfg.config_hash
    prediction = loaded.predict_conditional(train["states"][:2], [0., cfg.repair.study.horizon])
    assert np.isfinite(prediction.mean).all()
    assert np.linalg.eigvalsh(prediction.scaled_covariance).min() >= -2e-14
    loaded.zero_grad(set_to_none=True)
    loaded.covariance_residual(train["states"], train["horizons"]).square().mean().backward()
    assert all(parameter.grad is None for parameter in loaded.mean_network.parameters())
    original_bytes = resumed.read_bytes()
    replay = fit.fit_collocation_ablation(cfg, bank, 0, resumed, train, val)
    assert replay["complete"] and resumed.read_bytes() == original_bytes


def test_crossed_candidates_share_pool_scales_and_initial_weights(tmp_path):
    cfg, bank, train, val = problem(lbfgs=0)
    initial, scaling, data = set(), set(), set()
    methods = set()
    for proposal in ("uniform", "stratified"):
        for loss in ("pseudo_huber", "squared"):
            candidate = replace(cfg, proposal=proposal, loss=loss)
            path = tmp_path/f"{proposal}-{loss}.pt"
            result = fit.fit_collocation_ablation(candidate, bank, 0, path, train, val,
                                                  deadline=time.monotonic()-1)
            assert not result["complete"]
            saved = read(path)
            initial.add(fit._digest(saved["initial_state"]))
            scaling.add(fit._digest(saved["loss_scaling"]))
            data.add(fit._digest(saved["identity"]["data"]))
            methods.add(result["method_id"])
    assert len(initial) == len(scaling) == len(data) == 1
    assert len(methods) == 4
    assert fit.CollocationAblationConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="proposal"):
        replace(cfg, proposal="balanced-unweighted")


@pytest.mark.parametrize("loss", ["pseudo_huber", "squared"])
def test_importance_weights_multiply_complete_per_row_loss_after_robustification(loss):
    theta = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    features = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.], [.1, .2, .3, .4]], dtype=torch.float64)
    residual = theta*features-.2
    q = torch.tensor([.5, .25, .25], dtype=torch.float64)
    weights = (1/3)/q
    rows = fit.row_objective(residual, torch.ones(4), .7, loss=loss, delta=.2)
    natural = rows.mean()
    corrected = (q*weights*rows).sum()
    natural_gradient = torch.autograd.grad(natural, theta, retain_graph=True)[0]
    corrected_gradient = torch.autograd.grad(corrected, theta, retain_graph=True)[0]
    torch.testing.assert_close(corrected, natural, atol=2e-16, rtol=2e-15)
    torch.testing.assert_close(corrected_gradient, natural_gradient, atol=2e-16, rtol=2e-15)
    if loss == "pseudo_huber":
        incorrectly_weighted = fit.row_objective(residual*weights[:, None], torch.ones(4), .7,
                                                loss=loss, delta=.2)
        assert not torch.isclose((q*incorrectly_weighted).sum(), natural, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("loss", ["pseudo_huber", "squared"])
def test_optimizer_loss_uses_literal_p_over_q_without_self_normalization(loss):
    cfg, *_ = problem(loss=loss)
    parameter = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    states = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.], [.1, .2, .3, .4]], dtype=torch.float64)
    times = torch.zeros(3, dtype=torch.float64)
    probabilities = torch.tensor([.5, .25, .25], dtype=torch.float64)
    importance = (1/3)/probabilities

    class ResidualControl:
        horizon = .7

        def mean_residual(self, initial, horizons):
            return parameter*initial-.2

    model, scaling = ResidualControl(), {"mean_scale": [1., 1., 1., 1.]}
    natural = fit._loss(model, states, times, scaling, "mean", cfg)
    expected_draw = sum(probabilities[row]*fit._loss(
        model, states[row:row+1], times[row:row+1], scaling, "mean", cfg, importance[row:row+1])
        for row in range(3))
    natural_gradient = torch.autograd.grad(natural, parameter, retain_graph=True)[0]
    draw_gradient = torch.autograd.grad(expected_draw, parameter)[0]
    torch.testing.assert_close(expected_draw, natural, atol=3e-16, rtol=3e-15)
    torch.testing.assert_close(draw_gradient, natural_gradient, atol=3e-16, rtol=3e-15)


def test_common_validation_and_checkpoint_selection_use_natural_mse(tmp_path, monkeypatch):
    cfg, bank, train, val = problem(lbfgs=0)
    tensors, fingerprint = fit._bank(val, cfg, "validation")
    model = fit._model(cfg, bank, 0)
    training, train_fingerprint = fit._bank(train, cfg, "train")
    scaling = fit._scaling(model, training, cfg.repair, train_fingerprint)
    robust = fit.natural_validation(model, tensors, scaling, "mean", cfg)
    squared = fit.natural_validation(model, tensors, scaling, "mean", replace(cfg, loss="squared"))
    assert robust == squared and robust["rows"] == fingerprint["rows"]
    residual = model.mean_residual(tensors["states"], tensors["horizons"])
    expected = ((cfg.repair.study.horizon*residual/torch.tensor(
        scaling["mean_scale"], dtype=torch.float64))**2).mean()
    assert robust["natural_mse"] == pytest.approx(float(expected.detach()), rel=2e-14)
    scores = {"mean": [(4., 0.), (1., 8.), (2., .1)], "covariance": [(3., 0.)]*4}
    calls = {stage: 0 for stage in fit.STAGES}
    observed = []

    def opposing_rankings(current, inputs, scales, stage, configuration):
        mse, huber = scores[stage][calls[stage]]
        calls[stage] += 1
        if stage == "mean":
            observed.append(fit._digest(fit._selected_mean(current.state_dict())))
        return {"natural_mse": mse, "natural_huber": huber, "rows": len(inputs["states"])}

    monkeypatch.setattr(fit, "natural_validation", opposing_rankings)
    path = tmp_path/"opposing-rankings.pt"
    fit.fit_collocation_ablation(cfg, bank, 0, path, train, val)
    saved = read(path)
    assert saved["best_step"]["mean"] == 1
    assert saved["best_validation"]["mean"] == {"natural_mse": 1., "natural_huber": 8., "rows": 5}
    assert fit._digest(fit._selected_mean(saved["best_mean_state"])) == observed[1]


@pytest.mark.parametrize("field", ["weights", "optimizer", "sampler", "completion"])
def test_corrupted_checkpoint_payloads_are_rejected(tmp_path, field):
    cfg, bank, train, val = problem(lbfgs=0)
    path = tmp_path/"partial.pt"
    fit.fit_collocation_ablation(cfg, bank, 0, path, train, val, max_updates=1)
    saved = read(path)
    if field == "weights":
        saved["model_state"]["mean_network.0.weight"][0, 0] += 1
    elif field == "optimizer":
        saved["optimizer_states"]["mean_adam"]["param_groups"][0]["lr"] *= 2
    elif field == "sampler":
        saved["sampler_state"]["rng_state"]["state"]["state"] += 1
    else:
        saved["complete"] = True
    torch.save(saved, path)
    with pytest.raises(ValueError, match="checksum changed"):
        fit.load_collocation_ablation_model(path)
    if field == "completion":
        write_resigned(path, saved)
        with pytest.raises(ValueError, match="completion/stage"):
            fit.load_collocation_ablation_model(path)


def test_resume_rejects_changed_pool_source_helpers_and_variant(tmp_path, monkeypatch):
    cfg, bank, train, val = problem(lbfgs=0)
    path = tmp_path/"partial.pt"
    fit.fit_collocation_ablation(cfg, bank, 0, path, train, val, max_updates=1)
    changed = copy.deepcopy(train)
    changed["states"][0, :2] += [.001, -.001]
    with pytest.raises(ValueError, match="content checksum"):
        fit.fit_collocation_ablation(cfg, bank, 0, path, changed, val)
    with pytest.raises(ValueError, match="resume"):
        fit.fit_collocation_ablation(replace(cfg, loss="squared"), bank, 0, path, train, val)
    actual_sources = fit._sources()
    monkeypatch.setattr(fit, "_sources", lambda: actual_sources | {"changed_helper": True})
    with pytest.raises(ValueError, match="source, helper"):
        fit.load_collocation_ablation_model(path)


def test_parent_groups_stress_augmentation_and_no_future_label_contract(tmp_path):
    cfg, bank, train, val = problem(lbfgs=0)
    provenance = train["provenance"] | {"stress_augmentation": True}
    augmented = fit.make_flat_bank(train["states"], train["horizons"],
        ["derived-a", "derived-b", "train-b", "train-c", "train-c"], role="train", config=cfg,
        parent_family_ids=train["parent_family_ids"], row_kinds=["stress", "stress", "native", "native", "native"],
        provenance=provenance)
    _, fingerprint = fit._bank(augmented, cfg, "train")
    assert fingerprint["rows"] == 5 and fingerprint["parent_families"] == 3
    assert fingerprint["row_counts"] == {"native": 3, "stress": 2}
    with pytest.raises(ValueError, match="native-only"):
        fit.make_flat_bank(val["states"], val["horizons"], val["family_ids"], role="validation", config=cfg,
            row_kinds=["stress"]*5, provenance=val["provenance"] | {"stress_augmentation": True})
    overlapping = fit.make_flat_bank(val["states"], val["horizons"], val["family_ids"],
        role="validation", config=cfg, parent_family_ids=["train-a"]*5, provenance=val["provenance"])
    with pytest.raises(ValueError, match="parent families must be disjoint"):
        fit.fit_collocation_ablation(cfg, bank, 0, tmp_path/"overlap.pt", augmented, overlapping)
    for field, value in (("future_payoff_labels", True), ("future_moment_labels", [1]*5)):
        future = train | {field: value}
        with pytest.raises(ValueError, match="future labels"):
            fit.fit_collocation_ablation(cfg, bank, 0, tmp_path/"labels.pt", future, val)
    with pytest.raises(ValueError, match="optimizer seed"):
        fit.fit_collocation_ablation(cfg, bank, 99, tmp_path/"seed.pt", train, val)


def test_changed_validation_rows_cannot_change_training_only_normalization(tmp_path):
    cfg, bank, train, val = problem(lbfgs=0)
    changed_val = fit.make_flat_bank(val["states"][::-1], val["horizons"][::-1], val["family_ids"][::-1],
        role="validation", config=cfg, provenance=val["provenance"])
    paths = [tmp_path/"a.pt", tmp_path/"b.pt"]
    for path, valid in zip(paths, (val, changed_val), strict=True):
        fit.fit_collocation_ablation(cfg, bank, 0, path, train, valid, max_updates=0)
    assert read(paths[0])["loss_scaling"] == read(paths[1])["loss_scaling"]


def test_resigned_payload_cannot_hide_nonfinite_optimizer_or_changed_frozen_mean(tmp_path):
    cfg, bank, train, val = problem()
    path = tmp_path/"covariance.pt"
    fit.fit_collocation_ablation(cfg, bank, 0, path, train, val, max_updates=4)
    saved = read(path)
    assert saved["stage"] == "covariance"
    changed = copy.deepcopy(saved)
    changed["model_state"]["mean_network.0.weight"][0, 0] += 1
    write_resigned(path, changed)
    with pytest.raises(ValueError, match="frozen selected mean"):
        fit.load_collocation_ablation_model(path)
    changed = copy.deepcopy(saved)
    state = next(iter(changed["optimizer_states"]["mean_adam"]["state"].values()))
    state["exp_avg"].flatten()[0] = float("inf")
    write_resigned(path, changed)
    with pytest.raises(ValueError, match="Nonfinite checkpoint optimizer"):
        fit.load_collocation_ablation_model(path)
