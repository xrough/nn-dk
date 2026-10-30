"""Distinct v3 identity, exact resumption, frozen means, and zero-time autodiff."""
import time
from dataclasses import replace

import numpy as np
import pytest
import torch

import nn_dk.payoff_repair as historical
from nn_dk.data import source_hash
from nn_dk.interacting_config import InteractingStudyConfig
from nn_dk.native import native_coefficients
from nn_dk.payoff_boundary import (
    BOUNDARY_PAYOFF_KIND,
    BoundaryPayoffModel,
    BoundaryRepairConfig,
    fit_boundary_payoff,
    load_boundary_payoff_model,
)
from nn_dk.payoff_learning import PayoffBank


def setup_problem():
    study = replace(InteractingStudyConfig(), n_cells=8, width=4, train_paths=1,
        validation_paths=1, state_fractions=(0., 1.), horizon_samples=1,
        adam_steps=2, lbfgs_steps=1, lbfgs_max_iter=1, batch_size=4, validation_every=1)
    cfg = BoundaryRepairConfig(study, quadrature_nodes=4)
    rng = np.random.default_rng(731)

    def pool(role):
        states = rng.uniform(.8, 1.2, (2, 8))
        states /= states.sum(-1, keepdims=True)
        return {"states": np.repeat(states, 2, 0),
            "horizons": rng.uniform(0, study.horizon, 4),
            "family_ids": np.repeat([f"{role}-{i}" for i in range(2)], 2),
            "source_hash": source_hash(), "config_hash": study.config_hash,
            "content_hash": f"boundary-literal-{role}", "role": role}

    return cfg, PayoffBank.mixed_terminal_integral(study.spec, study.horizon), pool("train"), pool("validation")


def test_boundary_fit_resume_and_historical_namespace_isolation(tmp_path):
    torch.set_num_threads(1)
    cfg, bank, train, val = setup_problem()
    old_kind, old_class, old_loss = historical.REPAIRED_PAYOFF_KIND, historical.RepairPayoffModel, historical._objective
    complete, resumed = tmp_path/"complete.pt", tmp_path/"resumed.pt"
    fitted = fit_boundary_payoff(cfg, bank, 0, complete, train, val)
    assert fitted["complete"] and fitted["kind"] == BOUNDARY_PAYOFF_KIND != old_kind
    partial = fit_boundary_payoff(cfg, bank, 0, resumed, train, val, deadline=time.monotonic()-1)
    assert not partial["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        load_boundary_payoff_model(resumed)
    fit_boundary_payoff(cfg, bank, 0, resumed, train, val)
    a, b = (torch.load(p, weights_only=True) for p in (complete, resumed))
    assert a["model_state_hash"] == b["model_state_hash"]
    assert a["best_state_hash"] == b["best_state_hash"]
    assert all(torch.equal(a["model_state"][k], a["best_mean_state"][k])
               for k in a["model_state"] if k.startswith("mean_network."))
    assert historical.RepairPayoffModel is old_class and historical._objective is old_loss
    assert historical.REPAIRED_PAYOFF_KIND == old_kind
    model = load_boundary_payoff_model(complete)
    assert model.config.config_hash == cfg.config_hash
    assert model.loss_scaling["inverse_time_weighting"] is False
    assert model.loss_scaling["normalization_time_powers"] == [0]*4
    assert model.loss_scaling["pseudo_huber_delta"] == 3
    prediction = model.predict_conditional(train["states"][:2], [0., cfg.study.horizon])
    assert np.isfinite(prediction.mean).all()
    assert np.linalg.eigvalsh(prediction.scaled_covariance).min() >= -1e-12
    # Frozen mean parameters cannot receive covariance-stage gradients.
    model.zero_grad(set_to_none=True)
    model.covariance_residual(train["states"], train["horizons"]).square().mean().backward()
    assert all(p.grad is None for p in model.mean_network.parameters())


def test_boundary_checkpoint_rejects_changed_robust_loss_identity(tmp_path):
    cfg, bank, train, val = setup_problem()
    path = tmp_path/"original.pt"
    fit_boundary_payoff(cfg, bank, 0, path, train, val)
    saved = torch.load(path, weights_only=True)
    saved["provenance"]["loss_scaling"]["inverse_time_weighting"] = True
    # Rehashing the outer metadata cannot conceal a changed objective.
    from nn_dk.types import stable_hash
    provenance = saved["provenance"]
    provenance["identity_hash"] = stable_hash({k: v for k, v in provenance.items() if k != "identity_hash"})
    changed = tmp_path/"changed.pt"
    torch.save(saved, changed)
    with pytest.raises(ValueError, match="robust loss identity"):
        load_boundary_payoff_model(changed)


def test_value_autodiff_has_exact_native_right_derivative_at_zero():
    cfg, bank, _, _ = setup_problem()
    model = BoundaryPayoffModel(cfg.study.spec, cfg.study.physics, cfg.study.horizon, bank, width=4)
    state = torch.full((8,), 1/8, dtype=torch.float64)
    state[0] = model.spec.mass_stop+model.boundary_width/2
    state[4] += 1/8-state[0]
    native = native_coefficients(state.numpy(), model.context, model.spec)
    phi = bank.terminal_weights/bank.scales[:, None]
    q = bank.running_weights/bank.scales[:, None]
    expected_mean = np.sqrt(model.context.n_particles)*(phi@native.drift+q@state.numpy())
    expected_covariance = phi@native.covariance@phi.T
    zero = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    for component, expected in ((0, expected_mean), (1, expected_covariance)):
        slope = torch.autograd.functional.jacobian(
            lambda t: model.moment_values(state[None], t[None])[component][0], zero)
        np.testing.assert_allclose(slope.detach(), expected, atol=3e-12, rtol=2e-12)


def test_boundary_configuration_roundtrip_and_invalid_loss_scale():
    cfg, _, _, _ = setup_problem()
    assert BoundaryRepairConfig.from_dict(cfg.to_dict()) == cfg
    assert replace(cfg, pseudo_huber_delta=2.).config_hash != cfg.config_hash
    with pytest.raises(ValueError, match="pseudo-Huber"):
        replace(cfg, pseudo_huber_delta=0.)
