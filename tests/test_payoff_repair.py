"""Independent analytic, dense-autodiff and checkpoint tests for repaired heads."""

import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.linalg import expm

import nn_dk.interacting_training as training
import nn_dk.payoff_repair as repair
from nn_dk.interacting_config import InteractingStudyConfig
from nn_dk.native import native_coefficients
from nn_dk.native_types import NativeDKSpec
from nn_dk.observations import incidence_matrix
from nn_dk.payoff_learning import PayoffBank
from nn_dk.payoff_repair import (
    RepairConfig,
    RepairPayoffModel,
    fit_repaired_payoff,
    load_repaired_payoff_model,
)
from nn_dk.types import Context


@pytest.fixture(autouse=True)
def fixed_source(monkeypatch):
    monkeypatch.setattr(repair, "source_hash", lambda: "repair-test-source")
    monkeypatch.setattr(training, "source_hash", lambda: "repair-test-source")
    torch.set_num_threads(1)


def _model(mixed=True, interacting=True, g=8):
    spec = NativeDKSpec(g)
    context = Context(kappa=.75 if interacting else 0., a2=.1 if interacting else 0.,
                      n_particles=2048)
    bank = (PayoffBank.mixed_terminal_integral if mixed else PayoffBank.terminal_high_modes)(spec, 1/16)
    torch.manual_seed(7)
    return RepairPayoffModel(spec, context, 1/16, bank, width=4)


def _states(g=8, count=2):
    states = np.random.default_rng(91).uniform(.8, 1.2, (count, g))
    return torch.tensor(states/states.sum(-1, keepdims=True))


def _perturb(model):
    with torch.no_grad():
        for network in (model.mean_network, model.covariance_network):
            network[-1].weight.normal_(0, .01)
            network[-1].bias.normal_(0, .01)


@pytest.mark.parametrize("mixed", [False, True])
def test_boundary_initial_autonomy_and_values_match_jets(mixed):
    g = 8 if mixed else 16
    model, states = _model(mixed, g=g), _states(g)
    _perturb(model)
    times = torch.tensor([.02, .04], dtype=torch.float64)
    mean, covariance = model.moment_values(states, times)
    mj, cj = model._moment_jets(states, times)
    torch.testing.assert_close(mean, mj.value, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(covariance, cj.value, atol=1e-13, rtol=1e-13)
    assert np.linalg.eigvalsh(covariance.detach()).min() >= -1e-13
    mean0, covariance0 = model.moment_values(states, torch.zeros(2))
    torch.testing.assert_close(mean0, model.scaled_observables(states))
    assert torch.count_nonzero(covariance0) == 0
    stopped = states.clone()
    stopped[:, 0] = model.spec.mass_stop
    stopped[:, 1:] *= (1-model.spec.mass_stop)/stopped[:, 1:].sum(-1, keepdim=True)
    sm, sc = model.moment_values(stopped, times)
    torch.testing.assert_close(sm, model.scaled_observables(stopped))
    assert torch.count_nonzero(sc) == 0
    assert all(torch.count_nonzero(r) == 0 for r in model.residual(stopped, times))
    # Profiles determine initialization, not this autonomous conditional map.
    changed_profile = replace(model.context, cos_coeffs=(.2, 0., 0.))
    model.require_physics(changed_profile)
    model2 = RepairPayoffModel(model.spec, changed_profile, model.horizon, model.bank, 4)
    model2.load_state_dict(model.state_dict())
    torch.testing.assert_close(model2.moment_values(states, times)[0], mean)
    result = model.predict_conditional(states.numpy(), [0., 1/32, 1/16])
    assert result.mean.shape == (2, 3, 4)
    assert result.diagnostics["covariance_floor"] is None


def test_free_heat_integral_mean_is_exact_native_semigroup_and_residual():
    model, states = _model(interacting=False), _states()
    times = torch.tensor([.02, .04], dtype=torch.float64)
    mean, covariance = model.moment_values(states, times)
    heat = model.context.diffusion/model.spec.h**2*(
        np.roll(np.eye(8), 1, 0)+np.roll(np.eye(8), -1, 0)-2*np.eye(8))
    phi, q = model.bank.terminal_weights, model.bank.running_weights
    for i, time_value in enumerate(times):
        augmented = np.zeros((16, 16))
        augmented[:8, :8] = heat
        augmented[:8, 8:] = np.eye(8)
        transition = expm(float(time_value)*augmented)
        expected = (phi @ transition[:8, :8]+q @ transition[:8, 8:]) @ states[i].numpy()
        np.testing.assert_allclose(mean[i].detach().numpy(),
            np.sqrt(model.context.n_particles)*expected/model.bank.scales, atol=3e-13, rtol=1e-12)
    mr, cr = model.residual(states, times)
    torch.testing.assert_close(mr, torch.zeros_like(mr), atol=3e-11, rtol=0)
    torch.testing.assert_close(cr, torch.zeros_like(cr), atol=3e-10, rtol=0)
    assert torch.linalg.eigvalsh(covariance).min() > 0


def test_free_heat_covariance_matches_independent_lifted_matrix_exponential():
    model, states = _model(interacting=False), _states(count=1)
    g, n = model.spec.n_cells, model.context.n_particles
    h = model.spec.h
    heat = model.context.diffusion/h**2*(np.roll(np.eye(g), 1, 0)
                                        +np.roll(np.eye(g), -1, 0)-2*np.eye(g))
    # Augmented process X=[m, four running integrals]. Its drift is linear and
    # N-scaled covariance is driven by the independently propagated E[m_s].
    f = np.zeros((g+4, g+4))
    f[:g, :g], f[g:, :g] = heat, model.bank.running_weights
    size = g+4
    lift = np.zeros((g+size**2, g+size**2))
    lift[:g, :g] = heat
    lift[g:, g:] = np.kron(f, np.eye(size))+np.kron(np.eye(size), f)
    incidence = incidence_matrix(g)
    for k in range(g):
        basis = np.eye(g)[k]
        intensity = model.context.diffusion/h**2*(basis+np.roll(basis, -1))
        qv = np.zeros((size, size))
        qv[:g, :g] = (incidence*intensity) @ incidence.T
        lift[g:, k] = qv.reshape(-1)
    initial = np.zeros(g+size**2)
    initial[:g] = states[0]
    readout = np.concatenate((model.bank.terminal_weights, np.eye(4)), axis=1)
    for time_value in (1e-4, 1/64, 1/32, 1/16):
        expected = (expm(time_value*lift) @ initial)[g:].reshape(size, size)
        expected = readout @ expected @ readout.T
        prediction = model.predict_conditional(states.numpy(), [time_value]).scaled_covariance[0, 0]
        np.testing.assert_allclose(prediction, expected, atol=2e-13, rtol=3e-12)
    assert n == 2048  # N-scaled covariance is independent of the N prefactor.


def test_learned_covariance_preserves_integral_leading_qv_and_small_time_powers():
    model, states = _model(), torch.full((2, 8), 1/8, dtype=torch.float64)
    _perturb(model)
    times = torch.tensor([1e-8, 2e-8], dtype=torch.float64)
    _, covariance = model.moment_values(states, times)
    qv = native_coefficients(states[0].numpy(), model.context, model.spec).covariance
    q = model.bank.running_weights/model.bank.scales[:, None]
    expected = q @ qv @ q.T/3
    np.testing.assert_allclose((covariance[0]/times[0]**3).detach().numpy()[2:, 2:],
                               expected[2:, 2:], rtol=2e-6, atol=1e-5)
    ratios = covariance[1].diagonal()/covariance[0].diagonal()
    torch.testing.assert_close(ratios, torch.tensor([2., 2., 8., 8.], dtype=torch.float64),
                               rtol=2e-6, atol=2e-6)


def test_same_mode_terminal_integral_nonzero_cross_leading_order_is_preserved():
    spec, horizon = NativeDKSpec(8), 1/16
    weights = PayoffBank._weights(spec, ((1, "sin"), (1, "cos"), (1, "sin"), (1, "cos")))
    phi, running = weights.copy(), weights.copy()/horizon
    phi[2:], running[:2] = 0., 0.
    bank = PayoffBank("same-mode-small-time-control", phi, running, horizon)
    context = Context(kappa=.75, a2=.1, n_particles=2048)
    model = RepairPayoffModel(spec, context, horizon, bank, 4)
    _perturb(model)
    state = torch.full((1, 8), 1/8, dtype=torch.float64)
    t = torch.tensor([1e-8], dtype=torch.float64)
    covariance = model.moment_values(state, t)[1][0].detach().numpy()
    qv = native_coefficients(state[0].numpy(), context, spec).covariance
    expected = ((phi/bank.scales[:, None]) @ qv @ (running/bank.scales[:, None]).T)/2
    assert abs(expected[0, 2]) > 1
    np.testing.assert_allclose(covariance[:2, 2:]/float(t[0])**2,
                               expected[:2, 2:], rtol=2e-6, atol=1e-5)


def test_dense_autodiff_matches_exact_jets_including_congruence_products():
    model, states = _model(), _states(count=1)
    _perturb(model)
    state = states[0].requires_grad_()
    time_value = torch.tensor(.03, dtype=torch.float64, requires_grad=True)
    jets = model._moment_jets(state[None], time_value[None])
    drift, amplitude = path_coefficients_for_test(model, state)
    qv = (model.incidence*amplitude) @ (model.incidence*amplitude).T/model.context.n_particles
    for component, jet in enumerate(jets):
        def function(m, t):
            return model.moment_values(m[None], t[None])[component][0]
        jacobian = torch.autograd.functional.jacobian(lambda m: function(m, time_value), state)
        hessian = torch.autograd.functional.hessian(
            lambda m: function(m, time_value).sum(), state)
        generator_sum = (jacobian @ drift).sum()+.5*(hessian*qv).sum()
        torch.testing.assert_close(jet.generator.sum(), generator_sum, atol=3e-9, rtol=2e-10)
        # Time is represented through deterministic cached coefficients in
        # inference; use a high-accuracy finite difference independent of jets.
        eps = 1e-6
        numerical = (function(state, time_value.detach()+eps)
                     -function(state, time_value.detach()-eps))/(2*eps)
        torch.testing.assert_close(jet.time_derivative[0], numerical, atol=3e-8, rtol=3e-7)
        expected_face = torch.einsum("...g,gf->...f", jacobian,
            model.incidence*amplitude/math_sqrt(model.context.n_particles))
        torch.testing.assert_close(jet.face_derivatives[0], expected_face, atol=2e-10, rtol=2e-10)


def path_coefficients_for_test(model, state):
    from nn_dk.path_numerics import path_coefficients

    return path_coefficients(state, model.context, model.spec)


def math_sqrt(value):
    return value**.5


def _config(**changes):
    base = replace(InteractingStudyConfig(), n_cells=8, width=4, train_paths=1, validation_paths=1,
                   state_fractions=(0., 1.), horizon_samples=1, adam_steps=2, lbfgs_steps=1,
                   lbfgs_max_iter=1, batch_size=4, validation_every=1)
    return RepairConfig(replace(base, **changes))


def _banks(cfg):
    generator = np.random.default_rng(12)
    study = cfg.study

    def make(role):
        paths = (study.train_paths if role == "train" else study.validation_paths)*len(study.cases)
        repeats = len(study.state_times)*study.horizon_samples
        states = generator.uniform(.8, 1.2, (paths, study.n_cells))
        states /= states.sum(-1, keepdims=True)
        return {"states": np.repeat(states, repeats, 0),
                "horizons": generator.uniform(0, study.horizon, paths*repeats),
                "family_ids": np.repeat([f"{role}-{i}" for i in range(paths)], repeats),
                "source_hash": repair.source_hash(), "config_hash": study.config_hash,
                "content_hash": f"literal-{role}", "role": role}

    return make("train"), make("validation")


def test_staged_fit_scales_freeze_mean_and_exact_resume(tmp_path):
    cfg = _config()
    train, val = _banks(cfg)
    bank = PayoffBank.mixed_terminal_integral(cfg.study.spec, cfg.study.horizon)
    full = tmp_path/"full.pt"
    resumed = tmp_path/"resumed.pt"
    result = fit_repaired_payoff(cfg, bank, 0, full, train, val)
    assert result["complete"] and result["stage"] == "covariance"
    partial = fit_repaired_payoff(cfg, bank, 0, resumed, train, val, deadline=time.monotonic()-1)
    assert not partial["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        load_repaired_payoff_model(resumed)
    continuation = fit_repaired_payoff(cfg, bank, 0, resumed, train, val)
    assert continuation["complete"]
    first, second = torch.load(full, weights_only=True), torch.load(resumed, weights_only=True)
    assert first["model_state_hash"] == second["model_state_hash"]
    assert first["best_state_hash"] == second["best_state_hash"]
    assert first["steps_done"] == {"mean_adam": 2, "mean_lbfgs": 1,
                                   "covariance_adam": 2, "covariance_lbfgs": 1}
    assert all(torch.equal(first["model_state"][k], first["best_mean_state"][k])
               for k in first["model_state"] if k.startswith("mean_network."))
    scaling = first["provenance"]["loss_scaling"]
    assert scaling["source_role"] == "train" and np.shape(scaling["mean_scale"]) == (4,)
    assert np.shape(scaling["covariance_scale"]) == (4, 4)
    assert min(scaling["mean_scale"]) > .01
    assert len(set(scaling["mean_scale"])) > 1
    loaded = load_repaired_payoff_model(full)
    assert loaded.config.config_hash == cfg.config_hash
    prediction = loaded.predict_conditional(_states().numpy(), [1/64, 1/32, 1/16])
    assert np.isfinite(prediction.mean).all()
    assert np.linalg.eigvalsh(prediction.scaled_covariance).min() >= -1e-12
    # Covariance fitting cannot backpropagate into the selected mean network.
    loaded.zero_grad(set_to_none=True)
    loaded.covariance_residual(_states(), torch.tensor([.02, .04])).square().mean().backward()
    assert all(p.grad is None for p in loaded.mean_network.parameters())


def test_resume_after_covariance_step_preserves_frozen_mean_and_optimizers(tmp_path, monkeypatch):
    cfg = _config()
    train, val = _banks(cfg)
    bank = PayoffBank.mixed_terminal_integral(cfg.study.spec, cfg.study.horizon)
    full, resumed = tmp_path/"full.pt", tmp_path/"interrupted.pt"
    fit_repaired_payoff(cfg, bank, 0, full, train, val)
    original_save = repair._atomic_save

    class Interrupted(RuntimeError):
        pass

    def interrupt_after_save(payload, path):
        original_save(payload, path)
        if payload["steps_done"]["covariance_adam"] == 1:
            raise Interrupted("simulated interruption after an atomic covariance-stage checkpoint")

    monkeypatch.setattr(repair, "_atomic_save", interrupt_after_save)
    with pytest.raises(Interrupted):
        fit_repaired_payoff(cfg, bank, 0, resumed, train, val)
    monkeypatch.setattr(repair, "_atomic_save", original_save)
    fit_repaired_payoff(cfg, bank, 0, resumed, train, val)
    first, second = torch.load(full, weights_only=True), torch.load(resumed, weights_only=True)
    assert first["best_state_hash"] == second["best_state_hash"]
    assert first["model_state_hash"] == second["model_state_hash"]
    assert torch.equal(first["shuffle_rng_state"], second["shuffle_rng_state"])
    assert torch.equal(first["torch_rng_state"], second["torch_rng_state"])


def test_covariance_residual_has_no_mean_parameter_gradient_even_when_unfrozen():
    model, states = _model(), _states()
    _perturb(model)
    model.covariance_residual(states, torch.tensor([.02, .04])).square().mean().backward()
    assert all(p.grad is None for p in model.mean_network.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.covariance_network.parameters())


def test_checkpoint_rejects_tampering_and_family_overlap(tmp_path):
    cfg = _config()
    train, val = _banks(cfg)
    bank = PayoffBank.mixed_terminal_integral(cfg.study.spec, cfg.study.horizon)
    path = tmp_path/"model.pt"
    fit_repaired_payoff(cfg, bank, 0, path, train, val)
    saved = torch.load(path, weights_only=True)
    saved["model_state"]["mean_network.0.weight"][0, 0] += 1
    torch.save(saved, path)
    with pytest.raises(ValueError, match="checksum"):
        load_repaired_payoff_model(path)
    val["family_ids"] = train["family_ids"].copy()
    with pytest.raises(ValueError, match="disjoint"):
        fit_repaired_payoff(cfg, bank, 0, tmp_path/"overlap.pt", train, val)


def test_configuration_roundtrip_and_invalid_allocations():
    cfg = _config()
    assert RepairConfig.from_dict(cfg.to_dict()).config_hash == cfg.config_hash
    with pytest.raises(ValueError, match="nonempty"):
        RepairConfig(cfg.study, mean_adam_steps=0, mean_lbfgs_steps=0)
    with pytest.raises(ValueError, match="quadrature"):
        RepairConfig(cfg.study, quadrature_nodes=1)
