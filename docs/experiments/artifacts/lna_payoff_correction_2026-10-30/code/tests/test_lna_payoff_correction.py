"""Finite supervised fixtures: boundary limits, geometry and strict replay."""
from __future__ import annotations

import copy
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import fit_lna_payoff_correction as engine  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402


def configuration(**changes):
    return engine.LNAPayoffCorrectionConfig(horizon=1/16, kappa=.75, n_cells=8,
        n_particles=128, width=4, depth=2, mean_adam_steps=2, covariance_adam_steps=2,
        mean_lbfgs_steps=1, covariance_lbfgs_steps=1, lbfgs_max_iter=2,
        batch_size=3, validation_every=1, **changes)


def inputs(config, count=5, times=None):
    rng = np.random.default_rng(91)
    states = np.full((count, config.n_cells), 1/config.n_cells)+rng.normal(0, .002, (count, config.n_cells))
    states /= states.sum(-1, keepdims=True)
    bank = PayoffBank.mixed_terminal_integral(config.spec, config.horizon)
    times = np.array(config.query_times if times is None else times)
    x = times/config.horizon
    start = states @ bank.terminal_weights.T
    mean = np.repeat(start[:, None], len(times), axis=1)+x[None, :, None]*.01
    raw = rng.normal(size=(4, 4))
    correlation = raw@raw.T
    d = x[:, None]**np.where(bank.integral_rows, 1.5, .5)[None]
    covariance = np.repeat((d[:, :, None]*correlation*d[:, None, :])[None], count, 0)
    return bank, states, times, mean, covariance


def supervised(config, role):
    bank, states, times, mean, covariance = inputs(config)
    label_mean = mean+.002*(times/config.horizon)[None, :, None]**2
    label_covariance = 1.1*covariance
    provenance = {"stream_namespace": "software-only-"+role, "role": role,
        "parent_pool_sha256": "a"*64, "label_manifest_sha256": "b"*64,
        "baseline_manifest_sha256": "c"*64, "normalizer_policy": "fixed-bank-scales",
        "training_mixture": {"native": 1.}, "all_stopped_paths_retained": True}
    value = engine.make_supervised_bank(states, times, mean, covariance, label_mean, label_covariance,
        [role+str(i) for i in range(len(states))], config=config, bank=bank, role=role, provenance=provenance)
    return bank, value


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_config_roundtrip_and_distinct_supervised_target():
    config = configuration()
    assert engine.LNAPayoffCorrectionConfig.from_dict(config.to_dict()) == config
    assert config.config_hash != replace(config, label_step=2**-11).config_hash
    broken = config.to_dict()
    broken.pop("label_step")
    with pytest.raises(ValueError, match="Complete"):
        engine.LNAPayoffCorrectionConfig.from_dict(broken)
    with pytest.raises(ValueError, match="grid"):
        replace(config, label_step=.003)


def test_zero_heads_are_exact_gated_baseline_and_initial_limits():
    config = configuration()
    bank, states, times, mean, covariance = inputs(config, times=(0., .015625, .0625))
    model = engine._model(config, bank, 0)
    states[1, 0] = config.spec.mass_stop
    states[1, 1:] *= (1-states[1, 0])/states[1, 1:].sum()
    trained = model.predict_conditional(states, times, mean, covariance)
    control = model.predict_conditional(states, times, mean, covariance, kind="gated_zero")
    for key in trained:
        torch.testing.assert_close(trained[key], control[key], rtol=0, atol=0)
    expected = torch.tensor(states) @ model.terminal_weights.T
    torch.testing.assert_close(trained["mean"][:, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(trained["mean"][1], expected[1].expand(3, -1), rtol=0, atol=0)
    assert torch.count_nonzero(trained["scaled_covariance"][:, 0]) == 0
    assert torch.count_nonzero(trained["scaled_covariance"][1]) == 0
    x = torch.tensor(times[1:]/config.horizon)
    gate = torch.erf((torch.tensor(states[0])-config.spec.mass_stop)/(
        .05/config.n_cells*torch.sqrt(x[:, None]))).prod(-1)
    torch.testing.assert_close(trained["scaled_covariance"][0, 1:],
        gate[:, None, None]*torch.tensor(covariance[0, 1:]), rtol=0, atol=0)


def test_congruence_matches_normalized_D_geometry_without_floor():
    config = configuration()
    bank, states, times, mean, covariance = inputs(config)
    model = engine._model(config, bank, 1)
    matrix = torch.arange(16, dtype=torch.float64).reshape(4, 4)/25-0.2
    with torch.no_grad():
        model.covariance_network[-1].bias.copy_(matrix.flatten())
    prediction = model.predict_conditional(states, times, mean, covariance)["scaled_covariance"]
    scale = torch.tensor(bank.scales)
    for index, t in enumerate(times):
        x = t/config.horizon
        d = torch.diag(torch.tensor(x**np.where(bank.integral_rows, 1.5, .5)))
        inverse = torch.diag(1/torch.diagonal(d))
        h = torch.eye(4)+x*matrix
        normalized = torch.tensor(covariance[:, index])/scale[:, None]/scale[None, :]
        expected = d@h@inverse@normalized@inverse@h.T@d
        expected = expected*scale[:, None]*scale[None, :]
        gate = torch.erf((torch.tensor(states)-config.spec.mass_stop)/(.05/config.n_cells*math_sqrt(x))).prod(-1)
        torch.testing.assert_close(prediction[:, index], gate[:, None, None]*expected, rtol=1e-13, atol=1e-15)
    assert torch.linalg.eigvalsh(prediction).min() > -1e-12


def math_sqrt(x):
    return float(np.sqrt(x))


def test_time_ordered_blocks_retain_leading_coefficients():
    config = configuration()
    tiny = config.horizon*np.array((1e-8, 2e-8, 4e-8))
    bank, states, times, mean, covariance = inputs(config, times=tiny)
    model = engine._model(config, bank, 0)
    with torch.no_grad():
        model.covariance_network[-1].bias.fill_(.5)
    result = model.predict_conditional(states, times, mean, covariance)["scaled_covariance"].detach().numpy()
    order = 1+bank.integral_rows[:, None].astype(int)+bank.integral_rows[None, :].astype(int)
    normalized = result/(times/config.horizon)[None, :, None, None]**order
    baseline = covariance/(times/config.horizon)[None, :, None, None]**order
    np.testing.assert_allclose(normalized, baseline, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("bad", ["nonfinite", "negative", "asymmetric"])
def test_invalid_baseline_fails_without_repair(bad):
    config = configuration()
    bank, states, times, mean, covariance = inputs(config)
    if bad == "nonfinite":
        mean[0, 0, 0] = np.nan
    elif bad == "negative":
        covariance[0, 0] = -np.eye(4)
    else:
        covariance[0, 0, 0, 1] += .1
    with pytest.raises(ValueError, match="baseline"):
        engine._model(config, bank, 0).predict_conditional(states, times, mean, covariance)


def test_signed_roundoff_is_retained_not_projected():
    config = configuration()
    bank, states, times, mean, covariance = inputs(config)
    covariance[:] = np.diag([1., 1., 1., -1e-12])
    result = engine._model(config, bank, 0).predict_conditional(states, times, mean, covariance)
    assert (result["scaled_covariance"][..., -1, -1] < 0).all()


def test_fixed_physical_loss_units_and_natural_validation_selection():
    config = configuration()
    bank, value = supervised(config, "validation")
    arrays, _ = engine._bank(value, config, bank, "validation")
    model = engine._model(config, bank, 0)
    expected_mean = ((torch.tensor(value["baseline_mean"]-value["label_mean"])*
                      np.sqrt(config.n_particles)/model.scales)**2).mean()
    expected_covariance = ((torch.tensor(value["baseline_scaled_covariance"]-value["label_scaled_covariance"])/
                           (model.scales[:, None]*model.scales[None, :]))**2).mean()
    torch.testing.assert_close(engine._loss(model, arrays, "mean"), expected_mean)
    torch.testing.assert_close(engine._loss(model, arrays, "covariance"), expected_covariance)
    arrays["natural"][0] = False
    arrays["label_mean"][0] += 100
    selected = engine.natural_validation(model, arrays, "mean")
    assert selected["anchors"] == 4
    assert selected["natural_mse"] == pytest.approx(float(expected_mean))


@pytest.mark.parametrize("cut", [0, 1, 2, 3, 4, 5])
def test_complete_resume_is_bitwise_across_both_optimizers_and_stages(tmp_path, cut):
    config = configuration()
    bank, train = supervised(config, "train")
    _, validation = supervised(config, "validation")
    full, resumed = tmp_path/"full.pt", tmp_path/"resume.pt"
    caller_rng = torch.get_rng_state().clone()
    complete = engine.fit_lna_correction(config, bank, 0, full, train, validation)
    assert torch.equal(caller_rng, torch.get_rng_state())
    part = engine.fit_lna_correction(config, bank, 0, resumed, train, validation, max_updates=cut)
    assert sum(part["steps_done"].values()) == cut
    done = engine.fit_lna_correction(config, bank, 0, resumed, train, validation)
    assert complete["complete"] and done["complete"]
    first, second = engine._read(full), engine._read(resumed)
    for key in ("model_state", "best_state", "best_mean_state", "optimizer_states", "sampler_rng_state",
                "steps_done", "best_validation", "best_step"):
        assert engine._digest(first[key]) == engine._digest(second[key])
    assert engine._digest(engine._mean_weights(second["best_state"])) == engine._digest(
        engine._mean_weights(second["best_mean_state"]))
    model = engine.load_lna_correction(resumed)
    assert model.complete and model.method_kind == engine.KIND
    assert model.provenance["data"]["train"]["content_sha256"] == train["content_sha256"]
    assert not any(parameter.requires_grad for parameter in model.mean_network.parameters())


def test_resume_rejects_changed_labels_baseline_config_environment_and_checkpoint(tmp_path, monkeypatch):
    config = configuration()
    bank, train = supervised(config, "train")
    _, validation = supervised(config, "validation")
    path = tmp_path/"fit.pt"
    engine.fit_lna_correction(config, bank, 1, path, train, validation, max_updates=1)
    for key in ("baseline_mean", "label_mean", "baseline_scaled_covariance", "label_scaled_covariance"):
        changed = copy.deepcopy(train)
        changed[key] += .001 if "covariance" not in key else .001*np.eye(4)
        changed["content_sha256"] = engine._digest({k:v for k,v in changed.items() if k != "content_sha256"})
        with pytest.raises(ValueError, match="resume"):
            engine.fit_lna_correction(config, bank, 1, path, changed, validation)
    altered = replace(config, learning_rate=.002)
    with pytest.raises(ValueError, match="schema"):
        engine.fit_lna_correction(altered, bank, 1, path, train, validation)
    old = path.read_bytes()
    saved = torch.load(path, weights_only=True)
    saved["model_state"]["mean_network.0.weight"][0, 0] += 1
    torch.save(saved, path)
    with pytest.raises(ValueError, match="checksum"):
        engine.load_lna_correction(path)
    path.write_bytes(old)
    monkeypatch.setattr(engine, "_environment", lambda: {"changed": True})
    with pytest.raises(ValueError, match="environment"):
        engine.load_lna_correction(path)


def test_parent_leakage_incomplete_load_and_no_overwrite(tmp_path):
    config = configuration()
    bank, train = supervised(config, "train")
    _, validation = supervised(config, "validation")
    leaking = copy.deepcopy(validation)
    leaking["parent_family_ids"][0] = train["parent_family_ids"][0]
    leaking["content_sha256"] = engine._digest({k:v for k,v in leaking.items() if k != "content_sha256"})
    with pytest.raises(ValueError, match="disjoint"):
        engine.fit_lna_correction(config, bank, 0, tmp_path/"leak.pt", train, leaking)
    path = tmp_path/"partial.pt"
    result = engine.fit_lna_correction(config, bank, 0, path, train, validation, deadline=time.monotonic()-1)
    assert not result["complete"] and sum(result["steps_done"].values()) == 0
    with pytest.raises(ValueError, match="incomplete"):
        engine.load_lna_correction(path)
    with pytest.raises(FileExistsError):
        engine.fit_lna_correction(config, bank, 0, path, train, validation, resume=False)


def test_rehashed_false_completion_and_changed_frozen_mean_are_rejected(tmp_path):
    config = configuration()
    bank, train = supervised(config, "train")
    _, validation = supervised(config, "validation")
    path = tmp_path/"fit.pt"
    engine.fit_lna_correction(config, bank, 0, path, train, validation, max_updates=4)
    saved = engine._read(path)
    changed = copy.deepcopy(saved)
    changed["complete"] = True
    with pytest.raises(ValueError, match="completion"):
        engine._save(path, changed)
    changed = copy.deepcopy(saved)
    changed["model_state"]["mean_network.0.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="frozen mean"):
        engine._save(path, changed)


@pytest.mark.parametrize("corruption", ["optimizer", "history", "scales", "RNG"])
def test_rehashed_semantic_corruption_is_rejected(tmp_path, corruption):
    config = configuration()
    bank, train = supervised(config, "train")
    _, validation = supervised(config, "validation")
    path = tmp_path/"fit.pt"
    engine.fit_lna_correction(config, bank, 0, path, train, validation, max_updates=1)
    changed = copy.deepcopy(engine._read(path))
    if corruption == "optimizer":
        changed["optimizer_states"]["mean_adam"]["state"] = {}
    elif corruption == "history":
        changed["history"][0]["optimizer"] = "lbfgs"
    elif corruption == "scales":
        changed["identity"]["loss_scaling"]["mean_scale"][0] *= 2
    else:
        changed["sampler_draws"] += 1
    with pytest.raises(ValueError):
        engine._save(path, changed)


def test_supervised_provenance_all_queries_stopped_labels_and_source_guards(tmp_path, monkeypatch):
    config = configuration()
    bank, states, times, bm, bc = inputs(config)
    _, train = supervised(config, "train")
    provenance = train["provenance"]
    with pytest.raises(ValueError, match="three"):
        engine.make_supervised_bank(states, times[:2], bm[:, :2], bc[:, :2], bm[:, :2], bc[:, :2],
            train["family_ids"], role="train", config=config, bank=bank, provenance=provenance)
    with pytest.raises(ValueError, match="retention"):
        engine.make_supervised_bank(states, times, bm, bc, bm, bc, train["family_ids"], role="train",
            config=config, bank=bank, provenance={**provenance, "all_stopped_paths_retained": False})
    states[0, 0] = config.spec.mass_stop
    states[0, 1:] *= (1-states[0, 0])/states[0, 1:].sum()
    with pytest.raises(ValueError, match="absorbed"):
        engine.make_supervised_bank(states, times, bm, bc, bm, bc, train["family_ids"], role="train",
            config=config, bank=bank, provenance=provenance)
    _, validation = supervised(config, "validation")
    path = tmp_path/"fit.pt"
    engine.fit_lna_correction(config, bank, 2, path, train, validation, max_updates=0)
    original = engine._sources()
    monkeypatch.setattr(engine, "_sources", lambda: {**original, "changed_helper": True})
    with pytest.raises(ValueError, match="source"):
        engine.load_lna_correction(path)


def test_seed_identity_learning_and_replay_do_not_silently_replace_completed_fit(tmp_path):
    config = configuration()
    bank, train = supervised(config, "train")
    _, validation = supervised(config, "validation")
    records = []
    for seed in (0, 1, 2):
        path = tmp_path/f"seed{seed}.pt"
        metadata = engine.fit_lna_correction(config, bank, seed, path, train, validation)
        records.append(metadata)
        for stage in engine.STAGES:
            assert metadata["best_validation"][stage]["natural_mse"] < metadata["initial_validation"][stage]["natural_mse"]
        original = path.read_bytes()
        assert engine.fit_lna_correction(config, bank, seed, path, train, validation) == metadata
        assert path.read_bytes() == original
    assert len({row["identity"]["initial_state_sha256"] for row in records}) == 3
    assert len({row["method_id"] for row in records}) == 3
