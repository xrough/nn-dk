"""Tests for autonomous native coefficient learning, independent of legacy models."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from nn_dk.dk_model import DKCoefficientModel, finite_volume_mode_weights
from nn_dk.dk_training import (
    _prepare,
    _validate_groups,
    coefficient_diagnostics,
    fit_a,
    load_coefficient_model,
)
from nn_dk.native import native_coefficients, physical_observable_weights
from nn_dk.native_types import DKTrainingBatch, NativeDKSpec
from nn_dk.types import Context, QueryBank


def batch(prefix, spec=None, context=None, samples=12, negative=False):
    spec = spec or NativeDKSpec(n_cells=8)
    context = context or Context(n_particles=2048)
    rng = np.random.default_rng(71 if prefix == "train" else 92)
    states = rng.dirichlet(np.full(spec.n_cells, 100.), size=samples)
    truth = native_coefficients(states, context, spec)
    incidence = np.roll(np.eye(spec.n_cells), 1, axis=0) - np.eye(spec.n_cells)
    dt = 1 / 128
    noise = rng.normal(size=states.shape) * truth.face_amplitudes
    proposals = states + dt * truth.drift + np.sqrt(dt / context.n_particles) * noise @ incidence.T
    if negative:
        proposals[0, 0] -= 1
        proposals[0, 1] += 1
    return DKTrainingBatch(states, spec, context,
                           tuple(f"{prefix}-family-{i // 3}" for i in range(samples)),
                           tuple(f"{prefix}-sample-{i}" for i in range(samples)),
                           proposals, dt)


def test_autonomous_coefficients_conserve_and_do_not_use_initial_profile_or_n():
    torch.set_num_threads(1)
    spec = NativeDKSpec(n_cells=8)
    model = DKCoefficientModel(spec, width=4)
    states = batch("train", spec).states[:2]
    c1 = Context(n_particles=2048)
    c2 = replace(c1, cos_coeffs=(0.2, 0., 0.), sin_coeffs=(0.1, 0., 0.), n_particles=512)
    c = model.coefficients(states, c1)
    other = model.coefficients(states, c2)
    assert all(p.dtype == torch.float64 for p in model.parameters())
    torch.testing.assert_close(c.drift.sum(-1), torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(c.covariance.sum(-1), torch.zeros(2, 8, dtype=torch.float64), atol=1e-14, rtol=0)
    assert torch.all(c.face_amplitudes > 0)
    assert torch.equal(c.drift, other.drift) and torch.equal(c.covariance, other.covariance)
    assert "step_size" not in model.state_dict()
    np.testing.assert_allclose(finite_volume_mode_weights(spec),
                               physical_observable_weights(spec, QueryBank()), atol=1e-15)
    assert model.coefficients(states[0], c1).drift.shape == (8,)
    assert model.coefficients(states[:, None], c1).covariance.shape == (2, 1, 8, 8)


def test_native_uniform_covariance_is_representable_without_floor():
    spec = NativeDKSpec(n_cells=8)
    model = DKCoefficientModel(spec, width=4)
    states = np.full((2, 8), 1 / 8)
    context = Context()
    truth = native_coefficients(states, context, spec)
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()
        model.output.bias[1] = np.log(np.expm1(truth.face_amplitudes[0, 0] / model.noise_unit))
    np.testing.assert_allclose(model.numpy_coefficients(states, context).covariance,
                               truth.covariance, atol=1e-14)
    with torch.no_grad():
        model.output.bias[1] = -40
    assert model.coefficients(states, context).covariance.abs().max() < 1e-30


def test_tangent_gaussian_nll_matches_distribution_and_includes_negative_response():
    model = DKCoefficientModel(NativeDKSpec(n_cells=8), width=4)
    data = batch("train", negative=True)
    states, proposals = torch.tensor(data.states), torch.tensor(data.proposals)
    c = model.coefficients(states, data.context)
    q = model.tangent
    dt, n = data.proposal_dt, data.context.n_particles
    mean = (states + dt * c.drift) @ q
    covariance = dt / n * (q.T @ c.covariance @ q)
    expected = -torch.distributions.MultivariateNormal(mean, covariance_matrix=covariance).log_prob(proposals @ q)
    actual = model.proposal_residuals(states, proposals, data.context, dt, n)[1]
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-9)
    assert float(proposals.min()) < 0
    assert torch.isfinite(model.nll(states, proposals, data.context, dt, n))


def test_family_splits_and_sample_ids():
    train, val = batch("train"), batch("val")
    _validate_groups([train], [val], train.spec)  # Same context is deliberately allowed.
    overlap = replace(val, family_ids=(train.family_ids[0], *val.family_ids[1:]))
    with pytest.raises(ValueError, match="families overlap"):
        _validate_groups([train], [overlap], train.spec)
    duplicate = replace(val, sample_ids=(train.sample_ids[0], *val.sample_ids[1:]))
    with pytest.raises(ValueError, match="Duplicate"):
        _validate_groups([train], [duplicate], train.spec)


def test_training_selection_retains_exit_proposals_and_only_excludes_inactive_starts():
    source = batch("train", negative=True)
    source.active[-1] = False
    prepared = _prepare([source], source.spec, require_proposals=True)
    assert len(prepared["states"]) == len(source.states) - 1
    assert prepared["proposals"][0].min() < 0
    assert prepared["metadata"]["negative_exit_proposals"] == 1
    assert prepared["metadata"]["inactive_starts"] == 1
    assert prepared["metadata"]["families"] == 4
    no_proposals = replace(source, proposals=None, proposal_dt=None)
    with pytest.raises(ValueError, match="uncensored"):
        _prepare([no_proposals], source.spec, require_proposals=True)


def test_exact_epoch_resume_and_native_checkpoint_identity(tmp_path):
    train, val = batch("train"), batch("val")
    config = dict(epochs=3, batch_size=5, patience=10)
    uninterrupted = tmp_path / "full.pt"
    resumed = tmp_path / "resumed.pt"
    a = fit_a([train], [val], train.spec, width=4, seed=7, config=config, output_path=uninterrupted)
    fit_a([train], [val], train.spec, width=4, seed=7,
          config={**config, "epochs": 1}, output_path=resumed)
    b = fit_a([train], [val], train.spec, width=4, seed=7, config=config, output_path=resumed)
    assert a["complete"] and b["complete"] and a["history"] == b["history"]
    full, restarted = load_coefficient_model(uninterrupted), load_coefficient_model(resumed)
    assert all(torch.equal(v, restarted.state_dict()[k]) for k, v in full.state_dict().items())
    diagnostics = coefficient_diagnostics(restarted, [val])
    assert diagnostics["proposal_samples"] == len(val.states)
    assert np.isfinite(diagnostics["validation_nll"])
    assert np.shape(diagnostics["whitened_covariance"]) == (7, 7)
    with pytest.raises(ValueError, match="different model"):
        fit_a([train], [val], train.spec, width=5, seed=7, config=config, output_path=resumed)
    legacy = tmp_path / "legacy.pt"
    torch.save(dict(provenance={"kind": "a"}), legacy)
    with pytest.raises(ValueError, match="not legacy"):
        load_coefficient_model(legacy)


def test_increment_fit_and_training_only_normalization(tmp_path, monkeypatch):
    train = batch("train")
    val = batch("val", context=Context(kappa=0.75))
    def reject_oracle(*args, **kwargs):
        raise AssertionError("Trajectory-only fitting must not access oracle coefficient labels")
    monkeypatch.setattr("nn_dk.native.native_coefficients", reject_oracle)
    fit = fit_a([train], [val], train.spec, mode="increments", width=4, seed=9,
                config=dict(epochs=2, batch_size=8), output_path=tmp_path / "a.pt")
    model = load_coefficient_model(fit["path"])
    np.testing.assert_array_equal(model.normalization.center.numpy(), [0, 0, 1])
    assert fit["epochs"] == 2 and fit["complete"]
    result = model.sample(train.context, [0, 1 / 64], 3, 5, step_size=1 / 128)
    assert result.states.shape == (3, 2, 8)
    assert result.diagnostics["all_paths_retained"]


def test_spatial_fourier_features_match_native_interaction_and_see_far_field():
    spec = NativeDKSpec(n_cells=32)
    context = Context(kappa=0.75, a2=0.1)
    model = DKCoefficientModel(spec, width=4)
    states = np.random.default_rng(12).dirichlet(np.full(32, 50.), size=2)
    channels, _ = model._inputs(states, context)
    centers = (np.arange(32) + 0.5) * spec.h
    faces = (np.arange(32) + 1) * spec.h
    for k in (1, 2):
        for channel, function in enumerate((np.cos, np.sin)):
            expected = states @ (np.sinc(k * spec.h / (2 * np.pi))
                                 * function(k * (faces[:, None] - centers[None, :]))).T
            np.testing.assert_allclose(channels[:, 1 + 2 * (k - 1) + channel].detach(),
                                       expected, atol=1e-15)
    velocity = -context.kappa * channels[:, 2] + 2 * context.a2 * channels[:, 4]
    face_density = (states + np.roll(states, -1, axis=-1)) / (2 * spec.h)
    flux = (face_density * velocity.detach().numpy()
            - context.diffusion / spec.h**2 * (np.roll(states, -1, axis=-1) - states))
    np.testing.assert_allclose(flux, native_coefficients(states, context, spec).flux, atol=1e-15)
    # Identical neighborhoods at cell 0 can have different nonlocal forces.
    changed = states[:1].copy()
    changed[0, 14] -= 0.001
    changed[0, 20] += 0.001
    different, _ = model._inputs(changed, context)
    assert torch.equal(channels[:1, 0, :7], different[:, 0, :7])
    assert not torch.allclose(channels[:1, 1:5, 0], different[:, 1:5, 0])


def test_native_coefficient_network_is_periodic_shift_equivariant():
    torch.manual_seed(5)
    spec = NativeDKSpec(n_cells=16)
    model = DKCoefficientModel(spec, width=4)
    context = Context(kappa=0.75, a2=0.1)
    states = np.random.default_rng(5).dirichlet(np.full(16, 20.), size=3)
    base = model.coefficients(states, context)
    shift = 5
    translated = model.coefficients(np.roll(states, shift, axis=-1), context)
    torch.testing.assert_close(translated.drift, torch.roll(base.drift, shift, dims=-1), atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(translated.face_amplitudes,
                               torch.roll(base.face_amplitudes, shift, dims=-1), atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(translated.covariance,
                               torch.roll(base.covariance, (shift, shift), dims=(-2, -1)), atol=1e-14, rtol=1e-12)
