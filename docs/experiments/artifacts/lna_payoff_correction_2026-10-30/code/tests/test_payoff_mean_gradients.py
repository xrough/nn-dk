"""Independent stopped-payoff sensitivities and immutable gradient evidence."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit_payoff_mean_gradients as audit  # noqa: E402
from analysis.compiled_native import compile_native, compiler_available  # noqa: E402
from nn_dk.interacting_config import InteractingStudyConfig  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.payoff_boundary import BoundaryPayoffModel  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.types import Context  # noqa: E402


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    if not compiler_available():
        pytest.skip("Optional C++ compiler unavailable")
    return compile_native(tmp_path_factory.mktemp("mean-gradient-compiler"))


@pytest.mark.parametrize("crossing", [False, True])
def test_killed_payoff_jvp_includes_stopping_exposure_derivative(backend, crossing):
    spec, context, dt = NativeDKSpec(n_cells=8), Context(n_particles=64), .01
    bank = PayoffBank.mixed_terminal_integral(spec, .08)
    initial = np.full((2, 8), .125)
    noise = np.random.default_rng(18).normal(0, .04, (2, 8, 8))
    if crossing:
        noise[0, 2, 2] = -4.
    direction = np.cos((np.arange(8)+.5)*spec.h)/8
    direction -= direction.mean()
    packet = backend(initial, noise, context, spec, dt, initial_tangent=direction)
    actual = audit.payoff_jvp(bank, packet, dt, (.04, .08))
    finite = []
    epsilon = 1e-6
    for sign in (1, -1):
        other = backend(initial+sign*epsilon*direction, noise, context, spec, dt)
        finite.append(bank.evaluate_paths(other["states"], np.arange(9)*dt,
                                          other["stopped_at"])[:, [4, 8]])
    expected = (finite[0]-finite[1])/(2*epsilon)
    np.testing.assert_allclose(actual, expected, atol=4e-10, rtol=3e-6)
    if crossing:
        assert np.isfinite(packet["stopped_at"][0])
        assert abs(packet["stopped_at_tangent"][0]) > 1e-6
        omitted = audit.payoff_jvp(bank, packet | {
            "stopped_at_tangent": np.zeros(2)}, dt, (.04, .08))
        assert abs(omitted[0, :, 2:]-actual[0, :, 2:]).max() > 1e-5


def test_heat_deterministic_gradient_and_time_zero_source(backend):
    spec, context, dt = NativeDKSpec(n_cells=8), Context(n_particles=2048), 1/128
    bank = PayoffBank.mixed_terminal_integral(spec, 1/32)
    initial = np.full((1, 8), .125)
    direction = np.cos(3*(np.arange(8)+.5)*spec.h)/8
    packet = backend(initial, np.zeros((1, 4, 8)), context, spec, dt, initial_tangent=direction)
    identity = np.eye(8)
    transition = identity+dt/spec.h**2*(np.roll(identity, 1, axis=0)
                                      +np.roll(identity, -1, axis=0)-2*identity)
    expected = np.stack([np.linalg.matrix_power(transition, k)@direction for k in range(5)])
    np.testing.assert_allclose(packet["tangent"][0], expected, atol=3e-16, rtol=1e-13)
    model = BoundaryPayoffModel(spec, context, 1/32, bank, width=4, quadrature_nodes=2)
    controls = audit._controls(model, initial[0])
    assert controls["zero_time_face_maximum_error"] < 1e-13
    assert controls["absorbed_face_source_exact_zero"]


def test_cross_gram_removes_finite_sample_quadratic_noise_and_keeps_pairing():
    gradient = np.array([2., 4., 6., 8.]).reshape(4, 1, 1, 1)
    mean, plugin, cross = audit.source_estimates(gradient)
    np.testing.assert_array_equal(mean, [[[5.]]])
    np.testing.assert_array_equal(plugin, [[[25.]]])
    np.testing.assert_array_equal(cross, [[[21.]]])
    # An independent-half cross product may be negative: never project it PSD.
    _, _, indefinite = audit.source_estimates(np.array([-2., -1., 1., 2.]).reshape(4, 1, 1, 1))
    assert indefinite[0, 0, 0] < 0
    repeated = np.broadcast_to(gradient, (4, 3, 2, 2)).copy()
    boot, sources = audit.bootstrap_sources(repeated, 8, 145)
    np.testing.assert_array_equal(boot[:, 0], boot[:, 2])
    np.testing.assert_array_equal(sources[:, 0], sources[:, 2])
    np.testing.assert_array_equal(boot, audit.bootstrap_sources(repeated, 8, 145)[0])


def test_anchor_selection_uses_only_distinct_active_current_states():
    states = np.full((6, 8), .125)
    states[0, :2], states[1, :2], states[2, :2] = (.01, .24), (.03, .22), (.08, .17)
    states[3:] = states[:3]
    bank = {"states": states, "case_ids": np.array(["a"]*6),
            "family_ids": np.array(["f0", "f1", "f2"]*2), "snapshot_times": np.zeros(6)}
    chosen = audit.select_anchors(bank, {"physical_active": np.ones(6, bool)}, NativeDKSpec(n_cells=8), 2)
    assert chosen["indices"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="Insufficient"):
        audit.select_anchors(bank, {"physical_active": np.zeros(6, bool)}, NativeDKSpec(n_cells=8), 1)


@pytest.fixture
def toy_study(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    study_root = tmp_path/"historical"
    (study_root/"payoffs").mkdir(parents=True)
    (study_root/"payoffs/fit_lock.json").write_text("frozen")
    cfg = replace(InteractingStudyConfig(), n_cells=8, n_particles=64, width=4, horizon=1/32,
                  query_times=(1/128, 1/64, 1/32))
    bank = PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon)
    states = np.full((4, 8), .125)
    states[:, :2] += np.array([[-.01, .01], [-.02, .02], [-.03, .03], [-.04, .04]])
    state_path = study_root/"validation.h5"
    state_path.write_text("frozen state-only validation bank")
    pool = {"states": states, "case_ids": np.array(["uniform"]*2+["nonuniform"]*2),
            "family_ids": np.array(["v0", "v1", "v2", "v3"]), "snapshot_times": np.zeros(4),
            "path": str(state_path)}
    model = BoundaryPayoffModel(cfg.spec, cfg.physics, cfg.horizon, bank, width=4, quadrature_nodes=2).eval()
    models = []
    for seed in (0, 1, 2):
        checkpoint, metadata = study_root/f"seed{seed}.pt", study_root/f"seed{seed}.json"
        checkpoint.write_bytes(bytes([seed]))
        metadata.write_text("fixed checkpoint metadata")
        models.append({"case": "toy", "seed": seed, "checkpoint_path": str(checkpoint),
                       "checkpoint_sha256": audit.sha(checkpoint), "metadata_path": str(metadata),
                       "metadata_sha256": audit.sha(metadata)})
    runner = SimpleNamespace(configuration=lambda *args: cfg, payoff_bank=lambda *args: bank,
        load_state_bank=lambda *args: pool, load_declared_model=lambda *args: model)
    monkeypatch.setattr(audit, "_inventory", lambda *args: (runner, None, ("toy",), models))
    import analysis.audit_payoff_regimes_validation as validation
    monkeypatch.setattr(validation, "validation_metadata", lambda _: {"physical_active": np.ones(4, bool)})
    plan = audit.GradientPlan(cases=("toy",), anchors_per_profile=1, paths=4, chunk_paths=2,
                             step=1/128, refinement_paths=4, n_bootstrap=4)
    return study_root, plan


def test_prepare_resume_complete_and_reject_mutated_numeric_evidence(tmp_path, toy_study):
    if not compiler_available():
        pytest.skip("Optional C++ compiler unavailable")
    study, plan = toy_study
    root = tmp_path/"diagnostic"
    prepared = audit.prepare(root, study, plan)
    assert audit.prepare(root, study, plan) == prepared
    with pytest.raises(ValueError, match="identity"):
        audit.prepare(root, study, replace(plan, seed=plan.seed+1))
    partial = audit.run(root, max_chunks=1)
    assert not partial["complete"]
    first = Path(partial["chunks"][0]["path"])
    before = first.read_bytes()
    with pytest.raises(ValueError, match="must complete"):
        audit.load_complete(root)
    complete = audit.run(root)
    assert complete["complete"] and first.read_bytes() == before
    result = audit.load_complete(root)
    assert len(result["records"]) == 6
    assert {r["seed"] for r in result["records"]} == {0, 1, 2}
    assert result["records"][0]["paired_refinement"]["families"] == 4
    assert result["records"][0]["boundary_stratum"]
    assert audit.run(root) == complete
    evidence = Path(result["numeric_evidence"]["path"])
    evidence.write_bytes(evidence.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="numeric evidence"):
        audit.load_complete(root)


def test_changed_anchor_or_checkpoint_and_chunk_inventory_rejected(tmp_path, toy_study):
    study, plan = toy_study
    root = tmp_path/"diagnostic"
    prepared = audit.prepare(root, study, plan)
    artifact = Path(prepared["anchors"]["toy"]["path"])
    original = artifact.read_bytes()
    artifact.write_bytes(original+b"modified")
    with pytest.raises(ValueError, match="anchors"):
        audit.run(root)
    artifact.write_bytes(original)
    checkpoint = Path(prepared["identity"]["models"][0]["checkpoint_path"])
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        audit.run(root)


@pytest.mark.parametrize("change", [{"paths": 3}, {"refinement_paths": 1}, {"query_fractions": (1., .5)},
                                  {"step": 0}, {"n_bootstrap": 1}])
def test_invalid_allocation_rejected(change):
    with pytest.raises(ValueError):
        audit.GradientPlan(**change)


def test_complete_index_cannot_hide_missing_chunks(tmp_path, toy_study):
    study, plan = toy_study
    root = tmp_path/"diagnostic"
    prepared = audit.prepare(root, study, plan)
    (root/"index.json").write_text(json.dumps({"complete": True,
        "identity_hash": prepared["identity_hash"], "expected_chunks": 0, "chunks": []}))
    with pytest.raises(ValueError, match="inventory"):
        audit.load_complete(root)


def test_paired_refinement_preserves_identical_family_noise_cancellation():
    values = np.random.default_rng(5).normal(size=(8, 3, 4, 8))
    summary, arrays = audit.refinement_comparison(values, values, 8, 901)
    assert summary["paired_whole_family_resampling"]
    for value in arrays.values():
        np.testing.assert_array_equal(value, 0.)


def test_committed_chunk_corruption_is_rejected_before_resuming(tmp_path, toy_study):
    if not compiler_available():
        pytest.skip("Optional C++ compiler unavailable")
    study, plan = toy_study
    root = tmp_path/"diagnostic"
    audit.prepare(root, study, plan)
    partial = audit.run(root, max_chunks=1)
    path = Path(partial["chunks"][0]["path"])
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="chunk changed"):
        audit.run(root)
