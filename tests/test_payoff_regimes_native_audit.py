"""Independent native evidence replay and stopped-integral controls."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "payoff_regimes_native_audit", Path(__file__).parents[1]/"analysis/audit_payoff_regimes_native.py")
audit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = audit
_spec.loader.exec_module(audit)

from analysis.compiled_native import compiler_available  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402


def test_fixed_cell_weights_agree_with_independent_cell_quadrature():
    spec = NativeDKSpec(n_cells=16)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    x = (np.arange(16)+.5)[:, None]*spec.h+spec.h/2*nodes
    expected = np.stack((np.sin(3*x)@weights/2, np.cos(3*x)@weights/2,
                         np.sin(x)@weights/2, np.cos(x)@weights/2))
    np.testing.assert_allclose(audit.physical_weights(spec), expected, atol=2e-15, rtol=0)


def test_independent_integral_kills_at_fractional_crossing_and_terminal_freezes():
    spec = NativeDKSpec(n_cells=16)
    initial = np.ones(16)/16
    boundary = initial.copy()
    boundary[0] -= .02
    boundary[1] += .02
    states = np.asarray([[initial, boundary, boundary]])
    times, stops = np.array([0., .5, 1.]), np.array([.125])
    readout = audit.reduce_native_polygon(states, times, stops, spec, 1., (.5, 1.))
    weights = audit.physical_weights(spec)
    expected_integral = .125*.5*(weights[2:]@initial+weights[2:]@boundary)
    np.testing.assert_allclose(readout["outcomes"][0, :, 2:], np.tile(expected_integral, (2, 1)), atol=1e-15)
    np.testing.assert_allclose(readout["outcomes"][0, :, :2], np.tile(weights[:2]@boundary, (2, 1)), atol=1e-15)
    old = PayoffBank.mixed_terminal_integral(spec, 1.).evaluate_paths(states, times, stops)
    np.testing.assert_allclose(readout["outcomes"], old[:, 1:], atol=1e-15)


def test_absorbed_initial_zero_and_path_failure_handling():
    spec = NativeDKSpec(n_cells=16)
    y = np.zeros((2, 3, 4))
    audit._check_path_values(y, np.array([0., np.inf]), np.array([0., .03]), 2, 1., spec.mass_stop)
    with pytest.raises(ValueError, match="every native estimate"):
        audit._check_path_values(y, np.array([0., np.nan]), np.array([0., .03]), 2, 1., spec.mass_stop)
    with pytest.raises(ValueError, match="first threshold"):
        audit._check_path_values(y, np.array([.5, np.inf]), np.array([.01, .03]), 2, 1., spec.mass_stop)


def test_stopping_replay_requires_identical_absorption_membership():
    assert audit._stop_error(np.array([np.inf, .2]), np.array([np.inf, .2])) == 0
    with pytest.raises(AssertionError, match="membership"):
        audit._stop_error(np.array([np.inf, .2]), np.array([np.inf, np.inf]))


def test_chunk_digest_identifies_dtype_shape_and_content():
    value = np.arange(8, dtype=np.int64)
    digest = audit._chunk_digest({"a": value})
    assert digest != audit._chunk_digest({"a": value.astype(np.float64)})
    assert digest != audit._chunk_digest({"a": value.reshape(2, 4)})
    assert digest != audit._chunk_digest({"b": value})
    assert audit._typed_array_sha(value) != audit._array_sha(value)


@pytest.mark.skipif(not compiler_available(), reason="Native candidate replay requires compiler")
def test_independent_ensemble_candidate_prefix_preserves_chunk_initial_rng_order(tmp_path):
    runner = audit.runner
    plan = runner.StudyPlan(horizons=(1/64,), kappas=(.75,), optimizer_seeds=(0,),
        n_cells=16, train_paths=2, validation_paths=1, state_step=1/256,
        reference_step=1/512, paths_per_group=4, refinement_paths=2, chunk_paths=2,
        adam_steps=1, lbfgs_steps=0, validation_every=1, width=2, batch_size=16, quadrature_nodes=4)
    case, key = runner.case_names(plan)[0], "nonuniform__ensemble__0"
    cfg, bank = runner.reference_configuration(case, plan), runner.payoff_bank(case, plan)
    probabilities = runner.profile_probabilities(cfg, "nonuniform")[0]
    seed = int(audit.stable_hash([audit.evaluator.SEED, case, key, "independent-candidate"])[:12], 16)
    backend = runner.compile_native(tmp_path/"compiled")
    y, diagnostic = audit.evaluator._native_mc(runner, cfg, bank, "nonuniform", None,
        probabilities, cfg.paths_per_group, seed, backend)
    arrays = {"outcomes": y, "stopped_at": diagnostic["stopped_at"],
              "path_minimum_mass": diagnostic["path_minimum_mass"],
              "family_ids": np.array([audit.stable_hash([audit.evaluator.SEED, case, key,
                                                         "independent-candidate", i]) for i in range(4)], dtype="S64")}
    packet = {"identity": {"regime": case}, "rng": {
        "seed": seed, "stream": [audit.evaluator.KIND, "candidate-native-MC"],
        "chunk": 32, "forcing": "independent-PCG64-normal"}}
    record = audit._candidate_prefix_replay(cfg, key, packet, arrays, probabilities, None)
    assert record["maximum_outcome_error"] < 3e-12
    assert record["conditional_initial_fixed"] is False


@pytest.mark.skipif(not compiler_available(), reason="End-to-end native evidence requires compiler")
def test_complete_small_reference_candidate_and_deployment_evidence_audit(tmp_path, monkeypatch):
    runner, evaluator, deployment = audit.runner, audit.evaluator, audit.deployment
    plan = runner.StudyPlan(horizons=(1/64,), kappas=(.75,), optimizer_seeds=(0,),
        n_cells=16, train_paths=2, validation_paths=1, state_step=1/256,
        reference_step=1/512, paths_per_group=4, refinement_paths=2, chunk_paths=2,
        adam_steps=1, lbfgs_steps=0, validation_every=1, width=2, batch_size=16, quadrature_nodes=4)
    root = tmp_path/"smoke"
    runner.prepare(root, plan)
    evaluator.prepare(root)
    deployment.prepare(root)
    # The unit fixture supplies the prerequisite proof. Production audits require
    # the separately executed full family/address audit before any outcome draw.
    runner._json(root/"audit/deployment_pretest_independence.json", {
        "complete": True, "source_hash": audit.source_hash(),
        "deployment_plan_sha256": runner._sha(root/"deployment/preregistration.json"),
        "planned_families_disjoint": True, "address_streams_disjoint": True})
    runner.fit_seed(root, 0)
    runner.lock(root)
    runner.generate_reference(root)
    evaluator.evaluate(root)
    deployment.evaluate(root)
    monkeypatch.setattr(audit.pretest, "historical_identifier_inventory", lambda *args: (set(), []))
    result = audit.audit(root)
    assert result["complete"] and result["unique_reference_families"] == 24
    assert result["unique_native_candidate_families"] == 24
    assert result["independent_NN_initial_pools"]["independent_families"] == 8
    assert result["independent_NN_initial_pools"]["all_histograms_reproduced"]
    assert result["maximum_native_reference_replay_error"] < 3e-12
    assert result["maximum_native_candidate_replay_error"] < 3e-12
    with pytest.raises(ValueError, match="overwrite"):
        audit.audit(root)

