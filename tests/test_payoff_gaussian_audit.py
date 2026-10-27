"""The coarse Gaussian cost is joined to its literal three-query algorithm."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.native_gaussian_baseline import linear_gaussian  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "audit_payoff_gaussian", Path(__file__).parents[1]/"analysis"/"audit_payoff_gaussian.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_literal_three_query_grid_differs_from_forced_common_knots_and_is_fine_audited():
    cfg = audit.ReferenceConfig(n_cells=16)
    context = cfg.context("nonuniform")
    centers = (np.arange(16)+.5)*cfg.spec.h
    initial = (1+.15*np.cos(centers)+.1*np.sin(3*centers))/16
    bank = PayoffBank.terminal_high_modes(cfg.spec, cfg.horizon)
    coarse = audit.lga_marginals(initial, context, cfg.spec, cfg.query_times, 2**-7, bank)
    fine = audit.lga_marginals(initial, context, cfg.spec, cfg.query_times, 2**-10, bank)
    forced = linear_gaussian(initial, context, cfg.spec, cfg.common_times, 2**-7,
                             terminal_weights=bank.terminal_weights, return_joint=False)
    qi = np.rint(np.asarray(cfg.query_times)/cfg.common_step).astype(int)
    assert coarse[2]["integration_steps"] == 8
    assert forced["diagnostics"]["integration_steps"] == 16
    assert fine[2]["integration_steps"] == 64
    assert np.linalg.norm(coarse[1]-forced["scaled_covariance"][qi]) > 1e-8
    assert np.linalg.norm(coarse[1]-fine[1])/np.linalg.norm(fine[1]) < .001


def test_group_audit_matches_literal_cost_predictions_and_whole_family_bootstrap():
    cfg = audit.ReferenceConfig(n_cells=16, paths_per_group=32, refinement_paths=8)
    rng = np.random.default_rng(777)
    initial = np.tile(np.full(16, 1/16), (32, 1))
    data = {"metadata": {"role": "conditional", "case": "uniform"}, "initial": initial,
            "bank_b": rng.normal(size=(32, 3, 4))/np.sqrt(cfg.n_particles),
            "bank_c": rng.normal(size=(32, 3, 4))/np.sqrt(cfg.n_particles)}
    counts = audit.bootstrap_counts(32, audit._group_seed("uniform__conditional__0"), 8)
    results = audit.evaluate_group(data, cfg, "uniform__conditional__0", counts)
    assert len(results) == 2
    assert [record["method"]["requested_max_step"] for record in results] == [2**-7, 2**-10]
    for record in results:
        for bank in audit._banks(cfg):
            direct = audit.lga_marginals(initial[0], cfg.context("uniform"), cfg.spec,
                cfg.query_times, record["method"]["requested_max_step"], bank)
            np.testing.assert_array_equal(record["predictions"][bank.name]["mean"], direct[0])
            np.testing.assert_array_equal(record["predictions"][bank.name]["scaled_covariance"], direct[1])
            metric = record["metrics"][bank.name]["same_time"]
            assert metric["bootstrap"]["counts_sha256"] == audit.array_sha(counts)
            assert metric["n_reference"] == 32
            assert all(row["mean_sampling95_simultaneous_radius"] > 0 for row in metric["records"])
        assert not record["initial_pool_integration_MC"]
        assert "exact killed-integral law" in record["unsupported"]


def test_ensemble_uses_analytic_iid_moments_not_empirical_histogram_pool(monkeypatch):
    cfg = audit.ReferenceConfig(n_cells=16, paths_per_group=32, refinement_paths=8)
    rng = np.random.default_rng(781)
    probabilities = np.arange(1, 17, dtype=float)
    probabilities /= probabilities.sum()
    initial = rng.multinomial(cfg.n_particles, probabilities, size=32)/cfg.n_particles
    data = {"metadata": {"role": "ensemble", "case": "nonuniform"},
            "initial": initial, "probabilities": probabilities,
            "bank_b": rng.normal(size=(32, 3, 4))/np.sqrt(cfg.n_particles),
            "bank_c": rng.normal(size=(32, 3, 4))/np.sqrt(cfg.n_particles)}
    calls = []
    original = audit.lga_marginals

    def recording_lga(initial, context, spec, times, step, bank, initial_covariance):
        calls.append((initial.copy(), initial_covariance.copy(), bank.name, step))
        return original(initial, context, spec, times, step, bank, initial_covariance)

    monkeypatch.setattr(audit, "lga_marginals", recording_lga)
    counts = audit.bootstrap_counts(32, audit._group_seed("nonuniform__ensemble__0"), 8)
    results = audit.evaluate_group(data, cfg, "nonuniform__ensemble__0", counts)
    assert len(calls) == 4
    expected = np.diag(probabilities)-np.outer(probabilities, probabilities)
    for used_initial, covariance, _, _ in calls:
        np.testing.assert_array_equal(used_initial, probabilities)
        np.testing.assert_array_equal(covariance, expected)
    for record in results:
        assert record["initialization"] == "analytic-IID-moments"
        assert not record["initial_pool_integration_MC"]
        for metrics in record["metrics"].values():
            assert metrics["same_time"]["bootstrap"]["counts_sha256"] == audit.array_sha(counts)


@pytest.mark.parametrize("ensemble", [False, True])
def test_literal_marginals_match_full_gaussian_algorithm_on_same_requested_grid(ensemble):
    cfg = audit.ReferenceConfig(n_cells=16)
    centers = (np.arange(16)+.5)*cfg.spec.h
    initial = (1+.2*np.cos(centers))/16
    covariance = audit.iid_scaled_covariance(initial) if ensemble else None
    for bank in audit._banks(cfg):
        literal = audit.lga_marginals(initial, cfg.context("nonuniform"), cfg.spec,
            cfg.query_times, 2**-7, bank, covariance)
        full = linear_gaussian(initial, cfg.context("nonuniform"), cfg.spec,
            cfg.query_times, 2**-7, initial_scaled_covariance=covariance,
            terminal_weights=bank.terminal_weights, running_weights=bank.running_weights,
            return_joint=False)
        # The full API returns terminal and integral outputs separately. The
        # fixed-bank cost algorithm returns their declared payoff-wise sums.
        projection = np.hstack((np.eye(4), np.eye(4)))
        projected_mean = full["mean"] @ projection.T
        projected_covariance = projection @ full["scaled_covariance"] @ projection.T
        np.testing.assert_allclose(literal[0], projected_mean, atol=2e-16, rtol=0)
        np.testing.assert_allclose(literal[1], projected_covariance, atol=2e-14, rtol=0)
        assert literal[2]["integration_steps"] == full["diagnostics"]["integration_steps"]


def test_bridge_bootstrap_seeds_match_all_declared_whole_family_groups():
    from analysis.run_generalization_evaluation import _seed

    cfg = audit.ReferenceConfig()
    seeds = [audit._group_seed(group) for group in cfg.group_keys]
    assert len(set(seeds)) == len(cfg.group_keys)
    for group, seed in zip(cfg.group_keys, seeds, strict=True):
        assert seed == _seed(group)
        np.testing.assert_array_equal(audit.bootstrap_counts(32, seed, 8),
                                      audit.bootstrap_counts(32, _seed(group), 8))


def _prepare_inputs(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(audit, "source_hash", lambda: audit.SOURCE_FREEZE)
    reference = {"status": "waiting-for-complete-fit-lock",
                 "identity": {"config_hash": cfg.config_hash, "source_hash": audit.SOURCE_FREEZE}}
    monkeypatch.setattr(audit, "prepare_reference", lambda *args: reference)
    (tmp_path/"payoffs").mkdir()
    (tmp_path/"timing").mkdir()
    (tmp_path/"reference").mkdir()
    (tmp_path/"payoffs"/"fit_lock.json").write_text(json.dumps({"complete": True}))
    (tmp_path/"timing"/"payoff_benchmark_plan.json").write_text("{}")
    return reference


def test_bridge_preregistration_is_immutable_and_cannot_follow_outcome_generation(tmp_path, monkeypatch):
    cfg = audit.ReferenceConfig(n_cells=16)
    _prepare_inputs(tmp_path, cfg, monkeypatch)
    directory, plan = audit.prepare(tmp_path, cfg)
    assert (directory/"preregistration.json").is_file()
    assert plan["groups"] == list(cfg.group_keys)
    assert plan["steps"] == [2**-7, 2**-10]
    assert not plan["selection_using_reference_outcomes"]
    assert not plan["killed_integral_and_stopping_equivalence"]
    (tmp_path/"timing"/"payoff_benchmark_plan.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="identity"):
        audit.prepare(tmp_path, cfg)


def test_new_bridge_plan_rejects_already_materialized_references(tmp_path, monkeypatch):
    cfg = audit.ReferenceConfig(n_cells=16)
    _prepare_inputs(tmp_path, cfg, monkeypatch)
    (tmp_path/"reference"/"reference.h5").write_bytes(b"physical-outcomes-already-exist")
    with pytest.raises(ValueError, match="before reference outcomes"):
        audit.prepare(tmp_path, cfg)
