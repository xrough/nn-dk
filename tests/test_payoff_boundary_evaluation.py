"""Successor accuracy adapters keep historical streams and modules isolated."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import evaluate_payoff_boundary_regimes as repaired  # noqa: E402
from analysis import evaluate_payoff_regimes as historical  # noqa: E402


def test_evaluator_has_separate_globals_without_mutating_historical_module():
    assert repaired.KIND != historical.KIND
    assert repaired.SEED != historical.SEED
    assert historical.SEED == 2026102819
    assert repaired._implementation.prepare.__globals__["KIND"] == repaired.KIND
    assert repaired._implementation.prepare.__globals__["SEED"] == repaired.SEED
    assert historical.prepare.__globals__["KIND"] == historical.KIND
    assert repaired._implementation._adapter_origin == historical.__file__
    assert Path(repaired._implementation.__file__) == Path(repaired.__file__)


def test_model_factory_uses_successor_strict_restore_and_exact_untrained_factory():
    seen = []
    fitted, baseline = object(), object()
    runner = SimpleNamespace(
        load_declared_model=lambda item: (seen.append(("strict", item)) or fitted),
        untrained_model=lambda case, study: (seen.append(("untrained", case, study)) or baseline))
    study = object()
    item = {"case": "T1_k6", "checkpoint_sha256": "checked-by-successor-loader"}
    assert repaired._model(item, runner, study) is fitted
    assert repaired._model({"case": "T1_k6", "untrained": True}, runner, study) is baseline
    assert seen == [("strict", item), ("untrained", "T1_k6", study)]


def test_estimator_is_unchanged_while_candidate_random_stream_is_fresh():
    rng = np.random.default_rng(39)
    reference = rng.normal(size=(64, 3, 4))
    mean, covariance = historical.empirical(reference)
    counts = historical.bootstrap_counts(64, 43, 16)
    first = historical.bootstrap_advantage(reference, mean, covariance,
                                           mean+.2, covariance, counts, 1)
    second = repaired.bootstrap_advantage(reference, mean, covariance,
                                         mean+.2, covariance, counts, 1)
    assert first == second
    seen = []

    def native(initial, increments, context, spec, step, bank, times, backend=None):
        seen.append(increments.copy())
        return {"outcomes": np.repeat(initial[:, None], 3, axis=1),
                "stopped_at": np.full(len(initial), np.inf),
                "minimum_mass": float(initial.min()), "mass_error": 0.}

    cfg = SimpleNamespace(n_cells=4, n_particles=32, horizon=1/64,
                          spec=object(), query_times=(1/256, 1/128, 1/64),
                          context=lambda profile: profile)
    runner = SimpleNamespace(native_payoff=native)
    probabilities = np.full(4, .25)
    historical._native_mc(runner, cfg, None, "uniform", probabilities,
                          probabilities, 4, 71, None)
    repaired._native_mc(runner, cfg, None, "uniform", probabilities,
                        probabilities, 4, 71, None)
    assert not np.array_equal(seen[0], seen[1])


def test_final_cost_gate_is_still_required_before_loading_any_model_or_reference():
    with pytest.raises(ValueError, match="workers"):
        repaired.benchmark("not-a-run")


def test_legacy_model_inventory_and_expanded_plan_counts_are_frozen(tmp_path, monkeypatch):
    fake = SimpleNamespace(load_plan=lambda root: SimpleNamespace(optimizer_seeds=(0, 1, 2)))
    monkeypatch.setattr(repaired, "_runner", lambda: fake)
    monkeypatch.setattr(repaired.legacy, "inventory", lambda cases, seeds: {
        "complete": True, "models": [{"case": case, "seed": s, "family": "legacy-v2"}
                                     for case in cases for s in seeds]})
    monkeypatch.setattr(repaired.legacy, "verify_targets", lambda *args: None)
    comparison = repaired._immutable(tmp_path/"comparison/preregistration.json",
        {"kind": repaired.KIND, "cases": ["T1_k6"]})
    assert comparison["records_per_group"] == 10
    assert (tmp_path/"comparison/legacy_controls.json").is_file()
    deployment = repaired._immutable(tmp_path/"deployment/preregistration.json", {
        "kind": "independent-boundary-repair-neural-initial-integration-v1",
        "cases": ["T1_k6"], "pools": ["uniform", "nonuniform"],
        "models_per_pool": 4, "expected_records": 8})
    assert deployment["models_per_pool"] == 7 and deployment["expected_records"] == 14
    assert deployment["legacy_controls_sha256"] == comparison["legacy_controls_sha256"]
    monkeypatch.setattr(repaired.legacy, "inventory", lambda *args: {"changed": True})
    with pytest.raises(ValueError, match="identity changed"):
        repaired._immutable(tmp_path/"comparison/preregistration.json",
            {"kind": repaired.KIND, "cases": ["T1_k6"]})


def test_two_predictor_contrast_reduces_to_gaussian_formula_for_fixed_control():
    rng = np.random.default_rng(313)
    reference = rng.normal(size=(256, 3, 4))
    mean, covariance = historical.empirical(reference)
    counts = historical.bootstrap_counts(256, 317, 32)
    first = {"mean": mean+.1, "scaled_covariance": 2048*covariance}
    second = {"mean": mean+.2, "scaled_covariance": 2048*covariance}
    expected = historical.bootstrap_advantage(reference, first["mean"], first["scaled_covariance"],
        second["mean"], second["scaled_covariance"], counts, 2048)
    actual = repaired.predictor_advantage(reference, first, second, counts, 2048)
    assert expected["records"] == actual["records"]


def test_common_predictor_pool_bootstrap_is_retained_for_both_heads():
    rng = np.random.default_rng(331)
    reference = rng.normal(size=(64, 3, 4))
    mean, covariance = historical.empirical(reference)
    counts = historical.bootstrap_counts(64, 337, 32)
    draws = np.broadcast_to(mean, (32, 3, 4)).copy()
    draws[::2] += 10.
    first = {"mean": mean, "scaled_covariance": covariance,
             "bootstrap_mean": draws,
             "bootstrap_scaled_covariance": np.broadcast_to(covariance, (32, 3, 4, 4))}
    same = repaired.predictor_advantage(reference, first, dict(first), counts, 1)
    assert all(row["mean_advantage_percentile95"] == [0., 0.] for row in same["records"])
    assert all(row["covariance_advantage_percentile95"] == [0., 0.] for row in same["records"])


def test_new_deployment_addresses_do_not_reuse_historical_histograms_or_bootstraps():
    from analysis import evaluate_payoff_boundary_deployment as new
    from analysis import evaluate_payoff_regimes_deployment as old

    assert new.original is repaired
    for profile in ("uniform", "nonuniform"):
        assert new.pool_address("T1_k6", profile) != old.pool_address("T1_k6", profile)
        assert new.initial_bootstrap_address("T1_k6", profile) != old.initial_bootstrap_address("T1_k6", profile)
        assert not set(new.planned_families("T1_k6", profile, 32)) & set(old.planned_families("T1_k6", profile, 32))
