"""Deployment address auditing never draws a held-out initial histogram."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit_payoff_regimes_deployment as audit  # noqa: E402
from analysis import evaluate_payoff_regimes_deployment as deployment  # noqa: E402


def _plan():
    pools, arrays = [], {}
    for regime in ("T0p25_k3", "T1_k6"):
        for profile in ("uniform", "nonuniform"):
            families = deployment.planned_families(regime, profile, 16)
            key = regime+"__"+profile
            arrays[key] = families
            pools.append({"key": key, "regime": regime, "profile": profile, "paths": 16,
                          "initial_seed": deployment.pool_address(regime, profile),
                          "initial_stream": [deployment.KIND, "multinomial-initials"],
                          "bootstrap_seed": deployment.initial_bootstrap_address(regime, profile),
                          "family_ids_array_sha256": deployment.array_sha(families)})
    return {"seed": deployment.SEED, "kind": deployment.KIND, "pools": pools}, arrays


def test_pool_contracts_reconstruct_real_predeclared_formulas_without_rng(monkeypatch):
    plan, arrays = _plan()

    def prohibited(*args, **kwargs):
        raise AssertionError("Pretest independence audit must not draw random histograms or counts")

    monkeypatch.setattr(deployment, "seeded_rng", prohibited)
    monkeypatch.setattr(np.random, "default_rng", prohibited)
    ids, addresses, bootstraps, records = audit.verify_pool_contracts(plan, arrays, deployment.array_sha)
    assert len(ids) == 64 and len(addresses) == len(bootstraps) == len(records) == 4
    for row, pool in zip(records, plan["pools"], strict=True):
        assert row["initial_seed"] == pool["initial_seed"]
        assert row["bootstrap_default_rng_integer_seed"] == pool["bootstrap_seed"]


def test_changed_family_or_rng_stream_is_rejected_before_outcomes():
    plan, arrays = _plan()
    changed = copy.deepcopy(arrays)
    changed[plan["pools"][0]["key"]][0] = b"reused-old-family"
    with pytest.raises(ValueError, match="formula changed"):
        audit.verify_pool_contracts(plan, changed, deployment.array_sha)
    changed_plan = copy.deepcopy(plan)
    changed_plan["pools"][0]["initial_stream"] = [deployment.KIND, "reference-forcing"]
    with pytest.raises(ValueError, match="formula changed"):
        audit.verify_pool_contracts(changed_plan, arrays, deployment.array_sha)


def test_independent_bootstrap_seeds_cannot_reuse_reference_or_candidate_counts():
    result = audit.check_bootstrap_seed_independence({17, 18}, {19, 20}, 21)
    assert not result["bootstrap_count_drawn_during_audit"]
    with pytest.raises(ValueError, match="overlap"):
        audit.check_bootstrap_seed_independence({17, 18}, {18, 19}, 20)
    with pytest.raises(ValueError, match="overlap"):
        audit.check_bootstrap_seed_independence({17, 18}, {19, 20}, 18)
