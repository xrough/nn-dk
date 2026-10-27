"""Independent audit adapters retain historical sources and robust strata."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit_payoff_boundary_regimes as pretest  # noqa: E402
from analysis import audit_payoff_boundary_validation as validation  # noqa: E402
from analysis import audit_payoff_regimes as historical  # noqa: E402


def test_actual_audit_function_local_import_routes_only_to_new_runner(tmp_path, monkeypatch):
    from analysis import run_payoff_boundary_regimes as new
    from analysis import run_payoff_regimes as old

    historical_function = historical.pretest_audit
    canonical_assert = old._assert_frozen

    def sentinel(root):
        raise RuntimeError("successor-only local dependency")

    monkeypatch.setattr(new._implementation, "_assert_frozen", sentinel)
    with pytest.raises(RuntimeError, match="successor-only"):
        pretest.pretest_audit(tmp_path)
    assert old._assert_frozen is canonical_assert
    assert historical.pretest_audit is historical_function


def test_fixed_unit_loss_has_no_remaining_time_amplification_and_robust_tail():
    scale = {"mean_scale": [1., 2., 3., 4.], "pseudo_huber_delta": 3.,
             "inverse_time_weighting": False, "normalization_time_powers": [0]*4}
    raw = np.ones((4, 4))
    horizons = np.array([0., 1e-9, .1, 1.])
    normalized = validation.normalized_residual(raw, horizons, scale, 1., "mean")
    np.testing.assert_array_equal(normalized, np.broadcast_to(1/np.arange(1, 5), (4, 4)))
    robust = validation.robust_values(np.array([1e-10, 1., 1000.]), 3.)
    np.testing.assert_allclose(robust[:2], 18*(np.hypot(1., np.array([1e-10, 1.])/3.)-1), atol=1e-19)
    assert robust[-1] < 6000. and robust[-1] > 5900.
    with pytest.raises(ValueError, match="disable"):
        validation.normalized_residual(raw, horizons, dict(scale, inverse_time_weighting=True), 1., "mean")


def test_strata_keep_raw_squared_and_robust_objectives_and_empty_rows():
    model = SimpleNamespace(horizon=1., context=SimpleNamespace(n_particles=2048),
        bank=SimpleNamespace(scales=np.ones(4), integral_rows=np.array([False, False, True, True])))
    scaling = {"mean_scale": [1.]*4, "pseudo_huber_delta": 3.,
               "inverse_time_weighting": False, "normalization_time_powers": [0]*4}
    residual = np.array([[0., 1., 2., 3.], [0., 0., 0., 100.]])
    summary = validation.residual_summary(residual, np.array([1e-8, 1.]), scaling, model, "mean")
    assert summary["rows"] == 2
    assert summary["normalized_squared_objective"] == np.mean(residual**2)
    np.testing.assert_allclose(summary["normalized_objective"], np.mean(validation.robust_values(residual, 3.)))
    assert summary["normalized_rms"] == np.sqrt(np.mean(residual**2))
    assert validation.residual_summary(np.empty((0, 4)), [], scaling, model, "mean") == {"rows": 0, "empty": True}


def test_independent_contrast_algebra_includes_shared_pool_uncertainty():
    from analysis import audit_payoff_boundary_metrics as independent
    from analysis import evaluate_payoff_boundary_regimes as production

    rng = np.random.default_rng(347)
    reference = rng.normal(size=(128, 3, 4))
    counts = np.random.default_rng(349).multinomial(128, np.full(128, 1/128), size=32)
    rm, rc = independent.mathematics.empirical(reference)
    bm, bc = independent.mathematics.empirical(reference, counts)
    first = {"mean": rm+.1, "scaled_covariance": 2048*rc,
             "bootstrap_mean": bm+.1, "bootstrap_scaled_covariance": 2048*bc}
    second = {"mean": rm+.2, "scaled_covariance": 2048*rc,
              "bootstrap_mean": bm+.2, "bootstrap_scaled_covariance": 2048*bc}
    expected = production.predictor_advantage(reference, first, second, counts, 2048)["records"]
    actual = independent.contrast_both((rm, rc), (bm, bc), first, second, 2048)
    assert independent.mathematics._compare(expected, actual) < 1e-14


def test_history_reads_npz_family_identifiers_without_outcome_or_state_arrays(tmp_path, monkeypatch):
    root, excluded = tmp_path/"runs", tmp_path/"runs/current"
    old = root/"historical"
    old.mkdir(parents=True)
    np.savez_compressed(old/"candidate.npz", family_ids=np.array(["native-a", "native-b"]),
                        outcomes=np.array([object()]))
    np.savez_compressed(old/"planned_initial_families.npz", pool=np.array(["iid-a", "iid-b"]))
    excluded.mkdir(parents=True)
    np.savez_compressed(excluded/"ignored.npz", family_ids=np.array(["new-family"]))
    monkeypatch.setattr(pretest, "_historical_h5_inventory", lambda *args: (set(), []))
    identifiers, records = pretest.historical_identifier_inventory(root, excluded, {"new-family"})
    assert identifiers == {"native-a", "native-b", "iid-a", "iid-b"}
    assert len(records) == 2
    assert all(not record["historical_states_or_outcomes_read"] for record in records)
    assert {key for record in records for key in record["datasets_read"]} == {"family_ids", "pool"}
    with pytest.raises(ValueError, match="NPZ trajectory-family overlap"):
        pretest.historical_identifier_inventory(root, excluded, {"native-a"})
