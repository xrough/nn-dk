"""All-family deployment, historical provenance and repaired report gates."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import report_payoff_boundary_regimes as report  # noqa: E402


def _cost_row(name="new", family="repaired", seed=0, batch=1, role="conditional"):
    return dict(slot=f"{name}/{role}/{batch}", regime="case", profile="uniform", name=name,
        family=family, seed=seed, role=role, batch=batch, median_seconds=.001,
        first_prediction_seconds=.02)


def test_historical_costs_charge_own_fit_data_restoration_and_cache_probe():
    new, old = _cost_row(), _cost_row("old", "legacy-v2")
    old_ensemble = dict(old, slot="old/ensemble", role="ensemble", batch=16384,
                        first_prediction_seconds=.3, median_seconds=.2)
    timing = {"records": [new, old, old_ensemble], "loading": [
        dict(regime="case", profile="uniform", name="new", seconds=.1),
        dict(regime="case", profile="uniform", name="old", seconds=.15)]}
    costs, totals = report.cost_tables(timing, {"models": [
        dict(case="case", name="new", training_seconds=10.)]}, {"case": 1.},
        [dict(case="case", name="old", historical_training_seconds=20.)], {"case": 2.})
    assert costs[0]["known_offline_seconds"] == pytest.approx(11.1)
    assert costs[1]["known_offline_seconds"] == pytest.approx(22.15)
    assert costs[2]["known_offline_seconds"] == pytest.approx(22.17)
    assert costs[0]["offline_data_source"] == "new-v3"
    assert costs[2]["offline_data_source"] == "historical-v2"
    assert costs[2]["offline_training_is_historical"]
    value = next(row for row in totals if row["slot"] == "old/ensemble" and row["query_calls"] == 100)
    assert value["total_known_seconds"] == pytest.approx(22.17+.3+99*.2)
    assert len(totals) == 12


def test_no_historical_pool_cost_can_silently_default_to_zero():
    with pytest.raises(KeyError):
        report.cost_tables({"records": [_cost_row("old", "legacy-v2")], "loading": []},
            {"models": []}, {}, [dict(case="case", name="old", historical_training_seconds=20.)])


def _record(name, family, independent=False):
    return {"identity": dict(regime="case", group="uniform__ensemble__0", name=name),
            "method": dict(name=name, family=family), "independent": independent}


def test_independent_primary_replaces_legacy_too_without_losing_original_records():
    records = [_record("new", "repaired"), _record("old", "legacy-v2"),
               _record("heat", "untrained-repaired"), _record("LGA", "LGA")]
    replacements = {(row["identity"]["regime"], row["identity"]["group"], row["identity"]["name"]):
                    dict(row, independent=True) for row in records[:3]}
    group = dict(regime="case", group="uniform__ensemble__0", NN_versus_fine_LGA="paired-LGA",
                 NN_versus_legacy_v2="paired-legacy", reference_refinement="unchanged-refinement")
    evidence = {"records": records, "groups": [group]}
    report._graft_independent_records(evidence, replacements, {("case", "uniform__ensemble__0"):
        dict(NN_versus_fine_LGA="independent-LGA", NN_versus_legacy_v2="independent-legacy")})
    assert len(evidence["records"]) == 4 and len(evidence["paired_initial_records"]) == 3
    assert all(row["independent"] for row in evidence["records"][:3])
    assert evidence["records"][-1] is records[-1]
    assert evidence["groups"][0]["NN_versus_legacy_v2"] == "independent-legacy"
    assert evidence["groups"][0]["reference_refinement"] == "unchanged-refinement"


def test_incomplete_independent_legacy_replacement_is_rejected():
    evidence = {"records": [_record("old", "legacy-v2")], "groups": []}
    with pytest.raises(ValueError, match="incomplete"):
        report._graft_independent_records(evidence, {}, {})


def test_three_seeds_are_required_for_both_families():
    rows = [{"method": dict(family=family, seed=seed)} for family in ("repaired", "legacy-v2")
            for seed in (0, 1, 2)] + [{"method": dict(family="untrained-repaired", seed=0)}]
    report._head_inventory(rows, (0, 1, 2))
    with pytest.raises(ValueError, match="historical optimizer"):
        report._head_inventory(rows[:5]+rows[6:], (0, 1, 2))


def test_timing_verifier_requires_all68_workloads_and14_restorations(monkeypatch):
    fake = SimpleNamespace(reference_configuration=lambda *args: SimpleNamespace(cases=("uniform", "nonuniform")))
    monkeypatch.setattr(report, "runner", fake)
    evidence = {"study": object(), "plan": {"cases": ["case"],
        "neural_conditional_batches": [1, 64, 256], "LGA_max_steps": [2**-7, 2**-10]},
        "lock": {"models": [{"case": "case", "name": f"new{s}"} for s in range(3)]},
        "legacy": {"models": [{"case": "case", "name": f"old{s}"} for s in range(3)]}}
    records, loading = [], []
    for profile in ("uniform", "nonuniform"):
        for name in [*(f"new{s}" for s in range(3)), *(f"old{s}" for s in range(3)), "case_untrained"]:
            loading.append(dict(regime="case", profile=profile, name=name, seconds=.1))
            for batch in (1, 64, 256):
                records.append(dict(slot=f"case/{profile}/{name}/conditional/{batch}", median_seconds=.1, first_prediction_seconds=.2))
            records.append(dict(slot=f"case/{profile}/{name}/ensemble", median_seconds=.1, first_prediction_seconds=.2))
        for step in (2**-7, 2**-10):
            for role in ("conditional", "ensemble"):
                records.append(dict(slot=f"case/{profile}/LGA/{step}/{role}", median_seconds=.1, first_prediction_seconds=.2))
        for role in ("conditional", "ensemble"):
            records.append(dict(slot=f"case/{profile}/native-MC/{role}", median_seconds=.1, first_prediction_seconds=.2))
    evidence["timing"] = dict(records=records, loading=loading)
    assert len(records) == 68 and len(loading) == 14
    report._verify_timing(evidence)
    evidence["timing"]["loading"] = loading[:-1]
    with pytest.raises(ValueError, match="Every declared"):
        report._verify_timing(evidence)


def test_amortization_uses_matched_legacy_seed_and_batch1_only():
    candidate = dict(_cost_row(), known_offline_seconds=10.)
    controls = [dict(_cost_row("old", "legacy-v2", seed=0), known_offline_seconds=20., median_seconds=.003),
                dict(_cost_row("other", "legacy-v2", seed=1), known_offline_seconds=20., median_seconds=.003),
                dict(_cost_row("old", "legacy-v2", seed=0, batch=64), known_offline_seconds=20., median_seconds=.003)]
    tables = {"costs": [candidate, *controls], "accuracy": [], "sampling": [], "reference_refinement": []}
    rows = report.amortization_tables(tables)
    assert len(rows) == 2 and {row["control"] for row in rows} == {"old"}
    assert all(row["descriptive_payback_calls"] == 1 for row in rows)
    assert all(row["accuracy_qualified_payback_calls"] is None for row in rows)


def test_report_adapter_preserves_canonical_historical_report():
    from analysis import report_payoff_regimes as old

    assert old.runner.WORKFLOW == "harder-native-dk-repaired-payoff-comparison-v1"
    assert report._implementation.runner.WORKFLOW == "boundary-consistent-native-dk-payoff-comparison-v1"
    assert report._implementation.__file__ == report.__file__
    assert report._implementation._adapter_origin == old.__file__


def test_robust_loss_concentration_does_not_hide_squared_residual_outliers():
    full = dict(rows=100, normalized_objective=10., normalized_squared_objective=100.)
    stratum = dict(rows=2, normalized_objective=50., normalized_squared_objective=1000.)
    shares = report.loss_shares(stratum, full)
    assert shares["fraction_of_full_robust_loss"] == pytest.approx(.1)
    assert shares["fraction_of_full_squared_loss"] == pytest.approx(.2)
    assert report.loss_shares(dict(rows=0), full) == {
        "fraction_of_full_robust_loss": None, "fraction_of_full_squared_loss": None}
    assert report.loss_shares(stratum, dict(full, normalized_objective=0.))[
        "fraction_of_full_robust_loss"] is None


def test_pool_labels_follow_each_legacy_record_in_primary_and_paired_diagnostics():
    paired = _record("old", "legacy-v2")
    primary = dict(paired, initial_pool_independent_of_reference=True)
    assert report.initial_pool_status(paired) == "shared-with-reference"
    assert report.initial_pool_status(primary) == "independent"
    assert report.initial_pool_status(_record("LGA", "LGA")) == "not-applicable"


def test_observable_blocks_use_declared_weights_and_labels_after_permutation(monkeypatch):
    bank = SimpleNamespace(labels=("integral-A", "terminal-B"),
        terminal_weights=np.array([[0., 0.], [1., -1.]]),
        running_weights=np.array([[1., -1.], [0., 0.]]))
    monkeypatch.setattr(report, "runner", SimpleNamespace(payoff_bank=lambda *args: bank))
    base = dict(regime="case", horizon=1., kappa=6., family="repaired", name="new0", seed=0,
                group="uniform__ensemble__0", time=1.)
    tables = {"components": [dict(base, observable="integral-A", normalized_mean_difference=-.2),
                              dict(base, observable="terminal-B", normalized_mean_difference=.1)],
              "covariances": [dict(base, observable_i="integral-A", observable_j="terminal-B",
                                   normalized_covariance_difference=-.3),
                               dict(base, observable_i="integral-A", observable_j="integral-A",
                                    normalized_covariance_difference=.4),
                               dict(base, observable_i="terminal-B", observable_j="terminal-B",
                                    normalized_covariance_difference=.5)]}
    evidence = dict(plan={"cases": ["case"]}, study=object())
    rows = report.observable_block_tables(evidence, tables)
    assert {row["observable_block"]: row["worst_normalized_error"] for row in rows} == {
        "terminal": .1, "integral": .2, "TT": .5, "TI": .3, "II": .4}
    assert next(row for row in rows if row["observable_block"] == "integral")["worst_observable"] == "integral-A"


def test_block_classification_rejects_mixed_weights_instead_of_guessing_from_label():
    bank = SimpleNamespace(labels=("terminal-mode3",), terminal_weights=np.ones((1, 2)),
                           running_weights=np.ones((1, 2)))
    with pytest.raises(ValueError, match="pure terminal"):
        report.observable_types(bank)
