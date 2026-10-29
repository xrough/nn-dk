"""Literal costs, historical charges, full-seed claims and qualification gates."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from analysis import audit_payoff_boundary_report as audit


def raw_fixture():
    case, profiles, seeds, paths = "T1_k6", {"T1_k6": ("uniform", "nonuniform")}, (0, 1, 2), 16384
    new = [{"case": case, "name": f"{case}_boundary_seed{s}", "seed": s, "training_seconds": 10.+s}
           for s in seeds]
    old = [{"case": case, "name": f"{case}_legacy_v2_seed{s}", "seed": s, "training_seconds": 20.+s}
           for s in seeds]
    models = [(m["name"], "repaired", m["seed"]) for m in new]
    models += [(m["name"], "legacy-v2", m["seed"]) for m in old]
    models += [(case+"_untrained", "untrained-repaired", 0)]
    timing = {"loading": [], "records": [], "compiled_control": {
        "compile_seconds": .05, "compile_seconds_this_process": .05, "setup_seconds_this_process": .08}}

    def measured(profile, name, family, seed, role, batch):
        repetitions, warmup = ((3, 1) if family == "native-MC" else
                               (5, 1) if role == "ensemble" and family != "LGA" else (21, 3))
        scale = (.02 if family == "native-MC" else .004 if family == "LGA" else
                 .003 if family == "legacy-v2" else .002)
        if role == "ensemble" and family not in ("LGA", "native-MC"):
            scale *= 100
        samples = scale*(1+np.linspace(-.1, .1, repetitions))
        timing["records"].append({"slot": f"{profile}/{name}/{role}/{batch}", "regime": case,
            "profile": profile, "name": name, "family": family, "seed": seed, "role": role, "batch": batch,
            "samples_seconds": samples.tolist(), "first_prediction_seconds": scale+.05,
            "median_seconds": float(np.median(samples)), "minimum_seconds": float(samples.min()),
            "maximum_seconds": float(samples.max()), "p10_seconds": float(np.quantile(samples, .1)),
            "p90_seconds": float(np.quantile(samples, .9)), "p95_seconds": float(np.quantile(samples, .95)),
            "repeats": repetitions, "warmup": warmup})

    for profile in profiles[case]:
        for name, family, seed in models:
            timing["loading"].append({"regime": case, "profile": profile, "name": name, "seconds": .01})
            for role, batch in (("conditional", 1), ("conditional", 64),
                                ("conditional", 256), ("ensemble", paths)):
                measured(profile, name, family, seed, role, batch)
        for name in ("LGA_dt7", "LGA_dt10"):
            for role in ("conditional", "ensemble"):
                measured(profile, name, "LGA", None, role, 1)
        for role in ("conditional", "ensemble"):
            measured(profile, "native-MC-dt10", "native-MC", None, role, paths)
    return timing, new, old, {case: 2.}, {case: 3.}, profiles, seeds, paths


def test_literal_costs_pay_one_model_own_data_restore_and_real_cache_probe():
    costs, totals = audit.replay_costs(*raw_fixture())
    assert len(costs) == 68 and len(totals) == 272
    legacy = next(row for row in costs if row["family"] == "legacy-v2"
                  and row["name"].endswith("seed0") and row["role"] == "ensemble")
    assert legacy["training_seconds"] == 20 and legacy["shared_data_seconds"] == 3
    assert legacy["restoration_seconds"] == .01
    assert legacy["cache_creation_B1_probe_seconds"] == pytest.approx(.053)
    assert legacy["known_offline_seconds"] == pytest.approx(23.063)
    assert legacy["offline_data_source"] == "historical-v2"
    first = next(row for row in costs if row["family"] == "repaired"
                 and row["role"] == "conditional" and row["batch"] == 1)
    assert first["cache_creation_B1_probe_seconds"] == 0
    native = next(row for row in costs if row["family"] == "native-MC")
    assert native["known_offline_seconds"] == pytest.approx(.08)
    assert native["training_seconds"] == native["shared_data_seconds"] == 0
    for row in totals:
        match = next(item for item in costs if item["slot"] == row["slot"])
        assert row["total_known_seconds"] == pytest.approx(match["known_offline_seconds"]
            +match["first_prediction_seconds"]+(row["query_calls"]-1)*match["median_seconds"])


def test_full_six_case_all_seed_inventory_has_408_costs_and_84_model_loads():
    timing, new, old, _, _, _, seeds, paths = raw_fixture()
    full = {"records": [], "loading": [], "compiled_control": timing["compiled_control"]}
    current, historical, profiles = [], [], {}
    for case in ("T0p25_k0p75", "T0p25_k3", "T0p25_k6", "T1_k0p75", "T1_k3", "T1_k6"):
        profiles[case] = ("uniform", "nonuniform")
        for source, destination in ((new, current), (old, historical)):
            destination.extend({**row, "case": case, "name": row["name"].replace("T1_k6", case)}
                               for row in source)
        for table in ("loading", "records"):
            for row in timing[table]:
                full[table].append({key: value.replace("T1_k6", case) if isinstance(value, str)
                                    else value for key, value in row.items()})
    costs, totals = audit.replay_costs(full, current, historical,
        {case: 2. for case in profiles}, {case: 3. for case in profiles}, profiles, seeds, paths)
    assert len(full["loading"]) == 84
    assert len(costs) == 408 and len(totals) == 1632


@pytest.mark.parametrize("field", ["restoration_seconds", "cache_creation_B1_probe_seconds"])
def test_tampered_legacy_restore_or_cache_cost_rejected(field):
    costs, _ = audit.replay_costs(*raw_fixture())
    altered = deepcopy(costs)
    row = next(row for row in altered if row["family"] == "legacy-v2" and row["role"] == "ensemble")
    row[field] = 0.
    with pytest.raises(ValueError, match="Independent report replay mismatch"):
        audit.compare_table(costs, altered, ("slot",), label="costs")


def test_missing_model_seed_and_missing_legacy_load_fail_closed():
    fixture = list(raw_fixture())
    fixture[1] = fixture[1][:-1]
    with pytest.raises(ValueError, match="dropped an optimizer seed"):
        audit.replay_costs(*fixture)
    fixture = list(raw_fixture())
    fixture[0]["loading"] = [row for row in fixture[0]["loading"] if "legacy_v2_seed2" not in row["name"]]
    with pytest.raises(ValueError, match="model loads"):
        audit.replay_costs(*fixture)
    fixture = list(raw_fixture())
    fixture[0]["records"] = [row for row in fixture[0]["records"] if "boundary_seed2" not in row["name"]]
    with pytest.raises(ValueError, match="dropped a seed"):
        audit.replay_costs(*fixture)


def gate_rows():
    return {"max_mean_error_in_reference_std": .01, "max_normalized_covariance_error": .01,
        "mean_error_plus_sampling95": .02, "covariance_error_plus_sampling95": .02,
        "mean_sampling95_simultaneous_radius": .005, "covariance_sampling95_simultaneous_radius": .005}


@pytest.mark.parametrize("statistic", ["mean", "covariance"])
def test_fixed_accuracy_and_each_uncertainty_component_gate(statistic):
    accuracy, refinement = gate_rows(), gate_rows()
    refinement[statistic+"_error_plus_sampling95"] = .005
    sampling = {statistic+"95_radius": .005}
    assert audit.gate([accuracy], [sampling], [refinement], statistic) == "finite-reference-diagnostic-pass"
    point = "max_mean_error_in_reference_std" if statistic == "mean" else "max_normalized_covariance_error"
    accuracy[point] = .051
    assert audit.gate([accuracy], [sampling], [refinement], statistic) == "point-error-above-5pct"
    accuracy[point], accuracy[statistic+"_error_plus_sampling95"] = .01, .046
    assert audit.gate([accuracy], [sampling], [refinement], statistic) == "error-plus-uncertainty-unresolved"
    accuracy[statistic+"_error_plus_sampling95"], accuracy[statistic+"_sampling95_simultaneous_radius"] = .02, .013
    assert audit.gate([accuracy], [sampling], [refinement], statistic) == "sampling-component-above-1.25pct"
    accuracy[statistic+"_sampling95_simultaneous_radius"] = .005
    refinement[statistic+"_error_plus_sampling95"] = .013
    assert audit.gate([accuracy], [sampling], [refinement], statistic) == "reference-refinement-component-unresolved"
    refinement[statistic+"_error_plus_sampling95"], sampling[statistic+"95_radius"] = .005, .013
    assert audit.gate([accuracy], [sampling], [refinement], statistic) == "standalone-MC-component-unresolved"


def test_amortization_uses_matched_legacy_seed_and_rejects_inaccurate_speed_claim():
    costs, _ = audit.replay_costs(*raw_fixture())
    accuracy, sampling, refinement = [], [], []
    for row in costs:
        if row["role"] == "conditional" and row["batch"] not in (1, 16384):
            continue
        base = {key: row[key] for key in ("regime", "profile", "role", "name")}
        accuracy.append({**base, **gate_rows()})
        sampling.append({**base, "mean95_radius": .005, "covariance95_radius": .005})
    for profile in ("uniform", "nonuniform"):
        for role in ("conditional", "ensemble"):
            row = {"regime": "T1_k6", "profile": profile, "role": role,
                   "level": "level0_versus_finest", **gate_rows()}
            row.update(mean_error_plus_sampling95=.005, covariance_error_plus_sampling95=.005)
            refinement.append(row)
    result = audit.replay_amortization(costs, accuracy, sampling, refinement)
    assert len(result) == 96
    legacy_rows = [row for row in result if "legacy_v2" in row["control"]]
    assert len(legacy_rows) == 24
    assert all(row["candidate"].rsplit("seed", 1)[1] == row["control"].rsplit("seed", 1)[1] for row in legacy_rows)
    for row in accuracy:
        if "boundary_seed" in row["name"]:
            row["max_mean_error_in_reference_std"] = .2
    failed = audit.replay_amortization(costs, accuracy, sampling, refinement)
    assert all(row["accuracy_qualified_payback_calls"] is None for row in failed if row["statistic"] == "mean")
    assert any(row["descriptive_payback_calls"] is not None for row in failed if row["statistic"] == "mean")


def test_consistency_cannot_select_a_subset_of_optimizer_seeds():
    rows = [{"regime": "T1_k6", "profile": "uniform", "role": "ensemble", "group": "uniform__ensemble__0",
        "time_index": 2, "time": 1., "statistic": "mean", "name": f"T1_k6_boundary_seed{seed}",
        "advantage": .2+seed*.1, "lower95": .1+seed*.1, "upper95": .3+seed*.1} for seed in (0, 1, 2)]
    result = audit.summarize_contrasts(rows, (0, 1, 2))
    assert len(result) == 1 and result[0]["all_seeds_NN_closer_pointwise95"]
    assert not result[0]["all_seeds_LGA_closer_pointwise95"]
    with pytest.raises(ValueError, match="dropped or duplicated"):
        audit.summarize_contrasts(rows[:2], (0, 1, 2))
    rows[-1]["name"] = "T1_k6_boundary_seed0"
    with pytest.raises(ValueError, match="dropped or duplicated"):
        audit.summarize_contrasts(rows, (0, 1, 2))


def test_worker_CPU_monotonic_durations_and_calendar_span_are_distinct():
    rows = [{"seed": 0, "wall_seconds": 2., "cpu_seconds": 1., "started_unix": 10., "finished_unix": 12.},
            {"seed": 0, "wall_seconds": 3., "cpu_seconds": 2., "started_unix": 100., "finished_unix": 103.},
            {"seed": 1, "wall_seconds": 4., "cpu_seconds": 3., "started_unix": 11., "finished_unix": 15.}]
    assert audit.worker_accounting(rows) == {"sum_CPU_seconds": 6., "sum_worker_wall_seconds": 9.,
        "longest_worker_wall_seconds": 5., "UTC_start_finish_span_seconds": 93.}
