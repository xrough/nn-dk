"""Independent, scalar-only audit of the final cost and accuracy-gate tables.

No model is imported, no numeric outcome array is opened, and no bootstrap is
drawn or replayed. The changing report manifest is deliberately not an input.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
INPUTS: dict[str, str] = {}
MAX_ERROR = 0.0
CHECKS = 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    INPUTS[str(path.resolve())] = sha(path)
    return json.loads(path.read_text())


def rows(path):
    path = Path(path)
    INPUTS[str(path.resolve())] = sha(path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    return None if value in (None, "") else float(value)


def near(actual, expected, label):
    global MAX_ERROR, CHECKS
    actual, expected = float(actual), float(expected)
    difference = abs(actual-expected)
    MAX_ERROR = max(MAX_ERROR, difference)
    CHECKS += 1
    if not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=3e-12):
        raise AssertionError((label, actual, expected, difference))


def boolean(value):
    if isinstance(value, bool):
        return value
    assert value in ("True", "False"), value
    return value == "True"


def quantile(values, probability):
    ordered = sorted(values)
    location = (len(ordered)-1)*probability
    left = math.floor(location)
    fraction = location-left
    return ordered[left]*(1-fraction)+ordered[min(left+1, len(ordered)-1)]*fraction


def gate(accuracy, sampling, refinement, statistic):
    """Independently evaluate all declared finite-reference gate ingredients."""
    if not accuracy or not refinement:
        return "unresolved-inventory", {}
    mean = statistic == "mean"
    point_key = "max_mean_error_in_reference_std" if mean else "max_normalized_covariance_error"
    upper_key = statistic+"_error_plus_sampling95"
    radius_key = statistic+"_sampling95_simultaneous_radius"
    maximum = lambda pool, key: max(float(row[key]) for row in pool)
    values = {
        "point_error": maximum(accuracy, point_key),
        "discrepancy_upper": maximum(accuracy, upper_key),
        "reference_refinement_upper": maximum(refinement, upper_key),
        "method_discrepancy_sampling_radius": maximum(accuracy, radius_key),
        "standalone_sampling_radius": maximum(sampling, statistic+"95_radius") if sampling else 0.,
    }
    failures = [
        (values["point_error"] > .05, "point-error-above-5pct"),
        (values["discrepancy_upper"]+values["reference_refinement_upper"] > .05,
         "error-plus-uncertainty-unresolved"),
        (values["method_discrepancy_sampling_radius"] > .0125, "sampling-component-above-1.25pct"),
        (values["reference_refinement_upper"] > .0125, "reference-refinement-component-unresolved"),
        (values["standalone_sampling_radius"] > .0125, "standalone-MC-component-unresolved"),
    ]
    values["all_failed_conditions"] = [reason for failed, reason in failures if failed]
    return next((reason for failed, reason in failures if failed), "finite-reference-diagnostic-pass"), values


def main():
    started, cpu = time.perf_counter(), time.process_time()
    assert (ROOT/"report/manifest.json").exists(), "Final report generation has not finished"
    study = read(ROOT/"plan.json")
    timing = read(ROOT/"comparison/timing.json")
    lock = read(ROOT/"payoffs/fit_lock.json")
    metrics = read(ROOT/"audit/metrics_replay.json")
    assert timing["complete"] and timing["final_timing_after_all_workers_finished"]
    assert timing["environment"]["cpu_threads"] == 1
    assert len(timing["records"]) == 264 and len(timing["loading"]) == 48
    assert len(lock["models"]) == 18 and lock["complete"] and not lock["selection_using_final_outcomes"]
    package = REPO/"src/nn_dk"
    package_content = {str(p.relative_to(package)): p.read_text() for p in sorted(package.rglob("*.py"))}
    package_hash = hashlib.sha256(json.dumps(package_content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert package_hash == timing["identity"]["source_hash"] == lock["source_hash"] == metrics["identity"]["source_hash"]
    assert timing["identity"]["fit_lock_sha256"] == INPUTS[str((ROOT/"payoffs/fit_lock.json").resolve())]
    assert timing["identity"]["plan_sha256"] == sha(ROOT/"comparison/preregistration.json")
    for helper in ("run_payoff_regimes.py", "evaluate_payoff_regimes.py", "evaluate_payoff_regimes_deployment.py",
                   "native_euler_control.cpp", "compiled_native.py", "native_gaussian_baseline.py"):
        path = REPO/"analysis"/helper
        INPUTS[str(path)] = sha(path)
    INPUTS[str(Path(__file__).resolve())] = sha(__file__)
    INPUTS[str((ROOT/"comparison/preregistration.json").resolve())] = sha(ROOT/"comparison/preregistration.json")
    tables = {p.stem: rows(p) for p in sorted((ROOT/"report").glob("*.csv"))}
    costs, totals, amortization = (tables[name] for name in ("costs", "total_costs", "amortization"))
    assert len(costs) == 264 and len(totals) == 1056 and len(amortization) == 432
    cost_by_slot = {row["slot"]: row for row in costs}
    assert len(cost_by_slot) == len(costs)
    fits = {(m["case"], m["name"]): m for m in lock["models"]}
    assert len(fits) == 18
    loads = {(m["regime"], m["profile"], m["name"]): m["seconds"] for m in timing["loading"]}
    probes = {(m["regime"], m["profile"], m["name"]): m["first_prediction_seconds"]
              for m in timing["records"] if m["role"] == "conditional" and m["batch"] == 1}
    regimes = sorted({r["regime"] for r in timing["records"]})
    assert len(regimes) == len(study["horizons"])*len(study["kappas"]) == 6
    data_seconds = {}
    data_rows = {(r["regime"], r["role"]): r for r in tables["data"]}
    assert len(data_rows) == 12
    for case in regimes:
        data_seconds[case] = 0.
        for role in ("train", "validation"):
            data = read(ROOT/f"payoffs/{case}/states_{role}.json")
            assert data["complete"] and data["all_paths_retained"] and not data["future_payoff_labels"]
            assert data["source_hash"] == package_hash
            data_seconds[case] += data["generation_seconds"]
            near(data_rows[case, role]["generation_seconds"], data["generation_seconds"], "generation table")
    measured_keys = set()
    stats = defaultdict(list)
    compiled = timing["compiled_control"]
    near(compiled["cpu_threads"], 1, "native thread count")
    assert compiled["fast_math"] is False and compiled["dtype"] == "float64"
    for measured in timing["records"]:
        row = cost_by_slot[measured["slot"]]
        case, profile, name, family, role, batch = (measured[k] for k in ("regime", "profile", "name", "family", "role", "batch"))
        key = (case, profile, name, role, batch)
        assert key not in measured_keys
        measured_keys.add(key)
        for field, value in measured.items():
            if field == "samples_seconds":
                continue
            if isinstance(value, (int, float)):
                near(row[field], value, "literal timing field "+field)
            else:
                assert row[field] == str(value), (field, row[field], value)
        sample = measured["samples_seconds"]
        assert len(sample) == measured["repeats"] and all(v > 0 for v in sample)
        for field, expected in (("median_seconds", statistics.median(sample)), ("minimum_seconds", min(sample)),
                                ("maximum_seconds", max(sample)), ("p10_seconds", quantile(sample, .1)),
                                ("p90_seconds", quantile(sample, .9)), ("p95_seconds", quantile(sample, .95))):
            near(row[field], expected, "timing reduction "+field)
        fit = fits.get((case, name))
        assert (fit is not None) == (family == "repaired")
        native = family == "native-MC"
        head = family in ("repaired", "untrained-repaired")
        expected = {
            "training_seconds": fit["training_seconds"] if fit else 0.,
            "shared_data_seconds": data_seconds[case] if fit else 0.,
            "restoration_seconds": loads.get((case, profile, name), 0.),
            "cold_compilation_seconds": compiled["compile_seconds"] if native else 0.,
            "compiled_setup_seconds": max(0., compiled["setup_seconds_this_process"]-compiled["compile_seconds_this_process"]) if native else 0.,
            "cache_creation_B1_probe_seconds": probes[case, profile, name] if head and (role != "conditional" or batch != 1) else 0.,
        }
        for field, value in expected.items():
            near(row[field], value, "offline component "+field)
        near(row["known_offline_seconds"], sum(expected.values()), "one-fit full-data complete offline total")
        if head:
            assert ((role == "conditional" and batch in (1, 64, 256)) or (role == "ensemble" and batch == study["paths_per_group"]))
        elif family == "LGA":
            assert batch == 1 and name in ("LGA_dt7", "LGA_dt10")
            near(row["step"], 2**(-int(name.removeprefix("LGA_dt"))), "Gaussian step")
        elif native:
            assert batch == study["paths_per_group"]
            near(row["step"], 2**-10, "native deployment step")
        else:
            raise AssertionError(family)
        stats[case, family, role].append(float(row["median_seconds"]))
    assert len(measured_keys) == 264
    for case in regimes:
        for profile in ("uniform", "nonuniform"):
            measured = [r for r in timing["records"] if r["regime"] == case and r["profile"] == profile]
            assert len(measured) == 22
            assert Counter((r["family"], r["role"], r["batch"]) for r in measured) == Counter({
                ("repaired", "conditional", 1): 3, ("repaired", "conditional", 64): 3,
                ("repaired", "conditional", 256): 3, ("repaired", "ensemble", 16384): 3,
                ("untrained-repaired", "conditional", 1): 1, ("untrained-repaired", "conditional", 64): 1,
                ("untrained-repaired", "conditional", 256): 1, ("untrained-repaired", "ensemble", 16384): 1,
                ("LGA", "conditional", 1): 2, ("LGA", "ensemble", 1): 2,
                ("native-MC", "conditional", 16384): 1, ("native-MC", "ensemble", 16384): 1})
    total_keys = set()
    for total in totals:
        key = (total["slot"], int(total["query_calls"]))
        assert key not in total_keys
        total_keys.add(key)
        cost, calls = cost_by_slot[total["slot"]], key[1]
        assert calls in (1, 100, 10000, 100000)
        for field in ("regime", "profile", "name", "family", "role", "batch"):
            assert total[field] == cost[field]
        expected = float(cost["known_offline_seconds"])+float(cost["first_prediction_seconds"])+(calls-1)*float(cost["median_seconds"])
        near(total["known_offline_seconds"], cost["known_offline_seconds"], "total offline")
        near(total["first_prediction_seconds"], cost["first_prediction_seconds"], "total first")
        near(total["subsequent_warm_seconds"], (calls-1)*float(cost["median_seconds"]), "subsequent calls")
        near(total["total_known_seconds"], expected, "complete total prediction cost")
    assert total_keys == {(slot, q) for slot in cost_by_slot for q in (1, 100, 10000, 100000)}

    # Check every published metric against its scalar source packet. This reads
    # existing JSON only, rather than recomputing outcomes or bootstrap arrays.
    source_records = {}
    for path in sorted((ROOT/"comparison/accuracy").glob("*/*/group.json")):
        group = read(path)
        for item in group["records"]:
            packet = read(item["path"])
            identity = packet["identity"]
            source_records[identity["regime"], identity["group"], identity["name"]] = packet
    assert len(source_records) == 252
    replacements = 0
    for path in sorted((ROOT/"deployment/accuracy").glob("*/*/group.json")):
        group = read(path)
        for item in group["records"]:
            packet = read(item["path"])
            assert packet["initial_pool_independent_of_reference"] and packet["reference_and_initial_bootstrap_independent"]
            identity = packet["identity"]
            key = (identity["regime"], identity["group"], identity["name"])
            assert key in source_records
            source_records[key] = packet
            replacements += 1
    assert replacements == 48
    accuracy_index = {(r["regime"], r["group"], r["name"], int(r["time_index"])): r for r in tables["accuracy"]}
    assert len(accuracy_index) == len(tables["accuracy"]) == 756
    metric_fields = ("max_mean_error_in_reference_std", "max_normalized_covariance_error",
                     "mean_sampling95_simultaneous_radius", "covariance_sampling95_simultaneous_radius",
                     "mean_error_plus_sampling95", "covariance_error_plus_sampling95")
    for key, packet in source_records.items():
        for item in packet["metrics"]["records"]:
            row = accuracy_index[*key, int(item["time_index"])]
            for field in metric_fields:
                near(row[field], item[field], "source accuracy scalar "+field)
            profile, role, anchor = key[1].split("__")
            assert (row["profile"], row["role"], row["anchor"]) == (profile, role, anchor)
            assert row["family"] == packet["method"]["family"]
            if role == "ensemble" and row["family"] in ("repaired", "untrained-repaired"):
                assert row["NN_initial_pool"] == "independent"
    independent_replays = metrics["records"]
    sampling_index = {(r["regime"], r["group"], r["name"], int(r["time_index"])): r for r in tables["sampling"]}
    for item in independent_replays:
        for t in range(3):
            key = (item["case"], item["group"], item["name"], t)
            for field in metric_fields:
                near(accuracy_index[key][field], item["scalar_metric_records"][t][field], "independently audited metric")
            for field in ("mean95_radius", "covariance95_radius"):
                near(sampling_index[key][field], item["standalone_sampling_in_reference_units"][field][t], "independently audited initial integration uncertainty")
    refinement_index = {(r["regime"], r["group"], r["level"], int(r["time_index"])): r for r in tables["reference_refinement"]}
    assert len(refinement_index) == 216
    for item in metrics["reference_refinement"]:
        for t in range(3):
            row = refinement_index[item["case"], item["group"], item["level"], t]
            for field in metric_fields:
                near(row[field], item["records"][t][field], "independently audited reference refinement")
    consistency_index = {(r["regime"], r["group"], r["statistic"], int(r["time_index"])): r for r in tables["seed_consistency"]}
    assert len(consistency_index) == 216
    for item in metrics["all_seed_consistency"]:
        row = consistency_index[item["case"], item["group"], item["statistic"], item["time_index"]]
        for field in ("all_seeds_NN_closer_pointwise95", "all_seeds_LGA_closer_pointwise95"):
            assert boolean(row[field]) == item[field]
        for field in ("minimum_lower95", "maximum_upper95"):
            near(row[field], item[field], "independent all-seed contrast")

    # Deployment accuracy qualification is task-, law- and physics-specific.
    eligible = [r for r in costs if r["family"] == "repaired" and
                (r["role"] != "conditional" or int(r["batch"]) == 1)]
    assert len(eligible) == 72
    compatible_costs = {(r["regime"], r["profile"], r["role"], r["name"]): r for r in costs
                        if r["family"] in ("LGA", "native-MC") or r in eligible}
    gate_details, amortization_keys, descriptive = [], set(), defaultdict(list)
    for row in amortization:
        case, profile, role = (row[field] for field in ("regime", "profile", "role"))
        key = (case, profile, role, row["candidate"], row["control"], row["statistic"])
        assert key not in amortization_keys
        amortization_keys.add(key)
        candidate, control = (compatible_costs[case, profile, role, row[field]] for field in ("candidate", "control"))
        assert candidate["family"] == "repaired" and control["name"] in ("LGA_dt7", "LGA_dt10", "native-MC-dt10")
        if role == "conditional":
            assert int(candidate["batch"]) == 1
            assert row["scope"].startswith("new supplied-state queries")
        else:
            assert role == "ensemble" and int(candidate["batch"]) == 16384
            assert "fresh estimates of the same fixed IID law" in row["scope"] and "memoized" in row["scope"]
        assert "fixed physics/bank/horizons" in row["scope"] and "continuous stopping bias unresolved" in row["scope"]
        same = lambda r: (r["regime"], r["profile"], r["role"]) == (case, profile, role)
        refinement = [r for r in tables["reference_refinement"] if same(r) and r["level"] == "level0_versus_finest"]
        assert len(refinement) == (6 if role == "conditional" else 3)
        evaluated = []
        for tag, cost in (("candidate", candidate), ("control", control)):
            pool = [r for r in tables["accuracy"] if same(r) and r["name"] == cost["name"]]
            uncertainty = [r for r in tables["sampling"] if same(r) and r["name"] == cost["name"]]
            assert len(pool) == len(refinement) == len(uncertainty)
            reason, details = gate(pool, uncertainty, refinement, row["statistic"])
            assert row[tag+"_gate"] == reason
            evaluated.append(reason)
            gate_details.append({"regime": case, "profile": profile, "role": role, "name": cost["name"],
                                 "statistic": row["statistic"], "reported_gate": reason, **details})
        saving = float(control["median_seconds"])-float(candidate["median_seconds"])
        near(row["online_saving_seconds"], saving, "descriptive saving")
        expected = None
        if saving > 0:
            extra = float(candidate["known_offline_seconds"])+float(candidate["first_prediction_seconds"])-float(control["known_offline_seconds"])-float(control["first_prediction_seconds"])
            expected = max(1, math.ceil(1+extra/saving))
            # Verify the integer crossover independently in direct total-cost units.
            at_cross = extra-(expected-1)*saving
            assert at_cross <= 1e-9, (key, expected, at_cross)
            if expected > 1:
                assert extra-(expected-2)*saving > -1e-9
            descriptive[role, control["name"]].append(expected)
        assert number(row["descriptive_payback_calls"]) == expected
        qualified = expected if all(r == "finite-reference-diagnostic-pass" for r in evaluated) else None
        assert number(row["accuracy_qualified_payback_calls"]) == qualified
    expected_keys = {(c["regime"], c["profile"], c["role"], c["name"], name, statistic)
                     for c in eligible for name in ("LGA_dt7", "LGA_dt10", "native-MC-dt10") for statistic in ("mean", "covariance")}
    assert amortization_keys == expected_keys
    qualified_count = sum(number(r["accuracy_qualified_payback_calls"]) is not None for r in amortization)
    assert qualified_count == 0
    assert sum(boolean(r["all_seeds_NN_closer_pointwise95"]) for r in consistency_index.values()) == 0
    assert sum(boolean(r["all_seeds_LGA_closer_pointwise95"]) for r in consistency_index.values()) == 57
    near(sum(float(r["training_seconds"]) for r in tables["training"]),
         lock["total_recorded_model_training_seconds"], "all-seed recorded fitting cost")
    for row in tables["training"]:
        near(row["training_seconds"], fits[row["regime"], row["name"]]["training_seconds"], "fit identity")
    for row in tables["study_costs"]:
        near(row["all_seed_fit_seconds"], sum(m["training_seconds"] for m in lock["models"] if m["case"] == row["regime"]), "study fitting durations")
        near(row["shared_state_generation_seconds"], data_seconds[row["regime"]], "study shared data")
        manifest = read(ROOT/f"reference/{row['regime']}/manifest.json")
        near(row["reference_generation_seconds"], manifest["generation_seconds"], "study reference recorded durations")
    result = {
        "complete": True, "kind": "independent-scalar-only-cost-gate-report-audit-v1",
        "source_hash": package_hash, "script_sha256": sha(__file__), "inputs": INPUTS,
        "report_manifest_deliberately_unbound": True,
        "no_model_inference_or_outcome_array_access": True, "no_bootstrap_draw_or_replay": True,
        "no_refitting_or_selection": True,
        "counts": {"timing_records": len(timing["records"]), "cost_rows": len(costs), "total_cost_rows": len(totals),
                   "amortization_rows": len(amortization), "scalar_accuracy_rows": len(accuracy_index),
                   "independent_NN_ensemble_replacements": replacements,
                   "independently_audited_NN_accuracy_records": len(independent_replays),
                   "reference_refinement_rows": len(refinement_index), "seed_consistency_rows": len(consistency_index),
                   "accuracy_qualified_payback_rows": qualified_count, "all_seed_NN_closer_cells": 0, "all_seed_LGA_closer_cells": 57},
        "verified_accounting": ["One declared model fit per fitted deployment, every seed retained in study costs",
            "Full case training and validation data charged to each fitted deployment",
            "Per-profile model restoration or untrained constructor time retained",
            "B1 cold first query charged directly; B64/B256/ensemble pay discarded B1 cold probe once",
            "Native historical cold compilation plus current setup, without current compile double counting",
            "First query plus Q-1 measured median warm queries for Q=1,100,10000,100000",
            "Both Gaussian integration steps compared at matching regime, initial law and query role",
            "Every declared mean/covariance 5% and uncertainty/refinement 1.25% gate recomputed"],
        "gate_reason_counts": dict(Counter((r["candidate_gate"], r["control_gate"]) for r in amortization)),
        "candidate_gate_counts": dict(Counter(r["candidate_gate"] for r in amortization)),
        "control_gate_counts": dict(Counter(r["control_gate"] for r in amortization)),
        "descriptive_payback_ranges": [{"role": role, "control": control, "minimum_calls": min(values), "maximum_calls": max(values),
                                         "accuracy_qualified": False} for (role, control), values in sorted(descriptive.items())],
        "gate_ingredients": gate_details,
        "recorded_all_model_fitting_seconds": lock["total_recorded_model_training_seconds"],
        "scalar_comparisons": CHECKS, "maximum_absolute_scalar_difference": MAX_ERROR,
        "limitations": ["Conditional accuracy was tested at two fixed anchors per profile; new-state timing is not an accuracy certificate for arbitrary histograms.",
            "Ensemble deployment costs are fresh estimates of one fixed IID law, not novel initial-law generalization; identical answers can be memoized.",
            "Descriptive payback does not establish equal-accuracy advantage. No strict accuracy-qualified payback is resolved here.",
            "Finite reference refinements and pointwise sampling intervals do not enclose continuous boundary-hitting bias.",
            "Sum of overlapping worker fitting durations is not elapsed batch time; measured data/reference durations are study components.",
            "No numerical model, bootstrap or timing computation was rerun in this audit."],
        "audit_cpu_seconds": time.process_time()-cpu, "audit_wall_seconds": time.perf_counter()-started,
    }
    # JSON object keys must be strings even for the paired gate reason inventory.
    result["gate_reason_counts"] = {str(key): value for key, value in result["gate_reason_counts"].items()}
    output = ROOT/"audit/cost_report_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("complete", "counts", "scalar_comparisons", "maximum_absolute_scalar_difference", "audit_cpu_seconds", "audit_wall_seconds")}, indent=2))
    print("audit_sha256", sha(output))


if __name__ == "__main__":
    main()
