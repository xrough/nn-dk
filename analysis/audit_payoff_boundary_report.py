"""Independent replay of deployment costs, accuracy gates and all-seed claims.

The producer's reporting functions are not imported. Literal benchmark samples,
locked fits, verified historical records and raw accuracy packets determine the
expected tables. A final report manifest pins every consumed byte and excludes
this audit's output to avoid a circular checksum dependency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402

from analysis import payoff_legacy_control as legacy  # noqa: E402
from analysis import run_payoff_boundary_regimes as runner  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402

KIND = "independent-boundary-report-cost-claim-replay-v1"
AUDIT_NAME = "report_claims_cost_replay.json"
NUMERIC_ACCURACY = (
    "max_mean_error_in_reference_std", "max_normalized_covariance_error",
    "mean_sampling95_simultaneous_radius", "covariance_sampling95_simultaneous_radius",
    "mean_error_plus_sampling95", "covariance_error_plus_sampling95",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, *, positive=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError("Finite nonnegative recorded costs required")
    return value


def assert_equal(actual, expected, label):
    if expected is None:
        equal = actual is None or actual == ""
    elif isinstance(expected, bool):
        equal = actual == expected or actual == str(expected)
    elif isinstance(expected, (float, int, np.number)):
        try:
            equal = math.isclose(float(actual), float(expected), rel_tol=2e-11, abs_tol=2e-12)
        except (ValueError, TypeError):
            equal = False
    else:
        equal = actual == expected
    if not equal:
        raise ValueError(f"Independent report replay mismatch: {label}: {actual!r} != {expected!r}")


def compare_table(expected, actual, keys, *, label):
    """Compare every independently reconstructed field; allow producer extras."""
    def index(rows):
        result = {}
        for row in rows:
            key = tuple(str(row[name]) for name in keys)
            if key in result:
                raise ValueError(f"Duplicate {label} table identity")
            result[key] = row
        return result

    first, second = index(expected), index(actual)
    if set(first) != set(second):
        raise ValueError(f"Incomplete or undeclared {label} table inventory")
    for key, row in first.items():
        for field, value in row.items():
            if field not in second[key]:
                raise ValueError(f"Missing {label} field: {field}")
            assert_equal(second[key][field], value, f"{label}/{key}/{field}")


def replay_costs(timing, new_models, historical_models, new_data, historical_data,
                 profiles, seeds, n_paths, query_counts=(1, 100, 10000, 100000)):
    """Rebuild literal cold/warm deployment totals without producer functions."""
    seeds = tuple(seeds)
    if seeds != (0, 1, 2):
        raise ValueError("Every declared optimizer seed 0, 1, 2 must be retained")
    fit_maps, names = {}, {}
    for family, models, pool in (("repaired", new_models, new_data),
                                  ("legacy-v2", historical_models, historical_data)):
        seen = set()
        for model in models:
            case, seed = model["case"], model["seed"]
            if case not in profiles or seed not in seeds or (case, seed) in seen:
                raise ValueError("Incomplete or duplicated model/seed inventory")
            seen.add((case, seed))
            key = case, model["name"]
            fit_maps[key] = (finite(model["training_seconds"], positive=True),
                             finite(pool[case]), "new-v3" if family == "repaired" else "historical-v2")
            names[key] = (family, seed)
        if seen != {(case, seed) for case in profiles for seed in seeds}:
            raise ValueError("A model cost cohort dropped an optimizer seed")
    neural = dict(names)
    for case in profiles:
        neural[case, case+"_untrained"] = ("untrained-repaired", 0)
    loads = {}
    for row in timing["loading"]:
        key = row["regime"], row["profile"], row["name"]
        if key in loads:
            raise ValueError("Duplicate model restoration timing")
        loads[key] = finite(row["seconds"], positive=True)
    expected_loads = {(case, profile, name) for case, name in neural
                      for profile in profiles[case]}
    if set(loads) != expected_loads:
        raise ValueError("All new, legacy and untrained model loads are required")
    expected, observed = {}, {}
    for case, profile_names in profiles.items():
        for profile in profile_names:
            for (regime, name), (family, seed) in neural.items():
                if regime != case:
                    continue
                for role, batch in (("conditional", 1), ("conditional", 64),
                                    ("conditional", 256), ("ensemble", n_paths)):
                    expected[case, profile, name, role, batch] = (family, seed)
            for name in ("LGA_dt7", "LGA_dt10"):
                for role in ("conditional", "ensemble"):
                    expected[case, profile, name, role, 1] = ("LGA", None)
            for role in ("conditional", "ensemble"):
                expected[case, profile, "native-MC-dt10", role, n_paths] = ("native-MC", None)
    for row in timing["records"]:
        key = tuple(row[name] for name in ("regime", "profile", "name", "role", "batch"))
        if key in observed:
            raise ValueError("Duplicate literal inference timing")
        observed[key] = row
    if set(observed) != set(expected):
        raise ValueError("Literal timing inventory dropped a seed, role or batch")
    probes = {key[:3]: finite(value["first_prediction_seconds"], positive=True)
              for key, value in observed.items()
              if key[3:] == ("conditional", 1) and value["family"] in
              ("repaired", "legacy-v2", "untrained-repaired")}
    compiler = timing["compiled_control"]
    compile_cost = finite(compiler["compile_seconds"])
    setup = max(0., finite(compiler["setup_seconds_this_process"])
                - finite(compiler["compile_seconds_this_process"]))
    costs, totals = [], []
    for key, measured in observed.items():
        case, profile, name, role, batch = key
        family, seed = expected[key]
        if measured["family"] != family or (seed is not None and measured["seed"] != seed):
            raise ValueError("Literal timing method/optimizer identity changed")
        samples = np.asarray(measured["samples_seconds"], dtype=float)
        if (samples.ndim != 1 or len(samples) != measured["repeats"]
                or not np.isfinite(samples).all() or np.any(samples <= 0)):
            raise ValueError("Complete finite positive benchmark repetitions required")
        repetitions, warmup = ((3, 1) if family == "native-MC" else
                               (5, 1) if role == "ensemble" and family != "LGA" else (21, 3))
        if (len(samples), measured["warmup"]) != (repetitions, warmup):
            raise ValueError("Preregistered literal benchmark repetition allocation changed")
        for field, value in (("median_seconds", np.median(samples)), ("minimum_seconds", samples.min()),
                             ("maximum_seconds", samples.max()), ("p10_seconds", np.quantile(samples, .1)),
                             ("p90_seconds", np.quantile(samples, .9)), ("p95_seconds", np.quantile(samples, .95))):
            assert_equal(measured[field], value, "raw-timing/"+field)
        fit, pool, origin = fit_maps.get((case, name), (0., 0., "none"))
        restoration = loads.get(key[:3], 0.)
        probe = probes[key[:3]] if (case, name) in neural and (role, batch) != ("conditional", 1) else 0.
        cold_compile, cold_setup = (compile_cost, setup) if family == "native-MC" else (0., 0.)
        offline = fit+pool+restoration+probe+cold_compile+cold_setup
        first, warm = finite(measured["first_prediction_seconds"], positive=True), float(np.median(samples))
        row = {"slot": measured["slot"], "regime": case, "profile": profile, "name": name,
               "family": family, "role": role, "batch": batch,
               "median_seconds": warm, "first_prediction_seconds": first,
               "training_seconds": fit, "shared_data_seconds": pool,
               "restoration_seconds": restoration, "cold_compilation_seconds": cold_compile,
               "compiled_setup_seconds": cold_setup, "cache_creation_B1_probe_seconds": probe,
               "known_offline_seconds": offline, "offline_data_source": origin}
        costs.append(row)
        for count in query_counts:
            totals.append({**{k: row[k] for k in ("slot", "regime", "profile", "name", "family", "role", "batch")},
                "query_calls": count, "known_offline_seconds": offline,
                "first_prediction_seconds": first, "subsequent_warm_seconds": (count-1)*warm,
                "total_known_seconds": offline+first+(count-1)*warm})
    return costs, totals


def gate(accuracy, sampling, refinement, statistic):
    """Independent implementation of the fixed 5%/1.25% diagnostic contract."""
    if not accuracy or not refinement:
        return "unresolved-inventory"
    error = "max_mean_error_in_reference_std" if statistic == "mean" else "max_normalized_covariance_error"
    upper, radius = statistic+"_error_plus_sampling95", statistic+"_sampling95_simultaneous_radius"
    if max(float(row[error]) for row in accuracy) > .05:
        return "point-error-above-5pct"
    if max(float(row[upper]) for row in accuracy)+max(float(row[upper]) for row in refinement) > .05:
        return "error-plus-uncertainty-unresolved"
    if max(float(row[radius]) for row in accuracy) > .0125:
        return "sampling-component-above-1.25pct"
    if max(float(row[upper]) for row in refinement) > .0125:
        return "reference-refinement-component-unresolved"
    if sampling and max(float(row[statistic+"95_radius"]) for row in sampling) > .0125:
        return "standalone-MC-component-unresolved"
    return "finite-reference-diagnostic-pass"


def replay_amortization(costs, accuracy, sampling, refinement):
    rows = []
    for candidate in costs:
        if candidate["family"] != "repaired" or (candidate["role"] == "conditional" and candidate["batch"] != 1):
            continue
        same = {key: candidate[key] for key in ("regime", "profile", "role")}
        def compatible(row):
            return all(row[key] == value for key, value in same.items())
        ref = [row for row in refinement if compatible(row) and row["level"] == "level0_versus_finest"]
        seed = int(candidate["name"].rsplit("seed", 1)[1])
        for control in costs:
            if not compatible(control) or control["family"] not in ("LGA", "native-MC", "legacy-v2"):
                continue
            if control["family"] == "legacy-v2" and (
                    int(control["name"].rsplit("seed", 1)[1]) != seed
                    or (control["role"] == "conditional" and control["batch"] != 1)):
                continue
            saving = control["median_seconds"]-candidate["median_seconds"]
            arithmetic = None
            if saving > 0:
                difference = (candidate["known_offline_seconds"]+candidate["first_prediction_seconds"]
                              -control["known_offline_seconds"]-control["first_prediction_seconds"])
                arithmetic = max(1, math.ceil(1+difference/saving))
            for statistic in ("mean", "covariance"):
                gates = [gate([row for row in accuracy if compatible(row) and row["name"] == item["name"]],
                              [row for row in sampling if compatible(row) and row["name"] == item["name"]],
                              ref, statistic) for item in (candidate, control)]
                rows.append({**same, "candidate": candidate["name"], "control": control["name"],
                    "statistic": statistic, "candidate_gate": gates[0], "control_gate": gates[1],
                    "online_saving_seconds": saving, "descriptive_payback_calls": arithmetic,
                    "accuracy_qualified_payback_calls": arithmetic if all(
                        value == "finite-reference-diagnostic-pass" for value in gates) else None})
    return rows


def summarize_contrasts(rows, seeds):
    """All fixed seeds must appear once per group/horizon/statistic."""
    groups = defaultdict(list)
    fields = ("regime", "profile", "role", "group", "time_index", "time", "statistic")
    for row in rows:
        groups[tuple(row[key] for key in fields)].append(row)
    output = []
    for key, entries in sorted(groups.items()):
        actual_seeds = [int(row["name"].rsplit("seed", 1)[1]) for row in entries]
        if len(entries) != len(seeds) or set(actual_seeds) != set(seeds):
            raise ValueError("An accuracy claim dropped or duplicated an optimizer seed")
        output.append({**dict(zip(fields, key, strict=True)), "optimizer_seeds": str(tuple(seeds)),
            "minimum_advantage": min(row["advantage"] for row in entries),
            "maximum_advantage": max(row["advantage"] for row in entries),
            "minimum_lower95": min(row["lower95"] for row in entries),
            "maximum_upper95": max(row["upper95"] for row in entries),
            "all_seeds_NN_closer_pointwise95": all(row["lower95"] > 0 for row in entries),
            "all_seeds_LGA_closer_pointwise95": all(row["upper95"] < 0 for row in entries)})
    return output


def worker_accounting(rows):
    """Keep active recorded durations separate from calendar timestamp spans."""
    by_seed = defaultdict(float)
    for row in rows:
        by_seed[row["seed"]] += row["wall_seconds"]
    return {"sum_CPU_seconds": sum(row["cpu_seconds"] for row in rows),
            "sum_worker_wall_seconds": sum(row["wall_seconds"] for row in rows),
            "longest_worker_wall_seconds": max(by_seed.values()),
            "UTC_start_finish_span_seconds": max(row["finished_unix"] for row in rows)
                -min(row["started_unix"] for row in rows)}


def _csv(root, name):
    with (root/"report"/(name+".csv")).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _pools(data_lock, inputs):
    totals, records = {}, []
    for case, banks in data_lock["banks"].items():
        if set(banks) != {"train", "validation"}:
            raise ValueError("Both training and validation pool costs required")
        totals[case] = 0.
        for role, entry in banks.items():
            path = Path(entry["metadata_path"])
            if sha(path) != entry["metadata_sha256"]:
                raise ValueError("A locked state-pool generation manifest changed")
            inputs[str(path.resolve())] = entry["metadata_sha256"]
            packet = read(path)
            seconds = finite(packet["generation_seconds"])
            assert_equal(seconds, entry["generation_seconds"], "pool generation duration")
            totals[case] += seconds
            records.append({"regime": case, "role": role, "generation_seconds": seconds})
    return totals, records


def _workers(root, seeds, *, historical, inputs):
    rows = []
    for seed in seeds:
        relative = f"payoffs/execution/worker_seed{seed}.json"
        path = root/relative
        worker = read(path)
        if historical and worker != legacy._archived_json(relative):
            raise ValueError("Historical worker records differ from their archived bytes")
        inputs[str(path.resolve())] = sha(path)
        if not worker.get("complete"):
            raise ValueError("A fitting worker is incomplete")
        for index, attempt in enumerate(worker["attempts"]):
            rows.append({"seed": seed, "attempt": index,
                         **{field: attempt[field] for field in (
                             "wall_seconds", "cpu_seconds", "started_unix", "finished_unix")}})
    return rows


def _raw_tables(root, study, inputs):
    accuracy, sampling, refinements = [], [], []
    contrasts = {"contrasts": [], "legacy_contrasts": []}
    expected_methods = 2*len(study.optimizer_seeds)+4
    for case in runner.case_names(study):
        cfg = runner.reference_configuration(case, study)
        for group in cfg.group_keys:
            profile, role, anchor = group.split("__")
            original = read(root/"comparison/accuracy"/case/group/"group.json")
            primary = read(root/("deployment" if role == "ensemble" else "comparison")
                           /"accuracy"/case/group/"group.json")
            neural_names = {f"{case}_boundary_seed{seed}" for seed in study.optimizer_seeds}
            neural_names |= {f"{case}_legacy_v2_seed{seed}" for seed in study.optimizer_seeds}
            neural_names.add(case+"_untrained")
            records = {read(item["path"])["identity"]["name"]: item for item in original["records"]}
            if role == "ensemble":
                replacements = {}
                for entry in primary["records"]:
                    packet = read(entry["path"])
                    if (not packet.get("initial_pool_independent_of_reference")
                            or not packet.get("reference_and_initial_bootstrap_independent")):
                        raise ValueError("Every primary neural ensemble must use independent initial integration")
                    replacements[packet["identity"]["name"]] = entry
                if set(replacements) != neural_names or len(primary["records"]) != len(neural_names):
                    raise ValueError("A primary ensemble omitted a new, historical or untrained head")
                records.update(replacements)
            if (len(records) != expected_methods
                    or set(records) != neural_names | {"LGA_dt7", "LGA_dt10", "native-MC-dt10"}):
                raise ValueError("A primary accuracy group omitted a new, historical or classical method")
            for name, entry in records.items():
                path = Path(entry["path"])
                if sha(path) != entry["sha256"]:
                    raise ValueError("A primary accuracy record changed")
                inputs[str(path.resolve())] = entry["sha256"]
                packet = read(path)
                values = packet["metrics"]["records"]
                if {row["time_index"] for row in values} != {0, 1, 2} or len(values) != 3:
                    raise ValueError("A primary accuracy record omitted a declared physical horizon")
                origin = {"regime": case, "profile": profile, "role": role, "group": group,
                          "name": name, "family": packet["method"]["family"]}
                array_path = Path(packet["arrays"]["path"])
                if sha(array_path) != packet["arrays"]["sha256"]:
                    raise ValueError("A primary moment/bootstrap array changed")
                inputs[str(array_path.resolve())] = packet["arrays"]["sha256"]
                with np.load(array_path, allow_pickle=False) as saved:
                    if "bootstrap_mean" in saved:
                        std = np.sqrt(np.diagonal(np.asarray([row["reference_scaled_covariance"]
                            for row in values])/cfg.n_particles, axis1=-2, axis2=-1))
                        mean_radius = np.quantile(np.max(abs(saved["bootstrap_mean"]-saved["mean"])
                                                       /std, axis=-1), .95, axis=0)
                        cov_radius = np.quantile(np.max(abs(saved["bootstrap_scaled_covariance"]
                            -saved["scaled_covariance"])/cfg.n_particles
                            /(std[..., :, None]*std[..., None, :]), axis=(-2, -1)), .95, axis=0)
                    else:
                        mean_radius, cov_radius = np.zeros(3), np.zeros(3)
                for row in values:
                    t = row["time_index"]
                    base = {**origin, "time_index": t, "time": cfg.query_times[t]}
                    accuracy.append({**base, **{key: row[key] for key in NUMERIC_ACCURACY}})
                    sampling.append({**base, "mean95_radius": mean_radius[t], "covariance95_radius": cov_radius[t]})
            base = {"regime": case, "profile": profile, "role": role, "group": group}
            for level, value in original["reference_refinement"].items():
                for row in value["records"]:
                    refinements.append({**base, "level": level, "time_index": row["time_index"],
                        "time": cfg.query_times[row["time_index"]],
                        **{key: row[key] for key in NUMERIC_ACCURACY}})
            for table, field in (("contrasts", "NN_versus_fine_LGA"),
                                 ("legacy_contrasts", "NN_versus_legacy_v2")):
                bank = primary[field]
                if set(bank) != {f"{case}_boundary_seed{seed}" for seed in study.optimizer_seeds}:
                    raise ValueError("A raw comparison omitted a fixed optimizer seed")
                for name, value in bank.items():
                    for row in value["records"]:
                        for statistic in ("mean", "covariance"):
                            draws = np.asarray(row[f"bootstrap_{statistic}_advantages"])
                            if draws.shape != (128,) or not np.isfinite(draws).all():
                                raise ValueError("Complete pointwise whole-family contrast bootstrap required")
                            interval = np.quantile(draws, [.025, .975])
                            for actual, expected in zip(row[statistic+"_advantage_percentile95"], interval, strict=True):
                                assert_equal(actual, expected, "contrast percentile")
                            contrasts[table].append({**base, "name": name, "time_index": row["time_index"],
                                "time": cfg.query_times[row["time_index"]], "statistic": statistic,
                                "advantage": row[statistic+"_error_advantage"],
                                "lower95": interval[0], "upper95": interval[1],
                                "NN_closer_pointwise95": bool(interval[0] > 0),
                                "LGA_closer_pointwise95": bool(interval[1] < 0)})
    return {"accuracy": accuracy, "sampling": sampling, "reference_refinement": refinements, **contrasts}


def run(root, *, require_full_design=True):
    root = Path(root).resolve()
    started, cpu = time.monotonic(), time.process_time()
    report_path = root/"report/manifest.json"
    manifest = read(report_path)
    own_output = root/"audit"/AUDIT_NAME
    if str(own_output) in manifest["inputs"]:
        raise ValueError("The final audit must be excluded from report inputs to avoid a checksum cycle")
    if not manifest.get("complete") or manifest["source_hash"] != source_hash():
        raise ValueError("A complete frozen final report is required")
    if manifest["generator_sha256"] != sha(REPO/"analysis/report_payoff_boundary_regimes.py"):
        raise ValueError("The report generator changed after its final manifest")
    for path, expected in manifest["inputs"].items():
        if sha(path) != expected:
            raise ValueError(f"A report input changed: {path}")
    for filename, expected in manifest["outputs"].items():
        if sha(root/"report"/filename) != expected:
            raise ValueError(f"A frozen report output changed: {filename}")
    study, design = runner._assert_frozen(root)
    cases, seeds = runner.case_names(study), study.optimizer_seeds
    if require_full_design and (len(cases) != 6 or tuple(seeds) != (0, 1, 2)):
        raise ValueError("The scientific audit requires six regimes and all three optimizer seeds")
    lock = runner.verify_fit_lock(root)
    controls = read(root/"comparison/legacy_controls.json")
    if controls != legacy.inventory(cases, seeds):
        raise ValueError("Historical cost controls differ from their verified original inventory")
    legacy.verify_targets(controls, runner, study)
    inputs = {}
    data, data_rows = _pools(read(root/"payoffs/data_lock.json"), inputs)
    historical_data, historical_rows = _pools(legacy._archived_json("payoffs/data_lock.json"), inputs)
    historical_data = {case: historical_data[case] for case in cases}
    historical_rows = [row for row in historical_rows if row["regime"] in cases]
    workers = _workers(root, seeds, historical=False, inputs=inputs)
    historical_workers = _workers(legacy.HISTORICAL, seeds, historical=True, inputs=inputs)
    timing = read(root/"comparison/timing.json")
    if not timing.get("complete") or not timing.get("final_timing_after_all_workers_finished"):
        raise ValueError("Complete uncontended final timings required")
    profiles = {case: runner.configuration(case, study).cases for case in cases}
    costs, totals = replay_costs(timing, lock["models"], controls["models"], data, historical_data,
                                 profiles, seeds, study.paths_per_group)
    groups = sum(len(runner.reference_configuration(case, study).group_keys) for case in cases)
    for field, expected in (("models", len(lock["models"])), ("historical_control_models", len(controls["models"])),
                            ("accuracy_records", groups*(2*len(seeds)+4)),
                            ("original_accuracy_records", groups*(2*len(seeds)+4)),
                            ("independent_initial_records", len(cases)*2*(2*len(seeds)+1)),
                            ("total_retained_accuracy_records", groups*(2*len(seeds)+4)
                                +len(cases)*2*(2*len(seeds)+1)),
                            ("timing_records", len(costs)), ("timing_restorations", len(timing["loading"])),
                            ("total_cost_rows", len(totals)), ("all_optimizer_seeds_retained", True),
                            ("primary_NN_ensemble_initials_independent", True),
                            ("test_based_refitting", False), ("continuous_DK_claim", False),
                            ("new_fit_seconds", sum(row["training_seconds"] for row in lock["models"])),
                            ("historical_fit_seconds", sum(row["training_seconds"] for row in controls["models"]))):
        assert_equal(manifest[field], expected, "report manifest/"+field)
    compare_table(costs, _csv(root, "costs"), ("slot",), label="costs")
    compare_table(totals, _csv(root, "total_costs"), ("slot", "query_calls"), label="total costs")
    for table, rows in (("data", data_rows), ("historical_data", historical_rows)):
        compare_table(rows, _csv(root, table), ("regime", "role"), label=table)
    for table, rows in (("workers", workers), ("historical_workers", historical_workers)):
        compare_table(rows, _csv(root, table), ("seed", "attempt"), label=table)
    for table, models in (("training", lock["models"]), ("historical_training", controls["models"])):
        rows = [{"regime": row["case"], "name": row["name"], "seed": row["seed"],
                 "training_seconds": row["training_seconds"]} for row in models]
        compare_table(rows, _csv(root, table), ("regime", "name"), label=table)
    study_costs = [{"regime": case,
        "all_new_seed_fit_seconds": sum(row["training_seconds"] for row in lock["models"] if row["case"] == case),
        "historical_all_seed_fit_seconds": sum(row["training_seconds"] for row in controls["models"] if row["case"] == case),
        "new_shared_state_generation_seconds": data[case],
        "historical_shared_state_generation_seconds": historical_data[case],
        "reference_generation_seconds": read(root/"reference"/case/"manifest.json")["generation_seconds"]}
        for case in cases]
    compare_table(study_costs, _csv(root, "study_costs"), ("regime",), label="study costs")
    tables = _raw_tables(root, study, inputs)
    for table in ("accuracy", "sampling", "reference_refinement", "contrasts", "legacy_contrasts"):
        key = ("regime", "group", "time_index", "statistic", "name") if "contrasts" in table else (
            ("regime", "group", "level", "time_index") if table == "reference_refinement"
            else ("regime", "group", "name", "time_index"))
        compare_table(tables[table], _csv(root, table), key, label=table)
    summaries = {}
    for source, table in (("contrasts", "seed_consistency"), ("legacy_contrasts", "legacy_seed_consistency")):
        summaries[table] = summarize_contrasts(tables[source], seeds)
        compare_table(summaries[table], _csv(root, table),
                      ("regime", "group", "time_index", "statistic"), label=table)
    amortization = replay_amortization(costs, tables["accuracy"], tables["sampling"], tables["reference_refinement"])
    compare_table(amortization, _csv(root, "amortization"),
                  ("regime", "profile", "role", "candidate", "control", "statistic"), label="amortization")
    qualified = sum(row["accuracy_qualified_payback_calls"] is not None for row in amortization)
    assert_equal(manifest["qualified_payback_rows"], qualified, "qualified payback count")
    if require_full_design:
        for actual, expected in ((len(timing["loading"]), 84), (len(costs), 408), (len(totals), 1632),
                                 (len(summaries["seed_consistency"]), 216),
                                 (len(summaries["legacy_seed_consistency"]), 216)):
            assert_equal(actual, expected, "fixed full-study table count")
    result = {"kind": KIND, "complete": True, "source_hash": source_hash(),
        "script_sha256": sha(__file__), "report_manifest_sha256": sha(report_path),
        "report_generator_sha256": manifest["generator_sha256"], "additional_inputs": inputs,
        "fit_lock_sha256": sha(root/"payoffs/fit_lock.json"),
        "legacy_controls_sha256": sha(root/"comparison/legacy_controls.json"),
        "timing_records": len(costs), "model_loads": len(timing["loading"]),
        "cost_rows": len(costs), "total_cost_rows": len(totals),
        "all_new_and_historical_optimizer_seeds_retained": True,
        "accuracy_gates_independently_replayed": True, "qualified_payback_rows": qualified,
        "amortization_rows": len(amortization),
        "all_seed_comparisons": {name: {"rows": len(rows),
            "all_seeds_new_closer_pointwise95": sum(row["all_seeds_NN_closer_pointwise95"] for row in rows),
            "all_seeds_control_closer_pointwise95": sum(row["all_seeds_LGA_closer_pointwise95"] for row in rows)}
            for name, rows in summaries.items()},
        "new_worker_accounting": worker_accounting(workers),
        "historical_worker_accounting": worker_accounting(historical_workers),
        "scope": "Pointwise intervals per fixed seed/horizon; all-seed consistency is not a family-wise certificate. "
                 "Costs concern matched tasks; refinement gates do not certify a continuous stopping law.",
        "software_fixture_only": not require_full_design, "no_fitting_refitting_selection_or_sampling": True,
        "audit_wall_seconds": time.monotonic()-started, "audit_CPU_seconds": time.process_time()-cpu}
    if own_output.exists():
        raise FileExistsError("Preserve the earlier independent report audit")
    own_output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.root)
    print(json.dumps({key: result[key] for key in (
        "complete", "timing_records", "model_loads", "cost_rows", "total_cost_rows",
        "qualified_payback_rows", "all_seed_comparisons")}))


if __name__ == "__main__":
    main()
