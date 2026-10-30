"""Verified harder-regime accuracy, stopping, and deployment-cost report.

This is postprocessing only. It neither chooses fits nor extends test allocations.
Every optimizer seed and unresolved accuracy/cost gate remains visible.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from analysis import run_payoff_regimes as runner  # noqa: E402
from analysis.evaluate_payoff_regimes import prepare as comparison_plan  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402

KIND = "harder-regime-verified-report-v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _artifact(path, inputs, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Changed scientific evidence: {path}")
    inputs[str(path)] = actual
    return path


def load_complete(root, *, independent_initials=True):
    root = Path(root).resolve()
    study = runner.load_plan(root)
    plan = comparison_plan(root)
    lock = runner.verify_fit_lock(root)
    index = read(root/"comparison/accuracy_index.json")
    timing = read(root/"comparison/timing.json")
    plan_sha, lock_sha = sha(root/"comparison/preregistration.json"), sha(root/"payoffs/fit_lock.json")
    if (not index.get("complete") or not timing.get("complete")
            or not timing.get("final_timing_after_all_workers_finished")
            or len(lock["models"]) != len(runner.slots(study))
            or timing.get("identity") != {"plan_sha256": plan_sha, "fit_lock_sha256": lock_sha, "source_hash": source_hash()}
            or index.get("plan_sha256") != plan_sha or index.get("source_hash") != source_hash()):
        raise ValueError("All declared fits, independent accuracy and final timings are required")
    inputs, groups, records, references = {}, [], [], {}
    for name in ("plan.json", "payoffs/preregistration.json", "payoffs/data_lock.json",
                 "payoffs/fit_lock.json", "comparison/preregistration.json",
                 "comparison/accuracy_index.json", "comparison/timing.json"):
        _artifact(root/name, inputs)
    for regime in plan["cases"]:
        manifest_path = _artifact(root/f"reference/{regime}/manifest.json", inputs)
        manifest = read(manifest_path)
        if manifest.get("status") != "complete":
            raise ValueError("Incomplete native reference")
        _artifact(root/f"reference/{regime}/reference.h5", inputs, manifest["reference_sha256"])
        references[regime] = manifest
    expected = {(c, k) for c in plan["cases"]
                for k in runner.reference_configuration(c, study).group_keys}
    if {tuple(pair) for pair in index["groups"]} != expected:
        raise ValueError("The full fixed regime/group inventory is required")
    for regime, key in sorted(expected):
        path = _artifact(root/f"comparison/accuracy/{regime}/{key}/group.json", inputs)
        group = read(path)
        base = {"source_hash": source_hash(), "comparison_plan_sha256": plan_sha,
                "fit_lock_sha256": lock_sha, "regime": regime,
                "reference_sha256": references[regime]["reference_sha256"],
                "bank_hash": runner.payoff_bank(regime, study).bank_hash}
        if (not group.get("complete") or len(group["records"]) != len(study.optimizer_seeds)+4
                or group.get("identity") != base or group.get("group") != key):
            raise ValueError("Every group must retain all seven method records")
        groups.append({**group, "regime": regime})
        families = []
        for item in group["records"]:
            packet = read(_artifact(item["path"], inputs, item["sha256"]))
            if (not packet.get("complete") or packet["identity"] != {
                    **base, "group": key, "name": packet["method"]["name"]}):
                raise ValueError("Invalid method record")
            _artifact(packet["arrays"]["path"], inputs, packet["arrays"]["sha256"])
            families.append((packet["method"]["family"], packet["method"].get("seed")))
            records.append(packet)
        if {seed for family, seed in families if family == "repaired"} != set(study.optimizer_seeds):
            raise ValueError("No optimizer seed may be discarded")
    for item in lock["models"]:
        for key in ("checkpoint", "metadata"):
            _artifact(item[f"{key}_path"], inputs, item[f"{key}_sha256"])
    for path in (root/"audit").glob("*.json"):
        _artifact(path, inputs)
    evidence = {"root": root, "study": study, "plan": plan, "lock": lock, "timing": timing,
                "groups": groups, "records": records, "references": references, "inputs": inputs,
                "original_accuracy_records": len(records), "paired_initial_records": [], "paired_initial_groups": []}
    if independent_initials:
        _independent_initial_primary(evidence)
    return evidence


def _independent_initial_primary(evidence):
    """Retain coupled diagnostics; use fresh initial integration for deployment."""
    from analysis import evaluate_payoff_regimes_deployment as deployment
    root, inputs, study = evidence["root"], evidence["inputs"], evidence["study"]
    declared = deployment.prepare(root)
    plan_path = _artifact(root/"deployment/preregistration.json", inputs)
    index = read(_artifact(root/"deployment/accuracy_index.json", inputs))
    expected_groups = {(p["regime"], p["group"]) for p in declared["pools"]}
    if (not index.get("complete") or index.get("deployment_plan_sha256") != sha(plan_path)
            or index.get("source_hash") != source_hash()
            or index.get("fit_lock_sha256") != sha(root/"payoffs/fit_lock.json")
            or {tuple(v) for v in index["groups"]} != expected_groups):
        raise ValueError("Complete independently initialized ensemble predictions are required")
    replacements, contrasts = {}, {}
    for regime, group_key in sorted(expected_groups):
        group = read(_artifact(root/f"deployment/accuracy/{regime}/{group_key}/group.json", inputs))
        if not group.get("complete") or not group.get("independent_initials"):
            raise ValueError("Independent ensemble group changed")
        base = group["identity"]
        if (base["deployment_plan_sha256"] != sha(plan_path) or base["source_hash"] != source_hash()
                or base["fit_lock_sha256"] != index["fit_lock_sha256"]
                or base["reference_sha256"] != evidence["references"][regime]["reference_sha256"]
                or base["comparison_plan_sha256"] != sha(root/"comparison/preregistration.json")
                or base["regime"] != regime or base["group"] != group_key
                or base["bank_hash"] != runner.payoff_bank(regime, study).bank_hash):
            raise ValueError("Independent ensemble target identity changed")
        initial_record = read(_artifact(group["initial_pool_record_path"], inputs, group["initial_pool_record_sha256"]))
        _artifact(initial_record["arrays"]["path"], inputs, initial_record["arrays"]["sha256"])
        if initial_record["arrays"]["sha256"] != base["independent_initial_pool_sha256"]:
            raise ValueError("Independent initial pool changed")
        found = set()
        for item in group["records"]:
            record = read(_artifact(item["path"], inputs, item["sha256"]))
            name = record["method"]["name"]
            if (not record.get("complete") or not record.get("initial_pool_independent_of_reference")
                    or not record.get("reference_and_initial_bootstrap_independent")
                    or record["identity"] != {**base, "name": name}):
                raise ValueError("Independent ensemble result changed")
            _artifact(record["arrays"]["path"], inputs, record["arrays"]["sha256"])
            found.add((record["method"]["family"], record["method"].get("seed")))
            replacements[(regime, group_key, name)] = record
        if {seed for family, seed in found if family == "repaired"} != set(study.optimizer_seeds):
            raise ValueError("All independent-ensemble optimizer seeds are required")
        if len(found) != len(study.optimizer_seeds)+1:
            raise ValueError("Independent untrained control is required")
        contrasts[(regime, group_key)] = group["NN_versus_fine_LGA"]
    records, paired = [], []
    for record in evidence["records"]:
        identity, family = record["identity"], record["method"]["family"]
        key = (identity["regime"], identity["group"], identity["name"])
        if "__ensemble__" in identity["group"] and family in ("repaired", "untrained-repaired"):
            if key not in replacements:
                raise ValueError("Independent primary ensemble inventory is incomplete")
            paired.append(record)
            records.append(replacements.pop(key))
        else:
            records.append(record)
    if replacements:
        raise ValueError("An undeclared independent ensemble result appeared")
    groups, paired_groups = [], []
    for group in evidence["groups"]:
        key = group["regime"], group["group"]
        if key in contrasts:
            paired_groups.append(group)
            groups.append({**group, "NN_versus_fine_LGA": contrasts[key],
                           "primary_NN_initial_integration": "independent-of-reference"})
        else:
            groups.append(group)
    evidence.update(records=records, groups=groups, paired_initial_records=paired,
                    paired_initial_groups=paired_groups, independent_initial_records=len(paired))


def normalized_sampling(mean, covariance, bootstrap_mean, bootstrap_covariance,
                        reference_scaled_covariance, n_particles):
    """Standalone uncertainty in the same reference units as accuracy errors."""
    std = np.sqrt(np.diagonal(np.asarray(reference_scaled_covariance)/n_particles, axis1=-2, axis2=-1))
    products = std[..., :, None]*std[..., None, :]
    dm = abs(np.asarray(bootstrap_mean)-np.asarray(mean))/std
    dc = abs(np.asarray(bootstrap_covariance)-np.asarray(covariance))/n_particles/products
    return (np.quantile(np.max(dm, axis=-1), .95, axis=0),
            np.quantile(np.max(dc, axis=(-2, -1)), .95, axis=0))


def accuracy_tables(evidence):
    rows, components, covariances, sampling, refinement, contrasts, stopping, boundary, law = ([] for _ in range(9))
    study = evidence["study"]
    # Full chunk verification is retained once per case and shared with the
    # paired-initial diagnostics, rather than repeated for every group.
    reference_cache = evidence.setdefault("_verified_reference_cache", {})
    for packet in evidence["records"]:
        identity, method = packet["identity"], packet["method"]
        regime, key = identity["regime"], identity["group"]
        profile, role, anchor = key.split("__")
        cfg, bank = runner.reference_configuration(regime, study), runner.payoff_bank(regime, study)
        origin = {"regime": regime, "kappa": cfg.kappa, "horizon": cfg.horizon,
                  "profile": profile, "role": role, "anchor": int(anchor), "group": key,
                  "name": identity["name"], "family": method["family"], "seed": method.get("seed"),
                  "NN_initial_pool": ("independent" if packet.get("initial_pool_independent_of_reference")
                     else "shared-with-reference" if role == "ensemble" and method["family"] in
                     ("repaired", "untrained-repaired") else "not-applicable")}
        if packet.get("law_diagnostics"):
            law.append({**origin, **{k: v if np.isscalar(v) or v is None else json.dumps(v)
                                   for k, v in packet["law_diagnostics"].items()}})
        values = packet["metrics"]["records"]
        with np.load(packet["arrays"]["path"]) as arrays:
            if "bootstrap_mean" in arrays:
                mr, cr = normalized_sampling(arrays["mean"], arrays["scaled_covariance"],
                    arrays["bootstrap_mean"], arrays["bootstrap_scaled_covariance"],
                    [r["reference_scaled_covariance"] for r in values], cfg.n_particles)
            else:
                mr, cr = np.zeros(3), np.zeros(3)
        for item in values:
            t = item["time_index"]
            base = {**origin, "time_index": t, "time": cfg.query_times[t]}
            rows.append({**base, **{k: item[k] for k in (
                "max_mean_error_in_reference_std", "max_normalized_covariance_error",
                "mean_sampling95_simultaneous_radius", "covariance_sampling95_simultaneous_radius",
                "mean_error_plus_sampling95", "covariance_error_plus_sampling95")}})
            sampling.append({**base, "mean95_radius": mr[t], "covariance95_radius": cr[t],
                "scope": "standalone estimator sampling; separate from method-discrepancy uncertainty"})
            rc, pc = np.asarray(item["reference_scaled_covariance"]), np.asarray(item["candidate_scaled_covariance"])
            for i, label in enumerate(bank.labels):
                components.append({**base, "observable": label,
                    "reference_mean": item["reference_mean"][i], "candidate_mean": item["candidate_mean"][i],
                    "normalized_mean_difference": item["normalized_mean_difference"][i],
                    "reference_scaled_variance": rc[i, i], "candidate_scaled_variance": pc[i, i],
                    "variance_ratio": pc[i, i]/rc[i, i],
                    **physical_bounds(bank, cfg.query_times[t], i, item["candidate_mean"][i],
                                      pc[i, i]/cfg.n_particles)})
                for j, other in enumerate(bank.labels):
                    covariances.append({**base, "observable_i": label, "observable_j": other,
                        "normalized_covariance_difference": item["normalized_covariance_difference"][i][j],
                        "reference_scaled_covariance": rc[i, j], "candidate_scaled_covariance": pc[i, j]})
    for group in evidence["groups"]:
        regime, key = group["regime"], group["group"]
        cfg = runner.reference_configuration(regime, study)
        profile, role, anchor = key.split("__")
        base = {"regime": regime, "kappa": cfg.kappa, "horizon": cfg.horizon,
                "group": key, "profile": profile, "role": role, "anchor": int(anchor)}
        boundary.append({**base, **{k: v if np.isscalar(v) or v is None else json.dumps(v)
                                   for k, v in group.get("near_boundary_diagnostics", {}).items()}})
        for level, metric in group["reference_refinement"].items():
            for item in metric["records"]:
                refinement.append({**base, "level": level, "time_index": item["time_index"],
                    "time": cfg.query_times[item["time_index"]], **{k: item[k] for k in (
                    "max_mean_error_in_reference_std", "max_normalized_covariance_error",
                    "mean_error_plus_sampling95", "covariance_error_plus_sampling95",
                    "mean_sampling95_simultaneous_radius", "covariance_sampling95_simultaneous_radius")}})
        for name, contrast in group["NN_versus_fine_LGA"].items():
            for item in contrast["records"]:
                for stat in ("mean", "covariance"):
                    low, high = item[f"{stat}_advantage_percentile95"]
                    contrasts.append({**base, "name": name, "time_index": item["time_index"],
                        "time": cfg.query_times[item["time_index"]], "statistic": stat,
                        "advantage": item[f"{stat}_error_advantage"], "lower95": low, "upper95": high,
                        "NN_closer_pointwise95": low > 0, "LGA_closer_pointwise95": high < 0,
                        "scope": "per fitted seed and horizon; no across-regime family-wise certificate"})
        if regime not in reference_cache:
            reference_cache[regime] = runner.load_reference(evidence["root"], regime)["groups"]
        data = reference_cache[regime][key]
        stops = np.asarray(data["stopped_at"])
        for t, time_value in enumerate(cfg.query_times):
            count = int(np.sum(stops <= time_value))
            low, high = binomial_interval(count, len(stops))
            stopping.append({**base, "time_index": t, "time": time_value, "stopped": count,
                "paths": len(stops), "probability": count/len(stops), "wilson95_low": low,
                "wilson95_high": high, "NN_prediction": "unsupported", "LGA_prediction": "unstopped-law"})
    return {"accuracy": rows, "components": components, "covariances": covariances,
            "sampling": sampling, "reference_refinement": refinement, "contrasts": contrasts,
            "stopping": stopping, "near_boundary": boundary, "gaussian_law_diagnostics": law}


def binomial_interval(count, total):
    if not 0 <= count <= total or total < 1:
        raise ValueError("A complete binomial allocation is required")
    z, p = 1.959963984540054, count/total
    center, half = (p+z*z/(2*total))/(1+z*z/total), z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return max(0., center-half), min(1., center+half)


def physical_bounds(bank, time_value, index, mean, variance):
    """Simplex support bounds, valid also for frozen terminal/killed rewards."""
    phi, running = bank.terminal_weights[index], bank.running_weights[index]
    low = float(phi.min()+time_value*min(0., running.min()))
    high = float(phi.max()+time_value*max(0., running.max()))
    # Stopping can shorten exposure, so zero belongs to the integral support.
    mean_inside = low-1e-12 <= mean <= high+1e-12
    variance_bound = max(0., (high-mean)*(mean-low)) if mean_inside else 0.
    return {"physical_mean_lower": low, "physical_mean_upper": high,
            "mean_inside_physical_bounds": mean_inside,
            "physical_variance_upper_at_mean": variance_bound,
            "variance_inside_physical_bounds": mean_inside and -1e-12 <= variance <= variance_bound+1e-12}


def cost_tables(timing, fit_lock, data_seconds):
    """Every deployment pays one fit, its literal data pool, and cold setup."""
    fits = {(m["case"], m["name"]): m for m in fit_lock["models"]}
    loads = {(m["regime"], m["profile"], m["name"]): m["seconds"] for m in timing["loading"]}
    probes = {(m["regime"], m["profile"], m["name"]): m["first_prediction_seconds"]
              for m in timing["records"] if m["role"] == "conditional" and m["batch"] == 1}
    compiled = timing.get("compiled_control", {})
    rows, totals = [], []
    for measured in timing["records"]:
        regime, name, profile = measured["regime"], measured["name"], measured["profile"]
        fit = fits.get((regime, name))
        fitting, data = (fit["training_seconds"], data_seconds[regime]) if fit is not None else (0., 0.)
        load = loads.get((regime, profile, name), 0.)
        native = measured["family"] == "native-MC"
        compile_seconds = compiled.get("compile_seconds", 0.) if native else 0.
        setup = max(0., compiled.get("setup_seconds_this_process", 0.)
                    - compiled.get("compile_seconds_this_process", 0.)) if native else 0.
        probe = (probes[(regime, profile, name)] if measured["family"] in ("repaired", "untrained-repaired")
                 and (measured["role"] != "conditional" or measured["batch"] != 1) else 0.)
        row = {k: v for k, v in measured.items() if k != "samples_seconds"}
        row.update(training_seconds=fitting, shared_data_seconds=data, restoration_seconds=load,
                   cold_compilation_seconds=compile_seconds, compiled_setup_seconds=setup,
                   cache_creation_B1_probe_seconds=probe)
        row["known_offline_seconds"] = fitting+data+load+compile_seconds+setup+probe
        rows.append(row)
        for q in (1, 100, 10000, 100000):
            totals.append({**{k: row[k] for k in ("slot", "regime", "profile", "name", "family", "role", "batch")},
                "query_calls": q, "known_offline_seconds": row["known_offline_seconds"],
                "first_prediction_seconds": row["first_prediction_seconds"],
                "subsequent_warm_seconds": (q-1)*row["median_seconds"],
                "total_known_seconds": row["known_offline_seconds"]+row["first_prediction_seconds"]
                    +(q-1)*row["median_seconds"]})
    return rows, totals


def statistic_gate(accuracy, sampling, refinement, statistic):
    """A finite-reference diagnostic, explicitly weaker than continuous-law proof."""
    if not accuracy or not refinement:
        return "unresolved-inventory"
    error = "max_mean_error_in_reference_std" if statistic == "mean" else "max_normalized_covariance_error"
    radius = statistic+"_sampling95_simultaneous_radius"
    upper = statistic+"_error_plus_sampling95"
    if max(a[error] for a in accuracy) > .05:
        return "point-error-above-5pct"
    if max(a[upper] for a in accuracy)+max(a[upper] for a in refinement) > .05:
        return "error-plus-uncertainty-unresolved"
    if max(a[radius] for a in accuracy) > .0125:
        return "sampling-component-above-1.25pct"
    if max(a[upper] for a in refinement) > .0125:
        return "reference-refinement-component-unresolved"
    if sampling and max(a[statistic+"95_radius"] for a in sampling) > .0125:
        return "standalone-MC-component-unresolved"
    return "finite-reference-diagnostic-pass"


def payback(candidate, control):
    saving = control["median_seconds"]-candidate["median_seconds"]
    if saving <= 0:
        return None
    extra = (candidate["known_offline_seconds"]+candidate["first_prediction_seconds"]
             -control["known_offline_seconds"]-control["first_prediction_seconds"])
    return max(1, math.ceil(1+extra/saving))


def amortization_tables(tables):
    rows = []
    for candidate in tables["costs"]:
        if candidate["family"] != "repaired" or (candidate["role"] == "conditional" and candidate["batch"] != 1):
            continue
        same = {k: candidate[k] for k in ("regime", "profile", "role")}
        def compatible(r):
            return all(r[k] == v for k, v in same.items())
        refinement = [r for r in tables["reference_refinement"] if compatible(r) and r["level"] == "level0_versus_finest"]
        for control in tables["costs"]:
            if control["family"] not in ("LGA", "native-MC") or not compatible(control):
                continue
            for statistic in ("mean", "covariance"):
                gates = []
                for item in (candidate, control):
                    accuracy = [r for r in tables["accuracy"] if compatible(r) and r["name"] == item["name"]]
                    sampling = [r for r in tables["sampling"] if compatible(r) and r["name"] == item["name"]]
                    gates.append(statistic_gate(accuracy, sampling, refinement, statistic))
                arithmetic = payback(candidate, control)
                qualified = arithmetic if all(g == "finite-reference-diagnostic-pass" for g in gates) else None
                rows.append({**same, "candidate": candidate["name"], "control": control["name"],
                    "statistic": statistic, "candidate_gate": gates[0], "control_gate": gates[1],
                    "online_saving_seconds": control["median_seconds"]-candidate["median_seconds"],
                    "descriptive_payback_calls": arithmetic, "accuracy_qualified_payback_calls": qualified,
                    "scope": ("new supplied-state queries" if candidate["role"] == "conditional"
                              else "repeated fresh estimates of the same fixed IID law; identical answers can be memoized")
                              + "; fixed physics/bank/horizons, continuous stopping bias unresolved"})
    return rows


def _table(headers, rows):
    return "\n".join(["| "+" | ".join(map(str, headers))+" |", "| "+" | ".join("---" for _ in headers)+" |"]
                     +["| "+" | ".join(map(str, row))+" |" for row in rows])


def _range(values, factor=1.):
    values = [v*factor for v in values]
    return "—" if not values else (f"{min(values):.3g}" if min(values) == max(values)
                                  else f"{min(values):.3g}–{max(values):.3g}")


def model_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["regime"], row["family"], row["name"])].append(row)
    return [{"regime": regime, "family": family, "name": name, "seed": values[0]["seed"],
             "kappa": values[0]["kappa"], "horizon": values[0]["horizon"],
             "worst_mean": max(r["max_mean_error_in_reference_std"] for r in values),
             "worst_covariance": max(r["max_normalized_covariance_error"] for r in values),
             "worst_mean_plus_sampling95": max(r["mean_error_plus_sampling95"] for r in values),
             "worst_covariance_plus_sampling95": max(r["covariance_error_plus_sampling95"] for r in values)}
            for (regime, family, name), values in sorted(groups.items())]


def seed_consistency(contrasts, seeds):
    grouped = defaultdict(list)
    keys = ("regime", "profile", "role", "group", "time_index", "time", "statistic")
    for row in contrasts:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        if len(rows) != len(seeds) or len({r["name"] for r in rows}) != len(seeds):
            raise ValueError("A regime advantage cannot drop optimizer seeds")
        result.append({**dict(zip(keys, key, strict=True)), "optimizer_seeds": str(tuple(seeds)),
            "minimum_advantage": min(r["advantage"] for r in rows),
            "maximum_advantage": max(r["advantage"] for r in rows),
            "minimum_lower95": min(r["lower95"] for r in rows),
            "maximum_upper95": max(r["upper95"] for r in rows),
            "all_seeds_NN_closer_pointwise95": all(r["lower95"] > 0 for r in rows),
            "all_seeds_LGA_closer_pointwise95": all(r["upper95"] < 0 for r in rows),
            "scope": "consistency across fixed seeds, not a family-wise grid certificate"})
    return result


def validation_tables(evidence):
    path = evidence["root"]/"audit/postfit_validation_diagnostic.json"
    diagnostic = read(_artifact(path, evidence["inputs"]))
    expected = {m["name"]: m for m in evidence["lock"]["models"]}
    if (not diagnostic.get("complete") or diagnostic["identity"]["source_hash"] != source_hash()
            or not diagnostic["all_references_absent_at_completion"]
            or not diagnostic["no_selection_refit_or_future_outcome_access"]
            or diagnostic["identity"]["future_test_outcomes_opened"] is not False
            or diagnostic["identity"]["data_lock_sha256"] != sha(evidence["root"]/"payoffs/data_lock.json")
            or diagnostic["identity"]["preregistration_sha256"] != sha(evidence["root"]/"payoffs/preregistration.json")
            or diagnostic["identity"]["fit_lock_sha256"] != sha(evidence["root"]/"payoffs/fit_lock.json")
            or len(diagnostic["records"]) != len(expected)
            or {r["name"] for r in diagnostic["records"]} != set(expected)):
        raise ValueError("Complete pre-reference validation replay is required")
    rows, phases, consistency, concentration = [], [], [], []
    for record in diagnostic["records"]:
        if (not record["selected_mean_unchanged"]
                or record["case"] != expected[record["name"]]["case"]
                or record["seed"] != expected[record["name"]]["seed"]
                or record["checkpoint_sha256"] != expected[record["name"]]["checkpoint_sha256"]):
            raise ValueError("Validation replay changed the frozen fit")
        _artifact(record["numeric_evidence"]["path"], evidence["inputs"], record["numeric_evidence"]["sha256"])
        base = {"regime": record["case"], "name": record["name"], "seed": record["seed"]}
        for stratum, stages in record["full_selected_stratified_residuals"].items():
            for stage, values in stages.items():
                rows.append({**base, "stratum": stratum, "stage": stage,
                             **{k: v for k, v in values.items() if np.isscalar(v) or v is None}})
                overall = record["full_selected_stratified_residuals"]["all"][stage]
                denominator = overall["rows"]*overall["normalized_objective"]
                concentration.append({**base, "stratum": stratum, "stage": stage,
                    "rows": values["rows"], "normalized_rms": values.get("normalized_rms", 0.),
                    "fraction_of_full_squared_loss": (values["rows"]*values.get("normalized_objective", 0.)
                                                     /denominator if denominator else 0.)})
        for phase, strata in record["diagnostic_subset_phase_residuals"].items():
            for stratum, stages in strata.items():
                for stage, values in stages.items():
                    phases.append({**base, "phase": phase, "stratum": stratum, "stage": stage,
                                   **{k: v for k, v in values.items() if np.isscalar(v) or v is None}})
        consistency.append({**base, **record["full_selected_active_counts"],
            **{k: v if np.isscalar(v) or v is None else json.dumps(v)
               for k, v in record["full_selected_payoff_consistency"].items()}})
    corner = read(_artifact(evidence["root"]/"audit/boundary_corner_control.json", evidence["inputs"]))
    if (not corner.get("complete") or corner["source_hash"] != source_hash()
            or not corner["strictly_interior"] or not corner["no_fit_or_model_change"]):
        raise ValueError("The unchanged strict-interior boundary control is required")
    controls = [{"setting": r["setting"], "gate": corner["barrier"],
        "maximum_integral_mean_identity_error": r["maximum_mean_identity_error"],
        "maximum_terminal_covariance_identity_error": r["maximum_covariance_identity_error"],
        "maximum_integral_mean_corner_defect": float(np.max(abs(np.array(r["exact_integral_mean_corner_defect"])))),
        "maximum_terminal_covariance_corner_defect": float(np.max(abs(np.array(r["exact_terminal_covariance_corner_defect"])))),
        "units": "training's internal scaled units"} for r in corner["records"]]
    return {"validation_residuals": rows, "matched_validation_phases": phases,
            "validation_consistency": consistency, "validation_loss_concentration": concentration,
            "boundary_corner_control": controls}


def figures(output, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"repaired": "#0072B2", "untrained-repaired": "#999999", "LGA": "#009E73", "native-MC": "#222222"}
    summary = tables["model_summary"]
    horizons = sorted({r["horizon"] for r in summary})
    fig, axes = plt.subplots(len(horizons), 2, figsize=(11, 4*len(horizons)), squeeze=False, constrained_layout=True)
    for i, horizon in enumerate(horizons):
        for j, stat in enumerate(("mean", "covariance")):
            ax = axes[i, j]
            for family, color in colors.items():
                selected = [r for r in summary if r["horizon"] == horizon and r["family"] == family]
                ax.scatter([r["kappa"] for r in selected], [100*r["worst_"+stat] for r in selected], color=color, alpha=.65, label=family)
            ax.axhline(5, linestyle="--", color=".4")
            ax.set(xlabel="Interaction κ", ylabel=f"Worst {stat} discrepancy (%)", title=f"T={horizon:g}: every seed, both initial laws", yscale="log")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output/"accuracy_regimes.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(len(horizons), 2, figsize=(11, 4*len(horizons)), squeeze=False, constrained_layout=True)
    for i, horizon in enumerate(horizons):
        regimes = {r["regime"] for r in summary if r["horizon"] == horizon}
        for j, role in enumerate(("conditional", "ensemble")):
            ax = axes[i, j]
            for family, color in colors.items():
                selected = [r for r in tables["costs"] if r["regime"] in regimes and r["family"] == family and r["role"] == role
                            and (role != "conditional" or r["batch"] == 1 or r["family"] == "native-MC")]
                kappas = {r["regime"]: r["kappa"] for r in summary}
                ax.scatter([kappas[r["regime"]] for r in selected], [1000*r["median_seconds"] for r in selected], color=color, alpha=.6, label=family)
            ax.set(xlabel="Interaction κ", ylabel="Warm literal query (ms)", title=f"T={horizon:g}, {role}", yscale="log")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output/"inference_regimes.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    regimes = sorted({r["regime"] for r in summary})
    for i, regime in enumerate(regimes):
        selected = [r for r in tables["stopping"] if r["regime"] == regime and r["time_index"] == 2]
        ax.scatter(np.full(len(selected), i), [r["probability"] for r in selected], alpha=.65)
    ax.set(xticks=range(len(regimes)), xticklabels=regimes, ylabel="Native reference stopping probability", title="All groups retained; stopping is a separate law diagnostic")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(output/"stopping_regimes.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for irole, role in enumerate(("conditional", "ensemble")):
        for istat, statistic in enumerate(("mean", "covariance")):
            ax = axes[irole, istat]
            for i, regime in enumerate(regimes):
                selected = [r for r in tables["contrasts"] if r["regime"] == regime
                            and r["role"] == role and r["statistic"] == statistic and r["time_index"] == 2]
                offsets = np.linspace(-.22, .22, len(selected))
                for offset, row in zip(offsets, selected, strict=True):
                    color = "#0072B2" if row["profile"] == "uniform" else "#D55E00"
                    ax.vlines(i+offset, 100*row["lower95"], 100*row["upper95"], alpha=.4, color=color)
                    ax.scatter(i+offset, 100*row["advantage"], s=12, color=color)
            ax.axhline(0, color=".4", linestyle="--")
            ax.set_yscale("symlog", linthresh=1.)
            ax.margins(y=.12)
            ax.set(xticks=range(len(regimes)), xticklabels=regimes, ylabel="LGA error − NN error (percentage points)",
                   title=f"{role}, final {statistic}: pointwise 95% intervals")
            ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Every optimizer seed; blue: uniform, orange: nonuniform initial law\n"
                 "Symmetric log scale outside ±1 percentage point", fontsize=12)
    fig.savefig(output/"accuracy_advantages.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for i, regime in enumerate(regimes):
        selected = [r for r in tables["training"] if r["regime"] == regime]
        ax.scatter(np.full(len(selected), i), [r["training_seconds"] for r in selected])
    ax.set(xticks=range(len(regimes)), xticklabels=regimes, ylabel="Seconds / two-stage model fit", title="All three optimizer seeds; training costs retained")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(output/"training_costs.png", dpi=180)
    plt.close(fig)
    for role in ("conditional", "ensemble"):
        fig, axes = plt.subplots(len(horizons), len(regimes)//len(horizons),
                                 figsize=(14, 4*len(horizons)), squeeze=False, constrained_layout=True)
        for ax, regime in zip(axes.flat, regimes, strict=True):
            for label, family, method in (("repaired", "repaired", None), ("LGA dt=1/128", "LGA", "LGA_dt7"),
                                           ("LGA dt=1/1024", "LGA", "LGA_dt10"), ("native MC", "native-MC", None)):
                values = [r for r in tables["total_costs"] if r["regime"] == regime and r["role"] == role
                          and r["family"] == family and (method is None or r["name"] == method)
                          and (role != "conditional" or r["batch"] == 1 or r["family"] == "native-MC")]
                qs = sorted({r["query_calls"] for r in values})
                ys = [[r["total_known_seconds"] for r in values if r["query_calls"] == q] for q in qs]
                if ys:
                    ax.plot(qs, [np.median(v) for v in ys], label=label)
                    ax.fill_between(qs, [min(v) for v in ys], [max(v) for v in ys], alpha=.12)
            ax.set(xscale="log", yscale="log", xlabel=("New supplied-state queries" if role == "conditional"
                   else "Repeated fresh estimates; fixed IID law"),
                   ylabel="Fitting + setup + predictions (s)", title=regime)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"{role}: descriptive cost arithmetic; accuracy gates remain separate")
        fig.savefig(output/f"total_cost_{role}.png", dpi=180)
        plt.close(fig)


def report_text(evidence, tables):
    summary, costs = tables["model_summary"], tables["costs"]
    study = evidence["study"]
    reference_groups = len(evidence["groups"])
    reference_count = reference_groups*study.paths_per_group
    consistency = tables["seed_consistency"]
    nn_wins = sum(r["all_seeds_NN_closer_pointwise95"] for r in consistency)
    lga_wins = sum(r["all_seeds_LGA_closer_pointwise95"] for r in consistency)
    strongest = max(evidence["plan"]["cases"], key=lambda c: (
        runner.reference_configuration(c, study).horizon,
        runner.reference_configuration(c, study).kappa))
    nn_strong = [r for r in summary if r["regime"] == strongest and r["family"] == "repaired"]
    gaussian_strong = [r for r in summary if r["regime"] == strongest and r["name"] == "LGA_dt10"]
    lines = ["# Longer horizons and stronger interactions: repaired statistics NN versus Gaussian", "",
        "This fixed-design study retains every declared optimizer seed. Native Monte Carlo is the reference; "
        "the Gaussian comparator is an unstopped linear-noise approximation. The NN predicts only the supported "
        "four means and same-time 4×4 covariances at three horizons. It does not predict stopping probabilities or sample paths.", "",
        "## Main findings", "",
        f"Of {len(consistency)} fixed group/horizon/statistic comparisons, **{nn_wins} favor the NN for all "
        f"three optimizer seeds** with positive pointwise 95% lower contrast bounds; {lga_wins} favor the fine "
        "Gaussian approximation for all seeds. The remaining comparisons are unresolved or inconsistent "
        "across seeds. These counts retain both initial laws and conditional/independent-ensemble tasks; "
        "they do not constitute a simultaneous grid-wide inference.", "",
        f"In the longest, strongest case ({strongest}), the repaired NN's worst mean discrepancy is "
        f"{_range([r['worst_mean'] for r in nn_strong])} reference standard deviations, versus "
        f"{_range([r['worst_mean'] for r in gaussian_strong])} for the fine Gaussian approximation. "
        "The worst normalized covariance discrepancies are "
        f"{_range([r['worst_covariance'] for r in nn_strong], 100)}% versus "
        f"{_range([r['worst_covariance'] for r in gaussian_strong], 100)}%. "
        "Stronger interaction exposes limitations of both methods; this NN formulation and fixed fitting "
        "allocation do not turn that difficulty into an accuracy advantage. The validation and controlled "
        "boundary audit below diagnose a substantial fitting problem. All three unsuccessful fits remain in the study.", "",
        "## Design and interpretation", "",
        f"The fixed grid is G={study.n_cells}, N={study.n_particles}, D={study.diffusion:g}, "
        f"a₂={study.a2:g}, κ∈{study.kappas}, T∈{study.horizons}. "
        "The mode-1 uniform-state growth rate changes sign at κ=2. Thus the grid separates a longer stable "
        "regime from two unstable interaction strengths. Both IID histogram laws—uniform and exp(0.2 cos x)—remain. "
        "The mixed bank contains terminal sine/cosine mode 3 and killed sine/cosine mode-1 time integrals divided "
        "by each problem's fixed final horizon. Queries are at T/4,T/2,T. Results across T concern these declared "
        "time-average units, not an unchanged unnormalized running weight.", "",
        f"There are {study.train_paths} training and {study.validation_paths} validation paths per initial law "
        f"and regime, with {len(study.state_fractions)*study.horizon_samples} state/horizon rows per trajectory. "
        f"Each mean and covariance stage receives {study.adam_steps} AdamW and {study.lbfgs_steps} LBFGS outer steps. "
        f"The selected mean is frozen during covariance fitting. All {len(evidence['lock']['models'])} models use "
        f"width {study.width} and {study.quadrature_nodes} deterministic "
        "heat-anchor quadrature nodes. No final-reference outcome selects a fit, seed, allocation or regime. "
        "This is fixed-problem fitting, not generalization to unseen physical parameters.", "",
        f"Each of {reference_groups} reference groups has {study.paths_per_group:,} independent paths: "
        f"{reference_count:,} reference families overall. The base step is {study.reference_step:g}, with "
        f"coupled refinement factors {study.reference_factors} on {study.refinement_paths:,} families/group. "
        f"A separately drawn {study.paths_per_group:,}-path native Monte Carlo deployment at 2⁻¹⁰ is also evaluated. "
        "Every stopped path remains in the estimates. Terminal values freeze at the first Euler-polygon crossing "
        "of ρ*=0.01/(2π); running integrals cease there.", "",
        "## Independent accuracy", ""]
    rows = []
    for regime in evidence["plan"]["cases"]:
        for family, name in (("repaired", None), ("untrained-repaired", None), ("LGA", "LGA_dt7"), ("LGA", "LGA_dt10"), ("native-MC", None)):
            selected = [r for r in summary if r["regime"] == regime and r["family"] == family and (name is None or r["name"] == name)]
            rows.append([regime, name or family, _range([r["worst_mean"] for r in selected], 100),
                         _range([r["worst_covariance"] for r in selected], 100),
                         _range([r["worst_covariance_plus_sampling95"] for r in selected], 100)])
    lines += [_table(["Regime", "Method", "Worst mean (% SD)", "Worst covariance (%)", "Covariance + sampling (%)"], rows), "",
        "Each per-model maximum covers both initial laws, all supplied histogram anchors/ensemble queries and "
        "all three horizons. Ranges preserve optimizer seeds; the LGA steps remain separate. "
        "Errors use reference standard deviations and their products, not relative errors of possibly zero means. "
        "Individual outputs, covariance entries and simplex support-bound diagnostics remain in components.csv "
        "and covariances.csv. The untrained heat control runs this architecture's implementation; its runtime "
        "is not presented as an optimized standalone analytic solver.", "",
        "![](accuracy_regimes.png)", "", "![](accuracy_advantages.png)", "",
        "Positive contrasts mean the repaired NN is closer than the fine LGA. Primary NN ensemble predictions "
        "use a separately preregistered independent initial-histogram integration pool, with an independent "
        "initial-pool bootstrap. Conditional predictions use the supplied histogram. The bootstrap resamples complete "
        "trajectory families and uses the same reference resample for both methods. The intervals are pointwise "
        "per seed, group and horizon; searching the grid does not create a family-wise certificate. A resolved "
        "error contrast is distinct from meeting the absolute 5% accuracy target. Every seed is reported, "
        "including failures and unresolved comparisons.", "",
        "seed_consistency.csv separately identifies fixed profile/role/horizon comparisons where all optimizer "
        "seeds favor the same method. An isolated positive cell remains exploratory; even consistency across "
        "seeds is not a simultaneous certificate over the searched physical grid.", "",
        "The original NN ensemble comparison sharing initial histograms with the native reference is retained "
        "in paired_initial_accuracy.csv and paired_initial_contrasts.csv as a lower-noise approximation-bias "
        "diagnostic. Primary deployment tables and payback gates use independent initialization. Both the "
        "original and supplemental records remain checksum-indexed; the supplement was frozen before any "
        "reference outcomes or independent integration histograms were drawn.", "",
        "## Stopping and reference uncertainty", "", "![](stopping_regimes.png)", "",
        "The Gaussian baseline does not implement absorbing dynamics. If stopping is appreciable, a discrepancy "
        "can reflect that law mismatch as well as nonlinear closure error. The NN enforces frozen boundary values "
        "but its near-boundary approximation is still learned. Native refinement differences do not enclose "
        "continuous boundary-hitting bias. Zero observed stops would not establish rare-event accuracy.", "",
        "near_boundary.csv retains path-minimum quantiles and boundary-strip frequencies even for paths that "
        "do not stop. gaussian_law_diagnostics.csv retains deterministic mean minima, covariance positivity and "
        "symmetry checks. These diagnostics help distinguish interior nonlinear closure errors from stopping effects.", "",
        "Reference refinement, standalone initial-law integration noise, independent native-deployment sampling "
        "and method-discrepancy intervals are saved separately. The strict deployment gate asks for 5% normalized "
        "error and each sampling/refinement contribution below 1.25%; passing a finite-reference diagnostic "
        "would still not be a continuous-DK theorem. See reference_refinement.csv, sampling.csv and amortization.csv.", ""]
    strong_stops = [r for r in tables["stopping"] if r["regime"] == strongest and r["time_index"] == 2]
    lines += [f"Final-time stopping counts in {strongest}:", "",
        _table(["Initial law / query", "Stopped / paths", "Probability (%)", "Wilson 95% interval (%)"],
            [[r["group"], f"{r['stopped']:,} / {r['paths']:,}", f"{100*r['probability']:.4g}",
              f"{100*r['wilson95_low']:.4g}–{100*r['wilson95_high']:.4g}"] for r in strong_stops]), ""]
    if len(evidence["plan"]["cases"]) > 1 and not any(
            r["stopped"] for r in tables["stopping"] if r["regime"] != strongest):
        lines += [f"No stopping was observed in the other {len(evidence['plan']['cases'])-1} regimes.", ""]
    uniform_errors = [r["max_mean_error_in_reference_std"] for r in tables["accuracy"]
                      if r["regime"] == strongest and r["profile"] == "uniform" and r["family"] == "repaired"]
    if (not any(r["stopped"] for r in strong_stops if r["profile"] == "uniform")
            and uniform_errors and max(uniform_errors) > .05):
        lines += ["Uniform initial-law groups in the strongest case also have no observed stops, while "
            "their NN mean predictions already exceed the accuracy target. Stopping therefore does not "
            "by itself explain the NN failure.", ""]
    worst_component = max((r for r in tables["components"] if r["regime"] == strongest
                           and r["family"] == "repaired"), key=lambda r: abs(r["normalized_mean_difference"]))
    lines += [f"The largest mean defect in {strongest} concerns `{worst_component['observable']}` in "
        f"`{worst_component['group']}` at t={worst_component['time']:g}: prediction "
        f"{worst_component['candidate_mean']:.6g}, reference {worst_component['reference_mean']:.6g}. "
        "The complete per-output breakdown is in components.csv.", ""]
    if tables.get("validation_residuals"):
        rows = []
        for regime in evidence["plan"]["cases"]:
            values = [r for r in tables["validation_residuals"] if r["regime"] == regime and r["stratum"] == "all"]
            rows.append([regime, *[_range([r["normalized_rms"] for r in values if r["stage"] == stage])
                                   for stage in ("mean", "covariance")]])
        strongest = max(evidence["plan"]["cases"], key=lambda c: (
            runner.reference_configuration(c, study).horizon,
            runner.reference_configuration(c, study).kappa))
        # These are validation residuals in training units, not test prediction errors.
        near = [r for r in tables["validation_loss_concentration"] if r["regime"] == strongest
                and r["stratum"] == "minimum_mass/five_to_ten_thresholds"]
        interior = [r for r in tables["validation_residuals"] if r["regime"] == strongest
                    and r["stratum"] == "minimum_mass/above_25_thresholds"]
        mismatch = sum(r["rounded_boundary_active_disagreement"] for r in tables["validation_consistency"])
        full_rows = max(r["rows"] for r in tables["validation_residuals"]
                        if r["regime"] == strongest and r["stratum"] == "all")
        lines += ["## Why the difficult fits plateau", "",
            "A complete post-fit validation replay was recorded before native reference generation. These "
            "are normalized generator-residual RMS values, not percentages or prediction-error estimates:", "",
            _table(["Regime", "Mean residual RMS, all seeds", "Covariance residual RMS, all seeds"], rows), "",
            f"In {strongest}, {_range([r['rows'] for r in near], 1)} of the {full_rows:,} validation rows have a "
            "minimum mass between five and ten times the stopping threshold. They contribute "
            f"{_range([r['fraction_of_full_squared_loss'] for r in near if r['stage'] == 'mean'], 100)}% "
            "of the mean objective and "
            f"{_range([r['fraction_of_full_squared_loss'] for r in near if r['stage'] == 'covariance'], 100)}% "
            "of the covariance objective. Late evolved states and small remaining horizons dominate. "
            "The interior fit is also imperfect: on rows more than 25 threshold masses from zero, the "
            f"mean/covariance RMS ranges are {_range([r['normalized_rms'] for r in interior if r['stage'] == 'mean'])} / "
            f"{_range([r['normalized_rms'] for r in interior if r['stage'] == 'covariance'])}. "
            "See validation_loss_concentration.csv and the matched initial/mean-selected/final phase replay.", "",
            "The controlled boundary audit identifies an exact small-time inconsistency in the current ansatz. "
            "For an interior state with boundary gate g<1, an integral mean starts as g·τ·q+O(τ²), "
            "although its required initial slope is q. Terminal covariance starts as g·τ·Γ+O(τ²), "
            "although its required slope is the native quadratic variation Γ. The declared higher-order "
            "learned corrections cannot change those leading terms. At g=0.5 the generator-residual "
            "defects agree with (g−1)q and (g−1)Γ to below 2×10⁻¹⁴ in internal units, including an "
            "arbitrary nonzero correction control. The inverse-time loss weighting magnifies the resulting "
            "short-horizon residuals. This is an asymptotic corner inconsistency, not a quantitative "
            "attribution of every finite-horizon prediction error; corrections can compensate away from zero. "
            "It diagnoses this fixed boundary ansatz, not neural moment solvers generally.", "",
            f"The active/absorbed replay found {mismatch} rounded-boundary classification disagreements. "
            "The frozen selected mean is unchanged during covariance fitting. Thus neither a changed mean "
            "nor a demonstrated active-mask roundoff bug explains this run. No model was repaired or refitted "
            "after the reference results. A subsequent experiment should preserve the correct short-time "
            "slopes while enforcing the stopped boundary values, then repeat independent validation.", ""]
    lines += ["## Fitting, inference and total prediction cost", "", "![](training_costs.png)", "",
        f"All {len(tables['training'])} fits recorded **{sum(r['training_seconds'] for r in tables['training']):.1f} model fitting seconds**. "
        "Per-fit wall durations were recorded in three concurrent, single-thread seed workers. "
        "Concurrent worker times are not summed and presented as elapsed time. Study cost includes all seeds; "
        "a fresh fixed-model deployment pays one model fit and its full shared training/validation data generation. "
        "No hyperparameter search cost is invented.", "", "![](inference_regimes.png)", ""]
    if tables.get("workers"):
        cpu = sum(r["cpu_seconds"] for r in tables["workers"])
        span = max(r["finished_unix"] for r in tables["workers"])-min(r["started_unix"] for r in tables["workers"])
        longest = max(r["wall_seconds"] for r in tables["workers"])
        lines += [f"Seed workers recorded {cpu:.1f} CPU seconds. The longest recorded worker execution was "
            f"{longest:.1f} seconds; the UTC start/finish span was {span:.1f} seconds. "
            f"State-pool generation recorded {sum(r['generation_seconds'] for r in tables['data']):.1f} seconds; "
            f"native reference chunks recorded {sum(r['reference_generation_seconds'] for r in tables['study_costs']):.1f} seconds. "
            "Those durations can overlap across workers. Reference construction is a study cost, separate from "
            "a model deployment or an independently drawn native-MC query. Tests, analysis and bootstrap costs "
            "are not invented when their execution times are unrecorded.", ""]
    rows = []
    for regime in evidence["plan"]["cases"]:
        for family, name in (("repaired", None), ("LGA", "LGA_dt7"), ("LGA", "LGA_dt10"), ("native-MC", None)):
            selected = [r for r in costs if r["regime"] == regime and r["family"] == family and (name is None or r["name"] == name)]
            conditional = [r for r in selected if r["role"] == "conditional" and (r["batch"] == 1 or r["family"] == "native-MC")]
            ensemble = [r for r in selected if r["role"] == "ensemble"]
            rows.append([regime, name or family, _range([r["median_seconds"] for r in conditional], 1000),
                         _range([r["median_seconds"] for r in ensemble], 1000),
                         _range([r["training_seconds"] for r in selected])])
    lines += [_table(["Regime", "Method", "Conditional warm ms", "Initial-ensemble warm ms", "Fit seconds/model"], rows), "",
        "All final timing uses one CPU thread after fitting/reference/accuracy workers have exited. A conditional "
        f"head receives one supplied histogram; an ensemble head actually draws and integrates {study.paths_per_group:,} histograms. "
        "Gaussian calls propagate the literal three-query moments, with analytic IID initial covariance for "
        f"ensemble queries. Native Monte Carlo calls actually draw forcing, solve {study.paths_per_group:,} stopped paths, reduce "
        "the requested bank and aggregate moments. No ensemble runtime is extrapolated from a small batch. "
        "Batched NN throughput for 64/256 supplied states remains in costs.csv; batched LGA throughput is not measured.", "",
        "Costs separately retain fitting/data, model restoration, compilation/setup, first prediction and warm "
        "calls. The first B=1 head call creates the known-horizon cache. Larger batches and ensemble calls follow "
        "it; their cold totals conservatively charge the measured B=1 first-query probe once. The probe is a "
        "complete discarded prediction, not a claim to pure cache-preparation timing. total_costs.csv shows "
        "1,100,10,000 and 100,000 calls at fixed physics, bank and horizons. Conditional calls can receive "
        "new supplied states; accuracy was assessed on only two declared anchors per initial law, which "
        "does not certify arbitrary unseen histograms. Ensemble curves only repeat fresh estimates of the same declared IID law; "
        "they do not establish generalization to new initial laws. Identical repeated inputs or deterministic "
        "ensemble answers could be memoized by any method.", "", "![](total_cost_conditional.png)", "",
        "![](total_cost_ensemble.png)", ""]
    qualified = sum(r["accuracy_qualified_payback_calls"] is not None for r in tables["amortization"])
    lines += [f"The strict finite-reference gates produce **{qualified} qualified payback rows**. "
        "Descriptive arithmetic payback and measured warm savings remain in amortization.csv even when the "
        "accuracy gate fails; they are not advertised as equal-accuracy wins. Choosing a method requires matching "
        "the task, accuracy and uncertainty as well as query volume. A fast inaccurate approximation is not a "
        "successful solver, and fitting amortization is not established by a warm-latency comparison alone.", "",
        "The compact evidence archive retains every JSON result, CSV, figure, audit and exact code snapshot. "
        "Full weights, trajectory/outcome HDF5 and numeric arrays stay in local run directories with checksums. "
        "The package and previous study artifacts are preserved.", ""]
    return "\n".join(lines)


def generate(root, documentation=None, *, independent_initials=True, validation_diagnostics=True):
    evidence = load_complete(root, independent_initials=independent_initials)
    tables = accuracy_tables(evidence)
    tables["seed_consistency"] = seed_consistency(tables["contrasts"], evidence["study"].optimizer_seeds)
    if validation_diagnostics:
        tables.update(validation_tables(evidence))
        native = read(_artifact(evidence["root"]/"audit/native_reference_candidate_audit.json", evidence["inputs"]))
        metrics = read(_artifact(evidence["root"]/"audit/metrics_replay.json", evidence["inputs"]))
        if (not native.get("complete") or native["source_hash"] != source_hash()
                or not metrics.get("complete") or metrics["identity"]["source_hash"] != source_hash()
                or metrics["identity"]["fit_lock_sha256"] != sha(evidence["root"]/"payoffs/fit_lock.json")
                or metrics["identity"]["original_index_sha256"] != sha(evidence["root"]/"comparison/accuracy_index.json")
                or metrics["identity"]["deployment_index_sha256"] != sha(evidence["root"]/"deployment/accuracy_index.json")):
            raise ValueError("Complete matching independent native and metric audits are required")
    if evidence["paired_initial_records"]:
        paired = accuracy_tables({**evidence, "records": evidence["paired_initial_records"],
                                  "groups": evidence["paired_initial_groups"]})
        tables["paired_initial_accuracy"] = paired["accuracy"]
        tables["paired_initial_sampling"] = paired["sampling"]
        tables["paired_initial_contrasts"] = paired["contrasts"]
    tables["model_summary"] = model_summary(tables["accuracy"])
    tables["training"] = [{"regime": m["case"], "name": m["name"], "seed": m["seed"],
                           "training_seconds": m["training_seconds"]} for m in evidence["lock"]["models"]]
    # State-bank manifests record actual generation seconds, including validation.
    data = []
    for regime in evidence["plan"]["cases"]:
        for role in ("train", "validation"):
            path = evidence["root"]/f"payoffs/{regime}/states_{role}.json"
            packet = read(_artifact(path, evidence["inputs"]))
            data.append({"regime": regime, "role": role, "generation_seconds": packet["generation_seconds"]})
    tables["data"] = data
    tables["workers"] = []
    for seed in evidence["study"].optimizer_seeds:
        path = evidence["root"]/f"payoffs/execution/worker_seed{seed}.json"
        worker = read(_artifact(path, evidence["inputs"]))
        if not worker.get("complete"):
            raise ValueError("All training workers must exit before final reporting")
        for i, attempt in enumerate(worker["attempts"]):
            tables["workers"].append({"seed": seed, "attempt": i, **attempt})
    tables["study_costs"] = [{"regime": c,
        "all_seed_fit_seconds": sum(m["training_seconds"] for m in evidence["lock"]["models"] if m["case"] == c),
        "shared_state_generation_seconds": sum(r["generation_seconds"] for r in data if r["regime"] == c),
        "reference_generation_seconds": evidence["references"][c]["generation_seconds"],
        "scope": "recorded component durations; overlaps are not elapsed batch time"} for c in evidence["plan"]["cases"]]
    data_seconds = {c: sum(r["generation_seconds"] for r in data if r["regime"] == c) for c in evidence["plan"]["cases"]}
    tables["costs"], tables["total_costs"] = cost_tables(evidence["timing"], evidence["lock"], data_seconds)
    tables["amortization"] = amortization_tables(tables)
    output = evidence["root"]/"report"
    output.mkdir(exist_ok=True)
    for name, rows in tables.items():
        csv_write(output/f"{name}.csv", rows)
    figures(output, tables)
    text = report_text(evidence, tables)
    (output/"report.md").write_text(text)
    if documentation:
        documentation = Path(documentation)
        prefix = f"artifacts/{documentation.stem}/report/"
        for p in output.glob("*.png"):
            text = text.replace(f"]({p.name})", f"]({prefix}{p.name})")
        documentation.write_text(text+f"\n[Compact evidence archive](artifacts/{documentation.stem}/README.md).\n")
    manifest = {"kind": KIND, "complete": True, "source_hash": source_hash(),
        "generator_sha256": sha(__file__), "inputs": evidence["inputs"],
        "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"},
        "models": len(evidence["lock"]["models"]), "accuracy_records": len(evidence["records"]),
        "original_accuracy_records": evidence["original_accuracy_records"],
        "independent_initial_records": evidence.get("independent_initial_records", 0),
        "total_retained_accuracy_records": evidence["original_accuracy_records"]+evidence.get("independent_initial_records", 0),
        "primary_NN_ensemble_initials_independent": independent_initials,
        "postfit_validation_diagnostics_included": validation_diagnostics,
        "timing_records": len(evidence["timing"]["records"]), "all_optimizer_seeds_retained": True,
        "test_based_refitting": False, "continuous_DK_claim": False,
        "qualified_payback_rows": sum(r["accuracy_qualified_payback_calls"] is not None for r in tables["amortization"])}
    json_write(output/"manifest.json", manifest)
    return manifest


def archive(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(root):
        raise ValueError("Use a new archive outside the authoritative run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".payoff-regimes-archive-", dir=destination.parent) as temporary:
        staging = Path(temporary)/"archive"
        _archive_contents(root, staging)
        staging.rename(destination)
    return verify_archive(destination)


def _archive_contents(root, destination):
    """Exact/lossless compact evidence, with full local raw-artifact inventory."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(root):
        raise ValueError("Use a new archive outside the authoritative run")
    manifest = read(root/"report/manifest.json")
    if not manifest.get("complete") or manifest["source_hash"] != source_hash() or manifest["generator_sha256"] != sha(__file__):
        raise ValueError("A complete matching report is required")
    for path, expected in manifest["inputs"].items():
        if sha(path) != expected:
            raise ValueError("A report input changed")
    for name, expected in manifest["outputs"].items():
        if sha(root/"report"/name) != expected:
            raise ValueError("A report output changed")
    tested = read(root/"audit/final_tests.json")
    if not tested.get("complete") or tested["source_hash"] != source_hash():
        raise ValueError("Final test evidence is required")
    destination.mkdir(parents=True)
    copied, inventory = {}, {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        relative = str(p.relative_to(root))
        inventory[relative] = {"sha256": sha(p), "bytes": p.stat().st_size}
        selected = p.suffix in (".json", ".py", ".md") or p.parent == root/"report"
        if selected:
            q = destination/relative
            compressed = p.suffix == ".json" and p.stat().st_size > 100000
            if compressed:
                q = q.with_name(q.name+".gz")
            q.parent.mkdir(parents=True, exist_ok=True)
            if compressed:
                with p.open("rb") as source, q.open("wb") as sink:
                    with gzip.GzipFile(fileobj=sink, mode="wb", filename="", mtime=0) as zipper:
                        shutil.copyfileobj(source, zipper)
            else:
                shutil.copyfile(p, q)
            copied[str(q.relative_to(destination))] = {"sha256": sha(q), "original_sha256": sha(p), "compressed": compressed}
    for p in [*(REPO/"src/nn_dk").rglob("*.py"), *(REPO/"analysis").glob("*.py"),
              *(REPO/"analysis").glob("*.cpp"), REPO/"pyproject.toml", REPO/"uv.lock"]:
        q = destination/"code"/p.relative_to(REPO)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        copied[str(q.relative_to(destination))] = {"sha256": sha(q), "original_sha256": sha(p), "compressed": False}
    (destination/"README.md").write_text("# Harder-regime comparison evidence\n\nSee the [report](report/report.md). "
        "All declared fits and independent outcomes remain in this checksum-indexed evidence archive. "
        "Large JSON is losslessly compressed. Full numerical banks and weights remain local, indexed by checksum. "
        "Code is MIT; authored prose, figures and tables are CC BY 4.0.\n")
    json_write(destination/"archive_manifest.json", {"complete": True, "source_hash": source_hash(),
        "report_manifest_sha256": sha(root/"report/manifest.json"), "copied": copied, "raw_inventory": inventory,
        "generated": {"README.md": sha(destination/"README.md")}, "raw_artifacts_retained_locally": True})
    return verify_archive(destination)


def verify_archive(destination):
    destination = Path(destination).resolve()
    manifest = read(destination/"archive_manifest.json")
    expected = set(manifest["copied"]) | set(manifest["generated"]) | {"archive_manifest.json"}
    if {str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()} != expected:
        raise ValueError("Archive inventory changed")
    for name, item in manifest["copied"].items():
        p = destination/name
        if not p.resolve().is_relative_to(destination) or sha(p) != item["sha256"]:
            raise ValueError("Archived evidence changed")
        restored = hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest() if item["compressed"] else sha(p)
        if restored != item["original_sha256"]:
            raise ValueError("Lossless original evidence checksum changed")
    for name, expected_sha in manifest["generated"].items():
        if sha(destination/name) != expected_sha:
            raise ValueError("Archive documentation changed")
    return {"complete": True, "copied_artifacts_verified": len(manifest["copied"]),
            "indexed_raw_files": len(manifest["raw_inventory"]), "manifest_sha256": sha(destination/"archive_manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("report", "archive", "verify-archive"))
    parser.add_argument("--run", type=Path, default=REPO/"runs/payoff-regimes-2026-10-27")
    parser.add_argument("--documentation", type=Path)
    parser.add_argument("--archive", type=Path, default=REPO/"docs/experiments/artifacts/payoff_regimes_2026-10-27")
    args = parser.parse_args()
    result = (generate(args.run, args.documentation) if args.action == "report" else
              archive(args.run, args.archive) if args.action == "archive" else verify_archive(args.archive))
    print(json.dumps(result if args.action != "report" else {k: v for k, v in result.items() if k not in ("inputs", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
