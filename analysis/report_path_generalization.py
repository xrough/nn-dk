"""Postprocess COMPLETE broad-task accuracy and independent cost experiments.

This script fits no model, draws no trajectory and selects no optimizer seed.
It refuses partial evaluation or development timing files. All conclusions are
restricted to the measured finite-grid targets and explicitly matched tasks.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

BANKS = ("terminal-modes34", "terminal3-integral1")
PERCENT = 100.


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _check_artifact(packet, cache):
    path = Path(packet["path"]).resolve()
    if path not in cache:
        cache[path] = sha(path)
    if cache[path] != packet["sha256"]:
        raise ValueError(f"Evidence artifact changed: {path}")


def load_complete(root, path_timing, payoff_timing=None, *, verify_artifacts=True, gaussian_audit=None):
    """Open final outcomes only once every predeclared experiment is complete."""
    root = Path(root).resolve()
    index_path = root / "evaluation/evaluation_index.json"
    path_timing = Path(path_timing).resolve()
    payoff_timing = Path(payoff_timing or root / "timing/payoff_benchmark.json").resolve()
    index = read(index_path)
    if not index.get("complete") or set(index["completed_groups"]) != set(index["declared_groups"]):
        raise ValueError("All evaluation groups and BOTH final, independent timing experiments must be complete")
    timing_paths, timing_payoffs = read(path_timing), read(payoff_timing)
    if (timing_paths.get("status") != "complete" or timing_payoffs.get("status") != "complete"
            or not timing_paths.get("final_timing_after_external_fit_completion")
            or not timing_payoffs.get("final_timing_after_external_worker_completion")):
        raise ValueError("All evaluation groups and BOTH final, independent timing experiments must be complete")
    design_path = root / "evaluation/preregistration.json"
    if sha(design_path) != index["preregistration_sha256"]:
        raise ValueError("Evaluation preregistration changed")
    design = read(design_path)
    observable_bank = timing_paths.get("many_observable_bank", {})
    if observable_bank.get("zero_time_included") is not False or observable_bank.get("dimension") != 480:
        raise ValueError("The final timing bank must be the literal 480 positive-time observations")
    native_method = ("compiled_native_euler" if timing_paths.get("compiled_classical_control", {}).get("available")
                     else "numpy_native_euler")
    expected_moment_slots = {(case, role, bank, method)
        for case in ("dk_interacting_uniform", "dk_interacting") for role in ("conditional", "IID-ensemble")
        for bank in BANKS for method in ("learned_dense_path", "exact_kernel_picard_depth3", native_method)}
    moment_rows = timing_paths.get("moment_tasks", [])
    actual_moment_slots = {(row["case"], row["role"], row["bank"], row["method"]) for row in moment_rows}
    if len(moment_rows) != len(expected_moment_slots) or actual_moment_slots != expected_moment_slots:
        raise ValueError("All 24 predeclared literal complete Monte Carlo query timings are required")
    if any(row["paths"] != design["paths_per_group"] or row["dt"] != 2 ** -10
           or row["query_times"] != design["reference_config"]["query_times"] for row in moment_rows):
        raise ValueError("Moment-query timing allocation or physical bank differs from final accuracy")
    fit_lock_path = root / "payoffs/fit_lock.json"
    fit_lock = read(fit_lock_path)
    if (not fit_lock.get("complete") or len(fit_lock.get("models", [])) != 12
            or fit_lock.get("expected_models") != 12
            or sha(fit_lock_path) != index["input_lock"]["fit_lock_sha256"]):
        raise ValueError("All twelve immutable new-payoff fits are required")
    sources = (index["source_hash"], fit_lock["source_hash"], timing_paths["source_hash"],
               timing_payoffs["identity"]["source_hash"])
    if len(set(sources)) != 1:
        raise ValueError("Accuracy and timing experiments use different package source freezes")
    cache, groups, records, audits = {}, [], [], []
    inputs = {str(path): sha(path) for path in (index_path, design_path, fit_lock_path,
                                              path_timing, payoff_timing)}
    if verify_artifacts:
        _check_artifact({"path": root / "reference/reference.h5",
                         "sha256": index["input_lock"]["reference_sha256"]}, cache)
    for key in index["declared_groups"]:
        group_path = root / "evaluation" / key / "group_results.json"
        group = read(group_path)
        inputs[str(group_path)] = sha(group_path)
        if (not group.get("complete") or not group.get("reference_audit_complete")
                or group["identity"]["input_lock"] != index["input_lock"]
                or group["identity"]["group"] != key):
            raise ValueError(f"Incomplete or incompatible evaluation group {key}")
        names = [item["name"] for item in group["methods"]]
        case = key.split("__")[0]
        expected = len([item for item in design["candidate_plan"] if item["case"] == case]) + 12 + 2
        if len(names) != len(set(names)) or len(names) != expected:
            raise ValueError(f"Predeclared candidate inventory is incomplete in {key}")
        if sha(group["reference_audit_path"]) != group["reference_audit_sha256"]:
            raise ValueError("Coupled reference audit changed")
        audit = read(group["reference_audit_path"])
        inputs[str(Path(group["reference_audit_path"]).resolve())] = group["reference_audit_sha256"]
        if not audit.get("complete"):
            raise ValueError("Reference refinement audit is incomplete")
        if verify_artifacts:
            _check_artifact(audit["bootstrap_artifact"], cache)
        audits.append({"group": key, **audit})
        for item in group["methods"]:
            packet = read(item["json_path"])
            inputs[str(Path(item["json_path"]).resolve())] = sha(item["json_path"])
            if (not packet.get("complete") or packet["method"]["name"] != item["name"]
                    or packet["identity"]["input_lock"] != index["input_lock"]):
                raise ValueError("A completed method result changed identity")
            if verify_artifacts:
                for field in ("bootstrap_artifact", "reduced_artifact"):
                    if field in packet:
                        _check_artifact(packet[field], cache)
            records.append({"group": key, **packet})
        if verify_artifacts:
            _check_artifact(group["reference_bootstrap"], cache)
        groups.append({"path": str(group_path), **group})
    if verify_artifacts:
        for case in timing_paths["cases"]:
            _check_artifact({"path": case["raw_arrays"], "sha256": case["raw_arrays_sha256"]}, cache)
        for pool in timing_paths.get("moment_input_pools", []):
            _check_artifact({"path": pool["artifact_path"], "sha256": pool["artifact_sha256"]}, cache)
        for row in timing_paths.get("moment_tasks", []):
            _check_artifact(row["output_artifact"], cache)
        for row in timing_payoffs["records"]:
            _check_artifact(row["output_artifact"], cache)
        for model in fit_lock["models"]:
            _check_artifact({"path": model["checkpoint_path"], "sha256": model["checkpoint_sha256"]}, cache)
            _check_artifact({"path": model["metadata_path"], "sha256": model["metadata_sha256"]}, cache)
    worker_paths = [root / f"payoffs/execution/worker_seed{seed}.json" for seed in (0, 1, 2)]
    workers = [read(path) for path in worker_paths]
    inputs.update({str(path): sha(path) for path in worker_paths})
    if any(not worker.get("complete") for worker in workers):
        raise ValueError("All optimizer workers must have completed before reporting")
    gaussian_path = Path(gaussian_audit or root / "evaluation/gaussian_marginals/index.json").resolve()
    marginal_index = read(gaussian_path)
    inputs[str(gaussian_path)] = sha(gaussian_path)
    if (marginal_index.get("status") != "complete" or marginal_index["source_hash"] != index["source_hash"]
            or marginal_index["reference_sha256"] != index["input_lock"]["reference_sha256"]
            or marginal_index["fit_lock_sha256"] != index["input_lock"]["fit_lock_sha256"]
            or marginal_index["expected_records"] != 12 or len(marginal_index["records"]) != 12):
        raise ValueError("The complete literal three-query Gaussian quality bridge is required")
    _check_artifact(marginal_index["preregistration"], cache)
    inputs[str(Path(marginal_index["preregistration"]["path"]).resolve())] = marginal_index["preregistration"]["sha256"]
    marginal_slots = set()
    for item in marginal_index["records"]:
        _check_artifact(item, cache)
        inputs[str(Path(item["path"]).resolve())] = item["sha256"]
        packet = read(item["path"])
        key = item["group"]
        if key not in index["declared_groups"] or packet["identity"]["group"] != key:
            raise ValueError("Supplemental Gaussian group is incompatible")
        slot = (key, packet["method"]["requested_max_step"])
        if slot in marginal_slots or packet["method"]["method"] != "linear-Gaussian-marginals":
            raise ValueError("Duplicate or incompatible Gaussian quality bridge")
        marginal_slots.add(slot)
        if verify_artifacts:
            _check_artifact(packet["bootstrap_artifact"], cache)
        records.append({"group": key, **packet})
    if marginal_slots != {(key, step) for key in index["declared_groups"] for step in (2 ** -7, 2 ** -10)}:
        raise ValueError("The Gaussian marginal audit must retain both steps in all six groups")
    training_design_path = root / "payoffs/preregistration.json"
    inputs[str(training_design_path)] = sha(training_design_path)
    return {"root": str(root), "index": index, "design": design, "fit_lock": fit_lock,
            "path_timing": timing_paths, "payoff_timing": timing_payoffs,
            "groups": groups, "records": records, "audits": audits, "workers": workers,
            "training_design": read(training_design_path), "input_files": inputs,
            "verified_artifact_count": len(cache)}


def _identity(packet):
    group = packet["group"]
    fields = group.split("__")
    method = packet["method"]
    return {"group": group, "case": fields[0], "role": fields[1], "anchor": fields[2],
            "method_name": method["name"], "method": method.get("method"),
            "optimizer_seed": method.get("seed"), "initialization": method.get("init"),
            "dt": method.get("dt"), "depth": method.get("depth"),
            "lga_max_step": method.get("requested_max_step")}


def flatten_accuracy(evidence):
    moments, many, components, strong, stopping, integrations, distributions, gaussian = [], [], [], [], [], [], [], []
    for packet in evidence["records"]:
        identity = _identity(packet)
        for bank, metric in packet["metrics"].items():
            if bank == "many480":
                many.append({**identity, **{key: value for key, value in metric.items()
                    if not isinstance(value, (list, dict))}})
                components.extend({**identity, "observable_index": index,
                    "normalized_mean_error": mean, "relative_variance_error": variance}
                    for index, (mean, variance) in enumerate(zip(metric["mean_errors"], metric["variance_errors"], strict=True)))
                continue
            tasks = ({"same_time": metric} if bank == "same_time" else
                     {"nonlinear_path_tasks": metric["moments"]} if bank == "nonlinear3" else
                     {key: value for key, value in metric.items() if key in ("same_time", "joint_time_12")})
            for task, summary in tasks.items():
                for row in summary["records"]:
                    moments.append({**identity, "bank": packet.get("bank", bank), "task": task,
                        "time_index": row["time_index"], "reference_paths": summary["n_reference"],
                        **{key: value for key, value in row.items() if not isinstance(value, (list, dict))}})
            distribution = metric.get("scalar_distributions")
            if distribution:
                distributions.extend({**identity, "bank": bank, "scalar_index": index,
                    "wasserstein1_in_reference_std": wasserstein, "ks_distance": ks}
                    for index, (wasserstein, ks) in enumerate(zip(distribution["wasserstein1_in_reference_std"],
                                                                  distribution["ks_distance"], strict=True)))
        for kind in ("strong_common_grid_sup", "strong_common_grid_time_rms"):
            if kind in packet:
                metric = packet[kind]
                strong.append({**identity, "task": kind, "normalized_rms": metric["normalized_rms"],
                               "lower95": metric["bootstrap95_interval"][0],
                               "upper95": metric["bootstrap95_interval"][1]})
        if "stopping" in packet:
            metric = packet["stopping"]
            stopping.append({**identity, "reference_probability": metric["reference_probability"],
                "candidate_probability": metric["candidate_probability"],
                "reference_lower95": metric["reference_exact_binomial95_interval"][0],
                "reference_upper95": metric["reference_exact_binomial95_interval"][1],
                "candidate_lower95": metric["candidate_exact_binomial95_interval"][0],
                "candidate_upper95": metric["candidate_exact_binomial95_interval"][1],
                "paired_difference": metric["paired_probability_difference"]})
        if "integration_uncertainty" in packet:
            metric = packet["integration_uncertainty"]
            integrations.append({**identity, "bank": packet["bank"], "initial_histograms": metric["initial_histograms"],
                "mean95_radius": max(metric.get("mean95_radius_in_reference_std_by_time", [0.])),
                "covariance95_normalized_radius": max(metric.get("covariance95_normalized_radius_by_time", [0.]))})
        if "gaussian_sampling_diagnostics" in packet:
            from scipy.stats import beta
            diagnostic = packet["gaussian_sampling_diagnostics"]
            n = diagnostic["n_paths"]
            row = {**identity, **{key: value for key, value in diagnostic.items()
                                if not isinstance(value, (dict, list))}}
            for field in ("negative_native_mass_fraction", "below_stopping_threshold_fraction"):
                count = int(round(n * diagnostic[field]))
                row[field + "_lower95"] = 0. if count == 0 else float(beta.ppf(.025, count, n - count + 1))
                row[field + "_upper95"] = 1. if count == n else float(beta.ppf(.975, count + 1, n - count))
            gaussian.append(row)
    return {"moments": moments, "many480_summary": many, "many480_components": components,
            "strong_paths": strong, "stopping": stopping, "initial_integration": integrations,
            "scalar_distributions": distributions, "gaussian_law_diagnostics": gaussian}


def flatten_reference(evidence):
    rows = []
    for group in evidence["audits"]:
        for comparison in group["records"]:
            common = {"group": group["group"], "coarse_factor": comparison["coarse_factor"],
                      "fine_factor": comparison["fine_factor"], "audit_paths": group["n_audited_families"]}
            for bank, metric in comparison["metrics"].items():
                if bank == "many480":
                    rows.append({**common, "bank": bank, "task": "means_and_variances",
                        "mean_point": metric["mean_max_std"], "covariance_point": metric["variance_max_relative"],
                        "mean_sampling95": metric["mean_sampling95_simultaneous_radius"],
                        "covariance_sampling95": metric["variance_sampling95_simultaneous_radius"]})
                else:
                    tasks = {"nonlinear_path_tasks": metric["moments"]} if bank == "nonlinear3" else {
                        key: value for key, value in metric.items() if key in ("same_time", "joint_time_12")}
                    for task, summary in tasks.items():
                        for row in summary["records"]:
                            rows.append({**common, "bank": bank, "task": task, "time_index": row["time_index"],
                                "mean_point": row["max_mean_error_in_reference_std"],
                                "covariance_point": row["max_normalized_covariance_error"],
                                "mean_sampling95": row["mean_sampling95_simultaneous_radius"],
                                "covariance_sampling95": row["covariance_sampling95_simultaneous_radius"]})
    return rows


def deployed_sampling(evidence):
    """Standalone candidate-estimator variability, separate from paired bias CIs."""
    n_particles = evidence["design"]["reference_config"]["n_particles"]
    reference_variances = {}
    for group in evidence["groups"]:
        with np.load(group["reference_bootstrap"]["path"]) as values:
            reference_variances[group["identity"]["group"]] = values["many_reference_variance"]
    rows = []
    for packet in evidence["records"]:
        identity = _identity(packet)
        if packet["method"].get("method") in ("prepared-learned-path", "native-euler", "exact-kernel-picard"):
            with np.load(packet["bootstrap_artifact"]["path"]) as bootstrap:
                for bank in BANKS:
                    metrics = packet["metrics"][bank]["same_time"]["records"]
                    mean = np.asarray([row["candidate_mean"] for row in metrics])
                    covariance = np.asarray([row["candidate_scaled_covariance"] for row in metrics])
                    reference = np.asarray([row["reference_scaled_covariance"] for row in metrics])
                    scaled_std = np.sqrt(np.diagonal(reference, axis1=-2, axis2=-1))
                    mean_radius = np.quantile(np.max(abs(bootstrap[bank + "_bootstrap_mean"] - mean)
                        / (scaled_std / np.sqrt(n_particles)), axis=-1), .95, axis=0)
                    covariance_radius = np.quantile(np.max(abs(bootstrap[bank + "_bootstrap_scaled_covariance"] - covariance)
                        / (scaled_std[..., :, None] * scaled_std[..., None, :]), axis=(-2, -1)), .95, axis=0)
                    for time_index in range(len(metrics)):
                        rows.append({**identity, "bank": bank, "time_index": time_index,
                            "mean95_radius": float(mean_radius[time_index]),
                            "covariance95_normalized_radius": float(covariance_radius[time_index]),
                            "scope": "standalone deployed trajectory-Monte-Carlo variability; not paired method-bias uncertainty"})
                if "many_bootstrap_mean" in bootstrap:
                    variance = reference_variances[packet["group"]]
                    mean_radius = np.quantile(np.max(abs(bootstrap["many_bootstrap_mean"] - bootstrap["many_mean"])
                        / np.sqrt(variance), axis=-1), .95)
                    covariance_radius = np.quantile(np.max(abs(bootstrap["many_bootstrap_variance"] - bootstrap["many_variance"])
                        / variance, axis=-1), .95)
                    rows.append({**identity, "bank": "many480", "time_index": "all-positive-common-times",
                        "mean95_radius": float(mean_radius), "covariance95_normalized_radius": float(covariance_radius),
                        "scope": "standalone 480-output candidate MC variability, simultaneous over marginal bank"})
        elif packet["method"].get("method") == "statistics-head":
            uncertainty = packet["integration_uncertainty"]
            rows.append({**identity, "bank": packet["bank"], "time_index": "maximum-over-three-times",
                "mean95_radius": max(uncertainty.get("mean95_radius_in_reference_std_by_time", [0.])),
                "covariance95_normalized_radius": max(uncertainty.get("covariance95_normalized_radius_by_time", [0.])),
                "scope": "initial-law integration MC" if identity["role"] == "ensemble" else "conditional deterministic head; zero inference Monte Carlo"})
        elif packet["method"].get("method") in ("linear-Gaussian", "linear-Gaussian-marginals"):
            for bank in BANKS:
                rows.append({**identity, "bank": bank, "time_index": "all-three-times",
                    "mean95_radius": 0., "covariance95_normalized_radius": 0.,
                    "scope": "deterministic Gaussian marginal moments; unresolved law bias is separate"})
    return rows


def flatten_costs(evidence):
    rows = []
    for case in evidence["path_timing"]["cases"]:
        for key in ("records", "derivative_records"):
            for item in case[key]:
                rows.append({"benchmark": "path", "case": item["case"], "method": item["method"],
                    "task": item["task"], "batch": item["batch_paths"], "steps": item["steps"],
                    "horizon": item["horizon"], "dt": item["dt"], "bank": None, "optimizer_seed": 0
                    if item["method"].startswith(("learned", "legacy_learned")) else None,
                    "median_seconds": item["median_seconds"], "p10_seconds": item["p10_seconds"],
                    "p90_seconds": item["p90_seconds"], "scope": item.get("timing_scope"),
                    "initial_gradient_relative_rms": item.get("initial_error", {}).get("relative_rms_error"),
                    "initial_gradient_rms_error": item.get("initial_error", {}).get("rms_error"),
                    "initial_gradient_reference_rms": item.get("initial_error", {}).get("reference_rms"),
                    "physics_gradient_relative_rms": item.get("physics_error", {}).get("relative_rms_error"),
                    "physics_gradient_rms_error": item.get("physics_error", {}).get("rms_error"),
                    "physics_gradient_reference_rms": item.get("physics_error", {}).get("reference_rms"),
                    "jvp_relative_rms": item.get("tangent_error", {}).get("relative_rms_error"),
                    "jvp_rms_error": item.get("tangent_error", {}).get("rms_error"),
                    "jvp_reference_rms": item.get("tangent_error", {}).get("reference_rms"),
                    "physics_interpretation": item.get("physics_interpretation")})
    for item in evidence["path_timing"].get("moment_tasks", []):
        rows.append({"benchmark": "path-moments", "case": item["case"].replace("dk_interacting_uniform", "uniform").replace("dk_interacting", "nonuniform"),
            "method": item["method"], "task": "conditional_same_time_moments" if item["role"] == "conditional" else "initial_ensemble_same_time_moments",
            "bank": item["bank"], "role": item["role"], "batch": item["paths"], "chunk_size": item["chunk_size"],
            "dt": item["dt"], "horizon": item["horizon"], "optimizer_seed": item.get("optimizer_seed"),
            "median_seconds": item["median_seconds"], "p10_seconds": item["p10_seconds"], "p90_seconds": item["p90_seconds"],
            "initialization_seconds_separate": item["initialization_seconds"], "physical_noise_seconds_separate": item["physical_noise_seconds"],
            "bank_readout_median_seconds": float(np.median(item["new_bank_readout_component_seconds"])),
            "simulation_median_seconds": float(np.median(item["simulation_component_seconds"])),
            "aggregation_median_seconds": float(np.median(item["aggregation_component_seconds"])),
            "scope": item["cost_scope"]})
    for item in evidence["payoff_timing"]["records"]:
        ensemble = item["role"] == "IID-ensemble"
        cost = item["fresh_initialization_inference_aggregation"] if ensemble and "fresh_initialization_inference_aggregation" in item else item
        rows.append({"benchmark": "payoff", "case": item["case"], "method": item["method"],
            "task": "initial_ensemble_same_time_moments" if ensemble else "conditional_same_time_moments",
            "bank": item["bank"], "role": item["role"], "batch": item.get("batch_states", item.get("initial_histograms", 0)),
            "dt": item.get("step"), "optimizer_seed": item.get("optimizer_seed"), "initialization": item.get("init"),
            "median_seconds": cost["median_seconds"], "p10_seconds": cost.get("p10_seconds"),
            "p90_seconds": cost.get("p90_seconds"), "fresh_training_seconds": item.get("fresh_head_training_seconds", 0.),
            "historical_training_seconds": item.get("historical_trunk_training_seconds", 0.),
            "scope": "literal same-time four-means/full-4x4-covariance; fresh initial-law integration included for NN ensemble"})
    return rows


def preparation_rows(evidence):
    rows = []
    for case in evidence["path_timing"]["cases"]:
        rows.extend({"component": "path-runtime-and-kernel-preparation", **{key: value for key, value in item.items()
            if not isinstance(value, (dict, list))}} for item in case["preparation"])
        rows.append({"component": "historical-path-restoration", "case": case["records"][0]["case"],
                     "seconds": case["checkpoint_loading_seconds"],
                     "checkpoint_sha256": case["checkpoint"]["checkpoint_sha256"],
                     "scope": "explicit historical source-strict restore; process/OS caches uncontrolled"})
    for item in evidence["payoff_timing"]["cold_restoration"]:
        rows.append({"component": "payoff-head-restoration", **{key: value for key, value in item.items()
            if not isinstance(value, (dict, list))}, "scope": evidence["payoff_timing"].get("cold_cache_scope")})
    for item in evidence["payoff_timing"]["initialization"]:
        rows.append({"component": "payoff-ensemble-multinomial-initialization", **{key: value for key, value in item.items()
            if not isinstance(value, (dict, list))}})
    compiler = evidence["path_timing"].get("compiled_classical_control", {})
    if compiler.get("available"):
        rows.append({"component": "optional-native-control-compilation", "seconds": compiler["compile_seconds"],
            "seconds_this_timing_process": compiler["compile_seconds_this_process"], "cache_hit": compiler["cache_hit"],
            "compiler": compiler["compiler"], "library_sha256": compiler["library_sha256"],
            "cpp_sha256": compiler["cpp_sha256"], "scope": "optional serial FP64 analysis control; no fast-math"})
    return rows
def accuracy_gate(rows, reference_rows, *, tolerance=.05, uncertainty=.0125,
                  standalone_mc_radius=None):
    """A finite-reference diagnostic gate, never a continuous-DK certificate."""
    if not rows:
        return "unsupported"
    point = max(max(row["max_mean_error_in_reference_std"], row["max_normalized_covariance_error"]) for row in rows)
    radius = max(max(row["mean_sampling95_simultaneous_radius"], row["covariance_sampling95_simultaneous_radius"]) for row in rows)
    upper = max(max(row["mean_error_plus_sampling95"], row["covariance_error_plus_sampling95"]) for row in rows)
    if point > tolerance:
        return "point-error-above-target"
    if upper > tolerance or radius > uncertainty:
        return "sampling-unresolved"
    if standalone_mc_radius is None:
        return "standalone-MC-not-assessed"
    if standalone_mc_radius > uncertainty:
        return "deployed-MC-unresolved"
    if not reference_rows or max(max(row["mean_point"] + row["mean_sampling95"],
                                    row["covariance_point"] + row["covariance_sampling95"])
                                  for row in reference_rows) > uncertainty:
        return "coupled-reference-unresolved"
    return "finite-reference-diagnostic-only"


def break_even(training_seconds, candidate_cost, baseline_cost, *, candidate_gate, baseline_gate):
    """Only compare repeated, literal matched queries after the accuracy gates."""
    if candidate_gate != "finite-reference-diagnostic-only" or baseline_gate != "finite-reference-diagnostic-only":
        return {"queries": None, "reason": "Matched finite-reference accuracy/uncertainty is unresolved"}
    saving = baseline_cost - candidate_cost
    if saving <= 0:
        return {"queries": None, "reason": "No positive measured online saving"}
    ratio = max(training_seconds, 0.) / saving
    return {"queries": int(np.ceil(np.nextafter(ratio, -np.inf))),
            "reason": "Measured online costs and declared training; finite-reference scope only"}


def amortization_rows(evidence, tables, costs, references):
    """Leave crossovers undefined unless the declared tasks and precision match."""
    rows, grouped = [], defaultdict(list)
    for row in tables["moments"]:
        if row["task"] == "same_time" and row["method"] in ("statistics-head", "linear-Gaussian", "linear-Gaussian-marginals"):
            grouped[(row["case"], row["role"], row["bank"], row["method_name"])].append(row)
    for (case, role, bank, name), values in grouped.items():
        head = values[0]
        if head["method"] != "statistics-head":
            continue
        cost = _matched_cost(head, costs)
        if cost is None:
            continue
        deployment = [row for row in tables["deployed_sampling"] if row["case"] == case and row["role"] == role
                      and row["bank"] == bank and row["method_name"] == name]
        reference = [row for row in references if row["bank"] == bank and row["task"] == "same_time"
                     and row["group"].split("__")[:2] == [case, role] and row["coarse_factor"] == 1 and row["fine_factor"] == 4]
        candidate_gate = accuracy_gate(values, reference,
            standalone_mc_radius=max((max(row["mean95_radius"], row["covariance95_normalized_radius"])
                                      for row in deployment), default=None))
        for (other_case, other_role, other_bank, baseline), other in grouped.items():
            if (other_case, other_role, other_bank) != (case, role, bank) or other[0]["method"] not in ("linear-Gaussian", "linear-Gaussian-marginals"):
                continue
            other_cost = _matched_cost(other[0], costs)
            if other_cost is None:
                continue
            baseline_gate = accuracy_gate(other, reference, standalone_mc_radius=0.)
            marginal = break_even(cost.get("fresh_training_seconds", 0.), cost["median_seconds"],
                other_cost["median_seconds"], candidate_gate=candidate_gate, baseline_gate=baseline_gate)
            complete = break_even(cost.get("fresh_training_seconds", 0.) + cost.get("historical_training_seconds", 0.),
                cost["median_seconds"], other_cost["median_seconds"], candidate_gate=candidate_gate, baseline_gate=baseline_gate)
            rows.append({"case": case, "role": role, "bank": bank, "candidate": name, "baseline": baseline,
                "optimizer_seed": head["optimizer_seed"], "initialization": head["initialization"],
                "candidate_gate": candidate_gate, "baseline_gate": baseline_gate,
                "candidate_online_seconds": cost["median_seconds"], "baseline_online_seconds": other_cost["median_seconds"],
                "marginal_new_head_fitting_queries": marginal["queries"], "marginal_reason": marginal["reason"],
                "including_historical_trunk_queries": complete["queries"], "complete_reason": complete["reason"],
                "cost_scope": "literal same-bank queries; data generation/restoration separate; existing-pool adaptation vs fitting recorded",
                "certification_scope": "empirical finite-reference diagnostic only; no independent combined coverage or continuous-DK claim"})
    return rows


def task_summary_rows(tables, costs):
    """Keep scientific task summaries separate from unsupported timing joins."""
    grouped = defaultdict(list)
    for row in tables["moments"]:
        grouped[(row["group"], row["method_name"], row["bank"], row["task"])].append(row)
    summaries = []
    for (_, _, bank, task), rows in grouped.items():
        identity = rows[0]
        sampling = [row for row in tables["deployed_sampling"]
                    if row["group"] == identity["group"]
                    and row["method_name"] == identity["method_name"] and row["bank"] == bank]
        cost = _matched_cost(identity, costs) if task == "same_time" else None
        summaries.append({**{key: identity[key] for key in (
            "group", "case", "role", "anchor", "method_name", "method", "optimizer_seed",
            "initialization", "dt", "depth", "lga_max_step")}, "bank": bank, "task": task,
            "mean_max_std": max(row["max_mean_error_in_reference_std"] for row in rows),
            "covariance_max_normalized": max(row["max_normalized_covariance_error"] for row in rows),
            "comparison_mean95_radius": max(row["mean_sampling95_simultaneous_radius"] for row in rows),
            "comparison_covariance95_radius": max(row["covariance_sampling95_simultaneous_radius"] for row in rows),
            "standalone_mean95_radius": max((row["mean95_radius"] for row in sampling), default=None)
                if task == "same_time" else None,
            "standalone_covariance95_radius": max((row["covariance95_normalized_radius"] for row in sampling), default=None)
                if task == "same_time" else None,
            "literal_query_median_seconds": None if cost is None else cost["median_seconds"],
            "cost_scope": "unsupported or separately timed task" if cost is None else cost["scope"],
            "time_summary": "maximum over the declared task's observation times"})
    return summaries


def write_csv(path, rows):
    path = Path(path)
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else ["no_records"]
    opening = gzip.open if path.suffix == ".gz" else open
    with opening(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _range(values, *, scale=100., digits=3):
    values = [value for value in values if value is not None]
    if not values:
        return "unavailable"
    return f"{min(values) * scale:.{digits}g}–{max(values) * scale:.{digits}g}"


def _plot_paths(plt, tables, output):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for column, case in enumerate(("uniform", "nonuniform")):
        for family, color in (("prepared-learned-path", "tab:blue"), ("native-euler", "tab:orange")):
            grouped = defaultdict(list)
            for row in tables["strong_paths"]:
                if row["case"] == case and row["method"] == family and row["task"] == "strong_common_grid_sup":
                    grouped[(row["group"], row["optimizer_seed"])].append(row)
            for index, values in enumerate(grouped.values()):
                values.sort(key=lambda row: row["dt"])
                axes[0, column].plot([row["dt"] for row in values], [100 * row["normalized_rms"] for row in values],
                    color=color, marker="o", linewidth=1, alpha=.65, label=family if index == 0 else None)
                axes[0, column].vlines([row["dt"] for row in values], [100 * row["lower95"] for row in values],
                                       [100 * row["upper95"] for row in values], color=color, alpha=.65)
            grouped = defaultdict(list)
            for row in tables["many480_summary"]:
                if row["case"] == case and row["method"] == family:
                    grouped[(row["group"], row["optimizer_seed"])].append(row)
            for index, values in enumerate(grouped.values()):
                values.sort(key=lambda row: row["dt"])
                axes[1, column].plot([row["dt"] for row in values], [100 * row["variance_max_relative"] for row in values],
                                     color=color, marker="o", linewidth=1, alpha=.65, label=family if index == 0 else None)
        axes[0, column].set_title(case)
        for axis in axes[:, column]:
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Fitted/integration step Δ")
            axis.grid(alpha=.2)
        axes[1, column].axhline(5, color="gray", linestyle="--", label="5% statistical research target")
        axes[0, column].set_ylabel("Sup-path RMS / IID fluctuation scale (%)")
        axes[1, column].set_ylabel("Worst relative variance error, 480 outputs (%)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Short T=1/16: errors versus fine Euler; separately fitted NN steps, all seeds retained")
    figure.savefig(output / "path_refinement.png", dpi=180)
    plt.close(figure)


def _plot_many(plt, tables, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = [f"path_dt10_seed{seed}" for seed in (0, 1, 2)] + ["native_euler_dt10"] + [
        f"classical_picard_depth{depth}" for depth in (1, 2, 3)] + ["linear_gaussian_rk4_7", "linear_gaussian_rk4_10"]
    for axis, case in zip(axes, ("uniform", "nonuniform"), strict=True):
        values, names = [], []
        for label in labels:
            selected = [100 * row["relative_variance_error"] for row in tables["many480_components"]
                        if row["case"] == case and row["method_name"] == label]
            if selected:
                values.append(selected)
                names.append(label.replace("path_dt10_", "NN ").replace("classical_picard_depth", "Picard d").replace(
                    "linear_gaussian_rk4_", "LGA ").replace("native_euler_dt10", "Euler"))
        axis.boxplot(values, tick_labels=names, showfliers=False)
        axis.tick_params(axis="x", rotation=70)
        axis.axhline(0, color="gray", linewidth=.7)
        axis.set_title(case)
        axis.set_ylabel("Relative variance discrepancy (%)")
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("480 marginal variances across fresh groups; no 480×480 covariance claim")
    figure.savefig(output / "many_variances.png", dpi=180)
    plt.close(figure)


def _plot_heads(plt, tables, output):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, bank in enumerate(BANKS):
        for row_index, role in enumerate(("conditional", "ensemble")):
            axis = axes[row_index, column]
            for seed, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
                for index, initialization in enumerate(("cold", "trunk-transfer")):
                    rows = [row for row in tables["moments"] if row["method"] == "statistics-head"
                        and row["bank"] == bank and row["optimizer_seed"] == seed
                        and row["role"] == role and row["initialization"] == initialization]
                    if not rows:
                        continue
                    point = max(row["max_normalized_covariance_error"] for row in rows)
                    upper = max(row["covariance_error_plus_sampling95"] for row in rows)
                    axis.errorbar(index + (seed - 1) * .1, point * 100, yerr=[[0], [(upper - point) * 100]],
                                  marker="o", color=color, label=f"seed {seed}" if index == 0 else None)
            axis.set_xticks((0, 1), ("cold", "trunk transfer"))
            axis.axhline(5, color="gray", linestyle="--")
            axis.set_title(bank + " / " + role)
            axis.set_ylabel("Worst same-time covariance discrepancy (%)")
            axis.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("New payoff heads: maxima over profiles, anchors and times; sampling-only upper bars")
    figure.savefig(output / "novel_payoff_heads.png", dpi=180)
    plt.close(figure)


def _plot_costs(plt, costs, output, *, derivatives=False):
    tasks = ("scalar_initial_and_physics_reverse", "initial_path_jvp") if derivatives else ("full_density_path", "many_new_observables")
    names = ({"compiled_native_analytic_adjoint", "compiled_native_analytic_jvp", "numpy_native_analytic_adjoint",
              "numpy_native_analytic_jvp", "torch_native_euler", "learned_dense_cached_depth3",
              "learned_causal_block_fft_depth3", "exact_kernel_dense_cached_depth3"} if derivatives else
             {"compiled_native_euler", "numpy_native_euler", "torch_native_euler", "legacy_learned_torch",
              "learned_dense_cached_depth3", "learned_causal_block_fft_depth3", "exact_kernel_dense_cached_depth3"})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, case in enumerate(("dk_interacting_uniform", "dk_interacting")):
        for row_index, task in enumerate(tasks):
            axis = axes[row_index, column]
            grouped = defaultdict(list)
            for row in costs:
                if row["benchmark"] == "path" and row["case"] == case and row["steps"] == 64 and row["task"] == task and row["method"] in names:
                    grouped[row["method"]].append(row)
            for label, values in grouped.items():
                values.sort(key=lambda row: row["batch"])
                axis.plot([row["batch"] for row in values], [1000 * row["median_seconds"] for row in values], marker="o", label=label)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xlabel("Realizations in one call")
            axis.set_ylabel("Complete measured latency (ms)")
            axis.set_title(case + " / " + task)
            axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6)
    figure.suptitle("Short T=1/16, Δ=1/1024; fixed supplied inputs; cold preparation separate")
    figure.savefig(output / ("derivative_costs.png" if derivatives else "forward_costs.png"), dpi=180)
    plt.close(figure)


def _matched_cost(row, costs):
    role = "conditional_same_time_moments" if row["role"] == "conditional" else "initial_ensemble_same_time_moments"
    candidates = [cost for cost in costs if cost["benchmark"] in ("payoff", "path-moments") and cost["case"] == row["case"]
        and cost["bank"] == row["bank"] and cost["task"] == role
        and (cost["benchmark"] == "path-moments" or role != "conditional_same_time_moments" or cost["batch"] == 1)]
    if row["method"] == "statistics-head":
        candidates = [cost for cost in candidates if cost["method"] == "new-payoff-statistics-head"
                      and cost["optimizer_seed"] == row["optimizer_seed"] and cost["initialization"] == row["initialization"]]
    elif row["method"] in ("linear-Gaussian", "linear-Gaussian-marginals"):
        # The full common-grid coarse ODE uses finer forced knots than the
        # marginal-only cost control. Supplemental marginal quality bridges it.
        if row["method"] == "linear-Gaussian" and row["lga_max_step"] != 2 ** -10:
            return None
        candidates = [cost for cost in candidates if cost["method"] == "native-linear-Gaussian-RK4"
                      and cost["dt"] == row["lga_max_step"]]
    elif row["method"] in ("prepared-learned-path", "native-euler", "exact-kernel-picard"):
        names = {"prepared-learned-path": ("learned_dense_path",),
                 "native-euler": ("compiled_native_euler", "numpy_native_euler"),
                 "exact-kernel-picard": ("exact_kernel_picard_depth3",)}
        candidates = [cost for cost in candidates if cost["method"] in names[row["method"]]
                      and row["dt"] == 2 ** -10
                      and (row["method"] != "prepared-learned-path" or row["optimizer_seed"] == 0)
                      and (row["method"] != "exact-kernel-picard" or row["depth"] == 3)]
    else:
        return None
    return candidates[0] if len(candidates) == 1 else None


def _plot_accuracy_cost(plt, tables, costs, output):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, bank in enumerate(BANKS):
        for row_index, role in enumerate(("conditional", "ensemble")):
            axis = axes[row_index, column]
            for row in tables["moments"]:
                if row["bank"] != bank or row["task"] != "same_time" or row["role"] != role:
                    continue
                cost = _matched_cost(row, costs)
                if cost is None:
                    continue
                is_head = row["method"] == "statistics-head"
                is_lga = row["method"] in ("linear-Gaussian", "linear-Gaussian-marginals")
                axis.scatter(1000 * cost["median_seconds"], 100 * row["covariance_error_plus_sampling95"],
                    marker="o" if is_head else "s" if is_lga else "^", color=("tab:blue" if row["initialization"] == "cold" else "tab:orange")
                    if is_head else "tab:green" if is_lga else "tab:red", alpha=.5, s=20)
            axis.set_xscale("log")
            axis.axhline(5, color="gray", linestyle="--")
            axis.set_xlabel("Literal matched moment-query latency (ms)")
            axis.set_ylabel("Method discrepancy + comparison sampling radius (%)")
            axis.set_title(bank + " / " + role)
            axis.grid(alpha=.2)
    figure.suptitle("Same-time 4×4 moments: blue=cold, orange=transfer, green=LGA, red=path MC; standalone MC separate")
    figure.savefig(output / "accuracy_cost_matched_moments.png", dpi=180)
    plt.close(figure)


def _plot_deployed_sampling(plt, tables, costs, output):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, bank in enumerate(BANKS):
        for row_index, role in enumerate(("conditional", "ensemble")):
            axis = axes[row_index, column]
            for row in tables["deployed_sampling"]:
                if row["bank"] != bank or row["role"] != role:
                    continue
                cost = _matched_cost(row, costs)
                if cost is None:
                    continue
                is_head = row["method"] == "statistics-head"
                is_lga = row["method"] in ("linear-Gaussian", "linear-Gaussian-marginals")
                axis.scatter(1000 * cost["median_seconds"], 100 * row["covariance95_normalized_radius"],
                    marker="o" if is_head else "s" if is_lga else "^",
                    color="tab:blue" if is_head else "tab:green" if is_lga else "tab:red", alpha=.5, s=20)
            axis.set_xscale("log")
            axis.axhline(1.25, color="gray", linestyle="--")
            axis.set_xlabel("Literal matched moment-query latency (ms)")
            axis.set_ylabel("Standalone candidate MC covariance 95% radius (%)")
            axis.set_title(bank + " / " + role)
            axis.grid(alpha=.2)
    figure.suptitle("Deployed estimator variability, separate from paired bias; deterministic approximation bias is NOT zero")
    figure.savefig(output / "deployed_sampling_cost.png", dpi=180)
    plt.close(figure)


def make_plots(tables, costs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_paths(plt, tables, output)
    _plot_many(plt, tables, output)
    _plot_heads(plt, tables, output)
    _plot_costs(plt, costs, output)
    _plot_costs(plt, costs, output, derivatives=True)
    _plot_accuracy_cost(plt, tables, costs, output)
    _plot_deployed_sampling(plt, tables, costs, output)


def render_markdown(evidence, tables, costs, references):
    native = [row for row in costs if row["benchmark"] == "path" and row["method"] == "compiled_native_euler"
              and row["task"] == "full_density_path" and row["steps"] == 64 and row["batch"] == 1 and row["horizon"] == 1 / 16]
    learned = [row for row in costs if row["benchmark"] == "path" and row["method"] == "learned_dense_cached_depth3"
               and row["task"] == "full_density_path" and row["steps"] == 64 and row["batch"] == 1 and row["horizon"] == 1 / 16]
    ratios = [nn["median_seconds"] / classical["median_seconds"] for nn in learned for classical in native if nn["case"] == classical["case"]]
    head_rows = [row for row in tables["moments"] if row["method"] == "statistics-head"]
    lga_rows = [row for row in tables["moments"] if row["method"] in ("linear-Gaussian", "linear-Gaussian-marginals") and row["task"] == "same_time"]
    path_rows = [row for row in tables["strong_paths"] if row["method"] == "prepared-learned-path" and row["task"] == "strong_common_grid_sup" and row["dt"] == 2 ** -10]
    reference_table = ["| Coupled 2⁻¹³ → 2⁻¹⁵ audit | Mean point (%) | Covariance/variance point (%) | Mean sampling radius (%) | Covariance sampling radius (%) |",
                       "|---|---:|---:|---:|---:|"]
    for bank, task in ((BANKS[0], "same_time"), (BANKS[0], "joint_time_12"),
                       (BANKS[1], "same_time"), (BANKS[1], "joint_time_12"),
                       ("many480", "means_and_variances"), ("nonlinear3", "nonlinear_path_tasks")):
        selected = [row for row in references if row["bank"] == bank and row["task"] == task
                    and row["coarse_factor"] == 1 and row["fine_factor"] == 4]
        reference_table.append(f"| {bank} / {task} | {_range([row['mean_point'] for row in selected])} | {_range([row['covariance_point'] for row in selected])} | {_range([row['mean_sampling95'] for row in selected])} | {_range([row['covariance_sampling95'] for row in selected])} |")
    nonlinear_table = ["| Fixed 17-node nonlinear tasks | Mean point (% SD) | Covariance point (%) | Comparison covariance sampling radius (%) |",
                       "|---|---:|---:|---:|"]
    nonlinear_names = [f"path_dt10_seed{seed}" for seed in (0, 1, 2)] + ["native_euler_dt10"] + [
        f"classical_picard_depth{depth}" for depth in (1, 2, 3)] + ["linear_gaussian_rk4_7", "linear_gaussian_rk4_10"]
    for name in nonlinear_names:
        selected = [row for row in tables["moments"] if row["bank"] == "nonlinear3" and row["method_name"] == name]
        nonlinear_table.append(f"| {name} | {_range([row['max_mean_error_in_reference_std'] for row in selected])} | {_range([row['max_normalized_covariance_error'] for row in selected])} | {_range([row['covariance_sampling95_simultaneous_radius'] for row in selected])} |")
    long_table = ["| Optional T=1 complete-path timing | Batch | Measured median (ms) |",
                  "|---|---:|---:|"]
    for row in costs:
        if (row["benchmark"] == "path" and row.get("horizon") == 1. and row["task"] == "full_density_path"
                and row["method"] in ("compiled_native_euler", "numpy_native_euler", "learned_dense_cached_depth4",
                                       "learned_causal_block_fft_depth4", "exact_kernel_dense_cached_depth4")):
            long_table.append(f"| {row['method']} | {row['batch']} | {1000 * row['median_seconds']:.4g} |")
    compiled_statement = (f"At Δ=1/1024 and one realization, the prepared learned path runtime takes **{_range(ratios, scale=1)}×** the compiled native Euler latency."
                          if ratios else "The complete timing inventory contains no comparable compiled single-path row; no forward winner is asserted.")
    lines = ["# Broader interacting DK task comparison — 26 October 2026", "",
        compiled_statement,
        f"Fresh statistical heads have normalized mean point errors **{_range([row['max_mean_error_in_reference_std'] for row in head_rows])}%** and same-time covariance point errors **{_range([row['max_normalized_covariance_error'] for row in head_rows])}%** across all declared groups, times, seeds and initializations. These ranges are descriptive; the sampling and coupled-reference diagnostics below control their interpretation.",
        "A full density-path model supplies realizations, new observables, joint-time covariances and pathwise derivatives. The fixed payoff heads supply their declared same-time means and 4×4 covariances. Capability alone is not a measured runtime or accuracy advantage.", "",
        "## What was compared", "",
        "The target is the stopped native finite-volume DK system with G=32, N=2048, D=1 and V(x)=−0.75 cos(x)+0.1 cos(2x). Initial histograms are IID multinomial draws from uniform or exp(0.2 cos(x)) profiles. The scientific comparison uses T=1/16, 16,384 independent families per group, four fresh conditional histograms and two initial-law ensemble groups. Every stopped trajectory is retained. Observation weights integrate physical Fourier functions exactly against cellwise densities.",
        "The reference step is 2⁻¹³, with literally coupled 2⁻¹⁴/2⁻¹⁵ audits on 1,024 families per group. Whole-family bootstrap intervals are simultaneous within each reported bank. Numerical reference differences and sampling uncertainty are separate; continuous hitting-boundary and continuum-DK bias remain unresolved.",
        "Historical path operators have width 16 and 64 training paths per profile. New payoff heads have width 32 and 128 paths per profile, sharing one literal current-state/horizon pool across both new banks, all optimizer seeds and cold/trunk-transfer starts. Thus this is a deployed/adaptation comparison, not an equal-data, equal-capacity sample-efficiency experiment.", "",
        "Trunk-transfer starts retain prior optimization and data exposure from the historical statistics task; their new output head is reset. Cold and transferred fits share the fresh pool, but their total historical training exposure differs and is costed separately.", "",
        "The density-path ansatz learns its causal Volterra response kernel while retaining the known interaction force and density-dependent physical face-noise mobility. Each integration resolution has a separately fitted checkpoint. The refinement ladder does not establish time convergence of a single frozen network. The statistical heads learn full-native-state conditional moment functions from the known-generator residual, with exact deterministic noise-trace calculations and no future-payoff training labels.", "",
        "## Accuracy and uncertainty", "",
        f"The finest learned paths have normalized strong sup-path point errors **{_range([row['normalized_rms'] for row in path_rows])}%** relative to the finite reference, on all cells and the common physical-time grid.",
        f"The no-training linear-Gaussian approximation has same-time normalized covariance point discrepancies **{_range([row['max_normalized_covariance_error'] for row in lga_rows])}%**. Its law is an unstopped Gaussian linearization; agreement here does not establish stopped-DK equivalence.",
        "![](path_refinement.png)", "![](many_variances.png)", "![](novel_payoff_heads.png)", "",
        "The 480-output bank reports marginal means and variances only. Full joint-time covariance is assessed separately for each 12-dimensional B/C bank (four observables at three times). The statistical heads do not predict these joint-time matrices, scalar distributions, full density realizations or stopping probabilities.", "",
        "\n".join(reference_table),
        "The table gives ranges over all declared profiles, groups and times, normalized using each audit's finer reference. Coupled differences assess finite-step sensitivity and do not enclose the remaining continuous-time boundary bias. All three pairwise refinement comparisons remain in reference_refinement.csv.", "",
        "\n".join(nonlinear_table),
        "These nonlinear functionals are terminal normalized cos(3x), a density-energy integral and a smooth maximum on the fixed common polygon. Scalar Wasserstein/KS diagnostics are retained separately; their empirical values are not population distribution certificates. Gaussian nonlinear outcomes use independent samples of an unstopped approximate law.", "",
        "## Cost and the role of learning", "",
        "Complete path timings include stopping; many-observable timings include the readout. Initial histogram generation, physical Brownian generation, tensor conversion, checkpoint restoration and kernel preparation are recorded separately. The optional serial C++ controls use FP64 without fast-math, with source/compiler/flag/library fingerprints and cold compilation cost. They supply full forward paths, conservative initial JVPs and the complete discrete adjoint. Exact-kernel Picard controls isolate improvements caused by cached/parallel algebra from improvements requiring trained weights.",
        "![](forward_costs.png)", "![](derivative_costs.png)", "",
        "Initial sensitivities are projected onto mass-zero directions. Physical-parameter AD of a neural checkpoint trained only at fixed D, κ and a₂ is a diagnostic derivative of that parametrization. Finite-difference agreement validates the derivative implementation; it does not validate parameter generalization or an inverse-problem solver.",
        "Sensitivity timing objectives use terminal cos(3x), the density-energy trapezoid and the smooth maximum on every solver integration node. The nonlinear accuracy bank uses the fixed 17-node common polygon. Derivative comparisons are literal NN/native comparisons within one step size; their costs are not attached to that distinct 17-node nonlinear accuracy task or used to claim a temporal derivative convergence rate.",
        "Optional T=1 timings concern a different, finite-Euler experiment and its original depth-four checkpoint. They do not extend the short-horizon fine-reference accuracy conclusions to T=1.", "",
        "\n".join(long_table) if len(long_table) > 2 else "No optional T=1 timing was included in this report.", "",
        "## Matched statistical queries and amortization", "",
        "![](accuracy_cost_matched_moments.png)", "",
        "This comparison uses literal same-time four-mean/4×4-covariance queries. Conditional heads use one supplied histogram. Path moment queries actually simulate all 16,384 realizations in chunks of 64, apply the exact bank and aggregate means/N-covariances; they are not extrapolated B≤256 throughput. Fresh initial-histogram and Brownian generation are separate components of the path costs. Ensemble statistics-head costs include a fresh 16,384-histogram multinomial pool, inference and total-covariance aggregation; Gaussian ensemble initialization uses analytic IID covariance.",
        "![](deployed_sampling_cost.png)",
        "A tight paired path/reference interval measures a method-discrepancy estimate under shared physical noise. It is not the standalone variability of a deployed 16,384-path Monte Carlo estimate. The latter is recomputed from complete candidate-family bootstrap estimates and reported in deployed_sampling.csv and the separate plot. Conditional heads have zero inference Monte Carlo noise; initial-ensemble heads still have initial-law integration variability. Adding bias, paired sampling, standalone sampling and reference-audit quantities does not automatically give a rigorous independent coverage statement.",
        "A newly declared bank can reuse stored full-path samples without retraining the path model, but a fresh-path query still pays simulation and Monte Carlo uncertainty. A new statistical head pays fitting and optional historical trunk costs before cheap repeated conditional queries. A numerical break-even count is meaningful only for the same supported query after matched finite-reference error and uncertainty gates; unresolved comparisons are left without a winner or crossover count.", "",
        "Classical full-path samplers also support newly declared payoffs without refitting. Reusing realizations establishes the flexibility of the path representation. A benefit attributable to learning requires a measured accuracy–cost improvement over the compiled native and exact-kernel controls.", "",
        "## Training, stopping and evidence", ""]
    workers = worker_rows(evidence)
    wall = [worker.get("wall_seconds", 0.) for worker in workers]
    cpu = [worker.get("cpu_seconds", 0.) for worker in workers]
    lines.extend([f"The twelve new fits used three independent optimizer workers. Recorded worker wall times are {_range(wall, scale=1, digits=5)} s; aggregate worker CPU time is {sum(cpu):.5g} s. Summing worker wall times ({sum(wall):.5g} s) is not the elapsed concurrent batch duration. Per-model fitting times, historical pretraining costs and checkpoint hashes remain in training.csv and the locked metadata.",
        "Stopping probabilities use exact binomial intervals; zero observed exits is not a zero-probability claim. Gaussian negative-density and below-boundary sample fractions concern its 17 stored common-grid nodes, with binomial sampling intervals in gaussian_law_diagnostics.csv. They are not between-node or continuous Gaussian hitting probabilities and do not establish stopped-DK equivalence. No trajectory or Gaussian sample is clipped or discarded to improve a comparison.",
        "All three optimizer seeds remain separate in the CSVs and plots. No seed, step, initialization or sample count was chosen after reading final outcomes. The 5% normalized error and 1.25% component uncertainty levels are research targets, not blanket certification.", "",
        "CSV evidence: moments.csv, many480_summary.csv, many480_components.csv.gz, strong_paths.csv, stopping.csv, scalar_distributions.csv, initial_integration.csv, deployed_sampling.csv, reference_refinement.csv, costs.csv and training.csv. The report manifest records exact input and output hashes; original arrays/checkpoints remain referenced by their immutable experiment manifests.", ""])
    return "\n\n".join(lines)


def training_rows(evidence):
    rows = []
    for item in evidence["fit_lock"]["models"]:
        metadata = read(item["metadata_path"])
        rows.append({"model": "new-payoff-head", "bank": item["bank"], "optimizer_seed": item["seed"],
            "initialization": item["init"], "training_seconds": metadata["training_seconds"],
            "checkpoint_sha256": item["checkpoint_sha256"], "source_hash": item["source_hash"]})
    seen = set()
    for packet in evidence["records"]:
        method = packet["method"]
        if method.get("method") != "prepared-learned-path" or method["checkpoint_sha256"] in seen:
            continue
        seen.add(method["checkpoint_sha256"])
        rows.append({"model": "historical-path", "case": method["case"], "optimizer_seed": method["seed"],
            "dt": method["dt"], "training_seconds": packet["cost"]["historical_training_seconds"],
            "checkpoint_sha256": method["checkpoint_sha256"], "historical_commit": method["historical_commit"]})
    for item in evidence["training_design"].get("historical_transfers", []):
        rows.append({"model": "historical-statistics-trunk", "optimizer_seed": item["optimizer_seed"],
            "training_seconds": item["original_training_seconds"], "checkpoint_sha256": item["checkpoint_sha256"]})
    for role, item in evidence["fit_lock"]["input_banks"].items():
        metadata = read(item["metadata_path"])
        rows.append({"model": "shared-new-current-state-data", "role": role,
                     "generation_seconds": metadata.get("generation_seconds"),
                     "independent_families": item.get("independent_families"),
                     "collocation_rows": item.get("collocation_rows"),
                     "shared_all_twelve_models": True, "metadata_sha256": item["metadata_sha256"]})
    return rows


def worker_rows(evidence):
    return [{"optimizer_seed": worker["seed"], "attempts": len(worker.get("attempts", [worker])),
        "wall_seconds": sum(attempt.get("wall_seconds", 0.) for attempt in worker.get("attempts", [worker])),
        "cpu_seconds": sum(attempt.get("cpu_seconds", 0.) for attempt in worker.get("attempts", [worker])),
        "complete": worker["complete"], "scope": "sum of this worker's measured attempts; concurrent workers are not elapsed-wall additive"}
        for worker in evidence["workers"]]


def generate(root, path_timing, *, payoff_timing=None, output=None, docs_output=None,
             verify_artifacts=True, plots=True, gaussian_audit=None):
    evidence = load_complete(root, path_timing, payoff_timing, verify_artifacts=verify_artifacts,
                             gaussian_audit=gaussian_audit)
    output = Path(output or Path(root) / "report").resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = flatten_accuracy(evidence)
    tables["deployed_sampling"] = deployed_sampling(evidence)
    references, costs = flatten_reference(evidence), flatten_costs(evidence)
    tables["task_summary"] = task_summary_rows(tables, costs)
    for name, rows in tables.items():
        write_csv(output / (name + (".csv.gz" if name == "many480_components" else ".csv")), rows)
    write_csv(output / "reference_refinement.csv", references)
    write_csv(output / "costs.csv", costs)
    write_csv(output / "preparation_costs.csv", preparation_rows(evidence))
    write_csv(output / "training.csv", training_rows(evidence))
    write_csv(output / "worker_costs.csv", worker_rows(evidence))
    write_csv(output / "amortization.csv", amortization_rows(evidence, tables, costs, references))
    if plots:
        make_plots(tables, costs, output)
    text = render_markdown(evidence, tables, costs, references)
    (output / "report.md").write_text(text)
    if docs_output is not None:
        target = Path(docs_output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        for figure in output.glob("*.png"):
            text = text.replace(f"]({figure.name})", f"]({os.path.relpath(figure, target.parent)})")
        target.write_text(text)
    manifest = {"kind": "broad-DK-task-comparison-report-v1", "complete": True,
        "source_hash": evidence["index"]["source_hash"], "generator_sha256": sha(__file__),
        "inputs": evidence["input_files"], "verified_artifact_count": evidence["verified_artifact_count"],
        "evaluation_groups": evidence["index"]["declared_groups"],
        "evaluation_methods": len(evidence["records"]), "optimizer_seeds_pooled": False,
        "continuous_DK_claim": False, "outputs": {path.name: sha(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--path-timing", type=Path, required=True)
    parser.add_argument("--payoff-timing", type=Path)
    parser.add_argument("--gaussian-audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--docs-output", type=Path)
    parser.add_argument("--skip-artifact-hashes", action="store_true")
    args = parser.parse_args()
    generate(args.root, args.path_timing, payoff_timing=args.payoff_timing, output=args.output,
             docs_output=args.docs_output, verify_artifacts=not args.skip_artifact_hashes,
             gaussian_audit=args.gaussian_audit)


if __name__ == "__main__":
    main()
