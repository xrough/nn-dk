"""Report the complete independently tested payoff-formulation repair study.

This postprocessor fits nothing, samples no physical trajectory and selects no
optimizer seed. It distinguishes paired method-error uncertainty, deployed
Monte Carlo variability, reference refinement and training amortization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/nn-dk-mpl-cache")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from analysis.evaluate_path_generalization import bootstrap_counts, empirical  # noqa: E402
from analysis.run_payoff_repair import reference_configuration, verify_fit_lock  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

KIND = "independent-payoff-repair-report-v1"
BANKS = ("terminal-modes34", "terminal3-integral1")
FAMILIES = ("old-cold", "old-trunk-transfer", "v1-none", "v1-balanced",
            "untrained-repaired", "repaired", "LGA", "native-Euler", "Picard", "learned-path")
COLORS = dict(zip(FAMILIES, ("#a6a6a6", "#777777", "#e69f00", "#b8860b", "#56b4e9",
                            "#0072b2", "#009e73", "#202020", "#cc79a7", "#d55e00"), strict=True))
LABELS = {"old-cold": "Previous raw head", "old-trunk-transfer": "Previous transferred head",
          "v1-none": "Fresh raw head", "v1-balanced": "Fresh balanced head",
          "untrained-repaired": "Untrained heat anchor", "repaired": "Repaired staged head",
          "LGA": "Linear-Gaussian", "native-Euler": "Compiled native MC",
          "Picard": "Exact-kernel Picard MC", "learned-path": "Learned path MC"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _artifact(path, expected, inputs):
    path = Path(path).resolve()
    key = str(path)
    actual = inputs.get(key)
    if actual is None:
        actual = sha(path)
        inputs[key] = actual
    if actual != expected:
        raise ValueError(f"Scientific evidence checksum mismatch: {path}")


def load_complete(root):
    """Refuse partial studies and verify every declared result/array/input."""
    root = Path(root).resolve()
    inputs = {}
    plan_path, index_path = (root / "comparison" / name for name in
                             ("preregistration.json", "accuracy_index.json"))
    plan, index = read(plan_path), read(index_path)
    cfg = reference_configuration()
    if (not index.get("complete") or set(index.get("groups", [])) != set(cfg.group_keys)
            or index.get("declared_method_records_per_group") != 33
            or plan.get("source_hash") != source_hash()):
        raise ValueError("A complete matching six-group, 198-result study is required")
    for path in (plan_path, index_path):
        inputs[str(path)] = sha(path)
    pretest_path = root / "audit/pretest_audit.json"
    pretest = read(pretest_path)
    if not pretest.get("complete"):
        raise ValueError("The independent pretest/RNG audit must be complete")
    inputs[str(pretest_path)] = sha(pretest_path)
    for path, expected in plan["helpers"].items():
        _artifact(path, expected, inputs)
    lock = verify_fit_lock(root)
    postfit_path = root / "audit/postfit_validation_diagnostic.json"
    postfit = read(postfit_path)
    verify_postfit_diagnostic(postfit, lock)
    inputs[str(postfit_path)] = sha(postfit_path)
    lock_path = root / "payoffs/fit_lock.json"
    _artifact(lock_path, index["identity"]["fit_lock_sha256"], inputs)
    for field in ("preregistration", "data_lock"):
        _artifact(lock[f"{field}_path"], lock[f"{field}_sha256"], inputs)
    reference = read(root / "reference/manifest.json")
    if (reference.get("status") != "complete"
            or index["identity"]["reference_sha256"] != reference.get("reference_sha256")
            or reference["identity"]["config_hash"] != cfg.config_hash):
        raise ValueError("The fresh reference does not match the complete accuracy index")
    _artifact(root / "reference/reference.h5", reference["reference_sha256"], inputs)
    inputs[str(root / "reference/manifest.json")] = sha(root / "reference/manifest.json")
    independence_path = root / "audit/reference_independence_audit.json"
    independence = read(independence_path)
    if (not independence.get("complete") or independence.get("source_hash") != source_hash()
            or independence.get("reference_sha256") != reference["reference_sha256"]
            or independence.get("reference_manifest_sha256") != sha(root / "reference/manifest.json")
            or independence.get("fit_lock_sha256") != sha(lock_path)
            or independence.get("unique_reference_families") != 98_304
            or independence.get("reference_family_overlap_with_current_or_historical") != 0):
        raise ValueError("A complete source/reference-matching independence audit is required")
    inputs[str(independence_path)] = sha(independence_path)
    for packet in lock["input_banks"].values():
        _artifact(packet["path"], packet["sha256"], inputs)
        _artifact(packet["metadata_path"], packet["metadata_sha256"], inputs)
    for item in lock["models"]:
        for kind in ("checkpoint", "metadata"):
            _artifact(item[f"{kind}_path"], item[f"{kind}_sha256"], inputs)
    _artifact(plan["old_fit_lock"]["path"], plan["old_fit_lock"]["sha256"], inputs)
    old_lock = read(plan["old_fit_lock"]["path"])
    _artifact(old_lock["preregistration_path"], old_lock["preregistration_sha256"], inputs)
    old_design = read(old_lock["preregistration_path"])
    for transfer in old_design["historical_transfers"]:
        for kind in ("checkpoint", "metadata"):
            _artifact(transfer[f"{kind}_path"], transfer[f"{kind}_sha256"], inputs)
    for item in plan["old_heads"] + plan["old_paths"]:
        _artifact(item["checkpoint_path"], item["checkpoint_sha256"], inputs)
        path = Path(item["checkpoint_path"]).with_suffix(".json")
        _artifact(path, item["metadata_sha256"], inputs)
    groups, records = [], []
    for key in cfg.group_keys:
        path = root / "comparison/accuracy" / key / "group.json"
        group = read(path)
        if (not group.get("complete") or group["identity"] != index["identity"]
                or group["group"] != key or len(group["records"]) != 33):
            raise ValueError("Every independent group must contain all 33 declared method records")
        inputs[str(path)] = sha(path)
        names = set()
        for item in group["records"]:
            _artifact(item["path"], item["sha256"], inputs)
            record = read(item["path"])
            identity = record["identity"]
            if (not record.get("complete") or identity["group"] != key
                    or any(identity[k] != value for k, value in index["identity"].items())
                    or identity["name"] in names):
                raise ValueError("Incomplete, duplicated or incompatible result record")
            names.add(identity["name"])
            _artifact(record["arrays"]["path"], record["arrays"]["sha256"], inputs)
            records.append(record)
        expected_names = {item["name"] for item in lock["models"]}
        expected_names.update(f"old_{item['bank']}_{item['init']}_seed{item['seed']}"
                              for item in plan["old_heads"])
        expected_names.update("untrained_" + bank for bank in BANKS)
        expected_names.update(f"LGA_dt{step}_{bank}" for step in (7, 10) for bank in BANKS)
        expected_names.update(item["name"] for item in plan["old_paths"]
                              if item["case"] == key.split("__")[0])
        expected_names.update(("native-Euler", "Picard"))
        if names != expected_names:
            raise ValueError("An independent group omitted or replaced a preregistered model")
        groups.append(group)
    replay_path = root / "audit/selected_accuracy_replay.json"
    replay = read(replay_path)
    if (not replay.get("complete") or replay.get("source_hash") != source_hash()
            or replay.get("refitting_performed") is not False
            or replay.get("optimizer_seed_selection_using_test") is not False
            or len(replay.get("checks", [])) < 5):
        raise ValueError("The independent selected-accuracy replay must be complete and source matching")
    for check in replay["checks"]:
        result_path = root / "comparison/accuracy" / check["group"] / (check["method"] + ".json")
        if (not check.get("strict_restoration") or check["result_sha256"] != inputs.get(str(result_path))
                or check["metric_maximum_error"] > 1e-11 or check["prediction_maximum_error"] > 1e-11):
            raise ValueError("Selected accuracy replay differs from this reference-matched result set")
    inputs[str(replay_path)] = sha(replay_path)
    timing_path = root / "comparison/timing.json"
    timing = read(timing_path)
    if (not timing.get("complete") or not timing.get("final_timing_after_all_workers_finished")
            or timing["identity"]["source_hash"] != source_hash()
            or timing["identity"]["fit_lock_sha256"] != sha(lock_path)
            or timing["identity"]["plan_sha256"] != sha(plan_path)
            or len(timing["records"]) != 232
            or len({row["slot"] for row in timing["records"]}) != 232):
        raise ValueError("All 232 uncontended literal timing rows must be complete")
    inputs[str(timing_path)] = sha(timing_path)
    for item in timing.get("compiled_control", []):
        _artifact(item["library_path"], item["library_sha256"], inputs)
    final_audit_path = root / "audit/final_comparison_audit.json"
    final_audit = read(final_audit_path)
    if (not final_audit.get("complete") or final_audit.get("source_hash") != source_hash()
            or final_audit.get("reference_and_fit_identity") != index["identity"]
            or final_audit.get("timing_sha256") != sha(timing_path)
            or final_audit.get("accuracy_records_checked") != 198
            or final_audit.get("timing_rows_checked") != 232):
        raise ValueError("The independent final comparison audit must match all accuracy/cost identities")
    inputs[str(final_audit_path)] = sha(final_audit_path)
    return {"root": root, "cfg": cfg, "plan": plan, "index": index, "lock": lock,
            "reference": reference, "groups": groups, "records": records, "timing": timing,
            "old_design": old_design, "postfit": postfit, "inputs": inputs}


def verify_postfit_diagnostic(diagnostic, fit_lock):
    expected = {(item["bank"], item["seed"]): item for item in fit_lock["models"]
                if item["variant"] == "repaired"}
    records = diagnostic.get("records", [])
    actual = {(item["bank"], item["seed"]): item for item in records}
    if (not diagnostic.get("complete") or diagnostic.get("pending")
            or diagnostic.get("source_hash") != source_hash() or len(actual) != 6
            or len(records) != 6 or set(actual) != set(expected)):
        raise ValueError("All six complete postfit validation/mean-freeze diagnostics are required")
    for key, record in actual.items():
        if (not record.get("strict_restore") or not record.get("selected_mean_unchanged")
                or record["checkpoint_sha256"] != expected[key]["checkpoint_sha256"]
                or len(set(record["selected_mean_hashes"].values())) != 1):
            raise ValueError("The postfit audit differs from a locked repaired fit or changed its frozen mean")
        if "covariance" not in record["validation"].get("selected_mean_covariance_initial", {}):
            raise ValueError("Covariance phase-start diagnostics must use the selected frozen mean")
    expected_controls = {(item["bank"], item["variant"], item["seed"]): item
                         for item in fit_lock["models"] if item["variant"] != "repaired"}
    controls = diagnostic.get("controls", [])
    actual_controls = {(r["bank"], r["variant"], r["seed"]): r for r in controls}
    if (diagnostic.get("pending_controls") or len(controls) != 4
            or set(actual_controls) != set(expected_controls)):
        raise ValueError("All four current v1 diagnostic controls are required")
    for key, record in actual_controls.items():
        if (not record.get("strict_restore")
                or record["checkpoint_sha256"] != expected_controls[key]["checkpoint_sha256"]):
            raise ValueError("A v1 validation diagnostic differs from its locked fit")


def _stage_initial_validation(metadata, diagnostic, stage):
    """Gamma(u) changes with mean fitting, so covariance uses its own phase start."""
    if stage == "covariance":
        value = diagnostic["validation"]["selected_mean_covariance_initial"]["covariance"]
        return {"global_step": 1030, "validation_objective": value["normalized_objective"],
                "validation_raw_rms": value["raw_global_rms"],
                "scope": "covariance phase start against selected frozen mean Gamma"}
    value = metadata["initial_validation"]["mean"]
    return {"global_step": 0, "validation_objective": value["objective"],
            "validation_raw_rms": value["raw_rms"], "scope": "initial model before mean fitting"}


def _identity(record, bank):
    method = record["method"]
    case, role, anchor = record["identity"]["group"].split("__")
    return {"group": record["identity"]["group"], "case": case, "role": role,
            "anchor": int(anchor), "name": record["identity"]["name"],
            "family": method["family"], "bank": bank, "seed": method.get("seed"),
            "dt": method.get("dt")}


def _bank_metrics(record):
    if "bank" in record["identity"]:
        return {record["identity"]["bank"]: record["metrics"]}
    return record["metrics"]


def accuracy_tables(evidence):
    accuracy, components, covariance, refinements = [], [], [], []
    times = evidence["cfg"].query_times
    bank_labels = {bank["name"]: bank["labels"] for bank in evidence["plan"]["banks"]}
    for record in evidence["records"]:
        for bank, metric in _bank_metrics(record).items():
            identity = _identity(record, bank)
            for row in metric["records"]:
                t = row["time_index"]
                base = {**identity, "time_index": t, "time": times[t], "paired": metric["paired"]}
                accuracy.append({**base, **{key: row[key] for key in
                    ("max_mean_error_in_reference_std", "max_normalized_covariance_error",
                     "mean_sampling95_simultaneous_radius", "covariance_sampling95_simultaneous_radius",
                     "mean_error_plus_sampling95", "covariance_error_plus_sampling95")}})
                reference = np.asarray(row["reference_scaled_covariance"])
                candidate = np.asarray(row["candidate_scaled_covariance"])
                for i, label in enumerate(bank_labels[bank]):
                    components.append({**base, "observable": label,
                        "reference_mean": row["reference_mean"][i],
                        "candidate_mean": row["candidate_mean"][i],
                        "normalized_mean_difference": row["normalized_mean_difference"][i],
                        "reference_scaled_variance": reference[i, i],
                        "candidate_scaled_variance": candidate[i, i],
                        "variance_ratio": candidate[i, i] / reference[i, i]})
                    for j, other in enumerate(bank_labels[bank]):
                        covariance.append({**base, "observable_i": label, "observable_j": other,
                            "reference_scaled_covariance": reference[i, j],
                            "candidate_scaled_covariance": candidate[i, j],
                            "normalized_covariance_difference": row["normalized_covariance_difference"][i][j]})
    for group in evidence["groups"]:
        case, role, anchor = group["group"].split("__")
        for bank, metric in group["reference_refinement"].items():
            for row in metric["records"]:
                refinements.append({"group": group["group"], "case": case, "role": role,
                    "anchor": int(anchor), "bank": bank, "time_index": row["time_index"],
                    "time": times[row["time_index"]], **{key: row[key] for key in
                        ("max_mean_error_in_reference_std", "max_normalized_covariance_error",
                         "mean_sampling95_simultaneous_radius", "covariance_sampling95_simultaneous_radius",
                         "mean_error_plus_sampling95", "covariance_error_plus_sampling95")}})
    return {"accuracy": accuracy, "components": components,
            "covariances": covariance, "reference_refinement": refinements}


def sampling_tables(evidence):
    """Standalone estimator variability; paired method-difference CIs are separate."""
    cfg, rows, reference_rows = evidence["cfg"], [], []
    n_particles = cfg.n_particles
    by_group = defaultdict(list)
    for record in evidence["records"]:
        by_group[record["identity"]["group"]].append(record)
    with h5py.File(evidence["root"] / "reference/reference.h5", "r") as store:
        for group, records in by_group.items():
            counts = bootstrap_counts(cfg.paths_per_group, int(stable_hash(
                [evidence["plan"]["seed"], group])[:12], 16), 128)
            normalizers = {}
            for bank, field in zip(BANKS, ("bank_b", "bank_c"), strict=True):
                values = store[group][field][:]
                rm, rc = empirical(values)
                brm, brc = empirical(values, counts)
                std = np.sqrt(np.diagonal(rc, axis1=-2, axis2=-1))
                products = std[..., :, None] * std[..., None, :]
                normalizers[bank] = (std, products)
                mean_radius, cov_radius = _sampling_radii(rm, n_particles * rc,
                    brm, n_particles * brc, std, products, n_particles)
                for t, time_value in enumerate(cfg.query_times):
                    reference_rows.append({"group": group, "bank": bank, "time": time_value,
                        "time_index": t, "mean95_radius": mean_radius[t],
                        "covariance95_radius": cov_radius[t],
                        "scope": "standalone finite-reference Monte Carlo estimator"})
            for record in records:
                with np.load(record["arrays"]["path"]) as arrays:
                    for bank in _bank_metrics(record):
                        identity = _identity(record, bank)
                        std, products = normalizers[bank]
                        if "outcomes_" + bank in arrays:
                            pm, pc = empirical(arrays["outcomes_" + bank])
                            bm, bc = empirical(arrays["outcomes_" + bank], counts)
                            mean_radius, cov_radius = _sampling_radii(pm, n_particles * pc,
                                bm, n_particles * bc, std, products, n_particles)
                            scope = "standalone trajectory-Monte-Carlo estimator; not paired bias uncertainty"
                        elif "bootstrap_mean" in arrays:
                            mean_radius, cov_radius = _sampling_radii(arrays["mean"],
                                arrays["scaled_covariance"], arrays["bootstrap_mean"],
                                arrays["bootstrap_scaled_covariance"], std, products, n_particles)
                            scope = "standalone initial-law integration Monte Carlo"
                        else:
                            mean_radius = cov_radius = np.zeros(len(cfg.query_times))
                            scope = "deterministic conditional or analytic-IID marginal prediction; no inference MC"
                        for t, time_value in enumerate(cfg.query_times):
                            rows.append({**identity, "time_index": t, "time": time_value,
                                "mean95_radius": float(mean_radius[t]),
                                "covariance95_radius": float(cov_radius[t]), "scope": scope})
    return rows, reference_rows


def _sampling_radii(mean, covariance, bootstrap_mean, bootstrap_covariance, std, products, n_particles):
    dm = np.abs(np.asarray(bootstrap_mean) - mean) / std
    dc = np.abs(np.asarray(bootstrap_covariance) - covariance) / (n_particles * products)
    return (np.quantile(dm.max(-1), .95, axis=0),
            np.quantile(dc.max(axis=(-2, -1)), .95, axis=0))


def training_tables(evidence):
    rows, history, workers, metadata = [], [], [], {}
    transfers = {item["optimizer_seed"]: item["original_training_seconds"]
                 for item in evidence["old_design"]["historical_transfers"]}
    entries = [(item, item["variant"], False) for item in evidence["lock"]["models"]]
    entries += [({**item, "name": f"old_{item['bank']}_{item['init']}_seed{item['seed']}"},
                 "old-" + item["init"], True) for item in evidence["plan"]["old_heads"]]
    entries += [(item, "learned-path", True) for item in evidence["plan"]["old_paths"]]
    for item, family, historical in entries:
        path = Path(item.get("metadata_path", Path(item["checkpoint_path"]).with_suffix(".json")))
        packet = read(path)
        name = item["name"]
        # Path names repeat across profiles; keep a case-specific offline slot.
        key = (item.get("case", "both"), name)
        metadata[key] = packet
        row = {"name": name, "family": family, "bank": item.get("bank", "both"),
               "case": item.get("case", "both"), "seed": item["seed"],
               "historical": historical, "training_seconds": packet["training_seconds"],
               "prior_trunk_training_seconds": transfers[item["seed"]]
                   if family == "old-trunk-transfer" else 0.,
               "parameter_count": packet.get("parameter_count"),
               "adam_steps": packet.get("adam_steps"), "lbfgs_steps": packet.get("lbfgs_steps"),
               "metadata_path": str(path), "metadata_sha256": sha(path),
               "complete": packet["complete"],
               "cost_scope": "known historical fitting components; original data costs incomplete"
                   if historical else "fresh complete fitting, including validation and checkpointing"}
        if family == "repaired":
            row.update(mean_stage_seconds=packet["stage_seconds"]["mean"],
                       covariance_stage_seconds=packet["stage_seconds"]["covariance"],
                       selected_mean_step=packet["best_step"]["mean"],
                       selected_covariance_step=packet["best_step"]["covariance"])
        rows.append(row)
        if historical:
            continue
        initial = packet["initial_validation"]
        if family == "repaired":
            diagnostic = next(r for r in evidence["postfit"]["records"]
                              if r["bank"] == item["bank"] and r["seed"] == item["seed"])
            for stage in ("mean", "covariance"):
                history.append({"name": name, "bank": item["bank"], "family": family,
                    "seed": item["seed"], "stage": stage, "step": 0,
                    **_stage_initial_validation(packet, diagnostic, stage),
                    "selected": packet["best_step"][stage] == 0})
        else:
            for stage in ("mean", "covariance"):
                history.append({"name": name, "bank": item["bank"], "family": family,
                    "seed": item["seed"], "stage": stage, "step": 0, "global_step": 0,
                    "validation_objective": initial["objective"],
                    "validation_raw_rms": initial[f"{stage}_raw_rms"], "selected": False})
        for point in packet["history"]:
            validation = point.get("validation")
            if validation is None:
                continue
            for stage in ((point["stage"],) if family == "repaired" else ("mean", "covariance")):
                step = point["step"]
                history.append({"name": name, "bank": item["bank"], "family": family,
                    "seed": item["seed"], "stage": stage, "step": step,
                    "global_step": step + (1030 if family == "repaired" and stage == "covariance" else 0),
                    "validation_objective": validation["objective"],
                    "validation_raw_rms": validation["raw_rms"] if family == "repaired"
                        else validation[f"{stage}_raw_rms"],
                    "selected": step == (packet["best_step"][stage] if family == "repaired"
                                          else packet["best_step"])})
    for seed in (0, 1, 2):
        path = evidence["root"] / f"payoffs/execution/worker_seed{seed}.json"
        packet = read(path)
        evidence["inputs"][str(path)] = sha(path)
        for index, attempt in enumerate(packet.get("attempts", [])):
            workers.append({"seed": seed, "attempt": index, **attempt})
    data = []
    for role in ("train", "validation"):
        packet = read(evidence["lock"]["input_banks"][role]["metadata_path"])
        data.append({"role": role, "generation_seconds": packet["generation_seconds"],
                     "independent_families": evidence["lock"]["input_banks"][role]["independent_families"],
                     "collocation_rows": evidence["lock"]["input_banks"][role]["collocation_rows"]})
    return rows, history, workers, data


def validation_diagnostic_tables(evidence):
    rows = []
    for record in evidence["postfit"]["records"] + evidence["postfit"]["controls"]:
        for point, diagnostics in record["validation"].items():
            for stage, values in diagnostics.items():
                rows.append({"bank": record["bank"], "seed": record["seed"], "stage": stage,
                    "variant": record.get("variant", "repaired"),
                    "point": point, "raw_rms": values["raw_global_rms"],
                    "normalized_rms": values["normalized_global_rms"],
                    "normalized_objective": values["normalized_objective"],
                    "selected_mean_unchanged": record.get("selected_mean_unchanged"),
                    "comparison_scope": "obsolete pre-mean Gamma; not the covariance phase baseline"
                        if point == "initial" and stage == "covariance"
                        and record.get("selected_mean_unchanged") else "validation-only diagnostic"})
    return rows


def collapse_diagnostic_tables(evidence):
    labels = {b["name"]: b["labels"] for b in evidence["plan"]["banks"]}
    rows = []
    for record in evidence["postfit"]["records"] + evidence["postfit"]["controls"]:
        diag = record["collapse_diagnostic"]
        for i, label in enumerate(labels[record["bank"]]):
            rows.append({"bank": record["bank"], "variant": record.get("variant", "repaired"),
                "seed": record["seed"], "observable": label,
                "mean_QV_source_over_heat_anchor": diag["source_final_over_heat_baseline_by_output"][i],
                "covariance_over_heat_anchor": diag["covariance_final_over_heat_baseline_by_output"][i],
                "minimum_covariance_eigenvalue": diag["minimum_covariance_eigenvalue"],
                "scope": "validation-only degeneration diagnostic; heat anchor is not interacting truth"})
    return rows


def validation_component_tables(evidence):
    """Preserve each output/pair in physical as well as internal residual units."""
    labels = {b["name"]: b["labels"] for b in evidence["plan"]["banks"]}
    rows = []
    for record in evidence["postfit"]["records"] + evidence["postfit"]["controls"]:
        for point, diagnostics in record["validation"].items():
            for stage, values in diagnostics.items():
                arrays = {key: np.asarray(values[key + "_by_output_or_pair"]) for key in
                          ("raw_internal_rms", "raw_physical_rms", "physical_horizon_normalized_rms")}
                for index in np.ndindex(arrays["raw_internal_rms"].shape):
                    rows.append({"bank": record["bank"], "variant": record.get("variant", "repaired"),
                        "seed": record["seed"], "stage": stage, "point": point,
                        "observable_i": labels[record["bank"]][index[0]],
                        "observable_j": labels[record["bank"]][index[1]] if len(index) == 2 else None,
                        **{key: float(value[index]) for key, value in arrays.items()},
                        "scope": "pre-mean Gamma; obsolete covariance-stage target"
                            if point == "initial" and stage == "covariance"
                            and record.get("selected_mean_unchanged") else "validation-only residual diagnostic"})
    return rows


def cost_tables(evidence, training, data):
    rows, totals = [], []
    shared_data = sum(row["generation_seconds"] for row in data)
    fits = {(row["case"], row["name"]): row for row in training}
    loading = {(item["case"], item["name"]): item["seconds"] for item in evidence["timing"]["loading"]}
    prep = {(item["case"], item["name"]): item["restoration_and_kernel_preparation_seconds"]
            for item in evidence["timing"].get("path_preparation", [])}
    compilation = {item["case"]: item["compile_seconds"]
                   for item in evidence["timing"].get("compiled_control", [])}
    compiled_setup = {item["case"]: max(0., item["setup_seconds_this_process"]
        - item.get("compile_seconds_this_process", 0.))
        for item in evidence["timing"].get("compiled_control", [])}
    first_b1 = {(item["case"], item["name"]): item["first_prediction_seconds"]
                for item in evidence["timing"]["records"]
                if item["role"] == "conditional" and item["batch"] == 1}
    for item in evidence["timing"]["records"]:
        fit = fits.get((item["case"], item["name"]), fits.get(("both", item["name"])))
        fitting = 0. if fit is None else fit["training_seconds"]
        prior = 0. if fit is None else fit["prior_trunk_training_seconds"]
        data_seconds = shared_data if fit is not None and not fit["historical"] else 0.
        restore = loading.get((item["case"], item["name"]), 0.)
        preparation = prep.get((item["case"], item["name"]), 0.)
        compile_seconds = compilation.get(item["case"], 0.) if item["family"] == "native-Euler" else 0.
        setup_seconds = compiled_setup.get(item["case"], 0.) if item["family"] == "native-Euler" else 0.
        probe = (first_b1[(item["case"], item["name"])] if item["family"] in
                 ("repaired", "untrained-repaired")
                 and (item["role"] != "conditional" or item["batch"] != 1) else 0.)
        row = {key: value for key, value in item.items() if key != "samples_seconds"}
        row.update(training_seconds=fitting, prior_trunk_training_seconds=prior,
                   shared_data_generation_seconds=data_seconds, restoration_seconds=restore,
                   runtime_preparation_seconds=preparation, cold_compilation_seconds=compile_seconds,
                   compiled_cache_setup_seconds=setup_seconds,
                   cache_creation_B1_probe_seconds=probe,
                   first_prediction_scope="first call after model restore" if item["role"] == "conditional"
                       and item["batch"] == 1 else "first call at this allocation; model may reuse earlier B1 horizon cache",
                   known_offline_seconds=fitting + prior + data_seconds,
                   prior_data_generation_unrecorded=fit is not None and fit["historical"],
                   warm_input_generation_included=item["role"] == "ensemble" or item["family"] in
                       ("learned-path", "native-Euler", "Picard"))
        rows.append(row)
        for q in (1, 100, 10_000, 100_000):
            totals.append({"slot": item["slot"], "case": item["case"], "name": item["name"],
                "family": item["family"], "bank": item["bank"], "role": item["role"],
                "batch": item["batch"], "query_calls": q,
                "fresh_fitting_seconds": fitting, "prior_trunk_training_seconds": prior,
                "shared_data_generation_seconds": data_seconds, "restoration_seconds": restore,
                "runtime_preparation_seconds": preparation, "cold_compilation_seconds": compile_seconds,
                "compiled_cache_setup_seconds": setup_seconds,
                "cache_creation_B1_probe_seconds": probe,
                "first_prediction_seconds": item["first_prediction_seconds"],
                "subsequent_warm_calls_seconds": (q - 1) * item["median_seconds"],
                "total_known_seconds": fitting + prior + data_seconds + restore + preparation
                    + compile_seconds + setup_seconds + probe + item["first_prediction_seconds"]
                    + (q - 1) * item["median_seconds"],
                "scope": "known components; historical data/startup/search costs not invented; calls retain actual batch"})
    return rows, totals


def study_costs(evidence, training, workers, data):
    """Study allocation totals, never charged to every deployment call."""
    starts = [r["started_unix"] for r in workers]
    finishes = [r["finished_unix"] for r in workers]
    transfers = evidence["old_design"]["historical_transfers"]
    components = {
        "fresh_ten_fit_recorded_model_seconds": sum(r["training_seconds"] for r in training if not r["historical"]),
        "fresh_shared_train_validation_generation_seconds": sum(r["generation_seconds"] for r in data),
        "fresh_worker_CPU_seconds_all_attempts": sum(r["cpu_seconds"] for r in workers),
        "fresh_worker_active_wall_sum_not_batch_elapsed": sum(r["wall_seconds"] for r in workers),
        "fresh_worker_earliest_start_to_latest_finish_wall_span": max(finishes) - min(starts),
        "historical_head_fitting_seconds_inventory": sum(r["training_seconds"] for r in training
                                                         if r["family"].startswith("old-")),
        "historical_unique_trunk_fitting_seconds": sum(r["original_training_seconds"] for r in transfers),
        "historical_fine_path_fitting_seconds_inventory": sum(r["training_seconds"] for r in training
                                                              if r["family"] == "learned-path"),
        "fresh_reference_generation_seconds": evidence["reference"]["generation_seconds"],
    }
    return [{"component": key, "seconds": value,
             "scope": "study allocation/accounting; not all costs are additive or a deployment charge"}
            for key, value in components.items()]


def accuracy_gate(rows, sampling, refinements, *, tolerance=.05, uncertainty=.0125):
    """A conservative finite-reference diagnostic gate; never a continuum claim."""
    if not rows:
        return "unsupported"
    if max(max(r["max_mean_error_in_reference_std"], r["max_normalized_covariance_error"])
           for r in rows) > tolerance:
        return "point-error-above-target"
    if (max(max(r["mean_error_plus_sampling95"], r["covariance_error_plus_sampling95"])
            for r in rows) > tolerance
            or max(max(r["mean_sampling95_simultaneous_radius"],
                       r["covariance_sampling95_simultaneous_radius"]) for r in rows) > uncertainty):
        return "method-comparison-sampling-unresolved"
    if not sampling or max(max(r["mean95_radius"], r["covariance95_radius"]) for r in sampling) > uncertainty:
        return "standalone-MC-unresolved"
    if (not refinements or max(max(r["mean_error_plus_sampling95"],
                                  r["covariance_error_plus_sampling95"]) for r in refinements) > uncertainty):
        return "coupled-reference-refinement-unresolved"
    return "finite-reference-diagnostic-only"


def amortization_tables(accuracy, sampling, refinements, costs):
    rows = []
    for candidate in costs:
        if candidate["family"] != "repaired" or (candidate["role"] == "conditional" and candidate["batch"] != 1):
            continue
        same = {key: candidate[key] for key in ("case", "bank", "role")}
        values = [r for r in accuracy if r["name"] == candidate["name"]
                  and all(r[k] == v for k, v in same.items())]
        deployed = [r for r in sampling if r["name"] == candidate["name"]
                    and all(r[k] == v for k, v in same.items())]
        reference = [r for r in refinements if all(r[k] == v for k, v in same.items())]
        candidate_gate = accuracy_gate(values, deployed, reference)
        for control in costs:
            if (control["family"] not in ("LGA", "native-Euler", "Picard", "learned-path")
                    or not all(control[k] == v for k, v in same.items())):
                continue
            # Conditional MC genuinely uses 16k realizations for a single histogram.
            if control["family"] == "LGA":
                name = control["name"] + "_" + control["bank"]
            else:
                name = control["name"]
            other = [r for r in accuracy if r["name"] == name and all(r[k] == v for k, v in same.items())]
            other_sampling = [r for r in sampling if r["name"] == name
                              and all(r[k] == v for k, v in same.items())]
            control_gate = accuracy_gate(other, other_sampling, reference)
            saving = control["median_seconds"] - candidate["median_seconds"]
            queries = None
            reason = "Matched finite-reference accuracy/uncertainty unresolved"
            if candidate_gate == control_gate == "finite-reference-diagnostic-only":
                if saving > 0:
                    setup_keys = ("known_offline_seconds", "restoration_seconds", "runtime_preparation_seconds",
                                  "cold_compilation_seconds", "compiled_cache_setup_seconds",
                                  "cache_creation_B1_probe_seconds")
                    # Q calls cost setup + first + (Q-1)*warm.
                    additional_setup = (sum(candidate[key] - control[key] for key in setup_keys)
                        + candidate["first_prediction_seconds"] - control["first_prediction_seconds"])
                    queries = max(1, math.ceil(1 + additional_setup / saving))
                    reason = "Finite-reference diagnostic crossover; approximate-law/stopping limits remain"
                else:
                    reason = "No positive measured warm online saving"
            rows.append({**same, "candidate": candidate["name"], "control": name,
                "candidate_gate": candidate_gate, "control_gate": control_gate,
                "online_saving_seconds_per_literal_call": saving,
                "queries": queries, "reason": reason})
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _range(values, scale=1., digits=3):
    values = [float(value) * scale for value in values if value is not None]
    if not values:
        return "—"
    a, b = min(values), max(values)
    return f"{a:.{digits}g}" if a == b else f"{a:.{digits}g}–{b:.{digits}g}"


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(str(value) for value in row) + " |" for row in rows)])


def _maxima(rows):
    return (max(r["max_mean_error_in_reference_std"] for r in rows),
            max(r["max_normalized_covariance_error"] for r in rows))


def _summary_rows(accuracy):
    grouped = defaultdict(list)
    for row in accuracy:
        grouped[(row["name"], row["family"], row["bank"], row["seed"])].append(row)
    return [{"name": name, "family": family, "bank": bank, "seed": seed,
             "worst_mean_error": _maxima(values)[0], "worst_covariance_error": _maxima(values)[1],
             "worst_mean_plus_sampling": max(r["mean_error_plus_sampling95"] for r in values),
             "worst_covariance_plus_sampling": max(r["covariance_error_plus_sampling95"] for r in values)}
            for (name, family, bank, seed), values in grouped.items()]


def make_figures(output, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    summary, costs = tables["model_summary"], tables["costs"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for col, bank in enumerate(BANKS):
        for row, key in enumerate(("worst_mean_error", "worst_covariance_error")):
            ax = axes[row, col]
            for x, family in enumerate(FAMILIES):
                values = [100 * item[key] for item in summary if item["bank"] == bank and item["family"] == family]
                ax.scatter(np.full(len(values), x), values, s=28, color=COLORS[family], alpha=.8)
            ax.axhline(5, color="#555555", ls="--", lw=1)
            ax.set_yscale("log")
            ax.set_xticks(range(len(FAMILIES)), [LABELS[f] for f in FAMILIES], rotation=60, ha="right")
            ax.set_ylabel("Worst mean error (% reference SD)" if row == 0 else "Worst covariance error (%)")
            ax.set_title(bank)
    fig.suptitle("All optimizer seeds: worst independent error over profiles, groups and horizons")
    fig.savefig(output / "accuracy_banks.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, role in zip(axes, ("conditional", "ensemble"), strict=True):
        for x, family in enumerate(FAMILIES):
            values = [r["variance_ratio"] for r in tables["components"] if r["bank"] == BANKS[1]
                and r["role"] == role and r["family"] == family and "integral" in r["observable"]]
            ax.scatter(np.full(len(values), x), np.maximum(values, 1e-6), s=12, color=COLORS[family], alpha=.4)
        ax.axhline(1, color="#555555", ls="--")
        ax.set_yscale("log")
        ax.set_xticks(range(len(FAMILIES)), [LABELS[f] for f in FAMILIES], rotation=60, ha="right")
        ax.set_title(role)
        ax.set_ylabel("Integral variance / reference variance")
    fig.suptitle("Time-integral covariance collapse check (plot display floor 10⁻⁶ only)")
    fig.savefig(output / "integral_variance_ratios.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, bank in enumerate(BANKS):
        for row, stage in enumerate(("mean", "covariance")):
            ax = axes[row, col]
            names = sorted({r["name"] for r in tables["training_history"] if r["bank"] == bank})
            used = set()
            for name in names:
                values = [r for r in tables["training_history"] if r["bank"] == bank
                          and r["stage"] == stage and r["name"] == name]
                family = values[0]["family"]
                label = LABELS[family] if family not in used else None
                used.add(family)
                ax.plot([r["global_step"] for r in values],
                        [max(r["validation_raw_rms"], 1e-14) for r in values],
                        color=COLORS[family], alpha=.65, lw=1, label=label)
                selected = [r for r in values if r["selected"]]
                ax.scatter([r["global_step"] for r in selected],
                           [r["validation_raw_rms"] for r in selected], color=COLORS[family], s=15)
            ax.axvline(1030, color="#999999", ls=":", lw=1)
            ax.set_yscale("log")
            ax.set_xlabel("Optimizer step (repaired covariance follows selected frozen mean)")
            ax.set_ylabel(f"Validation {stage} residual RMS, original units")
            ax.set_title(bank)
            if row == col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Independent validation histories; marker = selected state, no test-based selection")
    fig.savefig(output / "training_losses.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, bank in enumerate(BANKS):
        for row, role in enumerate(("conditional", "ensemble")):
            ax = axes[row, col]
            for family in FAMILIES:
                matching = [r for r in costs if r["family"] == family and r["bank"] == bank
                    and r["role"] == role and (role == "ensemble" or r["batch"] == 1
                        or family in ("learned-path", "native-Euler", "Picard"))]
                x, y = [], []
                for cost in matching:
                    name = cost["name"] + "_" + bank if family == "LGA" else cost["name"]
                    values = [r for r in tables["accuracy"] if r["case"] == cost["case"]
                        and r["role"] == role and r["name"] == name and r["bank"] == bank]
                    if values:
                        x.append(1000 * cost["median_seconds"])
                        y.append(100 * max(r["max_normalized_covariance_error"] for r in values))
                ax.scatter(x, y, label=LABELS[family], color=COLORS[family], s=25, alpha=.8)
            ax.axhline(5, color="#555555", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Measured warm literal-call time (ms)")
            ax.set_ylabel("Worst covariance error (%)")
            ax.set_title(f"{bank}: {role}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    fig.savefig(output / "accuracy_cost.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, role in zip(axes, ("conditional", "ensemble"), strict=True):
        for x, family in enumerate(FAMILIES):
            values = [1000 * r["median_seconds"] for r in costs if r["family"] == family
                and r["role"] == role and (role == "ensemble" or r["batch"] == 1
                    or family in ("learned-path", "native-Euler", "Picard"))]
            ax.scatter(np.full(len(values), x), values, color=COLORS[family], alpha=.65, s=20)
        ax.set_yscale("log")
        ax.set_xticks(range(len(FAMILIES)), [LABELS[f] for f in FAMILIES], rotation=60, ha="right")
        ax.set_ylabel("Warm literal-call time (ms)")
        ax.set_title(role + (": one supplied state" if role == "conditional" else ": actual 16,384-state integration"))
    fig.suptitle("One bank, three horizons; same-time four means and 4×4 covariance")
    fig.savefig(output / "inference_costs.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, role in zip(axes, ("conditional", "ensemble"), strict=True):
        for x, family in enumerate(FAMILIES):
            values = [100 * r["covariance95_radius"] for r in tables["sampling"]
                      if r["family"] == family and r["role"] == role]
            ax.scatter(np.full(len(values), x), np.maximum(values, 1e-3),
                       color=COLORS[family], alpha=.35, s=12)
        ax.axhline(1.25, color="#555555", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xticks(range(len(FAMILIES)), [LABELS[f] for f in FAMILIES], rotation=60, ha="right")
        ax.set_ylabel("Standalone covariance sampling radius (%)")
        ax.set_title(role + "; zeros displayed at 0.001%")
    fig.suptitle("Deployment MC variability is separate from paired method-error intervals")
    fig.savefig(output / "deployed_sampling.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, bank in enumerate(BANKS):
        for row, role in enumerate(("conditional", "ensemble")):
            ax = axes[row, col]
            for family in ("repaired", "LGA", "native-Euler", "learned-path"):
                values = [r for r in tables["total_costs"] if r["family"] == family
                    and r["bank"] == bank and r["role"] == role
                    and (role == "ensemble" or r["batch"] == 1 or family in ("native-Euler", "learned-path"))]
                by_q = defaultdict(list)
                for value in values:
                    by_q[value["query_calls"]].append(value["total_known_seconds"])
                q = sorted(by_q)
                ax.plot(q, [np.median(by_q[n]) for n in q], color=COLORS[family], label=LABELS[family])
                ax.fill_between(q, [min(by_q[n]) for n in q], [max(by_q[n]) for n in q],
                                color=COLORS[family], alpha=.12)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Repeated literal calls")
            ax.set_ylabel("Known total fitting/setup/prediction cost (s)")
            ax.set_title(f"{bank}: {role}")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Descriptive amortization; bands retain profiles/seeds, no accuracy-qualified winner implied")
    fig.savefig(output / "total_costs.png", dpi=180)
    plt.close(fig)


def report_text(evidence, tables):
    accuracy, summary, costs, training = (tables[k] for k in
                                         ("accuracy", "model_summary", "costs", "training"))
    repaired = [r for r in accuracy if r["family"] == "repaired"]
    mixed = [r for r in repaired if r["bank"] == BANKS[1]]
    untrained = [r for r in accuracy if r["family"] == "untrained-repaired" and r["bank"] == BANKS[1]]
    final = [r for r in training if not r["historical"]]
    fresh_fit_seconds = sum(r["training_seconds"] for r in final)
    worker_seconds = sum(r["cpu_seconds"] for r in tables["workers"])
    stops = sum(g["stopped_reference_paths"] for g in evidence["groups"])
    qualified = [r for r in tables["amortization"] if r["queries"] is not None]
    lines = ["# Statistics neural solver: formulation repair and independent comparison", "",
        f"The repaired mixed terminal/time-integral head has worst mean and covariance point discrepancies of "
        f"**{100 * _maxima(mixed)[0]:.3g}% of reference SD** and **{100 * _maxima(mixed)[1]:.3g}%**. "
        f"The untrained heat anchor gives **{100 * _maxima(untrained)[0]:.3g}% / {100 * _maxima(untrained)[1]:.3g}%**. "
        "All optimizer seeds, profiles, anchors and three physical horizons remain in these maxima.", "",
        f"The repaired point errors are below the 5% research target, but the worst covariance discrepancy plus "
        f"its sampling diagnostic is **{100 * max(r['covariance_error_plus_sampling95'] for r in repaired):.3g}%**. "
        "The independent sample allocation therefore does not certify the 5% target or an accuracy-qualified cost crossover.", "",
        "The independent test uses 98,304 fresh reference trajectory families, disjoint from the learner pools, "
        "with 198 complete method records. No fitted variant or seed was selected, discarded or refitted using these outcomes. "
        "This report concerns the fixed-grid stopped native DK target and same-time moments only.", "",
        "## Diagnosis and revised formulation", "",
        "The previous audit found no source-sign or output-normalization defect in the integral equations. "
        "The old shared-network ansatz left the leading integral-mean derivative to be learned, and its unbalanced "
        "sum of mean/covariance residual losses allowed mean progress alongside worse covariance residuals. "
        "Those observations motivated the following preregistered intervention; they do not establish a unique failure cause.", "",
        "The revised model inserts the exact discrete free-heat mean for terminal observables and running integrals. "
        "A deterministic eight-node Gaussian quadrature supplies a positive free-heat covariance anchor with the correct "
        "terminal/integral time orders. Separate mean and covariance networks learn corrections using the full interacting, "
        "stopped native generator residual. The mean is selected on independent validation and frozen before covariance fitting. "
        "Loss scales are fixed per output and covariance pair using training states only. The physical noise is handled by the "
        "exact generator trace over face directions; there is no stochastic variational trace noise and no covariance floor.", "",
        "The anchor itself is an unstopped free-heat approximation, not the interacting stopped solution. Its untrained "
        "control is evaluated separately. Boundary gates and killed running sources remain in the learned formulation. "
        "The revised networks and staged optimizer allocation increase capacity and fitting budget; this experiment tests "
        "a combined formulation/training repair, not an equal-capacity attribution to one architectural change.", "",
        "The pretest audit verifies the fresh learner families against prior learner and previously exposed test families, "
        "replays the addressed horizon/state streams, and verifies reference RNG-address separation before any final "
        "reference outcome is materialized. Its immutable audit is retained as report evidence.", "",
        "The fresh v1 validation controls distinguish two failure mechanisms. With the raw joint loss, a time-integral "
        "covariance can shrink while its mean-derived quadratic-variation source remains close to the heat anchor. With "
        "balanced joint loss, the optimizer can also shrink that source by flattening the learned mean's state derivatives. "
        "Consequently, balancing the two losses alone need not repair the statistical task. Staging freezes the selected "
        "mean source during covariance fitting, while the analytic mean anchor protects the known integral leading term. "
        "These observed diagnostics support this mechanism for the tested fits; they do not identify a universal failure mode.", "",
        "## Frozen experiment", "",
        "G=32, N=2048, D=1, κ=0.75, a₂=0.1 and T=1/16. Initial histograms use IID multinomial draws from uniform "
        "and exp(0.2 cos x) laws. The stopping threshold is 0.01/(2π), with freezing at the first polygon crossing. "
        "Running integrals stop accumulating at that hit. All stopped realizations are retained; averaging survivors is forbidden.", "",
        "The two four-output banks are terminal sine/cosine modes 3 and 4, and terminal mode 3 plus killed integrals of "
        "mode 1 divided by the fixed final horizon. Queries are at 1/64, 1/32 and 1/16. Native cellwise reconstruction uses "
        "exact Fourier cell integrals, with identical definitions for every solver.", "",
        "Fresh training and validation share literal pools across all ten fits: 128 and 64 independent paths per profile, "
        "respectively, producing 4,096 and 2,048 collocation state/horizon rows. Four seed-zero v1 ablations retain raw and "
        "training-initial-RMS losses for both banks. Six repaired fits retain all three optimizer seeds. Each repaired stage "
        "has 1,000 AdamW steps plus 30 LBFGS steps; each v1 fit has one such allocation. All twelve historical heads, "
        "the two untrained repaired controls, three finest path seeds, compiled native Euler, exact-kernel Picard and two "
        "literal three-query Gaussian steps remain in the independent comparison.", "",
        "## Independent accuracy", ""]
    rows = []
    for bank in BANKS:
        for family in FAMILIES:
            values = [r for r in summary if r["bank"] == bank and r["family"] == family]
            if values:
                rows.append([bank, LABELS[family], _range([r["worst_mean_error"] for r in values], 100),
                             _range([r["worst_covariance_error"] for r in values], 100)])
    lines += [_table(["Bank", "Method", "Per-model worst mean (% SD)", "Worst covariance (%)"], rows), "",
        "Ranges retain individual optimizer seeds and, for LGA, the two numerical steps. Every maximum covers both "
        "profiles, all fresh conditional/ensemble groups and all query times. Covariance errors are normalized by products "
        "of reference standard deviations. A large mean percentage measures reference standard deviations, not relative error "
        "in a possibly tiny mean. Individual means, variances and all covariance entries are saved in the CSVs.", "",
        "The learned correction is needed for the mixed-bank integral means: the untrained anchor's worst mean "
        "discrepancy is about 39.4% of reference SD. For the terminal-only bank, the untrained anchor is already "
        "comparable to the repaired head and LGA. This experiment supports the integral repair; it does not show "
        "a general learned accuracy improvement for the high terminal modes.", "",
        "![](accuracy_banks.png)", "", "![](integral_variance_ratios.png)", "",
        "The variance-ratio figure displays every integral output, horizon, profile, anchor and seed. Its plotting floor "
        "does not modify any prediction or CSV value. A ratio near one is distinct from a guarantee on the complete covariance matrix.", "",
        "## Training behavior", "", "![](training_losses.png)", "",
        "The loss figure uses raw residual RMS in the original internal units, since normalized objectives differ between "
        "formulations. The repaired covariance stage begins after step 1,030 with the selected mean frozen; its checkpoint "
        "loader verifies equality of all mean-network tensors throughout covariance-stage states. Selected validation states "
        "are marked. The covariance curve starts from a reconstructed baseline with that selected mean and the still-initial "
        "covariance head. Its source Γ(u) differs from the source before mean fitting, so the saved original covariance "
        "validation is retained only as an explicitly obsolete-target diagnostic in validation_diagnostics.csv; it is not "
        "plotted as the covariance-stage baseline. A low residual by itself does not establish prediction accuracy.", ""]
    fit_rows = []
    for family in ("v1-none", "v1-balanced", "repaired", "old-cold", "old-trunk-transfer", "learned-path"):
        values = [r for r in training if r["family"] == family]
        fit_rows.append([LABELS[family], len(values), _range([r["parameter_count"] for r in values]),
                         _range([r["training_seconds"] for r in values]),
                         _range([r["prior_trunk_training_seconds"] for r in values])])
    lines += [_table(["Fit family", "Models", "Parameters", "Fit seconds/model", "Prior trunk seconds"], fit_rows), "",
        f"All ten fresh fits together recorded **{fresh_fit_seconds:.5g} model fitting seconds**, including ablations "
        f"and every optimizer seed. Fresh workers recorded **{worker_seconds:.5g} CPU seconds** across attempts. "
        "Concurrent worker wall times are not summed and presented as batch elapsed time. Shared data generation is "
        f"**{sum(r['generation_seconds'] for r in tables['data']):.4g} seconds**, charged once in a fresh deployment. "
        "Historical transferred heads additionally pay their original trunk fit; historical data-generation costs are "
        "not fully available and remain explicitly unrecorded.", "",
        _table(["Mixed-bank validation control", "Seed", "Sine-integral QV / heat", "Sine-integral covariance / heat"],
            [[r["variant"], r["seed"], f"{r['mean_QV_source_over_heat_anchor']:.5g}",
              f"{r['covariance_over_heat_anchor']:.5g}"] for r in tables["collapse_diagnostics"]
             if r["bank"] == BANKS[1] and r["observable"] == "killed-integral-sin1-over-T"]), "",
        "These ratios use validation states and the known heat anchor, not independent interacting truth. "
        "collapse_diagnostics.csv retains every output, including cosine integrals and terminal channels. "
        "validation_diagnostics.csv retains per-stage raw and normalized residuals; the audit JSON preserves "
        "the frozen-mean hashes. validation_components.csv preserves per-output and per-pair physical, internal "
        "and dimensionless residual diagnostics without pooling their scales.", "",
        "## Inference and total cost", "", "![](accuracy_cost.png)", "", "![](inference_costs.png)", ""]
    cost_rows = []
    for family in FAMILIES:
        values = [r for r in costs if r["family"] == family]
        steps = (2**-7, 2**-10) if family == "LGA" else (None,)
        for step in steps:
            selected = [r for r in values if step is None or r["dt"] == step]
            conditional = [r for r in selected if r["role"] == "conditional" and
                           (r["batch"] == 1 or family in ("learned-path", "native-Euler", "Picard"))]
            ensemble = [r for r in selected if r["role"] == "ensemble"]
            label = LABELS[family] + (f", max step 1/{round(1 / step)}" if step is not None else "")
            cost_rows.append([label, _range([r["median_seconds"] for r in conditional], 1000),
                              _range([r["median_seconds"] for r in ensemble], 1000),
                              _range([r["restoration_seconds"] + r["runtime_preparation_seconds"]
                                      for r in selected], 1000)])
    lines += [_table(["Method", "Conditional warm ms", "Initial-ensemble warm ms", "Restore/preparation ms"], cost_rows), "",
        "Each call returns one declared bank's four means and same-time 4×4 covariance matrices at three horizons. "
        "Conditional neural timings use one supplied histogram; batched 64/256-state throughput remains in costs.csv. "
        "Ensemble heads actually draw and integrate 16,384 fresh initial histograms. LGA uses analytic IID initial "
        "covariance and the literal three-query algorithm in both accuracy and timing; its two step sizes remain separate "
        "in the CSV. Path-based calls actually simulate 16,384 realizations, include fresh physical Brownian forcing "
        "and applicable initial histograms, apply exactly one bank and aggregate moments. Only path seed zero is timed; "
        "all path seeds remain in accuracy. No large-ensemble cost is extrapolated from a small batch.", "",
        "Warm timing follows fitting and scientific evaluation, using one CPU thread. Restoration/constructor time "
        "and the first measured prediction at each allocation are recorded separately. Cache creation is charged to the "
        "first B=1 conditional call of each restored/constructed model; later B=64/B=256 and ensemble timings reuse its "
        "known-horizon cache. Their recorded first prediction is therefore first at that allocation, not a fully cold model. "
        "For repaired/untrained later-allocation total costs, the measured B=1 first-prediction probe is conservatively "
        "charged once as a cache-creation probe, with its output discarded. It is labelled a probe rather than pure preparation. Compilation and "
        "path-kernel preparation remain explicit. Uncontrolled Python/OS startup and unknown historical costs are not invented.", "",
        "![](total_costs.png)", "",
        "total_costs.csv gives 1, 100, 10,000 and 100,000 literal repeated calls: known fitting, prior trunk, shared-data, "
        "restoration/preparation, compilation, applicable B=1 cache-creation probe, one first prediction at the requested "
        "allocation and subsequent warm calls. This is a cost accounting "
        "table, not an accuracy-qualified winner. The separate full-study total includes all fresh ablations and seeds; "
        "deploying one fixed model does not require fitting the entire study inventory.", "",
        f"The declared finite-reference accuracy/uncertainty gates produce **{len(qualified)} resolved crossover rows**. "
        "All others retain an explicit unresolved or no-saving reason in amortization.csv. Approximate Gaussian law "
        "bias and continuous stopping bias remain separate even if a finite-reference diagnostic gate passes.", "",
        "## Sampling, refinement and interpretation", "", "![](deployed_sampling.png)", "",
        "Whole-family bootstrap intervals use 128 replicates and preserve all horizons/observables from each family. "
        "Paired path/reference intervals quantify uncertainty in a method-discrepancy estimate under shared physical "
        "noise. They are distinct from the standalone Monte Carlo variability of deploying 16,384 trajectories. "
        "Ensemble heads have initial-law integration MC variability; deterministic conditional heads do not. Both "
        "quantities and the reference's own sampling variability are saved separately.", "",
        "The reference step is 2⁻¹³, with coupled 2⁻¹⁴ and 2⁻¹⁵ checks on 1,024 families per group. The report's "
        "refinement CSV uses the coupled base-versus-finest comparison. These finite-step differences do not enclose "
        "continuous boundary-hitting bias. The 5% normalized statistical error and 1.25% component uncertainty targets "
        "remain research targets, not automatic certifications from the fixed sample allocation.", "",
        f"Observed reference stops: **{stops} / 98,304**. Every stopped path is retained. Zero observed stops does "
        "not establish rare-boundary accuracy. The cheap LGA and free-heat controls are unstopped approximations, and "
        "agreement in this weak-noise short-horizon setting does not identify them with the stopped DK law.", "",
        "For fixed supported conditional moments, compare the repaired head's accuracy with LGA and its fitting/setup "
        "amortization before choosing it. The repaired head has cheaper warm conditional calls here; the no-training "
        "coarse LGA is the inexpensive choice for occasional queries and initial-ensemble moments at comparable "
        "observed statistical errors, subject to its approximation bias. Native Monte Carlo gives smaller finite-reference "
        "point discrepancies for these banks at a larger online sampling cost. Initial-ensemble questions also pay "
        "initial-law integration for neural heads. "
        "Full paths, new observables, joint-time quantities, scalar laws and stopping probabilities remain tasks for "
        "path samplers; these heads do not claim those outputs. Classical full paths offer the same observable reuse "
        "as neural paths, so flexibility alone is not a learned computational advantage.", "",
        "Every result, model sidecar, input and report output is checksum indexed in manifest.json. Full weights and "
        "raw trajectories remain in their local run directories. The compact documentation archive preserves the "
        "report, tables, individual JSON records, stage histories, exact analysis code and a full raw-artifact inventory.", ""]
    return "\n".join(lines)


def generate(root, *, output=None, documentation=None):
    evidence = load_complete(root)
    output = Path(output or evidence["root"] / "report").resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = accuracy_tables(evidence)
    tables["sampling"], tables["reference_sampling"] = sampling_tables(evidence)
    tables["training"], tables["training_history"], tables["workers"], tables["data"] = training_tables(evidence)
    tables["validation_diagnostics"] = validation_diagnostic_tables(evidence)
    tables["collapse_diagnostics"] = collapse_diagnostic_tables(evidence)
    tables["validation_components"] = validation_component_tables(evidence)
    tables["costs"], tables["total_costs"] = cost_tables(evidence, tables["training"], tables["data"])
    tables["study_costs"] = study_costs(evidence, tables["training"], tables["workers"], tables["data"])
    tables["model_summary"] = _summary_rows(tables["accuracy"])
    tables["amortization"] = amortization_tables(tables["accuracy"], tables["sampling"],
                                                tables["reference_refinement"], tables["costs"])
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), rows)
    make_figures(output, tables)
    text = report_text(evidence, tables)
    (output / "report.md").write_text(text)
    summary = {"kind": KIND, "complete": True, "source_hash": source_hash(),
               "generator_sha256": sha(__file__), "inputs": evidence["inputs"],
               "groups": list(evidence["cfg"].group_keys), "method_records": len(evidence["records"]),
               "accuracy_rows": len(tables["accuracy"]), "timing_records": len(evidence["timing"]["records"]),
               "fresh_fit_count": len(evidence["lock"]["models"]),
               "fresh_total_fit_seconds": sum(r["training_seconds"] for r in tables["training"] if not r["historical"]),
               "resolved_crossover_rows": sum(r["queries"] is not None for r in tables["amortization"]),
               "seeds_pooled": False, "post_test_refitting": False,
               "continuous_DK_claim": False, "continuous_stopping_bias": "unresolved",
               "outputs": {p.name: sha(p) for p in sorted(output.iterdir())
                           if p.is_file() and p.name != "manifest.json"}}
    json_write(output / "manifest.json", summary)
    if documentation is not None:
        path = Path(documentation).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = "artifacts/payoff_repair_2026-10-27/report/"
        for figure in output.glob("*.png"):
            text = text.replace(f"![]({figure.name})", f"![]({archive}{figure.name})")
        text += ("\nThe [compact evidence archive](artifacts/payoff_repair_2026-10-27/README.md) "
                 "preserves report outputs and checksum-indexed evidence.\n")
        path.write_text(text)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "runs/payoff-repair-2026-10-27")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--documentation", type=Path,
                        default=REPO / "docs/experiments/payoff_repair_2026-10-27.md")
    args = parser.parse_args()
    result = generate(args.root, output=args.output, documentation=args.documentation)
    print(json.dumps({"complete": result["complete"], "records": result["method_records"],
                      "outputs": len(result["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
