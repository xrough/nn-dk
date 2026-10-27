"""Postprocess the frozen boundary-repair experiment without selecting fits.

All legacy and repaired seeds are retained. Independent initial-law estimates
replace the paired diagnostic estimates for every neural family. Historical
costs are distinguished from the incremental cost of the fresh experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402

from analysis import evaluate_payoff_boundary_deployment as deployment  # noqa: E402
from analysis import evaluate_payoff_boundary_regimes as evaluator  # noqa: E402
from analysis import payoff_legacy_control as legacy  # noqa: E402
from analysis import run_payoff_boundary_regimes as runner  # noqa: E402
from analysis.payoff_study_adapter import bind_study, isolated_module  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402

KIND = "boundary-repaired-all-seed-independent-accuracy-cost-report-v1"
NEURAL = ("repaired", "legacy-v2", "untrained-repaired")
_implementation = isolated_module("report_payoff_regimes.py", "analysis._boundary_regimes_report_impl")
bind_study(_implementation, __file__, runner=runner, comparison_plan=evaluator.prepare, KIND=KIND)
read, sha, json_write = _implementation.read, _implementation.sha, _implementation.json_write
csv_write, _artifact = _implementation.csv_write, _implementation._artifact


def _expected_names(evidence, case):
    names = {row["name"] for row in evidence["lock"]["models"] if row["case"] == case}
    names |= {row["name"] for row in evidence["legacy"]["models"] if row["case"] == case}
    return names | {case+"_untrained", "LGA_dt7", "LGA_dt10", "native-MC-dt10"}


def _head_inventory(records, seeds):
    for family in ("repaired", "legacy-v2"):
        found = [row["method"]["seed"] for row in records if row["method"]["family"] == family]
        if len(found) != len(seeds) or set(found) != set(seeds):
            raise ValueError("Every repaired and historical optimizer seed must be retained")
    if sum(row["method"]["family"] == "untrained-repaired" for row in records) != 1:
        raise ValueError("Exactly one untrained architecture control is required")


def _verify_timing(evidence):
    timing, study = evidence["timing"], evidence["study"]
    slots, loads = set(), set()
    expected_slots, expected_loads = set(), set()
    for case in evidence["plan"]["cases"]:
        cfg = runner.reference_configuration(case, study)
        heads = _expected_names(evidence, case)-{"LGA_dt7", "LGA_dt10", "native-MC-dt10"}
        for profile in cfg.cases:
            for name in heads:
                expected_loads.add((case, profile, name))
                for batch in evidence["plan"]["neural_conditional_batches"]:
                    expected_slots.add(f"{case}/{profile}/{name}/conditional/{batch}")
                expected_slots.add(f"{case}/{profile}/{name}/ensemble")
            for step in evidence["plan"]["LGA_max_steps"]:
                for role in ("conditional", "ensemble"):
                    expected_slots.add(f"{case}/{profile}/LGA/{step}/{role}")
            for role in ("conditional", "ensemble"):
                expected_slots.add(f"{case}/{profile}/native-MC/{role}")
    for row in timing["records"]:
        if row["slot"] in slots or row["median_seconds"] < 0 or row["first_prediction_seconds"] < 0:
            raise ValueError("Duplicate or invalid literal timing record")
        slots.add(row["slot"])
    for row in timing["loading"]:
        key = row["regime"], row["profile"], row["name"]
        if key in loads or row["seconds"] < 0:
            raise ValueError("Duplicate or invalid neural restoration timing")
        loads.add(key)
    if slots != expected_slots or loads != expected_loads:
        raise ValueError("Every declared inference workload and restoration must be timed")


def load_complete(root, *, independent_initials=True):
    root = Path(root).resolve()
    study, _ = runner._assert_frozen(root)
    plan, lock = evaluator.prepare(root), runner.verify_fit_lock(root)
    inputs = {}
    for name in ("plan.json", "payoffs/preregistration.json", "payoffs/data_lock.json",
                 "payoffs/fit_lock.json", "comparison/preregistration.json", "comparison/legacy_controls.json",
                 "comparison/accuracy_index.json", "comparison/timing.json"):
        _artifact(root/name, inputs)
    controls = read(root/"comparison/legacy_controls.json")
    if (sha(root/"comparison/legacy_controls.json") != plan["legacy_controls_sha256"]
            or controls != legacy.inventory(plan["cases"], study.optimizer_seeds)):
        raise ValueError("Historical source and all selected controls must match their frozen inventory")
    legacy.verify_targets(controls, runner, study)
    index, timing = read(root/"comparison/accuracy_index.json"), read(root/"comparison/timing.json")
    plan_sha, lock_sha = sha(root/"comparison/preregistration.json"), sha(root/"payoffs/fit_lock.json")
    if (not index.get("complete") or index["source_hash"] != source_hash()
            or index["plan_sha256"] != plan_sha or index["records_per_group"] != plan["records_per_group"]
            or not timing.get("complete") or not timing.get("final_timing_after_all_workers_finished")
            or timing["identity"] != {"source_hash": source_hash(), "plan_sha256": plan_sha,
                                      "fit_lock_sha256": lock_sha}):
        raise ValueError("Complete matching independent accuracy and uncontended final timing required")
    evidence = {"root": root, "study": study, "plan": plan, "lock": lock, "legacy": controls,
                "timing": timing, "inputs": inputs, "records": [], "groups": [], "references": {},
                "paired_initial_records": [], "paired_initial_groups": []}
    expected = {(case, group) for case in plan["cases"]
                for group in runner.reference_configuration(case, study).group_keys}
    if {tuple(pair) for pair in index["groups"]} != expected:
        raise ValueError("The entire fixed physical group inventory is required")
    for case in plan["cases"]:
        manifest = read(_artifact(root/"reference"/case/"manifest.json", inputs))
        if manifest.get("status") != "complete":
            raise ValueError("Incomplete native reference")
        _artifact(root/"reference"/case/"reference.h5", inputs, manifest["reference_sha256"])
        evidence["references"][case] = manifest
    for case, group in sorted(expected):
        directory = root/"comparison/accuracy"/case/group
        packet = read(_artifact(directory/"group.json", inputs))
        base = {"source_hash": source_hash(), "comparison_plan_sha256": plan_sha,
                "fit_lock_sha256": lock_sha, "regime": case,
                "reference_sha256": evidence["references"][case]["reference_sha256"],
                "bank_hash": runner.payoff_bank(case, study).bank_hash}
        if (not packet.get("complete") or packet["identity"] != base or packet["group"] != group
                or len(packet["records"]) != plan["records_per_group"]):
            raise ValueError("Incomplete or incompatible main method group")
        rows = []
        for item in packet["records"]:
            record = read(_artifact(item["path"], inputs, item["sha256"]))
            if (not record.get("complete") or record["identity"] != {
                    **base, "group": group, "name": record["method"]["name"]}):
                raise ValueError("Main accuracy record identity changed")
            _artifact(record["arrays"]["path"], inputs, record["arrays"]["sha256"])
            rows.append(record)
        if {row["method"]["name"] for row in rows} != _expected_names(evidence, case):
            raise ValueError("An accuracy method is missing or duplicated")
        _head_inventory(rows, study.optimizer_seeds)
        for name in ("NN_versus_fine_LGA", "NN_versus_legacy_v2"):
            if set(packet.get(name, {})) != {r["name"] for r in lock["models"] if r["case"] == case}:
                raise ValueError("Every new optimizer seed needs both independent comparisons")
        evidence["records"].extend(rows)
        evidence["groups"].append({**packet, "regime": case})
    for item in lock["models"]+controls["models"]:
        for key in ("checkpoint", "metadata"):
            _artifact(item[key+"_path"], inputs, item[key+"_sha256"])
    evidence["original_accuracy_records"] = len(evidence["records"])
    if independent_initials:
        _independent_initial_primary(evidence)
    _verify_timing(evidence)
    return evidence


def _graft_independent_records(evidence, replacements, groups):
    """Every neural family shares the independent initial-law integration."""
    primary, paired = [], []
    for record in evidence["records"]:
        identity = record["identity"]
        key = identity["regime"], identity["group"], identity["name"]
        if "__ensemble__" in identity["group"] and record["method"]["family"] in NEURAL:
            if key not in replacements:
                raise ValueError("Independent neural initial-pool replacement is incomplete")
            paired.append(record)
            primary.append(replacements.pop(key))
        else:
            primary.append(record)
    if replacements:
        raise ValueError("An undeclared independent neural estimate appeared")
    paired_groups, primary_groups = [], []
    for group in evidence["groups"]:
        key = group["regime"], group["group"]
        if key in groups:
            paired_groups.append(group)
            replacement = groups[key]
            primary_groups.append({**group, "NN_versus_fine_LGA": replacement["NN_versus_fine_LGA"],
                "NN_versus_legacy_v2": replacement["NN_versus_legacy_v2"],
                "primary_NN_initial_integration": "independent-of-reference"})
        else:
            primary_groups.append(group)
    evidence.update(records=primary, groups=primary_groups, paired_initial_records=paired,
                    paired_initial_groups=paired_groups, independent_initial_records=len(paired))


def _independent_initial_primary(evidence):
    root, inputs, study = evidence["root"], evidence["inputs"], evidence["study"]
    declared = deployment.prepare(root)
    path = _artifact(root/"deployment/preregistration.json", inputs)
    index = read(_artifact(root/"deployment/accuracy_index.json", inputs))
    expected = {(pool["regime"], pool["group"]) for pool in declared["pools"]}
    if (not index.get("complete") or index["source_hash"] != source_hash()
            or index["deployment_plan_sha256"] != sha(path)
            or index["fit_lock_sha256"] != sha(root/"payoffs/fit_lock.json")
            or {tuple(v) for v in index["groups"]} != expected
            or index["expected_records"] != declared["expected_records"]
            or index["records_per_group"] != declared["models_per_pool"]):
        raise ValueError("All independent ensemble records are required")
    replacements, groups = {}, {}
    for case, key in sorted(expected):
        group = read(_artifact(root/"deployment/accuracy"/case/key/"group.json", inputs))
        base = group["identity"]
        if (not group.get("complete") or not group.get("independent_initials")
                or len(group["records"]) != declared["models_per_pool"]
                or base["source_hash"] != source_hash() or base["regime"] != case or base["group"] != key
                or base["deployment_plan_sha256"] != sha(path)
                or base["fit_lock_sha256"] != index["fit_lock_sha256"]
                or base["comparison_plan_sha256"] != sha(root/"comparison/preregistration.json")
                or base["reference_sha256"] != evidence["references"][case]["reference_sha256"]
                or base["bank_hash"] != runner.payoff_bank(case, study).bank_hash):
            raise ValueError("Independent ensemble law identity changed")
        initial = read(_artifact(group["initial_pool_record_path"], inputs, group["initial_pool_record_sha256"]))
        _artifact(initial["arrays"]["path"], inputs, initial["arrays"]["sha256"])
        if initial["arrays"]["sha256"] != base["independent_initial_pool_sha256"]:
            raise ValueError("Independent initial pool changed")
        records = []
        for item in group["records"]:
            record = read(_artifact(item["path"], inputs, item["sha256"]))
            if (not record.get("complete") or not record.get("initial_pool_independent_of_reference")
                    or not record.get("reference_and_initial_bootstrap_independent")
                    or record["identity"] != {**base, "name": record["method"]["name"]}):
                raise ValueError("Independent neural estimator evidence changed")
            _artifact(record["arrays"]["path"], inputs, record["arrays"]["sha256"])
            replacements[(case, key, record["method"]["name"])] = record
            records.append(record)
        _head_inventory(records, study.optimizer_seeds)
        if any(row["method"]["family"] not in NEURAL for row in records):
            raise ValueError("Only neural families belong to the initial integration supplement")
        groups[(case, key)] = group
    _graft_independent_records(evidence, replacements, groups)
    if evidence["independent_initial_records"] != declared["expected_records"]:
        raise ValueError("Independent initial record allocation changed")


def accuracy_tables(evidence):
    tables = _implementation.accuracy_tables(evidence)
    pools = {(record["identity"]["regime"], record["identity"]["group"], record["identity"]["name"]):
             initial_pool_status(record) for record in evidence["records"]}
    for table in ("accuracy", "components", "covariances", "sampling"):
        for row in tables[table]:
            row["NN_initial_pool"] = pools[(row["regime"], row["group"], row["name"])]
    extra = []
    for group in evidence["groups"]:
        case, key = group["regime"], group["group"]
        cfg = runner.reference_configuration(case, evidence["study"])
        profile, role, anchor = key.split("__")
        for name, contrast in group["NN_versus_legacy_v2"].items():
            for item in contrast["records"]:
                for statistic in ("mean", "covariance"):
                    lower, upper = item[statistic+"_advantage_percentile95"]
                    extra.append({"regime": case, "kappa": cfg.kappa, "horizon": cfg.horizon,
                        "group": key, "profile": profile, "role": role, "anchor": int(anchor),
                        "name": name, "time_index": item["time_index"], "time": cfg.query_times[item["time_index"]],
                        "statistic": statistic, "advantage": item[statistic+"_error_advantage"],
                        "lower95": lower, "upper95": upper, "NN_closer_pointwise95": lower > 0,
                        "LGA_closer_pointwise95": upper < 0, "control_family": "legacy-v2",
                        "scope": "matched optimizer seed; both initial integration uncertainties; pointwise, no family-wise certificate"})
    tables["legacy_contrasts"] = extra
    return tables


def initial_pool_status(record):
    if ("__ensemble__" not in record["identity"]["group"]
            or record["method"]["family"] not in NEURAL):
        return "not-applicable"
    return "independent" if record.get("initial_pool_independent_of_reference") else "shared-with-reference"


def observable_types(bank):
    """Join labels to their declared weights, regardless of bank ordering."""
    if len(set(bank.labels)) != len(bank.labels):
        raise ValueError("Observable block summaries require unique bank labels")
    result = {}
    for label, terminal, running in zip(bank.labels, bank.terminal_weights, bank.running_weights, strict=True):
        has_terminal, has_integral = bool(np.any(terminal != 0)), bool(np.any(running != 0))
        if has_terminal == has_integral:
            raise ValueError("Declare pure terminal or pure killed-integral outputs for block diagnostics")
        result[label] = "terminal" if has_terminal else "integral"
    return result


def observable_block_tables(evidence, tables):
    classes = {case: observable_types(runner.payoff_bank(case, evidence["study"]))
               for case in evidence["plan"]["cases"]}
    grouped = {}
    for table, statistic in (("components", "mean"), ("covariances", "covariance")):
        for row in tables[table]:
            category = classes[row["regime"]]
            if statistic == "mean":
                block = category[row["observable"]]
                signed = row["normalized_mean_difference"]
                labels = row["observable"]
            else:
                first, second = category[row["observable_i"]], category[row["observable_j"]]
                block = "TT" if first == second == "terminal" else "II" if first == second == "integral" else "TI"
                signed = row["normalized_covariance_difference"]
                labels = row["observable_i"]+" / "+row["observable_j"]
            key = row["regime"], row["family"], row["name"], statistic, block
            value = grouped.setdefault(key, {**{field: row[field] for field in (
                "regime", "horizon", "kappa", "family", "name", "seed")},
                "statistic": statistic, "observable_block": block, "entries": 0,
                "worst_normalized_error": -1.,
                "normalization": "reference SD" if statistic == "mean" else "reference SD products"})
            value["entries"] += 1
            if abs(signed) > value["worst_normalized_error"]:
                value.update(worst_normalized_error=abs(signed), signed_error_at_worst=signed,
                             worst_group=row["group"], worst_time=row["time"], worst_observable=labels)
    return [grouped[key] for key in sorted(grouped)]


def cost_tables(timing, fit_lock, data_seconds, legacy_models=(), historical_data_seconds=None):
    fits = {(row["case"], row["name"]): row for row in fit_lock["models"]}
    historical = {(row["case"], row["name"]): row for row in legacy_models}
    historical_data_seconds = historical_data_seconds or {}
    loading = {(row["regime"], row["profile"], row["name"]): row["seconds"] for row in timing["loading"]}
    probes = {(row["regime"], row["profile"], row["name"]): row["first_prediction_seconds"]
              for row in timing["records"] if row["role"] == "conditional" and row["batch"] == 1}
    compiled = timing.get("compiled_control", {})
    costs, totals = [], []
    for measured in timing["records"]:
        case, name, profile = measured["regime"], measured["name"], measured["profile"]
        key = case, name
        fitting, data, source = 0., 0., "none"
        if key in fits:
            fitting, data, source = fits[key]["training_seconds"], data_seconds[case], "new-v3"
        elif key in historical:
            fitting = historical[key]["historical_training_seconds"]
            data, source = historical_data_seconds[case], "historical-v2"
        native = measured["family"] == "native-MC"
        compile_seconds = compiled.get("compile_seconds", 0.) if native else 0.
        setup = max(0., compiled.get("setup_seconds_this_process", 0.)
                    - compiled.get("compile_seconds_this_process", 0.)) if native else 0.
        probe = (probes[(case, profile, name)] if measured["family"] in NEURAL
                 and (measured["role"] != "conditional" or measured["batch"] != 1) else 0.)
        restoration = loading.get((case, profile, name), 0.)
        row = {key: value for key, value in measured.items() if key != "samples_seconds"}
        row.update(training_seconds=fitting, shared_data_seconds=data, restoration_seconds=restoration,
                   cold_compilation_seconds=compile_seconds, compiled_setup_seconds=setup,
                   cache_creation_B1_probe_seconds=probe, offline_data_source=source,
                   offline_training_is_historical=source == "historical-v2")
        row["known_offline_seconds"] = fitting+data+restoration+compile_seconds+setup+probe
        costs.append(row)
        for calls in (1, 100, 10000, 100000):
            totals.append({**{key: row[key] for key in ("slot", "regime", "profile", "name", "family", "role", "batch")},
                "query_calls": calls, "known_offline_seconds": row["known_offline_seconds"],
                "first_prediction_seconds": row["first_prediction_seconds"],
                "subsequent_warm_seconds": (calls-1)*row["median_seconds"],
                "total_known_seconds": row["known_offline_seconds"]+row["first_prediction_seconds"]
                    +(calls-1)*row["median_seconds"], "offline_data_source": source})
    return costs, totals


def amortization_tables(tables):
    rows = []
    for candidate in tables["costs"]:
        if candidate["family"] != "repaired" or (candidate["role"] == "conditional" and candidate["batch"] != 1):
            continue
        keys = {key: candidate[key] for key in ("regime", "profile", "role")}
        def compatible(row):
            return all(row[key] == value for key, value in keys.items())
        refinement = [row for row in tables["reference_refinement"] if compatible(row) and row["level"] == "level0_versus_finest"]
        for control in tables["costs"]:
            if control["family"] not in ("LGA", "native-MC", "legacy-v2") or not compatible(control):
                continue
            if control["family"] == "legacy-v2" and (control["seed"] != candidate["seed"]
                    or (control["role"] == "conditional" and control["batch"] != 1)):
                continue
            for statistic in ("mean", "covariance"):
                gates = []
                for item in (candidate, control):
                    accuracy = [row for row in tables["accuracy"] if compatible(row) and row["name"] == item["name"]]
                    sampling = [row for row in tables["sampling"] if compatible(row) and row["name"] == item["name"]]
                    gates.append(_implementation.statistic_gate(accuracy, sampling, refinement, statistic))
                arithmetic = _implementation.payback(candidate, control)
                qualified = arithmetic if all(gate == "finite-reference-diagnostic-pass" for gate in gates) else None
                rows.append({**keys, "candidate": candidate["name"], "control": control["name"],
                    "statistic": statistic, "candidate_gate": gates[0], "control_gate": gates[1],
                    "online_saving_seconds": control["median_seconds"]-candidate["median_seconds"],
                    "descriptive_payback_calls": arithmetic, "accuracy_qualified_payback_calls": qualified,
                    "scope": ("new supplied-state queries" if candidate["role"] == "conditional" else
                              "repeated fresh estimates of fixed IID law; identical answers may be memoized")
                              + "; includes each model's own fit/data; continuous stopping bias unresolved"})
    return rows


def validation_tables(evidence):
    root, inputs = evidence["root"], evidence["inputs"]
    packet = read(_artifact(root/"audit/postfit_validation_diagnostic.json", inputs))
    expected = {row["name"]: row for row in evidence["lock"]["models"]}
    identity = packet["identity"]
    if (not packet.get("complete") or identity["source_hash"] != source_hash()
            or not packet["all_references_absent_at_completion"] or not packet["no_selection_refit_or_future_outcome_access"]
            or identity["future_test_outcomes_opened"] is not False
            or identity["fit_lock_sha256"] != sha(root/"payoffs/fit_lock.json")
            or identity["data_lock_sha256"] != sha(root/"payoffs/data_lock.json")
            or identity["preregistration_sha256"] != sha(root/"payoffs/preregistration.json")
            or {row["name"] for row in packet["records"]} != set(expected)):
        raise ValueError("Matching pre-reference robust validation replay is required")
    residuals, phases, consistency, concentration = [], [], [], []
    for record in packet["records"]:
        if (not record["selected_mean_unchanged"]
                or record["checkpoint_sha256"] != expected[record["name"]]["checkpoint_sha256"]):
            raise ValueError("Validation replay changed the selected network")
        _artifact(record["numeric_evidence"]["path"], inputs, record["numeric_evidence"]["sha256"])
        base = {"regime": record["case"], "name": record["name"], "seed": record["seed"]}
        for stratum, stages in record["full_selected_stratified_residuals"].items():
            for stage, values in stages.items():
                residuals.append({**base, "stratum": stratum, "stage": stage,
                    **{key: value for key, value in values.items() if np.isscalar(value) or value is None}})
                overall = record["full_selected_stratified_residuals"]["all"][stage]
                concentration.append({**base, "stratum": stratum, "stage": stage,
                    "rows": values["rows"], "empty": values.get("empty", False),
                    "normalized_rms": values.get("normalized_rms"),
                    **loss_shares(values, overall)})
        for phase, strata in record["diagnostic_subset_phase_residuals"].items():
            for stratum, stages in strata.items():
                for stage, values in stages.items():
                    phases.append({**base, "phase": phase, "stratum": stratum, "stage": stage,
                        **{key: value for key, value in values.items() if np.isscalar(value) or value is None}})
        consistency.append({**base, **record["full_selected_active_counts"],
            **{key: value if np.isscalar(value) or value is None else json.dumps(value)
               for key, value in record["full_selected_payoff_consistency"].items()}})
    formulation = read(_artifact(root/"audit/formulation_control.json", inputs))
    if (not formulation.get("complete") or formulation["source_hash"] != source_hash()
            or not formulation["no_fitting_checkpoint_trajectory_or_final_test_data_opened"]):
        raise ValueError("Independent new-formulation controls required")
    controls = [{"control": key, "complete": True, "details_json": json.dumps(value, sort_keys=True)}
                for key, value in formulation["controls"].items()]
    return {"validation_residuals": residuals, "matched_validation_phases": phases,
            "validation_consistency": consistency, "validation_loss_concentration": concentration,
            "formulation_control": controls}


def loss_shares(stratum, overall):
    """Robust concentration and squared-residual concentration stay distinct."""
    result = {}
    for label, key in (("robust", "normalized_objective"), ("squared", "normalized_squared_objective")):
        denominator = overall["rows"]*overall[key]
        result["fraction_of_full_"+label+"_loss"] = (
            stratum["rows"]*stratum[key]/denominator if denominator and stratum["rows"] else None)
    return result


def _data_tables(evidence):
    new, historical = [], []
    archive = read(legacy.ARCHIVE/"archive_manifest.json")
    locked = read(evidence["root"]/"payoffs/data_lock.json")
    for case in evidence["plan"]["cases"]:
        for role in ("train", "validation"):
            relative = f"payoffs/{case}/states_{role}.json"
            path = _artifact(evidence["root"]/relative, evidence["inputs"],
                             locked["banks"][case][role]["metadata_sha256"])
            packet = read(path)
            if (not packet.get("complete") or packet["source_hash"] != source_hash()
                    or packet["generation_seconds"] != locked["banks"][case][role]["generation_seconds"]):
                raise ValueError("New shared-data cost differs from its pre-fit lock")
            new.append({"regime": case, "role": role, "generation_seconds": packet["generation_seconds"],
                        "manifest_path": str(path), "manifest_sha256": sha(path), "source_hash": source_hash()})
            archived = legacy._archived_json(relative)
            item = next(value for key, value in archive["copied"].items() if key in (relative, relative+".gz"))
            old_path = _artifact(legacy.HISTORICAL/relative, evidence["inputs"], item["original_sha256"])
            if read(old_path) != archived or archived["source_hash"] != legacy.SOURCE_HASH:
                raise ValueError("Historical state-generation cost provenance changed")
            historical.append({"regime": case, "role": role, "generation_seconds": archived["generation_seconds"],
                "manifest_path": str(old_path), "manifest_sha256": sha(old_path),
                "archived_original_sha256": item["original_sha256"], "source_hash": legacy.SOURCE_HASH})
    return new, historical


def _worker_tables(evidence, historical=False):
    rows = []
    for seed in evidence["study"].optimizer_seeds:
        relative = f"payoffs/execution/worker_seed{seed}.json"
        path = (legacy.HISTORICAL if historical else evidence["root"])/relative
        expected = None
        if historical:
            archive = read(legacy.ARCHIVE/"archive_manifest.json")
            expected = next(value["original_sha256"] for key, value in archive["copied"].items()
                            if key in (relative, relative+".gz"))
        packet = read(_artifact(path, evidence["inputs"], expected))
        if not packet.get("complete"):
            raise ValueError("A training worker is incomplete")
        if historical and packet != legacy._archived_json(relative):
            raise ValueError("Historical worker accounting differs from its archived record")
        for index, attempt in enumerate(packet["attempts"]):
            rows.append({"seed": seed, "attempt": index, "historical": historical, **attempt})
    return rows


def figures(output, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    definitions = (("v3 repaired", "repaired", None, "#0072B2"),
        ("v2 historical", "legacy-v2", None, "#D55E00"),
        ("untrained v3", "untrained-repaired", None, "#999999"),
        ("LGA dt=1/128", "LGA", "LGA_dt7", "#009E73"),
        ("LGA dt=1/1024", "LGA", "LGA_dt10", "#CC79A7"),
        ("native MC", "native-MC", None, "#222222"))
    summary = tables["model_summary"]
    horizons = sorted({row["horizon"] for row in summary})
    cases = sorted({row["regime"] for row in summary})
    kappas = {row["regime"]: row["kappa"] for row in summary}
    for kind in ("accuracy", "inference"):
        fig, axes = plt.subplots(len(horizons), 2, figsize=(12, 4*len(horizons)), squeeze=False,
                                 constrained_layout=True)
        for i, horizon in enumerate(horizons):
            regimes = {row["regime"] for row in summary if row["horizon"] == horizon}
            for j, statistic in enumerate(("mean", "covariance") if kind == "accuracy" else ("conditional", "ensemble")):
                ax = axes[i, j]
                for label, family, name, color in definitions:
                    if kind == "accuracy":
                        selected = [row for row in summary if row["regime"] in regimes and row["family"] == family
                                    and (name is None or row["name"] == name)]
                        values = [max(1e-8, 100*row["worst_"+statistic]) for row in selected]
                    else:
                        selected = [row for row in tables["costs"] if row["regime"] in regimes and row["family"] == family
                                    and (name is None or row["name"] == name) and row["role"] == statistic
                                    and (statistic != "conditional" or row["batch"] == 1 or family == "native-MC")]
                        values = [1000*row["median_seconds"] for row in selected]
                    ax.scatter([kappas[row["regime"]] for row in selected], values, color=color, alpha=.65, s=23, label=label)
                if kind == "accuracy":
                    ax.axhline(5, linestyle="--", color=".45")
                accuracy_label = ("Mean error (% of reference SD)" if statistic == "mean"
                                  else "Covariance error (% of SD products)")
                ax.set(xlabel="Interaction κ", ylabel=(accuracy_label if kind == "accuracy"
                       else "Warm three-horizon prediction (ms)"), yscale="log", title=f"T={horizon:g}, {statistic}")
        axes[0, 0].legend(fontsize=8)
        fig.savefig(output/("accuracy_regimes.png" if kind == "accuracy" else "inference_regimes.png"), dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for index, case in enumerate(cases):
        selected = [row for row in tables["stopping"] if row["regime"] == case and row["time_index"] == 2]
        ax.scatter(np.full(len(selected), index), [row["probability"] for row in selected], alpha=.65)
    ax.set(xticks=range(len(cases)), xticklabels=cases, ylabel="Native reference stopping probability",
           title="All stopped paths retained; neither moment predictor supplies stopping probabilities")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(output/"stopping_regimes.png", dpi=180)
    plt.close(fig)
    for table, control, filename in (("contrasts", "LGA", "accuracy_advantages.png"),
                                     ("legacy_contrasts", "v2", "legacy_accuracy_advantages.png")):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for i, role in enumerate(("conditional", "ensemble")):
            for j, statistic in enumerate(("mean", "covariance")):
                ax, limits = axes[i, j], [-1., 1.]
                for index, case in enumerate(cases):
                    selected = [row for row in tables[table] if row["regime"] == case and row["role"] == role
                                and row["statistic"] == statistic and row["time_index"] == 2]
                    for offset, row in zip(np.linspace(-.22, .22, len(selected)), selected, strict=True):
                        color = "#0072B2" if row["profile"] == "uniform" else "#D55E00"
                        low, high = 100*row["lower95"], 100*row["upper95"]
                        limits.extend((low, high))
                        ax.vlines(index+offset, low, high, alpha=.4, color=color)
                        ax.scatter(index+offset, 100*row["advantage"], color=color, s=14)
                ax.axhline(0, color=".4", linestyle="--")
                ax.axhspan(-1, 1, alpha=.06, color=".5")
                ax.set_yscale("symlog", linthresh=1.)
                ax.set_ylim(min(limits)*1.12, max(limits)*1.12)
                ax.set(xticks=range(len(cases)), xticklabels=cases,
                       ylabel=f"{control} error − v3 error\n(percentage points of normalized error)",
                       title=f"{role}, final {statistic}: pointwise 95% intervals")
                ax.tick_params(axis="x", rotation=35)
        fig.suptitle("All optimizer seeds; blue: uniform, orange: nonuniform\nSymmetric log outside ±1 percentage point of normalized error", fontsize=12)
        fig.savefig(output/filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for index, case in enumerate(cases):
        for table, shift, label, color in (("training", -.1, "new v3 fits", "#0072B2"),
                                           ("historical_training", .1, "historical v2 fits", "#D55E00")):
            selected = [row for row in tables[table] if row["regime"] == case]
            ax.scatter(np.full(len(selected), index+shift), [row["training_seconds"] for row in selected],
                       color=color, label=label if index == 0 else None)
    ax.set(xticks=range(len(cases)), xticklabels=cases, ylabel="Recorded fit wall seconds / model",
           title="Every optimizer seed; historical fits are separate from this batch's new work")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.savefig(output/"training_costs.png", dpi=180)
    plt.close(fig)
    for role in ("conditional", "ensemble"):
        columns = max(1, len(cases)//len(horizons))
        fig, axes = plt.subplots(len(horizons), columns, figsize=(max(6, 4.5*columns), 4*len(horizons)),
                                 squeeze=False, constrained_layout=True)
        for ax, case in zip(axes.flat, cases, strict=True):
            for label, family, name, color in definitions:
                if family == "untrained-repaired":
                    continue
                selected = [row for row in tables["total_costs"] if row["regime"] == case and row["role"] == role
                            and row["family"] == family and (name is None or row["name"] == name)
                            and (role != "conditional" or row["batch"] == 1 or family == "native-MC")]
                calls = sorted({row["query_calls"] for row in selected})
                ys = [[row["total_known_seconds"] for row in selected if row["query_calls"] == count] for count in calls]
                if ys:
                    ax.plot(calls, [np.median(value) for value in ys], color=color, label=label)
                    ax.fill_between(calls, [min(value) for value in ys], [max(value) for value in ys], color=color, alpha=.12)
            ax.set(xscale="log", yscale="log", xlabel=("New supplied-state queries" if role == "conditional"
                   else "Repeated fresh estimates; fixed IID law"), ylabel="Fit + own data + setup + predictions (s)", title=case)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"{role}: descriptive cost arithmetic; accuracy qualification is separate")
        fig.savefig(output/f"total_cost_{role}.png", dpi=180)
        plt.close(fig)


def report_text(evidence, tables):
    rows, costs = tables["model_summary"], tables["costs"]
    study = evidence["study"]
    nn_wins = sum(row["all_seeds_NN_closer_pointwise95"] for row in tables["seed_consistency"])
    lga_wins = sum(row["all_seeds_LGA_closer_pointwise95"] for row in tables["seed_consistency"])
    repair_wins = sum(row["all_seeds_NN_closer_pointwise95"] for row in tables["legacy_seed_consistency"])
    legacy_wins = sum(row["all_seeds_LGA_closer_pointwise95"] for row in tables["legacy_seed_consistency"])
    strongest = max(evidence["plan"]["cases"], key=lambda case: (
        runner.reference_configuration(case, study).horizon, runner.reference_configuration(case, study).kappa))
    new_cost = sum(row["training_seconds"] for row in tables["training"])
    old_cost = sum(row["training_seconds"] for row in tables["historical_training"])
    qualified = sum(row["accuracy_qualified_payback_calls"] is not None for row in tables["amortization"])
    lines = ["# Boundary-consistent statistics NN: fresh longer/stronger comparison", "",
        f"All **{len(tables['training'])} new v3 fits** and all **{len(tables['historical_training'])} frozen v2 controls** are retained. "
        f"The primary comparison has **{len(evidence['records'])} estimates**; "
        f"**{evidence['original_accuracy_records']+evidence.get('independent_initial_records', 0)} records** are retained including paired-initial diagnostics.", "",
        f"Across {len(tables['seed_consistency'])} group/horizon/statistic comparisons, v3 is closer than fine LGA "
        f"with resolved pointwise intervals for all three seeds in **{nn_wins}** comparisons; LGA is closer in **{lga_wins}**. "
        f"Against matched-seed frozen v2, v3 wins consistently in **{repair_wins}**, and v2 in **{legacy_wins}**. "
        "Other comparisons are unresolved or optimizer-dependent. These counts do not form a family-wise certificate.", "",
        "## Target and the repair", "",
        f"The target remains stopped native finite-volume DK at G={study.n_cells}, N={study.n_particles}, D={study.diffusion:g}, "
        f"a₂={study.a2:g}. Horizons {study.horizons} and interactions {study.kappas} are fixed before fresh outcomes. "
        "Each initial law is IID uniform or exp(0.2 cos x). Conditional predictions use two fresh supplied histograms per law; "
        "ensemble predictions use a separate fresh IID histogram integration pool shared across neural heads. "
        "References, deployment MC, learner families and neural integration pools use disjoint declared streams.", "",
        "The moving gate is a product of erf(d√(T/(w²τ))). It has linear boundary behavior at positive remaining time and "
        "becomes flat at zero time at every strict interior state. Exact native terminal drift and killed-integral initial slopes "
        "replace the fixed-gate slope defect. Every output's learned mean correction starts at τ²; covariance remains a PSD "
        "congruence without a fixed floor. Independent controls verify the zero-time equations, small-time covariance orders, "
        "dense derivatives, absorption and finite corner gradients.", "",
        "Training uses full-horizon scales computed only from training states, with no inverse-time powers. The normalized "
        "residual z uses pseudo-Huber loss 2δ²(√(1+(z/δ)²)−1), δ=3. This retains quadratic behavior near zero and bounded "
        "influence for large residuals. Raw residuals, normalized RMS/MSE and robust losses are all reported, so a smaller "
        "robust criterion cannot hide large physical residuals. Mean selection precedes and remains frozen during covariance fitting.", "",
        "The covariance residual of the initial checkpoint uses the untrained mean's source. The matched covariance-training "
        "comparison is selected_mean_covariance_initial → final_selected: both use the same selected, frozen mean. "
        "An initial → final covariance-loss difference also changes that source and is therefore not an isolated covariance-head improvement. "
        "Absent validation strata and undefined zero-denominator concentration fractions are left empty rather than reported as zero error.", "",
        "The frozen v2 controls were trained on the earlier independent learner pools. Fresh common tests compare the fitted "
        "methods fairly, but prediction differences do not isolate the boundary/loss change from training-pool randomness. "
        "The analytic controls isolate consistency properties; no test outcome selected a seed or triggered refitting.", "",
        "## Independent accuracy", "",
        "Outputs are terminal sine/cosine mode 3 and killed sine/cosine mode 1 integrals divided by the case's fixed T. "
        "Every method returns four means and a same-time 4×4 covariance at T/4, T/2 and T. Mean error is divided by the "
        "reference standard deviation; covariance error by products of reference standard deviations. Covariances are stored N-scaled. "
        "Tables below take the maximum over times, conditional/ensemble groups and observable components; ranges retain optimizer seeds.", ""]
    summary = []
    for case in evidence["plan"]["cases"]:
        for label, family, method in (("v3", "repaired", None), ("v2", "legacy-v2", None),
                ("untrained v3", "untrained-repaired", None), ("LGA dt7", "LGA", "LGA_dt7"),
                ("LGA dt10", "LGA", "LGA_dt10"), ("native MC", "native-MC", None)):
            selected = [row for row in rows if row["regime"] == case and row["family"] == family
                        and (method is None or row["name"] == method)]
            summary.append([case, label, _implementation._range([row["worst_mean"] for row in selected], 100),
                            _implementation._range([row["worst_covariance"] for row in selected], 100)])
    lines += [_implementation._table(["Case", "Method", "Worst mean (% of reference SD)",
                                      "Worst covariance (% of SD products)"], summary), "",
        "![Accuracy by regime](accuracy_regimes.png)", "",
        "Whole-trajectory-family bootstrap intervals are pointwise per fixed fit/horizon. Independent initial-law estimates "
        "use product resampling: reference families and neural initial families are independent, with the same initial resample "
        "shared across both neural predictors in v3/v2 contrasts. The original paired-initial results remain separate diagnostics. "
        "No optimizer seeds are pooled. LGA is unstopped; neither neural head predicts stopping probabilities or a sampled law.", "",
        "### Terminal and time-integral blocks", "",
        "The following split uses each bank label's declared terminal/running weights, independently of output ordering. "
        "TT denotes terminal–terminal covariance, TI terminal–integral covariance, and II integral–integral covariance. "
        "Each number is the worst normalized error over both initial laws, conditional/ensemble groups and the three query times; "
        "ranges retain the optimizer seeds. Every family/seed block is retained in observable_blocks.csv.", ""]
    mean_blocks, covariance_blocks = [], []
    for case in evidence["plan"]["cases"]:
        for label, family, method in (("v3", "repaired", None), ("v2", "legacy-v2", None),
                                     ("LGA dt10", "LGA", "LGA_dt10")):
            chosen = [row for row in tables["observable_blocks"] if row["regime"] == case and row["family"] == family
                      and (method is None or row["name"] == method)]
            values = {block: _implementation._range([row["worst_normalized_error"] for row in chosen
                      if row["observable_block"] == block], 100) for block in ("terminal", "integral", "TT", "TI", "II")}
            mean_blocks.append([case, label, values["terminal"], values["integral"]])
            covariance_blocks.append([case, label, values["TT"], values["TI"], values["II"]])
    integral_details = []
    for label, family, method in (("v3", "repaired", None), ("v2", "legacy-v2", None),
                                  ("fine LGA", "LGA", "LGA_dt10")):
        chosen = [row for row in tables["observable_blocks"] if row["regime"] == strongest
                  and row["family"] == family and (method is None or row["name"] == method)]
        mean_range = _implementation._range([row["worst_normalized_error"] for row in chosen
                                             if row["observable_block"] == "integral"], 100)
        covariance_range = _implementation._range([row["worst_normalized_error"] for row in chosen
                                                   if row["observable_block"] == "II"], 100)
        integral_details.append(f"{label}: integral mean {mean_range}% of reference SD, II covariance {covariance_range}% of SD products")
    lines += [_implementation._table(["Case", "Method", "Terminal mode 3 (% ref SD)",
                                       "Time integral mode 1 (% ref SD)"], mean_blocks), "",
        _implementation._table(["Case", "Method", "TT (% SD products)", "TI (% SD products)",
                                 "II (% SD products)"], covariance_blocks), "",
        f"For {strongest}, the time-integral comparison is "+"; ".join(integral_details)+". "
        "These are worst group/time discrepancies in fluctuation units, rather than relative errors of possibly small means.", "",
        "![v3 against fine LGA](accuracy_advantages.png)", "",
        "![v3 against historical v2](legacy_accuracy_advantages.png)", "",
        "## Stopping and numerical uncertainty", "",
        f"The native reference uses {study.paths_per_group:,} families/group at step {study.reference_step:g}; "
        f"the first {study.refinement_paths:,} families are coupled to two finer levels by summed Brownian increments. "
        "Every first-crossing Euler polygon is frozen, killed integrals cease there, and every stopped path remains in statistics. "
        "Reference refinement and sampling uncertainty are separate. Refinement differences do not prove continuous stopping-boundary bias is absent.", "",
        "![Stopping probabilities](stopping_regimes.png)", "",
        "The 5% normalized-error target requires error plus uncertainty and reference refinement to resolve it; each sampling "
        "or reference contribution is targeted below 1.25%. Small point error alone does not pass these gates. "
        "All gate failures, standalone integration uncertainty and physical payoff bounds are retained in CSV tables.", "",
        "## Training and inference costs", "",
        f"This batch adds **{new_cost:,.1f} summed v3 fitting wall seconds**, plus "
        f"**{sum(row['generation_seconds'] for row in tables['data']):,.2f} shared state-generation seconds**. "
        f"The frozen v2 models retain **{old_cost:,.1f} historical fitting seconds** and "
        f"**{sum(row['generation_seconds'] for row in tables['historical_data']):,.2f} historical state-generation seconds**. "
        "Historical costs are verified against archived manifest bytes and are not new work in this batch. "
        "A deployment cost nevertheless charges each chosen model's own fit and shared data pool once.", ""]
    workers = tables["workers"]
    worker_wall = {}
    for row in workers:
        worker_wall[row["seed"]] = worker_wall.get(row["seed"], 0.)+row.get("wall_seconds", 0.)
    lines += [f"The new worker attempts record **{sum(row.get('cpu_seconds', 0.) for row in workers):,.1f} process CPU seconds** "
        f"and at most **{max(worker_wall.values(), default=0.):,.1f} seconds** for one worker's summed attempts. "
        "Three optimizer workers execute concurrently. Per-model training_seconds and summed worker wall durations are not total elapsed batch time; "
        "UTC start/finish timestamps and monotonic wall/CPU accounting are retained separately.", ""]
    cost_rows = []
    for case in evidence["plan"]["cases"]:
        for label, family, name in (("v3", "repaired", None), ("v2", "legacy-v2", None),
                ("LGA dt7", "LGA", "LGA_dt7"), ("LGA dt10", "LGA", "LGA_dt10"), ("native MC", "native-MC", None)):
            selected = [row for row in costs if row["regime"] == case and row["family"] == family
                        and (name is None or row["name"] == name)]
            conditional = [row for row in selected if row["role"] == "conditional"
                           and (row["batch"] == 1 or family == "native-MC")]
            ensemble = [row for row in selected if row["role"] == "ensemble"]
            cost_rows.append([case, label, _implementation._range([row["median_seconds"] for row in conditional], 1000),
                              _implementation._range([row["median_seconds"] for row in ensemble], 1000)])
    lines += [_implementation._table(["Case", "Method", "Warm conditional (ms)", "Warm IID estimate (ms)"], cost_rows), "",
        "Conditional timings return the full three-time bank for one supplied histogram; batching 64/256 states is retained separately. "
        f"Neural IID predictions include {study.paths_per_group:,} fresh histogram draws and total-covariance integration. "
        "LGA integrates the IID covariance analytically. Native MC draws fresh physical noise and, for the ensemble task, fresh initials, "
        "runs the stopped solver, and computes moments. All final inference timings are measured after fitting, references, accuracy and audits exit.", "",
        "![Warm inference](inference_regimes.png)", "", "![Training](training_costs.png)", "",
        "Cold totals include one model fit, its full shared training/validation pool, strict restoration, required compiled-native setup, "
        "the literal first prediction and subsequent warm predictions. Larger neural batches/ensembles additionally pay the measured cold B1 "
        "cache-creation probe. Arithmetic payback is distinct from accuracy-qualified payback; "
        f"**{qualified}** comparison/statistic rows meet the current finite-reference qualification. "
        "Repeated estimates of an identical fixed IID law may be memoized; their arithmetic curve is not a recommendation to recompute them.", "",
        "![Conditional total costs](total_cost_conditional.png)", "", "![Ensemble total costs](total_cost_ensemble.png)", "",
        "## Evidence and limits", "",
        f"The strongest declared case is {strongest}. Detailed scalar means, variances, covariance entries, all optimizer seeds, "
        "stopping intervals, robust/raw validation strata and matched v3/v2 contrasts are retained in the adjacent CSVs. "
        "Checkpoint, source, configuration, reference, bootstrap and manifest hashes bind every result. "
        "Software acceptance and a reproducible negative outcome remain separate from an accuracy or speed claim. "
        "This experiment concerns the specified stopped finite-grid DK law and fixed observable bank; it is not continuum-DK convergence.", ""]
    return "\n".join(lines)


def generate(root, documentation=None, *, independent_initials=True, validation_diagnostics=True):
    evidence = load_complete(root, independent_initials=independent_initials)
    tables = accuracy_tables(evidence)
    tables["seed_consistency"] = _implementation.seed_consistency(tables["contrasts"], evidence["study"].optimizer_seeds)
    tables["legacy_seed_consistency"] = _implementation.seed_consistency(tables["legacy_contrasts"], evidence["study"].optimizer_seeds)
    if validation_diagnostics:
        tables.update(validation_tables(evidence))
        native = read(_artifact(evidence["root"]/"audit/native_reference_candidate_audit.json", evidence["inputs"]))
        metrics = read(_artifact(evidence["root"]/"audit/metrics_replay.json", evidence["inputs"]))
        if (not native.get("complete") or native["source_hash"] != source_hash()
                or not metrics.get("complete") or metrics["source_hash"] != source_hash()
                or metrics["fit_lock_sha256"] != sha(evidence["root"]/"payoffs/fit_lock.json")):
            raise ValueError("Matching independent native and mathematical audits required")
    if evidence["paired_initial_records"]:
        paired = accuracy_tables({**evidence, "records": evidence["paired_initial_records"],
                                  "groups": evidence["paired_initial_groups"]})
        for key in ("accuracy", "sampling", "contrasts", "legacy_contrasts"):
            tables["paired_initial_"+key] = paired[key]
    tables["model_summary"] = _implementation.model_summary(tables["accuracy"])
    tables["observable_blocks"] = observable_block_tables(evidence, tables)
    tables["training"] = [{"regime": row["case"], "name": row["name"], "seed": row["seed"],
        "training_seconds": row["training_seconds"]} for row in evidence["lock"]["models"]]
    tables["historical_training"] = [{"regime": row["case"], "name": row["name"], "seed": row["seed"],
        "training_seconds": row["historical_training_seconds"], "checkpoint_sha256": row["checkpoint_sha256"]}
        for row in evidence["legacy"]["models"]]
    _artifact(legacy.ARCHIVE/"archive_manifest.json", evidence["inputs"])
    tables["data"], tables["historical_data"] = _data_tables(evidence)
    tables["workers"], tables["historical_workers"] = _worker_tables(evidence), _worker_tables(evidence, True)
    data_seconds = {case: sum(row["generation_seconds"] for row in tables["data"] if row["regime"] == case)
                    for case in evidence["plan"]["cases"]}
    historical_data_seconds = {case: sum(row["generation_seconds"] for row in tables["historical_data"] if row["regime"] == case)
                               for case in evidence["plan"]["cases"]}
    tables["study_costs"] = [{"regime": case,
        "all_new_seed_fit_seconds": sum(row["training_seconds"] for row in tables["training"] if row["regime"] == case),
        "historical_all_seed_fit_seconds": sum(row["training_seconds"] for row in tables["historical_training"] if row["regime"] == case),
        "new_shared_state_generation_seconds": data_seconds[case],
        "historical_shared_state_generation_seconds": historical_data_seconds[case],
        "reference_generation_seconds": evidence["references"][case]["generation_seconds"],
        "scope": "component recorded durations; historical work excluded from incremental new batch"}
        for case in evidence["plan"]["cases"]]
    tables["costs"], tables["total_costs"] = cost_tables(evidence["timing"], evidence["lock"], data_seconds,
        evidence["legacy"]["models"], historical_data_seconds)
    tables["amortization"] = amortization_tables(tables)
    output = evidence["root"]/"report"
    output.mkdir(exist_ok=True)
    for name, rows in tables.items():
        csv_write(output/(name+".csv"), rows)
    figures(output, tables)
    text = report_text(evidence, tables)
    (output/"report.md").write_text(text)
    if documentation:
        documentation = Path(documentation)
        documentation.parent.mkdir(parents=True, exist_ok=True)
        prefix = f"artifacts/{documentation.stem}/report/"
        for path in output.glob("*.png"):
            text = text.replace(f"]({path.name})", f"]({prefix}{path.name})")
        documentation.write_text(text+f"\n[Compact evidence archive](artifacts/{documentation.stem}/README.md).\n")
    manifest = {"kind": KIND, "complete": True, "source_hash": source_hash(), "generator_sha256": sha(__file__),
        "inputs": evidence["inputs"], "outputs": {path.name: sha(path) for path in output.iterdir()
            if path.is_file() and path.name != "manifest.json"},
        "models": len(evidence["lock"]["models"]), "historical_control_models": len(evidence["legacy"]["models"]),
        "accuracy_records": len(evidence["records"]), "original_accuracy_records": evidence["original_accuracy_records"],
        "independent_initial_records": evidence.get("independent_initial_records", 0),
        "total_retained_accuracy_records": evidence["original_accuracy_records"]+evidence.get("independent_initial_records", 0),
        "timing_records": len(evidence["timing"]["records"]), "timing_restorations": len(evidence["timing"]["loading"]),
        "total_cost_rows": len(tables["total_costs"]), "all_optimizer_seeds_retained": True,
        "primary_NN_ensemble_initials_independent": independent_initials,
        "postfit_validation_diagnostics_included": validation_diagnostics,
        "test_based_refitting": False, "continuous_DK_claim": False,
        "qualified_payback_rows": sum(row["accuracy_qualified_payback_calls"] is not None for row in tables["amortization"]),
        "new_fit_seconds": sum(row["training_seconds"] for row in tables["training"]),
        "historical_fit_seconds": sum(row["training_seconds"] for row in tables["historical_training"]),
        "excluded_post_report_audits": ["audit/report_claims_cost_replay.json"]}
    json_write(output/"manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("report", "archive", "verify-archive"))
    parser.add_argument("--run", type=Path, default=REPO/"runs/payoff-boundary-regimes-2026-10-27")
    parser.add_argument("--documentation", type=Path)
    parser.add_argument("--archive", type=Path, default=REPO/"docs/experiments/artifacts/payoff_boundary_regimes_2026-10-27")
    args = parser.parse_args(argv)
    result = (generate(args.run, args.documentation) if args.action == "report" else
              _implementation.archive(args.run, args.archive) if args.action == "archive" else
              _implementation.verify_archive(args.archive))
    print(json.dumps({key: value for key, value in result.items() if key not in ("inputs", "outputs")}, indent=2))


if __name__ == "__main__":
    main()


def __getattr__(name):
    return getattr(_implementation, name)
