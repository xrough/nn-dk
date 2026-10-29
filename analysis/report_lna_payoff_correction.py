"""All-seed supervised-LNA evidence, measured costs and offline compact archive.

One cold and one warm observation do not estimate a latency distribution.
Signed whole-family contrasts are pointwise, and passing finite-reference
diagnostics would not certify continuous stopping, a particle law or an SPDE.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

from analysis import run_lna_payoff_correction_study as runner  # noqa: E402
from analysis import run_payoff_collocation_study as collocation  # noqa: E402
from analysis.payoff_study_adapter import bind_study, isolated_module  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402

KIND = "all-seed-supervised-nonlinear-lna-correction-report-v1"
TEST_GATE_KIND = "lna-correction-current-source-test-gate-v1"
TEST_COMMANDS = {"ruff": [".venv/bin/ruff", "check", "."],
                 "pytest": [".venv/bin/python", "-m", "pytest", "-q"]}
PRODUCER_WORK_STAGES = ("prepare", "data", "fit")
EVALUATOR_WORK_STAGES = ("prepare", "freeze", "reference", "evaluate", "contrasts", "benchmark")
CONTROLS = {"gated_zero", "raw_conditional_LNA_dt7", "raw_conditional_LNA_dt10"}
_base = isolated_module("report_payoff_regimes.py", "analysis._lna_correction_report_impl")
bind_study(_base, __file__, runner=collocation)


def _evaluator():
    from analysis import evaluate_lna_payoff_correction

    return evaluate_lna_payoff_correction


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Unsupported report value: {type(item).__name__}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=encode)+"\n")
    temporary.replace(path)


def artifact(path, inputs, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Frozen report input checksum changed")
    inputs[str(path)] = digest
    return path


def _finite(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def collect_test_inputs():
    """Current maintained code/test/configuration map; no run/model reads."""
    files = set()
    for directory, suffixes in (("analysis", {".py", ".cpp", ".h", ".hpp"}), ("src", {".py"}),
            ("tests", {".py"}), ("configs", {".toml", ".json", ".yaml", ".yml"}),
            (".github/workflows", {".yaml", ".yml"})):
        files.update(p for p in (REPO/directory).rglob("*") if p.is_file() and p.suffix in suffixes)
    for name in ("pyproject.toml", "uv.lock", "LICENSE", "LICENSE-DOCS", "LICENSING.md"):
        path = REPO/name
        if not path.is_file():
            raise ValueError("Maintained build and license files must exist before the final test gate")
        files.add(path)
    return {str(path.relative_to(REPO)): sha(path) for path in sorted(files)}


def verify_final_tests(root, *, helpers=None):
    """Require full successful Ruff/pytest logs against exactly current bytes.

    Root writes this receipt only after comparing source maps before and after
    both commands. The receipt/logs are archival inputs, not a new experiment.
    """
    root = Path(root).resolve()
    path = root/"audit/final_tests.json"
    value = read(path)
    current = collect_test_inputs()
    digest = stable_hash(current)
    if (value.get("kind") != TEST_GATE_KIND or value.get("schema_version") != 1
            or value.get("complete") is not True or value.get("source_hash") != source_hash()
            or value.get("source_inputs") != current or value.get("before_sha256") != digest
            or value.get("after_sha256") != digest):
        raise ValueError("Final tests must bind the exact current analysis/package/tests/config source, not package hash alone")
    for name, expected in (helpers or {}).items():
        helper = Path(name).resolve()
        if not helper.is_relative_to(REPO) or current.get(str(helper.relative_to(REPO))) != expected:
            raise ValueError("Final tests do not cover every frozen joint helper")
    commands = value.get("commands", [])
    if len(commands) != 2 or {c.get("kind") for c in commands} != set(TEST_COMMANDS):
        raise ValueError("Both complete project Ruff and pytest commands are required")
    logs = []
    for command in commands:
        log = Path(command.get("log_path", "")).resolve()
        if (command.get("argv") != TEST_COMMANDS[command["kind"]] or command.get("cwd") != str(REPO.resolve())
                or type(command.get("exit_code")) is not int or command["exit_code"] != 0
                or not log.is_relative_to(root/"audit") or not log.is_file()
                or command.get("log_sha256") != sha(log)):
            raise ValueError("Full successful command allocation and exact immutable test logs required")
        if command["kind"] == "pytest":
            counts = re.findall(r"\b(\d+) passed\b", log.read_text())
            if (not counts or type(command.get("test_count")) is not int
                    or command["test_count"] < 1 or command["test_count"] != int(counts[-1])):
                raise ValueError("Pytest count must match the successful full-suite log")
        logs.append(log)
    if len(set(logs)) != len(commands):
        raise ValueError("Ruff and pytest require separate immutable logs")
    return value, logs


def _queries(packet, count=3):
    rows = packet.get("records", [])
    if len(rows) != count or {r["time_index"] for r in rows} != set(range(count)):
        raise ValueError("Every physical query must appear exactly once")
    return rows


def validate_evidence(evidence):
    study, old = evidence["study"], evidence["collocation_study"]
    cases = tuple(runner.cases(study))
    if tuple(evidence["cases"]) != cases or tuple(study.optimizer_seeds) != (0, 1, 2):
        raise ValueError("Every declared regime and all three seeds are required")
    expected_models = {(m["case"], m["seed"], m["name"]) for m in runner.slots(study)}
    models = evidence["candidate_lock"]["models"]
    actual_models = [(m["case"], m["seed"], m["name"]) for m in models]
    if len(actual_models) != len(expected_models) or set(actual_models) != expected_models:
        raise ValueError("Complete candidate model inventory required; no best-seed selection")
    expected_groups = {(case, group) for case in cases for group in collocation.reference_configuration(case, old).group_keys}
    groups = [(g["case"], g["group"]) for g in evidence["groups"]]
    if len(groups) != len(expected_groups) or set(groups) != expected_groups:
        raise ValueError("Every conditional and IID comparison group must remain")
    method_names = {case: {m["name"] for m in models if m["case"] == case}|CONTROLS for case in cases}
    method_specs = {case: {m["name"]: m for m in _evaluator().methods(evidence, case)} for case in cases}
    expected = {(case, group, name) for case, group in expected_groups for name in method_names[case]}
    actual = []
    fields = ("max_mean_error_in_reference_std", "max_normalized_covariance_error",
        "mean_sampling95_simultaneous_radius", "covariance_sampling95_simultaneous_radius",
        "mean_error_plus_sampling95", "covariance_error_plus_sampling95")
    for record in evidence["records"]:
        identity, method = record["identity"], record["method"]
        case, group, name = identity["case"], identity["group"], method["name"]
        actual.append((case, group, name))
        if (not record.get("complete") or record.get("all_families_retained") is not True
                or identity.get("joint_lock_sha256") != evidence["joint_lock_sha256"]
                or identity.get("method") != method or method != method_specs.get(case, {}).get(name)):
            raise ValueError("Complete source-bound all-family method evidence required")
        if name not in CONTROLS:
            slot = next((m for m in models if (m["case"], m["name"]) == (case, name)), None)
            if slot is None or method["seed"] != slot["seed"] or method["kind"] != "trained" or method["family"] != "LNA-correction":
                raise ValueError("A fitted seed identity changed")
        if "__ensemble__" in group and record.get("initial_pool_independent_of_reference") is not True:
            raise ValueError("Primary IID integration must be independent of reference")
        if record["available"]:
            if record["failed_family_ids"] or any(not _finite(row[k]) for row in _queries(record["metrics"]) for k in fields):
                raise ValueError("Available predictions require complete finite metrics and no failed families")
        elif record["metrics"] is not None or not record["failed_family_ids"]:
            raise ValueError("Unavailable predictions must retain explicit failures, never survivor averages")
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("All candidate/control accuracy records required")
    baseline = evidence["baseline"]
    old_groups = [(g["regime"], g["group"]) for g in baseline["groups"]]
    if len(old_groups) != len(expected_groups) or set(old_groups) != expected_groups:
        raise ValueError("Every comparative reference/refinement group must remain")
    old_models = baseline["lock"]["models"]
    if {(m["case"], m["seed"], m["proposal"], m["loss"]) for m in old_models} != {
            (s["case"], s["seed"], s["proposal"], s["loss"]) for s in collocation.slots(old)} or len(old_models) != len(collocation.slots(old)):
        raise ValueError("All collocation comparative models must remain")
    expected_old = {(case, group, name) for case, group in expected_groups for name in
        ({m["name"] for m in old_models if m["case"] == case}|{case+"_untrained", "LGA_dt7", "LGA_dt10", "native-MC-dt10"})}
    actual_old = [(r["identity"]["regime"], r["identity"]["group"], r["method"]["name"]) for r in baseline["records"]]
    if len(actual_old) != len(expected_old) or set(actual_old) != expected_old:
        raise ValueError("Complete collocation primary comparative context required")
    for group in baseline["groups"]:
        cfg = collocation.reference_configuration(group["regime"], old)
        if set(group["reference_refinement"]) != {f"level{i}_versus_finest" for i in range(len(cfg.reference_factors)-1)}:
            raise ValueError("Every coupled reference-refinement level required")
        for packet in group["reference_refinement"].values():
            if any(not _finite(row[k]) for row in _queries(packet) for k in fields):
                raise ValueError("Finite signed-reference comparison metrics required")
    contrast_rows = contrast_table(evidence)
    timing_rows = timing_table(evidence)
    return {"cases": len(cases), "candidate_models": len(models), "new_primary_records": len(actual),
        "collocation_primary_records": len(actual_old), "unavailable_new_records": sum(not r["available"] for r in evidence["records"]),
        "contrast_packets": len(evidence["contrasts"]["records"]), "contrast_statistic_rows": len(contrast_rows),
        "timing_jobs": len(timing_rows)}


def contrast_table(evidence):
    expected = set()
    lookup = {(r["identity"]["case"], r["identity"]["group"], r["method"]["name"]): r for r in evidence["records"]}
    old = evidence["baseline"]
    control_lookup = {(r["identity"]["regime"], r["identity"]["group"], r["method"]["name"]): r for r in old["records"]}
    control_lookup.update(lookup)
    for case in evidence["cases"]:
        for group in collocation.reference_configuration(case, evidence["collocation_study"]).group_keys:
            for model in (m for m in evidence["candidate_lock"]["models"] if m["case"] == case):
                controls = CONTROLS|{"LGA_dt7", "LGA_dt10", "native-MC-dt10"}|{
                    m["name"] for m in old["lock"]["models"] if m["case"] == case and m["seed"] == model["seed"]}
                expected.update((case, group, model["name"], control) for control in controls)
    actual, result = [], []
    for packet in evidence["contrasts"]["records"]:
        key = packet["case"], packet["group"], packet["candidate"], packet["control"]
        actual.append(key)
        model = next((m for m in evidence["candidate_lock"]["models"] if (m["case"], m["name"]) == (key[0], key[2])), None)
        first, second = lookup.get(key[:3]), control_lookup.get((key[0], key[1], key[3]))
        if (model is None or packet["seed"] != model["seed"] or first is None or second is None
                or packet["available"] != (first["available"] and second.get("available", True))):
            raise ValueError("Matched-seed contrast availability and endpoint identities must agree")
        if packet.get("pointwise_not_simultaneous") is not True or packet.get("same_initials_assumed") is not False:
            raise ValueError("Contrasts must retain pointwise and independent-initial qualifications")
        if not packet["available"]:
            if packet["contrast"] is not None:
                raise ValueError("Failed predictions cannot have successful contrast intervals")
            result.append({k: packet[k] for k in ("case", "group", "candidate", "seed", "control", "available")})
            continue
        if not lookup[key[:3]]["available"] or packet["contrast"].get("both_predictor_initial_pool_uncertainties_included") is not True:
            raise ValueError("Both initial-pool uncertainties must remain in an available contrast")
        for item in _queries(packet["contrast"]):
            for statistic in ("mean", "covariance"):
                point = item[statistic+"_error_advantage"]
                low, high = item[statistic+"_advantage_percentile95"]
                samples = item["bootstrap_"+statistic+"_advantages"]
                if (not np.isfinite([point, low, high]).all() or low > high or np.shape(samples) != (128,)
                        or not np.isfinite(samples).all() or not np.allclose([low, high], np.quantile(samples, [.025, .975]), rtol=1e-10, atol=1e-12)):
                    raise ValueError("Finite signed whole-family percentile contrasts required")
                result.append({**{k: packet[k] for k in ("case", "group", "candidate", "seed", "control", "available")},
                    "time_index": item["time_index"], "statistic": statistic, "advantage": point,
                    "lower95": low, "upper95": high, "candidate_closer_pointwise95": low > 0,
                    "control_closer_pointwise95": high < 0, "scope": "signed pointwise whole-family contrast; no simultaneous proof"})
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Every candidate/control/matched-seed contrast required exactly once")
    return result


def timing_table(evidence):
    index = evidence["timing"]
    if (not index.get("complete") or index.get("external_competing_heavy_work_finished_attested") is not True
            or index["environment"].get("cpu_threads") != 1):
        raise ValueError("Complete isolated one-thread timings required")
    expected = {job["slot"]: job for job in _evaluator()._timing_slots(evidence)}
    actual, rows = [], []
    for entry in index["jobs"]:
        value = read(artifact(entry["path"], evidence["inputs"], entry["sha256"]))
        job = value["identity"]["job"]
        actual.append(job["slot"])
        parts = value["components"]
        attempts = value["restoration_and_resume_attempts"]
        if (job != expected.get(job["slot"]) or value["identity"]["joint_lock_sha256"] != evidence["joint_lock_sha256"]
                or set(parts) != {"conditional_LNA_seconds", "head_seconds", "initialization_seconds", "aggregation_seconds"}
                or value.get("timing_mode") != "uninterrupted-full-uncached-workload-v1"
                or not all(_finite(v) for v in parts.values()) or not _finite(value.get("workload_seconds"))
                or not _finite(value.get("component_sum_seconds"))
                or not np.isclose(sum(parts.values()), value["component_sum_seconds"], rtol=1e-12)
                or value["component_sum_seconds"] > value["workload_seconds"]+1e-9
                or not _finite(value.get("paid_attempt_seconds")) or value["paid_attempt_seconds"] < value["workload_seconds"]-1e-9
                or type(value.get("attempt_cost_complete")) is not bool
                or (job["phase"] == "cold" and (not _finite(value.get("cold_end_to_end_seconds"))
                    or value["cold_end_to_end_seconds"] < value["workload_seconds"]-1e-9))
                or value.get("full_uncached_LNA_per_histogram") is not True or value.get("accuracy_cache_used") is not False
                or value.get("all_families_retained") is not True or not attempts
                or any(not _finite(a.get(k)) for a in attempts for k in ("restoration_seconds", "warm_preparation_seconds"))):
            raise ValueError("Full uncached ODE/head/initialization/aggregation and restoration observations required")
        rows.append({"case": job["case"], "profile": job["profile"], "name": job["method"]["name"],
            "family": job["method"]["family"], "seed": job["method"]["seed"], "role": job["role"], "phase": job["phase"],
            "families": job["families"], "available": value["available"], "workload_seconds": value["workload_seconds"], **parts,
            "component_sum_seconds": value["component_sum_seconds"], "paid_attempt_seconds": value["paid_attempt_seconds"],
            "attempt_cost_complete": value["attempt_cost_complete"], "cold_end_to_end_seconds": value.get("cold_end_to_end_seconds"),
            "restoration_seconds": sum(a["restoration_seconds"] for a in attempts),
            "resume_preparation_seconds": sum(a["warm_preparation_seconds"] for a in attempts),
            "scope": "one uninterrupted full initialization/ODE/head/aggregation workload observation; nested timers not additive; no stable latency distribution"})
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Every literal cold/warm B1 and full-IID timing workload required")
    return rows


def _cost_log(path, evidence, identity):
    value = read(artifact(path, evidence["inputs"]))
    payload = {k: v for k, v in value.items() if k != "content_sha256"}
    if value.get("identity") != identity or value.get("content_sha256") != stable_hash(payload) or not value["attempts"]:
        raise ValueError("Complete checksum-bound preparation cost history required")
    if any(not _finite(a.get(k)) for a in value["attempts"] for k in ("wall_seconds", "cpu_seconds")):
        raise ValueError("Interrupted attempts need measured durations, not invented zero costs")
    return value["attempts"]


def _label_replay_audit(evidence):
    from analysis import audit_lna_supervised_labels as auditor

    auditor.allocation(evidence["study"])
    receipt = auditor.verify(evidence["root"], require_complete=True)
    artifact(evidence["root"]/auditor.RECEIPT, evidence["inputs"])
    helpers = {str((REPO/name).resolve()): digest for name, digest in receipt["identity"]["helpers"].items()}
    helpers[str(Path(auditor.__file__).resolve())] = receipt["identity"]["script_sha256"]
    for path, digest in helpers.items():
        artifact(path, evidence["inputs"], digest)
    for row in receipt["records"]:
        for path, digest in row["input_files"].items():
            artifact(path, evidence["inputs"], digest)
    if not receipt["attempts"] or any(not _finite(a.get(k)) for a in receipt["attempts"] for k in ("wall_seconds", "cpu_seconds")):
        raise ValueError("Complete independent label-replay audit attempt costs required")
    prior = receipt.get("prior_audit_costs", [])
    for item in prior:
        if any(not _finite(item.get(k)) for k in ("wall_seconds", "cpu_seconds")):
            raise ValueError("Preserved earlier replay-audit costs must remain explicit and finite")
        for prefix in ("manifest", "receipt", "auditor_source"):
            artifact(item[prefix+"_path"], evidence["inputs"], item[prefix+"_sha256"])
    current_wall = sum(a["wall_seconds"] for a in receipt["attempts"])
    prior_wall = sum(item["wall_seconds"] for item in prior)
    current_cpu = sum(a["cpu_seconds"] for a in receipt["attempts"])
    prior_cpu = sum(item["cpu_seconds"] for item in prior)
    return {"required": True, "complete": True, "verified_families": len(receipt["records"]),
        "wall_seconds": current_wall+prior_wall, "current_audit_wall_seconds": current_wall,
        "prior_audit_wall_seconds": prior_wall, "prior_audit_costs": prior,
        "cpu_seconds": current_cpu+prior_cpu, "current_audit_cpu_seconds": current_cpu,
        "prior_audit_cpu_seconds": prior_cpu,
        "scope": "independent prefix replay of saved learner families; separate audit overhead, not added to producer preparation"}, helpers


def preparation_costs(evidence):
    root, study, inputs = evidence["root"], evidence["study"], evidence["inputs"]
    preparations = _cost_log(root/"execution/prepare_costs.json", evidence, {"plan_hash": study.plan_hash, "stage": "anchors"})
    data_attempts = _cost_log(root/"execution/data_costs.json", evidence, {"plan_hash": study.plan_hash, "stage": "labels-and-baselines"})
    if not any(a.get("complete") for a in preparations) or not any(a.get("complete") for a in data_attempts):
        raise ValueError("Full anchor and learner-data preparation must finish")
    components, families, learner_baselines = [], 0, 0.
    for case in evidence["cases"]:
        for role in ("train", "validation"):
            directory = root/"data"/case/role
            for relative in ("anchors/record.json", "anchors/arrays.npz", "bank/record.json", "bank/arrays.npz"):
                artifact(directory/relative, inputs)
            label = read(artifact(directory/"label_manifest.json", inputs))
            expected_anchors = 2*(study.train_natural_per_profile+study.train_stress_per_profile) if role == "train" else 2*study.validation_per_profile
            continuations = study.train_continuations if role == "train" else study.validation_continuations
            if (not label.get("complete") or label.get("anchors") != expected_anchors or label.get("paths_per_anchor") != continuations
                    or label.get("all_stopped_paths_retained") is not True or label.get("unbiased_sample_covariance") is not True
                    or sum(c["families"] for c in label["chunks"]) != expected_anchors*continuations):
                raise ValueError("Complete declared supervised label families and unbiased moments required")
            for chunk in label["chunks"]:
                record = read(artifact(Path(chunk["path"])/"record.json", inputs, chunk["record_sha256"]))
                artifact(Path(chunk["path"])/"arrays.npz", inputs, chunk["arrays_sha256"])
                seconds = record["costs"]["wall_seconds"]
                if not _finite(seconds) or not 0 <= chunk["stopped_families"] <= chunk["families"]:
                    raise ValueError("Every retained label chunk needs measured cost and stopping counts")
                families += chunk["families"]
                components.append({"case": case, "role": role, "component": "native-MC-labels", "families": chunk["families"],
                    "stopped_families": chunk["stopped_families"], "seconds": seconds})
            baseline = read(artifact(directory/"baseline_manifest.json", inputs))
            cache = baseline["cache"]
            record = read(artifact(Path(cache["path"])/"record.json", inputs, cache["record_sha256"]))
            artifact(Path(cache["path"])/"arrays.npz", inputs, cache["arrays_sha256"])
            seconds = record["costs"]["wall_seconds"]
            if not _finite(seconds) or baseline.get("analytic_IID_replacement") is not False or baseline.get("conditional_initial_covariance") != "zero":
                raise ValueError("Measured per-anchor conditional LNA preparation required")
            learner_baselines += seconds
            components.append({"case": case, "role": role, "component": "learner-conditional-LNA-cache", "seconds": seconds, "families": expected_anchors})
    worker_seconds, pools, worker_by_seed, worker_pids = 0., {}, {}, set()
    seed_pids = {seed: set() for seed in study.optimizer_seeds}
    for seed in study.optimizer_seeds:
        worker = read(artifact(root/"payoffs/execution"/f"worker_seed{seed}.json", inputs))
        if not worker.get("complete") or worker.get("seed") != seed or not worker.get("attempts"):
            raise ValueError("Every fixed-seed worker must complete")
        for attempt in worker["attempts"]:
            if not _finite(attempt.get("wall_seconds")) or not _finite(attempt.get("cpu_seconds")) or "pool_preparation" not in attempt:
                raise ValueError("Complete worker attempt and cached-bank preparation costs required")
            worker_seconds += attempt["wall_seconds"]
            worker_by_seed[seed] = worker_by_seed.get(seed, 0.)+attempt["wall_seconds"]
            worker_pids.add(attempt["pid"])
            seed_pids[seed].add(attempt["pid"])
            for case, value in attempt["pool_preparation"].items():
                if case not in evidence["cases"] or not _finite(value["seconds"]):
                    raise ValueError("Valid case-specific bank conversion costs required")
                pools[(case, seed)] = pools.get((case, seed), 0.)+value["seconds"]
    if set(pools) != {(case, seed) for case in evidence["cases"] for seed in study.optimizer_seeds}:
        raise ValueError("Every seed/case pool-preparation entry must remain")
    fit_total, fit_by_seed = 0., {}
    for model in evidence["candidate_lock"]["models"]:
        artifact(model["metadata_path"], inputs, model["metadata_sha256"])
        artifact(model["checkpoint_path"], inputs, model["checkpoint_sha256"])
        if not model.get("full_invocation_cost_complete") or not _finite(model["full_invocation_seconds"]):
            raise ValueError("All complete fit invocation costs required")
        artifact(model["invocation_record_path"], inputs, model["invocation_record_sha256"])
        fit_total += model["full_invocation_seconds"]
        fit_by_seed[model["seed"]] = fit_by_seed.get(model["seed"], 0.)+model["full_invocation_seconds"]
    prep = sum(a["wall_seconds"] for a in preparations)+sum(a["wall_seconds"] for a in data_attempts)
    nested = sum(row["seconds"] for row in components)
    if nested > sum(a["wall_seconds"] for a in data_attempts)+1e-5 or fit_total+sum(pools.values()) > worker_seconds+1e-5:
        raise ValueError("Nested component times cannot exceed complete invocation totals")
    accuracy_cache_seconds = 0.
    expected_caches = set()
    for case in evidence["cases"]:
        cfg = collocation.reference_configuration(case, evidence["collocation_study"])
        for group in cfg.group_keys:
            count = cfg.paths_per_group if "__ensemble__" in group else 1
            for start in range(0, count, evidence["plan"].chunk_paths):
                expected_caches.update(root/"evaluation/baselines"/case/group/f"dt{level}_{start}.json" for level in (7, 10))
    actual_caches = set((root/"evaluation/baselines").rglob("*.json"))
    if actual_caches != expected_caches:
        raise ValueError("Every declared shared accuracy-only baseline cache cost required")
    for path in sorted(actual_caches):
        cache = read(artifact(path, inputs))
        if cache.get("identity", {}).get("joint_lock_sha256") != evidence["joint_lock_sha256"] or not _finite(cache.get("baseline_ODE_seconds")):
            raise ValueError("Accuracy-only baseline cache source/cost changed")
        artifact(cache["arrays"]["path"], inputs, cache["arrays"]["sha256"])
        accuracy_cache_seconds += cache["baseline_ODE_seconds"]
    processes = []
    for path in sorted((root/"execution").glob("cli_*.json")):
        if not any(path.name.startswith("cli_"+stage+"_") for stage in PRODUCER_WORK_STAGES):
            continue
        process = read(artifact(path, inputs))
        if (process.get("stage") not in PRODUCER_WORK_STAGES or not path.name.startswith("cli_"+process["stage"]+"_")
                or process.get("source_hash") != source_hash() or any(not _finite(process.get(k)) for k in ("wall_seconds", "process_cpu_seconds"))):
            raise ValueError("Valid complete-process cost receipts required")
        processes.append(process)
    evaluation_processes = []
    evaluator = _evaluator()
    for process in evaluator.invocation_receipts(root, include_verification=False):
        if (process["stage"] not in EVALUATOR_WORK_STAGES or process.get("nonadditive_contains_nested_work") is not True
                or type(process.get("whole_process")) is not bool):
            raise ValueError("Evaluator work receipts must preserve their measured whole-process scope")
        artifact(process["receipt_path"], inputs, process["receipt_sha256"])
        evaluation_processes.append(process)
    coverage = {stage: any(p["stage"] == stage and p["whole_process"] and p["status"] == "success"
                          for p in evaluation_processes) for stage in EVALUATOR_WORK_STAGES}
    seen = {(p["stage"], p["pid"]) for p in processes}
    process_complete = (all(("prepare", a["pid"]) in seen for a in preparations)
        and all(("data", a["pid"]) in seen for a in data_attempts)
        and all(("fit", pid) in seen for pid in worker_pids))
    shared_overhead = {str(seed): worker_by_seed[seed]-fit_by_seed[seed] for seed in study.optimizer_seeds}
    standalone_build = prep
    if process_complete:
        standalone_build = sum(p["wall_seconds"] for p in processes if p["stage"] in ("prepare", "data"))
        shared_overhead = {str(seed): sum(p["wall_seconds"] for p in processes if p["stage"] == "fit"
            and p["pid"] in seed_pids[seed])-fit_by_seed[seed] for seed in study.optimizer_seeds}
        if standalone_build < prep-1e-5:
            raise ValueError("Complete process times cannot be shorter than nested preparation")
    if any(value < -1e-5 for value in shared_overhead.values()):
        raise ValueError("Fit invocations cannot exceed their parent worker durations")
    summary = {"native_MC_label_families": families, "full_shared_input_preparation_seconds": prep,
        "native_MC_label_component_seconds": sum(r["seconds"] for r in components if r["component"] == "native-MC-labels"),
        "learner_LNA_cache_component_seconds": learner_baselines, "all_fit_invocation_seconds": fit_total,
        "all_worker_invocation_seconds": worker_seconds, "full_prepare_plus_worker_seconds": prep+worker_seconds,
        "accuracy_only_shared_LNA_cache_seconds": accuracy_cache_seconds,
        "seed_shared_worker_overhead_seconds": shared_overhead,
        "standalone_full_input_build_charge_seconds": standalone_build,
        "complete_process_cost_receipts": process_complete, "process_receipts": processes,
        "all_recorded_CLI_process_seconds_nonadditive": sum(p["wall_seconds"] for p in processes),
        "evaluation_process_receipts": evaluation_processes, "evaluation_CLI_stage_coverage": coverage,
        "evaluation_complete_process_cost_receipts": all(coverage.values()),
        "evaluation_work_process_seconds_nonadditive": sum(p["wall_seconds"] for p in evaluation_processes),
        "evaluation_work_process_cpu_seconds_nonadditive": sum(p["cpu_seconds"] for p in evaluation_processes),
        "posthoc_verification_cost_scope": "producer verify-lock and evaluator verify remain separate local receipts; excluded before reading/frozen input identity",
        "scope": "summed invocations, not elapsed concurrent wall time; label/cache/fit/pool components are nested, never added again"}
    default = runner.StudyPlan().to_dict()
    summary["full_scientific_allocation"] = all(study.to_dict()[key] == default[key] for key in (
        "horizons", "kappas", "n_cells", "n_particles", "diffusion", "a2", "rho_stop", "train_natural_per_profile",
        "train_stress_per_profile", "validation_per_profile", "train_continuations", "validation_continuations", "label_step", "baseline_step"))
    summary["native_label_replay_audit"], evidence["additional_helpers"] = _label_replay_audit(evidence)
    return summary, components, pools


def accuracy_tables(evidence):
    adapted = []
    for record in evidence["records"]:
        if record["available"]:
            identity = record["identity"]
            adapted.append({**record, "identity": {**identity, "regime": identity["case"], "name": record["method"]["name"]}})
    base = evidence["baseline"]
    tables = _base.accuracy_tables({**base, "records": [*base["records"], *adapted]})
    for record in evidence["records"]:
        if not record["available"]:
            identity, method = record["identity"], record["method"]
            for index, t in enumerate(runner.query_times(identity["case"], evidence["study"])):
                profile, role, anchor = identity["group"].split("__")
                tables["accuracy"].append({"regime": identity["case"], "group": identity["group"],
                    "profile": profile, "role": role, "anchor": int(anchor), "name": method["name"],
                    "family": method["family"], "seed": method["seed"], "time_index": index, "time": t,
                    "available": False, "failed_family_ids": json.dumps(record["failed_family_ids"])})
    for row in tables["accuracy"]:
        row.setdefault("available", True)
    tables["signed_contrasts"] = contrast_table(evidence)
    tables["failures"] = [{"case": r["identity"]["case"], "group": r["identity"]["group"], "name": r["method"]["name"],
        "available": r["available"], "failed_family_ids": json.dumps(r["failed_family_ids"])} for r in evidence["records"]]
    return tables


def deployment_costs(evidence, timings, preparation, pools):
    models = {(m["case"], m["name"]): m for m in evidence["candidate_lock"]["models"]}
    by_phase = {(r["case"], r["profile"], r["name"], r["role"], r["phase"]): r for r in timings}
    costs, totals = [], []
    for key, cold in by_phase.items():
        if key[-1] != "cold":
            continue
        warm = by_phase[(*key[:-1], "warm")]
        model = models.get((cold["case"], cold["name"]))
        offline = 0. if model is None else preparation["standalone_full_input_build_charge_seconds"]+model["full_invocation_seconds"]+preparation["seed_shared_worker_overhead_seconds"][str(model["seed"])]
        overhead = cold["paid_attempt_seconds"]+warm["paid_attempt_seconds"]-cold["workload_seconds"]-warm["workload_seconds"]
        row = {**{k: cold[k] for k in ("case", "profile", "name", "family", "seed", "role", "families")},
            "available": cold["available"] and warm["available"], "known_offline_seconds": offline+overhead,
            "cold_observed_seconds": cold["workload_seconds"], "warm_observed_seconds": warm["workload_seconds"],
            "cold_successful_end_to_end_seconds": cold["cold_end_to_end_seconds"],
            "known_paid_attempt_overhead_seconds": overhead,
            "timing_attempt_cost_complete": cold["attempt_cost_complete"] and warm["attempt_cost_complete"],
            "complete_process_cost_receipts": preparation["complete_process_cost_receipts"],
            "evaluation_complete_process_cost_receipts": preparation["evaluation_complete_process_cost_receipts"],
            "native_label_replay_verified": preparation["native_label_replay_audit"]["complete"],
            "full_scientific_allocation": preparation["full_scientific_allocation"],
            "seed_case_pool_loading_component_seconds": 0. if model is None else pools[(model["case"], model["seed"])],
            "data_cost_scope": "standalone chosen model conservatively pays full multi-case input build and whole-seed shared worker overhead once; pool loading is nested, never added again" if model else "no supervised input or fit charge",
            "observation_scope": "one cold and one warm uninterrupted full-workload observation; no latency distribution"}
        costs.append(row)
        for count in (1, 100, 10000, 100000):
            totals.append({**row, "query_calls": count, "total_known_seconds": row["known_offline_seconds"]+row["cold_observed_seconds"]+(count-1)*row["warm_observed_seconds"],
                "scope": "descriptive repeated-service arithmetic; new supplied anchors or fresh IID estimates of one fixed law; identical answers may be cached"})
    return costs, totals


def amortization_tables(tables):
    result = []
    for candidate in (c for c in tables["costs"] if c["family"] == "LNA-correction"):
        for control in tables["costs"]:
            if control["family"] == "LNA-correction" or any(candidate[k] != control[k] for k in ("case", "profile", "role")):
                continue
            for statistic in ("mean", "covariance"):
                def gate(item):
                    def compatible(row):
                        return all(row[k] == item[v] for k, v in (("regime", "case"), ("profile", "profile"), ("role", "role")))
                    if item.get("complete_process_cost_receipts") is not True:
                        return "complete-process-cost-unresolved"
                    if item.get("evaluation_complete_process_cost_receipts") is not True:
                        return "evaluation-process-cost-unresolved"
                    if item.get("timing_attempt_cost_complete") is not True:
                        return "timing-attempt-cost-unresolved"
                    if item.get("native_label_replay_verified") is not True:
                        return "native-label-replay-unresolved"
                    if item.get("full_scientific_allocation") is not True:
                        return "reduced-software-allocation"
                    rows = [r for r in tables["accuracy"] if compatible(r) and r["name"] == item["name"]]
                    if not item["available"] or not rows or any(not r["available"] for r in rows):
                        return "prediction-unavailable"
                    sampling = [r for r in tables["sampling"] if compatible(r) and r["name"] == item["name"]]
                    refinement = [r for r in tables["reference_refinement"] if compatible(r)]
                    if len(sampling) != len(rows):
                        return "standalone-sampling-unresolved"
                    bounds = [r for r in tables["components"] if compatible(r) and r["name"] == item["name"]]
                    if not bounds or any(not r["mean_inside_physical_bounds"] or not r["variance_inside_physical_bounds"] for r in bounds):
                        return "physical-moment-bound-failure"
                    return _base.statistic_gate(rows, sampling, refinement, statistic)
                saving = control["warm_observed_seconds"]-candidate["warm_observed_seconds"]
                arithmetic = max(1, math.ceil(1+(candidate["known_offline_seconds"]+candidate["cold_observed_seconds"]-
                    control["known_offline_seconds"]-control["cold_observed_seconds"])/saving)) if saving > 0 else None
                first, second = gate(candidate), gate(control)
                result.append({**{k: candidate[k] for k in ("case", "profile", "role", "seed")},
                    "candidate": candidate["name"], "control": control["name"], "statistic": statistic,
                    "candidate_gate": first, "control_gate": second, "descriptive_payback_calls": arithmetic,
                    "accuracy_qualified_payback_calls": arithmetic if first == second == "finite-reference-diagnostic-pass" else None,
                    "scope": "finite-reference diagnostic only; continuous stopping/time/particle/continuum bias unresolved"})
    return result


def load_complete(root):
    evidence = _evaluator().load_complete(Path(root).resolve())
    validate_evidence(evidence)
    return evidence


def report_text(allocation, summary, tables):
    qualified = sum(row["accuracy_qualified_payback_calls"] is not None for row in tables["amortization"])
    text = ("# Supervised nonlinear-LNA payoff correction\n\n"
        f"This complete allocation retains {allocation['candidate_models']} fitted models, {allocation['new_primary_records']} new primary records and "
        f"{allocation['collocation_primary_records']} collocation primary records. No optimizer seed or failed prediction is removed.\n\n"
        "The Gaussian control already evolves a nonlinear deterministic mean. This candidate learns supervised finite-Euler payoff moments from "
        "native continuation labels; it is a different information and cost regime from generator-residual learning. It predicts only the fixed "
        "four-output bank and same-time full covariance, not density paths, stopping probabilities or arbitrary observables.\n\n"
        f"All {summary['native_MC_label_families']:,} learner continuation families remain, including stopped paths. The labeled target uses frozen terminal states "
        "and killed running integrals. Euler refinement and continuous hitting bias remain distinct.\n\n"
        "Primary IID estimates use fresh initials independent of reference and total covariance. Each histogram has its own conditional nonlinear LNA ODE. "
        "The accuracy cache is shared across seeds only for evaluation; every standalone timing reruns every ODE.\n\n"
        "Signed whole-family bootstrap contrasts include independent initial-pool uncertainty for both methods. Intervals and exclusion counts are pointwise; "
        "they establish no simultaneous statement. Positive bootstrap distances or norms are not used to assert a nonzero bias.\n\n"
        "Cold and warm each contain one uninterrupted full-workload observation including histogram initialization, all conditional ODEs, correction "
        "heads and aggregation. Nested component timers are breakdowns, never added again. These observations are not a stable latency distribution. "
        "Restoration, complete resume warm-up and paid interrupted attempts remain explicit. Unknown hard-kill costs block qualified payback.\n\n"
        f"The independent learner-label prefix audit verifies {summary['native_label_replay_audit']['verified_families']} saved families. "
        f"Its status is {'complete' if summary['native_label_replay_audit']['complete'] else 'not a completed scientific prefix audit for this reduced software allocation'}. "
        "Replay overhead is separate from producer preparation; a reduced software fixture cannot supply the scientific audit or qualified payback.\n\n"
        "The study cost counts full preparation plus worker invocations once; MC, baseline-cache, optimizer and pool components are nested. A standalone chosen "
        "model conservatively pays the full observed multi-case input build once, its own fit and measured setup. Neither training nor per-anchor ODE work is free.\n\n"
        f"Measured evaluator work-command process time is {summary['evaluation_work_process_seconds_nonadditive']:.6g} seconds "
        "(nonadditive: includes imports, preflight and nested prediction/cache/timing work). CLI stage coverage is explicit; missing whole-process work receipts "
        "block qualified payback. Later verification commands remain separate local overhead and do not change this frozen report.\n\n"
        f"There are {qualified} accuracy-qualified descriptive payback rows. Failed or unresolved sampling/refinement/physical-bound gates remain in amortization.csv. "
        "No continuous stopping, microscopic-particle, continuum-DK or universal neural advantage follows.\n")
    accuracy = [[r["case"], r["name"], r["seed"], "unavailable" if r["worst_mean"] is None else f"{100*r['worst_mean']:.3g}",
        "unavailable" if r["worst_covariance"] is None else f"{100*r['worst_covariance']:.3g}", r["unavailable_query_rows"]]
        for r in tables["model_summary"]]
    text += ("\nThe following point errors retain every candidate seed and new control. Mean percentages use reference SD; covariance percentages use "
        "products of reference SDs. They are not relative errors of possibly zero moments, and point estimates alone do not pass the uncertainty gates.\n\n"+
        _base._table(("Case", "Method", "Seed", "Worst mean (% SD)", "Worst covariance (% SD products)", "Unavailable queries"), accuracy)+"\n")
    return text


def model_summary(tables):
    grouped = {}
    for row in tables["accuracy"]:
        if row["family"] in ("LNA-correction", "gated-LNA", "raw-conditional-LNA"):
            grouped.setdefault((row["regime"], row["name"], row["seed"]), []).append(row)
    return [{"case": case, "name": name, "seed": seed, "unavailable_query_rows": sum(not r["available"] for r in values),
        "worst_mean": max((r["max_mean_error_in_reference_std"] for r in values if r["available"]), default=None),
        "worst_covariance": max((r["max_normalized_covariance_error"] for r in values if r["available"]), default=None)}
        for (case, name, seed), values in sorted(grouped.items())]


def generate(root, documentation=None, output=None):
    evidence = load_complete(root)
    if documentation is not None:
        document = Path(documentation).resolve()
        if document.suffix != ".md" or any(document.is_relative_to(p) for p in (evidence["root"], evidence["collocation_root"])):
            raise ValueError("Optional documentation must be a Markdown file outside authoritative study roots")
    allocation = validate_evidence(evidence)
    summary, components, pools = preparation_costs(evidence)
    tables = accuracy_tables(evidence)
    tables["timing_observations"] = timing_table(evidence)
    tables["costs"], tables["total_costs"] = deployment_costs(evidence, tables["timing_observations"], summary, pools)
    tables["amortization"] = amortization_tables(tables)
    tables["model_summary"] = model_summary(tables)
    tables["preparation_components"] = components
    tables["evaluation_process_receipts"] = summary["evaluation_process_receipts"]
    tables["training"] = [{"case": m["case"], "name": m["name"], "seed": m["seed"],
        "full_fit_invocation_seconds": m["full_invocation_seconds"], "recorded_optimizer_loop_seconds": m["training_seconds"],
        "mean_selected_step": m["selected_step"]["mean"], "covariance_selected_step": m["selected_step"]["covariance"],
        "mean_natural_validation_mse": m["natural_validation"]["mean"]["natural_mse"],
        "covariance_natural_validation_mse": m["natural_validation"]["covariance"]["natural_mse"]} for m in evidence["candidate_lock"]["models"]]
    for case, manifest in evidence["baseline"]["references"].items():
        reference_directory = evidence["collocation_root"]/"reference"/case
        artifact(reference_directory/"manifest.json", evidence["inputs"])
        artifact(reference_directory/"reference.h5", evidence["inputs"], manifest["reference_sha256"])
    helpers = {**evidence["baseline"].get("plan", {}).get("helpers", {}),
        **evidence["packet"]["helpers"], **evidence.get("additional_helpers", {})}
    for path, digest in helpers.items():
        artifact(path, evidence["inputs"], digest)
    identity = {"kind": KIND, "source_hash": source_hash(), "reporter_sha256": sha(__file__),
        "joint_lock_sha256": evidence["joint_lock_sha256"], "helpers": helpers, "inputs": evidence["inputs"]}
    output = evidence["root"]/"report" if output is None else Path(output).resolve()
    if any(output == root or root.is_relative_to(output) or output.is_relative_to(root/name)
           for root in (evidence["root"], evidence["collocation_root"]) for name in ("data", "payoffs", "reference", "evaluation", "comparison", "deployment")):
        raise ValueError("Report output cannot replace authoritative study evidence")
    if (output/"manifest.json").exists():
        previous = load_report(output)
        if previous["identity"] != identity:
            raise ValueError("Report source or inputs changed; choose a new output directory")
        if documentation is not None:
            Path(documentation).parent.mkdir(parents=True, exist_ok=True)
            Path(documentation).write_text((output/"report.md").read_text())
        return previous
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("A new empty report output directory is required")
    for name, rows in tables.items():
        _base.csv_write(output/(name+".csv"), rows)
    (output/"report.md").write_text(report_text(allocation, summary, tables))
    write(output/"report.json", {"complete": True, "identity": identity, "allocation": allocation,
        "study_costs": summary, "tables": tables, "all_seeds_retained": True,
        "pointwise_not_simultaneous": True, "active_readme_goal_complete": False})
    manifest = {"kind": KIND, "complete": True, "identity": identity, "allocation": allocation,
        "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write(output/"manifest.json", manifest)
    if documentation is not None:
        Path(documentation).parent.mkdir(parents=True, exist_ok=True)
        Path(documentation).write_text((output/"report.md").read_text())
    return manifest


def load_report(output):
    output = Path(output).resolve()
    value = read(output/"manifest.json")
    identity = value["identity"]
    if (value.get("kind") != KIND or not value.get("complete") or identity["source_hash"] != source_hash()
            or identity["reporter_sha256"] != sha(__file__)):
        raise ValueError("Complete current-source LNA report required")
    for path, digest in identity["inputs"].items():
        if sha(path) != digest:
            raise ValueError("Frozen report input changed")
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p != output/"manifest.json"}
    if actual != set(value["outputs"]) or any(sha(output/name) != digest for name, digest in value["outputs"].items()):
        raise ValueError("Complete exact report output inventory required")
    result = read(output/"report.json")
    if not result.get("complete") or result["identity"] != identity or result["allocation"] != value["allocation"]:
        raise ValueError("Report manifest/result identity differs")
    return value


def archive(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    report = load_report(root/"report")
    evidence = load_complete(root)
    tested, test_logs = verify_final_tests(root, helpers=report["identity"]["helpers"])
    roots = {"candidate": root, "collocation": evidence["collocation_root"]}
    if destination.exists() or any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in roots.values()):
        raise ValueError("Choose a new compact archive outside both authoritative studies")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lna-evidence-", dir=destination.parent) as temporary:
        staging = Path(temporary)/"archive"
        staging.mkdir()
        copied, inventory = {}, {}

        def copy_file(source, logical):
            target = staging/logical
            compressed = source.suffix == ".json" and source.stat().st_size > 100000
            if compressed:
                target = target.with_name(target.name+".gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            if compressed:
                with source.open("rb") as src, target.open("wb") as dst:
                    with gzip.GzipFile(fileobj=dst, mode="wb", filename="", mtime=0) as zipped:
                        shutil.copyfileobj(src, zipped)
            else:
                shutil.copyfile(source, target)
            copied[str(target.relative_to(staging))] = {"sha256": sha(target), "original_sha256": sha(source),
                "compressed": compressed, "logical_path": logical}

        for label, directory in roots.items():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    logical = label+"/"+str(path.relative_to(directory))
                    inventory[logical] = {"sha256": sha(path), "bytes": path.stat().st_size}
                    if path.suffix in (".json", ".md", ".csv", ".png", ".py") or path.resolve() in test_logs:
                        copy_file(path, logical)
        sources = set(Path(path) for path in report["identity"]["helpers"])
        sources.update(REPO/name for name in tested["source_inputs"])
        sources.update((REPO/"analysis").glob("*.py"))
        sources.update((REPO/"analysis").glob("*.cpp"))
        sources.update((REPO/"src/nn_dk").rglob("*.py"))
        sources.update((REPO/"tests").glob("test_lna*.py"))
        sources.add(REPO/"tests/test_batched_native_gaussian.py")
        sources.update(REPO/name for name in ("pyproject.toml", "uv.lock", "LICENSE", "LICENSE-DOCS", "LICENSING.md"))
        for path in sorted(sources):
            if not path.resolve().is_relative_to(REPO):
                raise ValueError("Source snapshot must use repository files")
            if path.is_file():
                copy_file(path, "code/"+str(path.relative_to(REPO)))
        helpers = {"code/"+str(Path(path).relative_to(REPO)): digest for path, digest in report["identity"]["helpers"].items()}
        (staging/"README.md").write_text("# Supervised LNA correction evidence\n\n"
            "See [the all-seed report](candidate/report/report.md). Large JSON is losslessly compressed. "
            "Full HDF5, checkpoints and numerical/trajectory/IID caches remain local and checksum-indexed. "
            "Every declared seed, failure and stopped family is retained. Code is MIT; project-authored documentation is CC BY 4.0; "
            "see [license coverage](code/LICENSING.md). This is a partial repository snapshot: tested source/configuration is included, "
            "while Git history and uncopied documentation remain in the [canonical repository](https://github.com/xrough/nn-dk) and "
            "[canonical documentation](https://github.com/xrough/nn-dk/tree/main/docs). Relative links into uncopied documents should be resolved there.\n")
        write(staging/"archive_manifest.json", {"kind": KIND, "complete": True, "allocation": report["allocation"],
            "copied": copied, "raw_inventory": inventory, "source_helpers": helpers,
            "generated": {"README.md": sha(staging/"README.md")}, "active_readme_goal_complete": False,
            "large_raw_artifacts_retained_locally": True, "report_manifest_sha256": sha(root/"report/manifest.json"),
            "test_receipt_sha256": sha(root/"audit/final_tests.json"), "tested_source_inputs": tested["source_inputs"],
            "test_log_paths": {str(p): "candidate/"+str(p.relative_to(root)) for p in test_logs}})
        verify_archive(staging)
        staging.rename(destination)
    return verify_archive(destination)


def verify_archive(destination):
    """Offline verification uses archived bytes only, never live source/paths."""
    destination = Path(destination).resolve()
    manifest = read(destination/"archive_manifest.json")
    if (manifest.get("kind") != KIND or not manifest.get("complete")
            or manifest.get("active_readme_goal_complete") is not False
            or manifest.get("large_raw_artifacts_retained_locally") is not True):
        raise ValueError("Complete scoped compact LNA archive required")
    expected = set(manifest["copied"])|set(manifest["generated"])|{"archive_manifest.json"}
    if {str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()} != expected:
        raise ValueError("Archive file inventory changed")
    logical = {}
    for name, item in manifest["copied"].items():
        path = destination/name
        if not path.resolve().is_relative_to(destination) or sha(path) != item["sha256"]:
            raise ValueError("Archive checksum/path changed")
        original = gzip.decompress(path.read_bytes()) if item["compressed"] else path.read_bytes()
        if hashlib.sha256(original).hexdigest() != item["original_sha256"] or item["logical_path"] in logical:
            raise ValueError("Lossless original archive checksum/inventory changed")
        logical[item["logical_path"]] = (item, original)
    for name, digest in manifest["generated"].items():
        if not (destination/name).resolve().is_relative_to(destination) or sha(destination/name) != digest:
            raise ValueError("Archive generated documentation changed")
    for path, digest in manifest["source_helpers"].items():
        if path not in logical or logical[path][0]["original_sha256"] != digest:
            raise ValueError("Archived frozen source helper changed")
    test_item = logical["candidate/audit/final_tests.json"]
    tested = json.loads(test_item[1])
    if (test_item[0]["original_sha256"] != manifest.get("test_receipt_sha256") or tested.get("kind") != TEST_GATE_KIND
            or tested.get("complete") is not True or tested.get("source_inputs") != manifest.get("tested_source_inputs")
            or tested.get("before_sha256") != stable_hash(tested["source_inputs"])
            or tested.get("after_sha256") != tested["before_sha256"]):
        raise ValueError("Archived final-test receipt/source identity changed")
    for name, digest in tested["source_inputs"].items():
        if "code/"+name not in logical or logical["code/"+name][0]["original_sha256"] != digest:
            raise ValueError("Archived maintained source differs from final tested bytes")
    commands = tested.get("commands", [])
    if len(commands) != 2 or {c.get("kind") for c in commands} != set(TEST_COMMANDS):
        raise ValueError("Archived full-suite command inventory changed")
    if len({c["log_path"] for c in commands}) != len(commands):
        raise ValueError("Archived full-suite commands must retain separate logs")
    for command in commands:
        log_name = manifest["test_log_paths"].get(command["log_path"])
        if (command.get("argv") != TEST_COMMANDS[command["kind"]] or command.get("exit_code") != 0
                or log_name not in logical or logical[log_name][0]["original_sha256"] != command["log_sha256"]):
            raise ValueError("Archived final-test log or command changed")
        if command["kind"] == "pytest":
            counts = re.findall(r"\b(\d+) passed\b", logical[log_name][1].decode())
            if not counts or command.get("test_count") != int(counts[-1]):
                raise ValueError("Archived pytest summary/count changed")
    for name, item in manifest["raw_inventory"].items():
        if (Path(name).is_absolute() or ".." in Path(name).parts or type(item.get("bytes")) is not int
                or item["bytes"] < 0 or not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64):
            raise ValueError("Invalid local raw-artifact checksum inventory")
        if name in logical and (logical[name][0]["original_sha256"] != item["sha256"] or len(logical[name][1]) != item["bytes"]):
            raise ValueError("Copied evidence disagrees with its local raw-artifact index")
    original_manifest = logical["candidate/report/manifest.json"]
    if original_manifest[0]["original_sha256"] != manifest["report_manifest_sha256"]:
        raise ValueError("Archived report manifest changed")
    report = json.loads(original_manifest[1])
    result = json.loads(logical["candidate/report/report.json"][1])
    if (not report.get("complete") or report.get("kind") != KIND or not result.get("complete")
            or report["allocation"] != manifest["allocation"] or result["allocation"] != report["allocation"]
            or result["identity"] != report["identity"]):
        raise ValueError("Archived report identity/allocation differs")
    for name, digest in report["outputs"].items():
        if logical["candidate/report/"+name][0]["original_sha256"] != digest:
            raise ValueError("Archived report output differs")
    return {"complete": True, "copied_artifacts_verified": len(manifest["copied"]),
        "indexed_raw_files": len(manifest["raw_inventory"]), "allocation": manifest["allocation"],
        "manifest_sha256": sha(destination/"archive_manifest.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("report", "archive", "verify-report", "verify-archive"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--documentation", type=Path)
    args = parser.parse_args(argv)
    if args.action in ("report", "archive") and args.root is None or args.action != "report" and args.output is None:
        parser.error("report/archive require --root; archive/verify actions require --output")
    result = generate(args.root, args.documentation, args.output) if args.action == "report" else (
        archive(args.root, args.output) if args.action == "archive" else load_report(args.output)
        if args.action == "verify-report" else verify_archive(args.output))
    print(json.dumps({k: v for k, v in result.items() if k not in ("identity", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
 
