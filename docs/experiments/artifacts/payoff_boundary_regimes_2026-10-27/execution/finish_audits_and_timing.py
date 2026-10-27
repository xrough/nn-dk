"""One-shot serialized audits, final software checks, then isolated timing.

Root must explicitly approve the script hash and the absence of untracked
heavy work in execution/timing_isolation_approved.json before launch. Existing
tracked fitting/reference/accuracy/review work may finish while this waits.
No fit, model selection, reference generation, or report generation occurs.
Any nonzero child status, including 75, closes this attempt before later stages.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SCRIPT = Path(__file__).resolve()
PYTHON = REPO/".venv/bin/python"
OUTPUT = ROOT/"execution/finalization.json"
APPROVAL = ROOT/"execution/timing_isolation_approved.json"
EXPECTED_SOURCE = "0d9bda04e5da1cc00cc3ec5917d23a4352e6ceb4585dd1a066779b364cf79576"
EXPECTED_COMMIT = "0d38243b36924f6752f6b8c1339788fcb6030074"
EXPECTED_REPORT = "9433203ad56e9dbe9fa7ed0356716558ac0593383caaf06d1642bd3b3d827f62"
EXPECTED_REPORT_AUDIT = "f32a053803c1bb098a189304f8e111fe77786953c60d985292c8e2dd61ad2a5c"
MAXIMUM_SECONDS = 8*60*60
ENVIRONMENT = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "MPLCONFIGDIR": "/private/tmp/nn-dk-mpl-cache"}
EXPECTED_CONTINUATION_LABELS = {"fit_lock", "postfit_validation", "accuracy", "independent_deployment"}
CASES = ("T0p25_k0p75", "T0p25_k3", "T0p25_k6", "T1_k0p75", "T1_k3", "T1_k6")
EXPECTED_CONTINUATION_LABELS |= {"reference_"+case for case in CASES}
SEEDS = (0, 1, 2)
APPROVAL_TRUE_FIELDS = ("approved", "no_untracked_heavy_jobs_or_agents",
                        "only_existing_tracked_pipeline_and_review_may_run_until_barrier",
                        "no_other_heavy_work_until_finalization_complete")
FROZEN_ROOT_PATHS = (
    "execution/source_freeze.json", "execution/continuation.json", "execution/finish_after_fits.py",
    "execution/finish_after_fits_attempt2.py", "execution/review_after_validation.py",
    "plan.json", "payoffs/preregistration.json", "payoffs/data_lock.json",
    "comparison/preregistration.json", "comparison/legacy_controls.json",
    "deployment/preregistration.json", "deployment/preparation.json",
    "audit/pretest_audit.json", "audit/deployment_pretest_independence.json",
    "audit/formulation_control.json",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def maintained_sources():
    paths = [path for folder in ("src", "analysis", "tests")
             for path in (REPO/folder).rglob("*.py")]
    paths += [REPO/"pyproject.toml", REPO/"uv.lock"]
    return {str(path.relative_to(REPO)): sha(path) for path in sorted(paths)}


def numerical_source():
    if str(REPO/"src") not in sys.path:
        sys.path.insert(0, str(REPO/"src"))
    from nn_dk.data import source_hash

    return source_hash()


def validate_static_sources():
    frozen = read(ROOT/"execution/source_freeze.json")
    if (not frozen.get("complete") or frozen.get("source_hash") != EXPECTED_SOURCE
            or frozen.get("commit") != EXPECTED_COMMIT or numerical_source() != EXPECTED_SOURCE):
        raise ValueError("Scientific package source differs from its original pre-fit freeze")
    if (sha(REPO/"analysis/report_payoff_boundary_regimes.py") != EXPECTED_REPORT
            or sha(REPO/"analysis/audit_payoff_boundary_report.py") != EXPECTED_REPORT_AUDIT):
        raise ValueError("Final report or report-audit code differs from root's reviewed SHA")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from analysis import run_payoff_boundary_regimes as runner

    plan, design = runner._assert_frozen(ROOT)
    if (tuple(plan.optimizer_seeds) != SEEDS or tuple(runner.case_names(plan)) != CASES
            or design["expected_models"] != 18 or len(design["helpers"]) != 27):
        raise ValueError("Only the full frozen six-case, three-seed study is permitted")
    return runner, plan, design


def validate_approval():
    approved = read(APPROVAL)
    if (any(approved.get(key) is not True for key in APPROVAL_TRUE_FIELDS)
            or approved.get("approved_by") != "root" or approved.get("script_sha256") != sha(SCRIPT)
            or approved.get("source_hash") != EXPECTED_SOURCE):
        raise ValueError("Root's script-specific timing-isolation approval is missing or incompatible")
    return approved


def log_closed_record(record, *, label_key="label", code_key="returncode"):
    if (record.get(code_key) != 0 or not isinstance(record.get("wall_seconds"), (int, float))
            or record["wall_seconds"] < 0 or record.get("finished_unix") is None
            or record.get("CPU_seconds", -1) < 0 or not record.get("log_sha256")
            or sha(record["log_path"]) != record["log_sha256"]):
        raise ValueError(f"Tracked child lacks successful waited/closed evidence: {record.get(label_key)}")


def completed_inputs(runner):
    continuation_path = ROOT/"execution/continuation_attempt2.json"
    review_packet_path = ROOT/"execution/fit_review_continuation.json"
    if not continuation_path.exists() or not review_packet_path.exists():
        return None
    continuation, review_packet = read(continuation_path), read(review_packet_path)
    if continuation.get("stage") == "failed" or review_packet.get("status") == "failed":
        raise RuntimeError("A tracked upstream continuation failed; preserve it and do not continue")
    if not continuation.get("complete") or review_packet.get("status") != "complete":
        return None
    if (continuation.get("source_hash") != EXPECTED_SOURCE or continuation.get("final_timing_launched")
            or continuation.get("continuation_attempt") != 2 or continuation.get("failure")
            or continuation.get("primary_accuracy_records") != 360
            or continuation.get("retained_accuracy_records") != 444):
        raise ValueError("Unexpected completed accuracy continuation")
    calls = continuation["calls"]
    if len(calls) != len(EXPECTED_CONTINUATION_LABELS) or {row["label"] for row in calls} != EXPECTED_CONTINUATION_LABELS:
        raise ValueError("Every lock/validation/reference/accuracy child must complete exactly once")
    for row in calls:
        log_closed_record(row)
    if (sha(continuation["script_path"]) != continuation["script_sha256"]
            or continuation.get("fit_lock_sha256") != sha(ROOT/"payoffs/fit_lock.json")
            or continuation.get("postfit_validation_sha256") != sha(ROOT/"audit/postfit_validation_diagnostic.json")):
        raise ValueError("Completed continuation's script, locked fits, or audit changed")
    for relative, digest in continuation["frozen_artifacts"].items():
        if sha(ROOT/relative) != digest:
            raise ValueError(f"An upstream frozen artifact changed: {relative}")
    if (review_packet.get("return_code") != 0 or review_packet.get("source_hash") != EXPECTED_SOURCE
            or review_packet.get("review_models") != 18 or review_packet.get("review_source_hash") != EXPECTED_SOURCE
            or review_packet.get("fit_lock_sha256") != sha(ROOT/"payoffs/fit_lock.json")
            or review_packet.get("audit_sha256") != sha(ROOT/"audit/postfit_validation_diagnostic.json")
            or sha(ROOT/"execution/fit_review.log") != review_packet.get("log_sha256")
            or sha(ROOT/"audit/fit_interpretation_review.json") != review_packet.get("review_sha256")
            or review_packet.get("command_wall_seconds", -1) < 0
            or review_packet.get("command_cpu_user_seconds", -1) < 0
            or review_packet.get("command_cpu_system_seconds", -1) < 0):
        raise ValueError("Fit review must have completed its owned child and closed its log")
    review = read(ROOT/"audit/fit_interpretation_review.json")
    if (not review.get("complete") or len(review.get("records", [])) != 18
            or review["identity"]["source_hash"] != EXPECTED_SOURCE
            or review["identity"]["fit_lock_sha256"] != sha(ROOT/"payoffs/fit_lock.json")
            or review["identity"]["future_outcomes_opened"] is not False
            or not review.get("no_fitting_selection_or_future_outcome_access")):
        raise ValueError("The all-model read-only fit review is incomplete or incompatible")
    lock = runner.verify_fit_lock(ROOT)
    if not lock.get("complete") or lock.get("expected_models") != 18 or len(lock.get("models", [])) != 18:
        raise ValueError("Strict all-eighteen checkpoint restoration is required")
    workers = {}
    for seed in SEEDS:
        path = ROOT/f"payoffs/execution/worker_seed{seed}.json"
        worker = read(path)
        attempts = worker.get("attempts", [])
        if (not worker.get("complete") or worker.get("source_hash") != EXPECTED_SOURCE
                or worker.get("seed") != seed or len(worker.get("models", [])) != 6
                or not attempts or not attempts[-1].get("complete")
                or attempts[-1].get("finished_unix") is None
                or worker.get("preregistration_sha256") != sha(ROOT/"payoffs/preregistration.json")):
            raise ValueError("Every worker must record a finished complete attempt with six models")
        workers[str(seed)] = {"path": str(path), "sha256": sha(path), "finished_attempt": attempts[-1]}
    for folder, expected_groups, records_per_group in (("comparison", 36, 10), ("deployment", 12, 7)):
        index = read(ROOT/folder/"accuracy_index.json")
        if (not index.get("complete") or index.get("source_hash") != EXPECTED_SOURCE
                or len(index["groups"]) != expected_groups or index["records_per_group"] != records_per_group):
            raise ValueError("Every original and independent-initial accuracy record is required")
    return {"continuation_attempt2_sha256": sha(continuation_path),
            "fit_review_continuation_sha256": sha(review_packet_path),
            "fit_interpretation_review_sha256": sha(ROOT/"audit/fit_interpretation_review.json"),
            "fit_lock_sha256": sha(ROOT/"payoffs/fit_lock.json"), "workers": workers,
            "all_tracked_heavy_children_successfully_waited_and_logs_closed": True,
            "PID_liveness_signals": "not used; completed waited-child packets and ownership flocks"}


def ownership_barrier():
    locks = [ROOT/f"payoffs/execution/.seed{seed}.lock" for seed in SEEDS]
    locks += [ROOT/"payoffs/execution/.fit-lock.lock"]
    locks += [ROOT/f"reference/{case}/execution/.generation.lock" for case in CASES]
    with ExitStack() as stack:
        for path in locks:
            stream = stack.enter_context(path.open("r"))
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return {str(path.relative_to(ROOT)): sha(path) for path in locks}


def expected_timing_slots(plan):
    controls = read(ROOT/"comparison/legacy_controls.json")
    lock = read(ROOT/"payoffs/fit_lock.json")
    comparison = read(ROOT/"comparison/preregistration.json")
    slots, loads = set(), set()
    for case in CASES:
        heads = {row["name"] for row in lock["models"] if row["case"] == case}
        heads |= {row["name"] for row in controls["models"] if row["case"] == case}
        heads.add(case+"_untrained")
        if len(heads) != 7:
            raise ValueError("Every timing problem requires all three new/old seeds and untrained control")
        for profile in ("uniform", "nonuniform"):
            for name in heads:
                loads.add((case, profile, name))
                for batch in comparison["neural_conditional_batches"]:
                    slots.add(f"{case}/{profile}/{name}/conditional/{batch}")
                slots.add(f"{case}/{profile}/{name}/ensemble")
            for step in comparison["LGA_max_steps"]:
                for role in ("conditional", "ensemble"):
                    slots.add(f"{case}/{profile}/LGA/{step}/{role}")
            for role in ("conditional", "ensemble"):
                slots.add(f"{case}/{profile}/native-MC/{role}")
    if len(slots) != 408 or len(loads) != 84 or tuple(plan.optimizer_seeds) != SEEDS:
        raise ValueError("Unexpected final timing allocation")
    return slots, loads


def pytest_summary(path):
    text = Path(path).read_text()
    lines = [line for line in text.splitlines() if re.search(r"\b\d+ passed\b", line)]
    if not lines:
        raise ValueError("A successful full-suite pytest summary is required")
    line = lines[-1]
    passed = int(re.search(r"(\d+) passed", line).group(1))
    warning = re.search(r"(\d+) warnings?", line)
    seconds = re.search(r"in ([0-9.]+)s", line)
    if passed < 1150 or seconds is None:
        raise ValueError("Full-suite success cannot omit the pre-freeze test inventory")
    return {"passed": passed, "warnings": int(warning.group(1)) if warning else 0,
            "seconds": float(seconds.group(1)), "summary": line.strip(),
            "skipped": int(found.group(1)) if (found := re.search(r"(\d+) skipped", line)) else 0,
            "xfailed": int(found.group(1)) if (found := re.search(r"(\d+) xfailed", line)) else 0}


def execute():
    started, cpu_started = time.monotonic(), time.process_time()
    if OUTPUT.exists():
        raise FileExistsError("Preserve the original one-shot finalization packet")
    approved = validate_approval()
    runner, plan, design = validate_static_sources()
    for path in (ROOT/"comparison/timing.json", ROOT/"audit/final_tests.json",
                 ROOT/"audit/native_reference_candidate_audit.json", ROOT/"audit/metrics_replay.json"):
        if path.exists():
            raise FileExistsError(f"A finalization-owned output already exists: {path}")
    frozen_sources = maintained_sources()
    frozen_inputs = {relative: sha(ROOT/relative) for relative in FROZEN_ROOT_PATHS}
    packet = {"kind": "one-shot-native-metrics-tests-isolated-timing-v1", "pid": os.getpid(),
              "script_path": str(SCRIPT), "script_sha256": sha(SCRIPT), "source_hash": EXPECTED_SOURCE,
              "scientific_source_commit": EXPECTED_COMMIT, "started_utc": utc(), "started_unix": time.time(),
              "stage": "waiting-for-all-tracked-heavy-work", "complete": False, "calls": [],
              "root_isolation_approval": approved, "root_isolation_approval_sha256": sha(APPROVAL),
              "maintained_sources_frozen_at_launch": frozen_sources, "frozen_scientific_prerequisites": frozen_inputs,
              "registered_helper_sha256": design["helpers"], "maximum_wait_seconds": MAXIMUM_SECONDS,
              "no_fitting_selection_or_report_generation": True,
              "child_accounting": "individual wait4 user+system CPU and monotonic wall; logs closed before next stage",
              "first_failed_continuation_preserved": {"path": str(ROOT/"execution/continuation.json"),
                                                     "sha256": sha(ROOT/"execution/continuation.json")},
              "timing_expected_records": 408, "timing_expected_restorations": 84}

    def save():
        packet.update(elapsed_wall_seconds=time.monotonic()-started,
                      parent_cpu_seconds=time.process_time()-cpu_started,
                      recorded_child_CPU_seconds=sum(row.get("CPU_seconds", 0.) for row in packet["calls"]))
        write(OUTPUT, packet)

    def guard(*, budget=True):
        if (sha(SCRIPT) != packet["script_sha256"] or sha(APPROVAL) != packet["root_isolation_approval_sha256"]
                or maintained_sources() != frozen_sources):
            raise ValueError("Script, timing-isolation approval, or maintained implementation/tests changed after launch")
        for relative, digest in frozen_inputs.items():
            if sha(ROOT/relative) != digest:
                raise ValueError(f"Frozen scientific prerequisite changed: {relative}")
        runner._assert_frozen(ROOT)
        if numerical_source() != EXPECTED_SOURCE:
            raise ValueError("Numerical source changed")
        if budget and time.monotonic()-started > MAXIMUM_SECONDS:
            raise TimeoutError("Eight-hour finalization budget reached; no further child stages may launch")

    def checked(label, command):
        guard()
        packet["stage"] = label
        log = ROOT/"execution"/("final_"+label+".log")
        record = {"label": label, "command": command, "display_command": shlex.join(command),
                  "log_path": str(log), "started_utc": utc(), "started_unix": time.time(), "returncode": None}
        packet["calls"].append(record)
        save()
        print(label, flush=True)
        tick = time.monotonic()
        try:
            with log.open("x") as stream:
                child = subprocess.Popen(command, cwd=REPO, env=ENVIRONMENT,
                                         stdout=stream, stderr=subprocess.STDOUT)
                record["pid"] = child.pid
                save()
                # Never signal a PID. The known owned child is waited to
                # completion, and no later stage starts if it exhausts budget.
                waited, status, usage = os.wait4(child.pid, 0)
                if waited != child.pid:
                    raise RuntimeError("wait4 returned an unexpected child")
                child.returncode = os.waitstatus_to_exitcode(status)
                record.update(returncode=child.returncode, user_CPU_seconds=usage.ru_utime,
                              system_CPU_seconds=usage.ru_stime, CPU_seconds=usage.ru_utime+usage.ru_stime)
        except BaseException as error:
            record["execution_exception"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            record.update(finished_utc=utc(), finished_unix=time.time(), wall_seconds=time.monotonic()-tick)
            if log.exists():
                record["log_sha256"] = sha(log)
            save()
        if record["returncode"] != 0:
            raise RuntimeError(f"{label} failed closed with status {record['returncode']}; see {log}")
        guard()
        return record

    save()
    try:
        while True:
            guard()
            prerequisites = completed_inputs(runner)
            if prerequisites is not None:
                try:
                    locks = ownership_barrier()
                    break
                except BlockingIOError:
                    pass
            time.sleep(10)
        packet.update(upstream_completion=prerequisites, released_ownership_locks=locks,
                      wait_before_audits_seconds=time.monotonic()-started,
                      timing_isolation_barrier_reached_utc=utc())
        save()
        checked("native_audit", [str(PYTHON), "analysis/audit_payoff_boundary_native.py", "--root", str(ROOT)])
        native = read(ROOT/"audit/native_reference_candidate_audit.json")
        if not native.get("complete") or native.get("source_hash") != EXPECTED_SOURCE:
            raise ValueError("Independent native audit did not finish completely")
        checked("metrics_audit", [str(PYTHON), "analysis/audit_payoff_boundary_metrics.py", "--root", str(ROOT)])
        metrics = read(ROOT/"audit/metrics_replay.json")
        if (not metrics.get("complete") or metrics.get("source_hash") != EXPECTED_SOURCE
                or metrics.get("fit_lock_sha256") != sha(ROOT/"payoffs/fit_lock.json")):
            raise ValueError("Independent numerical metrics audit did not finish completely")
        tests = checked("full_pytest", [str(PYTHON), "-m", "pytest", "-q"])
        summary = pytest_summary(tests["log_path"])
        lint = checked("ruff", [str(REPO/".venv/bin/ruff"), "check", "."])
        final_tests = {"complete": True, "source_hash": EXPECTED_SOURCE,
            "full_suite": {"command": tests["display_command"], "log_sha256": tests["log_sha256"],
                           **summary, "returncode": tests["returncode"], "wait4_CPU_seconds": tests["CPU_seconds"],
                           "monotonic_wall_seconds": tests["wall_seconds"]},
            "lint": {"command": lint["display_command"], "log_sha256": lint["log_sha256"],
                     "returncode": lint["returncode"], "wait4_CPU_seconds": lint["CPU_seconds"],
                     "monotonic_wall_seconds": lint["wall_seconds"],
                     "immutable_archive_exclusion": "pyproject excludes immutable archived source snapshots"},
            "dependency_lock_sha256": sha(REPO/"uv.lock"), "pyproject_sha256": sha(REPO/"pyproject.toml"),
            "maintained_sources_verified_unchanged_since_launch": True,
            "maintained_sources_sha256": frozen_sources, "script_sha256": packet["script_sha256"],
            "numerical_source_unchanged_throughout": True, "recorded_utc": utc()}
        write(ROOT/"audit/final_tests.json", final_tests)
        packet["final_tests_sha256"] = sha(ROOT/"audit/final_tests.json")
        packet["all_owned_pre_timing_children_waited_and_logs_closed"] = True
        save()
        # All tracked heavy children are closed. Recheck worker/reference
        # ownership locks and the unchanged upstream completion packets.
        if completed_inputs(runner) != prerequisites or ownership_barrier() != locks:
            raise ValueError("Tracked process completion or released ownership changed before timing")
        expected_slots, expected_loads = expected_timing_slots(plan)
        if (ROOT/"comparison/timing.json").exists():
            raise FileExistsError("Final timing must start fresh; no duplicate-load resume")
        checked("benchmark", [str(PYTHON), "analysis/evaluate_payoff_boundary_regimes.py", "benchmark",
                              "--root", str(ROOT), "--final-timing"])
        timing = read(ROOT/"comparison/timing.json")
        slots = [row["slot"] for row in timing["records"]]
        loads = [(row["regime"], row["profile"], row["name"]) for row in timing["loading"]]
        if (not timing.get("complete") or not timing.get("final_timing_after_all_workers_finished")
                or len(slots) != 408 or set(slots) != expected_slots
                or len(loads) != 84 or set(loads) != expected_loads
                or timing["identity"] != {"source_hash": EXPECTED_SOURCE,
                    "plan_sha256": sha(ROOT/"comparison/preregistration.json"),
                    "fit_lock_sha256": sha(ROOT/"payoffs/fit_lock.json")}):
            raise ValueError("Final timing omitted or duplicated a declared workload/restoration")
        guard()
        packet.update(complete=True, stage="complete-audits-tests-and-isolated-timing",
                      finished_utc=utc(), finished_unix=time.time(), timing_records=408, timing_restorations=84,
                      timing_sha256=sha(ROOT/"comparison/timing.json"),
                      native_audit_sha256=sha(ROOT/"audit/native_reference_candidate_audit.json"),
                      metrics_audit_sha256=sha(ROOT/"audit/metrics_replay.json"),
                      all_owned_children_waited_and_execution_logs_closed=True,
                      no_final_report_generated=True)
        save()
        print(json.dumps({"complete": True, "timing_records": 408, "restorations": 84,
                          "output": str(OUTPUT)}), flush=True)
    except BaseException as error:
        packet.update(stage="failed", failure={"type": type(error).__name__, "message": str(error),
                      "traceback": traceback.format_exc()}, finished_utc=utc(), finished_unix=time.time())
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        _, _, design = validate_static_sources()
        print(json.dumps({"source_hash": EXPECTED_SOURCE, "registered_scientific_helpers": len(design["helpers"]),
            "maintained_files_to_freeze_at_launch": len(maintained_sources()), "script_sha256": sha(SCRIPT),
            "approval_path": str(APPROVAL), "approval_true_fields": APPROVAL_TRUE_FIELDS,
            "timing_records": 408, "restorations": 84, "no_output_or_child_process_created": True}))
        return
    # This finalization-owned lock is held through process exit. Opening it
    # does not touch any seed/reference owner's lock contents.
    own_lock = ROOT/"execution/.finalization.lock"
    with own_lock.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute()


if __name__ == "__main__":
    main()
