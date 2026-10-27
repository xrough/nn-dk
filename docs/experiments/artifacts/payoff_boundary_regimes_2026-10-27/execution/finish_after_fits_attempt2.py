"""One-shot frozen boundary study continuation; no fitting, selection or timing.

Run only after the three predeclared optimizer workers have been launched.
Every successful stage is retained. Any nonzero child status, including the
resumable budget status 75, closes this continuation before further stages.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PYTHON = REPO/".venv/bin/python"
OUTPUT = ROOT/"execution/continuation_attempt2.json"
EXPECTED_SOURCE = "0d9bda04e5da1cc00cc3ec5917d23a4352e6ceb4585dd1a066779b364cf79576"
EXPECTED_COMMIT = "0d38243b36924f6752f6b8c1339788fcb6030074"
ENVIRONMENT = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "MPLCONFIGDIR": "/private/tmp/nn-dk-mpl-cache"}
sys.path.insert(0, str(REPO/"src"))
from nn_dk.data import source_hash  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


if OUTPUT.exists():
    raise FileExistsError("Preserve the earlier one-shot continuation packet; do not overwrite it")

STARTED = time.monotonic()
CPU_STARTED = time.process_time()
PLAN = read(ROOT/"plan.json")
DESIGN = read(ROOT/"payoffs/preregistration.json")
CASES = tuple(DESIGN["configs"])
SEEDS = tuple(PLAN["optimizer_seeds"])
FROZEN_PATHS = (
    "execution/source_freeze.json", "execution/continuation.json",
    "execution/finish_after_fits.py", "plan.json", "payoffs/preregistration.json",
    "payoffs/data_lock.json", "comparison/preregistration.json", "comparison/legacy_controls.json",
    "deployment/preregistration.json", "deployment/preparation.json",
    "audit/pretest_audit.json", "audit/deployment_pretest_independence.json",
    "audit/formulation_control.json",
)
PACKET = {
    "kind": "one-shot-frozen-boundary-regime-continuation-v1", "pid": os.getpid(),
    "continuation_attempt": 2,
    "predecessor_packet": str(ROOT/"execution/continuation.json"),
    "predecessor_packet_sha256": sha(ROOT/"execution/continuation.json"),
    "predecessor_failed_before_child_jobs_or_reference_draws": True,
    "PID_liveness_probe": "unavailable in sandbox; completion/finished-attempt packets plus ownership-flock barrier",
    "script_path": str(SCRIPT), "script_sha256": sha(SCRIPT),
    "source_hash": EXPECTED_SOURCE, "scientific_source_commit": EXPECTED_COMMIT,
    "started_unix": time.time(), "stage": "checking-frozen-prerequisites",
    "calls": [], "complete": False, "no_fitting_or_selection": True,
    "final_timing_launched": False, "all_optimizer_seeds": list(SEEDS),
    "cases": list(CASES), "maximum_parallel_reference_workers": 3,
    "frozen_artifacts": {relative: sha(ROOT/relative) for relative in FROZEN_PATHS},
    "child_CPU_accounting": "per-process wait4 user+system CPU; excludes fitting workers",
    "elapsed_accounting": "monotonic wall seconds; UTC timestamps retained separately",
}


def save():
    PACKET["continuation_wall_seconds"] = time.monotonic()-STARTED
    PACKET["continuation_parent_CPU_seconds"] = time.process_time()-CPU_STARTED
    PACKET["recorded_child_CPU_seconds"] = sum(
        record.get("CPU_seconds", 0.) for record in PACKET["calls"])
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(PACKET, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(OUTPUT)


def guard():
    if sha(SCRIPT) != PACKET["script_sha256"] or source_hash() != EXPECTED_SOURCE:
        raise ValueError("Frozen continuation script or numerical package source changed")
    for relative, digest in PACKET["frozen_artifacts"].items():
        if sha(ROOT/relative) != digest:
            raise ValueError(f"Frozen prerequisite changed: {relative}")
    if (read(ROOT/"execution/source_freeze.json")["commit"] != EXPECTED_COMMIT
            or DESIGN["source_hash"] != EXPECTED_SOURCE
            or SEEDS != (0, 1, 2) or len(CASES) != 6 or DESIGN["expected_models"] != 18):
        raise ValueError("Unexpected frozen source or complete six-case/three-seed design")
    for relative in ("audit/pretest_audit.json", "audit/deployment_pretest_independence.json",
                     "audit/formulation_control.json"):
        audit = read(ROOT/relative)
        if not audit.get("complete") or audit.get("source_hash") != EXPECTED_SOURCE:
            raise ValueError(f"A complete pre-outcome gate is missing: {relative}")


def stage(name):
    guard()
    PACKET["stage"] = name
    save()
    print(name, flush=True)


def run(label, arguments):
    """Retain the literal child command, full log and exact child CPU usage."""
    wall, started = time.monotonic(), time.time()
    log = ROOT/"execution"/(label+".log")
    record = {"label": label, "arguments": arguments,
              "command": [str(PYTHON), *arguments], "log_path": str(log),
              "started_unix": started, "returncode": None}
    try:
        with log.open("x") as stream:
            process = subprocess.Popen([str(PYTHON), *arguments], cwd=REPO, env=ENVIRONMENT,
                                       stdout=stream, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            waited_pid, status, usage = os.wait4(process.pid, 0)
            if waited_pid != process.pid:
                raise RuntimeError("Child process accounting returned an unexpected PID")
            process.returncode = os.waitstatus_to_exitcode(status)
            record.update(returncode=process.returncode, user_CPU_seconds=usage.ru_utime,
                          system_CPU_seconds=usage.ru_stime,
                          CPU_seconds=usage.ru_utime+usage.ru_stime)
    except BaseException as error:
        record["execution_exception"] = {"type": type(error).__name__, "message": str(error)}
    record.update(finished_unix=time.time(), wall_seconds=time.monotonic()-wall)
    if log.exists():
        record["log_sha256"] = sha(log)
    return record


def checked(label, arguments):
    guard()
    result = run(label, arguments)
    PACKET["calls"].append(result)
    save()
    if result["returncode"] != 0:
        raise RuntimeError(f"{label} failed with status {result['returncode']}; retained log: {result['log_path']}")
    return result


def wait_workers():
    waiting_started = time.monotonic()
    while True:
        guard()
        packets = []
        for seed in SEEDS:
            path = ROOT/f"payoffs/execution/worker_seed{seed}.json"
            if not path.exists():
                continue
            worker = read(path)
            if (worker.get("source_hash") != EXPECTED_SOURCE or worker.get("seed") != seed
                    or worker.get("preregistration_sha256") != PACKET["frozen_artifacts"][
                        "payoffs/preregistration.json"]):
                raise ValueError("A worker belongs to another source, seed or preregistration")
            if not worker.get("complete"):
                attempt = worker.get("attempts", [{}])[-1]
                if "finished_unix" in attempt:
                    raise RuntimeError(f"Optimizer seed {seed} stopped incomplete; continuation cannot refit")
                # A PID signal probe raises PermissionError under this sandbox
                # and cannot establish process death. Never infer completion
                # from liveness: only final worker packets, released ownership
                # locks and the unchanged strict all-18 model lock can unlock.
            packets.append(worker)
        PACKET["completed_optimizer_workers"] = sum(bool(p.get("complete")) for p in packets)
        save()
        if len(packets) == 3 and all(p.get("complete") for p in packets):
            break
        if time.monotonic()-waiting_started >= 8*60*60:
            raise TimeoutError("Eight-hour waiting budget reached; no refitting or reference unlock")
        time.sleep(10)
    # The final worker packet is written just before its ownership lock closes.
    # Wait for every lock without changing its contents, then release the
    # barrier so the strict lock CLI can acquire those same ownership locks.
    lock_deadline = time.monotonic()+60
    while True:
        try:
            with ExitStack() as stack:
                for seed in SEEDS:
                    stream = stack.enter_context((ROOT/f"payoffs/execution/.seed{seed}.lock").open("r"))
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                PACKET["worker_packets"] = {
                    str(seed): {"sha256": sha(ROOT/f"payoffs/execution/worker_seed{seed}.json"),
                                "models": len(read(ROOT/f"payoffs/execution/worker_seed{seed}.json")["models"])}
                    for seed in SEEDS}
            break
        except BlockingIOError:
            if time.monotonic() >= lock_deadline:
                raise TimeoutError("Completed worker ownership locks did not close")
            time.sleep(.1)
    PACKET["waiting_for_workers_wall_seconds"] = time.monotonic()-waiting_started
    save()


def reference_stage():
    abort = threading.Event()

    def reference(case):
        if abort.is_set():
            return {"label": "reference_"+case, "returncode": None,
                    "skipped_after_reference_failure": True, "CPU_seconds": 0.}
        result = run("reference_"+case, ["analysis/run_payoff_boundary_regimes.py", "reference",
            "--root", str(ROOT), "--case", case])
        if result["returncode"] != 0:
            abort.set()
        return result

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(reference, case) for case in CASES]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            PACKET["calls"].append(result)
            save()
            if result["returncode"] != 0:
                failed.append(result["label"])
                abort.set()
    # Already-running HDF5 writers finish and close normally. Pending cases do
    # not start after failure; no accuracy stage runs unless all six succeeded.
    if failed:
        raise RuntimeError(f"Native reference stage failed closed: {failed}")
    for case in CASES:
        manifest = read(ROOT/"reference"/case/"manifest.json")
        if manifest.get("status") != "complete" or not manifest.get("reference_sha256"):
            raise ValueError("A native reference failed to commit its complete manifest")
    PACKET["native_reference_calls_CPU_seconds"] = sum(
        result["CPU_seconds"] for result in PACKET["calls"] if result["label"].startswith("reference_"))
    save()


def accuracy_inventory(directory, expected_groups, records_per_group):
    index_path = ROOT/directory/"accuracy_index.json"
    index = read(index_path)
    if (not index.get("complete") or index.get("source_hash") != EXPECTED_SOURCE
            or len(index["groups"]) != expected_groups
            or index["records_per_group"] != records_per_group):
        raise ValueError(f"Incomplete {directory} accuracy inventory")
    total = 0
    for case, group in index["groups"]:
        metadata = read(ROOT/directory/"accuracy"/case/group/"group.json")
        if not metadata.get("complete") or len(metadata["records"]) != records_per_group:
            raise ValueError(f"Incomplete {directory} group methods")
        if set(metadata.get("NN_versus_legacy_v2", {})) != {
                f"{case}_boundary_seed{seed}" for seed in SEEDS}:
            raise ValueError("All matched-seed v3 versus legacy contrasts are required")
        total += len(metadata["records"])
    return {"groups": expected_groups, "records_per_group": records_per_group,
            "records": total, "index_sha256": sha(index_path)}


save()
try:
    guard()
    if any((ROOT/"reference"/case/"reference.h5").exists() for case in CASES):
        raise ValueError("A fresh one-shot continuation must precede every final reference draw")
    if (ROOT/"audit/postfit_validation_diagnostic.json").exists():
        raise ValueError("The owned postfit audit already exists; do not duplicate it")
    stage("waiting-for-all-three-predeclared-fit-workers")
    wait_workers()
    stage("strict-eighteen-fit-lock")
    checked("fit_lock", ["analysis/run_payoff_boundary_regimes.py", "lock", "--root", str(ROOT)])
    locked = read(ROOT/"payoffs/fit_lock.json")
    if not locked.get("complete") or locked.get("expected_models") != 18 or len(locked["models"]) != 18:
        raise ValueError("Every declared fit must be locked before held-out draws")
    PACKET["fit_lock_sha256"] = sha(ROOT/"payoffs/fit_lock.json")
    stage("robust-postfit-validation-before-reference")
    checked("postfit_validation", ["analysis/audit_payoff_boundary_validation.py", "--root", str(ROOT)])
    validation = read(ROOT/"audit/postfit_validation_diagnostic.json")
    if not validation.get("complete") or len(validation.get("records", [])) != 18:
        raise ValueError("Complete all-seed robust validation replay is required")
    PACKET["postfit_validation_sha256"] = sha(ROOT/"audit/postfit_validation_diagnostic.json")
    stage("fresh-native-reference-at-most-three-parallel-cases")
    reference_stage()
    stage("fresh-original-accuracy-all-new-and-legacy-seeds")
    checked("accuracy", ["analysis/evaluate_payoff_boundary_regimes.py", "evaluate", "--root", str(ROOT)])
    PACKET["original_accuracy_inventory"] = accuracy_inventory("comparison", 36, 10)
    stage("fresh-independent-neural-initial-integration")
    checked("independent_deployment", ["analysis/evaluate_payoff_boundary_deployment.py", "evaluate",
                                       "--root", str(ROOT)])
    PACKET["independent_deployment_inventory"] = accuracy_inventory("deployment", 12, 7)
    stage("accuracy-complete-ready-for-independent-audits")
    PACKET.update(complete=True, finished_unix=time.time(),
                  retained_accuracy_records=360+84, primary_accuracy_records=360,
                  final_timing_launched=False)
    save()
except BaseException as error:
    PACKET.update(stage="failed", failure={"type": type(error).__name__, "message": str(error),
                  "traceback": traceback.format_exc()}, finished_unix=time.time())
    save()
    raise
