"""One-shot continuation of the explicitly authorized eight-hour path pilot.

Wait for the three exclusive seed writers to EXIT, verify their final records
and locks, restore every frozen fit, then resume the original numerical workflow.
This is a bounded continuation of one existing run, not a recurring automation.
No worker is restarted and no observed final-input unlock is rolled back.

Freeze source/config/helper hashes into execution/parallel_plan.json before
launching. --check-only reads and reports the wait gate without starting a CLI,
creating locks, writing production records or reading final inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from nn_dk.data import source_hash, write_json  # noqa: E402
from nn_dk.path_config import PathConfig  # noqa: E402
from nn_dk.path_study import _artifacts, _inventory  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
HELPERS = (
    "analysis/fit_path_seed_batch.py",
    "analysis/finish_path_batch.py",
    "analysis/plot_density_paths.py",
    "analysis/benchmark_path_solver.py",
    "analysis/archive_density_paths.py",
)
EXIT_CHECKPOINTED = 75


class ContinuationStopped(RuntimeError):
    """Stop without starting another stage; saved experiment state is retained."""


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def probe_pid(pid):
    """Only ESRCH establishes absence. EPERM is unknown, never a dead process."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("An actual positive integer PID is required")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {"pid": pid, "state": "exited"}
    except PermissionError as error:
        return {"pid": pid, "state": "permission-unknown", "reason": str(error)}
    return {"pid": pid, "state": "alive"}


def inspect_workers(directory, config, plan, *, pid_probe=probe_pid):
    """Read-only gate, callable independently for fake/live-process dry checks."""
    directory = Path(directory)
    expected_seeds = tuple(config.optimizer_seeds)
    if (len(expected_seeds) != 3 or set(plan.get("workers", {})) != {str(s) for s in expected_seeds}
            or config.expected_models != 15):
        raise ValueError("This continuation requires the declared three-seed, fifteen-fit pilot")
    sequential = pid_probe(plan.get("sequential_pid"))
    states, failures, unknown = [], [], sequential["state"] == "permission-unknown"
    for seed in expected_seeds:
        record_path = directory / "execution" / f"seed_worker_{seed}.json"
        lock = directory / f".path-seed-worker-{seed}.lock"
        if not record_path.is_file():
            states.append({"seed": seed, "state": "missing-record", "lock_present": lock.exists()})
            continue
        packet = _read(record_path)
        if (packet.get("seed") != seed or packet.get("source_hash") != plan.get("source_hash")
                or packet.get("config_hash") != config.config_hash
                or packet.get("threads") != 1 or packet.get("final_inputs_read") is not False
                or packet.get("sequential_workflow_must_be_stopped") is not True
                or not isinstance(packet.get("complete"), bool)):
            raise ValueError(f"Worker {seed} identity, completion type or physical-input contract changed")
        actual = pid_probe(packet.get("pid"))
        unknown |= actual["state"] == "permission-unknown"
        state = {"seed": seed, **actual, "lock_present": lock.exists(),
                 "complete": packet["complete"], "record_path": str(record_path),
                 "record_sha256": _sha(record_path)}
        if actual["state"] == "exited" and not packet["complete"]:
            failures.append(seed)
        if packet["complete"]:
            if (not all(_number(packet.get(k)) for k in
                        ("started_unix", "finished_unix", "wall_seconds", "process_cpu_seconds"))
                    or packet["wall_seconds"] < 0 or packet["process_cpu_seconds"] < 0):
                raise ValueError(f"Completed worker {seed} lacks its final execution/cost record")
        states.append(state)
    ready = (sequential["state"] == "exited" and len(states) == len(expected_seeds)
             and all(item.get("state") == "exited" and item.get("complete") is True
                     and item.get("lock_present") is False for item in states))
    status = ("permission-unknown" if unknown else "worker-incomplete" if failures
              else "ready" if ready else "waiting")
    return {"status": status, "ready": ready, "sequential_process": sequential,
            "workers": states, "failed_worker_seeds": failures,
            "final_inputs_read": False,
            "interpretation": "Process exit, complete records and absent locks are all required"}


def validate_frozen(directory, config, plan, plan_sha):
    """Validate the immutable plan and executable helpers; never execute plan text."""
    directory = Path(directory)
    if _sha(directory / "execution" / "parallel_plan.json") != plan_sha:
        raise ContinuationStopped("The frozen parallel continuation plan changed")
    manifest = _read(directory / "manifest.json")
    if (plan.get("source_hash") != source_hash()
            or manifest.get("source_hash") != plan.get("source_hash")
            or plan.get("config_hash") != config.config_hash
            or manifest.get("config_hash") != config.config_hash
            or plan.get("config_file_sha256") != _sha(directory / "config.json")):
        raise ContinuationStopped("Package/configuration hashes do not match the frozen parallel plan")
    helpers = plan.get("helper_sha256", {})
    if (not isinstance(helpers, dict) or any(name not in helpers for name in HELPERS)
            or plan.get("worker_script") != HELPERS[0]
            or plan.get("worker_script_sha256") != helpers.get(HELPERS[0])):
        raise ContinuationStopped("The parallel plan must freeze every continuation/helper SHA")
    for name in HELPERS:
        if helpers[name] != _sha(PROJECT / name):
            raise ContinuationStopped(f"Frozen helper changed: {name}")
    created = manifest.get("created_unix")
    recorded = manifest.get("elapsed_seconds")
    if (not _number(created) or created <= 0 or not _number(recorded) or recorded < 0
            or config.budget_seconds != 8 * 60 * 60):
        raise ContinuationStopped("Valid original execution metadata and the unchanged eight-hour budget are required")
    return manifest


def parallel_active_ledger(workers, *, maximum_launch_spread=60.):
    """Reconstruct overlapping monotonic worker cost from their final records.

    UTC is used only for small relative launch offsets. Each worker's elapsed
    interval is measured by time.monotonic(), so a later UTC jump alone does not
    change its native budget. Suspend behavior follows that clock, rather than
    an inference from UTC. A broad launch spread is ambiguous and
    is rejected instead of silently mixing UTC wall time into the budget.
    """
    if len(workers) != 3:
        raise ContinuationStopped("Three completed worker records are required for the active ledger")
    for worker in workers:
        if (worker.get("complete") is not True
                or not all(_number(worker.get(key)) for key in
                           ("started_unix", "finished_unix", "wall_seconds", "process_cpu_seconds"))
                or worker["wall_seconds"] < 0 or worker["process_cpu_seconds"] < 0):
            raise ContinuationStopped("The active ledger requires complete monotonic worker timing records")
    first = min(worker["started_unix"] for worker in workers)
    spread = max(worker["started_unix"] for worker in workers) - first
    if spread > maximum_launch_spread:
        raise ContinuationStopped("Worker launches are too widely separated for an unambiguous active parallel ledger")
    active = max(worker["started_unix"] - first + worker["wall_seconds"] for worker in workers)
    utc_span = max(worker["finished_unix"] for worker in workers) - first
    return {"active_parallel_span_seconds": float(active),
            "worker_launch_spread_seconds": float(spread),
            "parallel_utc_wall_span_seconds": float(utc_span),
            "worker_process_cpu_seconds": float(sum(worker["process_cpu_seconds"] for worker in workers)),
            "utc_wall_minus_active_span_seconds": float(utc_span - active),
            "clock_scope": "UTC gaps can reflect suspension or clock changes; their cause is not established by this ledger"}


def verify_inventory(directory, config, plan):
    """All fifteen actual restorable models, matched to each worker's final SHAs."""
    inventory = _inventory(directory, config)
    if inventory.get("expected_models") != 15 or len(inventory.get("models", [])) != 15:
        raise ContinuationStopped("The complete fifteen-model fit inventory is required")
    slots = {(item["case"], item["dt"], item["seed"]): item for item in inventory["models"]}
    expected = {(case, dt, seed) for case in config.cases for dt in config.step_sizes
                for seed in config.optimizer_seeds}
    if set(slots) != expected:
        raise ContinuationStopped("The restored inventory has duplicate or missing declared fits")
    for seed in config.optimizer_seeds:
        packet = _read(Path(directory) / "execution" / f"seed_worker_{seed}.json")
        expected_worker = {(case, dt) for case in config.cases if config.problem(case) == "dk"
                           for dt in config.step_sizes}
        seen = set()
        for item in packet.get("models", []):
            key = (item.get("case"), item.get("dt"))
            actual = slots.get((*key, seed))
            if (key not in expected_worker or key in seen or actual is None
                    or item.get("checkpoint_path") != actual["checkpoint_path"]
                    or item.get("checkpoint_sha256") != actual["checkpoint_sha256"]):
                raise ContinuationStopped(f"Worker {seed} final model inventory differs from restored fits")
            seen.add(key)
        if seen != expected_worker or packet.get("source_hash") != plan["source_hash"]:
            raise ContinuationStopped(f"Worker {seed} did not complete its entire declared fitting allocation")
    return inventory


def _emit(directory, message):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}"
    print(line, flush=True)
    with (Path(directory) / "execution" / "finish_continuation.log").open("a") as stream:
        stream.write(line + "\n")


def _remaining(deadline, stage):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ContinuationStopped(f"Eight-hour active execution budget exhausted before {stage}; no fresh stage started")
    return remaining


def _logged(directory, stage, command):
    path = Path(directory) / "execution" / f"finish_{stage}.log"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("MPLCONFIGDIR", "/private/tmp/nn-dk-mpl-cache")
    tick, active_start = time.time(), time.monotonic()
    _emit(directory, f"Starting {stage}; stdout/stderr: {path}")
    with path.open("a") as stream:
        stream.write(json.dumps({"started_unix": tick, "argv": list(map(str, command))}) + "\n")
        stream.flush()
        completed = subprocess.run(list(map(str, command)), cwd=PROJECT, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT, check=False)
    packet = {"kind": "one-shot-stage", "stage": stage,
              "returncode": completed.returncode, "started_unix": tick,
              "finished_unix": time.time(), "active_seconds": time.monotonic() - active_start,
              "log_path": str(path)}
    write_json(Path(directory) / "execution" / f"finish_{stage}.json", packet)
    _emit(directory, f"Finished {stage}, exit={completed.returncode}")
    return packet


def _assert_complete(directory, config, inventory):
    manifest = _read(Path(directory) / "manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("fit_lock") != inventory
            or manifest.get("artifact_lock") != _artifacts(directory)
            or manifest.get("continuous_time_accuracy_certified") is not False):
        raise ContinuationStopped("Native workflow has not produced a complete unchanged discrete-target result")
    records = _read(Path(directory) / "evaluations.json")
    expected = {(case, dt, seed) for case in config.cases for dt in config.step_sizes
                for seed in config.optimizer_seeds}
    if len(records) != 15 or {(r["case"], r["dt"], r["seed"]) for r in records} != expected:
        raise ContinuationStopped("The final report does not contain every declared fit")
    passed = sum(r["same_step"]["accuracy_gate"]["status"] == "resolved_within_tolerance"
                 and r["prediction_diagnostics"]["causality"] is True
                 and r["prediction_diagnostics"]["replay"] is True for r in records)
    if manifest.get("same_step_accuracy_passed") != (passed == 15):
        raise ContinuationStopped("Native manifest accuracy status differs from its complete final comparisons")
    return manifest, passed


def _archive_readme(destination, directory, config, passed):
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "README.md"
    if path.exists():
        return
    path.write_text(
        "# Project A: full density-path pilot\n\n"
        f"Evidence prepared from `{directory}`; {config.expected_models} fixed-weight fits and "
        f"{passed}/15 same-step Euler accuracy gates passed. Continuous-time stopped-DK "
        "and continuum-SPDE accuracy remain uncertified. The archive is complete only "
        "when its `archive_manifest.json` and listed files are present.\n\n"
        "- [Machine findings and cost scopes](findings.json)\n"
        "- [Complete report](path_results.md) and [all optimizer seeds](path_table.csv)\n"
        "- [Warm validation timing diagnostics](path_timings.json)\n"
        "- [Checkpoint metadata](checkpoints.json) and [checksums](archive_manifest.json)\n\n"
        "Per-fit training wall times overlap across three one-thread seed workers. "
        "CLI elapsed excludes those standalone intervals; monotonic active parallel span, "
        "worker process-CPU time and UTC wall span are reported separately. UTC clock gaps "
        "do not by themselves establish suspension or count against active execution. Workflow completion excludes later "
        "plotting, timing and archival. Inference timings exclude training, model loading, "
        "initialization, generation and bootstrap; restoration/initialization are separate. "
        "Validation strong point errors are not matched continuous-time or marginal-law certification.\n"
    )


def finish_batch(directory, destination, *, poll_seconds=30., check_only=False):
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    config = PathConfig.load(directory / "config.json")
    plan_path = directory / "execution" / "parallel_plan.json"
    plan, plan_sha = _read(plan_path), _sha(plan_path)
    manifest = validate_frozen(directory, config, plan, plan_sha)
    original_cli_active = float(manifest["elapsed_seconds"])
    wait_allowance = config.budget_seconds - original_cli_active
    if check_only:
        snapshot = inspect_workers(directory, config, plan)
        ledger = (parallel_active_ledger([
            _read(directory / "execution" / f"seed_worker_{seed}.json")
            for seed in config.optimizer_seeds]) if snapshot["ready"] else None)
        remaining = None if ledger is None else wait_allowance - ledger["active_parallel_span_seconds"]
        print(json.dumps({**snapshot, "wait_watchdog_active_seconds": wait_allowance,
                          "parallel_active_ledger": ledger,
                          "post_worker_active_seconds_remaining": remaining,
                          "utc_elapsed_is_not_budget_gate": True}, indent=2))
        return 0 if snapshot["ready"] and remaining > 0 else EXIT_CHECKPOINTED
    if not _number(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError("The bounded worker poll interval must be positive and at most 60 seconds")
    supervisor_lock = directory / "execution" / ".finish_path_batch.lock"
    descriptor = os.open(supervisor_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    stage, started, stages = "wait-workers", time.time(), []
    supervisor_active_start = time.monotonic()
    deadline = supervisor_active_start + wait_allowance
    ledger, post_worker_active_start = None, None
    status_path = directory / "execution" / "continuation_result.json"
    try:
        while True:
            _remaining(deadline, stage)
            current = validate_frozen(directory, config, plan, plan_sha)
            if (current["created_unix"] != manifest["created_unix"]
                    or current["elapsed_seconds"] != original_cli_active):
                raise ContinuationStopped("Original batch execution ledger changed while waiting for workers")
            snapshot = inspect_workers(directory, config, plan)
            if snapshot["status"] == "permission-unknown":
                raise ContinuationStopped("PID liveness is unknown due to permission; rerun with authorized process visibility")
            if snapshot["status"] == "worker-incomplete":
                raise ContinuationStopped(f"Exited incomplete/failed workers: {snapshot['failed_worker_seeds']}")
            if snapshot["ready"]:
                break
            _emit(directory, "Waiting for actual worker exit, complete final records and released locks")
            time.sleep(min(poll_seconds, _remaining(deadline, stage)))
        ledger = parallel_active_ledger([
            _read(directory / "execution" / f"seed_worker_{seed}.json")
            for seed in config.optimizer_seeds])
        remaining_after_workers = wait_allowance - ledger["active_parallel_span_seconds"]
        post_worker_active_start = time.monotonic()
        deadline = post_worker_active_start + remaining_after_workers
        _remaining(deadline, "post-worker continuation")
        _emit(directory, f"Active worker span={ledger['active_parallel_span_seconds']:.3f}s; "
              f"remaining active budget={remaining_after_workers:.3f}s; UTC wall span is recorded separately")
        _emit(directory, "All three workers and the original sequential process have exited; restoring fifteen fits")
        stage = "restore-inventory"
        _remaining(deadline, stage)
        inventory = verify_inventory(directory, config, plan)
        if not inspect_workers(directory, config, plan)["ready"]:
            raise ContinuationStopped("Worker gate changed before native workflow continuation")
        validate_frozen(directory, config, plan, plan_sha)
        stage = "cli"
        remaining = _remaining(deadline, stage)
        if _read(directory / "manifest.json").get("status") != "complete":
            result = _logged(directory, stage, [PROJECT / ".venv/bin/nn-dk", "path-solver",
                "--config", directory / "config.json", "--output", directory, "--stage", "all",
                "--budget-seconds", f"{remaining:.9f}"])
            stages.append(result)
            if result["returncode"] != 0:
                raise ContinuationStopped(f"Native workflow returned {result['returncode']}; preserve its exact resumable state")
        else:
            stages.append({"stage": "cli", "returncode": 0, "skipped": "already complete"})
        completed_manifest, passed = _assert_complete(directory, config, inventory)
        validate_frozen(directory, config, plan, plan_sha)
        finish_path = directory / "execution" / "finish.json"
        if finish_path.exists():
            finished = _read(finish_path)
            if (finished.get("kind") != "workflow-finished"
                    or finished.get("source_hash") != plan["source_hash"]
                    or finished.get("config_hash") != config.config_hash):
                raise ContinuationStopped("Existing workflow completion record has incompatible provenance")
        else:
            workflow_active = time.monotonic() - post_worker_active_start
            write_json(finish_path, {
                "kind": "workflow-finished", "finished_unix": time.time(),
                "source_hash": plan["source_hash"], "config_hash": config.config_hash,
                "parallel_plan_sha256": plan_sha, "expected_models": 15,
                "cli_status": completed_manifest["status"], "cli_exit": 0,
                "cli_stage": next((result for result in stages if result["stage"] == "cli"), None),
                "same_step_gates_passed": passed, "same_step_accuracy_passed": passed == 15,
                "continuous_time_accuracy_certified": False,
                **ledger,
                "recorded_cli_active_seconds_before_continuation": original_cli_active,
                "post_worker_active_seconds_to_workflow_finish": workflow_active,
                "active_execution_ledger_seconds": (original_cli_active + ledger["active_parallel_span_seconds"]
                                                     + workflow_active),
                "active_budget_target_seconds": config.budget_seconds,
                "cost_scope": "UTC creation-to-native-completion wall span includes any clock gap; the separate active ledger combines recorded CLI execution, overlapping monotonic worker span and post-worker monotonic continuation. Both exclude subsequent plotting, timing and archival. Worker CPU seconds cover standalone workers only.",
                "budget_scope": "No fresh stage after the eight-hour monotonic active execution budget. UTC elapsed alone is not a budget gate. Native atomic units can finish just beyond a budget; legitimate final-input unlocks are never erased.",
            })
        for stage, arguments in (
            ("plot", [PROJECT / "analysis/plot_density_paths.py", directory]),
            ("benchmark", [PROJECT / "analysis/benchmark_path_solver.py", "--run", directory,
                           "--output", directory / "report/path_timings.json"]),
        ):
            _remaining(deadline, stage)
            validate_frozen(directory, config, plan, plan_sha)
            if not inspect_workers(directory, config, plan)["ready"]:
                raise ContinuationStopped("Registered fitting processes are not quiet before postprocessing")
            result = _logged(directory, stage, [PROJECT / ".venv/bin/python", "-u", *arguments])
            stages.append(result)
            if result["returncode"] != 0:
                raise ContinuationStopped(f"{stage} returned {result['returncode']}; no further stage started")
        stage = "archive"
        _remaining(deadline, stage)
        validate_frozen(directory, config, plan, plan_sha)
        _assert_complete(directory, config, inventory)
        _archive_readme(destination, directory, config, passed)
        result = _logged(directory, stage, [PROJECT / ".venv/bin/python", "-u",
            PROJECT / "analysis/archive_density_paths.py", directory, destination])
        stages.append(result)
        if result["returncode"] != 0:
            raise ContinuationStopped(f"Archival returned {result['returncode']}")
        packet = {"kind": "one-shot-continuation-finished", "status": "complete",
                  "started_unix": started, "finished_unix": time.time(),
                  "supervisor_active_seconds": time.monotonic() - supervisor_active_start,
                  "parallel_active_ledger": ledger,
                  "post_worker_active_seconds": time.monotonic() - post_worker_active_start,
                  "source_hash": plan["source_hash"], "config_hash": config.config_hash,
                  "parallel_plan_sha256": plan_sha, "same_step_gates_passed": passed,
                  "continuous_time_accuracy_certified": False, "archive": str(destination),
                  "stages": stages, "recurring_or_scheduled": False}
        write_json(status_path, packet)
        _emit(directory, json.dumps(packet))
        return 0
    except (ContinuationStopped, ValueError, OSError) as error:
        current = _read(directory / "manifest.json")
        packet = {"kind": "one-shot-continuation-checkpointed", "status": "not-complete",
                  "stage": stage, "reason": str(error), "finished_unix": time.time(),
                  "supervisor_active_seconds": time.monotonic() - supervisor_active_start,
                  "parallel_active_ledger": ledger,
                  "source_hash": plan.get("source_hash"), "config_hash": config.config_hash,
                  "stages": stages, "native_manifest_status": current.get("status"),
                  "final_inputs_unlocked": bool(current.get("test_unlocked_unix")),
                  "state_preserved": True, "recurring_or_scheduled": False,
                  "resume_instruction": (
                      "Inspect this exact stage and stdout logs. Do not restart live workers or erase final unlocks. "
                      "Resume the original configured workflow in its next active execution batch if this one is exhausted. "
                      "Keep frozen files, seeds and recorded ledgers intact. A UTC gap alone does not establish active-budget exhaustion."
                  )}
        write_json(status_path, packet)
        _emit(directory, json.dumps(packet))
        return EXIT_CHECKPOINTED
    finally:
        supervisor_lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path,
                        default=PROJECT / "docs/experiments/artifacts/path_first_batch_2026-10-25")
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        return finish_batch(args.directory, args.archive, poll_seconds=args.poll_seconds,
                            check_only=args.check_only)
    except (ContinuationStopped, ValueError, OSError) as error:
        print(f"Continuation not started: {error}", file=sys.stderr, flush=True)
        return EXIT_CHECKPOINTED


if __name__ == "__main__":
    raise SystemExit(main())
