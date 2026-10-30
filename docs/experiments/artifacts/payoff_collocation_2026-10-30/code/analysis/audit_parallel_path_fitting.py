"""Small software audit: sequential fits versus the real per-seed workers.

This never generates final inputs and never touches production run directories.
Use a new output directory on each invocation. Physical inputs and declared
training allocations are identical in both comparison directories. Timing
values are intentionally excluded from numerical reproducibility comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for thread_variable in THREAD_ENV:
    os.environ[thread_variable] = "1"

import torch  # noqa: E402

from nn_dk.data import source_hash, write_json  # noqa: E402
from nn_dk.path_config import CASES, PathConfig  # noqa: E402
from nn_dk.path_data import ensure_path_bank, load_path_bank  # noqa: E402
from nn_dk.path_study import (  # noqa: E402
    _checkpoint,
    _development,
    calibrate_paths,
    prepare_path_run,
)
from nn_dk.path_training import fit_path, load_path_model  # noqa: E402
from nn_dk.types import stable_hash  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(directory):
    return {str(path.relative_to(directory)): sha(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def without_wall(value):
    if isinstance(value, dict):
        return {key: without_wall(item) for key, item in value.items()
                if key not in ("training_seconds", "step_seconds")}
    if isinstance(value, list):
        return [without_wall(item) for item in value]
    return value


def equal_nested(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(equal_nested(value, right[key]) for key, value in left.items()))
    if isinstance(left, (tuple, list)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(equal_nested(a, b) for a, b in zip(left, right, strict=True)))
    return left == right


def one_fit(directory, config, case, seed, deadline=None):
    dt = config.step_sizes[0]
    train = load_path_bank(directory, config, "train", case, dt)
    validation = load_path_bank(directory, config, "validation", case, dt)
    checkpoint = _checkpoint(directory, case, dt, seed)
    metadata = fit_path(config, case, dt, seed, checkpoint, train, validation, deadline)
    write_json(checkpoint.with_suffix(".json"), metadata)
    return metadata


def run(output):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = output.resolve()
    if output.exists():
        raise ValueError("Use a new software-audit output directory; existing results are preserved")
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    helper = root/"analysis"/"fit_path_seed_batch.py"
    script = Path(__file__).resolve()
    config = replace(PathConfig(), name="path-worker-software-repro",
                     seed=202610250904, n_cells=8, width=8, depth=2,
                     horizon=1/16, step_sizes=(1/128,), reference_step=1/256,
                     query_times=(1/32, 1/16), cases=CASES, optimizer_seeds=(0, 1),
                     train_paths=12, validation_paths=8, test_paths=8,
                     batch_size=5, adam_steps=40, lbfgs_steps=20,
                     lbfgs_max_iter=5, n_bootstrap=20, budget_seconds=300.)
    identity = {"audit_script_sha256": sha(script), "worker_helper_sha256": sha(helper),
                "package_source_hash": source_hash(), "config_hash": config.config_hash}
    archive = output/"source"
    archive.mkdir()
    shutil.copy2(script, archive/script.name)
    shutil.copy2(helper, archive/helper.name)
    write_json(output/"config.json", config.to_dict())
    result = {"role": "software-only sequential/worker reproducibility; no scientific accuracy claim",
              "identity": identity, "config": config.to_dict(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "platform": platform.platform(), "thread_environment": {
                                  name: os.environ[name] for name in THREAD_ENV},
                              "torch_threads": torch.get_num_threads(),
                              "torch_interop_threads": torch.get_num_interop_threads()},
              "control_validation": [], "comparisons": [], "complete": False}
    write_json(output/"result.json", result)
    sequential, parallel = output/"sequential", output/"parallel"
    for directory in (sequential, parallel):
        manifest = prepare_path_run(directory, config)
        calibrate_paths(directory, config)
        for role in ("train", "validation"):
            assert ensure_path_bank(directory, config, role)["complete"]
        manifest["status"] = "fitting"
        write_json(directory/"manifest.json", manifest)
        for case in ("heat", "additive"):
            for seed in config.optimizer_seeds:
                print(f"[audit] controls {directory.name} {case} seed={seed}", flush=True)
                metadata = one_fit(directory, config, case, seed)
                assert metadata["complete"]
                bank = load_path_bank(directory, config, "validation", case, config.step_sizes[0])
                checkpoint = _checkpoint(directory, case, config.step_sizes[0], seed)
                record = _development(directory, config, case, config.step_sizes[0], seed,
                                      bank, load_path_model(checkpoint))
                result["control_validation"].append({"directory": directory.name, **record})
                write_json(output/"result.json", result)
                if not record["within_tolerance"]:
                    raise RuntimeError("Software control gate failed; no workers or final inputs started")
    for case in (case for case in config.cases if config.problem(case) == "dk"):
        for seed in config.optimizer_seeds:
            print(f"[audit] sequential {case} seed={seed}", flush=True)
            assert one_fit(sequential, config, case, seed)["complete"]
    # Resume a genuine deadline-limited partial checkpoint in worker zero.
    # Timing controls only when it stops; the checkpoint's numerical state is
    # produced by the unmodified production training implementation.
    partial = one_fit(parallel, config, "dk_uniform", 0, time.monotonic()+.15)
    result["partial_resume"] = {key: partial[key] for key in (
        "complete", "adam_steps", "lbfgs_steps", "current_step", "identity_hash")}
    assert not partial["complete"] and partial["current_step"] > 0
    before = snapshot(parallel)
    workers, streams = {}, []
    lock_observations = {}
    logs = output/"logs"
    logs.mkdir()
    try:
        for seed in config.optimizer_seeds:
            stream = (logs/f"worker_{seed}.log").open("w")
            streams.append(stream)
            process = subprocess.Popen([sys.executable, str(helper), str(parallel),
                                        "--seed", str(seed)], cwd=root,
                                       env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
            workers[seed] = process
        while any(process.poll() is None for process in workers.values()):
            for seed, process in workers.items():
                lock = parallel/f".path-seed-worker-{seed}.lock"
                if lock.exists():
                    try:
                        lock_pid = int(lock.read_text())
                    except (FileNotFoundError, ValueError):
                        # Creation/write and final unlink are separate system
                        # calls; observe the next complete lock value instead.
                        continue
                    lock_observations[seed] = {"lock_pid": lock_pid, "child_pid": process.pid}
            time.sleep(.02)
    finally:
        for stream in streams:
            stream.close()
    result["workers"] = {str(seed): {"pid": process.pid, "exit_code": process.returncode}
                         for seed, process in workers.items()}
    result["observed_seed_locks"] = lock_observations
    assert all(process.returncode == 0 for process in workers.values())
    assert len({process.pid for process in workers.values()}) == len(workers)
    assert set(lock_observations) == set(config.optimizer_seeds)
    assert all(item["lock_pid"] == item["child_pid"] for item in lock_observations.values())
    after = snapshot(parallel)
    allowed_existing_changes = set()
    for case in (case for case in config.cases if config.problem(case) == "dk"):
        for seed in config.optimizer_seeds:
            path = _checkpoint(parallel, case, config.step_sizes[0], seed)
            allowed_existing_changes.update((str(path.relative_to(parallel)),
                                             str(path.with_suffix(".json").relative_to(parallel))))
    result["unexpected_preexisting_file_changes"] = [
        name for name, digest in before.items()
        if name not in allowed_existing_changes and after.get(name) != digest]
    assert not result["unexpected_preexisting_file_changes"]
    for seed in config.optimizer_seeds:
        record = json.loads((parallel/"execution"/f"seed_worker_{seed}.json").read_text())
        assert record["complete"] and record["seed"] == seed
        assert record["final_inputs_read"] is False
        assert {item["case"] for item in record["models"]} == set(CASES)-{"heat", "additive"}
        expected = {_checkpoint(parallel, case, config.step_sizes[0], seed).as_posix()
                    for case in CASES if config.problem(case) == "dk"}
        assert {item["checkpoint_path"] for item in record["models"]} == expected
        assert all(sha(item["checkpoint_path"]) == item["checkpoint_sha256"]
                   for item in record["models"])
    for case in config.cases:
        for seed in config.optimizer_seeds:
            left = torch.load(_checkpoint(sequential, case, config.step_sizes[0], seed),
                              map_location="cpu", weights_only=True)
            right = torch.load(_checkpoint(parallel, case, config.step_sizes[0], seed),
                               map_location="cpu", weights_only=True)
            fields = ("provenance", "model_state_hash", "best_state_hash", "model_state",
                      "best_state", "adam_state", "lbfgs_state", "torch_rng_state",
                      "shuffle_rng_state", "best_validation", "initial_validation",
                      "best_step", "adam_steps_done", "lbfgs_steps_done", "complete")
            mismatches = [field for field in fields if not equal_nested(left[field], right[field])]
            history_equal = equal_nested(without_wall(left["history"]), without_wall(right["history"]))
            result["comparisons"].append({"case": case, "seed": seed,
                "model_state_hash_equal": left["model_state_hash"] == right["model_state_hash"],
                "best_state_hash_equal": left["best_state_hash"] == right["best_state_hash"],
                "provenance_equal": left["provenance"] == right["provenance"],
                "history_equal_ignoring_wall_time": history_equal, "mismatching_fields": mismatches,
                "model_state_hash": left["model_state_hash"],
                "best_state_hash": left["best_state_hash"],
                "data_hash": left["provenance"]["data_hash"],
                "history_hash_without_wall_time": stable_hash(without_wall(left["history"]))})
    result["no_final_bank"] = all(not (directory/"path_test.h5").exists()
                                   and not json.loads((directory/"manifest.json").read_text()).get("fit_lock")
                                   and not json.loads((directory/"manifest.json").read_text()).get("test_unlocked_unix")
                                   for directory in (sequential, parallel))
    result["source_unchanged"] = identity["package_source_hash"] == source_hash()
    result["helper_unchanged"] = identity["worker_helper_sha256"] == sha(helper)
    result["passed"] = bool(result["no_final_bank"] and result["source_unchanged"]
                             and result["helper_unchanged"] and all(
                                 not item["mismatching_fields"]
                                 and item["history_equal_ignoring_wall_time"]
                                 for item in result["comparisons"]))
    result["complete"] = True
    write_json(output/"result.json", result)
    print(json.dumps({"passed": result["passed"], "model_pairs": len(result["comparisons"]),
                      "no_final_bank": result["no_final_bank"], "output": str(output)}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/path-worker-repro-2026-10-25"))
    raise SystemExit(run(parser.parse_args().output))
