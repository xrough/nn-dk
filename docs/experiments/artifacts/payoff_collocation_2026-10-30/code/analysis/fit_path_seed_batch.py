"""Fit one declared optimizer seed, with exclusive checkpoint ownership.

Stop the sequential path-solver invocation before starting these workers.
Run distinct seeds in separate processes; wait for all workers to exit before
resuming the original path-solver command. Workers never open final inputs or
change the manifest/design. They reuse the exact same fit_path implementation.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import torch  # noqa: E402

from nn_dk.data import source_hash, write_json  # noqa: E402
from nn_dk.path_config import PathConfig  # noqa: E402
from nn_dk.path_data import load_path_bank  # noqa: E402
from nn_dk.path_study import _checkpoint, _development, _sha, prepare_path_run  # noqa: E402
from nn_dk.path_training import fit_path, load_path_model  # noqa: E402


def fit_seed(directory, seed):
    directory = Path(directory).resolve()
    if not (directory/"manifest.json").is_file():
        raise ValueError("Workers require an existing prepared run")
    config = PathConfig.load(directory/"config.json")
    manifest = prepare_path_run(directory, config)
    if (seed not in config.optimizer_seeds or manifest.get("fit_lock")
            or manifest.get("status") not in ("fitting", "budget-exhausted")):
        raise ValueError("Workers require a declared seed and an unlocked fitting stage")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for case in (c for c in ("heat", "additive") if c in config.cases):
        for dt in config.step_sizes:
            validation = load_path_bank(directory, config, "validation", case, dt)
            for control_seed in config.optimizer_seeds:
                model = load_path_model(_checkpoint(directory, case, dt, control_seed))
                if not _development(directory, config, case, dt, control_seed,
                                    validation, model)["within_tolerance"]:
                    raise ValueError("Every declared control must pass before DK workers start")
    lock = directory/f".path-seed-worker-{seed}.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    start, cpu_start = time.monotonic(), time.process_time()
    packet = {"seed": seed, "source_hash": source_hash(), "config_hash": config.config_hash,
              "started_unix": time.time(), "pid": os.getpid(), "threads": 1,
              "sequential_workflow_must_be_stopped": True, "final_inputs_read": False,
              "models": [], "complete": False}
    record_path = directory/"execution"/f"seed_worker_{seed}.json"
    remaining_budget = config.budget_seconds-manifest["elapsed_seconds"]
    deadline = start+max(0., remaining_budget)
    try:
        write_json(record_path, packet)
        for case in (c for c in config.cases if config.problem(c) == "dk"):
            for dt in config.step_sizes:
                print(f"[worker {seed}] fit {case} dt={dt:g}", flush=True)
                train = load_path_bank(directory, config, "train", case, dt)
                validation = load_path_bank(directory, config, "validation", case, dt)
                path = _checkpoint(directory, case, dt, seed)
                metadata = fit_path(config, case, dt, seed, path, train, validation, deadline)
                write_json(path.with_suffix(".json"), metadata)
                if not metadata["complete"]:
                    return False
                packet["models"].append({"case": case, "dt": dt,
                                         "checkpoint_path": str(path),
                                         "checkpoint_sha256": _sha(path)})
                write_json(record_path, packet)
        packet["complete"] = True
        return True
    finally:
        packet.update(finished_unix=time.time(), wall_seconds=time.monotonic()-start,
                      process_cpu_seconds=time.process_time()-cpu_start)
        write_json(record_path, packet)
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    raise SystemExit(0 if fit_seed(args.directory, args.seed) else 75)
