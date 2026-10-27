"""Archive completed, unchanged density-path evidence without duplicating large banks.

This performs no fitting, selection, simulation or confidence recalculation.
Full checkpoints, path banks and raw proposals remain in the original run.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import shutil
from pathlib import Path

import h5py

from nn_dk.data import source_hash
from nn_dk.path_config import PathConfig
from nn_dk.path_study import _artifacts


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, packet):
    Path(path).write_text(json.dumps(packet, indent=2, sort_keys=True)+"\n")


def active_parallel_span(workers):
    """Cost of overlapping native monotonic worker intervals, separate from UTC."""
    if not workers:
        return {"active_parallel_span_seconds": 0., "worker_launch_spread_seconds": 0.}
    if any(not all(isinstance(worker.get(key), (int, float))
                       and not isinstance(worker.get(key), bool) and math.isfinite(worker[key])
                       for key in ("started_unix", "wall_seconds"))
           or worker["wall_seconds"] < 0 for worker in workers):
        raise ValueError("Worker active-span records are missing finite monotonic durations")
    first = min(worker["started_unix"] for worker in workers)
    spread = max(worker["started_unix"] for worker in workers) - first
    if spread > 60.:
        raise ValueError("Worker launches are too far apart for the short-launch active parallel ledger")
    return {"active_parallel_span_seconds": max(
                worker["started_unix"]-first+worker["wall_seconds"] for worker in workers),
            "worker_launch_spread_seconds": spread}


def archive(directory, destination):
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    manifest = json.loads((directory/"manifest.json").read_text())
    config = PathConfig.load(directory/"config.json")
    if (manifest.get("status") != "complete" or manifest.get("source_hash") != source_hash()
            or manifest.get("config_hash") != config.config_hash
            or manifest.get("artifact_lock") != _artifacts(directory)):
        raise ValueError("Archive requires completed unchanged density-path artifacts")
    with (directory/"report/path_table.csv").open() as stream:
        table = list(csv.DictReader(stream))
    if len(table) != config.expected_models:
        raise ValueError("The report does not retain every configured model")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "preregistration.json", "manifest.json", "calibration.json"):
        shutil.copyfile(directory/name, destination/name)
    for name in ("path_results.md", "path_table.csv", "path_strong_errors.png",
                 "path_weak_errors.png", "density_realizations.png", "path_timings.json"):
        if (directory/"report"/name).exists():
            shutil.copyfile(directory/"report"/name, destination/name)
    for name in ("path_results.json",):
        data = (directory/"report"/name).read_bytes()
        (destination/(name+".gz")).write_bytes(gzip.compress(data, mtime=0))
    checkpoints = []
    for item in manifest["fit_lock"]["models"]:
        metadata = json.loads(Path(item["metadata_path"]).read_text())
        checkpoints.append({**item, "training": metadata})
    (destination/"checkpoint_training_history.json.gz").write_bytes(
        gzip.compress(json.dumps(checkpoints, sort_keys=True).encode(), mtime=0))
    write_json(destination/"checkpoints.json", [
        {**item, "training": {key: value for key, value in item["training"].items()
                             if key != "history"}} for item in checkpoints])
    execution = []
    for path in sorted((directory/"execution").glob("*.json")):
        target = destination/"execution"/path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        execution.append(json.loads(path.read_text()))
    workers = [packet for packet in execution if "process_cpu_seconds" in packet]
    finishes = [packet for packet in execution if packet.get("kind") == "workflow-finished"]
    active_parallel = active_parallel_span(workers)
    finished = max(finishes, key=lambda packet: packet["finished_unix"]) if finishes else None
    active_execution = None if finished is None else finished.get("active_execution_ledger_seconds")
    utc_elapsed = None if finished is None else finished["finished_unix"]-manifest["created_unix"]
    if (finished is not None and "active_parallel_span_seconds" in finished
            and not math.isclose(finished["active_parallel_span_seconds"],
                                 active_parallel["active_parallel_span_seconds"],
                                 rel_tol=0, abs_tol=1e-9)):
        raise ValueError("Workflow completion and worker active-span ledgers disagree")
    banks = {}
    for role in ("train", "validation", "test"):
        with h5py.File(directory/f"path_{role}.h5", "r") as bank:
            banks[role] = {
                "generation_seconds": float(bank.attrs["generation_seconds"]),
                "complete": bool(bank.attrs["complete"]),
                "paths": sum(len(bank[case]["initial"]) for case in config.cases),
                "scope": ("initial histograms, physical-noise generation, storage and hashing"
                          + (", fine Euler reference simulation" if role == "test" else ""))}
    def maximum(column):
        return max(float(row[column]) for row in table)
    findings = {
        "run_directory": str(directory), "source_hash": manifest["source_hash"],
        "config_hash": config.config_hash, "configured_models": config.expected_models,
        "cases": list(config.cases), "optimizer_seeds": list(config.optimizer_seeds),
        "step_sizes": list(config.step_sizes), "horizon": config.horizon,
        "paths_per_case": {"train": config.train_paths, "validation": config.validation_paths,
                           "test": config.test_paths},
        "same_step_accuracy_passed": manifest["same_step_accuracy_passed"],
        "continuous_time_accuracy_certified": False,
        "maximum_same_step_strong_rms": maximum("same_step_strong_supremum_rms"),
        "maximum_same_step_strong_ci_upper": maximum("same_step_strong_ci95_upper"),
        "maximum_same_step_mean_ci_endpoint": maximum("same_step_mean_maximum_pointwise_ci_endpoint"),
        "maximum_same_step_joint_covariance_ci_endpoint": maximum("same_step_covariance_maximum_pointwise_ci_endpoint"),
        "maximum_euler_fine_strong_rms": maximum("euler_fine_strong_supremum_rms"),
        "maximum_neural_fine_strong_rms": maximum("neural_fine_strong_supremum_rms"),
        "maximum_mass_error": maximum("maximum_mass_error"),
        "minimum_mass": min(float(row["minimum_mass"]) for row in table),
        "maximum_neural_stopped_paths": maximum("prediction_stopped_paths"),
        "maximum_neural_invalid_paths": maximum("prediction_invalid_paths"),
        "all_causality_checks_passed": all(row["causality_passed"] == "True" for row in table),
        "all_same_noise_replays_passed": all(row["same_noise_replay_passed"] == "True" for row in table),
        "training_seconds": sum(float(row["training_seconds"]) for row in table),
        "evaluation_seconds": sum(float(row["evaluation_seconds"]) for row in table),
        "input_and_reference_generation": banks,
        "total_generation_seconds": sum(bank["generation_seconds"] for bank in banks.values()),
        "recorded_workflow_seconds": manifest["elapsed_seconds"],
        "workflow_cost_scope": "cumulative recorded invocations, including resume/report replay",
        "standalone_worker_wall_span_seconds": (
            max(worker["finished_unix"] for worker in workers)
            -min(worker["started_unix"] for worker in workers) if workers else 0.),
        "standalone_worker_process_cpu_seconds": sum(worker["process_cpu_seconds"] for worker in workers),
        **active_parallel,
        "active_execution_ledger_seconds": active_execution,
        "active_budget_target_seconds": config.budget_seconds,
        "end_to_end_wall_seconds": utc_elapsed,
        "utc_wall_minus_active_ledger_seconds": (
            utc_elapsed-active_execution if utc_elapsed is not None and active_execution is not None else None),
        "parallel_cost_scope": "worker monotonic fit durations overlap and are excluded from manifest CLI elapsed; active parallel span uses bounded launch offsets plus those monotonic durations. CPU seconds cover standalone worker processes only",
        "clock_scope": "End-to-end UTC wall time includes any suspension or clock adjustment. A UTC/monotonic discrepancy does not establish its cause; UTC elapsed alone is not the active execution budget",
        "uncertainty_scope": "pointwise paired whole-family bootstrap; not simultaneous or marginal-law certification",
        "fine_reference_scope": "finite Euler difference, not a continuous stopping-bias bound",
        "evaluation_cost_scope": "includes oracle comparison, diagnostics, bootstrap and output; not neural latency"
    }
    write_json(destination/"findings.json", findings)
    write_json(destination/"environment.json", {
        "python": manifest["python"], "platform": manifest["platform"],
        "device": manifest["device"], "dtype": manifest["dtype"], "threads": manifest["threads"],
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "torch", "h5py", "matplotlib", "pytest", "ruff")}
    })
    contents = {str(path.relative_to(destination)): sha(path)
                for path in sorted(destination.rglob("*"))
                if path.is_file() and path.name != "archive_manifest.json"}
    write_json(destination/"archive_manifest.json", {
        "original_run": str(directory), "source_hash": manifest["source_hash"],
        "config_hash": config.config_hash, "files": contents,
        "large_local_artifacts": manifest["artifact_lock"]})
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(archive(args.directory, args.destination), indent=2))
