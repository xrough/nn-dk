"""Time standalone native DK after a completed convergence experiment.

This removes the fine-Brownian-buffer and coupling audit overhead from oracle
timings. It measures cost at specified timesteps, not accuracy-matched speedup.
No trajectory results, training state, or primary experiment files are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nn_dk.convergence_config import ConvergenceConfig
from nn_dk.data import source_hash, write_json
from nn_dk.dk_training import load_coefficient_model
from nn_dk.native import simulate_native
from nn_dk.types import seeded_rng


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measure(config, paths, step_size, repetitions, model=None):
    context = config.context("uniform")
    repeats = []
    for repetition in range(repetitions):
        # Identical initial histograms and internally generated Brownian streams
        # for native/NN timing at the same step, batch size, and repetition.
        seed = int(seeded_rng(config.seed, "standalone-native-timing", step_size,
                              paths, repetition).integers(0, 2**63 - 1))
        started = time.perf_counter()
        batch = simulate_native(context, config.spec, config.times,
                                step_size, seed, n_paths=paths,
                                coefficients=None if model is None else model.sampling_coefficients,
                                ablation="oracle" if model is None else "both")
        wall_seconds = time.perf_counter() - started
        repeats.append({
            "repetition": repetition, "seed": seed,
            "external_wall_seconds": wall_seconds,
            **{key: batch.diagnostics[key] for key in (
                "total_seconds", "initialization_seconds", "simulation_seconds",
                "stopping_probability", "initial_stopping_probability",
                "mass_error", "minimum_mass")},
        })
    total = np.asarray([row["total_seconds"] for row in repeats])
    wall = np.asarray([row["external_wall_seconds"] for row in repeats])
    median = float(np.median(total))
    return {
        "case": "uniform", "method": "oracle" if model is None else "neural-both",
        "step_size": step_size, "paths": paths,
        "steps": round(config.horizon / step_size), "repetitions": repeats,
        "median_total_seconds": median,
        "minimum_total_seconds": float(total.min()),
        "maximum_total_seconds": float(total.max()),
        "median_external_wall_seconds": float(np.median(wall)),
        "median_initialization_seconds": float(np.median([
            row["initialization_seconds"] for row in repeats])),
        "median_simulation_seconds": float(np.median([
            row["simulation_seconds"] for row in repeats])),
        "throughput_paths_per_second_from_median_total": paths / median,
        "median_total_seconds_per_path": median / paths,
    }


def benchmark(run: Path, repetitions: int = 3, include_neural: bool = False) -> dict:
    """Return all repeat costs for one and one configured batch of paths."""
    manifest_path, config_path = run / "manifest.json", run / "config.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Standalone timing requires a complete run; avoid CPU contention with its primary experiment")
    config = ConvergenceConfig(**json.loads(config_path.read_text()))
    if config.config_hash != manifest["config_hash"]:
        raise ValueError("Saved configuration does not match the completed-run manifest")
    current_hash = source_hash()
    if current_hash != manifest["source_hash"]:
        raise ValueError("Current source differs from the completed run; benchmark its recorded source revision")
    if "uniform" not in config.cases:
        raise ValueError("The run must contain the noninteracting uniform case")
    if repetitions != 3:
        raise ValueError("This predefined timing diagnostic uses exactly three repetitions")
    torch.set_num_threads(1)
    records = []
    for step_size in config.step_sizes:
        for paths in (1, config.evaluation_batch_size):
            records.append(_measure(config, paths, step_size, repetitions))
    neural_records, skipped_checkpoints = [], []
    if include_neural:
        checkpoints = sorted((run / "checkpoints" / "uniform").glob("*.pt"))
        if not checkpoints:
            raise ValueError("No uniform neural coefficient checkpoints found")
        for path in checkpoints:
            tick = time.perf_counter()
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if not checkpoint.get("complete", False):
                skipped_checkpoints.append(str(path.resolve()))
                continue
            model = load_coefficient_model(path)
            config.spec.require_compatible(model.spec)
            model_load_seconds = time.perf_counter() - tick
            provenance = checkpoint["provenance"]
            for paths in (1, config.evaluation_batch_size):
                record = _measure(config, paths, min(config.step_sizes), repetitions, model)
                record.update({
                    "checkpoint": str(path.resolve()), "checkpoint_sha256": sha256(path),
                    "checkpoint_identity_hash": provenance["identity_hash"],
                    "checkpoint_kind": provenance["kind"], "training_mode": provenance["mode"],
                    "optimizer_seed": provenance["seed"], "width": provenance["width"],
                    "training_data_hash": provenance["train_hash"],
                    "validation_data_hash": provenance["validation_hash"],
                    "model_load_seconds_excluded_from_rollout_timing": model_load_seconds,
                })
                neural_records.append(record)
    return {
        "analysis": "standalone-native-dk-cpu-timing-v2",
        "scope": "Standalone stopped oracle and optionally neural solvers with histogram initialization and internal coarse-step RNG",
        "limitation": "Specified-timestep cost only; no matched-accuracy speedup claim",
        "implementation_scope": (
            "Actual shared simulation backend: even learned-both rollouts compute native coefficients "
            "before overriding them with network coefficients. Neural times include this overhead; "
            "these are not intrinsic isolated neural-versus-analytic coefficient evaluation costs."),
        "threading_scope": (
            "Torch intra-operation threads are set to one. NumPy/BLAS thread limits are not separately "
            "enforced; platform thread environment is recorded below. Desktop timing is descriptive."),
        "timing_convention": (
            "Three repeats, no discarded warmup; median simulator total includes histogram initialization, "
            "internal Brownian RNG, stepping and snapshots. External wall also includes returned-batch validation. "
            "No fine Brownian buffer or coupling aggregation is constructed. Oracle runs first at all configured "
            "steps. Optional neural checkpoints then run at the smallest configured step only, with model loading "
            "excluded and recorded separately. Repetition seeds match oracle at the same step and batch size."),
        "run_directory": str(run.resolve()), "run_status": manifest["status"],
        "run_config_hash": manifest["config_hash"], "run_source_hash": manifest["source_hash"],
        "current_source_hash": current_hash, "source_matches_run": True,
        "config_file_sha256": sha256(config_path), "manifest_file_sha256": sha256(manifest_path),
        "analysis_script_sha256": sha256(Path(__file__)),
        "n_cells": config.n_cells, "n_particles": config.n_particles,
        "horizon": config.horizon, "observation_times": config.times.tolist(),
        "platform": {
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "processor": platform.processor(),
            "python": sys.version, "logical_cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "dependencies": {name: importlib.metadata.version(name) for name in (
                "numpy", "scipy", "torch")},
        },
        "records": records,
        "include_neural": include_neural, "neural_records": neural_records,
        "skipped_incomplete_checkpoints": skipped_checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, help="Default: RUN/report/native_standalone_timing.json")
    parser.add_argument("--include-neural", action="store_true",
                        help="Also time every complete uniform neural checkpoint at the smallest configured step")
    args = parser.parse_args()
    try:
        report = benchmark(args.run, include_neural=args.include_neural)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output or args.run / "report" / "native_standalone_timing.json"
    write_json(output, report)
    print(json.dumps({"report": str(output), "configurations": len(report["records"]),
                      "neural_configurations": len(report["neural_records"]),
                      "repeats_per_configuration": 3}, indent=2))


if __name__ == "__main__":
    main()
