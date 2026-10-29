"""Append-only trajectory records with independently addressable RNG streams."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from .config import ExperimentConfig, contexts_for
from .types import Context, TargetSpec, TrajectoryBatch, seeded_rng, stable_hash


def json_ready(value):
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hash() -> str:
    root = Path(__file__).parent
    return stable_hash({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))})


def prepare_run(directory: str | Path, config: ExperimentConfig) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config_hash"] != config.config_hash:
            raise ValueError("Run directory belongs to a different configuration")
        if manifest["source_hash"] != source_hash():
            raise ValueError("Source changed; use a new run directory to preserve provenance")
        return manifest
    dependencies = {}
    for package in ("nn-dk", "numpy", "scipy", "torch", "h5py", "matplotlib"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "uninstalled"
    manifest = {"schema_version": 1, "config": config.to_dict(), "config_hash": config.config_hash,
                "source_hash": source_hash(), "target": asdict(config.target),
                "queries": asdict(config.queries), "python": platform.python_version(),
                "platform": platform.platform(), "dependencies": dependencies,
                "created_unix": time.time(), "status": "prepared", "elapsed_seconds": 0.0,
                "interpretation": "Software pilot; no accuracy or speedup superiority assumed."}
    write_json(manifest_path, manifest)
    write_json(directory / "config.json", config.to_dict())
    return manifest


def generate_dataset(directory: str | Path, config: ExperimentConfig, deadline: float | None = None) -> dict:
    from .particles import simulate_particles

    directory = Path(directory)
    prepare_run(directory, config)
    start = time.perf_counter()
    generated = 0
    path = directory / "trajectories.h5"
    with h5py.File(path, "a") as store:
        if "config_hash" in store.attrs and store.attrs["config_hash"] != config.config_hash:
            raise ValueError("Dataset/config hash mismatch")
        store.attrs["config_hash"] = config.config_hash
        store.attrs["target"] = json.dumps(asdict(config.target))
        store.attrs["source_hash"] = source_hash()
        # Oracle provenance is physically separate from the learner path pool.
        oracle = store.require_group("oracle")
        oracle.attrs["interaction"] = "V=-kappa*cos(x)+a2*cos(2*x);U=0;independent Brownian"
        oracle.attrs["reference_dt"] = config.dt
        learner = store.require_group("learner")
        learner.attrs["information_regime"] = "K"
        seen = set()
        for split, n_reps in (("train", config.train_replicates), ("validation", config.validation_replicates), ("test", config.test_replicates)):
            split_group = learner.require_group(split)
            for context in contexts_for(config, split):
                if context.context_id in seen:
                    raise ValueError("Context leakage across splits")
                seen.add(context.context_id)
                group = split_group.require_group(context.context_id)
                if "states" not in group:
                    group.attrs["context"] = json.dumps(asdict(context))
                    group.create_dataset("times", data=config.times)
                    group.create_dataset("states", shape=(n_reps, len(config.times), config.n_cells), dtype="f8", chunks=(1, len(config.times), config.n_cells), compression="gzip", fillvalue=np.nan)
                    group.create_dataset("completed", data=np.zeros(n_reps, dtype=bool))
                    group.create_dataset("seeds", shape=(n_reps,), dtype="u8")
                    group.create_dataset("generation_seconds", data=np.zeros(n_reps))
                    group.create_dataset("valid", data=np.zeros(n_reps, dtype=bool))
                    group.create_dataset("failure_times", data=np.full(n_reps, np.nan))
                    group.create_dataset("trajectory_ids", data=np.array([f"{split}/{context.context_id}/{r}" for r in range(n_reps)], dtype=h5py.string_dtype()))
                for r in range(n_reps):
                    if group["completed"][r]:
                        continue
                    if deadline is not None and time.monotonic() >= deadline:
                        store.flush()
                        return {"complete": False, "generated_this_call": generated,
                                "seconds_this_call": time.perf_counter() - start}
                    seed = int(seeded_rng(config.seed, "trajectory", split, context.context_id, r).integers(0, 2 ** 63, dtype=np.int64))
                    tick = time.perf_counter()
                    batch = simulate_particles(context, config.target, config.times, config.dt, seed, n_paths=1)
                    group["states"][r] = batch.states[0]
                    group["seeds"][r] = seed
                    group["valid"][r] = batch.valid[0]
                    group["failure_times"][r] = batch.failure_times[0]
                    group["generation_seconds"][r] = time.perf_counter() - tick
                    # Last write is the commit marker; incomplete work can safely rerun.
                    group["completed"][r] = True
                    store.flush()
                    generated += 1
        store.attrs["complete"] = True
    return {"complete": True, "generated_this_call": generated,
            "seconds_this_call": time.perf_counter() - start}


def load_split(directory: str | Path, split: str) -> list[tuple[Context, TrajectoryBatch]]:
    result = []
    with h5py.File(Path(directory) / "trajectories.h5", "r") as store:
        target = TargetSpec(**json.loads(store.attrs["target"]))
        for context_id in sorted(store[f"learner/{split}"]):
            group = store[f"learner/{split}/{context_id}"]
            if not np.all(group["completed"][:]):
                raise ValueError("Incomplete trajectory group; resume generation first")
            context = Context(**json.loads(group.attrs["context"]))
            batch = TrajectoryBatch(group["times"][:], group["states"][:], target,
                                    group["valid"][:], group["failure_times"][:],
                                    {"context_id": context_id, "independent_paths": len(group["states"]),
                                     "trajectory_ids": group["trajectory_ids"].asstr()[:].tolist(),
                                     "generation_seconds": float(group["generation_seconds"][:].sum())})
            result.append((context, batch))
    return result


def generation_cost(directory: str | Path) -> dict:
    cost = {}
    with h5py.File(Path(directory) / "trajectories.h5", "r") as store:
        for split in ("train", "validation", "test"):
            cost[split] = sum(float(g["generation_seconds"][:].sum()) for g in store[f"learner/{split}"].values())
    return cost
