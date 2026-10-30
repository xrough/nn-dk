"""Resumable native trajectory-family pools and uncensored Euler proposals."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from .convergence_config import ConvergenceConfig
from .data import source_hash, write_json
from .native_types import DKTrainingBatch
from .types import seeded_rng


def prepare_convergence_run(directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("workflow") != "native-dk-convergence-v1" or manifest["config_hash"] != config.config_hash or manifest["source_hash"] != source_hash():
            raise ValueError("Convergence run source/configuration mismatch; use a new output directory")
        return manifest
    manifest = {"workflow": "native-dk-convergence-v1", "status": "prepared", "config": config.to_dict(),
                "config_hash": config.config_hash, "source_hash": source_hash(), "target": asdict(config.spec),
                "query_bank": asdict(config.queries), "python": platform.python_version(),
                "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "torch", "h5py", "matplotlib")},
                "device": "cpu", "created_unix": time.time(), "elapsed_seconds": 0.,
                "scope": "Weak errors for stopped native DK at fixed grid; no continuum convergence claim"}
    write_json(path, manifest)
    write_json(directory / "config.json", config.to_dict())
    return manifest


def generate_native_pool(directory, config: ConvergenceConfig, case, deadline=None):
    from .native import native_coefficients, simulate_native
    from .observations import incidence_matrix

    directory = Path(directory)
    prepare_convergence_run(directory, config)
    context, spec = config.context(case), config.spec
    h5path = directory / f"pool_{case}.h5"
    b = incidence_matrix(spec.n_cells)
    with h5py.File(h5path, "a") as store:
        store.attrs["config_hash"] = config.config_hash
        store.attrs["source_hash"] = source_hash()
        store.attrs["context"] = json.dumps(asdict(context))
        store.attrs["target"] = json.dumps(asdict(spec))
        for split, count in (("train", max(config.train_path_counts)), ("validation", config.validation_paths), ("test", config.coefficient_test_paths)):
            group = store.require_group(split)
            if "states" not in group:
                shape = (count, config.states_per_path, spec.n_cells)
                for key in ("states", "proposals", "oracle_drift", "oracle_face_amplitudes"):
                    parent = group.require_group("oracle") if key.startswith("oracle_") else group
                    parent.create_dataset(key.replace("oracle_", ""), shape=shape, dtype="f8", chunks=(1, config.states_per_path, spec.n_cells), compression="gzip")
                group.create_dataset("active", shape=shape[:2], dtype="?", chunks=(1, config.states_per_path))
                group.create_dataset("proposal_crosses", shape=shape[:2], dtype="?")
                group.create_dataset("completed", data=np.zeros(count, dtype=bool))
                group.create_dataset("seconds", data=np.zeros(count))
                group.create_dataset("family_ids", data=np.asarray(
                    [f"{config.config_hash}/{case}/{split}/{i}" for i in range(count)], dtype=h5py.string_dtype()))
                group.create_dataset("batch_seed", shape=(count,), dtype="i8")
                group.create_dataset("stream_index", shape=(count,), dtype="i8")
                group.attrs["proposal_dt"] = config.proposal_dt
                group.create_dataset("times", data=config.sample_times)
            for start in range(0, count, config.trajectory_batch_size):
                end = min(start + config.trajectory_batch_size, count)
                if np.all(group["completed"][start:end]):
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    return {"complete": False, "case": case}
                ids = tuple(group["family_ids"].asstr()[start:end])
                seed = int(seeded_rng(config.seed, "native-pool", case, split, start).integers(0, 2 ** 63))
                tick = time.perf_counter()
                batch = simulate_native(context, spec, config.sample_times, config.reference_dt,
                                         seed, n_paths=end-start, family_ids=ids)
                states = batch.states.reshape(-1, spec.n_cells)
                active = (batch.stopped_at[:, None] > config.sample_times[None]).reshape(-1)
                coeff = native_coefficients(states, context, spec)
                z = seeded_rng(config.seed, "native-proposals", case, split, start).normal(size=states.shape)
                proposals = states + config.proposal_dt * coeff.drift + np.sqrt(config.proposal_dt / context.n_particles) * (z * coeff.face_amplitudes) @ b.T
                proposals[~active] = states[~active]
                shape = (end-start, config.states_per_path, spec.n_cells)
                group["states"][start:end] = batch.states
                group["proposals"][start:end] = proposals.reshape(shape)
                group["active"][start:end] = active.reshape(shape[:2])
                group["proposal_crosses"][start:end] = ((proposals.min(-1) <= spec.mass_stop) & active).reshape(shape[:2])
                group["oracle/drift"][start:end] = coeff.drift.reshape(shape)
                group["oracle/face_amplitudes"][start:end] = coeff.face_amplitudes.reshape(shape)
                group["seconds"][start:end] = (time.perf_counter() - tick) / (end-start)
                group["batch_seed"][start:end] = seed
                group["stream_index"][start:end] = np.arange(end-start)
                group["completed"][start:end] = True
                store.flush()
        store.attrs["complete"] = True
    return {"complete": True, "case": case}


def load_native_pool(directory, config, case, split, n_paths=None):
    context, spec = config.context(case), config.spec
    with h5py.File(Path(directory) / f"pool_{case}.h5", "r") as store:
        if store.attrs["config_hash"] != config.config_hash or store.attrs["source_hash"] != source_hash():
            raise ValueError("Native pool provenance mismatch")
        group = store[split]
        count = len(group["states"]) if n_paths is None else n_paths
        if not 1 <= count <= len(group["states"]) or not np.all(group["completed"][:count]):
            raise ValueError("Native trajectory pool incomplete; resume generation")
        states = group["states"][:count].reshape(-1, spec.n_cells)
        ids = group["family_ids"].asstr()[:count]
        families = tuple(identifier for identifier in ids for _ in config.sample_times)
        samples = tuple(f"{identifier}/state{j}" for identifier in ids for j in range(config.states_per_path))
        return DKTrainingBatch(states, spec, context, families, samples,
                               group["proposals"][:count].reshape(-1, spec.n_cells),
                               float(group.attrs["proposal_dt"]), group["active"][:count].reshape(-1))


def native_generation_cost(directory, case):
    with h5py.File(Path(directory) / f"pool_{case}.h5", "r") as store:
        return {split: float(store[split]["seconds"][:].sum()) for split in ("train", "validation", "test")}
