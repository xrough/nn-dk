"""Independent, resumable Brownian input banks with locked final references."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import source_hash
from .native import initialize_masses
from .path_config import PathConfig
from .path_numerics import aggregate_brownian, torch_euler
from .types import seeded_rng, stable_hash

PATH_BANK_KIND = "causal-density-physical-input-bank-v1"
_ROLES = ("train", "validation", "test")
_CHUNK_PATHS = 8


def _path(directory, role):
    if role not in _ROLES:
        raise ValueError("Path bank role must be train, validation or test")
    return Path(directory)/f"path_{role}.h5"


def _count(config, role):
    return {"train": config.train_paths, "validation": config.validation_paths,
            "test": config.test_paths}[role]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _test_lock(directory, config):
    """Verify all declared fits are immutable before even reading final inputs."""
    manifest_path = Path(directory)/"manifest.json"
    if not manifest_path.exists():
        raise ValueError("Final path inputs require a completed fit lock")
    manifest = json.loads(manifest_path.read_text())
    lock = manifest.get("fit_lock")
    if (manifest.get("config_hash") != config.config_hash
            or manifest.get("source_hash") != source_hash()
            or not manifest.get("test_unlocked_unix") or not isinstance(lock, dict)
            or lock.get("expected_models") != config.expected_models
            or not isinstance(lock.get("models"), list)
            or len(lock["models"]) != config.expected_models):
        raise ValueError("Final path inputs require a matching complete fit lock")
    expected = {(case, dt, seed) for case in config.cases for dt in config.step_sizes
                for seed in config.optimizer_seeds}
    identities = set()
    for item in lock["models"]:
        identity = (item.get("case"), item.get("dt"), item.get("seed"))
        if identity in identities or identity not in expected:
            raise ValueError("Final fit lock has duplicate or incompatible models")
        identities.add(identity)
        for key in ("checkpoint", "metadata"):
            path = Path(item.get(f"{key}_path", ""))
            if (not path.is_absolute() or not path.is_file()
                    or _sha(path) != item.get(f"{key}_sha256")):
                raise ValueError("A locked path checkpoint or metadata file changed")
        checkpoint = torch.load(item["checkpoint_path"], map_location="cpu", weights_only=True)
        provenance = checkpoint.get("provenance", {})
        if (not checkpoint.get("complete")
                or provenance.get("config_hash") != config.config_hash
                or provenance.get("source_hash") != source_hash()
                or (provenance.get("case"), provenance.get("dt"), provenance.get("seed")) != identity
                or provenance.get("target_id") != config.target_id(identity[0], identity[1])):
            raise ValueError("Final fit lock includes an incomplete or incompatible checkpoint")


def _expected(config, role):
    return {"kind": PATH_BANK_KIND, "config_hash": config.config_hash,
            "source_hash": source_hash(), "input_role": role,
            "base_step": min(config.step_sizes), "reference_step": config.reference_step,
            "horizon": config.horizon}


def _check_store(store, config, role):
    expected = _expected(config, role)
    if any(store.attrs.get(key) != value for key, value in expected.items()):
        raise ValueError("Path bank source, configuration, input role or resolution changed")
    if set(store) != set(config.cases):
        raise ValueError("Path bank case inventory differs from its declared configuration")
    size = _count(config, role)
    steps = round(config.horizon/min(config.step_sizes))
    for case in config.cases:
        group = store[case]
        shapes = {"initial": (size, config.n_cells),
                  "brownian": (size, steps, config.n_cells), "family_ids": (size,),
                  "complete": (size,), "row_hash": (size,)}
        if role == "test":
            shapes.update(reference_states=(size, steps+1, config.n_cells),
                          reference_stopped_at=(size,))
        if (set(group) != set(shapes)
                or any(group[name].shape != shape for name, shape in shapes.items())
                or group.attrs.get("problem") != config.problem(case)
                or group.attrs.get("context_id") != config.context(case).context_id):
            raise ValueError("Path bank datasets or physical case metadata are incompatible")
        for index in np.flatnonzero(group["complete"][:]):
            if _row_hash(group, int(index)) != group["row_hash"][index].decode():
                raise ValueError("A committed path bank row changed")


def _create(path, config, role):
    size = _count(config, role)
    steps, g = round(config.horizon/min(config.step_sizes)), config.n_cells
    with h5py.File(path, "w") as store:
        store.attrs.update(**_expected(config, role), config=json.dumps(config.to_dict(), sort_keys=True),
                           generation_seconds=0., complete=False,
                           brownian_convention="physical independent faces; variance equals step; nested by summation",
                           reference_not_learner_input=True)
        for case in config.cases:
            group = store.create_group(case)
            group.attrs.update(problem=config.problem(case), context_id=config.context(case).context_id)
            for key, shape, dtype in (("initial", (size, g), "f8"),
                                      ("brownian", (size, steps, g), "f8"),
                                      ("family_ids", (size,), "S24"),
                                      ("row_hash", (size,), "S64"),
                                      ("complete", (size,), "?")):
                chunks = (min(_CHUNK_PATHS, size), *shape[1:])
                group.create_dataset(key, shape, dtype=dtype, chunks=chunks,
                                     compression="gzip", shuffle=True)
            if role == "test":
                group.create_dataset("reference_states", (size, steps+1, g), dtype="f8",
                                     chunks=(min(_CHUNK_PATHS, size), min(steps+1, 128), g),
                                     compression="gzip", shuffle=True)
                group.create_dataset("reference_stopped_at", (size,), dtype="f8",
                                     chunks=(min(_CHUNK_PATHS, size),), compression="gzip")
        store.flush()


def _content_hash(group):
    digest = hashlib.sha256()
    for name in sorted(set(group)-{"complete"}):
        dataset = group[name]
        digest.update(name.encode())
        digest.update(str(dataset.shape).encode())
        digest.update(str(dataset.dtype).encode())
        for start in range(0, len(dataset), _CHUNK_PATHS):
            digest.update(np.ascontiguousarray(dataset[start:start+_CHUNK_PATHS]).tobytes())
    return digest.hexdigest()


def _row_hash(group, index):
    digest = hashlib.sha256()
    for name in sorted(set(group)-{"complete", "row_hash"}):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(group[name][index]).tobytes())
    return digest.hexdigest()


def _validate_completed(store, config):
    cases = {}
    for case in config.cases:
        group = store[case]
        if not np.all(group["complete"][:]):
            raise ValueError("Cannot read an incomplete path input bank")
        digest = _content_hash(group)
        if digest != group.attrs.get("content_hash"):
            raise ValueError("Path bank content hash mismatch")
        cases[case] = digest
    if not store.attrs.get("complete") or stable_hash(cases) != store.attrs.get("content_hash"):
        raise ValueError("Path bank completion or combined content hash mismatch")
    return cases


def _metadata(store, config, role, path, generated):
    completed = sum(int(np.sum(store[case]["complete"][:])) for case in config.cases)
    total = _count(config, role)*len(config.cases)
    return {"path": str(path), "role": role, "complete": bool(store.attrs.get("complete")),
            "completed_paths": completed, "total_paths": total, "generated_this_call": generated,
            "source_hash": str(store.attrs["source_hash"]), "config_hash": config.config_hash,
            "content_hash": str(store.attrs.get("content_hash", "")),
            "generation_seconds": float(store.attrs["generation_seconds"]),
            "reference_not_learner_input": True}


def _one_input(config, role, case, index):
    keys = ("density-path-bank-v1", config.config_hash, role, case, index)
    family = stable_hash([config.seed, *keys])[:24]
    initial_seed = int(seeded_rng(config.seed, *keys, "initial").integers(0, 2**63-1))
    initial = initialize_masses(config.context(case), config.spec, 1, initial_seed)[0]
    step = config.reference_step if role == "test" else min(config.step_sizes)
    brownian = seeded_rng(config.seed, *keys, "physical-face-noise").normal(
        0, np.sqrt(step), (round(config.horizon/step), config.n_cells))
    return initial, brownian, family


def ensure_path_bank(directory, config: PathConfig, role, deadline=None):
    """Generate independent inputs in path-atomic chunks; final references require a fit lock.

    Rows become visible as complete only AFTER all datasets have been flushed.
    Interrupted unfinished rows are regenerated from addressable per-path RNG
    streams. Completed replay verifies stored content without modifying bytes.
    """
    path = _path(directory, role)
    if role == "test":
        _test_lock(directory, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with h5py.File(path, "r") as store:
            _check_store(store, config, role)
            if store.attrs.get("complete"):
                _validate_completed(store, config)
                return _metadata(store, config, role, path, 0)
    else:
        _create(path, config, role)
    started, generated = time.perf_counter(), 0
    with h5py.File(path, "r+") as store:
        _check_store(store, config, role)
        previous_seconds = float(store.attrs["generation_seconds"])
        for case in config.cases:
            group = store[case]
            for start in range(0, _count(config, role), _CHUNK_PATHS):
                end = min(start+_CHUNK_PATHS, _count(config, role))
                unfinished = np.flatnonzero(~group["complete"][start:end])+start
                if not len(unfinished):
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    store.attrs["generation_seconds"] = previous_seconds+time.perf_counter()-started
                    store.flush()
                    return _metadata(store, config, role, path, generated)
                inputs = [_one_input(config, role, case, int(index)) for index in unfinished]
                initial = np.stack([item[0] for item in inputs])
                brownian = np.stack([item[1] for item in inputs])
                families = np.asarray([item[2] for item in inputs], dtype="S24")
                if role == "test":
                    with torch.no_grad():
                        reference = torch_euler(initial, brownian, config.context(case), config.spec,
                                                config.reference_step, config.problem(case))
                    factor = round(min(config.step_sizes)/config.reference_step)
                    group["reference_states"][unfinished] = reference["states"][:, ::factor].numpy()
                    group["reference_stopped_at"][unfinished] = reference["stopped_at"].numpy()
                    brownian = aggregate_brownian(brownian, factor)
                group["initial"][unfinished] = initial
                group["brownian"][unfinished] = brownian
                group["family_ids"][unfinished] = families
                group["row_hash"][unfinished] = np.asarray(
                    [_row_hash(group, int(index)) for index in unfinished], dtype="S64")
                store.flush()
                group["complete"][unfinished] = True
                store.flush()
                generated += len(unfinished)
            group.attrs["content_hash"] = _content_hash(group)
        store.attrs["content_hash"] = stable_hash({case: str(store[case].attrs["content_hash"])
                                                   for case in config.cases})
        store.attrs["complete"] = True
        store.attrs["generation_seconds"] = previous_seconds+time.perf_counter()-started
        store.flush()
        return _metadata(store, config, role, path, generated)


def load_path_bank(directory, config: PathConfig, role, case, dt):
    """Read only complete verified inputs, summing the common Brownian grid for ``dt``."""
    path = _path(directory, role)
    if case not in config.cases or dt not in config.step_sizes:
        raise ValueError("A declared path bank case and numerical step are required")
    if role == "test":
        _test_lock(directory, config)
    with h5py.File(path, "r") as store:
        _check_store(store, config, role)
        _validate_completed(store, config)
        group = store[case]
        ratio = dt/min(config.step_sizes)
        if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-10):
            raise ValueError("Fit steps must be nested in the bank's minimum fit step")
        factor = round(ratio)
        packet = {"initial": group["initial"][:],
                  "brownian": aggregate_brownian(group["brownian"][:], factor),
                  "family_ids": group["family_ids"][:].astype(str),
                  "content_hash": str(group.attrs["content_hash"]), "role": role}
        if role == "test":
            packet.update(reference_states=group["reference_states"][:, ::factor],
                          reference_stopped_at=group["reference_stopped_at"][:])
        return packet
