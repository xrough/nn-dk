"""Locked, resumable fresh references for broad density-path/statistics tasks.

No outcome or held-out conditional anchor is materialized before all declared
payoff fits are complete and independently restored. Every stopped realization
is retained. Fine Euler polygons define killed integrals; a separate fixed
common-grid polygon defines reusable path functionals. Coupled refinements
audit numerical reference bias without certifying a continuous stopping law.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import i0  # noqa: E402

from nn_dk.data import source_hash  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.payoff_learning import PayoffBank, load_payoff_model  # noqa: E402
from nn_dk.types import Context, seeded_rng, stable_hash  # noqa: E402

_probe_path = Path(__file__).with_name("probe_path_workloads.py")
_module_spec = importlib.util.spec_from_file_location("_reference_classical_probe", _probe_path)
_probe = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_probe)

KIND = "locked-broad-native-path-payoff-reference-v1"
STREAM = "fresh-path-task-comparison-2026102611-v1"


@dataclass(frozen=True)
class ReferenceConfig:
    seed: int = 2026102611
    n_cells: int = 32
    n_particles: int = 2048
    diffusion: float = 1.
    kappa: float = .75
    a2: float = .1
    rho_stop: float = .01/(2*np.pi)
    horizon: float = 1/16
    query_times: tuple[float, ...] = (1/64, 1/32, 1/16)
    reference_step: float = 2**-13
    reference_factors: tuple[int, ...] = (1, 2, 4)
    common_step: float = 2**-8
    model_step: float = 2**-10
    cases: tuple[str, ...] = ("uniform", "nonuniform")
    conditional_anchors: int = 2
    paths_per_group: int = 16384
    refinement_paths: int = 1024
    chunk_paths: int = 64
    expected_models: int = 12

    def __post_init__(self):
        for name in ("query_times", "reference_factors", "cases"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("seed", "n_cells", "n_particles", "conditional_anchors",
                     "paths_per_group", "refinement_paths", "chunk_paths", "expected_models"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (0 if name == "seed" else 1):
                raise ValueError("Positive integer allocation/settings required")
        if (self.n_cells < 10 or self.n_cells % 2 or self.refinement_paths > self.paths_per_group
                or not self.cases or len(set(self.cases)) != len(self.cases)
                or any(case not in ("uniform", "nonuniform") for case in self.cases)):
            raise ValueError("Even native grid, declared profiles, and valid refinement allocation required")
        if (len(self.reference_factors) < 2 or self.reference_factors[0] != 1
                or any(type(factor) is not int or factor < 1 for factor in self.reference_factors)
                or any(b % a or b <= a for a, b in zip(
                    self.reference_factors[:-1], self.reference_factors[1:]))):
            raise ValueError("Strictly nested integer reference refinement factors starting at one required")
        values = (self.diffusion, self.kappa, self.a2, self.rho_stop, self.horizon,
                  self.reference_step, self.common_step, self.model_step, *self.query_times)
        if (not np.isfinite(values).all() or self.diffusion <= 0 or self.horizon <= 0
                or min(self.reference_step, self.common_step, self.model_step) <= 0
                or not self.query_times or self.query_times[-1] != self.horizon
                or min(self.query_times) <= 0 or any(b <= a for a, b in zip(
                    self.query_times[:-1], self.query_times[1:]))):
            raise ValueError("Finite positive times and diffusion required")
        for step in (self.reference_step, self.common_step, self.model_step):
            if not np.isclose(self.horizon/step, round(self.horizon/step), atol=1e-10, rtol=0):
                raise ValueError("Horizon must lie on each declared numerical grid")
            if not np.allclose(np.asarray(self.query_times)/step,
                               np.rint(np.asarray(self.query_times)/step), atol=1e-10, rtol=0):
                raise ValueError("All query times must lie on each numerical grid")
        for ratio in (self.model_step/self.reference_step, self.common_step/self.model_step):
            if ratio < 1 or not np.isclose(ratio, round(ratio), atol=1e-10, rtol=0):
                raise ValueError("Reference/model/common steps must be nested")
        velocity = abs(self.kappa)+2*abs(self.a2)
        if (self.reference_step*(2*self.diffusion/self.spec.h**2+velocity/self.spec.h) > 1
                or velocity*self.spec.h > 2*self.diffusion):
            raise ValueError("Native reference centered Euler stability conditions violated")

    @property
    def spec(self):
        return NativeDKSpec(n_cells=self.n_cells, rho_stop=self.rho_stop)

    def context(self, case):
        if case not in self.cases:
            raise ValueError("Profile not in the configured reference plan")
        return Context(cos_coeffs=(.2 if case == "nonuniform" else 0., 0., 0.),
                       kappa=self.kappa, a2=self.a2, diffusion=self.diffusion,
                       n_particles=self.n_particles)

    @property
    def config_hash(self):
        return stable_hash(asdict(self))

    @property
    def group_keys(self):
        return tuple(f"{case}__conditional__{anchor}" for case in self.cases
                     for anchor in range(self.conditional_anchors))+tuple(
            f"{case}__ensemble__0" for case in self.cases)

    @property
    def common_times(self):
        return np.arange(round(self.horizon/self.common_step)+1)*self.common_step


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


@lru_cache(maxsize=16)
def profile_probabilities(cfg, case):
    """Native histogram probabilities from audited smooth-profile quadrature."""
    def calculate(order):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        centers = (np.arange(cfg.n_cells)+.5)*cfg.spec.h
        positions = centers[:, None]+cfg.spec.h/2*nodes
        alpha = .2 if case == "nonuniform" else 0.
        return cfg.spec.h/2*(np.exp(alpha*np.cos(positions)) @ weights)/(2*np.pi*i0(alpha))
    coarse, fine = calculate(32), calculate(64)
    if not np.isclose(fine.sum(), 1, atol=2e-14, rtol=0):
        raise ValueError("Initial-law cell quadrature failed normalization")
    return fine, {"method": "32/64-point-cell-Gauss-Legendre-with-analytic-von-Mises-normalizer",
                  "normalization_error": float(abs(fine.sum()-1)),
                  "maximum_probability_refinement_difference": float(abs(fine-coarse).max()),
                  "probabilities_sha256": _array_sha(fine)}


def _identity(cfg):
    return {"kind": KIND, "config_hash": cfg.config_hash, "config": asdict(cfg),
            "source_hash": source_hash(), "script_sha256": _sha(__file__),
            "classical_helper_sha256": _sha(_probe_path),
            "native_spec_id": cfg.spec.target_id, "stream": STREAM,
            "numpy_version": np.__version__, "h5py_version": h5py.__version__,
            "rng": "NumPy-default-PCG64-with-sha256-addressed-SeedSequence-v1",
            "payoff_banks": {bank.name: bank.to_dict() for bank in _banks(cfg)}}


def _banks(cfg):
    return (PayoffBank.terminal_high_modes(cfg.spec, cfg.horizon),
            PayoffBank.mixed_terminal_integral(cfg.spec, cfg.horizon))


def prepare_reference(directory, cfg=ReferenceConfig()):
    """Persist a predeclared plan only; this never creates anchors or outcomes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"manifest.json"
    identity = _identity(cfg)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("identity") != json.loads(json.dumps(identity)):
            raise ValueError("Reference configuration/source/helper/payoff identity changed")
        return manifest
    manifest = {"identity": identity, "status": "waiting-for-complete-fit-lock",
                "groups": list(cfg.group_keys), "committed_chunks": 0,
                "generation_seconds": 0., "continuous_reference_bias": "unresolved",
                "sampling_accuracy_certificate": "unresolved-fixed-allocation"}
    _json(path, manifest)
    return manifest


def _fit_lock(value, cfg):
    lock = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    if not isinstance(lock, dict) or lock.get("expected_models") != cfg.expected_models:
        raise ValueError("All declared complete payoff fits are required before reference unlock")
    models = lock.get("models", [])
    if len(models) != cfg.expected_models:
        raise ValueError("Complete payoff model inventory has the wrong allocation")
    seen, normalized = set(), []
    for item in models:
        path = Path(item["checkpoint_path"]).resolve()
        if (str(path) in seen or item.get("complete") is not True
                or item.get("source_hash") != source_hash()
                or not path.is_file() or item.get("checkpoint_sha256") != _sha(path)):
            raise ValueError("Incomplete, duplicate, mutated or source-incompatible payoff fit lock")
        model = load_payoff_model(path)
        if not model.complete or model.provenance.get("source_hash") != source_hash():
            raise ValueError("Locked payoff model did not restore as complete at the current source")
        seen.add(str(path))
        normalized.append({"checkpoint_path": str(path), "checkpoint_sha256": _sha(path),
                           "source_hash": item["source_hash"], "complete": True})
    return {"expected_models": cfg.expected_models, "models": normalized,
            "inventory_hash": stable_hash(normalized)}


def _shapes(cfg):
    n, g = cfg.paths_per_group, cfg.n_cells
    q, c, k = len(cfg.query_times), len(cfg.common_times), round(cfg.horizon/cfg.model_step)
    a, f = cfg.refinement_paths, len(cfg.reference_factors)
    many = (c-1)*2*(g//2-1)
    return {"initial": ((n, g), "f8"), "family_ids": ((n,), "S64"),
            "base_brownian_sha256": ((n,), "S64"), "drawn_brownian_sha256": ((n,), "S64"),
            "model_increments": ((n, k, g), "f8"), "common_paths": ((n, c, g), "f8"),
            "bank_a": ((n, q, 4), "f8"), "bank_b": ((n, q, 4), "f8"),
            "bank_c": ((n, q, 4), "f8"), "many_observables": ((n, many), "f8"),
            "workloads": ((n, 3), "f8"), "stopped_at": ((n,), "f8"),
            "audit_bank_a": ((a, f, q, 4), "f8"), "audit_bank_b": ((a, f, q, 4), "f8"),
            "audit_bank_c": ((a, f, q, 4), "f8"),
            "audit_common_paths": ((a, f, c, g), "f8"),
            "audit_many_observables": ((a, f, many), "f8"),
            "audit_workloads": ((a, f, 3), "f8"), "audit_stopped_at": ((a, f), "f8"),
            "audit_brownian_sha256": ((a, f), "S64")}


def _group_plan(key):
    case, role, anchor = key.split("__")
    return case, role, int(anchor)


def _create_store(path, cfg, identity, lock):
    with h5py.File(path, "w") as out:
        out.attrs["identity_json"] = json.dumps(identity, sort_keys=True)
        out.attrs["fit_lock_json"] = json.dumps(lock, sort_keys=True)
        out.create_dataset("common_times", data=cfg.common_times)
        out.create_dataset("query_times", data=cfg.query_times)
        out.create_dataset("reference_factors", data=cfg.reference_factors)
        chunks = (cfg.paths_per_group+cfg.chunk_paths-1)//cfg.chunk_paths
        for key in cfg.group_keys:
            case, role, anchor = _group_plan(key)
            group = out.create_group(key)
            probabilities, audit = profile_probabilities(cfg, case)
            group.attrs["profile_probability_audit_json"] = json.dumps(audit, sort_keys=True)
            group.attrs["case"], group.attrs["role"], group.attrs["anchor_id"] = case, role, anchor
            group.attrs["context_id"] = cfg.context(case).context_id
            group.create_dataset("probabilities", data=probabilities)
            if role == "conditional":
                anchor_rng = seeded_rng(cfg.seed, STREAM, cfg.config_hash, "held-out-anchor", case, anchor)
                initial = anchor_rng.multinomial(cfg.n_particles, probabilities)/cfg.n_particles
                group.attrs["conditional_initial_json"] = json.dumps(initial.tolist())
                group.attrs["conditional_initial_sha256"] = _array_sha(initial)
            for name, (shape, dtype) in _shapes(cfg).items():
                group.create_dataset(name, shape, dtype=dtype,
                                     chunks=(min(cfg.chunk_paths, shape[0]), *shape[1:]),
                                     compression="lzf", shuffle=True)
            group.create_dataset("chunk_complete", (chunks,), dtype="?")
            group.create_dataset("chunk_sha256", (chunks,), dtype="S64")
            group.create_dataset("chunk_seconds", (chunks,), dtype="f8")


def _draw(cfg, key, start, count, conditional_initial=None):
    case, role, anchor = _group_plan(key)
    rng = seeded_rng(cfg.seed, STREAM, cfg.config_hash, "reference-chunk", case, role, anchor,
                     start//cfg.chunk_paths)
    probabilities, _ = profile_probabilities(cfg, case)
    initial = (rng.multinomial(cfg.n_particles, probabilities, size=count)/cfg.n_particles
               if role == "ensemble" else np.tile(conditional_initial, (count, 1)))
    factor = cfg.reference_factors[-1] if start < cfg.refinement_paths else 1
    steps = round(cfg.horizon/cfg.reference_step)*factor
    increments = rng.normal(0, np.sqrt(cfg.reference_step/factor), (count, steps, cfg.n_cells))
    families = np.asarray([stable_hash([STREAM, cfg.seed, cfg.config_hash, case, role, anchor, index])
                           for index in range(start, start+count)], dtype="S64")
    return initial, increments, factor, families


def _aggregate(increments, factor):
    return increments.reshape(len(increments), increments.shape[1]//factor,
                              factor, increments.shape[-1]).sum(2)


def _weights(cfg, modes):
    centers = (np.arange(cfg.n_cells)+.5)*cfg.spec.h
    return np.sqrt(2)*np.asarray([function(mode*centers)*np.sinc(mode/cfg.n_cells)
                                  for mode in modes for function in (np.cos, np.sin)])


def _reduce(packet, cfg, step):
    states = packet["states"]
    stops = np.asarray(packet["stopped_at"])
    if (not np.isfinite(states).all() or states.min() < -1e-12
            or not np.allclose(states.sum(-1), 1, atol=1e-10, rtol=0)
            or np.isnan(stops).any() or (stops < 0).any()
            or (np.isfinite(stops) & (stops > cfg.horizon)).any()):
        raise ValueError("Invalid native path packet; survivor-conditioned reduction is forbidden")
    times = np.arange(states.shape[1])*step
    common_indices = np.rint(cfg.common_times/step).astype(int)
    query_indices = np.rint(np.asarray(cfg.query_times)/step).astype(int)
    common = states[:, common_indices]
    bank_b, bank_c = (bank.evaluate_paths(states, times, packet["stopped_at"])
                       for bank in _banks(cfg))
    many = common[:, 1:] @ _weights(cfg, range(1, cfg.n_cells//2)).T
    return {"bank_a": states[:, query_indices] @ _weights(cfg, (1, 2)).T,
            "bank_b": bank_b[:, query_indices], "bank_c": bank_c[:, query_indices],
            "common_paths": common, "many_observables": many.reshape(len(states), -1),
            "workloads": _probe.workload_values(common, cfg.spec, cfg.common_step),
            "stopped_at": packet["stopped_at"]}


def _simulate_chunk(cfg, key, start, count, conditional_initial):
    initial, finest_dw, finest_factor, families = _draw(cfg, key, start, count, conditional_initial)
    base_dw = _aggregate(finest_dw, finest_factor)
    base = _probe.numpy_euler(initial, base_dw, cfg.context(_group_plan(key)[0]),
                              cfg.spec, cfg.reference_step)
    result = {"initial": initial, "family_ids": families,
              "base_brownian_sha256": np.asarray([_array_sha(row) for row in base_dw], dtype="S64"),
              "drawn_brownian_sha256": np.asarray([_array_sha(row) for row in finest_dw], dtype="S64"),
              "model_increments": _aggregate(base_dw, round(cfg.model_step/cfg.reference_step)),
              **_reduce(base, cfg, cfg.reference_step)}
    audited = min(count, max(0, cfg.refinement_paths-start))
    if audited:
        levels, level_hashes = [], []
        for factor in cfg.reference_factors:
            increments = _aggregate(finest_dw[:audited], finest_factor//factor)
            packet = (dict(states=base["states"][:audited], stopped_at=base["stopped_at"][:audited])
                      if factor == 1 else _probe.numpy_euler(
                          initial[:audited], increments, cfg.context(_group_plan(key)[0]),
                          cfg.spec, cfg.reference_step/factor))
            levels.append(_reduce(packet, cfg, cfg.reference_step/factor))
            level_hashes.append(np.asarray([_array_sha(row) for row in increments], dtype="S64"))
        for name in levels[0]:
            result["audit_"+name] = np.stack([level[name] for level in levels], axis=1)
        result["audit_brownian_sha256"] = np.stack(level_hashes, axis=1)
    return result


def _chunk_hash(group, cfg, start, count):
    digest = hashlib.sha256()
    for name in sorted(_shapes(cfg)):
        stop = min(start+count, cfg.refinement_paths) if name.startswith("audit_") else start+count
        if stop <= start:
            continue
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(group[name][start:stop]).tobytes())
    return digest.hexdigest()


def _verify_store(store, cfg, identity, lock):
    if (json.loads(store.attrs["identity_json"]) != json.loads(json.dumps(identity))
            or json.loads(store.attrs["fit_lock_json"]) != lock):
        raise ValueError("Reference store identity/fit lock mismatch")
    chunks = (cfg.paths_per_group+cfg.chunk_paths-1)//cfg.chunk_paths
    expected_names = set(_shapes(cfg)) | {"probabilities", "chunk_complete", "chunk_sha256", "chunk_seconds"}
    if set(store) != set(cfg.group_keys) | {"common_times", "query_times", "reference_factors"}:
        raise ValueError("Reference store group inventory changed")
    for name, expected in (("common_times", cfg.common_times), ("query_times", cfg.query_times),
                           ("reference_factors", cfg.reference_factors)):
        if not np.array_equal(store[name][:], expected):
            raise ValueError("Reference observation grids/refinement factors changed")
    for key in cfg.group_keys:
        group = store[key]
        if set(group) != expected_names or group["chunk_complete"].shape != (chunks,):
            raise ValueError("Reference dataset inventory changed")
        for name, (shape, dtype) in _shapes(cfg).items():
            if group[name].shape != shape or group[name].dtype != np.dtype(dtype):
                raise ValueError("Reference dataset shape/dtype changed")
        case, role, anchor = _group_plan(key)
        probabilities, audit = profile_probabilities(cfg, case)
        if (not np.array_equal(group["probabilities"][:], probabilities)
                or json.loads(group.attrs["profile_probability_audit_json"]) != audit
                or group.attrs["context_id"] != cfg.context(case).context_id
                or (group.attrs["case"], group.attrs["role"], group.attrs["anchor_id"])
                != (case, role, anchor)):
            raise ValueError("Reference profile/context/group metadata changed")
        if role == "conditional":
            rng = seeded_rng(cfg.seed, STREAM, cfg.config_hash, "held-out-anchor", case, anchor)
            expected = rng.multinomial(cfg.n_particles, probabilities)/cfg.n_particles
            if (group.attrs["conditional_initial_sha256"] != _array_sha(expected)
                    or not np.array_equal(json.loads(group.attrs["conditional_initial_json"]), expected)):
                raise ValueError("Held-out conditional initial histogram changed")
        for index in np.flatnonzero(group["chunk_complete"][:]):
            start = int(index)*cfg.chunk_paths
            count = min(cfg.chunk_paths, cfg.paths_per_group-start)
            if _chunk_hash(group, cfg, start, count) != group["chunk_sha256"][index].decode():
                raise ValueError("Committed reference chunk checksum mismatch")


def generate_reference(directory, cfg=ReferenceConfig(), *, fit_lock, max_chunks=None, deadline=None):
    """Commit independent chunks; interruption resumes without resampling a family.

    ``fit_lock`` is a JSON path or mapping with expected_models and models entries
    (checkpoint_path,checkpoint_sha256,complete,source_hash). Each complete model
    is restored through the strict package loader before outcome generation.
    ``max_chunks`` and absolute monotonic ``deadline`` permit resumable batches.
    """
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 0):
        raise ValueError("Nonnegative chunk limit required")
    if deadline is not None and not np.isfinite(deadline):
        raise ValueError("Finite absolute monotonic deadline required")
    directory = Path(directory)
    manifest = prepare_reference(directory, cfg)
    lock = _fit_lock(fit_lock, cfg)
    if manifest.get("fit_lock") not in (None, lock):
        raise ValueError("Reference fit inventory changed after unlocking")
    manifest["fit_lock"] = lock
    path, manifest_path = directory/"reference.h5", directory/"manifest.json"
    if not path.exists():
        _create_store(path, cfg, manifest["identity"], lock)
    completed_now = 0
    manifest["status"] = "generating"
    if "failure" in manifest:
        manifest.setdefault("recovered_attempt_failures", []).append(manifest.pop("failure"))
    _json(manifest_path, manifest)
    with h5py.File(path, "r+") as store:
        _verify_store(store, cfg, manifest["identity"], lock)
        for key in cfg.group_keys:
            group = store[key]
            conditional = json.loads(group.attrs["conditional_initial_json"]) if _group_plan(key)[1] == "conditional" else None
            for chunk, done in enumerate(group["chunk_complete"][:]):
                if done:
                    continue
                if ((max_chunks is not None and completed_now >= max_chunks)
                        or (deadline is not None and time.monotonic() >= deadline)):
                    manifest["status"] = "partial"
                    _json(manifest_path, manifest)
                    return manifest
                started = time.monotonic()
                start = chunk*cfg.chunk_paths
                count = min(cfg.chunk_paths, cfg.paths_per_group-start)
                try:
                    values = _simulate_chunk(cfg, key, start, count, conditional)
                    for name, array in values.items():
                        group[name][start:start+len(array)] = array
                    group["chunk_sha256"][chunk] = _chunk_hash(group, cfg, start, count).encode()
                    group["chunk_seconds"][chunk] = time.monotonic()-started
                    store.flush()
                    group["chunk_complete"][chunk] = True
                    store.flush()
                except Exception as exc:
                    manifest["status"] = "failed"
                    manifest["failure"] = {"group": key, "chunk": chunk,
                                           "type": type(exc).__name__, "message": str(exc),
                                           "survivor_conditioned_estimates_allowed": False}
                    _json(manifest_path, manifest)
                    raise
                completed_now += 1
                manifest["committed_chunks"] = sum(int(store[group_key]["chunk_complete"][:].sum())
                                                     for group_key in cfg.group_keys)
                manifest["generation_seconds"] = sum(float(store[group_key]["chunk_seconds"][:].sum())
                                                        for group_key in cfg.group_keys)
                _json(manifest_path, manifest)
        manifest["status"] = "complete"
        manifest["failed_paths"] = 0
        manifest["stopped_paths_by_group"] = {
            key: int(np.isfinite(store[key]["stopped_at"][:]).sum()) for key in cfg.group_keys}
        manifest["group_content_hashes"] = {
            key: stable_hash([value.decode() for value in store[key]["chunk_sha256"][:]])
            for key in cfg.group_keys}
    manifest["reference_path"] = str(path.resolve())
    manifest["reference_sha256"] = _sha(path)
    _json(manifest_path, manifest)
    return manifest


def load_reference(directory, cfg=ReferenceConfig(), *, group_key=None):
    """Load a complete, integrity-checked reference group or the whole bank."""
    directory = Path(directory)
    manifest = prepare_reference(directory, cfg)
    path = directory/"reference.h5"
    if (manifest.get("status") != "complete" or not path.is_file()
            or manifest.get("reference_sha256") != _sha(path)):
        raise ValueError("Only complete unmutated reference ensembles may produce estimates")
    if group_key is not None and group_key not in cfg.group_keys:
        raise ValueError("Unknown reference group")
    keys = cfg.group_keys if group_key is None else (group_key,)
    groups = {}
    with h5py.File(path, "r") as store:
        _verify_store(store, cfg, manifest["identity"], manifest["fit_lock"])
        seen = set()
        for key in cfg.group_keys:
            ids = [value.decode() for value in store[key]["family_ids"][:]]
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError("Independent reference groups contain overlapping trajectory families")
            seen.update(ids)
        for key in keys:
            group = store[key]
            groups[key] = {name: group[name][:] for name in _shapes(cfg)}
            groups[key]["probabilities"] = group["probabilities"][:]
            groups[key]["metadata"] = dict(group.attrs)
    result = {"manifest": manifest, "common_times": cfg.common_times,
              "query_times": np.asarray(cfg.query_times), "groups": groups}
    if group_key is not None:
        result.update(groups[group_key])
    return result


def iter_reference_groups(directory, cfg=ReferenceConfig()):
    """Yield one checked group at a time, avoiding loading the entire bank at once."""
    for key in cfg.group_keys:
        yield key, load_reference(directory, cfg, group_key=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path("runs/path-generalization-2026-10-26/reference"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fit-lock", type=Path,
                        default=Path("runs/path-generalization-2026-10-26/payoffs/fit_lock.json"))
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cfg = ReferenceConfig(**json.loads(args.config.read_text())) if args.config else ReferenceConfig()
    result = (prepare_reference(args.directory, cfg) if args.prepare_only else
              generate_reference(args.directory, cfg, fit_lock=args.fit_lock, max_chunks=args.max_chunks))
    print(json.dumps({"status": result["status"], "committed_chunks": result["committed_chunks"],
                      "config_hash": cfg.config_hash}, indent=2))


if __name__ == "__main__":
    main()
