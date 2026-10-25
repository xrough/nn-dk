"""Addressable interacting-DK state collocation and locked reference ensembles.

No future payoff labels enter collocation. Reference paths retain their stopped
values; refinements of a trajectory always share its family identifier.
"""
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

from .data import source_hash
from .native import initialize_masses, physical_observable_weights, simulate_native
from .types import seeded_rng, stable_hash

KIND = "interacting-generator-state-and-reference-bank-v1"
DATA_KEY_KIND = "interacting-independent-physical-data-streams-v2"


@lru_cache(maxsize=32)
def collocation_data_id(cfg):
    """Physical state/horizon pool identity, independent of the fitted method.

    Strict artifact provenance still includes the complete configuration and
    source hash. Two separately locked candidates can therefore share literal
    current states and residual horizons without accepting each other's files.
    """
    return stable_hash(dict(
        kind=DATA_KEY_KIND, role="collocation", seed=cfg.seed,
        target_id=cfg.target_id, native_spec_id=cfg.spec.target_id,
        context_ids={case: cfg.context(case).context_id for case in cfg.cases},
        train_paths=cfg.train_paths, validation_paths=cfg.validation_paths,
        state_step=cfg.state_step, state_times=cfg.state_times,
        horizon_samples=cfg.horizon_samples))


@lru_cache(maxsize=32)
def reference_data_id(cfg):
    """Predeclared independent reference/integration pool, excluding fit knobs."""
    return stable_hash(dict(
        kind=DATA_KEY_KIND, role="reference", seed=cfg.seed,
        target_id=cfg.target_id, native_spec_id=cfg.spec.target_id,
        context_ids={case: cfg.context(case).context_id for case in cfg.cases},
        query_times=cfg.query_times, reference_step=cfg.reference_step,
        reference_factors=cfg.reference_factors,
        conditional_anchors=cfg.conditional_anchors,
        conditional_paths=cfg.conditional_paths, ensemble_paths=cfg.ensemble_paths,
        ensemble_initial_paths=cfg.ensemble_initial_paths,
        refinement_paths=cfg.refinement_paths))


def _stream_data_id(cfg, keys):
    if keys and keys[0] in ("train", "validation"):
        return collocation_data_id(cfg)
    if keys and keys[0] in ("reference", "reference-anchor", "integration-initial"):
        return reference_data_id(cfg)
    raise ValueError("Undeclared interacting independent data stream")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    temporary = Path(str(path)+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2)+"\n")
    temporary.replace(path)


def _expired(deadline):
    return deadline is not None and time.monotonic() >= deadline


def _seed(cfg, *keys):
    return int(seeded_rng(cfg.seed, DATA_KEY_KIND, _stream_data_id(cfg, keys), *keys).integers(0, 2**63-1))


def _family(cfg, *keys):
    return stable_hash([DATA_KEY_KIND, cfg.seed, _stream_data_id(cfg, keys), *keys])[:24]


def _identity(cfg, role):
    return dict(kind=KIND, source_hash=source_hash(), config_hash=cfg.config_hash,
                target_id=cfg.target_id, role=role,
                bank_data_id=(reference_data_id(cfg) if role == "reference"
                              else collocation_data_id(cfg)))


def _row_hash(group, index):
    return _array_row_hash({name: group[name][index] for name in group
                            if name not in ("complete", "row_hash")})


def _array_row_hash(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    return digest.hexdigest()


def _group_hash(group):
    return stable_hash([value.decode() for value in group["row_hash"][:]])


def _datasets(group, shapes):
    for name, (shape, dtype) in shapes.items():
        group.create_dataset(name, shape, dtype=dtype,
                             chunks=(min(16, shape[0]), *shape[1:]),
                             compression="gzip", shuffle=True)


def _state_shapes(cfg, role):
    count = cfg.train_paths if role == "train" else cfg.validation_paths
    return {"states": ((count, len(cfg.state_times), cfg.n_cells), "f8"),
            "horizons": ((count, len(cfg.state_times), cfg.horizon_samples), "f8"),
            "stopped_at": ((count,), "f8"), "family_ids": ((count,), "S24"),
            "complete": ((count,), "?"), "row_hash": ((count,), "S64")}


def _reference_shapes(cfg, role):
    count = (cfg.conditional_anchors*cfg.conditional_paths if role == "conditional"
             else cfg.ensemble_paths)
    return {"initial": ((count, cfg.n_cells), "f8"),
            "values": ((count, len(cfg.query_times), 4), "f8"),
            "refinement_values": ((count, len(cfg.reference_factors), len(cfg.query_times), 4), "f8"),
            "refinement_stopped_at": ((count, len(cfg.reference_factors)), "f8"),
            "stopped_at": ((count,), "f8"), "family_ids": ((count,), "S24"),
            "anchor_ids": ((count,), "i8"), "failed": ((count,), "?"),
            "complete": ((count,), "?"), "row_hash": ((count,), "S64")}


def _validate_group(group, shapes):
    if set(group) != set(shapes):
        raise ValueError("Interacting bank dataset inventory changed")
    for name, (shape, dtype) in shapes.items():
        if group[name].shape != shape or group[name].dtype != np.dtype(dtype):
            raise ValueError("Interacting bank dataset shape or dtype changed")
    complete = group["complete"][:]
    # Read blocks once per dataset rather than issuing a separate HDF5 read for
    # every small scalar/vector row of large diagnostic ensembles.
    for start in range(0, len(complete), 256):
        stop = min(start+256, len(complete))
        indices = np.flatnonzero(complete[start:stop])
        if not len(indices):
            continue
        arrays = {name: group[name][start:stop] for name in group
                  if name not in ("complete", "row_hash")}
        hashes = group["row_hash"][start:stop]
        for index in indices:
            if _array_row_hash({name: values[index] for name, values in arrays.items()}) != hashes[index].decode():
                raise ValueError("A committed interacting trajectory-family row changed")
    if bool(group.attrs.get("complete")):
        if not np.all(group["complete"][:]) or group.attrs.get("content_hash") != _group_hash(group):
            raise ValueError("Interacting group completion or content hash changed")


def _validate_store(store, cfg, role):
    if any(store.attrs.get(k) != v for k, v in _identity(cfg, role).items()):
        raise ValueError("Interacting bank source, target, role or configuration changed")
    if set(store) != set(cfg.cases):
        raise ValueError("Interacting bank initial-profile inventory changed")
    for case in cfg.cases:
        group = store[case]
        if group.attrs.get("context_id") != cfg.context(case).context_id:
            raise ValueError("Interacting bank physical context changed")
        if role == "reference":
            if set(group) != {"conditional", "ensemble", "anchors", "integration_initial", "integration_family_ids"}:
                raise ValueError("Interacting reference inventory changed")
            if (group["anchors"].shape != (cfg.conditional_anchors, cfg.n_cells)
                    or group["integration_initial"].shape != (cfg.ensemble_initial_paths, cfg.n_cells)
                    or group["integration_family_ids"].shape != (cfg.ensemble_initial_paths,)):
                raise ValueError("Interacting reference initialization shapes changed")
            for subrole in ("conditional", "ensemble"):
                _validate_group(group[subrole], _reference_shapes(cfg, subrole))
            initial_hash = stable_hash({name: hashlib.sha256(group[name][:].tobytes()).hexdigest()
                                       for name in ("anchors", "integration_initial", "integration_family_ids")})
            if initial_hash != group.attrs.get("initial_hash"):
                raise ValueError("Interacting reference initialization content changed")
        else:
            _validate_group(group, _state_shapes(cfg, role))
    if store.attrs.get("complete"):
        hashes = {case: ({r: _group_hash(store[case][r]) for r in ("conditional", "ensemble")}
                         if role == "reference" else _group_hash(store[case])) for case in cfg.cases}
        if store.attrs.get("content_hash") != stable_hash(hashes):
            raise ValueError("Interacting bank combined content hash changed")


def _finish(store, cfg, role):
    hashes, complete = {}, True
    for case in cfg.cases:
        groups = ([store[case][r] for r in ("conditional", "ensemble")]
                  if role == "reference" else [store[case]])
        for group in groups:
            done = bool(np.all(group["complete"][:]))
            group.attrs["complete"] = done
            if done:
                group.attrs["content_hash"] = _group_hash(group)
            complete &= done
        hashes[case] = ({r: _group_hash(store[case][r]) for r in ("conditional", "ensemble")}
                        if role == "reference" else _group_hash(store[case]))
    store.attrs["complete"] = complete
    if complete:
        store.attrs["content_hash"] = stable_hash(hashes)
    store.flush()


def _packet(path, cfg, role):
    with h5py.File(path, "r") as store:
        _validate_store(store, cfg, role)
        result = dict(_identity(cfg, role), path=str(Path(path).resolve()),
                      complete=bool(store.attrs.get("complete")),
                      content_hash=str(store.attrs.get("content_hash", "")),
                      generation_seconds=float(store.attrs.get("generation_seconds", 0.)),
                      sha256=file_sha256(path), all_paths_retained=True,
                      reference_bias_status="unresolved" if role == "reference" else None)
    return result


def _stored_packet(path, cfg, role):
    packet = _packet(path, cfg, role)
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.exists() or json.loads(sidecar.read_text()) != packet:
        raise ValueError("Interacting bank JSON manifest or file SHA changed")
    return packet


def _complete_replay(path, cfg, role):
    """Recover a crash after HDF5 completion but before its final JSON rename."""
    sidecar = Path(path).with_suffix(".json")
    if sidecar.exists() and json.loads(sidecar.read_text()).get("complete"):
        return _stored_packet(path, cfg, role)
    packet = _packet(path, cfg, role)
    _json(sidecar, packet)
    return packet


def ensure_interacting_bank(directory, cfg, role, deadline=None):
    """Generate native stopped state collocation; one addressable family per row."""
    if role not in ("train", "validation"):
        raise ValueError("Learner state bank role must be train or validation")
    path = Path(directory)/f"interacting_{role}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with h5py.File(path, "r") as store:
            _validate_store(store, cfg, role)
            completed = bool(store.attrs.get("complete"))
        if completed:
            return _complete_replay(path, cfg, role)
    else:
        with h5py.File(path, "w") as store:
            store.attrs.update(**_identity(cfg, role), complete=False, generation_seconds=0.,
                               future_payoff_labels=False, config=json.dumps(cfg.to_dict(), sort_keys=True))
            for case in cfg.cases:
                group = store.create_group(case)
                group.attrs.update(context_id=cfg.context(case).context_id, complete=False)
                _datasets(group, _state_shapes(cfg, role))
    started = time.perf_counter()
    with h5py.File(path, "r+") as store:
        for case in cfg.cases:
            group = store[case]
            for index in np.flatnonzero(~group["complete"][:]):
                if _expired(deadline):
                    break
                seed = _seed(cfg, role, case, int(index))
                family = _family(cfg, role, case, int(index))
                initial = initialize_masses(cfg.context(case), cfg.spec, 1, seed)
                batch = simulate_native(cfg.context(case), cfg.spec, cfg.state_times,
                                        cfg.state_step, seed, 1, initial_masses=initial,
                                        family_ids=(family,))
                horizons = seeded_rng(seed, "independent-residual-horizons").uniform(
                    0, cfg.horizon, (len(cfg.state_times), cfg.horizon_samples))
                group["states"][index] = batch.states[0]
                group["horizons"][index] = horizons
                group["stopped_at"][index] = batch.stopped_at[0]
                group["family_ids"][index] = family.encode()
                group["row_hash"][index] = _row_hash(group, int(index)).encode()
                store.flush()
                group["complete"][index] = True
                store.flush()
        store.attrs["generation_seconds"] += time.perf_counter()-started
        _finish(store, cfg, role)
    packet = _packet(path, cfg, role)
    _json(path.with_suffix(".json"), packet)
    return packet


def load_interacting_bank(directory, cfg, role):
    """Flatten derived collocation samples, retaining their independent family IDs."""
    if role not in ("train", "validation"):
        raise ValueError("Only training/validation states can be learner inputs")
    path = Path(directory)/f"interacting_{role}.h5"
    packet = _stored_packet(path, cfg, role)
    if not packet["complete"]:
        raise ValueError("Cannot read an incomplete interacting collocation bank")
    states, horizons, families, cases, snapshot_times = [], [], [], [], []
    with h5py.File(path, "r") as store:
        for case in cfg.cases:
            group = store[case]
            shape = group["horizons"].shape
            states.append(np.repeat(group["states"][:], cfg.horizon_samples, axis=1).reshape(-1, cfg.n_cells))
            horizons.append(group["horizons"][:].reshape(-1))
            repeats = len(cfg.state_times)*cfg.horizon_samples
            families.extend(np.repeat(group["family_ids"][:].astype(str), repeats))
            cases.extend([case]*int(np.prod(shape)))
            snapshot_times.append(np.broadcast_to(np.asarray(cfg.state_times)[None, :, None], shape).reshape(-1))
    return dict(packet, states=np.concatenate(states), horizons=np.concatenate(horizons),
                family_ids=tuple(families), case_ids=tuple(cases),
                snapshot_times=np.concatenate(snapshot_times),
                future_payoff_labels=False, split_unit="independent-trajectory-family")


def verify_interacting_fit_lock(directory, cfg):
    """Require every predeclared fit before opening any reference/test stream."""
    path = Path(directory)/"manifest.json"
    if not path.exists():
        raise ValueError("Interacting references require a complete fit lock")
    manifest = json.loads(path.read_text())
    lock = manifest.get("fit_lock", {})
    if (manifest.get("source_hash") != source_hash() or manifest.get("config_hash") != cfg.config_hash
            or not manifest.get("test_unlocked_unix")
            or lock.get("expected_models") != len(cfg.optimizer_seeds)
            or len(lock.get("models", [])) != len(cfg.optimizer_seeds)):
        raise ValueError("Interacting references require a matching complete fit lock")
    from .interacting_training import load_interacting_model
    identities = set()
    for item in lock["models"]:
        seed = item.get("seed")
        if seed not in cfg.optimizer_seeds or seed in identities:
            raise ValueError("Interacting fit lock seed inventory changed")
        identities.add(seed)
        for kind in ("checkpoint", "metadata"):
            artifact = Path(item.get(f"{kind}_path", ""))
            if not artifact.is_absolute():
                artifact = Path(directory)/artifact
            if (not artifact.is_file()
                    or file_sha256(artifact) != item.get(f"{kind}_sha256")):
                raise ValueError("Locked interacting checkpoint or metadata changed")
        metadata_path = Path(item["metadata_path"])
        checkpoint_path = Path(item["checkpoint_path"])
        metadata_path = metadata_path if metadata_path.is_absolute() else Path(directory)/metadata_path
        checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else Path(directory)/checkpoint_path
        metadata = json.loads(metadata_path.read_text())
        model = load_interacting_model(checkpoint_path)
        provenance = model.provenance
        expected = dict(source_hash=source_hash(), config_hash=cfg.config_hash,
                        target_id=cfg.target_id, seed=seed)
        if (not metadata.get("complete") or not getattr(model, "complete", False)
                or any(provenance.get(k) != v for k, v in expected.items())
                or any(metadata.get(k) != v for k, v in expected.items())):
            raise ValueError("Interacting fit lock contains an incomplete or incompatible fit")
    return manifest


def _create_reference(path, cfg):
    # A crash while drawing a large initialization pool cannot leave a partially
    # declared final bank. Only initialized inventories receive the final name.
    temporary = Path(str(path)+".initializing")
    with h5py.File(temporary, "w") as store:
        store.attrs.update(**_identity(cfg, "reference"), complete=False, generation_seconds=0.,
                           finite_reference_law="stopped-Euler-first-polygon-crossing-v1",
                           continuous_bias_status="unresolved", all_paths_retained=True)
        for case in cfg.cases:
            group = store.create_group(case)
            group.attrs["context_id"] = cfg.context(case).context_id
            anchors = np.stack([initialize_masses(cfg.context(case), cfg.spec, 1,
                                _seed(cfg, "reference-anchor", case, a))[0]
                                for a in range(cfg.conditional_anchors)])
            integration = np.stack([initialize_masses(cfg.context(case), cfg.spec, 1,
                                    _seed(cfg, "integration-initial", case, i))[0]
                                    for i in range(cfg.ensemble_initial_paths)])
            group.create_dataset("anchors", data=anchors)
            group.create_dataset("integration_initial", data=integration)
            group.create_dataset("integration_family_ids", data=np.asarray([
                _family(cfg, "integration-initial", case, i).encode()
                for i in range(cfg.ensemble_initial_paths)], dtype="S24"))
            group.attrs["initial_hash"] = stable_hash({name: hashlib.sha256(group[name][:].tobytes()).hexdigest()
                                                       for name in ("anchors", "integration_initial", "integration_family_ids")})
            for role in ("conditional", "ensemble"):
                subgroup = group.create_group(role)
                subgroup.attrs["complete"] = False
                _datasets(subgroup, _reference_shapes(cfg, role))
    temporary.replace(path)


def ensure_interacting_reference(directory, cfg, deadline=None):
    """Generate complete reference allocations only after immutable all-seed lock.

    First predeclared refinement paths share literal finest Brownian sums across
    all levels. Remaining paths use the base Euler law. A small difference never
    certifies continuous-time bias, which remains unresolved.
    """
    verify_interacting_fit_lock(directory, cfg)
    path = Path(directory)/"interacting_reference.h5"
    if path.exists():
        with h5py.File(path, "r") as store:
            _validate_store(store, cfg, "reference")
            completed = bool(store.attrs.get("complete"))
        if completed:
            return _complete_replay(path, cfg, "reference")
    else:
        _create_reference(path, cfg)
    started = time.perf_counter()
    weights = physical_observable_weights(cfg.spec, cfg.queries)
    with h5py.File(path, "r+") as store:
        for case in cfg.cases:
            for role in ("conditional", "ensemble"):
                group = store[case][role]
                missing = np.flatnonzero(~group["complete"][:])
                # Batch only paths with the same finest level; each row still has
                # an independent addressable stream and an atomic commit marker.
                for refine in (True, False):
                    selected = [int(i) for i in missing if
                                ((int(i) % cfg.conditional_paths if role == "conditional" else int(i))
                                 < cfg.refinement_paths) == refine]
                    for offset in range(0, len(selected), cfg.reference_batch_size):
                        if _expired(deadline):
                            break
                        indices = selected[offset:offset+cfg.reference_batch_size]
                        seeds = [_seed(cfg, "reference", role, case, i) for i in indices]
                        families = [_family(cfg, "reference", role, case, i) for i in indices]
                        anchors = [i//cfg.conditional_paths if role == "conditional" else -1 for i in indices]
                        initials = np.stack([
                            store[case]["anchors"][a] if a >= 0 else
                            initialize_masses(cfg.context(case), cfg.spec, 1, seed)[0]
                            for a, seed in zip(anchors, seeds, strict=True)])
                        factors = cfg.reference_factors if refine else (1,)
                        fine_dt = cfg.reference_step/max(factors)
                        increments = np.stack([
                            seeded_rng(seed, "physical-reference-face-noise").normal(
                                0, np.sqrt(fine_dt), (round(cfg.horizon/fine_dt), cfg.n_cells))
                            for seed in seeds])
                        shape = (len(indices), len(cfg.reference_factors), len(cfg.query_times), 4)
                        values = np.full(shape, np.nan)
                        stops = np.full((len(indices), len(cfg.reference_factors)), np.nan)
                        failed = np.zeros(len(indices), dtype=bool)
                        for level, factor in enumerate(factors):
                            aggregate = max(factors)//factor
                            dw = increments.reshape(len(indices), -1, aggregate, cfg.n_cells).sum(axis=2)
                            try:
                                batch = simulate_native(cfg.context(case), cfg.spec, cfg.query_times,
                                                        cfg.reference_step/factor, seeds[0], len(indices),
                                                        initial_masses=initials, brownian_increments=dw,
                                                        family_ids=families)
                                # Explicit fixed-axis reduction avoids changing
                                # BLAS GEMV/GEMM summation with the batch size.
                                values[:, level] = np.einsum("ptg,og->pto", batch.states, weights,
                                                            optimize=False)
                                stops[:, level] = batch.stopped_at
                                failed |= ~np.all(np.isfinite(values[:, level]), axis=(1, 2))
                            except (FloatingPointError, ValueError):
                                # Retain every failed row; no survivor-conditioned estimator.
                                failed[:] = True
                        rows = dict(initial=initials, values=values[:, 0], refinement_values=values,
                                    refinement_stopped_at=stops, stopped_at=stops[:, 0],
                                    family_ids=np.asarray(families, dtype="S24"),
                                    anchor_ids=np.asarray(anchors, dtype="i8"), failed=failed)
                        hashes = np.asarray([
                            _array_row_hash({name: array[row] for name, array in rows.items()}).encode()
                            for row in range(len(indices))], dtype="S64")
                        # Each row has its own commit marker. All arrays/hashes
                        # reach storage before any marker can become committed.
                        for name, array in rows.items():
                            group[name][indices] = array
                        group["row_hash"][indices] = hashes
                        store.flush()
                        group["complete"][indices] = True
                        store.flush()
        store.attrs["generation_seconds"] += time.perf_counter()-started
        _finish(store, cfg, "reference")
    packet = _packet(path, cfg, "reference")
    _json(path.with_suffix(".json"), packet)
    return packet


def load_interacting_reference(directory, cfg, case):
    verify_interacting_fit_lock(directory, cfg)
    if case not in cfg.cases:
        raise ValueError("Undeclared interacting initial law")
    path = Path(directory)/"interacting_reference.h5"
    packet = _stored_packet(path, cfg, "reference")
    if not packet["complete"]:
        raise ValueError("Cannot estimate statistics from incomplete reference allocation")
    with h5py.File(path, "r") as store:
        group = store[case]
        return dict(packet, case=case, anchors=group["anchors"][:],
                    integration_initial=group["integration_initial"][:],
                    integration_family_ids=tuple(group["integration_family_ids"][:].astype(str)),
                    **{role: {name: group[role][name][:] for name in group[role]
                              if name not in ("complete", "row_hash")}
                       for role in ("conditional", "ensemble")})
