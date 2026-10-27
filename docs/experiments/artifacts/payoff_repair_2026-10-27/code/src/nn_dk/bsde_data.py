"""Predictable forward traces and reproducible anchor-grouped BSDE pools.

Physical Brownian increments are shared with the BSDE martingale sum. Stopping
never rescales those increments: the only predictable multiplier is the flag
that the physical path was active at the beginning of the step.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from .bsde_types import BSDETrace
from .data import json_ready, source_hash
from .native import initialize_masses, simulate_native
from .types import seeded_rng, stable_hash


def ou_rates(config) -> np.ndarray:
    """Discrete heat decay rates for normalized cos/sin modes one and two."""
    return np.asarray(config.mode_rates, dtype=float)


def ou_exact_moments(initial, horizons, config, dt=None):
    """Conditional moments of the continuous OU SDE or its Euler recursion.

    An initial batch ``(...,4)`` gives means ``(...,times,4)`` and covariances
    ``(...,times,4,4)``. Euler covariance accounts for its stationary bias;
    continuous-time covariance tends to identity, independently of the grid.
    """
    initial = np.asarray(initial, dtype=float)
    horizons = np.atleast_1d(np.asarray(horizons, dtype=float))
    if initial.shape[-1:] != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("Finite four-dimensional OU initial states required")
    if (horizons.ndim != 1 or not len(horizons) or not np.all(np.isfinite(horizons))
            or np.any(horizons < 0)):
        raise ValueError("Finite nonnegative OU horizons required")
    rates = ou_rates(config)
    if dt is None:
        decay = np.exp(-horizons[:, None] * rates)
        variance = -np.expm1(-2 * horizons[:, None] * rates)
    else:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite Euler step required")
        ratios = horizons / dt
        if not np.allclose(ratios, np.rint(ratios), atol=1e-8, rtol=0):
            raise ValueError("OU horizons must lie on the Euler time grid")
        transition = 1 - dt * rates
        if np.any(np.abs(transition) >= 1):
            raise ValueError("OU Euler step must have stable nonconstant modes")
        decay = transition[None, :] ** np.rint(ratios).astype(int)[:, None]
        variance = 2 * rates * dt * (1 - decay**2) / (1 - transition**2)
    mean = initial[..., None, :] * decay
    covariance = variance[..., :, None] * np.eye(4)
    covariance = np.broadcast_to(covariance, (*initial.shape[:-1], len(horizons), 4, 4)).copy()
    return mean, covariance


def _grid(dt, horizon):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Finite positive forward step and horizon required")
    n_steps = int(round(horizon / dt))
    if n_steps < 1 or not np.isclose(n_steps * dt, horizon, atol=1e-10, rtol=0):
        raise ValueError("Forward horizon must lie on the integration grid")
    return np.arange(n_steps + 1, dtype=float) * dt


def simulate_forward_trace(problem, config, initial, seed, dt=None, horizon=None,
                           anchor_ids=None, family_ids=None,
                           brownian_increments=None) -> BSDETrace:
    """Simulate native DK or normalized OU while exposing predictable traces."""
    if problem not in ("ou", "dk"):
        raise ValueError("BSDE forward problem must be ou or dk")
    dt = config.dt if dt is None else dt
    horizon = config.horizon if horizon is None else horizon
    times = _grid(dt, horizon)
    initial = np.asarray(initial, dtype=float)
    if initial.ndim == 1:
        initial = initial[None, :]
    dimension = 4 if problem == "ou" else config.n_cells
    if initial.ndim != 2 or initial.shape[1] != dimension or not np.all(np.isfinite(initial)):
        raise ValueError("Forward initial states have incompatible shape or nonfinite values")
    n_paths, n_steps = len(initial), len(times) - 1
    if n_paths < 1:
        raise ValueError("A nonempty forward initial-state batch is required")
    anchors = (tuple(stable_hash(["bsde-initial", problem, int(seed), i])[:24]
                     for i in range(n_paths)) if anchor_ids is None else tuple(anchor_ids))
    families = (tuple(stable_hash(["bsde-continuation", problem, int(seed), i])[:24]
                      for i in range(n_paths)) if family_ids is None else tuple(family_ids))
    if len(anchors) != n_paths or len(families) != n_paths or len(set(families)) != n_paths:
        raise ValueError("Per-path anchor IDs and unique trajectory-family IDs required")
    if brownian_increments is None:
        # One stream per independent continuation, invariant to path batching.
        dw = np.stack([seeded_rng(seed, "bsde-forward-noise", problem, family).normal(
            0, np.sqrt(dt), (n_steps, dimension)) for family in families])
    else:
        dw = np.asarray(brownian_increments, dtype=float)
        if dw.shape != (n_paths, n_steps, dimension) or not np.all(np.isfinite(dw)):
            raise ValueError("Full Brownian increments must match paths, steps, and driver dimension")
    tick = time.perf_counter()
    if problem == "ou":
        rates = ou_rates(config)
        if np.any(np.abs(1 - dt * rates) >= 1):
            raise ValueError("OU Euler step must have stable nonconstant modes")
        states = np.empty((n_paths, n_steps + 1, dimension))
        states[:, 0] = initial
        for step in range(n_steps):
            states[:, step + 1] = ((1 - dt * rates) * states[:, step]
                                   + np.sqrt(2 * rates) * dw[:, step])
        active = np.ones((n_paths, n_steps), dtype=bool)
        stopped = np.full(n_paths, np.inf)
        proposals = states[:, 1:].copy()
        diagnostics = {"method": "normalized-ou-euler-v1", "rates": rates.tolist(),
                       "all_paths_retained": True, "stopping_probability": 0.0}
    else:
        context = config.context(config.cases[0])
        if context.kappa != 0 or context.a2 != 0:
            raise ValueError("The initial BSDE DK workflow requires noninteracting physics")
        capture = {}
        batch = simulate_native(context, config.spec, times, dt, seed, n_paths,
                                initial_masses=initial, brownian_increments=dw,
                                family_ids=families, trace_capture=capture)
        states, active, proposals = batch.states, capture["active"], capture["proposals"]
        stopped, diagnostics = batch.stopped_at, dict(batch.diagnostics)
    diagnostics.update(seed=int(seed), dt=float(dt), horizon=float(horizon),
                       generation_seconds=time.perf_counter() - tick,
                       brownian_convention="full increments; active-at-start martingale mask",
                       increment_variance=float(dt),
                       proposals_are_learner_features=False)
    return BSDETrace(times=times, states=states, increments=dw, active=active,
                     anchor_ids=anchors, family_ids=families, problem=problem,
                     stopped_at=stopped, proposals=proposals, diagnostics=diagnostics)


def _pool_path(directory, problem):
    if problem not in ("ou", "dk"):
        raise ValueError("BSDE pool problem must be ou or dk")
    return Path(directory) / f"{problem}_pool.h5"


def _anchor_plan(config, problem, split):
    if split not in ("train", "validation"):
        raise ValueError("Forward pools contain train and validation anchors")
    count = config.train_anchors if split == "train" else config.validation_anchors
    continuations = (config.train_continuations if split == "train"
                     else config.validation_continuations)
    for index in range(count):
        case = "ou" if problem == "ou" else config.cases[index % len(config.cases)]
        anchor = stable_hash(["bsde-anchor-v1", config.seed, problem, split, index, case])[:24]
        initial_seed = int(seeded_rng(config.seed, "bsde-anchor-initial", problem, split,
                                     anchor).integers(0, 2**63 - 1))
        forward_seed = int(seeded_rng(config.seed, "bsde-anchor-continuations", problem,
                                     split, anchor).integers(0, 2**63 - 1))
        yield index, case, anchor, initial_seed, forward_seed, continuations


def _check_pool(store, config, problem, fingerprint, require_existing=False):
    expected = {"config_hash": config.config_hash, "source_hash": fingerprint,
                "workflow": "conditional-bsde-forward-pool-v1", "problem": problem,
                "target_id": config.target_id(problem)}
    exists = require_existing or len(store) > 0 or len(store.attrs) > 0
    if any((exists and key not in store.attrs)
           or (key in store.attrs and store.attrs[key] != value)
           for key, value in expected.items()):
        raise ValueError("BSDE pool target, source, or configuration changed; use a new directory")
    return expected


def generate_bsde_pool(directory, config, problem, deadline=None) -> dict:
    """Generate anchor-atomic HDF5 pools; interruption never commits partial anchors."""
    path = _pool_path(directory, problem)
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = source_hash()
    tick, generated = time.perf_counter(), 0
    with h5py.File(path, "a") as store:
        for key, value in _check_pool(store, config, problem, fingerprint).items():
            store.attrs[key] = value
        store.attrs["config"] = json.dumps(config.to_dict(), sort_keys=True)
        store.attrs["oracle_metadata_separate"] = True
        for split in ("train", "validation"):
            parent = store.require_group(split)
            for stale in [name for name in parent if name.startswith("_pending_")]:
                del parent[stale]
            for index, case, anchor, initial_seed, forward_seed, n_paths in _anchor_plan(
                    config, problem, split):
                name = f"anchor_{index:06d}"
                if name in parent:
                    group = parent[name]
                    if not group.attrs.get("complete", False) or group.attrs["anchor_id"] != anchor:
                        raise ValueError("Invalid committed BSDE anchor; use a new pool directory")
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    store.flush()
                    return {"complete": False, "problem": problem,
                            "generated_this_call": generated,
                            "seconds_this_call": time.perf_counter() - tick}
                if problem == "ou":
                    initial = seeded_rng(initial_seed, "ou-initial-anchor").normal(size=(1, 4))
                else:
                    initial = initialize_masses(config.context(case), config.spec, 1, initial_seed)
                families = tuple(stable_hash(["bsde-family-v1", anchor, child])[:24]
                                 for child in range(n_paths))
                trace = simulate_forward_trace(
                    problem, config, np.repeat(initial, n_paths, axis=0), forward_seed,
                    anchor_ids=(anchor,) * n_paths, family_ids=families)
                pending = f"_pending_{name}"
                group = parent.create_group(pending)
                group.attrs.update(anchor_id=anchor, case=case, initial_seed=initial_seed,
                                   forward_seed=forward_seed, source_hash=fingerprint,
                                   diagnostics=json.dumps(json_ready(trace.diagnostics)))
                learner, oracle = group.create_group("learner"), group.create_group("oracle")
                learner.create_dataset("times", data=trace.times)
                for key in ("states", "increments", "active"):
                    values = getattr(trace, key)
                    learner.create_dataset(key, data=values, chunks=(1, *values.shape[1:]),
                                           compression="gzip", shuffle=True)
                learner.create_dataset("anchor_ids", data=np.asarray(trace.anchor_ids,
                                       dtype=h5py.string_dtype()))
                learner.create_dataset("family_ids", data=np.asarray(trace.family_ids,
                                       dtype=h5py.string_dtype()))
                oracle.create_dataset("proposals", data=trace.proposals,
                                      chunks=(1, *trace.proposals.shape[1:]),
                                      compression="gzip", shuffle=True)
                oracle.create_dataset("stopped_at", data=trace.stopped_at)
                oracle.attrs["not_network_inputs"] = True
                store.flush()
                group.attrs["complete"] = True
                store.flush()
                parent.move(pending, name)
                store.flush()
                generated += 1
        store.attrs["complete"] = True
    return {"complete": True, "problem": problem, "generated_this_call": generated,
            "seconds_this_call": time.perf_counter() - tick}


def load_bsde_pool(directory, config, problem, split) -> BSDETrace:
    """Load a complete split, checking provenance and trajectory-family isolation."""
    path = _pool_path(directory, problem)
    with h5py.File(path, "r") as store:
        _check_pool(store, config, problem, source_hash(), require_existing=True)
        plans = list(_anchor_plan(config, problem, split))
        if split not in store:
            raise ValueError("BSDE split is absent; resume generation")
        groups, anchor_ids, families, cases, seconds = [], [], [], [], 0.0
        for index, case, anchor, *_ in plans:
            name = f"anchor_{index:06d}"
            if name not in store[split] or not store[split][name].attrs.get("complete", False):
                raise ValueError("BSDE split is incomplete; resume generation")
            group = store[split][name]
            if group.attrs["anchor_id"] != anchor:
                raise ValueError("BSDE anchor identity mismatch")
            groups.append(group)
            anchor_ids.extend(group["learner/anchor_ids"].asstr()[:].tolist())
            families.extend(group["learner/family_ids"].asstr()[:].tolist())
            cases.extend([case] * len(group["learner/states"]))
            seconds += json.loads(group.attrs["diagnostics"])["generation_seconds"]
        if len(set(families)) != len(families):
            raise ValueError("Duplicate independent trajectory families in BSDE split")
        other = "validation" if split == "train" else "train"
        other_anchors = {plan[2] for plan in _anchor_plan(config, problem, other)}
        if set(anchor_ids) & other_anchors:
            raise ValueError("Initial-anchor leakage between training and validation")
        return BSDETrace(
            times=groups[0]["learner/times"][:],
            states=np.concatenate([g["learner/states"][:] for g in groups]),
            increments=np.concatenate([g["learner/increments"][:] for g in groups]),
            active=np.concatenate([g["learner/active"][:] for g in groups]),
            anchor_ids=tuple(anchor_ids), family_ids=tuple(families), problem=problem,
            stopped_at=np.concatenate([g["oracle/stopped_at"][:] for g in groups]),
            proposals=np.concatenate([g["oracle/proposals"][:] for g in groups]),
            diagnostics={"problem": problem, "split": split, "cases": cases,
                         "config_hash": config.config_hash, "source_hash": store.attrs["source_hash"],
                         "independent_anchors": len(groups), "independent_paths": len(families),
                         "generation_seconds": seconds,
                         "proposals_are_learner_features": False})
