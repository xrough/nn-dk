"""Nested, fixed-budget initial-anchor ablations for conditional BSDE models.

Only the union of the requested trajectory families is simulated.  For example,
128x16, 512x4 and 2048x1 require 5120 training continuations, rather than a dense
2048x16 pool.  Every architecture receives the same selected family view.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from .bsde_data import simulate_forward_trace
from .bsde_types import BSDETrace
from .data import json_ready, source_hash
from .native import initialize_masses
from .types import seeded_rng, stable_hash

DEFAULT_ALLOCATIONS = ((128, 16), (512, 4), (2048, 1))
DEFAULT_VALIDATION = (256, 4)
WORKFLOW = "conditional-bsde-anchor-study-pool-v1"


def _plan(config, allocations, validation, seed):
    allocations = tuple(tuple(value) for value in allocations)
    validation = tuple(validation)
    values = (*allocations, validation)
    if (not allocations or any(len(value) != 2 for value in values)
            or any(type(n) is not int or n < 1 for value in values for n in value)):
        raise ValueError("Positive integer (anchors, continuations) allocations required")
    if (any(a[0] >= b[0] or a[1] <= b[1] for a, b in zip(allocations, allocations[1:]))
            or len({anchors * children for anchors, children in allocations}) != 1):
        raise ValueError("Increasing anchor counts and decreasing children at a fixed path budget required")
    if any(anchors % len(config.cases) for anchors, _ in values):
        raise ValueError("Study anchor counts must divide equally across initial profiles")
    seed = config.seed if seed is None else seed
    if type(seed) is not int or seed < 0:
        raise ValueError("Nonnegative integer study seed required")
    return allocations, validation, seed


def study_pool_identity(config, problem, *, allocations=DEFAULT_ALLOCATIONS,
                        validation=DEFAULT_VALIDATION, seed=None):
    """Return the architecture-independent pool identity and declared plan."""
    if problem not in ("ou", "dk"):
        raise ValueError("Study forward problem must be ou or dk")
    allocations, validation, seed = _plan(config, allocations, validation, seed)
    plan = {"workflow": WORKFLOW, "config_hash": config.config_hash,
            "target_id": config.target_id(problem), "problem": problem,
            "allocations": [list(value) for value in allocations],
            "validation": list(validation), "seed": seed,
            "training_path_budget": allocations[0][0] * allocations[0][1],
            "training_union_paths": sum(max(children for count, children in allocations
                                             if index < count)
                                        for index in range(allocations[-1][0]))}
    return {**plan, "pool_id": stable_hash(plan)}


def _anchor_plan(config, problem, split, allocations, validation, seed):
    count = allocations[-1][0] if split == "train" else validation[0]
    for index in range(count):
        case = "ou" if problem == "ou" else config.cases[index % len(config.cases)]
        children = (max(n for anchors, n in allocations if index < anchors)
                    if split == "train" else validation[1])
        anchor = stable_hash([WORKFLOW, seed, problem, split, index, case])[:24]
        initial_seed = int(seeded_rng(seed, "bsde-study-initial", problem, split,
                                     anchor).integers(0, 2**63 - 1))
        forward_seed = int(seeded_rng(seed, "bsde-study-continuations", problem, split,
                                     anchor).integers(0, 2**63 - 1))
        families = tuple(stable_hash(["bsde-study-family-v1", anchor, child])[:24]
                         for child in range(children))
        yield index, case, anchor, initial_seed, forward_seed, families


def _pool_path(directory, problem):
    if problem not in ("ou", "dk"):
        raise ValueError("Study forward problem must be ou or dk")
    return Path(directory) / f"{problem}_study_pool.h5"


def _check_pool(store, identity, fingerprint, require_existing=False):
    expected = {"workflow": WORKFLOW, "pool_id": identity["pool_id"],
                "config_hash": identity["config_hash"], "source_hash": fingerprint,
                "problem": identity["problem"], "target_id": identity["target_id"]}
    exists = require_existing or len(store) > 0 or len(store.attrs) > 0
    if any((exists and key not in store.attrs)
           or (key in store.attrs and store.attrs[key] != value)
           for key, value in expected.items()):
        raise ValueError("Study pool target, source or plan changed; use a new directory")
    return expected


def _check_anchor(group, case, anchor, families):
    if (not group.attrs.get("complete", False)
            or group.attrs.get("anchor_id") != anchor
            or group.attrs.get("case") != case
            or group.attrs.get("n_paths") != len(families)):
        raise ValueError("Invalid committed study anchor; resume generation or use a new directory")
    if (tuple(group["learner/anchor_ids"].asstr()[:]) != (anchor,) * len(families)
            or tuple(group["learner/family_ids"].asstr()[:]) != families):
        raise ValueError("Study anchor or continuation-family identity mismatch")


def _write_trace(group, trace):
    learner, oracle = group.create_group("learner"), group.create_group("oracle")
    learner.create_dataset("times", data=trace.times)
    for key in ("states", "increments", "active"):
        values = getattr(trace, key)
        learner.create_dataset(key, data=values, chunks=(1, *values.shape[1:]),
                               compression="gzip", shuffle=True)
    for key in ("anchor_ids", "family_ids"):
        learner.create_dataset(key, data=np.asarray(getattr(trace, key),
                                                    dtype=h5py.string_dtype()))
    oracle.create_dataset("proposals", data=trace.proposals,
                          chunks=(1, *trace.proposals.shape[1:]),
                          compression="gzip", shuffle=True)
    oracle.create_dataset("stopped_at", data=trace.stopped_at)
    oracle.attrs["not_network_inputs"] = True


def generate_study_pool(directory, config, problem, *, allocations=DEFAULT_ALLOCATIONS,
                        validation=DEFAULT_VALIDATION, seed=None, deadline=None):
    """Generate the union pool with atomic per-anchor commits and resumability."""
    allocations, validation, seed = _plan(config, allocations, validation, seed)
    identity = study_pool_identity(config, problem, allocations=allocations,
                                   validation=validation, seed=seed)
    path = _pool_path(directory, problem)
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint, tick = source_hash(), time.perf_counter()
    generated, paths = 0, 0
    with h5py.File(path, "a") as store:
        store.attrs.update(_check_pool(store, identity, fingerprint))
        store.attrs["config"] = json.dumps(config.to_dict(), sort_keys=True)
        store.attrs["plan"] = json.dumps(identity, sort_keys=True)
        store.attrs["oracle_metadata_separate"] = True
        for split in ("train", "validation"):
            parent = store.require_group(split)
            for stale in [name for name in parent if name.startswith("_pending_")]:
                del parent[stale]
            for index, case, anchor, initial_seed, forward_seed, families in _anchor_plan(
                    config, problem, split, allocations, validation, seed):
                name = f"anchor_{index:06d}"
                if name in parent:
                    _check_anchor(parent[name], case, anchor, families)
                    continue
                if deadline is not None and time.monotonic() >= deadline:
                    store.flush()
                    return {"complete": False, "problem": problem, "pool_id": identity["pool_id"],
                            "generated_this_call": generated, "paths_this_call": paths,
                            "seconds_this_call": time.perf_counter() - tick}
                initial = (seeded_rng(initial_seed, "ou-initial-anchor").normal(size=(1, 4))
                           if problem == "ou" else initialize_masses(
                               config.context(case), config.spec, 1, initial_seed))
                trace = simulate_forward_trace(
                    problem, config, np.repeat(initial, len(families), axis=0), forward_seed,
                    anchor_ids=(anchor,) * len(families), family_ids=families)
                pending = f"_pending_{name}"
                group = parent.create_group(pending)
                group.attrs.update(anchor_id=anchor, case=case, n_paths=len(families),
                                   initial_seed=initial_seed, forward_seed=forward_seed,
                                   diagnostics=json.dumps(json_ready(trace.diagnostics)))
                _write_trace(group, trace)
                store.flush()
                group.attrs["complete"] = True
                store.flush()
                parent.move(pending, name)
                store.flush()
                generated += 1
                paths += len(families)
        store.attrs["complete"] = True
    return {"complete": True, "problem": problem, "pool_id": identity["pool_id"],
            "generated_this_call": generated, "paths_this_call": paths,
            "seconds_this_call": time.perf_counter() - tick}


def load_study_pool(directory, config, problem, split, *, allocations=DEFAULT_ALLOCATIONS,
                    validation=DEFAULT_VALIDATION, allocation=None, seed=None,
                    include_proposals=False) -> BSDETrace:
    """Load one fixed-budget view, omitting oracle proposal arrays by default."""
    allocations, validation, seed = _plan(config, allocations, validation, seed)
    if split not in ("train", "validation"):
        raise ValueError("Study pools contain train and validation splits")
    if split == "train":
        if allocation is None or tuple(allocation) not in allocations:
            raise ValueError("Select a declared training allocation")
        selected = tuple(allocation)
    else:
        if allocation is not None:
            raise ValueError("Validation uses one fixed allocation, independently of training")
        selected = validation
    identity = study_pool_identity(config, problem, allocations=allocations,
                                   validation=validation, seed=seed)
    with h5py.File(_pool_path(directory, problem), "r") as store:
        _check_pool(store, identity, source_hash(), require_existing=True)
        if split not in store:
            raise ValueError("Study split is absent; resume generation")
        groups, anchor_ids, families, cases, seconds = [], [], [], [], 0.0
        for index, case, anchor, _, _, expected in _anchor_plan(
                config, problem, split, allocations, validation, seed):
            if index >= selected[0]:
                break
            name = f"anchor_{index:06d}"
            if name not in store[split]:
                raise ValueError("Study split is incomplete; resume generation")
            group = store[split][name]
            _check_anchor(group, case, anchor, expected)
            groups.append(group)
            anchor_ids.extend([anchor] * selected[1])
            families.extend(expected[:selected[1]])
            cases.extend([case] * selected[1])
            seconds += json.loads(group.attrs["diagnostics"])["generation_seconds"]
        if len(set(families)) != len(families):
            raise ValueError("Duplicate independent continuation families in study view")
        other = "validation" if split == "train" else "train"
        other_anchors = {item[2] for item in _anchor_plan(
            config, problem, other, allocations, validation, seed)}
        if set(anchor_ids) & other_anchors:
            raise ValueError("Initial-anchor leakage between study splits")

        def concatenate(key):
            return np.concatenate([group[key][:selected[1]] for group in groups])

        return BSDETrace(
            times=groups[0]["learner/times"][:], states=concatenate("learner/states"),
            increments=concatenate("learner/increments"), active=concatenate("learner/active"),
            anchor_ids=tuple(anchor_ids), family_ids=tuple(families), problem=problem,
            stopped_at=concatenate("oracle/stopped_at"),
            proposals=concatenate("oracle/proposals") if include_proposals else None,
            diagnostics={"problem": problem, "split": split, "cases": cases,
                         "config_hash": config.config_hash, "source_hash": store.attrs["source_hash"],
                         "pool_id": identity["pool_id"], "allocation": list(selected),
                         "independent_anchors": selected[0], "independent_paths": len(families),
                         "shared_union_anchor_simulation_seconds": seconds,
                         "generation_seconds": seconds,
                         "generation_cost_scope": "full union children of selected anchors, not view-only",
                         "proposals_are_learner_features": False})
