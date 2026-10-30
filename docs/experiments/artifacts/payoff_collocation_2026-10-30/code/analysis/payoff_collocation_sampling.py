"""Frozen, training-input-only proposals and grouped boundary stress states.

Stratified proposals change the chance of drawing a row, not the natural-pool
objective: every draw supplies p/q with p=1/number_of_rows. Repeated draws and
synthetic stress transforms never create new independent trajectory families.
No validation/reference outcomes or fitted-model residuals define these strata.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

SAMPLER_KIND = "importance-corrected-boundary-time-row-proposals-v1"
STRESS_KIND = "parent-grouped-simplex-boundary-stress-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_identity(value):
    value = np.ascontiguousarray(value, dtype="<f8")
    return {"shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def _edges(value, name, *, unit_interval=False):
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Finite increasing {name} required") from error
    if (not result or not np.isfinite(result).all() or min(result) <= 0
            or any(right <= left for left, right in zip(result, result[1:]))
            or (unit_interval and max(result) >= 1)):
        raise ValueError(f"Finite increasing {name} required")
    return result


@dataclass(frozen=True)
class BoundaryStrataConfig:
    """Fixed mass/time units and upper-open bin edges.

    At positive time use xi=min(m-m_stop)*sqrt(T/(width**2*time)), plus the
    remaining-time fraction time/T. Equality with an edge enters its upper bin.
    Absorption (any m<=m_stop) and exactly zero time have explicit labels.
    """

    horizon: float
    mass_stop: float
    boundary_width: float
    distance_edges: tuple[float, ...] = (.25, 1., 4.)
    time_edges: tuple[float, ...] = (.25, .5, .75)
    mass_tolerance: float = 1e-10

    def __post_init__(self):
        if (not np.isfinite((self.horizon, self.mass_stop, self.boundary_width,
                             self.mass_tolerance)).all()
                or self.horizon <= 0 or self.mass_stop < 0 or self.boundary_width <= 0
                or self.mass_tolerance <= 0):
            raise ValueError("Finite positive horizon, width/tolerance and nonnegative stop required")
        object.__setattr__(self, "distance_edges", _edges(self.distance_edges, "distance edges"))
        object.__setattr__(self, "time_edges", _edges(self.time_edges, "time edges", unit_interval=True))

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Complete frozen boundary strata configuration required")
        return cls(**value)


def _states(states, cfg):
    if not isinstance(cfg, BoundaryStrataConfig):
        raise TypeError("A frozen BoundaryStrataConfig is required")
    value = np.array(states, dtype=np.float64, copy=True)
    if (value.ndim != 2 or not len(value) or value.shape[1] < 2
            or not np.isfinite(value).all() or np.any(value < 0)
            or not np.allclose(value.sum(-1), 1., rtol=0, atol=cfg.mass_tolerance)
            or value.shape[1]*cfg.mass_stop >= 1):
        raise ValueError("Nonempty finite nonnegative unit-mass rows and feasible stopping mass required")
    return value


def _horizons(horizons, size, cfg):
    value = np.array(horizons, dtype=np.float64, copy=True)
    if (value.shape != (size,) or not np.isfinite(value).all()
            or np.any(value < 0) or np.any(value > cfg.horizon)):
        raise ValueError("One finite remaining horizon in [0,T] per row required")
    return value


def _families(family_ids, size):
    values = np.asarray(family_ids)
    if values.shape != (size,):
        raise ValueError("One nonempty parent family ID per row required")
    result = []
    for value in values.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str) or not value:
            raise ValueError("One nonempty parent family ID per row required")
        result.append(value)
    return np.asarray(result, dtype=str)


def boundary_time_strata(states, horizons, config):
    """Return deterministic row labels using only masses and remaining time.

    The log-coordinate calculation avoids division by zero and overflow in the
    short-time tail. All absorbed/zero-time labels are assigned before logs.
    """
    states = _states(states, config)
    horizons = _horizons(horizons, len(states), config)
    active, zero = (states > config.mass_stop).all(-1), horizons == 0
    labels = np.empty(len(states), dtype="<U64")
    labels[~active & zero] = "absorbed/zero-time"
    labels[~active & ~zero] = "absorbed/positive-time"
    labels[active & zero] = "interior/zero-time"
    positive = active & ~zero
    if positive.any():
        distance = states[positive].min(-1)-config.mass_stop
        log_xi = (np.log(distance)+.5*(math.log(config.horizon)
                  -2*math.log(config.boundary_width)-np.log(horizons[positive])))
        distance_bin = np.searchsorted(np.log(config.distance_edges), log_xi, side="right")
        time_bin = np.searchsorted(config.time_edges, horizons[positive]/config.horizon, side="right")
        labels[positive] = [f"active/time-{time}/distance-{distance}"
                            for time, distance in zip(time_bin, distance_bin, strict=True)]
    return labels


@dataclass(frozen=True)
class CollocationDraw:
    indices: np.ndarray
    probabilities: np.ndarray
    importance_weights: np.ndarray
    family_ids: np.ndarray
    labels: np.ndarray


class CollocationSampler:
    """Private-PCG64 IID proposals with replacement and exact p/q weights.

    ``uniform`` samples every row equally. ``stratified`` first allocates equal
    probability to each occupied stratum, then uniformly within that stratum.
    Empty strata receive no allocation; every actual row always has support.
    The weight belongs INSIDE the sample mean of each per-row loss, not inside
    a nonlinear robust loss and not divided by the observed weight sum.
    """

    def __init__(self, states, horizons, family_ids, config, *, mode="uniform", seed=0):
        if mode not in ("uniform", "stratified"):
            raise ValueError("Use uniform or stratified importance-corrected proposals")
        if type(seed) is not int or seed < 0:
            raise ValueError("A nonnegative integer private RNG seed is required")
        states = _states(states, config)
        horizons = _horizons(horizons, len(states), config)
        families = _families(family_ids, len(states))
        labels = boundary_time_strata(states, horizons, config)
        names, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        probabilities = (np.full(len(states), 1/len(states)) if mode == "uniform"
                         else 1/(len(names)*counts[inverse]))
        weights = (np.ones(len(states)) if mode == "uniform"
                   else (1/len(states))/probabilities)
        self._labels, self._families = labels, families
        self._probabilities, self._weights = probabilities, weights
        for array in (self._labels, self._families, self._probabilities, self._weights):
            array.setflags(write=False)
        self._identity = {"kind": SAMPLER_KIND, "mode": mode, "seed": seed,
            "config": config.to_dict(), "states": _array_identity(states),
            "horizons": _array_identity(horizons), "families_sha256": _digest(families.tolist()),
            "labels_sha256": _digest(labels.tolist()),
            "probabilities": _array_identity(probabilities),
            "importance_weights": _array_identity(weights), "rows": len(states),
            "occupied_strata": dict(zip(names.tolist(), counts.tolist(), strict=True)),
            "target": "uniform-over-original-pool-rows", "replacement": True,
            "draws_are_not_independent_trajectory_families": True}
        self._identity_hash = _digest(self._identity)
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._draws = 0

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    @property
    def probabilities(self):
        return self._probabilities.copy()

    @property
    def importance_weights(self):
        return self._weights.copy()

    @property
    def labels(self):
        return self._labels.copy()

    def draw(self, size):
        if type(size) is not int or size < 1:
            raise ValueError("A positive integer row allocation is required")
        indices = self._rng.choice(len(self._probabilities), size=size, replace=True,
                                   p=self._probabilities)
        self._draws += size
        return CollocationDraw(indices=indices, probabilities=self._probabilities[indices],
            importance_weights=self._weights[indices], family_ids=self._families[indices],
            labels=self._labels[indices])

    def state_dict(self):
        payload = {"schema_version": 1, "kind": SAMPLER_KIND, "identity": self.identity,
            "identity_sha256": self._identity_hash, "draws": self._draws,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state)}
        return {**payload, "state_sha256": _digest(payload)}

    def load_state_dict(self, state):
        required = {"schema_version", "kind", "identity", "identity_sha256", "draws",
                    "rng_state", "state_sha256"}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("Complete sampler resume state required")
        payload = {key: value for key, value in state.items() if key != "state_sha256"}
        try:
            checksum = _digest(payload)
        except (TypeError, ValueError) as error:
            raise ValueError("Sampler resume state checksum is invalid") from error
        if checksum != state["state_sha256"]:
            raise ValueError("Sampler resume state checksum changed")
        if (type(state["schema_version"]) is not int or state["schema_version"] != 1
                or state["kind"] != SAMPLER_KIND
                or state["identity"] != self._identity
                or state["identity_sha256"] != self._identity_hash
                or _digest(state["identity"]) != self._identity_hash):
            raise ValueError("Sampler pool, labels, configuration or proposal identity changed")
        if type(state["draws"]) is not int or state["draws"] < 0:
            raise ValueError("Nonnegative sampler draw count required")
        try:
            if state["rng_state"]["bit_generator"] != "PCG64":
                raise ValueError("Private sampler bit generator changed")
            candidate = np.random.Generator(np.random.PCG64())
            candidate.bit_generator.state = copy.deepcopy(state["rng_state"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid private sampler RNG state") from error
        self._rng, self._draws = candidate, state["draws"]


@dataclass(frozen=True)
class BoundaryStressConfig:
    """Explicit mass distances, cell IDs and times for grouped stress rows.

    A distance is in cell-MASS units: selected cells equal mass_stop+distance.
    Distance zero creates exact boundary/corner states. Positive distances give
    strict interior rows when the remaining slack is representable.
    """

    strata: BoundaryStrataConfig
    target_distances: tuple[float, ...]
    cell_sets: tuple[tuple[int, ...], ...]
    horizons: tuple[float, ...]

    def __post_init__(self):
        if not isinstance(self.strata, BoundaryStrataConfig):
            raise TypeError("Frozen boundary strata units are required")
        distances = tuple(float(value) for value in self.target_distances)
        horizons = tuple(float(value) for value in self.horizons)
        cells = tuple(tuple(cell for cell in group) for group in self.cell_sets)
        if (not distances or len(set(distances)) != len(distances)
                or not np.isfinite(distances).all() or min(distances) < 0
                or not horizons or len(set(horizons)) != len(horizons)
                or not np.isfinite(horizons).all() or min(horizons) < 0
                or max(horizons) > self.strata.horizon or not cells
                or any(not group or any(type(cell) is not int or cell < 0 for cell in group)
                       or len(set(group)) != len(group) for group in cells)):
            raise ValueError("Distinct finite distances/times and nonempty unique cell sets required")
        cells = tuple(tuple(sorted(group)) for group in cells)
        if len(set(cells)) != len(cells):
            raise ValueError("Distinct selected cell sets required")
        object.__setattr__(self, "target_distances", distances)
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "cell_sets", cells)

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Complete frozen boundary stress configuration required")
        return cls(**(value | {"strata": BoundaryStrataConfig.from_dict(value["strata"])}))


@dataclass(frozen=True)
class BoundaryStressPool:
    states: np.ndarray
    horizons: np.ndarray
    family_ids: np.ndarray
    parent_indices: np.ndarray
    selected_cells: np.ndarray
    target_distances: np.ndarray
    transform_ids: np.ndarray
    labels: np.ndarray
    identity: dict


def make_boundary_stress_pool(parents, family_ids, config):
    """Redistribute parent slack while fixing selected masses and family IDs.

    These are controlled conditional-state probes, NOT new native trajectories
    or draws from the physical initial law. No path is clipped/renormalized.
    All transforms of a parent retain its original independent-family group.
    """
    if not isinstance(config, BoundaryStressConfig):
        raise TypeError("A frozen BoundaryStressConfig is required")
    parents = _states(parents, config.strata)
    families = _families(family_ids, len(parents))
    stop, cells = config.strata.mass_stop, parents.shape[1]
    if np.any(parents <= stop):
        raise ValueError("Strict interior parent histograms are required for positive slack redistribution")
    if any(max(group) >= cells or len(group) >= cells for group in config.cell_sets):
        raise ValueError("Selected cell IDs must leave at least one unselected cell")
    for group in config.cell_sets:
        for distance in config.target_distances:
            if 1-cells*stop-len(group)*distance <= 0:
                raise ValueError("Infeasible target distances exhaust the remaining simplex slack")
    rows, times, ids, parent_indices, masks, distances, transforms = [], [], [], [], [], [], []
    for parent_index, parent in enumerate(parents):
        for group in config.cell_sets:
            selected = np.zeros(cells, dtype=bool)
            selected[list(group)] = True
            slack = parent[~selected]-stop
            for distance in config.target_distances:
                remaining = 1-cells*stop-len(group)*distance
                state = np.empty(cells, dtype=np.float64)
                state[selected] = stop+distance
                state[~selected] = stop+remaining*slack/slack.sum()
                if (np.any(state[~selected] <= stop)
                        or (distance > 0 and np.any(state[selected] <= stop))):
                    raise ValueError("Strict interior target allocation is not representable in float64")
                for horizon in config.horizons:
                    rows.append(state.copy())
                    times.append(horizon)
                    ids.append(families[parent_index])
                    parent_indices.append(parent_index)
                    masks.append(selected.copy())
                    distances.append(distance)
                    transforms.append(_digest({"parent_index": parent_index,
                        "parent_family": families[parent_index], "cells": list(group),
                        "distance": distance, "horizon": horizon}))
    states = _states(rows, config.strata)
    horizons = np.asarray(times, dtype=np.float64)
    labels = boundary_time_strata(states, horizons, config.strata)
    identity = {"kind": STRESS_KIND, "config": config.to_dict(),
        "parents": _array_identity(parents), "parent_family_sha256": _digest(families.tolist()),
        "states": _array_identity(states), "horizons": _array_identity(horizons),
        "labels_sha256": _digest(labels.tolist()), "transform_ids_sha256": _digest(transforms),
        "rows": len(rows), "independent_parent_families": len(set(families.tolist())),
        "physical_initial_law_sample": False, "transforms_are_independent_families": False}
    identity["identity_sha256"] = _digest(identity)
    arrays = (states, horizons, np.asarray(ids), np.asarray(parent_indices, dtype=np.int64),
              np.asarray(masks), np.asarray(distances), np.asarray(transforms), labels)
    for array in arrays:
        array.setflags(write=False)
    return BoundaryStressPool(*arrays, identity)
