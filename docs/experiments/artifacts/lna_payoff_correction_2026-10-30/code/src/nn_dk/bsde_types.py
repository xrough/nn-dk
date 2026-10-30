"""Causal forward traces and same-time moment contracts for Deep BSDE."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .types import stable_hash


@dataclass
class BSDETrace:
    times: np.ndarray
    states: np.ndarray
    increments: np.ndarray
    active: np.ndarray
    anchor_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    problem: str
    stopped_at: np.ndarray
    proposals: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.states = np.asarray(self.states, dtype=np.float64)
        self.increments = np.asarray(self.increments, dtype=np.float64)
        self.active = np.asarray(self.active, dtype=bool)
        self.stopped_at = np.asarray(self.stopped_at, dtype=np.float64)
        self.anchor_ids, self.family_ids = tuple(self.anchor_ids), tuple(self.family_ids)
        if self.problem not in ("ou", "dk") or self.states.ndim != 3:
            raise ValueError("A supported BSDE problem and (paths,nodes,state) array are required")
        n, nodes, dimension = self.states.shape
        if n < 1 or nodes < 2 or self.times.shape != (nodes,) or self.times[0] != 0:
            raise ValueError("BSDE traces require positive paths and a time grid starting at zero")
        if (not np.all(np.isfinite(self.times)) or np.any(np.diff(self.times) <= 0)
                or not np.allclose(np.diff(self.times), self.times[1], atol=1e-12, rtol=0)):
            raise ValueError("Uniform finite BSDE time grid required")
        if (self.increments.ndim != 3 or self.increments.shape[:2] != (n, nodes-1)
                or self.active.shape != (n, nodes-1)):
            raise ValueError("Pre-step active masks and Brownian increments must match the trace")
        if (len(self.anchor_ids) != n or len(self.family_ids) != n
                or len(set(self.family_ids)) != n or self.stopped_at.shape != (n,)):
            raise ValueError("Per-path anchor IDs, unique continuation IDs and stopping times required")
        if (not np.all(np.isfinite(self.states)) or not np.all(np.isfinite(self.increments))
                or np.any(np.isnan(self.stopped_at)) or np.any(self.stopped_at < 0)):
            raise ValueError("Invalid BSDE states, increments or stopping times")
        if self.problem == "dk" and (dimension < 8 or np.any(self.states < -1e-12)
                or not np.allclose(self.states.sum(-1), 1., atol=1e-10, rtol=0)):
            raise ValueError("Native BSDE states must be nonnegative conservative masses")
        if self.problem == "dk" and self.increments.shape[-1] != dimension:
            raise ValueError("Native Brownian driver and finite-volume face counts must agree")
        if self.problem == "ou" and (dimension != 4 or self.increments.shape[-1] != 4):
            raise ValueError("The calibration OU trace has four states and Brownian drivers")
        if self.problem == "ou" and (not np.all(self.active) or np.any(np.isfinite(self.stopped_at))):
            raise ValueError("OU calibration is unstopped")
        finite = np.isfinite(self.stopped_at)
        if np.any(self.stopped_at[finite] > self.times[-1]):
            raise ValueError("Finite stopping times cannot exceed the trace horizon")
        if not np.array_equal(self.active, self.times[None, :-1] < self.stopped_at[:, None]):
            raise ValueError("Active-at-start masks must agree with declared stopping times")
        for i in np.flatnonzero(finite):
            node = int(np.searchsorted(self.times, self.stopped_at[i], side="left"))
            if not np.allclose(self.states[i, node:], self.states[i, node], atol=1e-12, rtol=0):
                raise ValueError("Absorbed BSDE states must remain frozen")
        anchors = {}
        for i, identifier in enumerate(self.anchor_ids):
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Nonempty initial-anchor identifiers required")
            if identifier in anchors and not np.array_equal(self.initial[i], self.initial[anchors[identifier]]):
                raise ValueError("Continuations of an initial anchor must share its initial state")
            anchors.setdefault(identifier, i)
        if any(not isinstance(identifier, str) or not identifier for identifier in self.family_ids):
            raise ValueError("Nonempty continuation-family identifiers required")
        if self.proposals is not None:
            self.proposals = np.asarray(self.proposals, dtype=np.float64)
            if self.proposals.shape != (n, nodes-1, dimension) or not np.all(np.isfinite(self.proposals)):
                raise ValueError("Finite uncensored pre-stopping proposals must match the trace")

    @property
    def dt(self):
        return float(self.times[1] - self.times[0])

    @property
    def initial(self):
        return self.states[:, 0]

    @property
    def n_paths(self):
        return len(self.states)

    @property
    def fingerprint(self):
        digest = hashlib.sha256(stable_hash([self.problem, self.anchor_ids, self.family_ids]).encode())
        for value in (self.times, self.states, self.increments, self.active, self.stopped_at):
            digest.update(memoryview(np.ascontiguousarray(value)))
        return digest.hexdigest()

    def subset(self, indices):
        indices = np.atleast_1d(np.asarray(indices, dtype=int))
        return BSDETrace(self.times, self.states[indices], self.increments[indices],
                         self.active[indices], tuple(self.anchor_ids[i] for i in indices),
                         tuple(self.family_ids[i] for i in indices), self.problem,
                         self.stopped_at[indices], None if self.proposals is None else self.proposals[indices],
                         dict(self.diagnostics))


@dataclass
class MomentPrediction:
    mean: np.ndarray
    scaled_covariance: np.ndarray
    horizons: np.ndarray
    problem: str
    target_id: str
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.scaled_covariance = np.asarray(self.scaled_covariance, dtype=np.float64)
        self.horizons = np.asarray(self.horizons, dtype=np.float64)
        if (self.horizons.ndim != 1 or not len(self.horizons)
                or np.any(np.diff(self.horizons) <= 0)):
            raise ValueError("Nonempty strictly increasing prediction horizons required")
        if (self.problem not in ("ou", "dk") or self.mean.ndim < 2
                or self.mean.shape[-2:] != (len(self.horizons), 4)
                or self.scaled_covariance.shape != (*self.mean.shape[:-1], 4, 4)):
            raise ValueError("Same-time moment prediction dimensions do not match its horizon bank")
        if (not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scaled_covariance))
                or not np.all(np.isfinite(self.horizons)) or np.any(self.horizons < 0)):
            raise ValueError("Finite moments and nonnegative horizons required")
        if not np.allclose(self.scaled_covariance, self.scaled_covariance.swapaxes(-1, -2), atol=1e-9):
            raise ValueError("Conditional covariance must be symmetric")
        if np.linalg.eigvalsh(self.scaled_covariance).min() < -1e-9:
            raise ValueError("Invalid covariance; no spectral repair is performed")
