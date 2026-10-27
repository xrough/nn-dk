"""Contracts for native stopped DK; deliberately distinct from particle projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .types import LENGTH, Context, stable_hash


@dataclass(frozen=True)
class NativeDKSpec:
    n_cells: int = 32
    rho_stop: float = 0.01 / LENGTH
    length: float = LENGTH
    law: str = "stopped-discrete-dk-v1"
    state_representation: str = "finite-volume-mass-v1"
    initialization: str = "iid-particle-histogram-v1"
    stopping: str = "euler-polygon-first-crossing-freeze-v1"

    def __post_init__(self):
        if not isinstance(self.n_cells, (int, np.integer)) or self.n_cells < 8:
            raise ValueError("At least eight integer finite-volume cells required")
        if self.length != LENGTH or not np.isfinite(self.rho_stop) or not 0 < self.rho_stop < 1 / self.length:
            raise ValueError("Use the fixed torus and a positive subuniform stopping density")
        if (self.law, self.state_representation, self.initialization, self.stopping) != (
                "stopped-discrete-dk-v1", "finite-volume-mass-v1", "iid-particle-histogram-v1",
                "euler-polygon-first-crossing-freeze-v1"):
            raise ValueError("Unsupported native DK law, state, initialization, or stopping convention")

    @property
    def h(self):
        return self.length / self.n_cells

    @property
    def mass_stop(self):
        return self.h * self.rho_stop

    @property
    def target_id(self):
        return stable_hash(asdict(self))

    def require_compatible(self, other):
        if not isinstance(other, NativeDKSpec) or self != other:
            raise ValueError("Incompatible native DK target identifiers")


@dataclass
class DKCoefficients:
    """Covariance is S S^T BEFORE division by N; arrays may be NumPy or Torch."""

    drift: Any
    covariance: Any
    face_amplitudes: Any
    flux: Any = None


@dataclass
class NativeBatch:
    times: np.ndarray
    states: np.ndarray
    spec: NativeDKSpec
    context: Context
    stopped_at: np.ndarray
    family_ids: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.states = np.asarray(self.states, dtype=float)
        self.stopped_at = np.asarray(self.stopped_at, dtype=float)
        self.family_ids = tuple(self.family_ids)
        if self.states.ndim != 3 or self.states.shape[1:] != (len(self.times), self.spec.n_cells):
            raise ValueError("Native states must have shape (paths, times, finite-volume cells)")
        n = len(self.states)
        if n < 1 or self.stopped_at.shape != (n,) or len(self.family_ids) != n or len(set(self.family_ids)) != n:
            raise ValueError("Unique trajectory family IDs and per-path stopping times required")
        if not np.all(np.isfinite(self.states)) or np.any(self.states < -1e-12):
            raise ValueError("Stopped DK states must remain finite and nonnegative")
        if not np.all(np.isfinite(self.times)) or np.any(np.diff(self.times) <= 0) or self.times[0] < 0:
            raise ValueError("Finite increasing nonnegative native observation times required")
        if np.any(np.isnan(self.stopped_at)) or np.any(self.stopped_at < 0):
            raise ValueError("Use infinity for paths that have not stopped")
        if not np.allclose(self.states.sum(-1), 1, atol=1e-10, rtol=0):
            raise ValueError("Native finite-volume masses must conserve total mass")


@dataclass
class DKTrainingBatch:
    """State samples/proposals grouped by independent parent trajectories.

    ``active`` is determined at proposal START. Exit proposals are never removed.
    Oracle labels can be recomputed separately and are never network features.
    """

    states: np.ndarray
    spec: NativeDKSpec
    context: Context
    family_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    proposals: np.ndarray | None = None
    proposal_dt: float | None = None
    active: np.ndarray | None = None

    def __post_init__(self):
        self.states = np.asarray(self.states, dtype=float)
        self.family_ids, self.sample_ids = tuple(self.family_ids), tuple(self.sample_ids)
        if self.states.ndim != 2 or self.states.shape[1] != self.spec.n_cells or not len(self.states):
            raise ValueError("Training states must have shape (samples, cells)")
        n = len(self.states)
        if len(self.family_ids) != n or len(self.sample_ids) != n or len(set(self.sample_ids)) != n:
            raise ValueError("Per-sample family IDs and unique sample IDs required")
        if not np.all(np.isfinite(self.states)) or np.any(self.states < -1e-12):
            raise ValueError("Finite nonnegative training origins required")
        self.active = np.ones(n, dtype=bool) if self.active is None else np.asarray(self.active, dtype=bool)
        if self.active.shape != (n,):
            raise ValueError("Active-at-start flags must be per sample")
        if self.proposals is not None:
            self.proposals = np.asarray(self.proposals, dtype=float)
            if self.proposals.shape != self.states.shape or not np.all(np.isfinite(self.proposals)):
                raise ValueError("Finite uncensored proposals must match the origin states")
            if self.proposal_dt is None or not np.isfinite(self.proposal_dt) or self.proposal_dt <= 0:
                raise ValueError("A positive native proposal timestep is required")

    @property
    def independent_paths(self):
        return len(set(self.family_ids))


def physics_features(context: Context):
    """Autonomous generator features: no time or initial-profile descriptor.

    N is applied explicitly by the integrator, not hidden in the coefficient map.
    """
    return np.asarray((context.kappa, context.a2, context.diffusion), dtype=float)
