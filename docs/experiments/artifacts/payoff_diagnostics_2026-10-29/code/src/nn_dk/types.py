"""Shared contracts. Every method uses the same target and observation identifiers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

LENGTH = 2.0 * np.pi


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Context:
    cos_coeffs: tuple[float, ...] = (0.0, 0.0, 0.0)
    sin_coeffs: tuple[float, ...] = (0.0, 0.0, 0.0)
    kappa: float = 0.0
    a2: float = 0.0
    n_particles: int = 512
    diffusion: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "cos_coeffs", tuple(float(x) for x in self.cos_coeffs))
        object.__setattr__(self, "sin_coeffs", tuple(float(x) for x in self.sin_coeffs))
        for name in ("kappa", "a2", "diffusion"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not isinstance(self.n_particles, (int, np.integer)):
            raise ValueError("Integer particle count required")
        object.__setattr__(self, "n_particles", int(self.n_particles))
        if len(self.cos_coeffs) != len(self.sin_coeffs) or not self.cos_coeffs:
            raise ValueError("Matching nonempty sine/cosine descriptors required")
        values = (*self.cos_coeffs, *self.sin_coeffs, self.kappa, self.a2, self.diffusion)
        if not np.all(np.isfinite(values)) or self.n_particles < 1 or self.diffusion <= 0:
            raise ValueError("Finite parameters, positive diffusion and particle count required")

    @property
    def context_id(self) -> str:
        return stable_hash(asdict(self))[:20]

    def features(self) -> np.ndarray:
        """Only declared prediction-time metadata; never trajectory outcomes."""
        return np.asarray((*self.cos_coeffs, *self.sin_coeffs, self.kappa, self.a2,
                           self.diffusion, np.log(self.n_particles)), dtype=np.float64)


@dataclass(frozen=True)
class TargetSpec:
    n_cells: int = 32
    law: str = "particle"
    observation: str = "periodic-cubic-bspline-v1"
    regime: str = "K"
    conditioning: str = "ensemble-start"
    length: float = LENGTH

    def __post_init__(self):
        if self.n_cells < 8 or self.law != "particle" or self.regime != "K":
            raise ValueError("v0.1 supports particle targets, regime K, and at least eight cells")
        if self.observation != "periodic-cubic-bspline-v1" or self.conditioning != "ensemble-start" or self.length != LENGTH:
            raise ValueError("Unsupported observation, conditioning, or domain")

    @property
    def target_id(self) -> str:
        return stable_hash(asdict(self))

    def require_compatible(self, other: TargetSpec) -> None:
        if self != other:
            raise ValueError("Incompatible target/observation/conditioning identifiers")


@dataclass(frozen=True)
class QueryBank:
    times: tuple[float, ...] = (0.25, 0.5, 1.0)
    modes: tuple[int, ...] = (1, 2)

    def __post_init__(self):
        object.__setattr__(self, "times", tuple(float(t) for t in self.times))
        object.__setattr__(self, "modes", tuple(self.modes))
        if not self.times or any(t <= 0 or not np.isfinite(t) for t in self.times) or tuple(sorted(set(self.times))) != self.times:
            raise ValueError("Query times must be finite, positive, unique, and increasing")
        if not self.modes or any(k < 1 or int(k) != k for k in self.modes):
            raise ValueError("Positive integer modes required")

    @property
    def dimension(self) -> int:
        return len(self.times) * len(self.modes) * 2

    @property
    def query_id(self) -> str:
        return stable_hash(asdict(self))

    def weights(self, n_cells: int) -> np.ndarray:
        if max(self.modes) >= n_cells / 2:
            raise ValueError("Observable modes must be below the coarse Nyquist frequency")
        x = LENGTH * np.arange(n_cells) / n_cells
        return np.sqrt(2.0) * np.asarray([f(k * x) for k in self.modes for f in (np.cos, np.sin)])

    def extract(self, batch: TrajectoryBatch) -> np.ndarray:
        indices = []
        for t in self.times:
            hits = np.flatnonzero(np.isclose(batch.times, t, rtol=0, atol=1e-10))
            if len(hits) != 1:
                raise ValueError(f"Query time {t} is not on the stored time grid")
            indices.append(hits[0])
        y = batch.states[:, indices] @ self.weights(batch.target.n_cells).T
        return y.reshape(len(batch.states), -1)


@dataclass
class TrajectoryBatch:
    times: np.ndarray
    states: np.ndarray
    target: TargetSpec
    valid: np.ndarray | None = None
    failure_times: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    particle_states: np.ndarray | None = None

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.states = np.asarray(self.states, dtype=np.float64)
        if self.states.ndim != 3 or self.states.shape[1:] != (len(self.times), self.target.n_cells):
            raise ValueError("States must have shape (paths, times, coarse cells)")
        if len(self.times) == 0 or not np.all(np.isfinite(self.times)) or np.any(np.diff(self.times) <= 0):
            raise ValueError("Finite increasing trajectory times required")
        n = len(self.states)
        self.valid = np.ones(n, dtype=bool) if self.valid is None else np.asarray(self.valid, dtype=bool)
        self.failure_times = np.full(n, np.nan) if self.failure_times is None else np.asarray(self.failure_times, dtype=float)
        if self.valid.shape != (n,) or self.failure_times.shape != (n,):
            raise ValueError("Validity and failure times must be per-path arrays")
        if not np.all(np.isfinite(self.states[self.valid])):
            raise ValueError("Valid paths must be finite")
        if np.any(self.states[self.valid] < -1e-12):
            raise ValueError("Negative-mass paths must be marked invalid")


@dataclass
class StatisticalPrediction:
    mean: np.ndarray
    scaled_covariance: np.ndarray
    target: TargetSpec
    query_bank: QueryBank
    context_id: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=float)
        self.scaled_covariance = np.asarray(self.scaled_covariance, dtype=float)
        d = self.query_bank.dimension
        if self.mean.shape != (d,) or self.scaled_covariance.shape != (d, d):
            raise ValueError("Prediction dimensions do not match the query bank")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scaled_covariance)):
            raise ValueError("Statistical predictions must be finite")


def seeded_rng(seed: int, *keys: str | int) -> np.random.Generator:
    """Stable independent streams, unaffected by ordering, process hash, or resumption."""
    digest = hashlib.sha256(json.dumps([seed, *keys], separators=(",", ":")).encode()).digest()
    entropy = np.frombuffer(digest, dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence(entropy))
