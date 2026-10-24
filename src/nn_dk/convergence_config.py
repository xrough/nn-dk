"""A-only native DK studies: explicit cases, axes, and finite experiment budgets."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .native_types import NativeDKSpec
from .types import Context, QueryBank, stable_hash


@dataclass(frozen=True)
class ConvergenceConfig:
    name: str = "a-first-batch"
    seed: int = 20261024
    cases: tuple[str, ...] = ("uniform",)
    n_cells: int = 32
    n_particles: int = 2048
    diffusion: float = 1.0
    rho_stop: float = 0.01 / (2 * np.pi)
    horizon: float = 1.0
    query_times: tuple[float, ...] = (0.25, 0.5, 1.0)
    reference_dt: float = 2 ** -13
    step_sizes: tuple[float, ...] = (2 ** -7, 2 ** -8, 2 ** -9, 2 ** -10)
    train_path_counts: tuple[int, ...] = (64,)
    widths: tuple[int, ...] = (32,)
    base_width: int = 32
    training_modes: tuple[str, ...] = ("oracle",)
    optimizer_seeds: tuple[int, ...] = (0, 1, 2)
    states_per_path: int = 32
    validation_paths: int = 64
    coefficient_test_paths: int = 64
    diagnostic_paths: int = 8192
    final_paths: int = 65536
    refinement_paths: int = 32
    proposal_dt: float = 2 ** -7
    trajectory_batch_size: int = 32
    evaluation_batch_size: int = 64
    bootstrap_samples: int = 200
    tolerance: float = 0.05
    uncertainty_budget: float = 0.0125
    budget_seconds: float = 28800.0
    particle_numbers: tuple[int, ...] = ()
    training: dict = field(default_factory=lambda: {
        "epochs": 100, "patience": 10, "batch_size": 128,
        "learning_rate": 1e-3, "weight_decay": 1e-5, "threads": 1,
    })

    def __post_init__(self):
        sequence_keys = ("cases", "query_times", "step_sizes", "train_path_counts", "widths",
                         "training_modes", "optimizer_seeds", "particle_numbers")
        for key in sequence_keys:
            object.__setattr__(self, key, tuple(getattr(self, key)))
        self.spec
        self.queries
        if not np.isclose(self.query_times[-1], self.horizon, rtol=0, atol=1e-12):
            raise ValueError("The last physical query must equal the declared horizon")
        if not self.cases or any(c not in ("uniform", "nonuniform", "interacting") for c in self.cases):
            raise ValueError("Unknown native DK case")
        if len(set(self.cases)) != len(self.cases):
            raise ValueError("Cases must be unique")
        if (any(m not in ("oracle", "increments") for m in self.training_modes) or not self.training_modes
                or len(set(self.training_modes)) != len(self.training_modes)):
            raise ValueError("Choose oracle and/or increments supervision")
        counts = ("n_particles", "states_per_path", "validation_paths", "coefficient_test_paths",
                  "diagnostic_paths", "final_paths", "refinement_paths", "trajectory_batch_size",
                  "evaluation_batch_size", "bootstrap_samples", "base_width")
        if any(not isinstance(getattr(self, k), int) or getattr(self, k) < 2 for k in counts):
            raise ValueError("Positive integer path, bootstrap, and architecture counts required")
        if not self.train_path_counts or any(not isinstance(n, int) or n < 2 for n in self.train_path_counts) or tuple(sorted(set(self.train_path_counts))) != self.train_path_counts:
            raise ValueError("Training path counts must be nested increasing sizes")
        if (self.base_width not in self.widths or any(not isinstance(w, int) or w < 2 for w in self.widths)
                or len(set(self.widths)) != len(self.widths) or not self.optimizer_seeds
                or len(set(self.optimizer_seeds)) != len(self.optimizer_seeds)
                or any(not isinstance(s, int) or s < 0 for s in self.optimizer_seeds)):
            raise ValueError("Architecture/seed axes are empty or inconsistent")
        if (any(not isinstance(n, int) or n < 2 for n in self.particle_numbers)
                or len(set(self.particle_numbers)) != len(self.particle_numbers)):
            raise ValueError("Particle-number follow-ups must be distinct positive integer counts")
        if not 0 < self.uncertainty_budget < self.tolerance < np.inf or not np.isfinite(self.budget_seconds) or self.budget_seconds <= 0:
            raise ValueError("Invalid accuracy or wall-clock budget")
        if not np.isfinite(self.diffusion) or self.diffusion <= 0:
            raise ValueError("Positive diffusion required")
        if not self.step_sizes or tuple(sorted(set(self.step_sizes), reverse=True)) != self.step_sizes:
            raise ValueError("Use a unique decreasing timestep ladder")
        grids = (self.reference_dt, self.reference_dt / 2, self.reference_dt / 4,
                 self.proposal_dt, *self.step_sizes)
        if any(not np.isfinite(t) or t <= 0 for t in grids) or not np.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("Finite positive integration times required")
        for dt in grids:
            for t in (*self.query_times, self.horizon):
                if t > self.horizon or not np.isclose(t / dt, round(t / dt), rtol=0, atol=1e-8):
                    raise ValueError("All physical query times must lie on every integration grid")
        for dt in (*self.step_sizes, self.horizon / self.states_per_path):
            if not np.isclose(dt / self.reference_dt, round(dt / self.reference_dt), atol=1e-8, rtol=0):
                raise ValueError("Reference increments and state sampling must nest within the study grid")
        # Bound the shared fine-increment buffer rather than silently allocating gigabytes.
        buffer_bytes = self.evaluation_batch_size * round(self.horizon / self.reference_dt) * self.n_cells * 8
        if buffer_bytes > 512 * 1024 ** 2:
            raise ValueError("Reduce evaluation_batch_size: coupled Brownian buffer exceeds 512 MiB")

    @property
    def spec(self):
        return NativeDKSpec(self.n_cells, self.rho_stop)

    @property
    def queries(self):
        return QueryBank(self.query_times)

    @property
    def times(self):
        return np.asarray((0.0, *self.query_times))

    @property
    def sample_times(self):
        return np.arange(self.states_per_path) * self.horizon / self.states_per_path

    @property
    def config_hash(self):
        return stable_hash(asdict(self))

    def context(self, case, n_particles=None):
        if case not in ("uniform", "nonuniform", "interacting"):
            raise ValueError("Unknown native DK case")
        return Context(cos_coeffs=(0.0 if case == "uniform" else 0.2, 0., 0.),
                       kappa=0.75 if case == "interacting" else 0.,
                       a2=0.1 if case == "interacting" else 0.,
                       n_particles=n_particles or self.n_particles, diffusion=self.diffusion)

    def fit_specs(self):
        """Data axis at base width; capacity axis only at the largest dataset."""
        result = []
        for mode in self.training_modes:
            counts = self.train_path_counts if mode == "oracle" else (max(self.train_path_counts),)
            for n in counts:
                widths = self.widths if n == max(self.train_path_counts) and mode == "oracle" else (self.base_width,)
                for width in widths:
                    for seed in self.optimizer_seeds:
                        result.append({"mode": mode, "paths": n, "width": width, "seed": seed,
                                       "key": f"{mode}_n{n}_w{width}_s{seed}"})
        return result

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix == ".json":
            return cls(**json.loads(path.read_text()))
        with path.open("rb") as stream:
            return cls(**tomllib.load(stream))


def convergence_preset(name):
    if name == "a-first-batch":
        return ConvergenceConfig()
    if name == "a-smoke":
        return ConvergenceConfig(name=name, n_cells=8, n_particles=128,
            cases=("uniform", "nonuniform", "interacting"), horizon=0.125,
            query_times=(0.03125, 0.0625, 0.125), reference_dt=2 ** -10,
            step_sizes=(2 ** -6, 2 ** -7, 2 ** -8), proposal_dt=2 ** -6,
            train_path_counts=(8, 16), widths=(8, 16), base_width=16,
            training_modes=("oracle", "increments"), optimizer_seeds=(0,),
            states_per_path=4, validation_paths=4, coefficient_test_paths=4,
            diagnostic_paths=32, refinement_paths=4, bootstrap_samples=40,
            trajectory_batch_size=4, evaluation_batch_size=16,
            training={"epochs": 3, "patience": 3, "batch_size": 32,
                      "learning_rate": 1e-3, "weight_decay": 1e-5, "threads": 1})
    if name in ("a-full", "a-final"):
        return ConvergenceConfig(name=name, cases=("uniform", "nonuniform", "interacting"),
            train_path_counts=(64, 256, 1024, 4096), widths=(16, 32, 64),
            training_modes=("oracle", "increments"),
            diagnostic_paths=65536 if name == "a-final" else 8192,
            particle_numbers=(512, 2048, 8192))
    raise ValueError(f"Unknown convergence preset {name}")
