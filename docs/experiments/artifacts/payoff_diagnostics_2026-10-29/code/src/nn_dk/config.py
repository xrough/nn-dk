"""Validated, hashable experiment configurations; no implicit hyperparameter search."""

from __future__ import annotations

import copy
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .types import Context, QueryBank, TargetSpec, seeded_rng, stable_hash


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "pilot"
    seed: int = 20261023
    n_cells: int = 32
    n_particles: int = 512
    diffusion: float = 1.0
    dt: float = 2 ** -11
    save_interval: float = 2 ** -7
    lag: float = 2 ** -6
    horizon: float = 1.0
    query_times: tuple[float, ...] = (0.25, 0.5, 1.0)
    train_contexts: int = 64
    train_replicates: int = 16
    validation_contexts: int = 16
    validation_replicates: int = 32
    test_contexts: int = 8
    test_replicates: int = 512
    model_paths: int = 512
    calibration_paths: int = 256
    n_grid: int = 32
    budget_seconds: float = 8 * 3600
    bootstrap_samples: int = 200
    optimizer_seeds: tuple[int, ...] = (0, 1, 2)
    training: dict = field(default_factory=lambda: {
        "epochs": 100, "patience": 10, "batch_size": 128,
        "learning_rate": 1e-3, "weight_decay": 1e-5, "threads": 1,
    })

    def __post_init__(self):
        object.__setattr__(self, "query_times", tuple(self.query_times))
        object.__setattr__(self, "optimizer_seeds", tuple(self.optimizer_seeds))
        self.target
        self.queries
        if not np.isfinite(self.diffusion) or self.diffusion <= 0 or self.n_particles < 1:
            raise ValueError("Positive diffusion and particle count required")
        for key in ("train_contexts", "train_replicates", "validation_contexts", "validation_replicates", "test_contexts", "test_replicates", "model_paths", "calibration_paths", "n_grid", "bootstrap_samples"):
            if getattr(self, key) < (2 if "replicates" in key else 1):
                raise ValueError(f"Invalid count: {key}")
        if self.n_grid < 8 or self.budget_seconds <= 0 or not self.optimizer_seeds:
            raise ValueError("Grid, budget, and optimizer seeds must be nonempty/positive")
        if any(not np.isfinite(t) or t <= 0 for t in (self.dt, self.save_interval, self.lag, self.horizon)):
            raise ValueError("Finite positive time settings required")
        for a, b in ((self.save_interval, self.dt), (self.lag, self.save_interval), (self.horizon, self.lag)):
            if not np.isclose(a / b, round(a / b), atol=1e-10, rtol=0):
                raise ValueError("Reference, saved, learned, and horizon grids must be nested")
        if any(t > self.horizon or not np.isclose(t / self.lag, round(t / self.lag), atol=1e-10, rtol=0) for t in self.query_times):
            raise ValueError("Queries must be on the supported learned time grid")
        if self.train_replicates % 2 or self.validation_replicates % 2:
            raise ValueError("Even train/validation replicas required for disjoint covariance labels")

    @property
    def target(self) -> TargetSpec:
        return TargetSpec(n_cells=self.n_cells)

    @property
    def queries(self) -> QueryBank:
        return QueryBank(times=self.query_times)

    @property
    def times(self) -> np.ndarray:
        return np.arange(round(self.horizon / self.save_interval) + 1) * self.save_interval

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        if path.suffix == ".json":
            return cls(**json.loads(path.read_text()))
        with path.open("rb") as stream:
            return cls(**tomllib.load(stream))


def contexts_for(config: ExperimentConfig, split: str) -> list[Context]:
    counts = {"train": config.train_contexts, "validation": config.validation_contexts, "test": config.test_contexts}
    rng = seeded_rng(config.seed, "contexts", split)
    result = []
    for i in range(counts[split]):
        c = rng.uniform(-0.25, 0.25, 6)
        # Include a uniform start in each suite, but with independently drawn environments.
        if i == 0:
            c[:] = 0
        result.append(Context(tuple(c[:3]), tuple(c[3:]), float(rng.uniform(0, 1.5)),
                              float(rng.uniform(-0.25, 0.25)), config.n_particles, config.diffusion))
    return result


def smoke_config() -> ExperimentConfig:
    return ExperimentConfig(name="smoke", n_cells=16, horizon=0.5,
                            query_times=(0.125, 0.25, 0.5), train_contexts=8,
                            validation_contexts=2, test_contexts=2,
                            train_replicates=8, validation_replicates=8, test_replicates=8,
                            model_paths=16, calibration_paths=128, n_grid=16,
                            optimizer_seeds=(0,), bootstrap_samples=100,
                            training={"epochs": 3, "patience": 3, "batch_size": 128,
                                      "learning_rate": 1e-3, "weight_decay": 1e-5, "threads": 1})


def preset(name: str) -> ExperimentConfig:
    if name == "smoke":
        return smoke_config()
    values = copy.deepcopy(ExperimentConfig().to_dict())
    if name == "pilot":
        return ExperimentConfig(**values)
    if name == "n2048":
        values.update(name=name, n_particles=2048)
    elif name == "resolution64":
        values.update(name=name, n_cells=64, n_particles=2048, n_grid=64)
    elif name == "long-horizon":
        values.update(name=name, horizon=4.0, query_times=(1.0, 2.0, 4.0))
    else:
        raise ValueError(f"Unknown preset {name}")
    return ExperimentConfig(**values)
