"""Independent configurations for conditional Deep BSDE moment prediction."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .native_types import NativeDKSpec
from .types import Context, QueryBank, stable_hash


@dataclass(frozen=True)
class BSDEConfig:
    name: str = "bsde-first-batch"
    seed: int = 20261024
    n_cells: int = 32
    n_particles: int = 2048
    diffusion: float = 1.0
    rho_stop: float = 0.01 / (2 * np.pi)
    horizon: float = 1.0
    query_times: tuple[float, ...] = (0.25, 0.5, 1.0)
    dt: float = 2 ** -7
    reference_dt: float = 2 ** -13
    cases: tuple[str, ...] = ("uniform", "nonuniform")
    optimizer_seeds: tuple[int, ...] = (0, 1, 2)
    width: int = 64
    train_anchors: int = 128
    train_continuations: int = 16
    validation_anchors: int = 32
    validation_continuations: int = 32
    ou_test_anchors: int = 256
    conditional_anchors: int = 8
    conditional_paths: int = 2048
    outer_paths: int = 8192
    refinement_paths: int = 32
    reference_batch_size: int = 64
    bootstrap_samples: int = 200
    tolerance: float = 0.05
    uncertainty_budget: float = 0.0125
    budget_seconds: float = 28800.0
    training: dict = field(default_factory=lambda: {
        "epochs": 100, "patience": 10, "batch_size": 32,
        "learning_rate": 1e-3, "weight_decay": 1e-5, "threads": 1,
        "gradient_clip": 100.0,
    })

    def __post_init__(self):
        defaults = {"epochs": 100, "patience": 10, "batch_size": 32,
                    "learning_rate": 1e-3, "weight_decay": 1e-5, "threads": 1,
                    "gradient_clip": 100.0}
        object.__setattr__(self, "training", {**defaults, **self.training})
        for key in ("query_times", "cases", "optimizer_seeds"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Nonnegative integer master seed required")
        self.spec
        self.queries
        if not np.isfinite(self.diffusion) or self.diffusion <= 0:
            raise ValueError("Positive finite diffusion required")
        integers = ("n_particles", "width", "train_anchors", "train_continuations",
                    "validation_anchors", "validation_continuations", "ou_test_anchors",
                    "conditional_anchors", "conditional_paths", "outer_paths",
                    "refinement_paths", "reference_batch_size", "bootstrap_samples")
        if any(not isinstance(getattr(self, k), int) or getattr(self, k) < 2 for k in integers):
            raise ValueError("BSDE path, anchor, architecture and bootstrap counts must be integers >=2")
        if (not self.cases or len(set(self.cases)) != len(self.cases)
                or any(case not in ("uniform", "nonuniform") for case in self.cases)):
            raise ValueError("The first BSDE workflow supports only noninteracting DK cases")
        if any(n % len(self.cases) for n in (self.train_anchors, self.validation_anchors)):
            raise ValueError("Training/validation anchors must divide equally across cases")
        if (not self.optimizer_seeds or len(set(self.optimizer_seeds)) != len(self.optimizer_seeds)
                or any(not isinstance(s, int) or s < 0 for s in self.optimizer_seeds)):
            raise ValueError("Distinct nonnegative integer optimizer seeds required")
        if (not np.isfinite(self.horizon) or self.horizon <= 0
                or not np.isclose(self.query_times[-1], self.horizon, atol=1e-12, rtol=0)):
            raise ValueError("The last BSDE query must equal its finite positive horizon")
        for step in (self.dt, self.reference_dt):
            if not np.isfinite(step) or step <= 0:
                raise ValueError("Positive finite forward/reference timesteps required")
            if any(not np.isclose(t / step, round(t / step), atol=1e-8, rtol=0)
                   for t in self.query_times):
                raise ValueError("Every query must lie on the forward and reference grids")
        if (self.dt < self.reference_dt
                or not np.isclose(self.dt / self.reference_dt, round(self.dt / self.reference_dt))):
            raise ValueError("The BSDE forward timestep must nest within the reference grid")
        if 2 * self.diffusion * self.dt / self.spec.h**2 > 1 + 1e-12:
            raise ValueError("BSDE forward Euler violates native diffusion CFL")
        if not 0 < self.uncertainty_budget < self.tolerance < np.inf:
            raise ValueError("Invalid BSDE accuracy budgets")
        if not np.isfinite(self.budget_seconds) or self.budget_seconds <= 0:
            raise ValueError("Positive BSDE invocation budget required")
        for key in ("epochs", "patience", "batch_size", "threads"):
            value = self.training.get(key, {"epochs": 100, "patience": 10, "batch_size": 32, "threads": 1}[key])
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"Positive integer training {key} required")
        for key in ("learning_rate", "gradient_clip"):
            if not np.isfinite(self.training[key]) or self.training[key] <= 0:
                raise ValueError(f"Positive finite training {key} required")
        if not np.isfinite(self.training["weight_decay"]) or self.training["weight_decay"] < 0:
            raise ValueError("Finite nonnegative weight decay required")
        if self.reference_batch_size * round(self.horizon / self.reference_dt) * self.n_cells * 8 > 512 * 1024**2:
            raise ValueError("Reduce reference_batch_size to bound coupled Brownian memory")

    @property
    def spec(self):
        return NativeDKSpec(n_cells=self.n_cells, rho_stop=self.rho_stop)

    @property
    def queries(self):
        return QueryBank(self.query_times)

    @property
    def times(self):
        return np.asarray((0.0, *self.query_times))

    @property
    def mode_rates(self):
        return 4 * self.diffusion / self.spec.h**2 * np.sin(
            np.pi * np.asarray((1, 1, 2, 2)) / self.n_cells)**2

    @property
    def observable_scale(self):
        return np.sinc(np.asarray((1, 1, 2, 2)) / self.n_cells)

    def context(self, case):
        if case not in self.cases:
            raise ValueError("Unsupported BSDE initial profile")
        return Context(cos_coeffs=(0.2 if case == "nonuniform" else 0.0, 0.0, 0.0),
                       n_particles=self.n_particles, diffusion=self.diffusion)

    def target_id(self, problem):
        if problem not in ("ou", "dk"):
            raise ValueError("Unknown BSDE problem")
        return stable_hash({"workflow": "conditional-bsde-moments-v1", "problem": problem,
                            "native": asdict(self.spec) if problem == "dk" else None,
                            "rates": self.mode_rates.tolist(), "n_particles": self.n_particles,
                            "diffusion": self.diffusion, "forward_dt": self.dt,
                            "observations": "exact-cell-averages-modes12-v1" if problem == "dk" else "normalized-ou4-v1",
                            "query_times": self.query_times})

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @property
    def config_hash(self):
        return stable_hash(self.to_dict())

    @classmethod
    def load(cls, path):
        path = Path(path)
        return cls(**(tomllib.loads(path.read_text()) if path.suffix == ".toml"
                      else json.loads(path.read_text())))


def bsde_preset(name):
    if name == "bsde-first-batch":
        return BSDEConfig()
    if name == "bsde-refinement":
        return replace(BSDEConfig(), name=name, dt=2**-8)
    if name == "bsde-smoke":
        return replace(BSDEConfig(), name=name, horizon=1/8, query_times=(1/32, 1/16, 1/8),
                       reference_dt=2**-10, width=16, optimizer_seeds=(0,),
                       train_anchors=16, train_continuations=2, validation_anchors=8,
                       validation_continuations=2, ou_test_anchors=16,
                       conditional_anchors=2, conditional_paths=16, outer_paths=32,
                       refinement_paths=2, reference_batch_size=16, bootstrap_samples=8,
                       budget_seconds=600., training={"epochs": 3, "patience": 3,
                           "batch_size": 8, "learning_rate": 1e-3, "weight_decay": 1e-5,
                           "threads": 1, "gradient_clip": 100.})
    raise ValueError("Unknown BSDE preset")
