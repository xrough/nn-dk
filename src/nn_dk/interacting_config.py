"""Full-state generator moments with interactions; no moment-closure assumption."""
from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .native_types import NativeDKSpec
from .types import Context, QueryBank, stable_hash


def interacting_target_id(spec, context, horizon):
    """Physical continuous target, distinct from its Euler reference polygons."""
    return stable_hash({
        "kind": "continuous-interacting-native-dk-generator-moments-v1",
        "law": "continuous-stopped-finite-volume-dk-v1",
        "stopping": "continuous-first-hitting-freeze-v1",
        "n_cells": spec.n_cells, "length": spec.length, "rho_stop": spec.rho_stop,
        "state_representation": spec.state_representation,
        "diffusion": context.diffusion, "n_particles": context.n_particles,
        "kappa": context.kappa, "a2": context.a2, "horizon": float(horizon),
        "observation": "exact-cell-integrated-modes12-v1",
        "covariance_scope": "conditional-and-ensemble-same-time-full-4x4-v1"})


@dataclass(frozen=True)
class InteractingStudyConfig:
    name: str = "interacting-moments-smoke"
    seed: int = 2026102505
    n_cells: int = 32
    n_particles: int = 2048
    diffusion: float = 1.
    kappa: float = .75
    a2: float = .1
    rho_stop: float = .01/(2*np.pi)
    horizon: float = .0625
    query_times: tuple[float, ...] = (.015625, .03125, .0625)
    cases: tuple[str, ...] = ("uniform", "nonuniform")
    optimizer_seeds: tuple[int, ...] = (0,)
    width: int = 32
    state_step: float = 2**-9
    reference_step: float = 2**-10
    reference_factors: tuple[int, ...] = (1, 2, 4)
    train_paths: int = 32
    validation_paths: int = 16
    state_fractions: tuple[float, ...] = (0., .25, .5, 1.)
    horizon_samples: int = 4
    adam_steps: int = 200
    lbfgs_steps: int = 10
    lbfgs_max_iter: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 64
    validation_every: int = 10
    conditional_anchors: int = 2
    conditional_paths: int = 128
    ensemble_paths: int = 512
    ensemble_initial_paths: int = 512
    refinement_paths: int = 64
    reference_batch_size: int = 16
    n_bootstrap: int = 100
    tolerance: float = .05
    uncertainty_budget: float = .0125
    budget_seconds: float = 28800.

    def __post_init__(self):
        for key in ("query_times", "cases", "optimizer_seeds", "reference_factors", "state_fractions"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A nonempty interacting experiment name is required")
        for key in ("seed", "n_cells", "n_particles", "width", "train_paths", "validation_paths",
                    "horizon_samples", "adam_steps", "lbfgs_steps", "lbfgs_max_iter", "batch_size",
                    "validation_every", "conditional_anchors", "conditional_paths", "ensemble_paths",
                    "ensemble_initial_paths", "refinement_paths", "reference_batch_size", "n_bootstrap"):
            value = getattr(self, key)
            minimum = (0 if key in ("seed", "adam_steps", "lbfgs_steps") else
                       2 if key in ("conditional_paths", "ensemble_paths", "ensemble_initial_paths",
                                    "refinement_paths", "n_bootstrap") else 1)
            if type(value) is not int or value < minimum:
                raise ValueError(f"Invalid integer interacting setting: {key}")
        if self.width < 2 or self.n_particles < 2 or self.adam_steps+self.lbfgs_steps < 1:
            raise ValueError("Require width/N >=2 and a nonempty optimizer allocation")
        if self.refinement_paths > min(self.conditional_paths, self.ensemble_paths):
            raise ValueError("Reference refinement paths cannot exceed either independent ensemble allocation")
        for key in ("diffusion", "rho_stop", "horizon", "state_step", "reference_step",
                    "learning_rate", "budget_seconds", "tolerance", "uncertainty_budget"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite interacting setting required: {key}")
        if any(type(getattr(self, key)) not in (int, float)
               or not math.isfinite(getattr(self, key)) for key in ("kappa", "a2")):
            raise ValueError("Finite physical interaction coefficients are required")
        if self.cases != ("uniform", "nonuniform"):
            raise ValueError("Retain both initial profile laws in this interacting comparison")
        if (not self.optimizer_seeds or len(set(self.optimizer_seeds)) != len(self.optimizer_seeds)
                or any(type(v) is not int or v < 0 for v in self.optimizer_seeds)):
            raise ValueError("Distinct nonnegative optimizer seeds are required")
        if not 0 < self.uncertainty_budget < self.tolerance < 1:
            raise ValueError("Require 0 < uncertainty budget < accuracy tolerance < 1")
        if len(self.query_times) != 3 or not np.allclose(
                self.query_times, np.asarray((.25, .5, 1.))*self.horizon,
                atol=1e-12, rtol=0):
            raise ValueError("Use quarter, half and final physical query times")
        if (not self.state_fractions or self.state_fractions[0] != 0
                or self.state_fractions[-1] != 1
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.state_fractions)
                or any(a >= b for a, b in zip(self.state_fractions, self.state_fractions[1:]))):
            raise ValueError("Increasing collocation snapshot fractions from zero to one required")
        if (not self.reference_factors or self.reference_factors[0] != 1
                or len(self.reference_factors) < 2
                or any(type(v) is not int or v < 1 for v in self.reference_factors)
                or any(b % a != 0 or b <= a for a, b in
                       zip(self.reference_factors, self.reference_factors[1:]))):
            raise ValueError("Nested reference refinement factors starting at one required")
        spec = self.spec
        velocity = abs(self.kappa)+2*abs(self.a2)
        if velocity*spec.h > 2*self.diffusion+1e-12:
            raise ValueError("Centered interacting flux violates native cell Peclet condition")
        for step in (self.state_step, self.reference_step):
            if step*(2*self.diffusion/spec.h**2+velocity/spec.h) > 1+1e-12:
                raise ValueError("Interacting Euler reference violates native CFL")
            for time in (self.horizon, *self.query_times, *self.state_times):
                if not math.isclose(time/step, round(time/step), rel_tol=0, abs_tol=1e-9):
                    raise ValueError("Queries and state snapshots must lie on the numerical grids")

    @property
    def spec(self):
        return NativeDKSpec(self.n_cells, self.rho_stop)

    @property
    def physics(self):
        return Context(kappa=self.kappa, a2=self.a2, diffusion=self.diffusion,
                       n_particles=self.n_particles)

    def context(self, case):
        if case not in self.cases:
            raise ValueError("Undeclared interacting initial-profile case")
        return Context(cos_coeffs=(.2, 0., 0.) if case == "nonuniform" else (0., 0., 0.),
                       kappa=self.kappa, a2=self.a2, diffusion=self.diffusion,
                       n_particles=self.n_particles)

    @property
    def state_times(self):
        return tuple(self.horizon*f for f in self.state_fractions)

    @property
    def queries(self):
        return QueryBank(self.query_times)

    @property
    def target_id(self):
        return interacting_target_id(self.spec, self.physics, self.horizon)

    @property
    def config_hash(self):
        return stable_hash(self.to_dict())

    @property
    def expected_models(self):
        return len(self.optimizer_seeds)

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path):
        path = Path(path)
        return cls(**(tomllib.loads(path.read_text()) if path.suffix == ".toml"
                      else json.loads(path.read_text())))


def interacting_preset(name):
    if name == "interacting-moments-smoke":
        return InteractingStudyConfig()
    if name == "interacting-moments-pilot":
        return replace(InteractingStudyConfig(), name=name, horizon=1., query_times=(.25, .5, 1.),
                       state_step=2**-9, reference_step=2**-13, train_paths=256,
                       validation_paths=64, optimizer_seeds=(0, 1, 2), adam_steps=1000,
                       lbfgs_steps=50, conditional_anchors=4, conditional_paths=1024,
                       ensemble_paths=8192, ensemble_initial_paths=8192, refinement_paths=256,
                       n_bootstrap=200)
    raise ValueError("Unknown interacting-moment preset")
