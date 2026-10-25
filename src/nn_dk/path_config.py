"""Prespecified density-path experiments, distinct from moment predictors."""
from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .native_types import NativeDKSpec
from .types import LENGTH, Context, stable_hash

CASES = ("heat", "additive", "dk_uniform", "dk_nonuniform", "dk_interacting")


@dataclass(frozen=True)
class PathConfig:
    name: str = "path-smoke"
    seed: int = 2026102503
    n_cells: int = 32
    n_particles: int = 2048
    diffusion: float = 1.
    rho_stop: float = .01 / LENGTH
    horizon: float = .0625
    step_sizes: tuple[float, ...] = (2**-7,)
    reference_step: float = 2**-10
    query_times: tuple[float, ...] = (.03125, .0625)
    cases: tuple[str, ...] = CASES
    optimizer_seeds: tuple[int, ...] = (0,)
    width: int = 16
    depth: int = 3
    train_paths: int = 32
    validation_paths: int = 16
    test_paths: int = 64
    adam_steps: int = 200
    lbfgs_steps: int = 20
    lbfgs_max_iter: int = 5
    learning_rate: float = .003
    batch_size: int = 32
    n_bootstrap: int = 100
    tolerance: float = .05
    uncertainty_budget: float = .0125
    budget_seconds: float = 28800.

    def __post_init__(self):
        for field in ("step_sizes", "query_times", "cases", "optimizer_seeds"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        for field in ("seed", "n_cells", "n_particles", "width", "depth", "train_paths",
                      "validation_paths", "test_paths", "adam_steps", "lbfgs_steps",
                      "lbfgs_max_iter", "batch_size", "n_bootstrap"):
            v = getattr(self, field)
            minimum = 0 if field in ("seed", "adam_steps", "lbfgs_steps") else 1
            if type(v) is not int or v < minimum:
                raise ValueError(f"Invalid integer density-path setting {field}")
        if self.width < 2 or self.n_cells < 8 or self.adam_steps + self.lbfgs_steps < 1:
            raise ValueError("Require width>=2, cells>=8 and an optimizer allocation")
        for field in ("diffusion", "rho_stop", "horizon", "reference_step", "learning_rate",
                      "tolerance", "uncertainty_budget", "budget_seconds"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"Positive finite density-path setting {field} required")
        if not 0 < self.uncertainty_budget < self.tolerance < 1:
            raise ValueError("Require 0 < uncertainty budget < accuracy tolerance < 1")
        if (not self.cases or len(set(self.cases)) != len(self.cases)
                or set(self.cases) - set(CASES)):
            raise ValueError("Unique supported density-path cases required")
        if (not self.optimizer_seeds or len(set(self.optimizer_seeds)) != len(self.optimizer_seeds)
                or any(type(s) is not int or s < 0 for s in self.optimizer_seeds)):
            raise ValueError("Unique nonnegative optimizer seeds required")
        if (not self.step_sizes or len(set(self.step_sizes)) != len(self.step_sizes)
                or any(not math.isfinite(dt) or dt <= 0 for dt in self.step_sizes)
                or self.reference_step > min(self.step_sizes)):
            raise ValueError("Unique positive steps and a finer reference step required")
        if any(not math.isclose(dt/min(self.step_sizes), round(dt/min(self.step_sizes)),
                                rel_tol=0, abs_tol=1e-9) for dt in self.step_sizes):
            raise ValueError("Fit grids must nest in the finest fit Brownian grid")
        if (not self.query_times or any(not math.isfinite(t) or t <= 0 for t in self.query_times)
                or any(a >= b for a, b in zip(self.query_times, self.query_times[1:]))
                or self.query_times[-1] > self.horizon):
            raise ValueError("Positive increasing query times within the horizon required")
        for dt in (*self.step_sizes, self.reference_step):
            for time in (self.horizon, *self.query_times):
                if not math.isclose(time / dt, round(time / dt), rel_tol=0, abs_tol=1e-9):
                    raise ValueError("Every physical query must lie on every numerical grid")
            if not math.isclose(dt / self.reference_step, round(dt / self.reference_step),
                                rel_tol=0, abs_tol=1e-9):
                raise ValueError("Numerical steps must nest in the fine Brownian grid")
            for case in self.cases:
                c = self.context(case)
                velocity = abs(c.kappa) + 2 * abs(c.a2)
                if dt * (2*c.diffusion/self.spec.h**2 + velocity/self.spec.h) > 1 + 1e-12:
                    raise ValueError("Density-path Euler discretization violates native CFL")
        if any(4*self.diffusion*dt/self.spec.h**2 >= 1-1e-8 for dt in self.step_sizes):
            raise ValueError("The neural positive-pole ansatz requires dt*lambda_max < 1")

    @property
    def spec(self):
        return NativeDKSpec(n_cells=self.n_cells, rho_stop=self.rho_stop)

    @property
    def expected_models(self):
        return len(self.cases)*len(self.step_sizes)*len(self.optimizer_seeds)

    @property
    def config_hash(self):
        return stable_hash(self.to_dict())

    def problem(self, case):
        if case not in self.cases:
            raise ValueError("Undeclared density-path case")
        return case if case in ("heat", "additive") else "dk"

    def context(self, case):
        if case not in CASES:
            raise ValueError("Unsupported density-path case")
        return Context(cos_coeffs=(0., 0., 0.) if case == "dk_uniform" else (.2, 0., 0.),
                       kappa=.75 if case == "dk_interacting" else 0.,
                       a2=.1 if case == "dk_interacting" else 0.,
                       diffusion=self.diffusion, n_particles=self.n_particles)

    def target_id(self, case, dt):
        return stable_hash({"law": "causal-density-path-euler-ito-v1",
                            "problem": self.problem(case), "spec": asdict(self.spec),
                            "context": asdict(self.context(case)), "step": dt,
                            "stopping": self.spec.stopping if self.problem(case) == "dk" else "none"})

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path):
        path = Path(path)
        return cls(**(tomllib.loads(path.read_text()) if path.suffix == ".toml"
                      else json.loads(path.read_text())))


def path_preset(name):
    if name == "path-smoke":
        return PathConfig()
    if name in ("path-controls", "path-pilot", "path-first-batch"):
        return replace(PathConfig(), name=name, horizon=1.,
                       step_sizes=(2**-7,) if name == "path-first-batch" else (2**-7, 2**-8, 2**-9),
                       reference_step=2**-13,
                       query_times=(.25, .5, 1.), width=32, depth=4,
                       train_paths=256, validation_paths=64, test_paths=256,
                       optimizer_seeds=(0, 1, 2), adam_steps=1000, lbfgs_steps=100,
                       n_bootstrap=200,
                       cases=("heat", "additive") if name == "path-controls" else CASES)
    raise ValueError("Unknown density-path preset")
