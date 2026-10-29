"""Configurations for continuous-generator neural conditional moment maps.

The BSDE adapter supplies existing geometry/observation helpers only.  Its Euler
settings and continuation counts are compatibility placeholders, never the
target law, training objective, or claimed reference allocation of this track.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .bsde_config import BSDEConfig
from .native_types import NativeDKSpec
from .types import QueryBank, stable_hash


def _training_defaults():
    return {"collocation_nodes": 65, "validation_nodes": 257, "adam_steps": 1000,
            "adam_learning_rate": 1e-3, "lbfgs_steps": 100, "lbfgs_max_iter": 10,
            "lbfgs_learning_rate": 1., "residual_tolerance": 1e-6, "threads": 1}


def _reference_defaults():
    return {"ode_step": 2**-16, "window_step": 2**-16, "tilt": 1.,
            "certificate_tolerance": 1e-8}


def generator_target_id(config, problem):
    """Identify continuous physics and queries independently of numerical steps.

    Accepts this study configuration or an existing ``BSDEConfig`` adapter.
    Euler step sizes and their polygon-crossing label must not leak into the
    continuous target identifier.  Physical changes and query/profile changes do.
    """
    if problem not in ("ou", "dk"):
        raise ValueError("The generator study supports only OU and native DK")
    spec = config.spec
    return stable_hash({
        "workflow": "continuous-generator-neural-moments-v1", "problem": problem,
        "law": "continuous-stopped-finite-volume-dk-v1" if problem == "dk"
        else "continuous-normalized-ou4-v1",
        "n_cells": spec.n_cells, "length": spec.length,
        "rho_stop": spec.rho_stop if problem == "dk" else None,
        "stopping": "continuous-first-hitting-freeze-v1" if problem == "dk" else "unstopped",
        "state_representation": spec.state_representation if problem == "dk" else "normalized-ou4-v1",
        "initialization": spec.initialization if problem == "dk" else "four-independent-standard-normal-coordinates",
        "initial_profile_laws": tuple({"case": case,
                                       "cos_coeffs": config.context(case).cos_coeffs,
                                       "sin_coeffs": config.context(case).sin_coeffs}
                                      for case in config.cases),
        "n_particles": config.n_particles,
        "diffusion": config.diffusion, "interaction": "zero",
        "observation": "exact-cell-integrated-modes12-v1" if problem == "dk" else "identity4-v1",
        "query_times": tuple(config.query_times),
        "covariance_scope": "conditional-and-ensemble-same-time-full-4x4-v1",
    })


@dataclass(frozen=True)
class GeneratorStudyConfig:
    name: str = "generator-accuracy"
    seed: int = 2026102601
    n_cells: int = 32
    n_particles: int = 2048
    diffusion: float = 1.
    rho_stop: float = .01 / (2 * np.pi)
    horizon: float = 1.
    query_times: tuple[float, ...] = (.25, .5, 1.)
    cases: tuple[str, ...] = ("uniform", "nonuniform")
    optimizer_seeds: tuple[int, ...] = (0, 1, 2)
    width: int = 32
    conditional_anchors: int = 32
    validation_anchors: int = 256
    ou_test_anchors: int = 1024
    tolerance: float = .05
    uncertainty_budget: float = .0125
    budget_seconds: float = 28800.
    training: dict = field(default_factory=_training_defaults)
    reference_settings: dict = field(default_factory=_reference_defaults)

    def __post_init__(self):
        for key in ("query_times", "cases", "optimizer_seeds"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        for key, defaults in (("training", _training_defaults()),
                              ("reference_settings", _reference_defaults())):
            settings = getattr(self, key)
            if not isinstance(settings, dict) or set(settings) - set(defaults):
                raise ValueError(f"Unsupported generator {key} keys")
            object.__setattr__(self, key, {**defaults, **settings})
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A nonempty generator study name is required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("A nonnegative integer generator master seed is required")
        if self.cases != ("uniform", "nonuniform"):
            raise ValueError("The generator accuracy study retains both declared initial-profile laws")
        if (not self.optimizer_seeds or len(set(self.optimizer_seeds)) != len(self.optimizer_seeds)
                or any(type(seed) is not int or seed < 0 for seed in self.optimizer_seeds)):
            raise ValueError("Distinct nonnegative integer optimizer seeds required")
        counts = ("n_particles", "width", "conditional_anchors", "validation_anchors", "ou_test_anchors")
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 2 for key in counts):
            raise ValueError("Generator architecture and diagnostic counts must be integers >=2")
        self.spec
        self.queries
        if (not np.isfinite(self.diffusion) or self.diffusion <= 0
                or not np.isfinite(self.horizon) or self.horizon <= 0):
            raise ValueError("Positive finite generator diffusion and horizon required")
        expected_times = np.asarray((.25, .5, 1.)) * self.horizon
        if (len(self.query_times) != 3
                or not np.allclose(self.query_times, expected_times, atol=1e-12, rtol=0)):
            raise ValueError("The generator study retains its three quarter/half/final horizons")
        if (not np.isfinite(self.tolerance) or not np.isfinite(self.uncertainty_budget)
                or not 0 < self.uncertainty_budget < self.tolerance):
            raise ValueError("Finite positive generator accuracy and uncertainty budgets required")
        if not np.isfinite(self.budget_seconds) or self.budget_seconds <= 0:
            raise ValueError("Positive finite generator invocation budget required")
        for key in ("collocation_nodes", "validation_nodes", "lbfgs_max_iter", "threads"):
            minimum = 3 if key.endswith("nodes") else 1
            if type(self.training[key]) is not int or self.training[key] < minimum:
                raise ValueError(f"Generator training {key} must be an integer >= {minimum}")
        if any(type(self.training[key]) is not int or self.training[key] < 0
               for key in ("adam_steps", "lbfgs_steps")):
            raise ValueError("Nonnegative integer generator optimizer step counts required")
        if self.training["adam_steps"] + self.training["lbfgs_steps"] < 1:
            raise ValueError("At least one generator optimizer step is required")
        for key in ("adam_learning_rate", "lbfgs_learning_rate", "residual_tolerance"):
            if not np.isfinite(self.training[key]) or self.training[key] <= 0:
                raise ValueError(f"Positive finite generator training {key} required")
        if any(not np.isfinite(value) or value <= 0 for value in self.reference_settings.values()):
            raise ValueError("Positive finite generator reference settings required")
        if self.reference_settings["tilt"] > 1:
            raise ValueError("Generator cooperative proof requires tilt <=1")
        for key in ("ode_step", "window_step"):
            step = self.reference_settings[key]
            if (step > self.horizon or not np.isclose(self.horizon / step,
                    round(self.horizon / step), atol=1e-8, rtol=0)):
                raise ValueError("Generator reference steps must divide the horizon")

    @property
    def spec(self):
        return NativeDKSpec(self.n_cells, self.rho_stop)

    @property
    def queries(self):
        return QueryBank(self.query_times)

    @property
    def times(self):
        return np.asarray((0., *self.query_times))

    @property
    def expected_models(self):
        return 2 * len(self.optimizer_seeds)

    @property
    def training_settings(self):
        return {"width": self.width, **self.training}

    @property
    def base_config(self):
        """Geometry adapter; Euler and path settings are explicitly not used."""
        exponent = max(7, math.ceil(math.log2(2 * self.diffusion * self.horizon / self.spec.h**2)))
        step = self.horizon / 2**exponent
        reference_step = step / 64
        batch = min(64, int(512 * 1024**2 / (self.horizon / reference_step * self.n_cells * 8)))
        return BSDEConfig(
            name=f"{self.name}-geometry-adapter", seed=self.seed, n_cells=self.n_cells,
            n_particles=self.n_particles, diffusion=self.diffusion, rho_stop=self.rho_stop,
            horizon=self.horizon, query_times=self.query_times, cases=self.cases,
            optimizer_seeds=self.optimizer_seeds, width=self.width, dt=step,
            reference_dt=reference_step, train_anchors=2, train_continuations=2,
            validation_anchors=self.validation_anchors * len(self.cases), validation_continuations=2,
            conditional_anchors=self.conditional_anchors, conditional_paths=2,
            ou_test_anchors=self.ou_test_anchors, outer_paths=2, refinement_paths=2,
            reference_batch_size=max(2, batch), bootstrap_samples=2,
            tolerance=self.tolerance, uncertainty_budget=self.uncertainty_budget,
            budget_seconds=self.budget_seconds, training={"threads": self.training["threads"]})

    def context(self, case):
        return self.base_config.context(case)

    def target_id(self, problem):
        return generator_target_id(self, problem)

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @property
    def config_hash(self):
        return stable_hash(self.to_dict())

    @classmethod
    def load(cls, path):
        path = Path(path)
        content = tomllib.loads(path.read_text()) if path.suffix == ".toml" else json.loads(path.read_text())
        return cls(**content)


def generator_preset(name):
    if name == "generator-accuracy":
        return GeneratorStudyConfig()
    if name == "generator-smoke":
        return replace(GeneratorStudyConfig(), name=name, optimizer_seeds=(0,), width=8,
                       conditional_anchors=2, validation_anchors=4, ou_test_anchors=8,
                       budget_seconds=600., training={"collocation_nodes": 9, "validation_nodes": 17,
                           "adam_steps": 2, "lbfgs_steps": 2, "lbfgs_max_iter": 2})
    raise ValueError("Unknown generator study preset")
