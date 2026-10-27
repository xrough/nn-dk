"""Fixed-path-budget factorial study of initial-state coverage and value structure."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from .bsde_config import BSDEConfig
from .types import stable_hash


def _base():
    return BSDEConfig(
        name="bsde-anchor-study-base", seed=20261025,
        validation_anchors=256, validation_continuations=4,
        ou_test_anchors=1024, conditional_anchors=32, conditional_paths=512,
    )


@dataclass(frozen=True)
class BSDEStudyConfig:
    name: str = "bsde-anchor-study"
    base: dict = field(default_factory=lambda: _base().to_dict())
    allocations: tuple[tuple[int, int], ...] = ((128, 16), (512, 4), (2048, 1))
    architectures: tuple[str, ...] = ("mlp", "structured")
    validation: tuple[int, int] = (256, 4)

    def __post_init__(self):
        object.__setattr__(self, "allocations", tuple(tuple(v) for v in self.allocations))
        object.__setattr__(self, "architectures", tuple(self.architectures))
        object.__setattr__(self, "validation", tuple(self.validation))
        config = self.base_config
        object.__setattr__(self, "base", config.to_dict())
        if not self.name or len(self.allocations) < 2:
            raise ValueError("A named study with at least two allocations is required")
        if any(len(v) != 2 or any(not isinstance(n, int) or n < 1 for n in v)
               for v in self.allocations):
            raise ValueError("Positive integer initial-anchor and continuation counts required")
        anchors = [a for a, _ in self.allocations]
        if anchors != sorted(set(anchors)) or any(a % len(config.cases) for a in anchors):
            raise ValueError("Study anchors must increase and balance the configured profiles")
        if len({a*c for a, c in self.allocations}) != 1:
            raise ValueError("The primary study holds the total training-path budget fixed")
        if (len(self.validation) != 2 or any(not isinstance(v, int) or v < 2 for v in self.validation)
                or self.validation != (config.validation_anchors, config.validation_continuations)):
            raise ValueError("Study validation counts must match the common base configuration")
        if self.architectures != ("mlp", "structured"):
            raise ValueError("The preregistered study compares MLP and structured value heads")

    @property
    def base_config(self):
        return BSDEConfig(**self.base)

    @property
    def training_paths(self):
        return self.allocations[0][0] * self.allocations[0][1]

    @property
    def expected_models(self):
        return 2 * len(self.allocations) * len(self.architectures) * len(self.base_config.optimizer_seeds)

    def to_dict(self):
        return json.loads(json.dumps({"name": self.name, "base": self.base,
                "allocations": [list(v) for v in self.allocations],
                "architectures": list(self.architectures), "validation": list(self.validation)}))

    @property
    def config_hash(self):
        return stable_hash(self.to_dict())

    @classmethod
    def load(cls, path):
        path = Path(path)
        return cls(**(tomllib.loads(path.read_text()) if path.suffix == ".toml"
                      else json.loads(path.read_text())))


def bsde_study_preset(name):
    if name == "bsde-anchor-study":
        return BSDEStudyConfig()
    if name == "bsde-study-smoke":
        base = replace(_base(), name="bsde-study-smoke-base", horizon=1/8,
                       query_times=(1/32, 1/16, 1/8), reference_dt=2**-10,
                       width=16, optimizer_seeds=(0,), train_anchors=4,
                       train_continuations=4, validation_anchors=8,
                       validation_continuations=2, ou_test_anchors=8,
                       conditional_anchors=2, conditional_paths=8, outer_paths=16,
                       refinement_paths=2, reference_batch_size=8, bootstrap_samples=8,
                       budget_seconds=600., training={"epochs": 2, "patience": 2,
                           "batch_size": 8, "learning_rate": 1e-3, "weight_decay": 1e-5,
                           "threads": 1, "gradient_clip": 100.})
        return BSDEStudyConfig(name=name, base=base.to_dict(),
                               allocations=((4, 4), (8, 2), (16, 1)), validation=(8, 2))
    raise ValueError("Unknown BSDE study preset")
