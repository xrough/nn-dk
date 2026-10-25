"""Continuous-generator targets and immutable scientific preset contracts."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nn_dk.bsde_config import BSDEConfig
from nn_dk.generator_config import GeneratorStudyConfig, generator_preset, generator_target_id


@pytest.mark.parametrize("name", ("generator-smoke", "generator-accuracy"))
def test_generator_presets_roundtrip_and_match_committed_tomls(tmp_path, name):
    config = generator_preset(name)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    assert GeneratorStudyConfig.load(path) == config
    assert GeneratorStudyConfig.load(path).config_hash == config.config_hash
    assert GeneratorStudyConfig.load(Path("configs") / f"{name.replace('-', '_')}.toml") == config
    assert config.target_id("dk") != config.target_id("ou")
    assert config.expected_models == 2 * len(config.optimizer_seeds)
    assert config.training_settings["width"] == config.width


def test_accuracy_preset_retains_all_profiles_horizons_seeds_and_thresholds():
    config = generator_preset("generator-accuracy")
    assert config.seed not in (20261024, 20261025)
    assert (config.n_cells, config.n_particles, config.diffusion, config.horizon) == (32, 2048, 1., 1.)
    assert config.query_times == (.25, .5, 1.)
    assert config.cases == ("uniform", "nonuniform")
    assert config.optimizer_seeds == (0, 1, 2)
    assert config.expected_models == 6
    assert config.conditional_anchors == 32
    assert config.validation_anchors == 256
    assert config.ou_test_anchors == 1024
    assert config.tolerance == .05 and config.uncertainty_budget == .0125
    assert config.budget_seconds == 8 * 3600
    assert config.reference_settings == {"ode_step": 2**-16, "window_step": 2**-16,
                                         "tilt": 1., "certificate_tolerance": 1e-8}
    assert config.training_settings == {
        "width": 32, "collocation_nodes": 65, "validation_nodes": 257,
        "adam_steps": 1000, "adam_learning_rate": 1e-3, "lbfgs_steps": 100,
        "lbfgs_max_iter": 10, "lbfgs_learning_rate": 1., "residual_tolerance": 1e-6,
        "threads": 1}


def test_bsde_adapter_supplies_geometry_without_defining_generator_target():
    config = GeneratorStudyConfig()
    base = config.base_config
    assert isinstance(base, BSDEConfig)
    assert base.spec == config.spec
    assert base.queries == config.queries
    assert base.width == config.width
    assert base.validation_anchors == len(config.cases) * config.validation_anchors
    assert base.conditional_anchors == config.conditional_anchors
    assert config.context("uniform") == base.context("uniform")
    assert config.context("nonuniform").cos_coeffs[0] == .2
    np.testing.assert_array_equal(config.times, (0., .25, .5, 1.))
    for problem in ("ou", "dk"):
        assert generator_target_id(base, problem) == config.target_id(problem)
        assert base.target_id(problem) != config.target_id(problem)
        assert generator_target_id(replace(base, dt=2**-8), problem) == config.target_id(problem)
        assert generator_target_id(replace(base, reference_dt=2**-14), problem) == config.target_id(problem)
        assert generator_target_id(replace(base, seed=base.seed + 1), problem) == config.target_id(problem)


def test_numerical_training_and_reference_settings_change_config_not_physics_target():
    config = GeneratorStudyConfig()
    variations = (replace(config, seed=config.seed + 1), replace(config, width=64),
                  replace(config, training={"adam_steps": 2000}),
                  replace(config, reference_settings={"ode_step": 2**-17}),
                  replace(config, conditional_anchors=64), replace(config, validation_anchors=128))
    for varied in variations:
        assert varied.config_hash != config.config_hash
        assert varied.target_id("dk") == config.target_id("dk")
        assert varied.target_id("ou") == config.target_id("ou")
    physical = (replace(config, n_particles=512), replace(config, n_cells=16),
                replace(config, diffusion=.5), replace(config, rho_stop=config.rho_stop * 2),
                replace(config, horizon=.5, query_times=(.125, .25, .5)))
    assert all(varied.target_id("dk") != config.target_id("dk") for varied in physical)


def test_partial_settings_merge_without_mutating_callers_or_other_configs():
    training, references = {"adam_steps": 2}, {"tilt": .5}
    config = GeneratorStudyConfig(training=training, reference_settings=references)
    training["adam_steps"] = 100
    references["tilt"] = 1.
    assert config.training["adam_steps"] == 2
    assert config.reference_settings["tilt"] == .5
    assert config.training["lbfgs_steps"] == 100
    assert GeneratorStudyConfig().training["adam_steps"] == 1000


@pytest.mark.parametrize("change", [
    {"name": ""}, {"seed": -1}, {"seed": True}, {"n_cells": 4}, {"n_particles": 1},
    {"n_particles": True}, {"diffusion": 0.}, {"diffusion": float("nan")},
    {"rho_stop": 0.}, {"horizon": 0.}, {"query_times": (.25, 1.)},
    {"query_times": (.1, .5, 1.)}, {"cases": ("uniform",)},
    {"optimizer_seeds": (0, 0)}, {"optimizer_seeds": (-1,)},
    {"optimizer_seeds": (True,)}, {"width": 1}, {"conditional_anchors": 1},
    {"validation_anchors": 1}, {"ou_test_anchors": 1},
    {"tolerance": .01}, {"uncertainty_budget": 0.}, {"tolerance": float("inf")},
    {"budget_seconds": -1.}, {"training": {"width": 64}},
    {"training": {"adam_steps": -1}}, {"training": {"adam_steps": 1.5}},
    {"training": {"adam_steps": 0, "lbfgs_steps": 0}},
    {"training": {"lbfgs_max_iter": 0}}, {"training": {"threads": 0}},
    {"training": {"collocation_nodes": 2}}, {"training": {"validation_nodes": 2}},
    {"training": {"adam_learning_rate": 0.}},
    {"training": {"lbfgs_learning_rate": float("nan")}},
    {"training": {"residual_tolerance": 0.}},
    {"reference_settings": {"unknown": 1}}, {"reference_settings": {"ode_step": -.1}},
    {"reference_settings": {"window_step": .003}}, {"reference_settings": {"tilt": 0.}},
    {"reference_settings": {"tilt": 2.}},
    {"reference_settings": {"certificate_tolerance": float("nan")}},
])
def test_generator_config_rejects_invalid_contracts(change):
    with pytest.raises(ValueError):
        GeneratorStudyConfig(**change)


def test_unknown_generator_problem_and_preset_rejected():
    with pytest.raises(ValueError, match="only OU"):
        generator_target_id(GeneratorStudyConfig(), "particle")
    with pytest.raises(ValueError, match="Unknown"):
        generator_preset("legacy")
