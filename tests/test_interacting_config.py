"""Continuous target identity and prespecified interacting-moment experiment design."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nn_dk.generator_config import GeneratorStudyConfig
from nn_dk.generator_model import GeneratorMomentModel
from nn_dk.generator_reference import _physics
from nn_dk.interacting_config import (
    InteractingStudyConfig,
    interacting_preset,
    interacting_target_id,
)
from nn_dk.path_config import path_preset


@pytest.mark.parametrize("name", ["interacting-moments-smoke", "interacting-moments-pilot"])
def test_committed_presets_and_json_round_trip(name, tmp_path):
    config = interacting_preset(name)
    root = Path(__file__).resolve().parents[1]
    assert InteractingStudyConfig.load(root / "configs" / (name.replace("-", "_") + ".toml")) == config
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    restored = InteractingStudyConfig.load(path)
    assert restored == config
    assert restored.config_hash == config.config_hash
    assert restored.target_id == config.target_id
    assert isinstance(restored.cases, tuple)
    assert restored.cases == ("uniform", "nonuniform")
    assert restored.reference_factors == (1, 2, 4)
    assert restored.query_times == tuple(np.asarray((.25, .5, 1.))*restored.horizon)


def test_pilot_preserves_both_physics_profiles_and_fixes_independent_allocations():
    smoke = interacting_preset("interacting-moments-smoke")
    pilot = interacting_preset("interacting-moments-pilot")
    assert (smoke.horizon, smoke.train_paths, smoke.validation_paths) == (1/16, 32, 16)
    assert (pilot.horizon, pilot.state_step, pilot.reference_step) == (1., 2**-9, 2**-13)
    assert (pilot.train_paths, pilot.validation_paths) == (256, 64)
    assert (pilot.conditional_anchors, pilot.conditional_paths, pilot.ensemble_paths,
            pilot.ensemble_initial_paths, pilot.refinement_paths) == (4, 1024, 8192, 8192, 256)
    assert pilot.optimizer_seeds == (0, 1, 2)
    assert (pilot.adam_steps, pilot.lbfgs_steps, pilot.lbfgs_max_iter) == (1000, 50, 5)
    assert pilot.budget_seconds == 8*60*60
    assert smoke.seed != path_preset("path-interacting-smoke").seed
    assert pilot.seed != GeneratorStudyConfig().seed
    for config in (smoke, pilot):
        uniform, nonuniform = config.context("uniform"), config.context("nonuniform")
        assert uniform.cos_coeffs == (0., 0., 0.)
        assert nonuniform.cos_coeffs == (.2, 0., 0.)
        for context in (uniform, nonuniform):
            assert (context.kappa, context.a2, context.diffusion,
                    context.n_particles) == (.75, .1, 1., 2048)
        assert config.queries.times == config.query_times
        assert config.state_times == tuple(config.horizon*f for f in config.state_fractions)


def test_continuous_target_ignores_initial_profile_and_numerical_reference():
    config = InteractingStudyConfig()
    for context in (config.context("uniform"), config.context("nonuniform"),
                    replace(config.physics, cos_coeffs=(.3, -.1, .05), sin_coeffs=(.1, 0., 0.))):
        assert interacting_target_id(config.spec, context, config.horizon) == config.target_id
    for changes in ({"state_step": 2**-10}, {"reference_step": 2**-11},
                    {"reference_factors": (1, 4, 8)},
                    {"optimizer_seeds": (3, 7), "train_paths": 64, "width": 16},
                    {"seed": config.seed+1}, {"conditional_paths": 256},
                    {"state_fractions": (0., .5, 1.)}):
        changed = replace(config, **changes)
        assert changed.target_id == config.target_id
        assert changed.config_hash != config.config_hash


@pytest.mark.parametrize("changes", [
    {"n_cells": 16}, {"n_particles": 512}, {"diffusion": .8},
    {"kappa": -.75}, {"a2": -.1}, {"rho_stop": .02/(2*np.pi)},
    {"horizon": .125, "query_times": (.03125, .0625, .125)},
])
def test_target_identifies_physical_generator_geometry_boundary_and_duration(changes):
    config = InteractingStudyConfig()
    changed = replace(config, **changes)
    assert changed.target_id != config.target_id
    assert changed.config_hash != config.config_hash


def test_new_target_and_model_do_not_reuse_noninteracting_certificates():
    config = InteractingStudyConfig()
    legacy = GeneratorStudyConfig()
    assert config.target_id != legacy.target_id("dk")
    # Reusing either legacy reference or moment-family implementation must fail,
    # rather than silently dropping the interaction from this target.
    with pytest.raises(ValueError, match="noninteracting"):
        _physics(config, config.physics)
    with pytest.raises(ValueError, match="noninteracting"):
        GeneratorMomentModel(config, "dk", width=8)
    with pytest.raises(ValueError, match="Undeclared"):
        config.context("interacting")


@pytest.mark.parametrize("changes", [
    {"name": ""}, {"name": 1}, {"cases": ()}, {"cases": ("uniform",)},
    {"cases": ("nonuniform", "uniform")}, {"cases": ("uniform", "uniform")},
    {"optimizer_seeds": ()}, {"optimizer_seeds": (0, 0)}, {"optimizer_seeds": (-1,)},
    {"optimizer_seeds": (True,)}, {"seed": -1}, {"seed": True},
    {"n_cells": 4}, {"n_cells": True}, {"n_particles": 1}, {"width": 1},
    {"train_paths": 0}, {"validation_paths": 0}, {"horizon_samples": 0},
    {"adam_steps": -1}, {"lbfgs_steps": -1}, {"adam_steps": 0, "lbfgs_steps": 0},
    {"lbfgs_max_iter": 0}, {"validation_every": 0}, {"batch_size": True},
    {"reference_batch_size": 0}, {"tolerance": 1.}, {"uncertainty_budget": .05},
    {"rho_stop": 1/(2*np.pi)},
])
def test_invalid_design_and_optimizer_contracts_rejected(changes):
    with pytest.raises(ValueError):
        replace(InteractingStudyConfig(), **changes)


@pytest.mark.parametrize("key", ["diffusion", "rho_stop", "horizon", "state_step",
                                  "reference_step", "learning_rate", "budget_seconds",
                                  "tolerance", "uncertainty_budget"])
@pytest.mark.parametrize("value", [0., -.1, float("nan"), float("inf")])
def test_positive_finite_numerical_and_physical_parameters(key, value):
    with pytest.raises(ValueError):
        replace(InteractingStudyConfig(), **{key: value})


@pytest.mark.parametrize("key", ["kappa", "a2"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.75"])
def test_interaction_parameters_are_finite_real_numbers(key, value):
    with pytest.raises(ValueError):
        replace(InteractingStudyConfig(), **{key: value})


@pytest.mark.parametrize("key", ["conditional_paths", "ensemble_paths",
                                  "ensemble_initial_paths", "refinement_paths", "n_bootstrap"])
def test_statistical_allocations_require_multiple_independent_samples(key):
    with pytest.raises(ValueError):
        replace(InteractingStudyConfig(), **{key: 1})


@pytest.mark.parametrize("changes", [
    {"query_times": ()}, {"query_times": (.03125,)},
    {"query_times": (.0625, .03125, .015625)},
    {"query_times": (.015625, .03125, .125)},
    {"state_fractions": ()}, {"state_fractions": (.25, 1.)},
    {"state_fractions": (0., .5)}, {"state_fractions": (0., .5, .5, 1.)},
    {"state_fractions": (0., -.1, 1.)}, {"state_fractions": (0., float("nan"), 1.)},
    {"reference_factors": ()}, {"reference_factors": (1,)},
    {"reference_factors": (2, 4)}, {"reference_factors": (1, 2, 3)},
    {"reference_factors": (1, 2, 2)}, {"reference_factors": (1, True, 4)},
    {"state_step": 2**-5}, {"reference_step": 2**-5},
    {"state_step": .003}, {"reference_step": .003},
    {"kappa": 11.},
])
def test_query_snapshot_refinement_and_native_stability_contracts(changes):
    with pytest.raises(ValueError):
        replace(InteractingStudyConfig(), **changes)


def test_unknown_interacting_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        interacting_preset("generator-accuracy")


def test_loss_balancing_is_explicit_default_preserves_unscaled_objective():
    default = InteractingStudyConfig()
    balanced = replace(default, loss_balancing="training-initial-rms")
    assert default.loss_balancing == "none"
    assert default.loss_scale_floor == 1e-6
    assert balanced.target_id == default.target_id
    assert balanced.config_hash != default.config_hash


@pytest.mark.parametrize("changes", [
    {"loss_balancing": "validation-rms"}, {"loss_balancing": "adaptive"},
    {"loss_balancing": None}, {"loss_scale_floor": 0.},
    {"loss_scale_floor": -1.}, {"loss_scale_floor": float("nan")},
    {"loss_scale_floor": float("inf")}, {"loss_scale_floor": True},
])
def test_loss_scaling_safeguard_and_train_only_modes_are_validated(changes):
    with pytest.raises(ValueError):
        replace(InteractingStudyConfig(), **changes)
