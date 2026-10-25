"""Prespecified full-density targets and reproducible path-experiment inputs."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from nn_dk.native_types import NativeDKSpec
from nn_dk.path_config import CASES, PathConfig, path_preset
from nn_dk.types import LENGTH, TargetSpec


@pytest.mark.parametrize("name", ["path-smoke", "path-controls", "path-pilot", "path-first-batch"])
def test_presets_json_toml_round_trip_exactly(name, tmp_path):
    config = path_preset(name)
    saved = tmp_path / "configuration.json"
    saved.write_text(json.dumps(config.to_dict()))
    assert PathConfig.load(saved) == config
    assert PathConfig.load(saved).config_hash == config.config_hash
    assert PathConfig.load(Path("configs") / (name.replace("-", "_") + ".toml")) == config
    assert isinstance(config.step_sizes, tuple)
    assert isinstance(config.cases, tuple)
    assert isinstance(config.optimizer_seeds, tuple)


def test_ladder_allocations_are_fixed_and_case_counts_are_explicit():
    controls = path_preset("path-controls")
    pilot = path_preset("path-pilot")
    assert controls.cases == ("heat", "additive")
    assert pilot.cases == CASES
    assert pilot.step_sizes == controls.step_sizes == (2**-7, 2**-8, 2**-9)
    assert pilot.reference_step == 2**-13
    assert pilot.query_times == (.25, .5, 1.)
    assert pilot.horizon == 1
    assert pilot.optimizer_seeds == (0, 1, 2)
    assert (pilot.train_paths, pilot.validation_paths, pilot.test_paths) == (256, 64, 256)
    assert (pilot.width, pilot.depth, pilot.adam_steps, pilot.lbfgs_steps,
            pilot.lbfgs_max_iter) == (32, 4, 1000, 100, 5)
    assert pilot.expected_models == 45
    assert controls.expected_models == 18
    assert pilot.tolerance == .05 and pilot.uncertainty_budget == .0125
    assert pilot.budget_seconds == 8 * 60 * 60


def test_first_batch_preserves_all_cases_and_seeds_at_one_resolution():
    first = path_preset("path-first-batch")
    pilot = path_preset("path-pilot")
    assert first.cases == CASES
    assert first.optimizer_seeds == (0, 1, 2)
    assert first.step_sizes == (2**-7,)
    assert first.expected_models == 15
    assert replace(first, name=pilot.name, step_sizes=pilot.step_sizes) == pilot


def test_physics_and_initialization_are_declared_separately_from_noise():
    config = path_preset("path-pilot")
    assert config.spec == NativeDKSpec(n_cells=32, rho_stop=.01 / LENGTH)
    assert config.context("dk_uniform").cos_coeffs == (0., 0., 0.)
    assert config.context("dk_nonuniform").cos_coeffs == (.2, 0., 0.)
    interacting = config.context("dk_interacting")
    assert (interacting.kappa, interacting.a2, interacting.diffusion,
            interacting.n_particles) == (.75, .1, 1., 2048)
    assert config.problem("dk_interacting") == config.problem("dk_uniform") == "dk"
    assert config.problem("additive") == "additive"


def test_targets_distinguish_law_resolution_physics_and_initialization():
    config = path_preset("path-pilot")
    identities = {config.target_id(case, dt)
                  for case in config.cases for dt in config.step_sizes}
    assert len(identities) == len(config.cases) * len(config.step_sizes)
    assert config.spec.target_id not in identities
    assert TargetSpec().target_id not in identities
    assert replace(config, n_particles=512).target_id("dk_uniform", 2**-7) not in identities
    # Training allocation is not part of the physical law, but remains in the
    # reproducibility hash. Optimizer seeds cannot create different target laws.
    larger = replace(config, train_paths=512, optimizer_seeds=(7,))
    assert larger.target_id("dk_uniform", 2**-7) == config.target_id("dk_uniform", 2**-7)
    assert larger.config_hash != config.config_hash
    assert replace(config, depth=8).config_hash != config.config_hash


@pytest.mark.parametrize("updates", [
    {"cases": ()}, {"cases": ("heat", "heat")}, {"cases": ("moment",)},
    {"optimizer_seeds": ()}, {"optimizer_seeds": (0, 0)},
    {"optimizer_seeds": (-1,)}, {"optimizer_seeds": (True,)},
    {"step_sizes": ()}, {"step_sizes": (2**-7, 2**-7)},
    {"step_sizes": (0.,)}, {"step_sizes": (math.nan,)},
    {"reference_step": 2**-6},
])
def test_unique_supported_cases_seeds_and_steps_required(updates):
    with pytest.raises(ValueError):
        replace(PathConfig(), **updates)


@pytest.mark.parametrize("updates", [
    {"query_times": ()}, {"query_times": (0.,)},
    {"query_times": (.0625, .03125)}, {"query_times": (.03125, .03125)},
    {"query_times": (.125,)}, {"query_times": (math.inf,)},
    {"query_times": (.03, .0625)}, {"horizon": .06},
])
def test_queries_must_be_ordered_and_on_every_grid(updates):
    with pytest.raises(ValueError):
        replace(PathConfig(), **updates)


def test_brownian_nesting_not_just_matching_horizon_and_queries():
    # Both grids contain each physical query, but a 3/1024 step cannot be made
    # by summing a fixed integer number of 2/1024 Brownian increments.
    with pytest.raises(ValueError, match="nest"):
        replace(PathConfig(), horizon=96 / 1024, query_times=(48 / 1024, 96 / 1024),
                step_sizes=(3 / 1024,), reference_step=2 / 1024)


def test_cfl_and_positive_neural_pole_domain_are_both_checked():
    with pytest.raises(ValueError, match="CFL|positive-pole"):
        replace(PathConfig(), step_sizes=(1 / 32,))
    # This step satisfies native Euler CFL, but its Nyquist Euler pole is
    # negative and therefore outside the declared neural kernel family.
    with pytest.raises(ValueError, match="positive-pole"):
        replace(PathConfig(), step_sizes=(1 / 64,))


@pytest.mark.parametrize("field", ["diffusion", "rho_stop", "horizon", "reference_step",
                                    "learning_rate", "tolerance", "uncertainty_budget",
                                    "budget_seconds"])
@pytest.mark.parametrize("value", [0., -1., math.nan, math.inf])
def test_positive_finite_real_settings(field, value):
    with pytest.raises(ValueError):
        replace(PathConfig(), **{field: value})


@pytest.mark.parametrize("updates", [
    {"seed": -1}, {"n_cells": 4}, {"n_particles": 0},
    {"width": 1}, {"depth": 0}, {"train_paths": 0},
    {"validation_paths": 0}, {"test_paths": 0},
    {"adam_steps": -1}, {"lbfgs_steps": -1},
    {"adam_steps": 0, "lbfgs_steps": 0}, {"lbfgs_max_iter": 0},
    {"batch_size": True}, {"n_bootstrap": 0},
])
def test_integer_allocations_and_nonzero_training_required(updates):
    with pytest.raises(ValueError):
        replace(PathConfig(), **updates)


@pytest.mark.parametrize("updates", [
    {"tolerance": 1}, {"uncertainty_budget": .05},
    {"uncertainty_budget": .1}, {"rho_stop": 1 / LENGTH},
])
def test_accuracy_budget_and_density_boundary_require_valid_order(updates):
    with pytest.raises(ValueError):
        replace(PathConfig(), **updates)


def test_invalid_preset_and_undeclared_case_are_rejected():
    with pytest.raises(ValueError):
        path_preset("statistics")
    with pytest.raises(ValueError, match="Undeclared"):
        replace(PathConfig(), cases=("heat",)).problem("dk_uniform")
    with pytest.raises(ValueError, match="Unsupported"):
        PathConfig().context("moment")


def test_fit_steps_must_share_a_nested_bank_grid_not_only_the_reference_grid():
    with pytest.raises(ValueError, match="nest"):
        replace(PathConfig(), horizon=12/1024, query_times=(12/1024,),
                step_sizes=(3/1024, 4/1024), reference_step=1/1024)
