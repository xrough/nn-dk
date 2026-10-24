"""Reject traces that would silently bias the martingale training objective."""

from dataclasses import replace

import numpy as np
import pytest

from nn_dk.bsde_config import BSDEConfig, bsde_preset
from nn_dk.bsde_model import BSDEMomentModel
from nn_dk.bsde_types import BSDETrace, MomentPrediction
from nn_dk.types import Context, QueryBank


def trace_arguments(problem="dk"):
    dimension = 8 if problem == "dk" else 4
    states = (np.full((2, 3, dimension), 1 / dimension) if problem == "dk"
              else np.zeros((2, 3, dimension)))
    return {"times": np.array([0., .1, .2]), "states": states,
            "increments": np.zeros((2, 2, dimension)), "active": np.ones((2, 2), bool),
            "anchor_ids": ("initial-a", "initial-b"), "family_ids": ("path-a", "path-b"),
            "problem": problem, "stopped_at": np.full(2, np.inf)}


def test_native_trace_rejects_incompatible_brownian_driver_dimension():
    arguments = trace_arguments()
    arguments["increments"] = np.zeros((2, 2, 1))
    with pytest.raises(ValueError):
        BSDETrace(**arguments)


@pytest.mark.parametrize("stop,active", [
    (np.inf, [True, False]),
    (.05, [True, True]),
    (0., [True, False]),
    (.3, [True, True]),
])
def test_trace_rejects_inconsistent_stopping_masks_and_times(stop, active):
    arguments = trace_arguments()
    arguments["stopped_at"][0] = stop
    arguments["active"][0] = active
    with pytest.raises(ValueError):
        BSDETrace(**arguments)


def test_trace_rejects_unfrozen_absorbed_states():
    arguments = trace_arguments()
    arguments["stopped_at"][0] = .05
    arguments["active"][0] = [True, False]
    arguments["states"][0, 2, :2] += [.01, -.01]
    with pytest.raises(ValueError):
        BSDETrace(**arguments)


def test_anchor_continuations_share_initial_state_but_have_unique_families():
    arguments = trace_arguments()
    arguments["anchor_ids"] = ("same-initial", "same-initial")
    valid = BSDETrace(**arguments)
    assert len(set(valid.anchor_ids)) == 1
    arguments["states"][1, 0, :2] += [.01, -.01]
    with pytest.raises(ValueError):
        BSDETrace(**arguments)
    arguments = trace_arguments()
    arguments["family_ids"] = ("duplicate", "duplicate")
    with pytest.raises(ValueError):
        BSDETrace(**arguments)


def test_ou_contract_rejects_stopping():
    arguments = trace_arguments("ou")
    arguments["stopped_at"][0] = .05
    arguments["active"][0] = [True, False]
    with pytest.raises(ValueError):
        BSDETrace(**arguments)


def test_time_zero_and_initially_stopped_traces_are_valid():
    arguments = trace_arguments()
    arguments["stopped_at"][:] = 0.
    arguments["active"][:] = False
    trace = BSDETrace(**arguments)
    assert trace.subset([1]).family_ids == ("path-b",)


@pytest.mark.parametrize("horizons", [[], [.2, .1], [.1, .1]])
def test_prediction_contract_rejects_invalid_horizon_banks(horizons):
    with pytest.raises(ValueError):
        MomentPrediction(np.zeros((len(horizons), 4)), np.zeros((len(horizons), 4, 4)),
                         np.asarray(horizons), "dk", "test-target")


@pytest.mark.parametrize("training", [
    {"learning_rate": 0.}, {"learning_rate": np.inf},
    {"weight_decay": -.1}, {"weight_decay": np.nan},
    {"gradient_clip": 0.}, {"gradient_clip": np.inf},
])
def test_config_rejects_invalid_optimization_settings(training):
    with pytest.raises(ValueError):
        BSDEConfig(training=training)


def test_config_partial_training_override_has_complete_defaults():
    config = BSDEConfig(training={"epochs": 2})
    assert config.training["epochs"] == 2
    assert config.training["learning_rate"] == 1e-3
    assert config.training["batch_size"] == 32


def test_model_rejects_untrained_horizons_and_incompatible_physics():
    config = bsde_preset("bsde-smoke")
    model = BSDEMomentModel(config, "dk")
    initial = np.full(config.n_cells, 1 / config.n_cells)
    with pytest.raises(ValueError, match="horizons"):
        model.predict_conditional(initial, [.3 * config.horizon])
    with pytest.raises(ValueError, match="noninteracting"):
        model.predict_ensemble(Context(kappa=.1, n_particles=config.n_particles),
                               config.queries, 2, 0)
    with pytest.raises(ValueError, match="physics"):
        model.predict_ensemble(Context(n_particles=config.n_particles, diffusion=2.),
                               config.queries, 2, 0)
    with pytest.raises(ValueError, match="modes"):
        model.predict_ensemble(config.context("uniform"), QueryBank(config.query_times, (1,)), 2, 0)


def test_target_identifiers_distinguish_forward_step_and_problem():
    config = bsde_preset("bsde-smoke")
    refined = replace(config, dt=config.dt / 2)
    assert config.target_id("dk") != config.target_id("ou")
    assert config.target_id("dk") != refined.target_id("dk")
    with pytest.raises(ValueError):
        config.target_id("particle")
