"""Boundary proposals preserve the natural objective and independent families."""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.payoff_collocation_sampling import (  # noqa: E402
    BoundaryStrataConfig,
    BoundaryStressConfig,
    CollocationSampler,
    boundary_time_strata,
    make_boundary_stress_pool,
)


def config():
    return BoundaryStrataConfig(1., .05, .1, distance_edges=(.5, 2.))


def pool():
    minimum = np.array([.055, .07, .2, .21, .22, .23, .24])
    states = np.column_stack((minimum, np.repeat(((1-minimum)/3)[:, None], 3, axis=1)))
    return states, np.full(7, .25), np.array(["family-a", "family-a", "family-b",
                                            "family-c", "family-d", "family-e", "family-e"])


def sampler(mode="stratified", seed=37):
    return CollocationSampler(*pool(), config(), mode=mode, seed=seed)


def resign(state):
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return state


def test_dimensionless_boundary_and_time_strata_handle_absorption_and_zero_time():
    states, _, _ = pool()
    boundary = np.array([.05, .3, .3, .35])
    inputs = np.vstack((boundary, boundary, states[[0, 0, 2, 2, 2]]))
    times = np.array([0., .25, 0., .25, .25, .75, np.nextafter(0., 1.)])
    with np.errstate(all="raise"):
        labels = boundary_time_strata(inputs, times, config())
    assert labels.tolist() == ["absorbed/zero-time", "absorbed/positive-time",
        "interior/zero-time", "active/time-1/distance-0", "active/time-1/distance-2",
        "active/time-3/distance-1", "active/time-0/distance-2"]
    # Identical d/sqrt(t) has the same distance bin even at different times.
    matching = np.array([[.06, .3, .3, .34], [.07, .3, .3, .33]])
    labels = boundary_time_strata(matching, [.25, 1.], config())
    assert labels[0].split("/")[-1] == labels[1].split("/")[-1]


def test_row_probabilities_only_allocate_to_occupied_strata_with_full_support():
    stratified = sampler()
    expected = np.array([.25, .25, .1, .1, .1, .1, .1])
    np.testing.assert_allclose(stratified.probabilities, expected, rtol=0, atol=0)
    assert len(stratified.identity["occupied_strata"]) == 2
    assert np.all(stratified.probabilities > 0)
    np.testing.assert_allclose(stratified.probabilities.sum(), 1., atol=2e-16)
    np.testing.assert_allclose(stratified.importance_weights, (1/7)/expected)
    uniform = sampler("uniform")
    np.testing.assert_array_equal(uniform.probabilities, np.full(7, 1/7))
    np.testing.assert_array_equal(uniform.importance_weights, np.ones(7))
    # Public inspection arrays cannot mutate the actual proposal.
    exposed = stratified.probabilities
    exposed[:] = 0
    np.testing.assert_array_equal(stratified.probabilities, expected)


@pytest.mark.parametrize("robust", [False, True])
def test_expected_importance_weighted_loss_and_gradient_equal_natural_pool(robust):
    proposal = sampler()
    states, _, _ = pool()
    parameter = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    x = torch.tensor(states[:, 0], dtype=torch.float64)
    target = torch.linspace(-.2, .3, 7, dtype=torch.float64)
    residual = parameter*x-target
    losses = (2*residual.square()/(torch.sqrt(1+(residual/.3).square())+1)
              if robust else residual.square())
    natural = losses.mean()
    q = torch.tensor(proposal.probabilities)
    importance = torch.tensor(proposal.importance_weights)
    expected = (q*importance*losses).sum()
    balanced_without_correction = (q*losses).sum()
    natural_gradient = torch.autograd.grad(natural, parameter, retain_graph=True)[0]
    corrected_gradient = torch.autograd.grad(expected, parameter)[0]
    torch.testing.assert_close(expected, natural, atol=2e-17, rtol=2e-15)
    torch.testing.assert_close(corrected_gradient, natural_gradient, atol=2e-17, rtol=2e-15)
    assert not torch.isclose(balanced_without_correction, natural, atol=1e-6, rtol=1e-6)


def test_all_absorbed_and_zero_time_rows_keep_support_and_natural_weights():
    states = np.tile([.05, .3, .3, .35], (4, 1))
    times = np.array([0., 0., 0., .5])
    proposal = CollocationSampler(states, times, ["one", "one", "two", "two"], config(),
                                 mode="stratified")
    np.testing.assert_allclose(proposal.probabilities, [1/6, 1/6, 1/6, 1/2])
    np.testing.assert_allclose(proposal.probabilities*proposal.importance_weights, .25)


def test_private_rng_exact_resume_preserves_row_order_weights_and_family_groups():
    before = np.random.get_state()
    original = sampler()
    original.draw(19)
    state = json.loads(json.dumps(original.state_dict()))
    expected = original.draw(73)
    restored = sampler()
    restored.load_state_dict(state)
    actual = restored.draw(73)
    for field in ("indices", "probabilities", "importance_weights", "family_ids", "labels"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert restored.state_dict() == original.state_dict()
    assert restored.state_dict()["draws"] == 92
    _, _, families = pool()
    np.testing.assert_array_equal(actual.family_ids, families[actual.indices])
    assert set(actual.family_ids) <= set(families)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize("change", ["pool", "horizons", "families", "thresholds", "mode", "seed"])
def test_resume_rejects_changed_ordered_pool_labels_configuration_and_proposal(change):
    states, times, families = pool()
    cfg, mode, seed = config(), "stratified", 37
    if change == "pool":
        states = states[::-1]
    elif change == "horizons":
        times = times+.01
    elif change == "families":
        families[0] = "fresh-parent"
    elif change == "thresholds":
        cfg = replace(cfg, distance_edges=(.4, 2.))
    elif change == "mode":
        mode = "uniform"
    else:
        seed += 1
    other = CollocationSampler(states, times, families, cfg, mode=mode, seed=seed)
    with pytest.raises(ValueError, match="identity changed"):
        other.load_state_dict(sampler().state_dict())


def test_resume_checksum_invalid_rng_and_failed_load_are_transactional():
    original = sampler()
    original.draw(9)
    state = original.state_dict()
    expected = original.draw(31)
    restored = sampler()
    restored.load_state_dict(state)
    damaged = json.loads(json.dumps(state))
    damaged["rng_state"]["state"]["state"] += 1
    with pytest.raises(ValueError, match="checksum changed"):
        restored.load_state_dict(damaged)
    damaged = json.loads(json.dumps(state))
    damaged["rng_state"]["bit_generator"] = "MT19937"
    with pytest.raises(ValueError, match="RNG state"):
        restored.load_state_dict(resign(damaged))
    damaged = json.loads(json.dumps(state))
    damaged["draws"] = -1
    with pytest.raises(ValueError, match="draw count"):
        restored.load_state_dict(resign(damaged))
    partial = state.copy()
    del partial["rng_state"]
    with pytest.raises(ValueError, match="Complete"):
        restored.load_state_dict(partial)
    np.testing.assert_array_equal(restored.draw(31).indices, expected.indices)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_sampler_rejects_empty_or_invalid_batch_allocations(size):
    with pytest.raises(ValueError, match="positive integer"):
        sampler().draw(size)


def test_sampler_rejects_empty_invalid_or_unidentified_training_rows():
    states, times, families = pool()
    with pytest.raises(ValueError, match="Nonempty"):
        CollocationSampler(states[:0], times[:0], families[:0], config())
    bad = states.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="unit-mass"):
        CollocationSampler(bad, times, families, config())
    with pytest.raises(ValueError, match="remaining horizon"):
        CollocationSampler(states, -times, families, config())
    with pytest.raises(ValueError, match="family ID"):
        CollocationSampler(states, times, [""]*7, config())
    with pytest.raises(ValueError, match="uniform or stratified"):
        CollocationSampler(states, times, families, config(), mode="balanced-unweighted")
    with pytest.raises(ValueError, match="private RNG seed"):
        CollocationSampler(states, times, families, config(), seed=True)
    with pytest.raises(ValueError, match="distance edges"):
        replace(config(), distance_edges=(1., .5))
    with pytest.raises(ValueError, match="time edges"):
        replace(config(), time_edges=(0., .5))


def stress_config():
    return BoundaryStressConfig(config(), target_distances=(0., .005, .1),
                                cell_sets=((0,), (0, 2)), horizons=(0., .25, 1.))


def test_stress_transform_preserves_mass_distances_parent_family_and_slack_ratios():
    parents = np.array([[.1, .2, .3, .4], [.25, .25, .25, .25], [.3, .2, .1, .4]])
    families = np.array(["parent-a", "parent-b", "parent-a"])
    cfg = stress_config()
    stress = make_boundary_stress_pool(parents, families, cfg)
    assert stress.states.shape == (3*2*3*3, 4)
    np.testing.assert_allclose(stress.states.sum(-1), 1., atol=5e-16, rtol=0)
    assert np.all(stress.states >= cfg.strata.mass_stop)
    np.testing.assert_array_equal(stress.family_ids, families[stress.parent_indices])
    assert stress.identity["independent_parent_families"] == 2
    assert stress.identity["transforms_are_independent_families"] is False
    assert stress.identity["physical_initial_law_sample"] is False
    assert len(set(stress.transform_ids)) == len(stress.states)
    assert not stress.states.flags.writeable
    for row, mask, distance, parent_index, label in zip(stress.states, stress.selected_cells,
            stress.target_distances, stress.parent_indices, stress.labels, strict=True):
        np.testing.assert_array_equal(row[mask], np.full(mask.sum(), cfg.strata.mass_stop+distance))
        if distance == 0:
            assert label.startswith("absorbed/")
        else:
            assert (row > cfg.strata.mass_stop).all()
            assert not label.startswith("absorbed/")
        original_slack = parents[parent_index, ~mask]-cfg.strata.mass_stop
        new_slack = row[~mask]-cfg.strata.mass_stop
        np.testing.assert_allclose(new_slack/new_slack.sum(), original_slack/original_slack.sum(),
                                   rtol=2e-15, atol=2e-16)
    replay = make_boundary_stress_pool(parents, families, cfg)
    assert replay.identity == stress.identity
    np.testing.assert_array_equal(replay.states, stress.states)
    changed = make_boundary_stress_pool(parents, families,
                                      replace(cfg, target_distances=(0., .01, .1)))
    assert changed.identity != stress.identity


def test_configuration_roundtrip_freezes_all_thresholds_and_stress_allocations():
    cfg = config()
    assert BoundaryStrataConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    stress = stress_config()
    assert BoundaryStressConfig.from_dict(json.loads(json.dumps(stress.to_dict()))) == stress
    incomplete = cfg.to_dict()
    del incomplete["distance_edges"]
    with pytest.raises(ValueError, match="Complete frozen"):
        BoundaryStrataConfig.from_dict(incomplete)
    incomplete = stress.to_dict()
    del incomplete["horizons"]
    with pytest.raises(ValueError, match="Complete frozen"):
        BoundaryStressConfig.from_dict(incomplete)


def test_stress_controls_reject_invalid_corner_targets_and_absorbed_parents():
    parent = np.array([[.1, .2, .3, .4]])
    cfg = stress_config()
    with pytest.raises(ValueError, match="remaining simplex slack"):
        make_boundary_stress_pool(parent, ["a"], replace(cfg, target_distances=(.5,)))
    with pytest.raises(ValueError, match="unselected cell"):
        make_boundary_stress_pool(parent, ["a"], replace(cfg, cell_sets=((0, 1, 2, 3),)))
    with pytest.raises(ValueError, match="cell IDs"):
        make_boundary_stress_pool(parent, ["a"], replace(cfg, cell_sets=((4,),)))
    with pytest.raises(ValueError, match="Strict interior parent"):
        make_boundary_stress_pool([[.05, .2, .3, .45]], ["a"], cfg)
    with pytest.raises(ValueError, match="Distinct selected"):
        replace(cfg, cell_sets=((0, 1), (1, 0)))
    with pytest.raises(ValueError, match="distinct|Distinct"):
        replace(cfg, target_distances=(-.01,))
