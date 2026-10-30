"""Forward traces retain full physical noise and split independent initial anchors."""

import time
from dataclasses import replace

import h5py
import numpy as np
import pytest

from nn_dk.bsde_config import bsde_preset
from nn_dk.bsde_data import (
    generate_bsde_pool,
    load_bsde_pool,
    ou_exact_moments,
    ou_rates,
    simulate_forward_trace,
)
from nn_dk.native import simulate_native


def small_config():
    return replace(bsde_preset("bsde-smoke"), train_anchors=4, validation_anchors=2)


def test_native_trace_preserves_default_path_and_uncensored_crossing():
    config = small_config()
    context, spec = config.context("uniform"), config.spec
    initial = np.full((2, spec.n_cells), 1 / spec.n_cells)
    initial[1, 0] = spec.mass_stop / 2
    initial[1, 1:] = (1 - initial[1, 0]) / (spec.n_cells - 1)
    increments = np.zeros((2, 3, spec.n_cells))
    increments[:, 0, 0] = 10.
    ids = ("crossing", "initially-stopped")
    trace = simulate_forward_trace("dk", config, initial, 7, horizon=3 * config.dt,
                                   family_ids=ids, brownian_increments=increments)
    plain = simulate_native(context, spec, trace.times, config.dt, 7, 2,
                            initial_masses=initial, brownian_increments=increments,
                            family_ids=ids)
    assert np.array_equal(trace.states, plain.states)
    assert np.array_equal(trace.stopped_at, plain.stopped_at)
    assert np.array_equal(trace.increments, increments)
    assert trace.active[0, 0] and not np.any(trace.active[0, 1:])
    assert not np.any(trace.active[1])
    assert trace.proposals[0, 0].min() < 0
    assert trace.states[0, 1].min() == pytest.approx(spec.mass_stop)
    assert 0 < trace.stopped_at[0] < config.dt
    assert trace.stopped_at[1] == 0
    assert np.all(trace.states[0, 1:] == trace.states[0, 1])


def test_capture_does_not_change_generated_native_noise_or_paths():
    config = small_config()
    context, spec = config.context("uniform"), config.spec
    times = np.arange(5) * config.dt
    plain = simulate_native(context, spec, times, config.dt, 22, n_paths=4)
    capture = {}
    traced = simulate_native(context, spec, times, config.dt, 22, n_paths=4,
                             trace_capture=capture)
    assert np.array_equal(plain.states, traced.states)
    assert np.array_equal(capture["pre_states"], traced.states[:, :-1])
    assert capture["increments"].shape == (4, 4, spec.n_cells)
    assert capture["active"].dtype == bool


def test_ou_exact_euler_moments_agree_with_independent_recursion():
    config = small_config()
    initial = np.array([[1., -.5, .3, 2.], [-1., 0., 0., 0.]])
    horizons = np.array([0., config.dt, 5 * config.dt])
    mean, covariance = ou_exact_moments(initial, horizons, config, dt=config.dt)
    rates = ou_rates(config)
    for j, horizon in enumerate(horizons):
        expected_mean, expected_variance = initial.copy(), np.zeros(4)
        for _ in range(round(horizon / config.dt)):
            expected_mean *= 1 - rates * config.dt
            expected_variance = ((1 - rates * config.dt)**2 * expected_variance
                                 + 2 * rates * config.dt)
        np.testing.assert_allclose(mean[:, j], expected_mean, atol=1e-15)
        np.testing.assert_allclose(covariance[:, j], np.broadcast_to(
            np.diag(expected_variance), (2, 4, 4)), atol=1e-15)
    long_mean, long_covariance = ou_exact_moments(initial[0], [100.], config)
    assert long_mean.shape == (1, 4)
    np.testing.assert_allclose(long_covariance[0], np.eye(4), atol=1e-15)
    with pytest.raises(ValueError, match="grid"):
        ou_exact_moments(initial, [1.2 * config.dt], config, dt=config.dt)


def test_ou_forward_reuses_full_supplied_increments():
    config = small_config()
    initial = np.array([[1., 2., 3., 4.]])
    dw = np.full((1, 2, 4), .1)
    trace = simulate_forward_trace("ou", config, initial, 1, horizon=2 * config.dt,
                                   brownian_increments=dw)
    rates = ou_rates(config)
    np.testing.assert_allclose(trace.states[:, 1], initial * (1 - config.dt * rates)
                               + np.sqrt(2 * rates) * dw[:, 0])
    assert trace.active.all() and np.isinf(trace.stopped_at).all()
    assert np.array_equal(trace.proposals, trace.states[:, 1:])


def test_continuation_noise_streams_do_not_depend_on_path_batching():
    config = small_config()
    initial = np.array([[1., 2., 3., 4.], [-1., -2., -3., -4.]])
    batched = simulate_forward_trace("ou", config, initial, 43,
                                    family_ids=("child-a", "child-b"))
    single = simulate_forward_trace("ou", config, initial[1], 43,
                                   family_ids=("child-b",))
    assert np.array_equal(batched.increments[1], single.increments[0])
    assert np.array_equal(batched.states[1], single.states[0])


@pytest.mark.parametrize("problem", ["ou", "dk"])
def test_anchor_pools_are_reproducible_disjoint_and_resumable(tmp_path, monkeypatch, problem):
    config = small_config()
    monkeypatch.setattr("nn_dk.bsde_data.source_hash", lambda: "frozen-test-source")
    expired = generate_bsde_pool(tmp_path / "resumed", config, problem,
                                 deadline=time.monotonic() - 1)
    assert not expired["complete"] and expired["generated_this_call"] == 0
    with pytest.raises(ValueError, match="incomplete|absent"):
        load_bsde_pool(tmp_path / "resumed", config, problem, "train")
    first = generate_bsde_pool(tmp_path / "resumed", config, problem)
    assert first["complete"] and first["generated_this_call"] == 6
    again = generate_bsde_pool(tmp_path / "resumed", config, problem)
    assert again["complete"] and again["generated_this_call"] == 0
    generate_bsde_pool(tmp_path / "fresh", config, problem)
    train = load_bsde_pool(tmp_path / "resumed", config, problem, "train")
    validation = load_bsde_pool(tmp_path / "resumed", config, problem, "validation")
    fresh = load_bsde_pool(tmp_path / "fresh", config, problem, "train")
    assert train.fingerprint == fresh.fingerprint
    assert not set(train.anchor_ids) & set(validation.anchor_ids)
    assert not set(train.family_ids) & set(validation.family_ids)
    assert len(set(train.anchor_ids)) == config.train_anchors
    assert np.array_equal(train.initial[0], train.initial[1])
    assert not np.array_equal(train.increments[0], train.increments[1])
    with h5py.File(tmp_path / "resumed" / f"{problem}_pool.h5", "r") as store:
        group = store["train/anchor_000000"]
        assert "proposals" not in group["learner"]
        assert group["oracle"].attrs["not_network_inputs"]
        assert group["learner/states"].chunks is not None
    changed_config = replace(config, seed=config.seed + 1)
    with pytest.raises(ValueError, match="changed"):
        load_bsde_pool(tmp_path / "resumed", changed_config, problem, "train")
    with pytest.raises(ValueError, match="changed"):
        generate_bsde_pool(tmp_path / "resumed", changed_config, problem)
    monkeypatch.setattr("nn_dk.bsde_data.source_hash", lambda: "changed-source")
    with pytest.raises(ValueError, match="changed"):
        load_bsde_pool(tmp_path / "resumed", config, problem, "train")
