from __future__ import annotations

import json

import numpy as np
import pytest

from nn_dk.native_types import NativeBatch, NativeDKSpec
from nn_dk.path_evaluation import (
    empirical_proposal_quadratic_variation,
    empirical_quadratic_variation,
    evaluate_native_paths,
    evaluate_paths,
    path_normalization,
    verify_causality,
    verify_same_noise_replay,
)
from nn_dk.types import Context


def paths(n=24, g=8):
    spec, context = NativeDKSpec(n_cells=g), Context(n_particles=2048)
    times = np.array([0.0, 0.25, 0.5, 1.0])
    states = np.full((n, len(times), g), 1 / g)
    rng = np.random.default_rng(24)
    for step in range(1, len(times)):
        noise = rng.normal(0, 0.001, (n, g))
        noise -= noise.mean(axis=1, keepdims=True)
        states[:, step] += noise
    return states, times, context, spec


def compare(prediction=None, reference=None, **kwargs):
    states, times, context, spec = paths()
    prediction = states if prediction is None else prediction
    reference = states if reference is None else reference
    defaults = {
        "coupling_verified": True,
        "expected_paths": len(prediction),
        "observation_times": times[1:],
        "n_bootstrap": 12,
    }
    return evaluate_paths(prediction, reference, times, context, spec, **(defaults | kwargs))


def test_fixed_uniform_histogram_fluctuation_energy_has_no_extra_grid_factor():
    spec = NativeDKSpec(n_cells=32)
    normalization = path_normalization(spec, 2048)
    assert normalization["uniform_density_fluctuation_energy"] == pytest.approx(
        (1 - 1 / 32) / (2048 * spec.h))
    assert normalization["n_scaled_cell_standard_deviation"]**2 == pytest.approx(
        31 / 32**2)
    states, times, context, spec = paths()
    prediction = states.copy()
    prediction[:, 1:, 0] += 0.001
    prediction[:, 1:, 1] -= 0.001
    result = evaluate_paths(prediction, states, times, context, spec,
                            coupling_verified=True, n_bootstrap=8)
    expected = np.sqrt(2048 * 2 * 0.001**2 / (1 - 1 / 8))
    assert result["strong"]["pathwise_supremum_rms"] == pytest.approx(expected)
    assert result["strong"]["rms_final_time"] == pytest.approx(expected)
    np.testing.assert_allclose(result["strong"]["rms_by_time"], [0, expected, expected, expected])


def test_high_grid_modes_cannot_pass_by_leaving_four_observables_unchanged():
    states, times, context, spec = paths(g=32)
    prediction = states.copy()
    high_mode = np.cos(3 * 2 * np.pi * (np.arange(32) + 0.5) / 32)
    prediction[:, 1:] += 0.001 * high_mode
    result = evaluate_paths(prediction, states, times, context, spec,
                            coupling_verified=True, observation_times=times[1:], n_bootstrap=8)
    assert result["strong"]["pathwise_supremum_rms"] > 0.15
    assert result["weak_full_state"]["maximum_mean_normalized_error"] > 0.2
    assert result["weak_observables"]["maximum_mean_normalized_error"] < 1e-12
    assert result["weak_observables"]["maximum_joint_covariance_normalized_error"] < 1e-12
    assert not result["accuracy_gate"]["strong_empirical_interval_within_tolerance"]


def test_joint_covariance_includes_two_time_errors_with_identical_one_time_laws():
    states, times, context, spec = paths(n=4)
    values = np.array([-2, -1, 1, 2]) * 0.001
    states[:] = 1 / 8
    for step in range(1, 4):
        states[:, step, 0] += values
        states[:, step, 1] -= values
    prediction = states.copy()
    prediction[:, -1] = prediction[::-1, -1]
    result = evaluate_paths(prediction, states, times, context, spec,
                            coupling_verified=True, observation_times=times[1:], n_bootstrap=0)
    weak = result["weak_full_state"]
    np.testing.assert_allclose(weak["prediction_mean"], weak["reference_mean"])
    np.testing.assert_allclose(weak["prediction_n_scaled_same_time_covariance"],
                               weak["reference_n_scaled_same_time_covariance"])
    assert weak["two_time_covariances_included"]
    assert weak["maximum_joint_covariance_normalized_error"] > 0.05
    np.testing.assert_allclose(weak["reference_n_scaled_joint_covariance"],
                               2048 * np.cov(states[:, 1:].reshape(4, -1), rowvar=False))


def test_failed_nonfinite_paths_prevent_survivor_conditioned_statistics():
    states, _, _, _ = paths()
    failed = np.zeros(len(states), dtype=bool)
    failed[4] = True
    result = compare(failures=failed)
    assert result["status"] == "invalid"
    assert result["strong"] is result["weak_full_state"] is result["weak_observables"] is None
    assert result["diagnostics"]["prediction"]["n_allocated_paths"] == len(states)
    assert "survivor" in result["accuracy_gate"]["reason"]
    bad = states.copy()
    bad[4, -1, 0] = np.nan
    result = compare(prediction=bad)
    assert result["status"] == "invalid"
    assert result["diagnostics"]["prediction"]["n_nonfinite_paths"] == 1
    with pytest.raises(ValueError, match="full allocated"):
        compare(prediction=states[:-1], reference=states[:-1], expected_paths=len(states))


def test_negative_nonconservative_and_unfrozen_states_are_recorded_as_failures():
    states, _, _, _ = paths()
    bad = states.copy()
    bad[0, 2, 0] = -0.001
    bad[0, 2, 1] += states[0, 2, 0] + 0.001
    result = compare(prediction=bad)
    assert result["diagnostics"]["prediction"]["n_negative_mass_paths"] == 1
    bad = states.copy()
    bad[0, 1, 0] += 0.1
    assert compare(prediction=bad)["diagnostics"]["prediction"]["n_nonconservative_paths"] == 1
    stops = np.full(len(states), np.inf)
    stops[0] = 0.1
    result = compare(stopped_at=stops, reference_stopped_at=np.full(len(states), np.inf))
    assert result["diagnostics"]["prediction"]["n_not_frozen_after_stop"] == 1


def test_initially_stopped_and_absorbed_paths_remain_in_all_moments():
    states, times, context, spec = paths(n=4)
    states[:] = 1 / 8
    states[0, :, 0] = 0
    states[0, :, 1] = 2 / 8
    states[1, 1:, 0] = spec.mass_stop
    states[1, 1:, 1] += 1 / 8 - spec.mass_stop
    stops = np.array([0, 0.2, np.inf, np.inf])
    result = evaluate_paths(states, states, times, context, spec,
                            stopped_at=stops, reference_stopped_at=stops,
                            coupling_verified=True, observation_times=times[1:], n_bootstrap=8)
    assert result["status"] == "available"
    assert result["stopping"]["all_stopped_paths_included"]
    assert result["stopping"]["prediction_initially_stopped"] == 1
    assert result["stopping"]["prediction_cdf"] == [0.5, 0.5, 0.5]
    assert result["stopping"]["prediction_restricted_mean_stop_time"] == pytest.approx(0.55)
    np.testing.assert_allclose(result["weak_full_state"]["prediction_mean"], states[:, 1:].mean(0))
    assert result["strong"]["pathwise_supremum_rms"] == 0


def test_coupling_requires_explicit_physical_noise_verification_and_same_initial_state():
    result = compare(coupling_verified=False)
    assert result["status"] == "unavailable"
    assert result["strong"] is None
    assert result["accuracy_gate"]["status"] == "unresolved"
    states, _, _, _ = paths()
    changed = states.copy()
    changed[0, 0, :2] += [0.001, -0.001]
    with pytest.raises(ValueError, match="identical initial"):
        compare(prediction=changed)
    ids = tuple(str(i) for i in range(len(states)))
    with pytest.raises(ValueError, match="ordered"):
        compare(family_ids=ids, reference_family_ids=ids[::-1])
    with pytest.raises(ValueError, match="unique"):
        compare(family_ids=("same",) * len(states), reference_family_ids=ids)
    assert compare(family_ids=ids, reference_family_ids=ids)["diagnostics"]["ordered_family_ids_verified"]


def test_reference_and_sampling_budgets_are_separate_from_empirical_agreement():
    result = compare()
    assert result["strong"]["pathwise_supremum_rms"] == 0
    assert result["accuracy_gate"]["strong_empirical_interval_within_tolerance"]
    assert result["accuracy_gate"]["weak_full_state_and_modes_empirical_intervals_within_tolerance"]
    assert result["accuracy_gate"]["status"] == "unresolved"
    assert not result["accuracy_gate"]["strong_reference_bias_resolved"]
    bounds = {"status": "resolved", "strong_normalized_bound": 0.001,
              "weak_normalized_bound": 0.002}
    certified = compare(reference_bias=bounds)
    assert certified["accuracy_gate"]["status"] == "resolved_within_tolerance"
    assert certified["accuracy_gate"]["strong_upper_plus_reference_bound"] == 0.001
    assert certified["accuracy_gate"]["weak_upper_plus_reference_bound"] == 0.002
    assert compare(reference_bias=bounds, expected_paths=None)["accuracy_gate"]["status"] == "unresolved"
    assert compare(reference_bias=bounds, n_bootstrap=0)["accuracy_gate"]["status"] == "unresolved"
    bad = bounds | {"strong_normalized_bound": 0.02}
    assert not compare(reference_bias=bad)["accuracy_gate"]["strong_reference_bias_resolved"]
    bad = bounds | {"weak_normalized_bound": True}
    assert not compare(reference_bias=bad)["accuracy_gate"]["weak_reference_bias_resolved"]


def test_coherent_low_mode_error_cannot_hide_behind_small_individual_cell_errors():
    states, times, context, spec = paths(g=32)
    changed = states.copy()
    changed[:, 1:] += 0.0001 * np.cos(2 * np.pi * (np.arange(32) + 0.5) / 32)
    bounds = {"status": "resolved", "strong_normalized_bound": 0,
              "weak_normalized_bound": 0}
    result = evaluate_paths(changed, states, times, context, spec, coupling_verified=True,
                            expected_paths=len(states), n_bootstrap=8, reference_bias=bounds)
    assert result["strong"]["pathwise_supremum_rms"] < 0.05
    assert result["weak_full_state"]["maximum_mean_normalized_error"] < 0.05
    assert result["weak_observables"]["maximum_mean_normalized_error"] > 0.1
    assert result["accuracy_gate"]["weak_status"] == "unresolved"


def test_deterministic_whole_family_bootstrap_and_strict_json():
    states, _, _, _ = paths()
    changed = states.copy()
    changed[:, 1:, 0] += np.linspace(0, 0.0002, len(states))[:, None]
    changed[:, 1:, 1] -= np.linspace(0, 0.0002, len(states))[:, None]
    first = compare(prediction=changed, seed=3)
    assert first == compare(prediction=changed, seed=3)
    assert first["strong"]["uncertainty"]["ci_halfwidth"] > 0
    assert not first["weak_full_state"]["uncertainty"]["simultaneous_band"]
    json.dumps(first, allow_nan=False)
    assert "continuous stopping" in first["reference_scope"]
    one, times, context, spec = paths(n=1)
    result = evaluate_paths(one, one, times, context, spec, coupling_verified=True)
    assert result["strong"]["pathwise_supremum_rms"] == 0
    assert result["weak_full_state"] is None
    assert result["accuracy_gate"]["status"] == "unresolved"


@pytest.mark.parametrize("kwargs", [
    {"n_bootstrap": -1}, {"n_bootstrap": 2.5}, {"tolerance": float("nan")},
    {"uncertainty_budget": 0}, {"coupling_verified": "yes"},
    {"failures": np.zeros(24)}, {"stopped_at": np.full(24, np.nan)},
    {"observation_times": (0.125,)}, {"observation_times": (0,)},
])
def test_invalid_contracts_are_rejected(kwargs):
    with pytest.raises(ValueError):
        compare(**kwargs)


def test_native_adapter_preserves_family_ids_and_target():
    states, times, context, spec = paths()
    batch = NativeBatch(times, states, spec, context, np.full(len(states), np.inf),
                        tuple(f"family:{i}" for i in range(len(states))))
    result = evaluate_native_paths(batch, batch, coupling_verified=True, n_bootstrap=0)
    assert result["strong"]["pathwise_supremum_rms"] == 0
    assert result["diagnostics"]["ordered_family_ids_verified"]
    assert result["target_id"] == spec.target_id


def test_qv_uses_all_active_proposals_and_native_N_scaling():
    p, k, g, n = 4096, 8, 3, 2048
    times = np.linspace(0, 0.5, k + 1)
    dt = times[1]
    rng = np.random.default_rng(64)
    factor = np.array([[1.0, 0, 0], [-1.0, 2.0, 0], [0, -2.0, 0]])
    covariance = factor @ factor.T
    increments = rng.normal(size=(p, k, g)) @ factor.T * np.sqrt(dt / n)
    active = np.ones((p, k), dtype=bool)
    active[p // 2:, 4:] = False
    increments *= active[:, :, None]
    states = np.concatenate((np.zeros((p, 1, g)), np.cumsum(increments, axis=1)), axis=1)
    result = empirical_quadratic_variation(
        states, times, n, active_at_start=active,
        predictable_covariance=np.broadcast_to(covariance, (p, k, g, g)))
    assert result["n_active_proposals"] == int(active.sum())
    assert result["mean_active_time_exposure"] == pytest.approx(0.375)
    assert result["coefficient_comparison"]["status"] == "descriptive-uncensored-proposals"
    assert result["coefficient_comparison"]["relative_frobenius_difference"] < 0.025
    np.testing.assert_allclose(result["n_scaled_predictable_integrated_covariance"],
                               0.375 * covariance)
    assert abs(result["n_scaled_realized_trace"] / (0.375 * np.trace(covariance)) - 1) < 0.025
    json.dumps(result, allow_nan=False)


def test_qv_does_not_treat_fractional_stopped_crossings_as_uncensored_noise():
    states = np.zeros((2, 3, 2))
    states[:, 1] = [0.1, -0.1]
    states[:, 2] = states[:, 1]
    active = np.array([[True, False], [True, False]])
    crossed = np.array([[True, False], [False, False]])
    result = empirical_quadratic_variation(
        states, [0, 0.25, 0.5], 32, active_at_start=active, crossed=crossed,
        predictable_covariance=np.broadcast_to(np.eye(2), (2, 2, 2, 2)))
    assert result["n_crossing_increments"] == 1
    assert result["all_pre_stop_increments_retained"]
    assert result["coefficient_comparison"]["status"] == "unresolved-stopped-crossing"
    with pytest.raises(ValueError, match="restart"):
        empirical_quadratic_variation(states, [0, 0.25, 0.5], 32,
                                      active_at_start=np.array([[False, True], [True, True]]))


def test_qv_left_endpoint_drift_is_removed_without_filtering_crossing_proposals():
    states = np.zeros((2, 3, 2))
    states[:, 1] = [0.25, -0.25]
    states[:, 2] = [0.5, -0.5]
    drift = np.broadcast_to(np.array([1, -1]), (2, 2, 2))
    result = empirical_quadratic_variation(states, [0, 0.25, 0.5], 32,
                                          predictable_drift=drift)
    assert result["drift_removed"]
    assert result["n_scaled_realized_trace"] == 0


def test_proposal_qv_retains_exit_proposals_and_avoids_dense_covariance_storage():
    origins = np.full((2, 2, 3), 1 / 3)
    proposals = origins.copy()
    proposals[:, 0, :2] += [-0.5, 0.5]
    active = np.array([[True, False], [True, False]])
    crossing = active.copy()
    amplitudes = np.full_like(origins, 2)
    result = empirical_proposal_quadratic_variation(
        origins, proposals, [0, 0.25, 0.5], 32,
        active_at_start=active, crossing_proposals=crossing,
        predictable_face_amplitudes=amplitudes)
    assert result["n_uncensored_exit_proposals"] == 2
    assert result["crossing_proposals_retained"]
    assert result["coefficient_comparison"]["status"] == "descriptive-uncensored-proposals"
    assert result["n_scaled_realized_trace"] == pytest.approx(16)
    incidence = np.roll(np.eye(3), 1, axis=0) - np.eye(3)
    np.testing.assert_allclose(result["n_scaled_predictable_integrated_covariance"],
                               incidence @ incidence.T)


def causal_operator(initial, increments):
    result = np.repeat(initial[:, None, :], increments.shape[1] + 1, axis=1)
    result[:, 1:, 0] += np.cumsum(increments[:, :, 0], axis=1)
    result[:, 1:, 1] -= np.cumsum(increments[:, :, 0], axis=1)
    return result


def test_future_noise_perturbation_checks_every_past_state_and_same_noise_replay():
    initial = np.full((3, 8), 1 / 8)
    increments = np.random.default_rng(18).normal(0, 0.001, (3, 8, 8))
    for cut in (0, 3, 7):
        result = verify_causality(causal_operator, initial, increments, cut)
        assert result["passed"]
        assert result["maximum_past_difference"] == 0
        assert result["maximum_future_difference"] > 0
        assert result["future_noise_was_perturbed"]
    assert verify_same_noise_replay(causal_operator, initial, increments)["passed"]


def test_future_leakage_and_extra_inference_randomness_are_detected():
    initial = np.full((3, 8), 1 / 8)
    increments = np.zeros((3, 8, 8))

    def leaking(initial, increments):
        result = causal_operator(initial, increments)
        result[:, 1:, 0] += increments.sum(axis=(1, 2))[:, None]
        return result

    assert not verify_causality(leaking, initial, increments, 3)["passed"]
    rng = np.random.default_rng(1)

    def extra_noise(initial, increments):
        result = causal_operator(initial, increments)
        result[:, 1:, 0] += rng.normal(size=result[:, 1:, 0].shape)
        return {"states": result}

    assert not verify_same_noise_replay(extra_noise, initial, increments)["passed"]
