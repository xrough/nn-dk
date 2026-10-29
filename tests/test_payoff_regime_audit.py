"""Independent analytic/stopping controls and pretest-only identifier access."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import audit_native_stopping as stopping  # noqa: E402
from analysis import audit_payoff_regimes as audit  # noqa: E402
from analysis.compiled_native import compiler_available  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.types import Context, seeded_rng  # noqa: E402


def test_sparse_lift_matches_independent_uniform_analytic_terminal_and_integral_covariance():
    spec = NativeDKSpec(16)
    context = Context(n_particles=2048, diffusion=1.)
    horizon = 1.
    bank = PayoffBank.mixed_terminal_integral(spec, horizon)
    state = np.full(16, 1/16)
    heat = context.diffusion/spec.h**2*(np.roll(np.eye(16), 1, 0)
        + np.roll(np.eye(16), -1, 0)-2*np.eye(16))
    steady = np.diag(state)-np.outer(state, state)
    for time in (.05, .25, 1.):
        result = audit.exact_free_heat_moments(state, spec, context, bank, time)
        transition = expm(time*heat)
        expected_terminal = bank.terminal_weights @ (steady-transition @ steady @ transition.T) @ bank.terminal_weights.T
        np.testing.assert_allclose(result["scaled_covariance"][:2, :2], expected_terminal[:2, :2],
                                   atol=3e-14, rtol=1e-12)
        # For a native Fourier eigenfunction lambda, the conditional integrated
        # OU variance is v*(2t/lambda-(3-4e^-lambda*t+e^-2lambda*t)/lambda²).
        lam = 4*np.sin(spec.h/2)**2/spec.h**2
        q = bank.running_weights[2:]
        amplitude = q @ steady @ q.T
        coefficient = 2*time/lam-(3-4*np.exp(-lam*time)+np.exp(-2*lam*time))/lam**2
        np.testing.assert_allclose(result["scaled_covariance"][2:, 2:], amplitude*coefficient,
                                   atol=3e-14, rtol=1e-10)
        np.testing.assert_allclose(result["mean"], 0., atol=3e-15)


def test_long_horizon_quadrature_and_instability_are_independently_audited():
    result = audit.analytic_controls((1.,), n_cells=16, quadrature_nodes=16)
    assert result["no_held_out_DK_outcomes_generated_or_read"]
    assert len(result["records"]) == 9
    for row in result["records"]:
        assert row["quadrature"]["16"]["maximum_normalized_covariance_error"] < 1e-9
        assert row["quadrature"]["16"]["maximum_scaled_mean_error"] < 2e-14
        assert row["quadrature"]["16"]["maximum_normalized_time_derivative_error"] < 1e-8
    rates = [row["observed_mode1_growth_rate"] for row in result["uniform_linearization"]]
    assert rates[0] < 0 < rates[1] < rates[2]
    assert all(row["maximum_mode1_eigenvector_error"] < 1e-12 for row in result["uniform_linearization"])


def test_identifier_inventory_never_opens_outcomes_and_rejects_family_overlap(tmp_path):
    historical = tmp_path/"runs/old/data.h5"
    historical.parent.mkdir(parents=True)
    with h5py.File(historical, "w") as store:
        store.create_dataset("group/family_ids", data=np.array(["old-a", "old-b"], dtype="S16"))
        # A dangling external link would fail if the audit dereferenced arbitrary
        # outcome material. Only named identifier datasets may be read.
        store["unavailable_outcomes"] = h5py.ExternalLink("missing.h5", "/outcomes")
    current = tmp_path/"runs/current"
    current.mkdir()
    with h5py.File(current/"excluded.h5", "w") as store:
        store.create_dataset("family_ids", data=np.array(["new-a"], dtype="S16"))
    ids, inventory = audit.historical_identifier_inventory(tmp_path/"runs", current, {"new-a"})
    assert ids == {"old-a", "old-b"}
    assert len(inventory) == 1
    assert list(inventory[0]["datasets_read"]) == ["group/family_ids"]
    with pytest.raises(ValueError, match="overlap"):
        audit.historical_identifier_inventory(tmp_path/"runs", current, {"old-a"})


def test_entropy_digest_reproduces_the_rng_seedsequence_without_a_draw():
    seed, keys = 2026102802, ("regime", "forcing", 17)
    digest = bytes.fromhex(audit.entropy_digest(seed, *keys))
    expected = np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype="<u4").tolist()))
    actual = seeded_rng(seed, *keys)
    np.testing.assert_array_equal(actual.normal(size=16), expected.normal(size=16))
    assert audit.entropy_digest(seed, *keys) == hashlib.sha256(json.dumps(
        [seed, *keys], separators=(",", ":")).encode()).hexdigest()
    assert audit.entropy_digest(seed, *keys) != audit.entropy_digest(seed, "regime", "forcing", 18)


@pytest.mark.skipif(not compiler_available(), reason="Optional serial C++ control unavailable")
def test_compiled_native_first_crossing_and_killed_integrals_match_package_controls():
    result = stopping.stopped_native_controls(n_cells=32)
    assert result["all_paths_retained"]
    assert result["first_segment_crossing"] == result["already_absorbed"] == 1
    assert result["maximum_cpp_vs_native_state_error"] < 2e-14
    assert result["maximum_cpp_vs_native_stopping_time_error"] < 2e-14
    assert result["maximum_killed_integral_manual_error"] < 2e-15
    assert result["mass_error"] < 2e-14
    assert result["kind"] == stopping.KIND
    assert result["frozen_readout_verification"]["frozen_states_bitwise_equal"]
    assert result["frozen_readout_verification"]["initially_absorbed_running_payoffs_exactly_zero"]
    assert result["frozen_readout_verification"]["killed_running_payoffs_exactly_frozen"]
    assert result["no_held_out_DK_outcomes_generated_or_read"]
    assert result["provenance"]["historical_records_rewritten"] is False
    assert hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest() == stopping.HISTORICAL_HELPER_SHA256


def frozen_readout_fixture():
    states = np.full((3, 5, 8), 1/8)
    states[1, 0, :2] += [.01, -.01]
    values = np.zeros((3, 5, 4))
    values[2, :, :2] = [.014719351239493379, -.009835156062260268]
    values[1, 1:] = [.014719351239493379, -.009835156062260268, .125, -.25]
    return states, values, np.array([False, False, True, True])


def test_frozen_linear_readouts_accept_blas_projection_roundoff_only():
    states, values, integral_rows = frozen_readout_fixture()
    values[2, -1, 0] -= 7e-18
    values[1, -1, 0] -= 7e-18
    result = stopping.verify_frozen_readouts(states, values, integral_rows)
    assert 0 < result["maximum_frozen_terminal_projection_error"] < 8e-18
    assert result["projected_payoff_constancy_atol"] == 2e-15
    assert result["projected_payoff_constancy_rtol"] == 0


@pytest.mark.parametrize("row", [1, 2])
def test_frozen_linear_readouts_reject_nonroundoff_projection_errors(row):
    states, values, integral_rows = frozen_readout_fixture()
    values[row, -1, 0] += 1e-8
    with pytest.raises(AssertionError, match="terminal projections"):
        stopping.verify_frozen_readouts(states, values, integral_rows)


@pytest.mark.parametrize("row", [1, 2])
def test_frozen_native_states_reject_even_one_ulp_of_motion(row):
    states, values, integral_rows = frozen_readout_fixture()
    states[row, -1, 0] = np.nextafter(states[row, -1, 0], np.inf)
    with pytest.raises(AssertionError, match="states must freeze bitwise"):
        stopping.verify_frozen_readouts(states, values, integral_rows)


@pytest.mark.parametrize("row", [1, 2])
def test_killed_running_outputs_do_not_receive_the_projection_allowance(row):
    states, values, integral_rows = frozen_readout_fixture()
    values[row, -1, 2] = np.nextafter(values[row, -1, 2], np.inf)
    with pytest.raises(AssertionError, match="running payoffs"):
        stopping.verify_frozen_readouts(states, values, integral_rows)
