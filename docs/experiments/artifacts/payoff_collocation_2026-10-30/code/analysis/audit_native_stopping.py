"""Portable synthetic stopping verification, separate from frozen study audits.

The historical ``audit_payoff_regimes.py`` is checksum-bound to completed
studies and stays unchanged. This successor retains its three-path control,
with exact stopped states and running-payoff checks. Repeated linear payoff
projections may differ by floating-point roundoff between BLAS kernels.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from analysis.compiled_native import compile_native  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.native import boundary_step, native_coefficients, simulate_native  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.types import Context  # noqa: E402

KIND = "portable-synthetic-native-stopping-verification-v1"
HISTORICAL_HELPER_SHA256 = "8d393f940959de5724ae55cf9f76da77160519fdb0e7225f069d58ae294ee211"
PROJECTED_PAYOFF_ATOL = 2e-15


def verify_frozen_readouts(states, observed, integral_rows):
    """Check the absorbed path 2 and first-segment crossing path 1 exactly.

    Only terminal linear projections receive a roundoff allowance. Physical
    state freezing is bitwise, and killed running-only outputs remain exact.
    No trajectory is dropped or repaired by this verifier.
    """
    states = np.asarray(states, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    integral_rows = np.asarray(integral_rows, dtype=bool)
    if (states.ndim != 3 or states.shape[0] != 3 or states.shape[1] < 2
            or observed.ndim != 3 or observed.shape[:2] != states.shape[:2]
            or integral_rows.shape != (observed.shape[2],)
            or not integral_rows.any() or integral_rows.all()
            or not np.isfinite(states).all() or not np.isfinite(observed).all()):
        raise ValueError("Complete finite three-path states and terminal/running readouts required")
    for row, start in ((2, 0), (1, 1)):
        actual = states[row, start:]
        expected = np.broadcast_to(states[row, start], actual.shape)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64),
                                      err_msg="Stopped physical states must freeze bitwise")
    running = observed[..., integral_rows]
    np.testing.assert_array_equal(running[2], np.zeros_like(running[2]),
                                  err_msg="Initially absorbed running payoffs must stay exactly zero")
    np.testing.assert_array_equal(running[1, 1:], np.broadcast_to(running[1, 1], running[1, 1:].shape),
                                  err_msg="Accumulated running payoffs must freeze exactly at killing")
    terminal = observed[..., ~integral_rows]
    maximum_projection_error = 0.
    for row, start in ((2, 0), (1, 1)):
        actual = terminal[row, start:]
        expected = np.broadcast_to(terminal[row, start], actual.shape)
        np.testing.assert_allclose(actual, expected, atol=PROJECTED_PAYOFF_ATOL, rtol=0,
                                   err_msg="Frozen terminal projections exceed floating-point roundoff")
        maximum_projection_error = max(maximum_projection_error, float(abs(actual-expected).max()))
    return {"frozen_states_bitwise_equal": True, "initially_absorbed_running_payoffs_exactly_zero": True,
            "killed_running_payoffs_exactly_frozen": True,
            "maximum_frozen_terminal_projection_error": maximum_projection_error,
            "projected_payoff_constancy_atol": PROJECTED_PAYOFF_ATOL,
            "projected_payoff_constancy_rtol": 0.}


def stopped_native_controls(backend=None, n_cells=32):
    """Synthetic controlled hit, already-absorbed path, and complete integral."""
    backend = compile_native() if backend is None else backend
    spec = NativeDKSpec(n_cells)
    context = Context(n_particles=2048, diffusion=1., kappa=6., a2=.1)
    dt, steps = 2**-10, 4
    initial = np.full((3, n_cells), 1/n_cells)
    initial[2, 0] = spec.mass_stop
    initial[2, 1] += 1/n_cells-spec.mass_stop
    dw = np.zeros((3, steps, n_cells))
    dw[1, 0, 0] = 3.
    times = np.arange(steps+1)*dt
    families = tuple(f"synthetic-control-{i}" for i in range(3))
    exact = simulate_native(context, spec, times, dt, 2026102804, 3, initial,
                            dw, family_ids=families)
    compiled = backend(initial, dw, context, spec, dt)
    np.testing.assert_allclose(compiled["states"], exact.states, atol=2e-14, rtol=0)
    np.testing.assert_allclose(compiled["stopped_at"], exact.stopped_at, atol=2e-14, rtol=0)
    for stops in (exact.stopped_at, compiled["stopped_at"]):
        if not (np.isinf(stops[0]) and 0 < stops[1] < dt and stops[2] == 0):
            raise AssertionError("The synthetic crossing/absorption control did not stop as designed")
    drift, amplitude = native_coefficients(initial[1], context, spec).drift, native_coefficients(
        initial[1], context, spec).face_amplitudes
    noise = amplitude * dw[1, 0] / np.sqrt(context.n_particles)
    proposal = initial[1] + dt*drift + np.roll(noise, 1)-noise
    stopped, crossed, stop = boundary_step(initial[1], proposal, 0., dt, spec)
    np.testing.assert_allclose(stopped, compiled["states"][1, 1], atol=2e-14, rtol=0)
    assert crossed and np.isclose(stop, exact.stopped_at[1], atol=2e-14, rtol=0)
    bank = PayoffBank.mixed_terminal_integral(spec, steps*dt)
    observed = bank.evaluate_paths(compiled["states"], times, compiled["stopped_at"])
    manual = compiled["states"] @ bank.terminal_weights.T
    accumulated = np.zeros((3, 4))
    for step in range(steps):
        exposure = np.minimum(dt, np.maximum(exact.stopped_at-times[step], 0))
        endpoints = (compiled["states"][:, step]+compiled["states"][:, step+1]) / 2
        accumulated += exposure[:, None] * (endpoints @ bank.running_weights.T)
        manual[:, step+1] += accumulated
    np.testing.assert_allclose(observed, manual, atol=2e-15, rtol=0)
    frozen = verify_frozen_readouts(compiled["states"], observed, bank.integral_rows)
    stop_error = float(abs(compiled["stopped_at"][1:]-exact.stopped_at[1:]).max())
    return {"complete": True, "paths": 3, "all_paths_retained": True,
            "never_stopped": 1, "first_segment_crossing": 1, "already_absorbed": 1,
            "maximum_cpp_vs_native_state_error": float(abs(compiled["states"]-exact.states).max()),
            "maximum_cpp_vs_native_stopping_time_error": stop_error,
            "maximum_killed_integral_manual_error": float(abs(observed-manual).max()),
            "mass_error": float(abs(compiled["states"].sum(-1)-1).max()),
            "backend_fingerprint": backend.metadata["fingerprint"],
            "kind": KIND, "source_hash": source_hash(), "frozen_readout_verification": frozen,
            "provenance": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "adapted_from": "analysis/audit_payoff_regimes.py:stopped_native_controls",
                "historical_helper_sha256": HISTORICAL_HELPER_SHA256,
                "historical_records_rewritten": False},
            "no_held_out_DK_outcomes_generated_or_read": True}
