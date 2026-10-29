"""Controlled boundary-strip initial-time consistency of the frozen ansatz.

No trained checkpoint or held-out outcome is used.  A strictly interior state
is chosen with barrier approximately one half.  The fixed gate multiplies the
known integral initial slope and terminal instantaneous covariance; corrections
of the declared higher time orders cannot change those leading coefficients.
This diagnoses an ansatz limitation, not failure of neural moment methods.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.audit_payoff_regimes import _json, _sha  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.native import native_coefficients  # noqa: E402
from nn_dk.native_types import NativeDKSpec  # noqa: E402
from nn_dk.payoff_learning import PayoffBank  # noqa: E402
from nn_dk.payoff_repair import RepairPayoffModel  # noqa: E402
from nn_dk.types import Context  # noqa: E402

KIND = "controlled-strict-interior-boundary-gate-initial-time-defect-v1"


def controlled_state(spec, boundary_width):
    state = np.full(spec.n_cells, 1/spec.n_cells)
    near = spec.mass_stop+boundary_width*np.arctanh(.5)
    state[spec.n_cells//2] += state[0]-near
    state[0] = near
    if not np.all(state > spec.mass_stop) or not np.isclose(state.sum(), 1, atol=1e-14, rtol=0):
        raise AssertionError("Controlled state must be strictly interior and conservative")
    return state


def run_control():
    torch.set_num_threads(1)
    spec, horizon = NativeDKSpec(32), 1.
    physics = Context(n_particles=2048, diffusion=1., kappa=6., a2=.1)
    bank = PayoffBank.mixed_terminal_integral(spec, horizon)
    with torch.random.fork_rng():
        torch.manual_seed(2026102831)
        model = RepairPayoffModel(spec, physics, horizon, bank, width=4, quadrature_nodes=16)
    initial = controlled_state(spec, model.boundary_width)
    state = torch.tensor(initial[None], dtype=torch.float64)
    gate = float(model.barrier(state)[0])
    source = np.sqrt(physics.n_particles)*(initial @ model.normalized_running_weights.numpy().T)
    phi = model.normalized_weights.numpy()
    qv = native_coefficients(initial, physics, spec).covariance
    terminal_qv = phi @ qv @ phi.T
    expected_mean_defect, expected_covariance_defect = (gate-1)*source, (gate-1)*terminal_qv
    records = []
    for setting in ("zero_correction", "nonzero_arbitrary_correction"):
        if setting == "nonzero_arbitrary_correction":
            with torch.no_grad():
                model.mean_network[-1].bias.copy_(torch.tensor([.25, -.5, .75, 1.2]))
                model.covariance_network[-1].bias.copy_(torch.linspace(-.3, .4, 16))
        with torch.no_grad():
            mean, covariance = model._moment_jets(state, torch.zeros(1, dtype=torch.float64))
            mean_residual, covariance_residual = model.residual(state, torch.zeros(1, dtype=torch.float64))
        observed_mean, observed_covariance = mean_residual[0].numpy(), covariance_residual[0].numpy()
        np.testing.assert_allclose(mean.time_derivative[0, 2:].numpy(), gate*source[2:], atol=2e-12, rtol=1e-12)
        np.testing.assert_allclose(observed_mean[2:], expected_mean_defect[2:], atol=2e-12, rtol=1e-12)
        np.testing.assert_allclose(observed_covariance[:2, :2], expected_covariance_defect[:2, :2],
                                   atol=2e-12, rtol=1e-12)
        records.append({"setting": setting,
                        "integral_mean_time_derivative_at_zero": mean.time_derivative[0, 2:].tolist(),
                        "required_integral_initial_slope": source[2:].tolist(),
                        "observed_integral_mean_residual_at_zero": observed_mean[2:].tolist(),
                        "exact_integral_mean_corner_defect": expected_mean_defect[2:].tolist(),
                        "observed_terminal_covariance_residual_at_zero": observed_covariance[:2, :2].tolist(),
                        "exact_terminal_covariance_corner_defect": expected_covariance_defect[:2, :2].tolist(),
                        "integral_mean_generator_at_zero": mean.generator[0, 2:].tolist(),
                        "maximum_mean_identity_error": float(abs(observed_mean[2:]-expected_mean_defect[2:]).max()),
                        "maximum_covariance_identity_error": float(abs(observed_covariance[:2, :2]
                                                                           -expected_covariance_defect[:2, :2]).max())})
    small_time = []
    for t in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
        with torch.no_grad():
            residual = model.mean_residual(state, torch.tensor([t], dtype=torch.float64))[0, 2:].numpy()
        small_time.append({"remaining_horizon": t, "integral_mean_raw_residual": residual.tolist(),
                           "residual_minus_corner_defect": (residual-expected_mean_defect[2:]).tolist(),
                           "loss_time_multiplier_with_fixed_1e-3_safeguard": 1/max(t/horizon, 1e-3)})
    return {"complete": True, "kind": KIND, "source_hash": source_hash(),
            "created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": _sha(__file__),
            "controlled_state": initial.tolist(), "minimum_mass": float(initial.min()),
            "stopping_mass": spec.mass_stop, "strictly_interior": True, "barrier": gate,
            "boundary_width": model.boundary_width, "horizon": horizon,
            "physics": {"N": physics.n_particles, "D": physics.diffusion, "kappa": physics.kappa, "a2": physics.a2},
            "integral_mean_identity": "residual(0,m)=(g(m)-1)*sqrt(N)*q_normalized(m)",
            "terminal_covariance_identity": "residual_TT(0,m)=(g(m)-1)*Phi_normalized*Q(m)*Phi_normalized.T",
            "correction_independence": "O(t^2) integral-mean and higher-order covariance congruence corrections cannot change these leading terms",
            "records": records, "small_time_control": small_time,
            "no_checkpoint_reference_learner_or_final_test_data_opened": True,
            "no_fit_or_model_change": True,
            "limitations": ["An exact initial-time corner inconsistency is not a quantitative attribution of every finite-time prediction error.",
                            "Large learned corrections can compensate partly away from zero; validation strata and independent accuracy remain necessary.",
                            "The diagnosis applies to this fixed gated ansatz, not neural moment methods generally."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.root/"audit/boundary_corner_control.json"
    if output.exists():
        raise ValueError("Preserve the original controlled audit snapshot")
    result = run_control()
    result["reference_H5_files_present_at_control"] = [str(path) for path in (args.root/"reference").rglob("reference.h5")]
    _json(output, result)
    print(json.dumps({"complete": True, "barrier": result["barrier"], "sha256": _sha(output),
                      "integral_mean_corner_defect": result["records"][0]["exact_integral_mean_corner_defect"]}))


if __name__ == "__main__":
    main()
