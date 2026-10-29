"""Analytic martingale calibration and coupled stopped-DK reference audits."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .bsde_evaluation import (
    _cache_chunk,
    _check_deadline,
    _load_chunk,
    _provenance,
    _validate_provenance,
)
from .convergence_metrics import compare_native
from .data import json_ready, write_json
from .native import coupled_native_refinement
from .native_types import NativeBatch
from .types import seeded_rng


def brownian_quadratic_calibration(config):
    """Exact first-order martingale residual: sum(dW**2-dt), Var=2*T*dt.

    This nonzero residual is temporal discretization of the stochastic integral,
    not a lower bound on value-head error. No second, independent Brownian motion
    is introduced by the BSDE representation.
    """
    count = max(4096, config.outer_paths)
    steps = round(config.horizon/config.dt)
    dw = seeded_rng(config.seed, "bsde-brownian-calibration").normal(
        0, np.sqrt(config.dt), (count, steps))
    states = np.concatenate((np.zeros((count, 1)), np.cumsum(dw, axis=1)), axis=1)
    residual = states[:, -1]**2 - config.horizon - np.sum(2*states[:, :-1]*dw, axis=1)
    identity = np.sum(dw**2 - config.dt, axis=1)
    expected = 2*config.horizon*config.dt
    observed = np.mean(residual**2)
    # For R=dt*(chi2_steps-steps), E[R^4]=12*T^2*dt^2+48*T*dt^3.
    fourth = 12*config.horizon**2*config.dt**2 + 48*config.horizon*config.dt**3
    se = np.sqrt((fourth-expected**2)/count)
    identity_error = float(np.max(np.abs(residual-identity)))
    return {"complete": True, "passed": bool(identity_error < 1e-10
            and abs(observed-expected) <= 6*se),
            "n_independent_paths": count, "step_size": config.dt,
            "horizon": config.horizon, "expected_terminal_residual_mse": expected,
            "empirical_terminal_residual_mse": float(observed),
            "analytic_sampling_standard_error": float(se),
            "empirical_mse_ci95_normal_approximation": [float(observed-1.96*se), float(observed+1.96*se)],
            "pathwise_discrete_identity_max_abs_error": identity_error,
            "noise_convention": "The forward process and BSDE martingale use identical Brownian increments",
            "interpretation": "Quadratic-payoff first-order integral residual; nonzero terminal loss is expected"}


def ou_matrix_calibration(config):
    """Independent matrix exponential and covariance recursion vs closed forms."""
    from .bsde_data import ou_exact_moments, ou_rates

    rates = np.asarray(ou_rates(config))
    if rates.shape != (4,):
        raise ValueError("OU calibration requires four positive mode rates")
    drift, diffusion_covariance = -np.diag(rates), np.diag(2*rates)
    initial = seeded_rng(config.seed, "bsde-ou-matrix-test").normal(size=(3, 4))
    continuous_mean, continuous_cov = ou_exact_moments(initial, config.query_times, config)
    euler_mean, euler_cov = ou_exact_moments(initial, config.query_times, config, dt=config.dt)
    rows = []
    for index, horizon in enumerate(config.query_times):
        # Van Loan block exponential independently integrates the covariance.
        block = np.block([[drift, diffusion_covariance], [np.zeros((4, 4)), -drift.T]])
        exponential = expm(horizon*block)
        transition = exponential[:4, :4]
        covariance = exponential[:4, 4:] @ transition.T
        steps = round(horizon/config.dt)
        euler_transition = np.eye(4)+config.dt*drift
        em = initial.copy()
        ec = np.zeros((4, 4))
        for _ in range(steps):
            em = em @ euler_transition.T
            ec = euler_transition @ ec @ euler_transition.T + config.dt*diffusion_covariance
        continuous_error = max(float(np.max(np.abs(initial@transition.T-continuous_mean[:, index]))),
                               float(np.max(np.abs(covariance-continuous_cov[:, index]))))
        euler_error = max(float(np.max(np.abs(em-euler_mean[:, index]))),
                          float(np.max(np.abs(ec-euler_cov[:, index]))))
        rows.append({"horizon": horizon, "continuous_matrix_closed_form_max_abs": continuous_error,
                     "euler_recursion_closed_form_max_abs": euler_error,
                     "euler_vs_continuous_mean_max_abs": float(np.max(np.abs(em-initial@transition.T))),
                     "euler_vs_continuous_covariance_max_abs": float(np.max(np.abs(ec-covariance))),
                     "passed": bool(max(continuous_error, euler_error) < 1e-10)})
    return {"complete": True, "passed": all(row["passed"] for row in rows),
            "mode_rates": rates, "noise_amplitudes": np.sqrt(2*rates), "rows": rows,
            "interpretation": "Four-dimensional unstopped OU analytic calibration; not stopped nonlinear DK"}


def _native_audit(directory, config, case, deadline):
    directory = Path(directory)
    batches = {key: [] for key in ("coarse", "half", "quarter")}
    seconds = 0.
    families = tuple(f"bsde-calibration/{config.config_hash}/{case}/family{i}"
                     for i in range(config.refinement_paths))
    for index, family in enumerate(families):
        path = directory / "calibration_cache" / case / f"family_{index:06d}.npz"
        expected = {"case": case, "family_id": family, "reference_dt": config.reference_dt}
        if path.exists():
            arrays, metadata = _load_chunk(path, config, expected)
        else:
            _check_deadline(deadline)
            tick = time.perf_counter()
            seed = int(seeded_rng(config.seed, "bsde-calibration-native", case, index).integers(0, 2**63-1))
            paths = coupled_native_refinement(config.context(case), config.spec, config.query_times,
                                              config.reference_dt, seed, family_ids=(family,))
            arrays = {}
            for key, batch in paths.items():
                arrays[f"{key}_states"] = batch.states
                arrays[f"{key}_stopped_at"] = batch.stopped_at
            metadata = {**_provenance(config), **expected, "seed": seed, "seconds": time.perf_counter()-tick}
            _cache_chunk(path, arrays, metadata)
        seconds += metadata["seconds"]
        for level in batches:
            batches[level].append((arrays[f"{level}_states"], arrays[f"{level}_stopped_at"]))
    reconstructed = {
        level: NativeBatch(np.asarray(config.query_times), np.concatenate([item[0] for item in values]),
                           config.spec, config.context(case), np.concatenate([item[1] for item in values]),
                           families, {"all_paths_retained": True})
        for level, values in batches.items()}
    comparisons = {}
    for coarse, fine in (("coarse", "half"), ("half", "quarter"), ("coarse", "quarter")):
        comparisons[f"{coarse}_vs_{fine}"] = compare_native(
            reconstructed[coarse], reconstructed[fine], config.queries,
            n_bootstrap=config.bootstrap_samples, seed=config.seed, paired=True,
            tolerance=config.tolerance, uncertainty_budget=config.uncertainty_budget)
    fine = comparisons["half_vs_quarter"]
    uncertainty = fine["uncertainty"]
    endpoint = None
    if uncertainty["available"]:
        endpoint = max(float(np.max(np.abs(uncertainty[key]))) for key in (
            "mean_normalized_ci95", "scaled_covariance_normalized_ci95",
            "stopping_probability_difference_ci95"))
    max_mass_error = max(float(np.max(np.abs(batch.states.sum(-1)-1))) for batch in reconstructed.values())
    min_mass = min(float(batch.states.min()) for batch in reconstructed.values())
    bias = {"status": "uncertified", "normalized_bound": None,
            "empirical_finest_refinement_ci95_max_endpoint": endpoint,
            "empirical_difference_within_budget": bool(endpoint is not None and endpoint <= config.uncertainty_budget),
            "reason": "A finite coupled refinement contrast does not bound remaining stopped-SDE bias; no order is assumed"}
    return json_ready({"complete": True, "passed": bool(max_mass_error < 1e-10 and min_mass >= -1e-12),
                       "reference_steps": [config.reference_dt/divisor for divisor in (1, 2, 4)],
                       "n_independent_families": len(families), "family_ids": families,
                       "comparisons": comparisons, "reference_bias": bias,
                       "maximum_mass_error": max_mass_error, "minimum_saved_mass": min_mass,
                       "all_stopped_paths_included": True, "seconds": seconds})


def calibrate_bsde(directory, config, deadline=None):
    """Run/resume Brownian, OU and native-DK calibration without fitting models."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "calibration.json"
    if path.exists():
        result = json.loads(path.read_text())
        _validate_provenance(result, config)
        if result.get("workflow") != "conditional-bsde-calibration-v1":
            raise ValueError("Output contains a different calibration workflow")
        if result.get("complete"):
            return result
    else:
        result = {"workflow": "conditional-bsde-calibration-v1", **_provenance(config),
                  "complete": False, "native_refinement": {}}
    _check_deadline(deadline)
    if "brownian" not in result:
        result["brownian"] = brownian_quadratic_calibration(config)
        write_json(path, result)
    if "ou" not in result:
        result["ou"] = json_ready(ou_matrix_calibration(config))
        write_json(path, result)
    for case in config.cases:
        if case not in result["native_refinement"]:
            result["native_refinement"][case] = _native_audit(directory, config, case, deadline)
            write_json(path, result)
    result["complete"] = True
    result["numerical_passed"] = bool(result["brownian"]["passed"] and result["ou"]["passed"]
                                     and all(value["passed"] for value in result["native_refinement"].values()))
    result["passed"] = result["numerical_passed"]
    result["status"] = "numerical_calibration_complete_reference_bias_uncertified" if result["numerical_passed"] else "failed"
    write_json(path, result)
    return result
