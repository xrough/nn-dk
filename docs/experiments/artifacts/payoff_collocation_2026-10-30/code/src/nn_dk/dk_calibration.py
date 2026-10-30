"""Resumable A0 calibration for native stopped-DK convergence studies.

Finite reference refinements provide empirical differences, never a certified
bound on the remaining discretization bias. Linear moment calibration uses a
different, explicitly unstopped, additive-noise heat system.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from .convergence_config import ConvergenceConfig
from .convergence_metrics import compare_native, native_normalization
from .data import json_ready, source_hash, write_json
from .native import (
    coupled_native_refinement,
    linear_gaussian_statistics,
    native_coefficients,
    physical_observable_weights,
)
from .native_types import NativeBatch
from .observations import incidence_matrix
from .types import seeded_rng

_WORKFLOW = "native-dk-a0-calibration-v1"
_LEVELS = ("coarse", "half", "quarter")


def _expired(deadline):
    return deadline is not None and time.monotonic() >= deadline


def _validate_provenance(record, config, fingerprint):
    if (record.get("config_hash") != config.config_hash
            or record.get("source_hash") != fingerprint):
        raise ValueError("Native calibration source/configuration mismatch; use a new directory")


def _euler_matrix_moments(context, spec, queries, dt):
    """Independent matrix recursion to audit the spectral Euler identities."""
    b = incidence_matrix(spec.n_cells)
    operator = -context.diffusion / spec.h**2 * b @ b.T
    transition = np.eye(spec.n_cells) + dt * operator
    mass = np.full(spec.n_cells, 1 / spec.n_cells)
    covariance = np.diag(mass) - np.outer(mass, mass)
    q = native_coefficients(mass, context, spec).covariance
    weights = physical_observable_weights(spec, queries)
    width = len(weights)
    mean = np.empty(queries.dimension)
    joint = np.zeros((queries.dimension, queries.dimension))
    saves = {round(t / dt): i for i, t in enumerate(queries.times)}
    crosses = []
    maximum_null = float(np.max(np.abs(covariance.sum(0))))
    for step in range(max(saves) + 1):
        if step in saves:
            i = saves[step]
            current = slice(i * width, (i + 1) * width)
            mean[current] = weights @ mass
            joint[current, current] = weights @ covariance @ weights.T
            for j, cross in enumerate(crosses):
                previous = slice(j * width, (j + 1) * width)
                block = weights @ cross @ weights.T
                joint[previous, current] = block
                joint[current, previous] = block.T
            crosses.append(covariance.copy())
        if step == max(saves):
            break
        mass = transition @ mass
        covariance = transition @ covariance @ transition.T + dt * q
        crosses = [cross @ transition.T for cross in crosses]
        maximum_null = max(maximum_null, float(np.max(np.abs(covariance.sum(0)))))
    return mean, joint, maximum_null


def _linear_calibration(config, existing=None, deadline=None):
    context, spec, queries = config.context("uniform"), config.spec, config.queries
    analytic = linear_gaussian_statistics(context, spec, queries)
    mass = np.full(spec.n_cells, 1 / spec.n_cells)
    oracle = native_coefficients(mass, context, spec)
    b = incidence_matrix(spec.n_cells)
    operator = -context.diffusion / spec.h**2 * b @ b.T
    null_mode = {
        "drift_operator_max_abs": float(np.max(np.abs(operator @ np.ones(spec.n_cells)))),
        "noise_covariance_max_abs": float(np.max(np.abs(oracle.covariance @ np.ones(spec.n_cells)))),
        "drift_sum_abs": float(abs(oracle.drift.sum())),
        "initial_covariance_max_abs": float(np.max(np.abs(
            (np.diag(mass) - np.outer(mass, mass)) @ np.ones(spec.n_cells)))),
    }
    null_mode["passed"] = bool(max(null_mode.values()) < 1e-11)
    normal = native_normalization(spec, queries)
    std, covscale = np.asarray(normal["mean"]), np.asarray(normal["scaled_covariance"])
    rows = [] if existing is None else list(existing.get("euler_refinement", []))
    done = {row["step_size"] for row in rows}
    for dt in config.step_sizes:
        if dt in done:
            continue
        if _expired(deadline):
            break
        spectral = linear_gaussian_statistics(context, spec, queries, step_size=dt)
        mean, covariance, conservation = _euler_matrix_moments(context, spec, queries, dt)
        mean_error = np.sqrt(context.n_particles) * (mean - analytic["mean"]) / std
        covariance_error = (covariance - analytic["scaled_covariance"]) / covscale
        agreement = max(float(np.max(np.abs(mean - spectral["mean"]))),
                        float(np.max(np.abs(covariance - spectral["scaled_covariance"]))))
        rows.append({
            "step_size": dt, "mean": mean, "scaled_covariance": covariance,
            "errors": {"mean_normalized": np.abs(mean_error),
                       "scaled_covariance_normalized": np.abs(covariance_error),
                       "mean_normalized_rmse": float(np.sqrt(np.mean(mean_error**2))),
                       "scaled_covariance_normalized_rmse": float(np.sqrt(np.mean(covariance_error**2)))},
            "matrix_spectral_max_abs_difference": agreement,
            "matrix_covariance_null_mode_max_abs": conservation,
            "passed": bool(agreement < 1e-10 and conservation < 1e-10),
        })
    complete = len(rows) == len(config.step_sizes)
    passed = null_mode["passed"] and all(row["passed"] for row in rows)
    return json_ready({
        "complete": complete, "status": "passed" if complete and passed else
        "failed" if complete else "partial", "exact": analytic,
        "euler_refinement": rows, "conserved_null_mode": null_mode,
        "normalization": normal,
        "interpretation": "Unstopped frozen-uniform-noise heat moments; not the stopped DK law. "
                          "Euler moments are independently propagated as matrices and checked against spectral identities.",
        "temporal_order": None,
    })


def _refinement_batches(config, case, directory, fingerprint, deadline):
    """Persist one complete coupled family at a time, bounding memory/work units."""
    context, spec = config.context(case), config.spec
    times = config.times
    count = config.refinement_paths
    families = tuple(f"a0/{config.seed}/{case}/family{i}" for i in range(count))
    path = directory / f"calibration_refinement_{case}.h5"
    with h5py.File(path, "a") as store:
        if "config_hash" in store.attrs:
            _validate_provenance(dict(store.attrs), config, fingerprint)
        else:
            store.attrs["config_hash"] = config.config_hash
            store.attrs["source_hash"] = fingerprint
            store.attrs["workflow"] = _WORKFLOW
            store.create_dataset("times", data=times)
            store.create_dataset("family_ids", data=np.asarray(families, dtype=h5py.string_dtype()))
            store.create_dataset("completed", data=np.zeros(count, dtype=bool))
            store.create_dataset("seconds", data=np.zeros(count))
            for level in _LEVELS:
                group = store.create_group(level)
                group.create_dataset("states", shape=(count, len(times), spec.n_cells),
                                     dtype="f8", chunks=(1, len(times), spec.n_cells))
                group.create_dataset("stopped_at", shape=(count,), dtype="f8")
            store.flush()
        for i in range(count):
            if store["completed"][i]:
                continue
            if _expired(deadline):
                break
            seed = int(seeded_rng(config.seed, "a0-reference-refinement", case, i).integers(0, 2**63))
            tick = time.perf_counter()
            result = coupled_native_refinement(context, spec, times, config.reference_dt,
                                               seed, n_paths=1, family_ids=(families[i],))
            for level, batch in result.items():
                store[f"{level}/states"][i] = batch.states[0]
                store[f"{level}/stopped_at"][i] = batch.stopped_at[0]
            store["seconds"][i] = time.perf_counter() - tick
            # A completed family is an atomic resume unit across all three levels.
            store["completed"][i] = True
            store.flush()
        completed = int(np.count_nonzero(store["completed"][:]))
        elapsed = float(store["seconds"][:].sum())
        if completed != count:
            return None, {"complete": False, "completed_paths": completed,
                          "requested_paths": count, "simulation_seconds": elapsed}
        batches = {}
        for level, divisor in zip(_LEVELS, (1, 2, 4), strict=True):
            batches[level] = NativeBatch(
                times, store[f"{level}/states"][:], spec, context,
                store[f"{level}/stopped_at"][:], families,
                {"dt": config.reference_dt / divisor, "all_paths_retained": True})
        return batches, {"complete": True, "completed_paths": count,
                         "requested_paths": count, "simulation_seconds": elapsed}


def _empirical_bias(comparison, config):
    errors = comparison["errors"]
    uncertainty = comparison["uncertainty"]
    means = np.asarray(errors["mean_normalized"])
    covariance = np.asarray(errors["scaled_covariance_normalized"])
    stops = np.abs(np.asarray(errors["stopping_probability_difference"]))
    estimate = max(float(np.max(means)), float(np.max(covariance)), float(np.max(stops)))
    bound = None
    intervals = {}
    if uncertainty["available"]:
        intervals = {key: uncertainty[key] for key in (
            "mean_normalized_ci95", "scaled_covariance_normalized_ci95",
            "stopping_probability_difference_ci95")}
        bound = max(float(np.max(np.abs(np.asarray(ci)))) for ci in intervals.values())
    return {
        "status": "uncertified", "normalized_bound": bound,
        "empirical_normalized_difference": estimate,
        "empirical_normalized_difference_upper_ci95": bound,
        "empirical_contrast_ci95": intervals,
        "uncertainty_budget": config.uncertainty_budget,
        "empirical_difference_within_budget": bool(bound is not None and bound <= config.uncertainty_budget),
        "reason": "A finite half/quarter refinement difference, even with paired Monte Carlo intervals, "
                  "does not bound the remaining bias to the continuous-time stopped discrete DK law. "
                  "No convergence order or asymptotic extrapolation is assumed.",
    }


def calibrate_native(config: ConvergenceConfig, output_dir, deadline=None) -> dict:
    """Run/resume A0 with pointwise paired intervals and strict provenance.

    ``deadline`` is an absolute ``time.monotonic()`` deadline. Work checks it
    between independent coupled-family units; a path already in progress finishes
    and commits all refinement levels. JSON and HDF5 checkpoints permit exact
    resumption without silently changing source, target, or configuration.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "calibration.json"
    fingerprint = source_hash()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        _validate_provenance(json.loads(manifest_path.read_text()), config, fingerprint)
    if path.exists():
        result = json.loads(path.read_text())
        if result.get("workflow") != _WORKFLOW:
            raise ValueError("Output contains a different calibration workflow")
        _validate_provenance(result, config, fingerprint)
        if result.get("complete"):
            return result
    else:
        result = {"workflow": _WORKFLOW, "config_hash": config.config_hash,
                  "source_hash": fingerprint, "complete": False, "status": "partial",
                  "linear_gaussian": {}, "native_refinement": {}, "reference_bias": {},
                  "seconds_total": 0.0, "scope": "Stopped spatially discrete DK at fixed G,N"}
    tick = time.perf_counter()
    previous_seconds = result.get("seconds_total", 0.0)

    def commit():
        result["seconds_this_call"] = time.perf_counter() - tick
        result["seconds_total"] = previous_seconds + result["seconds_this_call"]
        write_json(path, result)

    if not _expired(deadline):
        result["linear_gaussian"] = _linear_calibration(config, result["linear_gaussian"], deadline)
        commit()
    for case in config.cases:
        if result["native_refinement"].get(case, {}).get("complete"):
            continue
        if _expired(deadline):
            break
        batches, progress = _refinement_batches(config, case, directory, fingerprint, deadline)
        if batches is None:
            result["native_refinement"][case] = progress
            commit()
            break
        comparisons = {}
        for coarse, fine in (("coarse", "half"), ("half", "quarter"), ("coarse", "quarter")):
            comparisons[f"{coarse}_vs_{fine}"] = compare_native(
                batches[coarse], batches[fine], config.queries,
                n_bootstrap=config.bootstrap_samples, seed=config.seed, paired=True,
                tolerance=config.tolerance, uncertainty_budget=config.uncertainty_budget)
        bias = _empirical_bias(comparisons["half_vs_quarter"], config)
        result["reference_bias"][case] = bias
        result["native_refinement"][case] = {
            **progress, "reference_steps": [config.reference_dt / divisor for divisor in (1, 2, 4)],
            "family_ids": list(batches["coarse"].family_ids), "comparisons": comparisons,
            "reference_bias": bias, "all_stopped_paths_included": True,
            "temporal_order": None,
        }
        commit()
    result["complete"] = bool(result["linear_gaussian"].get("complete") and all(
        result["native_refinement"].get(case, {}).get("complete") for case in config.cases))
    result["status"] = ("numerical_calibration_complete_reference_bias_uncertified"
                        if result["complete"] and result["linear_gaussian"]["status"] == "passed"
                        else "numerical_calibration_failed" if result["complete"] else "partial")
    commit()
    return json_ready(result)
