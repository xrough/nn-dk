"""Read-only postfit, pre-reference diagnostics for all harder-regime heads.

All selection criteria are replayed on the complete independent validation
pool.  Initial/post-mean/final comparisons use one deterministic 256-row
stratified subset shared by every optimizer seed.  These generator residuals
diagnose fitting and ansatz limitations; they do not measure prediction error.
No future DK outcomes, new fits, or model selection enter this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.audit_payoff_regimes import _json, _sha  # noqa: E402
from nn_dk.data import source_hash  # noqa: E402
from nn_dk.interacting_training import _bank, _state_hash  # noqa: E402
from nn_dk.payoff_repair import RepairPayoffModel  # noqa: E402

KIND = "postfit-pre-reference-stratified-generator-validation-v1"
PHASES = ("initial", "selected_mean_covariance_initial", "final_selected")
SAMPLE_POLICY = "profile-x-snapshot-x-horizon-quartile; eight equally-spaced minimum-mass ranks per stratum"
POOL_KEYS = ("mean", "covariance", "source", "mean_residual", "covariance_residual")


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _npz(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": _sha(path),
            "arrays": {name: {"shape": list(np.asarray(value).shape), "sha256": _array_sha(value)}
                       for name, value in values.items()}}


def deterministic_subset(states, horizons, profiles, snapshot_times, horizon, *, per_stratum=8):
    """Current-state-only coverage, preserving rare low-mass validation rows.

    This diagnostic sample is deliberately stratified rather than IID.  Its
    pooled residual is not the full validation selection criterion, which is
    replayed separately.  No future payoff or optimizer outcome chooses rows.
    """
    states, horizons = np.asarray(states), np.asarray(horizons)
    profiles, snapshots = np.asarray(profiles, dtype=str), np.asarray(snapshot_times)
    n = len(states)
    if (states.ndim != 2 or horizons.shape != (n,) or profiles.shape != (n,)
            or snapshots.shape != (n,) or n < 1 or per_stratum < 1
            or not np.isfinite(states).all() or not np.isfinite(horizons).all()
            or np.any(horizons < 0) or np.any(horizons > horizon)):
        raise ValueError("Compatible finite validation current-state metadata required")
    quartile = np.minimum((4*horizons/horizon).astype(int), 3)
    minimum = states.min(-1)
    selected = []
    for profile in sorted(set(profiles)):
        for snapshot in np.unique(snapshots):
            for interval in range(4):
                candidates = np.flatnonzero((profiles == profile) & (snapshots == snapshot) & (quartile == interval))
                if len(candidates):
                    ordered = candidates[np.lexsort((candidates, minimum[candidates]))]
                    positions = np.linspace(0, len(ordered)-1, min(per_stratum, len(ordered)), dtype=int)
                    selected.extend(ordered[positions].tolist())
    indices = np.array(sorted(selected), dtype=np.int64)
    if len(set(indices)) != len(indices) or not len(indices):
        raise AssertionError("Deterministic validation subset duplicated or omitted every row")
    return indices


def validation_metadata(bank):
    """Recover historical absorption flags separately from rounded mass tests."""
    stop_map = {}
    with h5py.File(bank["path"], "r") as store:
        for profile in store:
            group = store[profile]
            for family, stop in zip(group["family_ids"].asstr()[:], group["stopped_at"][:], strict=True):
                if family in stop_map:
                    raise ValueError("Duplicated validation trajectory family")
                stop_map[family] = float(stop)
    families = np.asarray(bank["family_ids"], dtype=str)
    snapshots = np.asarray(bank["snapshot_times"], dtype=float)
    stops = np.array([stop_map[family] for family in families])
    return {"family_ids": families, "profiles": np.asarray(bank["case_ids"], dtype=str),
            "snapshot_times": snapshots, "stopped_at": stops,
            "physical_active": snapshots < stops}


def stratum_masks(states, horizons, metadata, spec, horizon):
    states, horizons = np.asarray(states), np.asarray(horizons)
    minimum = states.min(-1)
    ratio = minimum/spec.mass_stop
    masks = {"all": np.ones(len(states), dtype=bool),
             "physical_active_only": np.asarray(metadata["physical_active"], dtype=bool),
             "physically_absorbed": ~np.asarray(metadata["physical_active"], dtype=bool),
             "model_threshold_active": (states > spec.mass_stop).all(-1),
             "model_threshold_absorbed": ~(states > spec.mass_stop).all(-1)}
    for profile in sorted(set(metadata["profiles"])):
        masks[f"profile/{profile}"] = metadata["profiles"] == profile
    for snapshot in np.unique(metadata["snapshot_times"]):
        masks[f"snapshot_fraction/{snapshot/horizon:.12g}"] = metadata["snapshot_times"] == snapshot
    quartile = np.minimum((4*horizons/horizon).astype(int), 3)
    for interval in range(4):
        masks[f"remaining_horizon_quartile/{interval}"] = quartile == interval
    cuts = (("at_or_below_boundary", -np.inf, 1+1e-10),
            ("one_to_two_thresholds", 1+1e-10, 2.), ("two_to_five_thresholds", 2., 5.),
            ("five_to_ten_thresholds", 5., 10.), ("ten_to_25_thresholds", 10., 25.),
            ("above_25_thresholds", 25., np.inf))
    for name, lower, upper in cuts:
        masks[f"minimum_mass/{name}"] = (ratio > lower) & (ratio <= upper)
    return masks


def normalized_residual(value, horizons, scaling, horizon, stage):
    value = np.asarray(value, dtype=float)
    power = np.asarray(scaling["small_time_powers"], dtype=int)
    clock = np.maximum(np.asarray(horizons)/horizon, scaling["scale_time_floor"])
    if stage == "covariance":
        power, clock = power[:, None]+power[None, :], clock[:, None, None]
    elif stage == "mean":
        clock = clock[:, None]
    else:
        raise ValueError("Use mean or covariance residuals")
    return horizon*value/np.asarray(scaling[f"{stage}_scale"])/clock**power


def residual_summary(value, horizons, scaling, model, stage):
    value = np.asarray(value)
    if not len(value):
        return {"rows": 0, "empty": True}
    normalized = normalized_residual(value, horizons, scaling, model.horizon, stage)
    factor = (model.bank.scales/np.sqrt(model.context.n_particles) if stage == "mean" else
              np.outer(model.bank.scales, model.bank.scales))
    physical = value*factor
    result = {"rows": len(value), "raw_internal_rms": float(np.sqrt(np.mean(value**2))),
              "normalized_rms": float(np.sqrt(np.mean(normalized**2))),
              "normalized_objective": float(np.mean(normalized**2)),
              "raw_internal_maximum": float(abs(value).max()),
              "normalized_maximum": float(abs(normalized).max()),
              "raw_internal_rms_by_output_or_pair": np.sqrt(np.mean(value**2, axis=0)).tolist(),
              "raw_physical_rms_by_output_or_pair": np.sqrt(np.mean(physical**2, axis=0)).tolist(),
              "normalized_rms_by_output_or_pair": np.sqrt(np.mean(normalized**2, axis=0)).tolist(),
              "physical_covariance_units": "N-scaled covariance per unit time" if stage == "covariance" else None}
    integral = model.bank.integral_rows
    groups = ({"terminal": ~integral, "integral": integral} if stage == "mean" else
              {"terminal_terminal": (~integral)[:, None] & (~integral)[None, :],
               "terminal_integral": integral[:, None] ^ integral[None, :],
               "integral_integral": integral[:, None] & integral[None, :]})
    result["groups"] = {name: {"raw_internal_rms": float(np.sqrt(np.mean(value[:, mask]**2))),
                               "normalized_rms": float(np.sqrt(np.mean(normalized[:, mask]**2))),
                               "raw_physical_rms": float(np.sqrt(np.mean(physical[:, mask]**2)))}
                        for name, mask in groups.items() if mask.any()}
    return result


def evaluate_pool(model, states, horizons, *, batch_size=128):
    """Recompute equations from the two jets with one pass per batch."""
    chunks = {name: [] for name in POOL_KEYS}
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            m = torch.tensor(states[start:start+batch_size], dtype=torch.float64)
            t = torch.tensor(horizons[start:start+batch_size], dtype=torch.float64)
            mean, covariance = model._moment_jets(m, t)
            source = mean.face_derivatives @ mean.face_derivatives.transpose(-1, -2)
            running = np.sqrt(model.context.n_particles)*(m @ model.normalized_running_weights.T)
            running = torch.where(model._active(m)[:, None], running, 0.)
            values = {"mean": mean.value, "covariance": covariance.value, "source": source,
                      "mean_residual": mean.time_derivative-mean.generator-running,
                      "covariance_residual": covariance.time_derivative-covariance.generator-source}
            for name, value in values.items():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite validation jet; no row may be omitted")
                chunks[name].append(value.numpy())
    return {name: np.concatenate(values) for name, values in chunks.items()}


def selected_mean_hashes(saved):
    names = ("best_mean_state", "model_state", "best_state")
    hashes = {name: _state_hash({key: value for key, value in saved[name].items()
                                if key.startswith("mean_network.")}) for name in names}
    if len(set(hashes.values())) != 1:
        raise ValueError("Selected mean changed during covariance fitting")
    for key in saved["initial_state"]:
        if key.startswith("covariance_network.") and not torch.equal(
                saved["initial_state"][key], saved["best_mean_state"][key]):
            raise ValueError("Covariance head changed before the mean phase completed")
    return hashes


def covariance_phase_baseline(model, saved):
    baseline = RepairPayoffModel(model.spec, model.context, model.horizon, model.bank,
                                 model.width, model.quadrature_nodes)
    state = {key: (saved["best_mean_state"][key] if key.startswith("mean_network.") else value)
             for key, value in saved["initial_state"].items()}
    baseline.load_state_dict(state)
    return baseline.eval()


def bounded_payoff_diagnostics(model, values, horizons):
    """Necessary bounded-variable checks, separate from residual smallness."""
    phi, q = model.bank.terminal_weights, model.bank.running_weights
    lower = phi.min(-1)[None]+np.minimum(0., np.asarray(horizons)[:, None]*q.min(-1))
    upper = phi.max(-1)[None]+np.maximum(0., np.asarray(horizons)[:, None]*q.max(-1))
    mean = values["mean"]*model.bank.scales/np.sqrt(model.context.n_particles)
    covariance = values["covariance"]*np.outer(model.bank.scales, model.bank.scales)/model.context.n_particles
    variance = np.diagonal(covariance, axis1=-2, axis2=-1)
    excess = np.maximum(lower-mean, mean-upper)
    variance_excess = variance-(upper-mean)*(mean-lower)
    tolerance = 1e-12
    return {"mean_bound_violations_by_output": (excess > tolerance).sum(0).tolist(),
            "maximum_mean_bound_excess_by_output": np.maximum(0., excess.max(0)).tolist(),
            "bounded_variance_violations_by_output": (variance_excess > tolerance).sum(0).tolist(),
            "maximum_bounded_variance_excess_by_output": np.maximum(0., variance_excess.max(0)).tolist(),
            "minimum_internal_covariance_eigenvalue": float(np.linalg.eigvalsh(values["covariance"]).min()),
            "maximum_internal_covariance_asymmetry": float(abs(values["covariance"]
                                                               -values["covariance"].swapaxes(-1, -2)).max()),
            "scope": "Necessary simplex-payoff mean/variance bounds, not sufficient law consistency"}


def _ratios(numerator, denominator):
    return [float(a/b) if b > 0 else None for a, b in zip(numerator, denominator, strict=True)]


def _describe_phases(model, pools, horizons, masks, scaling):
    result = {}
    for phase, values in pools.items():
        result[phase] = {}
        for name in ("all", "physical_active_only", "physically_absorbed"):
            mask = masks[name]
            result[phase][name] = {stage: residual_summary(values[f"{stage}_residual"][mask],
                horizons[mask], scaling, model, stage) for stage in ("mean", "covariance")}
    return result


def _sample_arrays(bank, metadata, indices):
    return {"indices": indices, "states": np.asarray(bank["states"])[indices],
            "horizons": np.asarray(bank["horizons"])[indices],
            "family_ids": np.asarray(metadata["family_ids"])[indices].astype("S64"),
            "profiles": np.asarray(metadata["profiles"])[indices].astype("S32"),
            "snapshot_times": np.asarray(metadata["snapshot_times"])[indices],
            "physical_active": np.asarray(metadata["physical_active"])[indices],
            "stopped_at": np.asarray(metadata["stopped_at"])[indices]}


def _existing_sample(path, expected):
    if path.exists():
        with np.load(path) as previous:
            if set(previous.files) != set(expected) or any(not np.array_equal(previous[key], value)
                                                         for key, value in expected.items()):
                raise ValueError("Saved diagnostic validation sample changed")
        return {"path": str(path.resolve()), "sha256": _sha(path),
                "arrays": {key: {"shape": list(value.shape), "sha256": _array_sha(value)}
                           for key, value in expected.items()}}
    return _npz(path, expected)


def run(root):
    from analysis import run_payoff_regimes as runner

    torch.set_num_threads(1)
    started, cpu_started = time.monotonic(), time.process_time()
    root = Path(root).resolve()
    plan, _ = runner._assert_frozen(root)
    cases = runner.case_names(plan)
    if any((root/"reference"/case/"reference.h5").exists() for case in cases):
        raise ValueError("This validation diagnostic must precede all final-reference outcomes")
    lock = runner.verify_fit_lock(root)
    if len(lock["models"]) != 3*len(cases):
        raise ValueError("Every fixed regime and all three optimizer seeds are required")
    output = root/"audit/postfit_validation_diagnostic.json"
    identity = {"kind": KIND, "source_hash": source_hash(), "script_sha256": _sha(__file__),
                "fit_lock_sha256": _sha(root/"payoffs/fit_lock.json"),
                "preregistration_sha256": _sha(root/"payoffs/preregistration.json"),
                "data_lock_sha256": _sha(root/"payoffs/data_lock.json"),
                "sample_policy": SAMPLE_POLICY, "maximum_diagnostic_rows_per_case": 256,
                "full_final_validation_criterion_replay": True,
                "all_seeds_retained": True, "future_test_outcomes_opened": False}
    previous = json.loads(output.read_text()) if output.exists() else None
    if previous is not None and previous.get("identity") != identity:
        raise ValueError("Postfit audit identity changed; preserve its earlier output")
    result = {"identity": identity, "complete": False,
              "created_utc": datetime.now(timezone.utc).isoformat(), "samples": {}, "records": [],
              "interpretation": "Residuals diagnose local equation fitting; only independent reference outcomes assess prediction accuracy.",
              "subset_scope": "Deliberately stratified coverage; its pooled objective is not the optimizer's validation criterion."}
    retained = {row["name"]: row for row in previous.get("records", [])} if previous else {}
    for case in cases:
        cfg = runner.configuration(case, plan)
        validation = runner.load_state_bank(root, case, "validation")
        training = runner.load_state_bank(root, case, "train")
        _, vf = _bank(validation, cfg, "validation")
        _, tf = _bank(training, cfg, "train")
        metadata = validation_metadata(validation)
        indices = deterministic_subset(validation["states"], validation["horizons"], metadata["profiles"],
                                       metadata["snapshot_times"], cfg.horizon)
        sample_path = root/"audit/validation_samples"/(case+".npz")
        sample_arrays = _sample_arrays(validation, metadata, indices)
        artifact = _existing_sample(sample_path, sample_arrays)
        result["samples"][case] = {"sample_indices": indices.tolist(),
                                   "sample_indices_sha256": _array_sha(indices),
                                   "diagnostic_rows": len(indices), "full_validation_rows": len(validation["states"]),
                                   "full_validation_fingerprint": vf,
                                   "sample_artifact": artifact}
        states, horizons = validation["states"], validation["horizons"]
        full_masks = stratum_masks(states, horizons, metadata, cfg.spec, cfg.horizon)
        subset_masks = {key: mask[indices] for key, mask in full_masks.items()}
        known_coefficients = None
        for item in (row for row in lock["models"] if row["case"] == case):
            model = runner.load_declared_model(item)
            path = Path(item["checkpoint_path"])
            saved = torch.load(path, map_location="cpu", weights_only=True)
            hashes = selected_mean_hashes(saved)
            if (saved["provenance"]["data_fingerprints"] != {"training": tf, "validation": vf}
                    or saved["provenance"]["source_hash"] != source_hash()
                    or saved["provenance"]["seed"] != item["seed"]
                    or saved["provenance"]["config_hash"] != runner.repaired_configuration(case, plan).config_hash
                    or _sha(path) != item["checkpoint_sha256"]
                    or _sha(item["metadata_path"]) != item["metadata_sha256"]):
                raise ValueError("Strict restored checkpoint differs from its physical pools or frozen slot")
            prior = retained.get(item["name"])
            if prior is not None:
                if (prior["checkpoint_sha256"] != _sha(path)
                        or prior["sample_indices_sha256"] != _array_sha(indices)
                        or _sha(prior["numeric_evidence"]["path"]) != prior["numeric_evidence"]["sha256"]):
                    raise ValueError("Previously replayed validation audit evidence changed")
                result["records"].append(prior)
                print(item["name"], "strictly restored; prior numerical audit reused", flush=True)
                continue
            scaling = saved["provenance"]["loss_scaling"]
            if known_coefficients is not None:
                model._coefficient_cache = known_coefficients.copy()
            final_full = evaluate_pool(model, states, horizons)
            known_coefficients = model._coefficient_cache
            full_summary = {stage: residual_summary(final_full[f"{stage}_residual"], horizons,
                                                    scaling, model, stage)
                            for stage in ("mean", "covariance")}
            selection_replay = {}
            for stage in ("mean", "covariance"):
                expected = saved["best_validation"][stage]
                actual = full_summary[stage]
                np.testing.assert_allclose(actual["normalized_objective"], expected["objective"], rtol=1e-10, atol=1e-14)
                np.testing.assert_allclose(actual["raw_internal_rms"], expected["raw_rms"], rtol=1e-10, atol=1e-14)
                selection_replay[stage] = {"saved_objective": expected["objective"],
                                           "replayed_objective": actual["normalized_objective"],
                                           "saved_raw_rms": expected["raw_rms"],
                                           "replayed_raw_rms": actual["raw_internal_rms"]}
            initial = RepairPayoffModel(model.spec, model.context, model.horizon, model.bank,
                                         model.width, model.quadrature_nodes)
            initial.load_state_dict(saved["initial_state"])
            postmean = covariance_phase_baseline(model, saved)
            # Only immutable known physical coefficients are shared. No fitted
            # value, derivative, label or optimizer state enters this cache.
            for control in (initial, postmean):
                control._coefficient_cache = model._coefficient_cache.copy()
            pools = {"initial": evaluate_pool(initial, states[indices], horizons[indices]),
                     "selected_mean_covariance_initial": evaluate_pool(postmean, states[indices], horizons[indices]),
                     "final_selected": {key: value[indices] for key, value in final_full.items()}}
            source_difference = float(abs(pools["selected_mean_covariance_initial"]["source"]
                                          -pools["final_selected"]["source"]).max())
            source_tolerance = 2e-12*max(1., float(abs(pools["final_selected"]["source"]).max()))
            if source_difference > source_tolerance:
                raise AssertionError("Covariance-stage fitting changed the frozen mean source")
            phase_summary = _describe_phases(model, pools, horizons[indices], subset_masks, scaling)
            full_stratified = {name: {stage: residual_summary(final_full[f"{stage}_residual"][mask],
                horizons[mask], scaling, model, stage) for stage in ("mean", "covariance")}
                for name, mask in full_masks.items()}
            source_initial = np.diagonal(pools["initial"]["source"], axis1=-2, axis2=-1).mean(0)
            source_final = np.diagonal(pools["final_selected"]["source"], axis1=-2, axis2=-1).mean(0)
            cov_initial = np.diagonal(pools["initial"]["covariance"], axis1=-2, axis2=-1).mean(0)
            cov_final = np.diagonal(pools["final_selected"]["covariance"], axis1=-2, axis2=-1).mean(0)
            arrays = {f"subset_{phase}_{key}": value for phase, pool in pools.items() for key, value in pool.items()}
            arrays.update({f"full_final_{key}": value for key, value in final_full.items()})
            numeric = _npz(root/"audit/validation_numeric"/(item["name"]+".npz"), arrays)
            row = {"name": item["name"], "case": case, "seed": item["seed"],
                   "bank": item["bank"], "checkpoint_path": str(path.resolve()),
                   "checkpoint_sha256": _sha(path), "metadata_sha256": _sha(item["metadata_path"]),
                   "source_hash": source_hash(), "strict_restoration_and_pool_fingerprints_verified": True,
                   "sample_indices_sha256": _array_sha(indices), "selected_mean_hashes": hashes,
                   "selected_mean_unchanged": True,
                   "covariance_phase_initialization": "selected mean tensors plus literal initial covariance/geometry tensors",
                   "initial_covariance_validation_uses_obsolete_mean_source": True,
                   "allocation": saved["steps_done"], "selected_steps": saved["best_step"],
                   "training_seconds": saved["training_seconds"],
                   "selection_criterion_replay": selection_replay,
                   "diagnostic_subset_phase_residuals": phase_summary,
                   "full_selected_stratified_residuals": full_stratified,
                   "full_selected_active_counts": {
                       "physical_active": int(full_masks["physical_active_only"].sum()),
                       "physical_absorbed": int(full_masks["physically_absorbed"].sum()),
                       "model_threshold_active": int(full_masks["model_threshold_active"].sum()),
                       "rounded_boundary_active_disagreement": int(np.sum(
                           full_masks["physical_active_only"] != full_masks["model_threshold_active"]))},
                   "source_and_covariance_ratios_to_heat_control": {
                       "source_final_over_heat_by_output": _ratios(source_final, source_initial),
                       "covariance_final_over_heat_by_output": _ratios(cov_final, cov_initial),
                       "mean_source_unchanged_during_covariance_maximum": source_difference,
                       "floating_point_batch_replay_tolerance": source_tolerance,
                       "scope": "Diagnostic anchor ratios, not accuracy against interacting truth"},
                   "full_selected_payoff_consistency": bounded_payoff_diagnostics(model, final_full, horizons),
                   "numeric_evidence": numeric}
            result["records"].append(row)
            result.update(audit_wall_seconds=time.monotonic()-started, audit_cpu_seconds=time.process_time()-cpu_started)
            _json(output, result)
            print(item["name"], "full criterion and stratified diagnostic verified", flush=True)
    if any((root/"reference"/case/"reference.h5").exists() for case in cases):
        raise ValueError("Held-out reference appeared before this validation audit completed")
    if source_hash() != identity["source_hash"] or _sha(root/"payoffs/fit_lock.json") != identity["fit_lock_sha256"]:
        raise ValueError("Source or fitted inventory changed during validation audit")
    result.update(complete=len(result["records"]) == len(lock["models"]),
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  audit_wall_seconds=time.monotonic()-started,
                  audit_cpu_seconds=time.process_time()-cpu_started,
                  all_references_absent_at_completion=True,
                  no_selection_refit_or_future_outcome_access=True)
    if not result["complete"]:
        raise AssertionError("Postfit diagnostic omitted a frozen model slot")
    _json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.root)
    print(json.dumps({"complete": result["complete"], "models": len(result["records"]),
                      "wall_seconds": result["audit_wall_seconds"],
                      "audit": str((args.root/"audit/postfit_validation_diagnostic.json").resolve())}))


if __name__ == "__main__":
    main()
