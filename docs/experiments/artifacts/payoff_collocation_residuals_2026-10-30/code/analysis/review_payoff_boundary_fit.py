"""Read-only fitting diagnostics in common physical and fixed residual units.

The fitted v3 and archived v2 heads are evaluated on the same NEW validation
states. The v3 fixed, training-only full-horizon normalization is shared for
this diagnostic comparison; historical time-weighted v2 objectives are never
compared with the v3 robust objective. No final outcomes are opened, and no
selection, fitting, or accuracy claim follows from these equation residuals.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nn_dk.data import source_hash  # noqa: E402
from nn_dk.interacting_training import _bank, _state_hash  # noqa: E402

KIND = "postfit-common-new-validation-fixed-unit-residual-review-v1"
PHASES = ("initial", "selected_mean_covariance_initial", "final_selected")
STAGES = ("mean", "covariance")
GROUP_PREFIXES = ("profile/", "snapshot_fraction/", "remaining_horizon_quartile/", "minimum_mass/")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def _safe_ratio(numerator, denominator):
    return float(numerator/denominator) if denominator > 0 else None


def robust_values(value, delta):
    """Stable ordinary-coordinate pseudo-Huber values with quadratic origin."""
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all() or not np.isfinite(delta) or delta <= 0:
        raise ValueError("Finite residuals and a positive pseudo-Huber parameter required")
    return 2*value**2/(np.hypot(1., value/delta)+1.)


def residual_statistics(value, scale, horizon, physical_factor, delta):
    """A common unit definition without remaining-time or distance weights."""
    value, scale = np.asarray(value, dtype=float), np.asarray(scale, dtype=float)
    physical_factor = np.asarray(physical_factor, dtype=float)
    if (value.ndim not in (2, 3) or value.shape[1:] != scale.shape
            or physical_factor.shape != scale.shape or not np.isfinite(value).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(physical_factor).all() or not np.isfinite(horizon) or horizon <= 0):
        raise ValueError("Compatible finite residuals, physical scales and positive horizon required")
    if not len(value):
        return {"rows": 0, "empty": True}
    normalized = horizon*value/scale
    physical = value*physical_factor
    robust = robust_values(normalized, delta)
    return {"rows": len(value), "raw_internal_rms": float(np.sqrt(np.mean(value**2))),
            "raw_internal_maximum": float(abs(value).max()),
            "fixed_unit_rms": float(np.sqrt(np.mean(normalized**2))),
            "fixed_unit_squared_rms": float(np.mean(normalized**2)),
            "fixed_unit_maximum": float(abs(normalized).max()),
            "raw_physical_rms": float(np.sqrt(np.mean(physical**2))),
            "raw_physical_rms_by_output_or_pair": np.sqrt(np.mean(physical**2, axis=0)).tolist(),
            "fixed_unit_rms_by_output_or_pair": np.sqrt(np.mean(normalized**2, axis=0)).tolist(),
            "common_pseudo_huber_objective": float(np.mean(robust)),
            "raw_internal_square_sum": float(np.sum(value**2)),
            "physical_square_sum": float(np.sum(physical**2)),
            "fixed_unit_square_sum": float(np.sum(normalized**2)),
            "common_robust_sum": float(np.sum(robust)),
            "inverse_remaining_time_or_boundary_distance_weighting": False}


def stratified_statistics(value, scale, horizon, physical_factor, delta, masks):
    value = np.asarray(value)
    masks = {name: np.asarray(mask, dtype=bool) for name, mask in masks.items()}
    if ("all" not in masks or not masks["all"].all()
            or any(mask.shape != (len(value),) for mask in masks.values())):
        raise ValueError("Complete named row masks with an all-rows group required")
    overall = residual_statistics(value, scale, horizon, physical_factor, delta)
    if overall.get("empty"):
        raise ValueError("A complete validation pool cannot be empty")
    groups = {}
    for name, mask in masks.items():
        result = residual_statistics(value[mask], scale, horizon, physical_factor, delta)
        result["row_fraction"] = float(mask.mean())
        if not result.get("empty"):
            for key, denominator in (("raw_internal_square_sum", "raw_square_fraction"),
                                     ("physical_square_sum", "physical_square_fraction"),
                                     ("fixed_unit_square_sum", "fixed_unit_square_fraction"),
                                     ("common_robust_sum", "robust_objective_fraction")):
                result[denominator] = _safe_ratio(result[key], overall[key])
        groups[name] = result
    return groups


def verify_selected_mean(saved, audited_hashes):
    names = ("best_mean_state", "model_state", "best_state")
    hashes = {name: _state_hash({key: value for key, value in saved[name].items()
                               if key.startswith("mean_network.")}) for name in names}
    if len(set(hashes.values())) != 1 or hashes != audited_hashes:
        raise ValueError("Selected mean changed or does not match audited phase hashes")
    for key, value in saved["initial_state"].items():
        if key.startswith("covariance_network.") and not torch.equal(value, saved["best_mean_state"][key]):
            raise ValueError("Covariance network changed before the mean stage completed")
    return {"selected_mean_unchanged": True, "mean_hashes": hashes,
            "covariance_initial_tensors_preserved_during_mean_phase": True}


def compare_saved_strata(summary, saved):
    """Replay the audit's literal metrics, keeping robust and squared separate."""
    verified = 0
    for name, value in summary.items():
        other = saved[name]
        if value.get("empty"):
            if other != {"rows": 0, "empty": True}:
                raise ValueError("An empty stratum was represented as a zero-loss populated stratum")
            continue
        if value["rows"] != other["rows"]:
            raise ValueError("Stratified row counts differ")
        for first, second in (("raw_internal_rms", "raw_internal_rms"),
                              ("raw_internal_maximum", "raw_internal_maximum"),
                              ("fixed_unit_rms", "normalized_rms"),
                              ("fixed_unit_squared_rms", "normalized_squared_objective"),
                              ("common_pseudo_huber_objective", "normalized_objective")):
            np.testing.assert_allclose(value[first], other[second], rtol=2e-12, atol=2e-14)
        np.testing.assert_allclose(value["raw_physical_rms_by_output_or_pair"],
            other["raw_physical_rms_by_output_or_pair"], rtol=2e-12, atol=2e-14)
        verified += 1
    return verified


def evaluate_residuals(model, states, horizons, batch_size=128):
    chunks = {stage: [] for stage in STAGES}
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            state = torch.tensor(states[start:start+batch_size], dtype=torch.float64)
            remaining = torch.tensor(horizons[start:start+batch_size], dtype=torch.float64)
            mean, covariance = model._moment_jets(state, remaining)
            source = mean.face_derivatives@mean.face_derivatives.transpose(-1, -2)
            running = np.sqrt(model.context.n_particles)*(state@model.normalized_running_weights.T)
            running = torch.where(model._active(state)[:, None], running, 0.)
            residuals = (mean.time_derivative-mean.generator-running,
                         covariance.time_derivative-covariance.generator-source)
            for stage, value in zip(STAGES, residuals, strict=True):
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Every old-model validation residual must remain finite")
                chunks[stage].append(value.numpy())
    return {stage: np.concatenate(values) for stage, values in chunks.items()}


def _evidence_arrays(record):
    artifact = record["numeric_evidence"]
    path = Path(artifact["path"])
    if sha(path) != artifact["sha256"]:
        raise ValueError("Postfit numeric evidence changed")
    with np.load(path) as store:
        arrays = {key: store[key] for key in store.files}
    for key, value in arrays.items():
        expected = artifact["arrays"][key]
        if (list(value.shape) != expected["shape"] or hashlib.sha256(
                np.ascontiguousarray(value).tobytes()).hexdigest() != expected["sha256"]):
            raise ValueError("Postfit numeric evidence array changed")
    return arrays


def _metadata(bank, spec, horizon):
    # Reuse only the frozen definition of physical historical absorption and
    # strata, not its numerical summaries or normalization implementation.
    from analysis.audit_payoff_regimes_validation import stratum_masks, validation_metadata

    metadata = validation_metadata(bank)
    masks = stratum_masks(bank["states"], bank["horizons"], metadata, spec, horizon)
    return metadata, masks


def _legacy_arrays(root, item, states, horizons):
    from analysis import payoff_legacy_control as legacy

    model = legacy.load_model(item)
    values = evaluate_residuals(model, states, horizons)
    path = root/"audit/fit_review_numeric"/(item["name"]+".npz")
    if path.exists():
        with np.load(path) as previous:
            if set(previous.files) != set(values) or any(not np.array_equal(previous[key], value)
                                                        for key, value in values.items()):
                raise ValueError("Historical new-state residual replay is not reproducible")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
        temporary.replace(path)
    return values, {"path": str(path.resolve()), "sha256": sha(path)}


def _ratio_comparison(current, historical):
    return {name: {stage: {key: _safe_ratio(current[name][stage][key], historical[name][stage][key])
                            for key in ("raw_internal_rms", "raw_physical_rms", "fixed_unit_rms",
                                        "common_pseudo_huber_objective")}
                   for stage in STAGES if not current[name][stage].get("empty")}
            for name in current}


def run(root, *, compare_legacy=True):
    from analysis import run_payoff_boundary_regimes as runner

    torch.set_num_threads(1)
    started, cpu_started = time.monotonic(), time.process_time()
    root = Path(root).resolve()
    plan, _ = runner._assert_frozen(root)
    audit_path = root/"audit/postfit_validation_diagnostic.json"
    audit = _read(audit_path)
    lock = runner.verify_fit_lock(root)
    if (not audit.get("complete") or audit["identity"]["source_hash"] != source_hash()
            or audit["identity"]["fit_lock_sha256"] != sha(root/"payoffs/fit_lock.json")
            or not audit.get("no_selection_refit_or_future_outcome_access")
            or len(audit["records"]) != len(lock["models"])):
        raise ValueError("A complete current-source all-model pre-reference validation audit is required")
    controls_path = root/"comparison/legacy_controls.json"
    controls = _read(controls_path) if compare_legacy else {"models": []}
    identity = {"kind": KIND, "source_hash": source_hash(), "script_sha256": sha(__file__),
                "postfit_audit_sha256": sha(audit_path), "fit_lock_sha256": sha(root/"payoffs/fit_lock.json"),
                "data_lock_sha256": sha(root/"payoffs/data_lock.json"), "compare_legacy": compare_legacy,
                "legacy_controls_sha256": sha(controls_path) if compare_legacy else None,
                "common_scales": "v3 frozen train-only full-horizon heat units shared by both models",
                "future_outcomes_opened": False}
    output = root/"audit/fit_interpretation_review.json"
    previous = _read(output) if output.exists() else None
    if previous is not None and previous.get("identity") != identity:
        raise ValueError("Fit review identity changed; preserve earlier evidence")
    retained = {record["name"]: record for record in previous.get("records", [])} if previous else {}
    result = {"identity": identity, "complete": False, "created_utc": datetime.now(timezone.utc).isoformat(),
              "records": [], "interpretation": "Equation-fit diagnostics; not accuracy against native law.",
              "phase_scope": "Initial, selected mean plus literal initial covariance, and final selected heads; subset is deliberately stratified.",
              "weighted_objectives_not_cross_compared": True,
              "strata_overlap": "Profile, snapshot, horizon and minimum-mass partitions overlap; fractions must not be summed across partitions."}
    case_cache = {}
    for record in audit["records"]:
        if record["name"] in retained:
            prior = retained[record["name"]]
            if (sha(record["checkpoint_path"]) != prior["current_checkpoint_sha256"]
                    or sha(record["numeric_evidence"]["path"]) != prior["current_numeric_evidence_sha256"]):
                raise ValueError("Previously reviewed checkpoint or current residual evidence changed")
            if (prior.get("legacy_numeric_evidence") and sha(prior["legacy_numeric_evidence"]["path"])
                    != prior["legacy_numeric_evidence"]["sha256"]):
                raise ValueError("Previously saved legacy residual evidence changed")
            result["records"].append(prior)
            continue
        case = record["case"]
        if case not in case_cache:
            cfg = runner.configuration(case, plan)
            bank = runner.load_state_bank(root, case, "validation")
            _, fingerprint = _bank(bank, cfg, "validation")
            if fingerprint != audit["samples"][case]["full_validation_fingerprint"]:
                raise ValueError("New validation pool differs from the postfit audit")
            metadata, masks = _metadata(bank, cfg.spec, cfg.horizon)
            sample = audit["samples"][case]
            if sha(sample["sample_artifact"]["path"]) != sample["sample_artifact"]["sha256"]:
                raise ValueError("Saved diagnostic subset changed")
            indices = np.asarray(sample["sample_indices"], dtype=int)
            case_cache[case] = cfg, bank, metadata, masks, indices
        cfg, bank, metadata, masks, indices = case_cache[case]
        if sha(record["checkpoint_path"]) != record["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        saved = torch.load(record["checkpoint_path"], map_location="cpu", weights_only=True)
        mean_proof = verify_selected_mean(saved, record["selected_mean_hashes"])
        scaling = saved["provenance"]["loss_scaling"]
        if scaling["inverse_time_weighting"] or scaling["normalization_time_powers"] != [0]*4:
            raise ValueError("Expected fixed units without inverse-time weighting")
        arrays = _evidence_arrays(record)
        physical = {"mean": runner.payoff_bank(case, plan).scales/np.sqrt(cfg.n_particles),
                    "covariance": np.outer(runner.payoff_bank(case, plan).scales,
                                            runner.payoff_bank(case, plan).scales)}

        def summaries(values, selected_masks):
            per_stage = {stage: stratified_statistics(values[stage], scaling[stage+"_scale"],
                cfg.horizon, physical[stage], scaling["pseudo_huber_delta"], selected_masks) for stage in STAGES}
            return {name: {stage: per_stage[stage][name] for stage in STAGES} for name in selected_masks}

        current = summaries({stage: arrays[f"full_final_{stage}_residual"] for stage in STAGES}, masks)
        summary_checks = sum(compare_saved_strata({name: values[stage] for name, values in current.items()},
            {name: values[stage] for name, values in record["full_selected_stratified_residuals"].items()}) for stage in STAGES)
        phases = {phase: summaries({stage: arrays[f"subset_{phase}_{stage}_residual"] for stage in STAGES},
                                  {name: mask[indices] for name, mask in masks.items()}) for phase in PHASES}
        actual_active = {"physical_active": int(metadata["physical_active"].sum()),
                         "physical_absorbed": int((~metadata["physical_active"]).sum()),
                         "model_threshold_active": int(masks["model_threshold_active"].sum()),
                         "rounded_boundary_active_disagreement": int(np.sum(
                             metadata["physical_active"] != masks["model_threshold_active"]))}
        if actual_active != record["full_selected_active_counts"]:
            raise ValueError("Physical and threshold absorption counts differ from their audit")
        row = {"name": record["name"], "case": case, "seed": record["seed"],
               "current_checkpoint_sha256": record["checkpoint_sha256"],
               "current_numeric_evidence_sha256": record["numeric_evidence"]["sha256"],
               "selected_mean_verification": mean_proof, "active_counts": actual_active,
               "new_validation_rows": len(bank["states"]), "diagnostic_subset_rows": len(indices),
               "selected_steps": record["selected_steps"], "allocation": record["allocation"],
               "training_seconds": record["training_seconds"], "current_full_strata": current,
               "current_phase_strata": phases, "literal_saved_strata_verified": summary_checks,
               "current_payoff_consistency": record["full_selected_payoff_consistency"],
               "fixed_scales": {stage: scaling[stage+"_scale"] for stage in STAGES},
               "common_pseudo_huber_delta": scaling["pseudo_huber_delta"]}
        if compare_legacy:
            matches = [item for item in controls["models"] if item["case"] == case and item["seed"] == record["seed"]]
            if len(matches) != 1:
                raise ValueError("A unique immutable v2 optimizer-seed control is required")
            legacy_values, evidence = _legacy_arrays(root, matches[0], bank["states"], bank["horizons"])
            historical = summaries(legacy_values, masks)
            row.update(legacy_name=matches[0]["name"], legacy_checkpoint_sha256=matches[0]["checkpoint_sha256"],
                       legacy_numeric_evidence=evidence, legacy_on_new_validation_full_strata=historical,
                       current_over_legacy_residual_ratios=_ratio_comparison(current, historical))
        result["records"].append(row)
        result.update(review_wall_seconds=time.monotonic()-started, review_cpu_seconds=time.process_time()-cpu_started)
        _json(output, result)
        print(record["name"], "common-unit residual review complete", flush=True)
    if source_hash() != identity["source_hash"] or sha(audit_path) != identity["postfit_audit_sha256"]:
        raise ValueError("Numerical source or upstream validation audit changed during review")
    result.update(complete=len(result["records"]) == len(audit["records"]),
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  review_wall_seconds=time.monotonic()-started, review_cpu_seconds=time.process_time()-cpu_started,
                  no_fitting_selection_or_future_outcome_access=True)
    _json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--no-legacy", action="store_true")
    args = parser.parse_args()
    result = run(args.root, compare_legacy=not args.no_legacy)
    print(json.dumps({"complete": result["complete"], "models": len(result["records"]),
                      "output": str((args.root/"audit/fit_interpretation_review.json").resolve()),
                      "wall_seconds": result["review_wall_seconds"]}))


if __name__ == "__main__":
    main()
