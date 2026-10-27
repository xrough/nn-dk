"""Read-only validation diagnostic. Never opens reference/test outcomes."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from analysis import run_payoff_repair as runner
from nn_dk.data import source_hash
from nn_dk.interacting_training import _state_hash
from nn_dk.payoff_repair import RepairPayoffModel, load_repaired_payoff_model
from nn_dk.payoff_learning import PayoffMomentModel, load_payoff_model

ROOT = Path("runs/payoff-repair-2026-10-27").resolve()
torch.set_num_threads(1)
STARTED_WALL, STARTED_CPU = time.monotonic(), time.process_time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pool(model, states, horizons):
    chunks = {name: [] for name in ("mean_residual", "covariance_residual", "source", "covariance")}
    with torch.no_grad():
        for start in range(0, len(states), 128):
            m, t = states[start:start+128], horizons[start:start+128]
            mean = model._mean_jet(m, t) if hasattr(model, "_mean_jet") else model._moment_jets(m, t)[0]
            chunks["source"].append(mean.face_derivatives @ mean.face_derivatives.transpose(-1, -2))
            mr, cr = (model.mean_residual(m, t), model.covariance_residual(m, t)) if hasattr(
                model, "mean_residual") else model.residual(m, t)
            chunks["mean_residual"].append(mr)
            chunks["covariance_residual"].append(cr)
            chunks["covariance"].append(model.moment_values(m, t)[1])
    return {name: torch.cat(values).numpy() for name, values in chunks.items()}


def describe(value, time, scaling, stage, bank):
    powers = np.asarray(scaling["small_time_powers"])
    t = np.maximum(time/bank.horizon, scaling["scale_time_floor"])
    if stage == "mean":
        time_scale = t[:, None]**powers
        physical = value*bank.scales/np.sqrt(2048)
    else:
        powers = powers[:, None]+powers[None, :]
        time_scale = t[:, None, None]**powers
        physical = value*bank.scales[:, None]*bank.scales[None, :]
    normalized = bank.horizon*value/np.asarray(scaling[f"{stage}_scale"])/time_scale
    result = {"raw_internal_rms_by_output_or_pair": np.sqrt(np.mean(value**2, axis=0)).tolist(),
              "raw_physical_rms_by_output_or_pair": np.sqrt(np.mean(physical**2, axis=0)).tolist(),
              "physical_horizon_normalized_rms_by_output_or_pair":
                  np.sqrt(np.mean(normalized**2, axis=0)).tolist(),
              "normalized_global_rms": float(np.sqrt(np.mean(normalized**2))),
              "raw_global_rms": float(np.sqrt(np.mean(value**2))),
              "normalized_objective": float(np.mean(normalized**2))}
    integral = bank.integral_rows
    if stage == "mean":
        masks = {"terminal": ~integral, "integral": integral}
    else:
        masks = {"terminal_terminal": (~integral)[:, None] & (~integral)[None, :],
                 "terminal_integral": integral[:, None] ^ integral[None, :],
                 "integral_integral": integral[:, None] & integral[None, :]}
    result["groups"] = {key: {"raw_internal_rms": float(np.sqrt(np.mean(value[:, mask]**2))),
                              "raw_physical_rms": float(np.sqrt(np.mean(physical[:, mask]**2))),
                              "normalized_rms": float(np.sqrt(np.mean(normalized[:, mask]**2)))}
                        for key, mask in masks.items() if mask.any()}
    return result


freeze = json.loads((ROOT/"execution/source_freeze.json").read_text())
assert source_hash() == freeze["source_hash"]
design = runner._assert_frozen(ROOT/"payoffs")
assert design["source_hash"] == freeze["source_hash"]
_, arrays = runner._data_identity(ROOT/"payoffs")
validation = arrays["validation"]
states, horizons = (torch.tensor(validation[key], dtype=torch.float64) for key in ("states", "horizons"))
result = {"kind": "independent-postfit-validation-only-diagnostic-v1",
          "created_utc": datetime.now(timezone.utc).isoformat(), "source_hash": source_hash(),
          "source_freeze_sha256": sha(ROOT/"execution/source_freeze.json"),
          "script_sha256": sha(__file__), "future_test_outcomes_opened": False,
          "reference_h5_present_at_diagnostic": (ROOT/"reference/reference.h5").exists(),
          "validation_data_sha256": sha(ROOT/"payoffs/interacting_validation.h5"),
          "validation_rows": len(states), "validation_families": len(set(validation["family_ids"])),
          "interpretation": "Generator residuals and baseline comparisons are diagnostics, not independent accuracy results",
          "records": [], "controls": [], "pending": [], "pending_controls": []}
prior_path = ROOT/"audit/postfit_validation_diagnostic.json"
prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
if prior:
    assert prior["source_hash"] == source_hash()
    assert prior["validation_data_sha256"] == result["validation_data_sha256"]
    result["prior_audit"] = {"sha256": sha(prior_path), "created_utc": prior["created_utc"],
                             "audit_wall_seconds": prior.get("audit_wall_seconds"),
                             "audit_cpu_seconds": prior.get("audit_cpu_seconds")}
retained = {(r["bank"], r["seed"]): r for r in prior.get("records", [])}
retained_controls = {(r["bank"], r["variant"]): r for r in prior.get("controls", [])}
result["numerical_records_replayed"] = 0
result["numerical_records_reused_after_strict_restore_and_hash_check"] = 0
heat_reference = {}
common_scaling = {}

for bank_name in ("terminal-modes34", "terminal3-integral1"):
    for seed in range(3):
        path = ROOT/f"payoffs/checkpoints/{bank_name}/repaired/seed{seed}.pt"
        if not path.exists():
            result["pending"].append(str(path))
            continue
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if not saved["complete"]:
            result["pending"].append(str(path))
            continue
        final = load_repaired_payoff_model(path)
        common_scaling[bank_name] = saved["provenance"]["loss_scaling"]
        previous = retained.get((bank_name, seed))
        if previous is not None and previous["checkpoint_sha256"] == sha(path):
            result["records"].append(previous)
            result["numerical_records_reused_after_strict_restore_and_hash_check"] += 1
            print(bank_name, seed, "strictly restored and prior numerical replay reused", flush=True)
            continue
        means = {name: _state_hash({k: v for k, v in saved[name].items() if k.startswith("mean_network.")})
                 for name in ("best_mean_state", "model_state", "best_state")}
        assert len(set(means.values())) == 1
        cfg, bank = final.config, final.bank
        baseline = RepairPayoffModel(cfg.study.spec, cfg.study.physics, cfg.study.horizon,
                                    bank, cfg.study.width, cfg.quadrature_nodes)
        baseline.load_state_dict(saved["initial_state"])
        postmean = RepairPayoffModel(cfg.study.spec, cfg.study.physics, cfg.study.horizon,
                                    bank, cfg.study.width, cfg.quadrature_nodes)
        postmean.load_state_dict(saved["best_mean_state"])
        pools = {"initial": pool(baseline, states, horizons), "selected_mean_covariance_initial": pool(postmean, states, horizons),
                 "final_selected": pool(final, states, horizons)}
        scaling = saved["provenance"]["loss_scaling"]
        heat_reference[bank_name] = pools["initial"]
        common_scaling[bank_name] = scaling
        record = {"bank": bank_name, "seed": seed, "path": str(path), "checkpoint_sha256": sha(path),
                  "strict_restore": True, "source_hash": final.provenance["source_hash"],
                  "selected_mean_hashes": means, "selected_mean_unchanged": True,
                  "allocation": saved["steps_done"], "selected_steps": saved["best_step"],
                  "training_seconds": saved["training_seconds"], "validation": {}}
        for stage_name, values in pools.items():
            record["validation"][stage_name] = {stage: describe(values[f"{stage}_residual"],
                horizons.numpy(), scaling, stage, bank) for stage in ("mean", "covariance")}
        for stage in ("mean", "covariance"):
            # Independent recomputation must reproduce the exact held-out
            # criterion used for selecting that stage's saved best state.
            actual = record["validation"]["final_selected"][stage]["normalized_objective"]
            expected = saved["best_validation"][stage]["objective"]
            assert np.isclose(actual, expected, rtol=2e-12, atol=1e-16), (actual, expected)
        source_initial = pools["initial"]["source"].diagonal(axis1=-2, axis2=-1).mean(0)
        source_final = pools["final_selected"]["source"].diagonal(axis1=-2, axis2=-1).mean(0)
        covariance_initial = pools["initial"]["covariance"].diagonal(axis1=-2, axis2=-1).mean(0)
        covariance_final = pools["final_selected"]["covariance"].diagonal(axis1=-2, axis2=-1).mean(0)
        assert np.all(source_initial > 0) and np.all(covariance_initial > 0)
        record["collapse_diagnostic"] = {
            "mean_qv_source_initial_by_output": source_initial.tolist(),
            "mean_qv_source_final_by_output": source_final.tolist(),
            "source_final_over_heat_baseline_by_output": (source_final/source_initial).tolist(),
            "mean_internal_covariance_initial_by_output": covariance_initial.tolist(),
            "mean_internal_covariance_final_by_output": covariance_final.tolist(),
            "covariance_final_over_heat_baseline_by_output": (covariance_final/covariance_initial).tolist(),
            "mean_source_unchanged_by_covariance_stage_maximum": float(np.max(np.abs(
                pools["selected_mean_covariance_initial"]["source"]-pools["final_selected"]["source"]))),
            "minimum_covariance_eigenvalue": float(np.linalg.eigvalsh(pools["final_selected"]["covariance"]).min()),
            "scope": "Ratios to known heat anchors diagnose degeneration; they do not prove interacting accuracy"}
        assert record["collapse_diagnostic"]["mean_source_unchanged_by_covariance_stage_maximum"] == 0.
        result["records"].append(record)
        result["numerical_records_replayed"] += 1
        print(bank_name, seed, "verified", flush=True)

for bank_name in ("terminal-modes34", "terminal3-integral1"):
    for variant in ("v1-none", "v1-balanced"):
        path = ROOT/f"payoffs/checkpoints/{bank_name}/{variant}/seed0.pt"
        if not path.exists():
            result["pending_controls"].append(str(path))
            continue
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if not saved["complete"]:
            result["pending_controls"].append(str(path))
            continue
        final = load_payoff_model(path)
        previous = retained_controls.get((bank_name, variant))
        if previous is not None and previous["checkpoint_sha256"] == sha(path):
            result["controls"].append(previous)
            result["numerical_records_reused_after_strict_restore_and_hash_check"] += 1
            print(bank_name, variant, "strictly restored and prior numerical replay reused", flush=True)
            continue
        if bank_name not in heat_reference:
            heat = RepairPayoffModel(final.spec, final.context, final.horizon, final.bank, final.width)
            heat_reference[bank_name] = pool(heat, states, horizons)
        baseline = PayoffMomentModel(final.spec, final.context, final.horizon, final.bank, final.width)
        baseline.load_state_dict(saved["initial_state"])
        pools = {"initial": pool(baseline, states, horizons), "final_selected": pool(final, states, horizons)}
        record = {"bank": bank_name, "variant": variant, "seed": 0, "path": str(path),
                  "checkpoint_sha256": sha(path), "strict_restore": True,
                  "source_hash": final.provenance["source_hash"], "joint_training": True,
                  "common_physical_scaling": "repaired TRAIN-only heat standard deviations/pair products",
                  "validation": {}}
        for name, values in pools.items():
            record["validation"][name] = {stage: describe(values[f"{stage}_residual"], horizons.numpy(),
                common_scaling[bank_name], stage, final.bank) for stage in ("mean", "covariance")}
        reference = heat_reference[bank_name]
        source_reference = reference["source"].diagonal(axis1=-2, axis2=-1).mean(0)
        covariance_reference = reference["covariance"].diagonal(axis1=-2, axis2=-1).mean(0)
        source_final = pools["final_selected"]["source"].diagonal(axis1=-2, axis2=-1).mean(0)
        covariance_final = pools["final_selected"]["covariance"].diagonal(axis1=-2, axis2=-1).mean(0)
        record["collapse_diagnostic"] = {
            "mean_qv_source_final_by_output": source_final.tolist(),
            "source_final_over_heat_baseline_by_output": (source_final/source_reference).tolist(),
            "mean_internal_covariance_final_by_output": covariance_final.tolist(),
            "covariance_final_over_heat_baseline_by_output": (covariance_final/covariance_reference).tolist(),
            "minimum_covariance_eigenvalue": float(np.linalg.eigvalsh(pools["final_selected"]["covariance"]).min())}
        result["controls"].append(record)
        result["numerical_records_replayed"] += 1
        print(bank_name, variant, "control verified", flush=True)

result["complete"] = len(result["records"]) == 6 and len(result["controls"]) == 4 and not result["pending"] and not result["pending_controls"]
result["audit_wall_seconds"] = time.monotonic()-STARTED_WALL
result["audit_cpu_seconds"] = time.process_time()-STARTED_CPU
result["audit_timing_scope"] = "This successful refresh only; prior successful timed audit recorded separately; earlier development attempts were not instrumented"
assert source_hash() == freeze["source_hash"]
target = ROOT/"audit/postfit_validation_diagnostic.json"
target.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps({"records": len(result["records"]), "complete": result["complete"], "sha256": sha(target)}))
