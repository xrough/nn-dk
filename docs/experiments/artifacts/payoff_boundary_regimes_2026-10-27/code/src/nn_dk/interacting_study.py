"""Source-frozen, fit-locked experiments for nonlinear generator moments."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .data import source_hash, write_json
from .experiments import BudgetExhausted
from .native import (
    boundary_step,
    initialize_masses,
    native_coefficients,
    physical_observable_weights,
)
from .path_numerics import path_coefficients
from .types import Context, QueryBank

WORKFLOW = "full-state-interacting-generator-study-v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _budget(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExhausted("Interacting study checkpointed; resume the identical configuration")


def prepare_interacting_run(directory, config):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    design = {"workflow": WORKFLOW, "config_hash": config.config_hash,
              "target_id": config.target_id, "cases": list(config.cases),
              "optimizer_seeds": list(config.optimizer_seeds),
              "training": "known-generator residual at current full states; no future payoff labels",
              "trace": "exact all physical face directions; no auxiliary noise",
              "reference": "independent stopped Euler with nested Brownian refinement",
              "continuous_reference_bias": "unresolved; finite differences are diagnostics, not bounds",
              "covariance_scope": "same-time full 4x4; no cross-time claim",
              "ensemble_initialization": "independent multinomial Monte Carlo plus total covariance",
              "final_references_locked_until_all_fits_complete": True,
              "accuracy_target": config.tolerance, "uncertainty_budget": config.uncertainty_budget,
              "seeds_pooled": False}
    path = directory/"manifest.json"
    if path.exists():
        saved = _read(path)
        if (saved.get("workflow") != WORKFLOW or saved.get("source_hash") != source_hash()
                or saved.get("config_hash") != config.config_hash
                or _read(directory/"config.json") != config.to_dict()
                or _read(directory/"preregistration.json") != design):
            raise ValueError("Interacting source/configuration/design changed; use a fresh run directory")
        if saved.get("artifact_lock") is not None:
            current = {name: _sha(directory/name) for name in saved["artifact_lock"]}
            if current != saved["artifact_lock"]:
                raise ValueError("A completed interacting study artifact changed")
        return saved
    saved = {"workflow": WORKFLOW, "source_hash": source_hash(),
             "config_hash": config.config_hash, "target_id": config.target_id,
             "expected_models": config.expected_models, "status": "prepared",
             "created_unix": time.time(), "elapsed_seconds": 0.,
             "python": platform.python_version(), "device": "cpu", "dtype": "float64"}
    write_json(directory/"config.json", config.to_dict())
    write_json(directory/"preregistration.json", design)
    write_json(path, saved)
    return saved


def calibrate_interacting(directory, cfg):
    """Independent coefficient/Fourier/stopping checks before any fitting."""
    path = Path(directory)/"calibration.json"
    if path.exists():
        result = _read(path)
        if (result["source_hash"] != source_hash() or result["config_hash"] != cfg.config_hash
                or not result.get("passed")):
            raise ValueError("Calibration provenance changed")
        return result
    spec = cfg.spec
    states = initialize_masses(cfg.context("nonuniform"), spec, 32, cfg.seed+73)
    exact = native_coefficients(states, cfg.physics, spec)
    drift, amplitude = path_coefficients(torch.tensor(states), cfg.physics, spec)
    weights = physical_observable_weights(spec, QueryBank())
    free = Context(diffusion=cfg.diffusion, n_particles=cfg.n_particles)
    free_drift = native_coefficients(states, free, spec).drift
    rates = np.repeat(4*cfg.diffusion*np.sin(np.arange(1, 3)*spec.h/2)**2/spec.h**2, 2)
    fourier_error = float(np.max(np.abs(free_drift@weights.T+(states@weights.T)*rates)))
    uniform = np.full(spec.n_cells, 1/spec.n_cells)
    proposal = uniform.copy()
    proposal[0] = -uniform[0]
    proposal[1] += 2*uniform[0]
    boundary, crossed, stop = boundary_step(uniform, proposal, 0., 1., spec)
    expected_stop = (uniform[0]-spec.mass_stop)/(2*uniform[0])
    errors = {"drift_numpy_torch": float(np.max(np.abs(exact.drift-drift.numpy()))),
              "amplitude_numpy_torch": float(np.max(np.abs(exact.face_amplitudes-amplitude.numpy()))),
              "mass_drift": float(np.max(np.abs(exact.drift.sum(-1)))),
              "mass_covariance": float(np.max(np.abs(exact.covariance.sum(-1)))),
              "free_fourier_eigenvalue": fourier_error,
              "boundary_time": float(abs(stop-expected_stop)),
              "boundary_mass": float(abs(boundary[0]-spec.mass_stop)),
              "boundary_conservation": float(abs(boundary.sum()-1))}
    result = {"source_hash": source_hash(), "config_hash": cfg.config_hash,
              "passed": bool(crossed and max(errors.values()) < 1e-10), "errors": errors,
              "scope": "software coefficient/observation/stopping calibration; no continuous weak-error bound"}
    write_json(path, result)
    if not result["passed"]:
        raise ValueError("Interacting numerical calibration failed")
    return result


def _fit_inventory(directory, cfg):
    from .interacting_training import load_interacting_model

    result = []
    for seed in cfg.optimizer_seeds:
        checkpoint = Path(directory)/"checkpoints"/f"moments_seed{seed}.pt"
        metadata = checkpoint.with_suffix(".json")
        if not checkpoint.exists() or not metadata.exists():
            raise ValueError("Every declared fit must finish before final references unlock")
        model = load_interacting_model(checkpoint)
        info = _read(metadata)
        if (not info.get("complete") or info.get("seed") != seed
                or model.provenance.get("seed") != seed
                or info.get("config_hash") != cfg.config_hash
                or info.get("source_hash") != source_hash()
                or info.get("target_id") != cfg.target_id):
            raise ValueError("Interacting fit inventory provenance mismatch")
        result.append({"seed": seed, "checkpoint_path": str(checkpoint.relative_to(directory)),
                       "checkpoint_sha256": _sha(checkpoint),
                       "metadata_path": str(metadata.relative_to(directory)),
                       "metadata_sha256": _sha(metadata)})
    return result


def _lock_fits(directory, cfg):
    path = Path(directory)/"manifest.json"
    saved = _read(path)
    inventory = _fit_inventory(directory, cfg)
    lock = {"expected_models": cfg.expected_models, "models": inventory,
            "source_hash": source_hash(), "config_hash": cfg.config_hash, "target_id": cfg.target_id}
    if saved.get("fit_lock") is not None and saved["fit_lock"] != lock:
        raise ValueError("An interacting fit changed after final-reference unlock")
    if saved.get("fit_lock") is None:
        saved["fit_lock"] = lock
        saved["test_unlocked_unix"] = time.time()
        saved["status"] = "fits-locked"
        write_json(path, saved)
    return lock


def report_interacting(directory, cfg, records):
    """Present empirical accuracy and uncertainty without a continuous certificate."""
    from .interacting_evaluation import evaluation_signature
    from .types import stable_hash

    directory = Path(directory)
    signature = evaluation_signature(directory, cfg)
    validated = []
    expected = {(case, scope, anchor) for case in cfg.cases
                for scope, anchors in (("conditional", range(cfg.conditional_anchors)), ("ensemble", (None,)))
                for anchor in anchors}
    for seed in cfg.optimizer_seeds:
        packet = _read(directory/f"interacting_evaluation_seed{seed}.json")
        body = {key: value for key, value in packet.items() if key != "content_hash"}
        if (packet.get("source_hash") != source_hash() or packet.get("config_hash") != cfg.config_hash
                or packet.get("target_id") != cfg.target_id or packet.get("seed") != seed
                or packet.get("evaluation_signature") != signature
                or packet.get("content_hash") != stable_hash(body)):
            raise ValueError("Interacting evaluation provenance or content changed before reporting")
        prediction = Path(packet["prediction_path"])
        if not prediction.is_absolute():
            prediction = directory/prediction
        if not prediction.is_file() or _sha(prediction) != packet.get("prediction_sha256"):
            raise ValueError("Interacting prediction content changed before reporting")
        identities = [(r["case"], r["kind"], r.get("anchor_id")) for r in packet["records"]]
        if (len(identities) != len(expected) or set(identities) != expected
                or any(r["seed"] != seed for r in packet["records"])):
            raise ValueError("Interacting report lacks a declared profile, seed or anchor")
        validated.extend(packet["records"])
    if records != validated:
        raise ValueError("Interacting flattened evaluations differ from their signed seed packets")
    metadata = [_read(directory/"checkpoints"/f"moments_seed{s}.json") for s in cfg.optimizer_seeds]
    lines = ["# Interacting full-state neural statistics experiment", "",
             f"Target: stopped native DK, G={cfg.n_cells}, N={cfg.n_particles}, "
             f"kappa={cfg.kappa}, a2={cfg.a2}, T={cfg.horizon}.", "",
             "One full-state conditional-moment network spans both initial laws. "
             "It learns from known-generator residuals, with no future payoff labels. "
             "Ensemble predictions use independent initial-state Monte Carlo and total covariance.", "",
             "Reference sampling error, initial-integration error and coupled time-step differences "
             "are reported separately. Continuous stopping/reference bias remains unresolved. "
             "This run cannot certify continuous-DK convergence or a 5% accuracy claim.", "",
             "| Seed | Best validation objective | Fit seconds |", "|---|---:|---:|"]
    for info in metadata:
        objective = info.get("best_validation", {}).get("objective", float("nan"))
        lines.append(f"| {info['seed']} | {objective:.6g} | {info['training_seconds']:.3f} |")
    rows = []
    lines.extend(["", "Errors use fixed uniform-particle fluctuation units. The values below are "
                  "maximum point estimates over all query times and moment entries, not certified bounds.", "",
                  "| Seed | Profile | Scope | Mean error | Covariance error | Reference sampling halfwidth | Initial MC halfwidth |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for record in records:
        comparison = record["comparison"]
        uncertainty = comparison["uncertainty"]
        row = {"seed": record["seed"], "case": record["case"], "scope": record["kind"],
               "anchor_id": record.get("anchor_id"),
               "mean_max_error": comparison["mean_error"]["maximum_absolute_point"],
               "covariance_max_error": comparison["covariance_error"]["maximum_absolute_point"],
               "reference_sampling_halfwidth": uncertainty["maximum_reference_sampling_halfwidth"],
               "initial_integration_halfwidth": uncertainty["maximum_initial_integration_halfwidth"],
               "prediction_seconds": record["prediction_seconds"],
               "stopped_paths": record["stopping"]["stopped_paths"],
               "reference_paths": record["stopping"]["paths"],
               "continuous_accuracy_certified": False}
        rows.append(row)
        scope = row["scope"]+(f" {row['anchor_id']}" if row["anchor_id"] is not None else "")
        lines.append(f"| {row['seed']} | {row['case']} | {scope} | "
                     f"{100*row['mean_max_error']:.3f}% | {100*row['covariance_max_error']:.3f}% | "
                     f"{100*row['reference_sampling_halfwidth']:.3f}% | {100*row['initial_integration_halfwidth']:.3f}% |")
    if rows:
        with (directory/"convergence_table.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines.extend(["", "Full per-case, per-anchor mean/covariance errors and uncertainty diagnostics "
                  "are saved in `evaluations.json`; optimizer seeds are never pooled.", ""])
    (directory/"report.md").write_text("\n".join(lines))
    # Residual learning curves are meaningful even when MC error is unresolved.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for info in metadata:
        history = info.get("history", [])
        points = [(v["step"], v["validation"]["objective"]) for v in history
                  if isinstance(v.get("validation"), dict)]
        if points:
            ax.semilogy(*np.asarray(points).T, label=f"seed {info['seed']}")
    ax.set(xlabel="Optimizer step", ylabel="Independent-state residual objective")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory/"learning_curves.png", dpi=160)
    plt.close(fig)
    return {"records": len(records), "continuous_accuracy_certified": False,
            "report": str(directory/"report.md"), "target_id": cfg.target_id}


def run_interacting_study(directory, cfg, deadline=None, stage="all"):
    from .interacting_data import (
        ensure_interacting_bank,
        ensure_interacting_reference,
        load_interacting_bank,
    )
    from .interacting_evaluation import evaluate_interacting_model
    from .interacting_training import fit_interacting, load_interacting_model

    if stage not in ("all", "calibrate", "generate", "fit", "evaluate", "report"):
        raise ValueError("Unknown interacting study stage")
    directory = Path(directory).resolve()
    manifest = prepare_interacting_run(directory, cfg)
    if manifest.get("status") == "complete":
        return _read(directory/"summary.json")
    started = time.monotonic()
    try:
        _budget(deadline)
        calibrate_interacting(directory, cfg)
        if stage == "calibrate":
            return {"status": "calibrated"}
        if stage in ("all", "generate", "fit"):
            for role in ("train", "validation"):
                packet = ensure_interacting_bank(directory, cfg, role, deadline=deadline)
                if not packet["complete"]:
                    raise BudgetExhausted("Interacting state generation checkpointed; resume this configuration")
            if stage == "generate":
                return {"status": "generated"}
            train, validation = [load_interacting_bank(directory, cfg, r) for r in ("train", "validation")]
            for seed in cfg.optimizer_seeds:
                _budget(deadline)
                path = directory/"checkpoints"/f"moments_seed{seed}.pt"
                info = fit_interacting(cfg, seed, path, train, validation, deadline=deadline)
                write_json(path.with_suffix(".json"), info)
                if not info["complete"]:
                    raise BudgetExhausted("Interacting coefficient fit checkpointed; resume this configuration")
            _lock_fits(directory, cfg)
            if stage == "fit":
                return {"status": "fits-locked"}
        _lock_fits(directory, cfg)
        if stage in ("all", "evaluate"):
            _budget(deadline)
            packet = ensure_interacting_reference(directory, cfg, deadline=deadline)
            if not packet["complete"]:
                raise BudgetExhausted("Interacting reference generation checkpointed; resume this configuration")
            records = []
            for seed in cfg.optimizer_seeds:
                model = load_interacting_model(directory/"checkpoints"/f"moments_seed{seed}.pt")
                packet = evaluate_interacting_model(directory, cfg, model, seed, deadline=deadline)
                records.extend(packet["records"])
            write_json(directory/"evaluations.json", records)
            if stage == "evaluate":
                return {"status": "evaluated", "records": len(records)}
        records = _read(directory/"evaluations.json")
        result = report_interacting(directory, cfg, records)
        write_json(directory/"summary.json", result)
        manifest = _read(directory/"manifest.json")
        manifest["status"] = "complete"
        # The HDF5 row digests are also checked by their loaders. Lock every
        # saved numerical and prediction artifact after reporting completes.
        manifest["artifact_lock"] = {str(p.relative_to(directory)): _sha(p) for p in sorted(directory.rglob("*"))
                                     if p.is_file() and p.name != "manifest.json"}
        write_json(directory/"manifest.json", manifest)
        return result
    finally:
        manifest = _read(directory/"manifest.json")
        manifest["elapsed_seconds"] = manifest.get("elapsed_seconds", 0.)+time.monotonic()-started
        write_json(directory/"manifest.json", manifest)
