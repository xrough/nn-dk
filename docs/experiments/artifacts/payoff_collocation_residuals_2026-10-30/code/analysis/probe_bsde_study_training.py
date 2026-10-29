#!/usr/bin/env python3
"""Inspect completed study fits without opening any held-out outcomes.

The probe reads configuration/provenance, committed checkpoint metadata and
training histories only.  It neither predicts at test states nor opens final
evaluation artifacts, forward trajectories, or reference banks.  Partial runs
are supported: active fits without complete atomic metadata are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from nn_dk.bsde_model import BSDEMomentModel
from nn_dk.bsde_structured import StructuredBSDEMomentModel
from nn_dk.bsde_study import WORKFLOW, _design
from nn_dk.bsde_study_config import BSDEStudyConfig
from nn_dk.data import json_ready, source_hash, write_json
from nn_dk.types import stable_hash


def _read_json(path):
    contents = path.read_bytes()
    return json.loads(contents), hashlib.sha256(contents).hexdigest()


def _normalization_hash(state):
    digest = hashlib.sha256()
    for key in ("feature_mean", "feature_scale"):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _history_summary(saved, metadata, base):
    history = saved["history"]
    epoch = saved["epoch"]
    if (not history or epoch != len(history) or metadata.get("epochs") != epoch
            or [row["epoch"] for row in history] != list(range(1, epoch+1))
            or metadata.get("history") != history):
        raise ValueError("Checkpoint and metadata training histories disagree")
    if not all(np.isfinite(row[key]) for row in history
               for key in ("training_loss", "validation_loss", "epoch_seconds")):
        raise ValueError("Nonfinite training history")
    best = min(history, key=lambda row: row["validation_loss"])
    last = history[-1]
    if (saved["best_loss"] != best["validation_loss"]
            or metadata.get("best_validation_loss") != saved["best_loss"]):
        raise ValueError("Selected validation loss does not match history")
    cap, patience = base.training["epochs"], base.training["patience"]
    if epoch >= cap:
        stopping = "epoch-cap"
    elif saved["patience_count"] >= patience:
        stopping = "validation-patience"
    else:
        raise ValueError("Completed checkpoint has no declared stopping condition")
    return {"epochs": epoch, "maximum_epochs": cap, "selected_epoch": best["epoch"],
            "stopping_reason": stopping, "reached_epoch_cap": epoch >= cap,
            "patience_count": saved["patience_count"], "patience_limit": patience,
            "selected_validation_loss": best["validation_loss"],
            "selected_epoch_training_loss": best["training_loss"],
            "selected_epoch_validation_minus_training": best["validation_loss"] - best["training_loss"],
            "last_training_loss": last["training_loss"],
            "last_validation_loss": last["validation_loss"],
            "last_validation_minus_training": last["validation_loss"] - last["training_loss"],
            "last_ten_validation_loss_change": last["validation_loss"] - history[max(0, epoch-10)]["validation_loss"],
            "training_seconds": metadata["training_seconds"],
            "gap_interpretation": "Training loss averages changing optimizer iterates within each epoch; validation uses the final iterate. This gap alone does not measure conditional generalization.",
            "validation_terminal_residuals": metadata.get("validation_residuals", {})}


def _structured_parameters(model, state):
    model.load_state_dict(state)
    return {"rates": model.value_network.rates.detach().cpu().numpy(),
            "asymptotic_variance_scales": model.value_network.scales.detach().cpu().numpy()}


def _row(directory, key, study, fingerprint, locked):
    problem, architecture, anchors, children, seed = key
    base = study.base_config
    path = (directory / "checkpoints" / problem / architecture
            / f"a{anchors}_c{children}" / f"seed_{seed}.pt")
    metadata_path = path.with_suffix(".json")
    factors = {"problem": problem, "architecture": architecture,
               "train_anchors": anchors, "train_continuations": children,
               "optimizer_seed": seed}
    if not path.exists() or not metadata_path.exists():
        if locked:
            raise ValueError("A locked checkpoint or its metadata is missing")
        return None, {**factors, "reason": "checkpoint-or-metadata-not-yet-committed"}
    metadata, metadata_sha = _read_json(metadata_path)
    if not metadata.get("complete"):
        if locked:
            raise ValueError("A locked checkpoint has incomplete metadata")
        return None, {**factors, "reason": "fit-metadata-incomplete"}
    # Read one atomically committed checkpoint snapshot, without re-reading an
    # active path later.  weights_only avoids loading executable pickle classes.
    contents = path.read_bytes()
    checkpoint_sha = hashlib.sha256(contents).hexdigest()
    saved = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if not saved.get("complete"):
        if locked:
            raise ValueError("A locked checkpoint is incomplete")
        return None, {**factors, "reason": "checkpoint-fit-incomplete"}
    provenance = saved["provenance"]
    identity = {name: value for name, value in provenance.items() if name != "identity_hash"}
    if (provenance.get("source_hash") != fingerprint
            or provenance.get("config_hash") != base.config_hash
            or provenance.get("config") != base.to_dict()
            or provenance.get("problem") != problem or provenance.get("seed") != seed
            or provenance.get("target_id") != base.target_id(problem)
            or stable_hash(identity) != provenance.get("identity_hash")
            or metadata.get("identity_hash") != provenance.get("identity_hash")
            or metadata.get("study_config_hash") != study.config_hash
            or metadata.get("architecture") != architecture
            or metadata.get("allocation") != [anchors, children]
            or metadata.get("training_anchors") != anchors
            or metadata.get("training_paths") != study.training_paths
            or metadata.get("validation_anchors") != study.validation[0]
            or metadata.get("validation_paths") != study.validation[0]*study.validation[1]
            or metadata.get("training_view_fingerprint") != provenance.get("training_fingerprint")
            or metadata.get("validation_fingerprint") != provenance.get("validation_fingerprint")):
        raise ValueError(f"Study/checkpoint/data provenance mismatch: {path}")
    if locked:
        if (locked.get("checkpoint_sha256") != checkpoint_sha
                or locked.get("training_fingerprint") != provenance["training_fingerprint"]
                or locked.get("validation_fingerprint") != provenance["validation_fingerprint"]):
            raise ValueError(f"Checkpoint bytes or data fingerprint differ from fit lock: {path}")
    factory = StructuredBSDEMomentModel if architecture == "structured" else BSDEMomentModel
    # Construction validates the complete state layout. No model evaluation or
    # forward simulation occurs, and the caller's Torch RNG is restored.
    with torch.random.fork_rng():
        model = factory(base, problem)
    if (saved.get("kind") != model.kind or provenance.get("kind") != model.kind
            or metadata.get("kind") != model.kind):
        raise ValueError("Checkpoint kind does not match the preregistered architecture")
    model.load_state_dict(saved["best_state"])
    counts = {"value": sum(parameter.numel() for parameter in model.value_network.parameters()),
              "integrand": sum(parameter.numel() for parameter in model.integrand_network.parameters())}
    counts["total"] = counts["value"] + counts["integrand"]
    row = {**factors, "checkpoint_path": str(path.relative_to(directory)),
           "checkpoint_sha256": checkpoint_sha, "metadata_sha256": metadata_sha,
           "checkpoint_sha_verified_against_fit_lock": bool(locked),
           "kind": model.kind, "training_fingerprint": provenance["training_fingerprint"],
           "validation_fingerprint": provenance["validation_fingerprint"],
           "training_normalization_sha256": _normalization_hash(saved["best_state"]),
           "parameter_counts": counts, "training": _history_summary(saved, metadata, base)}
    if architecture == "structured":
        row["structured_selected_parameters"] = _structured_parameters(model, saved["best_state"])
        row["structured_final_parameters"] = _structured_parameters(model, saved["model_state"])
        transition = 1 - base.dt * base.mode_rates[::2]
        if np.any(transition <= 0):
            raise ValueError("The exponential OU diagnostic requires positive Euler mode factors")
        reference = {"effective_decay_rates": -np.log(transition)/base.dt,
                     "asymptotic_variance_scales": 2*base.mode_rates[::2]*base.dt/(1-transition**2)}
        selected = row["structured_selected_parameters"]
        row["known_linear_euler_coefficient_diagnostic"] = {
            "reference": reference,
            "selected_rate_difference": selected["rates"] - reference["effective_decay_rates"],
            "selected_variance_scale_difference": selected["asymptotic_variance_scales"] - reference["asymptotic_variance_scales"],
            "interpretation": ("Exact representable OU family; discrepancy diagnoses fitted parameters, not test error."
                               if problem == "ou" else
                               "Unstopped linear heat backbone only. Stopped DK may need its full-state residual; that residual can also compensate backbone rates, so those rates are not identifiable coefficient estimates or an oracle for the stopped target.")}
    return row, None


def probe(run_directory, *, require_complete=False):
    directory = Path(run_directory).resolve()
    study = BSDEStudyConfig.load(directory / "config.json")
    base = study.base_config
    manifest, manifest_sha = _read_json(directory / "manifest.json")
    preregistration, preregistration_sha = _read_json(directory / "preregistration.json")
    base_content, base_sha = _read_json(directory / "base_config.json")
    fingerprint = source_hash()
    if (manifest.get("workflow") != WORKFLOW
            or manifest.get("source_hash") != fingerprint
            or manifest.get("config_hash") != study.config_hash
            or manifest.get("config") != study.to_dict()
            or base_content != base.to_dict()
            or preregistration != _design(study)
            or stable_hash(preregistration) != manifest.get("preregistration_hash")):
        raise ValueError("Training probe requires matching frozen source, study and preregistration")
    torch.set_num_threads(base.training["threads"])
    inventory = {tuple(item["key"]): item for item in manifest.get("fit_lock", [])}
    expected = [(problem, architecture, anchors, children, seed)
                for problem in ("ou", "dk") for anchors, children in study.allocations
                for architecture in study.architectures for seed in base.optimizer_seeds]
    if inventory and set(inventory) != set(expected):
        raise ValueError("Study fit-lock allocation is incomplete or incompatible")
    rows, skipped = [], []
    training_views, validation_views, normalizations = {}, {}, {}
    for key in expected:
        row, pending = _row(directory, key, study, fingerprint, inventory.get(key))
        if pending:
            skipped.append(pending)
            continue
        view = (row["problem"], row["train_anchors"], row["train_continuations"])
        if (training_views.setdefault(view, row["training_fingerprint"]) != row["training_fingerprint"]
                or validation_views.setdefault(row["problem"], row["validation_fingerprint"]) != row["validation_fingerprint"]
                or normalizations.setdefault(view, row["training_normalization_sha256"]) != row["training_normalization_sha256"]):
            raise ValueError("Paired architectures/seeds do not share training, validation or normalization")
        rows.append(row)
    if require_complete and skipped:
        raise ValueError(f"Training allocation incomplete: {len(skipped)} fits remain")
    payload = {"diagnostic": "Preregistered BSDE study training-only checkpoint probe",
               "scope": "No held-out outcomes, test states, evaluation JSON, reference banks or forward trajectories are read. No fitting or model prediction occurs.",
               "snapshot_unix": time.time(), "fit_lock_present": bool(inventory),
               "source_hash": fingerprint, "study_config_hash": study.config_hash,
               "base_config_hash": base.config_hash,
               "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "manifest_sha256": manifest_sha, "preregistration_sha256": preregistration_sha,
               "config_file_sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
               "base_config_file_sha256": base_sha,
               "expected_fits": len(expected), "completed_fits_read": len(rows),
               "incomplete_fits_remaining": len(skipped), "allocation_complete": not skipped,
               "skipped_fits": skipped, "fits": rows,
               "interpretation": "Learned backbone coefficients and loss histories distinguish potential optimization limitations from structural restrictions. They do not measure held-out accuracy or prove either explanation. Epoch-cap completion does not establish optimization convergence."}
    output = directory / "report" / "training_probe.json"
    write_json(output, payload)
    return output, json_ready(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    output, payload = probe(arguments.run_directory, require_complete=arguments.require_complete)
    print(output)
    print(json.dumps({name: payload[name] for name in
                      ("completed_fits_read", "incomplete_fits_remaining", "allocation_complete", "fit_lock_present")}))
    for row in payload["fits"]:
        if row["architecture"] == "structured":
            print(json.dumps({"problem": row["problem"], "anchors": row["train_anchors"],
                              "seed": row["optimizer_seed"],
                              "selected_parameters": row["structured_selected_parameters"],
                              "stopping_reason": row["training"]["stopping_reason"]}))


if __name__ == "__main__":
    main()
