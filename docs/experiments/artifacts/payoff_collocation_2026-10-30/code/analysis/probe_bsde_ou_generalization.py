#!/usr/bin/env python3
"""Posthoc OU diagnostics on the unchanged, prespecified analytic test bank.

This probe never refits a network or changes a test allocation. It describes
errors on the fixed test states, without certification outside that bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nn_dk.bsde_config import BSDEConfig
from nn_dk.bsde_data import ou_exact_moments
from nn_dk.bsde_training import load_bsde_model
from nn_dk.data import json_ready, source_hash, write_json
from nn_dk.types import seeded_rng, stable_hash

COMPONENTS = ("cos1", "sin1", "cos2", "sin2")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _largest(error, initial, horizons):
    index = np.unravel_index(np.argmax(np.abs(error)), error.shape)
    anchor = int(index[0])
    norms = np.linalg.norm(initial, axis=1)
    result = {"array_index": list(map(int, index)), "anchor": anchor,
              "horizon": float(horizons[index[1]]),
              "component": COMPONENTS[index[2]], "signed_error": float(error[index]),
              "initial": initial[anchor], "initial_norm": float(norms[anchor]),
              "initial_norm_rank_descending": int(np.sum(norms > norms[anchor]) + 1)}
    if error.ndim == 4:
        result["second_component"] = COMPONENTS[index[3]]
    return result


def _error_summary(error, initial, horizons, tolerance):
    per_anchor = np.abs(error).reshape(len(initial), -1)
    return {"rmse": float(np.sqrt(np.mean(error**2))),
            "maximum": _largest(error, initial, horizons),
            "entries_above_tolerance": int(np.sum(np.abs(error) > tolerance)),
            "entries": int(error.size),
            "anchors_with_entry_above_tolerance": int(np.sum(per_anchor.max(-1) > tolerance)),
            "anchors": len(initial)}


def _irreducible_euler_loss(config):
    """Population second-chaos loss for exact values and unrestricted linear-dW Z.

    For each terminal Fourier component, q_i,j is its remaining noise weight.
    Predictable first-order integrands cannot represent centered quadratic
    increments. Diagonal residual variances are 2 dt^2 sum_j q_i,j^4; svec
    off-diagonal residual variances are 2 dt^2 sum_j q_i,j^2 q_k,j^2.
    """
    rates, dt = config.mode_rates, config.dt
    values = []
    for horizon in config.query_times:
        steps = round(horizon / dt)
        weights = np.sqrt(2 * rates)[None] * (1 - dt * rates)[None]**np.arange(steps)[:, None]
        squares = weights**2
        diagonal = 2 * dt**2 * np.sum(squares**2, axis=0)
        off_diagonal = [2 * dt**2 * np.sum(squares[:, i] * squares[:, j])
                        for i in range(4) for j in range(i)]
        values.append({"horizon": horizon,
                       "second_moment_block_population_floor": float(
                           (diagonal.sum() + np.sum(off_diagonal)) / 10),
                       "diagonal_payoff_residual_variance": diagonal})
    return {"scope": "Population Euler OU objective, exact values, unrestricted predictable linear-in-dW integrands; not a finite-sample acceptance gate",
            "horizons": values, "mean_block_floor": 0.,
            "mean_horizon_objective_floor": float(np.mean(
                [row["second_moment_block_population_floor"] for row in values]))}


def probe(run_directory):
    directory = Path(run_directory).resolve()
    config_path, manifest_path = directory / "config.json", directory / "manifest.json"
    config = BSDEConfig.load(config_path)
    manifest = json.loads(manifest_path.read_text())
    current_source = source_hash()
    if manifest.get("config_hash") != config.config_hash or manifest.get("source_hash") != current_source:
        raise ValueError("Probe source/configuration differs from the immutable experiment")
    evaluations = {}
    for path in sorted((directory / "evaluation").glob("ou_seed*.json")):
        record = json.loads(path.read_text())
        if record.get("problem") != "ou":
            continue
        seed = record["seed"]
        if seed in evaluations:
            raise ValueError("Duplicate OU evaluation records")
        evaluations[seed] = (path, record)
    if set(evaluations) != set(config.optimizer_seeds):
        raise ValueError("Complete every configured OU evaluation before probing the test bank")
    # Validate every allocated checkpoint and record BEFORE loading any network.
    inputs = []
    for seed in config.optimizer_seeds:
        checkpoint = directory / "checkpoints" / "ou" / f"seed_{seed}.pt"
        metadata_path = checkpoint.with_suffix(".json")
        evaluation_path, evaluation = evaluations[seed]
        metadata = json.loads(metadata_path.read_text())
        if not metadata.get("complete") or not evaluation.get("complete"):
            raise ValueError("Incomplete OU checkpoint or evaluation")
        if evaluation.get("config_hash") != config.config_hash or evaluation.get("source_hash") != current_source:
            raise ValueError("OU evaluation source/configuration mismatch")
        checkpoint_sha = _sha(checkpoint)
        if evaluation.get("model", {}).get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("OU evaluation checkpoint SHA mismatch")
        inputs.append((seed, checkpoint, checkpoint_sha, metadata, metadata_path,
                       evaluation, evaluation_path))
    torch.set_num_threads(config.training.get("threads", 1))
    initial = seeded_rng(config.seed, "bsde-analytic-test", "ou").normal(
        size=(config.ou_test_anchors, 4))
    euler_mean, euler_covariance = ou_exact_moments(initial, config.query_times, config, dt=config.dt)
    continuous_mean, continuous_covariance = ou_exact_moments(initial, config.query_times, config)
    rows = []
    for seed, checkpoint, checkpoint_sha, metadata, metadata_path, evaluation, evaluation_path in inputs:
        model = load_bsde_model(checkpoint)
        if model.config.config_hash != config.config_hash or model.problem != "ou":
            raise ValueError("Checkpoint is not this experiment's OU model")
        prediction = model.predict_conditional(initial, config.query_times)
        mean_error = prediction.mean - euler_mean
        covariance_error = prediction.scaled_covariance - euler_covariance
        history = metadata["history"]
        best = min(history, key=lambda row: row["validation_loss"])
        by_horizon_component = []
        for h, horizon in enumerate(config.query_times):
            for k, component in enumerate(COMPONENTS):
                mean = mean_error[:, h, k]
                variance = covariance_error[:, h, k, k]
                by_horizon_component.append({"horizon": horizon, "component": component,
                    "mean_rmse": float(np.sqrt(np.mean(mean**2))),
                    "mean_maximum_absolute_error": float(np.max(np.abs(mean))),
                    "mean_entries_above_tolerance": int(np.sum(np.abs(mean) > config.tolerance)),
                    "variance_rmse": float(np.sqrt(np.mean(variance**2))),
                    "variance_maximum_absolute_error": float(np.max(np.abs(variance))),
                    "variance_entries_above_tolerance": int(np.sum(np.abs(variance) > config.tolerance)),
                    "predicted_variance_std_over_initial_states": float(
                        prediction.scaled_covariance[:, h, k, k].std())})
        rows.append({"seed": seed, "checkpoint_sha256": checkpoint_sha,
                     "metadata_sha256": _sha(metadata_path), "evaluation_sha256": _sha(evaluation_path),
                     "training": {"epochs": metadata["epochs"], "best_epoch": best["epoch"],
                         "best_validation_loss": metadata["best_validation_loss"],
                         "best_epoch_training_loss": best["training_loss"],
                         "last_training_loss": history[-1]["training_loss"],
                         "last_validation_loss": history[-1]["validation_loss"],
                         "last_validation_minus_training": history[-1]["validation_loss"] - history[-1]["training_loss"],
                         "validation_residuals": metadata["validation_residuals"]},
                     "euler_target_mean": _error_summary(mean_error, initial, config.query_times, config.tolerance),
                     "euler_target_covariance": _error_summary(covariance_error, initial, config.query_times, config.tolerance),
                     "continuous_target_mean": _error_summary(prediction.mean - continuous_mean, initial, config.query_times, config.tolerance),
                     "continuous_target_covariance": _error_summary(prediction.scaled_covariance - continuous_covariance, initial, config.query_times, config.tolerance),
                     "per_horizon_component": by_horizon_component,
                     "mean_signed_error_by_anchor_horizon_component": mean_error,
                     "covariance_signed_error_by_anchor_horizon_components": covariance_error,
                     "evaluation_gate": evaluation["neural_error"]["accuracy_gate"]})
    payload = {"diagnostic": "Posthoc exploration on the unchanged prespecified OU analytic test bank",
               "interpretation": "No refitting, no allocation changes, no certification outside this bank; covariance entry counts include both symmetric halves",
               "config_hash": config.config_hash, "source_hash": current_source,
               "analysis_sha256": _sha(Path(__file__)),
               "config_file_sha256": _sha(config_path), "n_test_anchors": len(initial),
               "horizons": config.query_times, "components": COMPONENTS,
               "tolerance": config.tolerance, "initial_states": initial,
               "test_bank_hash": stable_hash(initial.tolist()),
               "irreducible_population_loss": _irreducible_euler_loss(config), "seeds": rows}
    output = directory / "report" / "ou_generalization_probe.json"
    write_json(output, payload)
    return output, json_ready(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    arguments = parser.parse_args()
    output, payload = probe(arguments.run_directory)
    print(output)
    for row in payload["seeds"]:
        print(json.dumps({"seed": row["seed"], "epochs": row["training"]["epochs"],
                          "mean": row["euler_target_mean"],
                          "covariance": row["euler_target_covariance"]}))


if __name__ == "__main__":
    main()
