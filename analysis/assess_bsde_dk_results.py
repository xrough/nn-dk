#!/usr/bin/env python3
"""Describe completed DK moment records without fitting or simulating paths.

This is posthoc analysis of the existing, fixed conditional and ensemble banks.
The two banks differ; cancellation within the conditional bank is described
separately from the accuracy of the ensemble bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nn_dk.bsde_config import BSDEConfig
from nn_dk.data import json_ready, source_hash, write_json

COMPONENTS = ("cos1", "sin1", "cos2", "sin2")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rms(value):
    return float(np.sqrt(np.mean(np.asarray(value)**2)))


def _interval_summary(interval, tolerance):
    interval = np.asarray(interval)
    low, high = interval[..., 0], interval[..., 1]
    return {"entries": int(low.size),
            "interval_excludes_zero": int(np.sum((low > 0) | (high < 0))),
            "interval_entirely_outside_tolerance": int(np.sum((low > tolerance) | (high < -tolerance))),
            "maximum_halfwidth": float(np.max((high-low)/2))}


def _comparison(record, times, tolerance, conditional):
    errors = record["errors"]
    mean = np.asarray(errors["mean_normalized_signed"])
    covariance = np.asarray(errors["scaled_covariance_normalized_signed"])
    mean_ci = np.asarray(record["uncertainty"]["mean_normalized_contrast_ci95"])
    covariance_ci = np.asarray(record["uncertainty"]["scaled_covariance_normalized_contrast_ci95"])
    if not conditional:
        mean, covariance = mean[None], covariance[None]
        mean_ci, covariance_ci = mean_ci[None], covariance_ci[None]
    off_diagonal = ~np.eye(4, dtype=bool)
    mode_numbers = np.repeat((1, 2), 2)
    cross_mode = mode_numbers[:, None] != mode_numbers[None, :]
    within_mode = off_diagonal & ~cross_mode
    rows = []
    for h, horizon in enumerate(times):
        rows.append({"horizon": horizon, "mean_rmse": _rms(mean[:, h]),
                     "covariance_rmse": _rms(covariance[:, h]),
                     "variance_rmse": _rms(np.diagonal(covariance[:, h], axis1=-2, axis2=-1)),
                     "off_diagonal_covariance_rmse": _rms(covariance[:, h, off_diagonal]),
                     "cross_mode_covariance_rmse": _rms(covariance[:, h, cross_mode]),
                     "within_mode_quadrature_covariance_rmse": _rms(covariance[:, h, within_mode]),
                     "mean_maximum_absolute_error": float(np.max(np.abs(mean[:, h]))),
                     "covariance_maximum_absolute_error": float(np.max(np.abs(covariance[:, h]))),
                     "mean_interval_counts": _interval_summary(mean_ci[:, h], tolerance),
                     "covariance_interval_counts": _interval_summary(covariance_ci[:, h], tolerance),
                     "components": [
                         {"component": component, "mean_rmse": _rms(mean[:, h, k]),
                          "mean_average_signed_error": float(mean[:, h, k].mean()),
                          "variance_rmse": _rms(covariance[:, h, k, k]),
                          "variance_average_signed_error": float(covariance[:, h, k, k].mean())}
                         for k, component in enumerate(COMPONENTS)]})
    result = {"mean_rmse": _rms(mean), "covariance_rmse": _rms(covariance),
              "variance_rmse": _rms(np.diagonal(covariance, axis1=-2, axis2=-1)),
              "off_diagonal_covariance_rmse": _rms(covariance[..., off_diagonal]),
              "cross_mode_covariance_rmse": _rms(covariance[..., cross_mode]),
              "within_mode_quadrature_covariance_rmse": _rms(covariance[..., within_mode]),
              "maximum_sampling_halfwidth": record["uncertainty"]["maximum_sampling_normalized_halfwidth"],
              "mean_interval_counts": _interval_summary(mean_ci, tolerance),
              "covariance_interval_counts": _interval_summary(covariance_ci, tolerance),
              "accuracy_gate": record["accuracy_gate"], "horizons": rows}
    if conditional:
        anchor_mean = mean.mean(0)
        result["fixed_anchor_mean_error_decomposition"] = {
            "all_anchor_mean_error_rmse": _rms(mean),
            "rms_after_averaging_signed_errors_across_fixed_anchors": _rms(anchor_mean),
            "rms_anchor_deviation_from_average_error": _rms(mean-anchor_mean),
            "interpretation": "Descriptive decomposition on the fixed eight-anchor bank, including its reference sampling error; not the independent ensemble bank"}
    return result


def assess(run_directory):
    directory = Path(run_directory).resolve()
    config = BSDEConfig.load(directory / "config.json")
    manifest = json.loads((directory / "manifest.json").read_text())
    current_source = source_hash()
    if manifest.get("config_hash") != config.config_hash or manifest.get("source_hash") != current_source:
        raise ValueError("Assessment source/configuration mismatch")
    rows, present = [], set()
    for path in sorted((directory / "evaluation").glob("dk_seed*.json")):
        record = json.loads(path.read_text())
        seed = record["seed"]
        if seed in present or seed not in config.optimizer_seeds:
            raise ValueError("Duplicate or unallocated DK evaluation seed")
        if not record.get("complete") or record.get("problem") != "dk":
            raise ValueError("Only completed DK evaluations can be assessed")
        if record.get("config_hash") != config.config_hash or record.get("source_hash") != current_source:
            raise ValueError("DK evaluation source/configuration mismatch")
        checkpoint = directory / "checkpoints" / "dk" / f"seed_{seed}.pt"
        checkpoint_sha = _sha(checkpoint)
        if record.get("model", {}).get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("DK evaluation checkpoint SHA mismatch")
        training = record["training"]
        best = min(training["history"], key=lambda row: row["validation_loss"])
        cases = {}
        for case, data in record["cases"].items():
            cases[case] = {
                key: _comparison(data[key], config.query_times, config.tolerance,
                                 conditional=key.startswith("conditional"))
                for key in ("conditional_neural_error", "conditional_total_error",
                            "ensemble_neural_error", "ensemble_total_error", "ensemble_time_error")}
            cases[case]["stopping"] = {"conditional": data["conditional_stopping"],
                                        "ensemble": data["ensemble_stopping"]}
        rows.append({"seed": seed, "evaluation_sha256": _sha(path),
                     "checkpoint_sha256": checkpoint_sha,
                     "training": {"epochs": training["epochs"], "best_epoch": best["epoch"],
                                  "best_epoch_training_loss": best["training_loss"],
                                  "best_validation_loss": training["best_validation_loss"],
                                  "last_epoch": training["history"][-1],
                                  "validation_residuals": training["validation_residuals"]},
                     "cases": cases})
        present.add(seed)
    if not rows:
        raise ValueError("No completed DK evaluations available")
    payload = {"analysis": "Posthoc assessment of completed fixed-allocation DK evaluation records",
               "complete_configured_allocation": present == set(config.optimizer_seeds),
               "missing_optimizer_seeds": sorted(set(config.optimizer_seeds)-present),
               "config_hash": config.config_hash, "source_hash": current_source,
               "analysis_sha256": _sha(Path(__file__)), "seeds": rows,
               "interpretation": "Intervals are pointwise; reference bias remains uncertified. Covariance entry counts include both symmetric halves. No models were fitted or evaluated by this JSON-only analysis."}
    output = directory / "report" / "dk_error_assessment.json"
    write_json(output, payload)
    return output, json_ready(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    arguments = parser.parse_args()
    output, payload = assess(arguments.run_directory)
    print(output)
    for row in payload["seeds"]:
        for case, data in row["cases"].items():
            print(json.dumps({"seed": row["seed"], "case": case,
                              "conditional_neural": {k: v for k, v in data["conditional_neural_error"].items() if k != "horizons"},
                              "ensemble_neural": {k: v for k, v in data["ensemble_neural_error"].items() if k != "horizons"}}))


if __name__ == "__main__":
    main()
