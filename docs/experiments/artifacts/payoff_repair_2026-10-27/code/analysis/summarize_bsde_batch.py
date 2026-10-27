#!/usr/bin/env python3
"""Summarize a completed conditional-BSDE run without simulation or fitting.

Usage: .venv/bin/python analysis/summarize_bsde_batch.py runs/bsde-first-batch

Reads immutable JSON and hashes checkpoint bytes without deserializing them.
HDF5 pools and active-run artifacts are never opened. Optimizer seeds remain
separate, and only same-time four-by-four covariance blocks are summarized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

_WORKFLOW = "conditional-bsde-moments-v1"


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _source_hash() -> str:
    root = Path(__file__).resolve().parents[1] / "src" / "nn_dk"
    return _stable_hash({str(path.relative_to(root)): path.read_text()
                         for path in sorted(root.rglob("*.py"))})


def _numbers(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numbers(item)]
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return [float(value)]
    return []


def _maximum(value: Any) -> float | None:
    numbers = _numbers(value)
    return max(map(abs, numbers)) if numbers else None


def _rms(value: Any) -> float | None:
    numbers = _numbers(value)
    return math.sqrt(sum(number**2 for number in numbers)/len(numbers)) if numbers else None


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _shape(value: Any, expected: tuple[int, ...], label: str):
    if expected:
        _require(isinstance(value, list) and len(value) == expected[0], f"Invalid {label} shape")
        for item in value:
            _shape(item, expected[1:], label)
    else:
        _require(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value), f"Nonfinite/non-numeric {label}")


def _provenance(value: dict, manifest: dict, label: str):
    for key in ("source_hash", "config_hash"):
        _require(value.get(key) == manifest.get(key), f"{label}/{key} mismatch")


def _training(metadata: dict, config: dict, problem: str, seed: int):
    _require(metadata.get("complete") is True, "Incomplete BSDE training metadata")
    expected_kind = "native-dk-bsde-moments-v1" if problem == "dk" else "linear-ou-bsde-moments-v1"
    _require(metadata.get("kind") == expected_kind, "Checkpoint model kind differs from the declared BSDE problem")
    _require((metadata.get("problem"), metadata.get("seed")) == (problem, seed),
             "Checkpoint metadata problem/optimizer-seed mismatch")
    counts = {"training_anchors": config["train_anchors"],
              "validation_anchors": config["validation_anchors"],
              "training_paths": config["train_anchors"]*config["train_continuations"],
              "validation_paths": config["validation_anchors"]*config["validation_continuations"]}
    _require(all(metadata.get(key) == count for key, count in counts.items()),
             "Checkpoint training allocation differs from declared anchors/continuations")
    history = metadata.get("history", [])
    _require(isinstance(metadata.get("epochs"), int) and 1 <= metadata["epochs"] <= config["training"]["epochs"],
             "Completed checkpoint has invalid epoch count")
    _require(len(history) == metadata["epochs"], "Training epoch history is incomplete")


def _comparison(comparison: dict, config: dict, conditional: bool, label: str):
    horizons = len(config["query_times"])
    leading = (config["conditional_anchors"],) if conditional else ()
    errors, uncertainty = comparison["errors"], comparison["uncertainty"]
    _shape(errors["mean_normalized_signed"], (*leading, horizons, 4), f"{label} mean errors")
    _shape(errors["scaled_covariance_normalized_signed"], (*leading, horizons, 4, 4), f"{label} covariance errors")
    for kind, expected in (("mean", errors["mean_normalized_signed"]),
                           ("scaled_covariance", errors["scaled_covariance_normalized_signed"])):
        actual = errors[f"{kind}_normalized_rmse"]
        _require(math.isclose(actual, _rms(expected), rel_tol=1e-10, abs_tol=1e-12),
                 f"{label} {kind} RMSE inconsistent with signed errors")
    if uncertainty.get("available") and "n_bootstrap" in uncertainty:
        _require(uncertainty["n_bootstrap"] == config["bootstrap_samples"],
                 "Bootstrap allocation differs from the manifest")
    if uncertainty.get("available"):
        for key, shape in (("mean_normalized_contrast_ci95", (*leading, horizons, 4, 2)),
                           ("scaled_covariance_normalized_contrast_ci95", (*leading, horizons, 4, 4, 2))):
            if key in uncertainty:
                _shape(uncertainty[key], shape, f"{label} {key}")
    if "minimum_predicted_covariance_eigenvalue" in comparison:
        _require(comparison["minimum_predicted_covariance_eigenvalue"] >= -1e-9,
                 "Negative predicted conditional covariance eigenvalue")
    if "minimum_predicted_conditional_covariance_eigenvalue" in comparison:
        _require(comparison["minimum_predicted_conditional_covariance_eigenvalue"] >= -1e-9,
                 "Negative predicted conditional covariance eigenvalue")


def _decomposition(total: dict, neural: dict, time_error: dict) -> dict:
    """Validate signed contrasts; aggregate error norms must not be added."""
    residuals = {}
    for key in ("mean_normalized_signed", "scaled_covariance_normalized_signed"):
        t, n, d = (_numbers(value["errors"][key]) for value in (total, neural, time_error))
        _require(len(t) == len(n) == len(d), "Signed error decomposition has inconsistent shapes")
        residuals[key] = max(abs(a-b-c) for a, b, c in zip(t, n, d, strict=True))
        _require(residuals[key] < 1e-9, "Total signed contrast is not neural plus time contrast")
    return {"maximum_absolute_signed_identity_residual": max(residuals.values()),
            "components": residuals,
            "interpretation": "Total = neural + forward-time signed contrasts. RMSE magnitudes are not additive."}


def _stopping(stop: dict, expected: int, config: dict) -> dict:
    _require(stop.get("n_paths") == expected and stop.get("all_stopped_paths_included") is True,
             "Reference path allocation/all-stopped-path inclusion mismatch")
    count = stop.get("n_stopped")
    _require(isinstance(count, int) and 0 <= count <= expected, "Invalid stopping count")
    _require(math.isclose(stop["probability_by_horizon"], count/expected), "Inconsistent stopping probability")
    _require(stop.get("times") == config["query_times"], "Stopping query times differ from configuration")
    cdf = stop.get("stopping_cdf")
    _shape(cdf, (len(config["query_times"]),), "stopping CDF")
    _require(all(0 <= x <= 1 for x in cdf) and cdf == sorted(cdf)
             and math.isclose(cdf[-1], stop["probability_by_horizon"]), "Invalid stopping CDF")
    result = dict(stop)
    result["zero_event_one_sided95_probability_upper"] = (
        -math.expm1(math.log(.05)/expected) if count == 0 else None)
    result["interpretation"] = "Zero stops do not establish zero probability; repeated resolutions/seeds are not extra independent trials."
    return result


def component_summary(comparison: dict, times: list[float], conditional=False) -> dict:
    """Retain signed errors, pointwise intervals and variance/cross-mode detail."""
    errors, uncertainty = comparison["errors"], comparison["uncertainty"]
    by_time = {}
    for index, time in enumerate(times):
        means = errors["mean_normalized_signed"]
        covs = errors["scaled_covariance_normalized_signed"]
        means = [anchor[index] for anchor in means] if conditional else means[index]
        covs = [anchor[index] for anchor in covs] if conditional else [covs[index]]
        diagonal = [matrix[i][i] for matrix in covs for i in range(4)]
        off_diagonal = [matrix[i][j] for matrix in covs for i in range(4) for j in range(4) if i != j]
        by_time[format(time, ".12g")] = {
            "mean_rmse": _rms(means), "covariance_rmse": _rms(covs),
            "variance_rmse": _rms(diagonal), "same_time_cross_mode_covariance_rmse": _rms(off_diagonal)}
    return {
        **errors, "by_time": by_time,
        "mean_ci95_max_abs_endpoint": _maximum(uncertainty.get("mean_normalized_contrast_ci95")),
        "covariance_ci95_max_abs_endpoint": _maximum(uncertainty.get("scaled_covariance_normalized_contrast_ci95")),
        "maximum_sampling_normalized_halfwidth": uncertainty.get("maximum_sampling_normalized_halfwidth"),
        "maximum_contrast_normalized_ci_endpoint": uncertainty.get("maximum_contrast_normalized_ci_endpoint"),
        "uncertainty": uncertainty, "accuracy_gate": comparison["accuracy_gate"],
        "minimum_predicted_conditional_covariance_eigenvalue": comparison.get(
            "minimum_predicted_conditional_covariance_eigenvalue",
            comparison.get("minimum_predicted_covariance_eigenvalue")),
        "conditional_fixed_anchor_bank": conditional,
        "covariance_scope": "same-time 4x4; no joint temporal covariance or path-sampling assertion",
    }


def _validate_native_case(value: dict, config: dict, case: str):
    for component in ("conditional_total_error", "conditional_neural_error",
                      "ensemble_total_error", "ensemble_neural_error", "ensemble_time_error"):
        _comparison(value[component], config, component.startswith("conditional"), f"{case}/{component}")
    conditional_paths = config["conditional_anchors"]*config["conditional_paths"]
    for kind, count in (("conditional", conditional_paths), ("ensemble", config["outer_paths"])):
        for level in ("coarse", "fine"):
            _stopping(value[f"{kind}_stopping"][level], count, config)
        diagnostics = value[f"{kind}_reference_diagnostics"]
        _require(diagnostics.get("all_stopped_paths_included") is True,
                 "Reference diagnostics omit stopped paths")
        _require(0 <= diagnostics["maximum_mass_error"] < 1e-10,
                 "Reference mass conservation check failed")
        _require(diagnostics["minimum_saved_mass"] >= -1e-12, "Reference contains negative saved states")
    cost = value["cost"]
    _require(cost["reference_paths"] == config["outer_paths"]
             and cost["n_initial_anchors"] == config["outer_paths"]
             and cost["conditional_reference"]["reference_paths"] == conditional_paths
             and cost["conditional_reference"]["n_initial_anchors"] == config["conditional_anchors"],
             "DK reference costs identify an incorrect path/anchor allocation")
    return _decomposition(value["ensemble_total_error"], value["ensemble_neural_error"], value["ensemble_time_error"])


def _flatten(value: dict[str, Any], prefix="") -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        elif isinstance(item, (list, tuple)):
            result[name] = json.dumps(item, separators=(",", ":"), allow_nan=False)
        else:
            result[name] = item
    return result


def summarize(directory: Path, output_dir: Path | None = None) -> dict[str, str]:
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = _read(manifest_path)
    _require(manifest.get("status") == "complete", "Findings require manifest status='complete'; resume the partial run first")
    _require(manifest.get("workflow") == _WORKFLOW, "Expected a conditional BSDE statistics workflow")
    config_path = directory / "config.json"
    config = _read(config_path)
    _require(config == manifest["config"] and _stable_hash(config) == manifest["config_hash"],
             "Saved configuration content/hash differs from the manifest")
    _require(_source_hash() == manifest["source_hash"],
             "Current numerical package source differs from the frozen BSDE source hash")
    calibration_path, records_path = directory / "calibration.json", directory / "evaluations.json"
    calibration, records = _read(calibration_path), _read(records_path)
    _provenance(calibration, manifest, "Calibration")
    _require(calibration.get("complete") is True and calibration.get("passed") is True,
             "Numerical BSDE calibration is incomplete or failed")
    for problem in ("brownian", "ou"):
        _require(calibration[problem].get("complete") is True and calibration[problem].get("passed") is True,
                 f"Incomplete/failed {problem} analytic calibration")
    _require(set(calibration["native_refinement"]) == set(config["cases"]),
             "Native calibration cases differ from configuration")
    for case, audit in calibration["native_refinement"].items():
        _require(audit.get("complete") is True and audit.get("passed") is True
                 and audit.get("n_independent_families") == config["refinement_paths"],
                 f"Native reference audit allocation/status mismatch: {case}")
        _require(audit.get("reference_steps") == [config["reference_dt"]/divisor for divisor in (1, 2, 4)],
                 "Native reference audit timesteps differ from configuration")
    keys = [(record["problem"], record["seed"]) for record in records]
    expected = {(problem, seed) for problem in ("ou", "dk") for seed in config["optimizer_seeds"]}
    _require(len(keys) == len(set(keys)) and set(keys) == expected,
             "Evaluation records do not match declared problems/optimizer seeds")
    rows, models, input_hashes = [], [], {}
    for record in records:
        problem, seed = record["problem"], record["seed"]
        _provenance(record, manifest, f"{problem}/seed{seed}")
        _require(record.get("complete") is True, "Incomplete individual BSDE evaluation")
        _require(record.get("times") == config["query_times"] and record.get("step_size") == config["dt"],
                 "Evaluation query times/forward step differ from configuration")
        checkpoint = directory / "checkpoints" / problem / f"seed_{seed}.pt"
        metadata_path = checkpoint.with_suffix(".json")
        metadata = _read(metadata_path)
        _training(metadata, config, problem, seed)
        _require(metadata == record.get("training"), "Evaluation training metadata differs from saved checkpoint metadata")
        _require(record.get("model", {}).get("checkpoint_sha256") == _hash_file(checkpoint),
                 "Evaluation/checkpoint SHA256 mismatch")
        _require((record["model"].get("problem"), record["model"].get("seed")) == (problem, seed),
                 "Evaluation model identity differs from problem/seed")
        identity = _stable_hash(record["model"])[:20]
        individual = directory / "evaluation" / f"{problem}_seed{seed}_{identity}.json"
        _require(_read(individual) == record, "Aggregate and individual BSDE evaluations differ")
        input_hashes[str(checkpoint.relative_to(directory))] = _hash_file(checkpoint)
        input_hashes[str(metadata_path.relative_to(directory))] = _hash_file(metadata_path)
        input_hashes[str(individual.relative_to(directory))] = _hash_file(individual)
        model = {"problem": problem, "seed": seed, "checkpoint_sha256": record["model"]["checkpoint_sha256"],
                 "training": metadata, "checkpoint_loading_seconds": record.get("checkpoint_loading_seconds"),
                 "evaluation_seconds_this_call": record.get("evaluation_seconds_this_call"),
                 "scope": "Training/online inference costs are distinct from shared coupled reference-audit cost"}
        if problem == "ou":
            _require(record.get("n_test_anchors") == config["ou_test_anchors"], "OU analytic test allocation mismatch")
            for component in ("neural_error", "total_error", "time_error"):
                # OU has a conditional anchor axis of a different declared size.
                ou_config = {**config, "conditional_anchors": config["ou_test_anchors"]}
                _comparison(record[component], ou_config, True, f"ou/{component}")
                rows.append({"problem": problem, "case": "analytic test states", "seed": seed,
                             "component": component, "conditioning": "complete initial OU state",
                             "summary": component_summary(record[component], config["query_times"], True)})
            model["decomposition"] = _decomposition(record["total_error"], record["neural_error"], record["time_error"])
            model["cost"] = record.get("cost", {})
        else:
            _require(set(record["cases"]) == set(config["cases"]), "DK evaluation cases differ from configuration")
            model["cases"] = {}
            for case in config["cases"]:
                value = record["cases"][case]
                decomposition = _validate_native_case(value, config, case)
                for component in ("conditional_total_error", "conditional_neural_error", "ensemble_total_error",
                                  "ensemble_neural_error", "ensemble_time_error"):
                    conditional = component.startswith("conditional")
                    rows.append({"problem": problem, "case": case, "seed": seed, "component": component,
                                 "conditioning": "fixed complete initial mass anchors" if conditional else "ensemble start",
                                 "summary": component_summary(value[component], config["query_times"], conditional)})
                model["cases"][case] = {"cost": value["cost"], "reference_bias": value["reference_bias"],
                    "decomposition": decomposition,
                    **{f"{kind}_reference_diagnostics": value[f"{kind}_reference_diagnostics"] for kind in ("conditional", "ensemble")},
                    **{f"{kind}_stopping": {level: _stopping(value[f"{kind}_stopping"][level],
                       config["conditional_anchors"]*config["conditional_paths"] if kind == "conditional" else config["outer_paths"], config)
                       for level in ("coarse", "fine")} for kind in ("conditional", "ensemble")}}
        models.append(model)
    for path in (manifest_path, config_path, calibration_path, records_path):
        input_hashes[str(path.relative_to(directory))] = _hash_file(path)
    calibration_summary = {"status": calibration["status"], "passed": calibration["passed"],
                           "brownian": calibration["brownian"], "ou": calibration["ou"],
                           "native_refinement": {case: {key: value[key] for key in (
                               "complete", "passed", "n_independent_families", "reference_steps",
                               "maximum_mass_error", "minimum_saved_mass", "reference_bias", "seconds")}
                               for case, value in calibration["native_refinement"].items()}}
    result = {"schema_version": 1, "status": "complete-artifact-summary", "run_directory": str(directory),
              "model_count": len(models), "expected_model_count": len(expected), "comparison_count": len(rows),
              "config": config, "run_elapsed_seconds": manifest.get("elapsed_seconds"),
              "provenance": {"analysis_script": str(Path(__file__).resolve()),
                  "analysis_script_sha256": _hash_file(Path(__file__)), "source_hash": manifest["source_hash"],
                  "config_hash": manifest["config_hash"], "current_package_source_matches": True,
                  "input_sha256": input_hashes, "validation_scope": "JSON declarations, immutable checkpoint bytes and current source; HDF5 traces are not reread"},
              "calibration": calibration_summary, "models": models, "rows": rows,
              "shared_forward_pool_generation_seconds": manifest.get("forward_pool_generation_seconds", {}),
              "interpretation": {"covariance_scope": "same-time 4x4; no joint temporal covariance",
                  "optimizer_seeds": "Separate fits, never pooled as Monte Carlo samples",
                  "sampling_intervals": "Pointwise trajectory-family bootstrap; conditional test anchors remain fixed",
                  "reference_bias": "Finite DK refinements do not certify remaining continuous-time bias",
                  "error_decomposition": "Signed total = neural + forward-time contrast; RMSEs are not additive",
                  "stopping": "Every frozen path retained; zero events have a positive probability upper bound",
                  "cost": "Shared generation/reference costs repeated across seeds must not be summed; paired reference audit is not standalone forecast timing",
                  "matched_accuracy_speedup": "Not assessed by this artifact summary",
                  "learning_or_temporal_convergence": "A fixed data/capacity/forward-step study does not establish a convergence rate"}}
    output = Path(output_dir).resolve() if output_dir else directory / "report"
    output.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output / "findings_summary.json", output / "findings_table.csv"
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(text)
    temporary.replace(json_path)
    flat_rows = []
    for row in rows:
        summary = row["summary"]
        selected = {key: value for key, value in summary.items() if key not in (
            "mean_normalized_signed", "scaled_covariance_normalized_signed", "uncertainty")}
        flat_rows.append(_flatten({**{key: value for key, value in row.items() if key != "summary"}, **selected}))
    columns = list(dict.fromkeys(key for row in flat_rows for key in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat_rows)
    temporary = csv_path.with_suffix(".csv.tmp")
    temporary.write_text(stream.getvalue())
    temporary.replace(csv_path)
    return {"summary": str(json_path), "table": str(csv_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.output), indent=2))


if __name__ == "__main__":
    main()
