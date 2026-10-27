#!/usr/bin/env python3
"""Summarize completed native-DK artifacts without fitting or resimulation.

Usage: .venv/bin/python analysis/summarize_first_batch.py runs/a-first-batch

Despite the historical filename, allocations, grids, cases, and optimizer seeds
come from the saved manifest. This script reads JSON artifacts only; it never
loads executable checkpoints or imports the numerical package.
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


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


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
    return math.sqrt(sum(x * x for x in numbers) / len(numbers)) if numbers else None


def _expected_keys(config: dict[str, Any]) -> set[tuple]:
    """Mirror the declared data/capacity axes, independent of numerical code."""
    keys = {(case, "oracle", None, dt)
            for case in config["cases"] for dt in config["step_sizes"]}
    largest = max(config["train_path_counts"])
    for case in config["cases"]:
        for mode in config["training_modes"]:
            counts = config["train_path_counts"] if mode == "oracle" else [largest]
            for count in counts:
                widths = config["widths"] if mode == "oracle" and count == largest else [config["base_width"]]
                for width in widths:
                    for seed in config["optimizer_seeds"]:
                        key = f"{mode}_n{count}_w{width}_s{seed}"
                        keys.update((case, key, ablation, dt)
                                    for ablation in ("drift", "noise", "both")
                                    for dt in config["step_sizes"])
    return keys


def _record_key(record: dict[str, Any]) -> tuple:
    return record["case"], record["fit_key"], record.get("ablation"), record["step_size"]


def _precision(uncertainty: dict[str, Any]) -> dict[str, Any]:
    return {side: {
        "mean_ci95_halfwidth_max": _maximum(values.get("mean_normalized_ci_halfwidth")),
        "covariance_ci95_halfwidth_max": _maximum(values.get("scaled_covariance_normalized_ci_halfwidth")),
        "stopping_ci95_halfwidth_max": _maximum(values.get("stopping_probability_ci_halfwidth")),
    } for side, values in uncertainty.get("sampling_components", {}).items()}


def _gate(comparison: dict[str, Any]) -> dict[str, Any]:
    gate = dict(comparison.get("accuracy_gate", {}))
    bias = gate.get("reference_bias")
    if isinstance(bias, dict):
        gate["reference_bias"] = {key: bias.get(key) for key in (
            "status", "normalized_bound", "empirical_difference_within_budget", "reason")}
    return gate


def component_summary(comparison: dict[str, Any] | None) -> dict[str, Any] | None:
    """Retain global, per-time variance, and cross-time covariance distinctions."""
    if comparison is None:
        return None
    errors = comparison.get("errors") or {}
    uncertainty = comparison.get("uncertainty", {})
    queries = errors.get("per_query", [])
    by_time: dict[str, list] = {}
    labels = {q["index"]: q for q in queries if "index" in q}
    for query in queries:
        if "time" in query:
            by_time.setdefault(format(query["time"], ".12g"), []).append(
                query.get("scaled_variance_normalized_error"))
    cross_time, by_pair = [], {}
    for pair in errors.get("covariance_pairs", []):
        if not pair.get("two_time"):
            continue
        error = pair.get("normalized_error")
        cross_time.append(error)
        first, second = labels.get(pair.get("i")), labels.get(pair.get("j"))
        if first and second:
            times = sorted((first["time"], second["time"]))
            key = "|".join(format(t, ".12g") for t in times)
            by_pair.setdefault(key, []).append(error)
    return {
        "mean_rmse": errors.get("mean_normalized_rmse"),
        "covariance_rmse": errors.get("scaled_covariance_normalized_rmse"),
        "mean_error_max": _maximum(errors.get("mean_normalized")),
        "covariance_error_max": _maximum(errors.get("scaled_covariance_normalized")),
        "variance_error_rmse_by_time": {time: _rms(values) for time, values in by_time.items()},
        "two_time_covariance_error_rmse": _rms(cross_time),
        "two_time_covariance_error_rmse_by_times": {times: _rms(values) for times, values in by_pair.items()},
        "mean_ci95_max_abs_endpoint": _maximum(uncertainty.get("mean_normalized_ci95")),
        "covariance_ci95_max_abs_endpoint": _maximum(uncertainty.get("scaled_covariance_normalized_ci95")),
        "stopping_ci95_max_abs_endpoint": _maximum(uncertainty.get("stopping_probability_difference_ci95")),
        "mean_standard_error_max": _maximum(uncertainty.get("mean_normalized_standard_error")),
        "covariance_standard_error_max": _maximum(uncertainty.get("scaled_covariance_normalized_standard_error")),
        "wasserstein_normalized_max": _maximum(errors.get("wasserstein_normalized")),
        "pool_precision": _precision(uncertainty),
        "paired": comparison.get("paired"),
        "accuracy_gate": _gate(comparison),
    }


def _outcome(estimate: dict[str, Any]) -> dict[str, Any]:
    result = dict(estimate.get("diagnostics", {}))
    stopping = estimate.get("stopping", {})
    result.update(stopping_probability=stopping.get("probability_by_horizon"),
                  stopping_cdf=stopping.get("cdf"),
                  restricted_mean_stopping_time=stopping.get("restricted_mean_stopping_time"))
    count, stopped = result.get("n_paths"), result.get("n_stopped")
    result["zero_event_one_sided95_probability_upper"] = (
        -math.expm1(math.log(0.05) / count) if stopped == 0 and count else None)
    return result


def _row(record: dict[str, Any]) -> dict[str, Any]:
    result = {key: record.get(key) for key in (
        "case", "stage", "fit_key", "method", "training_mode", "train_paths", "width", "seed", "step_size")}
    result["ablation"] = record.get("ablation", "oracle")
    result["total"] = component_summary(record["comparison"])
    result["same_step_neural"] = component_summary(record.get("neural_error"))
    result["oracle_time"] = component_summary(
        record.get("timestep_error", record["comparison"] if record["method"] == "oracle" else None))
    result["prediction"] = _outcome(record["comparison"].get("prediction", {}))
    result["reference"] = _outcome(record["comparison"].get("reference", {}))
    result["cost"] = record.get("cost", {})
    return result


def _checkpoint_summaries(directory: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = {}
    for record in records:
        if record["fit_key"] != "oracle":
            unique.setdefault((record["case"], record["fit_key"]), record)
    summaries = []
    for (case, key), record in sorted(unique.items()):
        path = directory / "checkpoints" / case / f"{key}.json"
        metadata = _read(path)
        if not metadata.get("complete"):
            raise ValueError(f"Completed run contains incomplete coefficient fit: {path}")
        history = metadata.get("history", [])
        finite = [row for row in history if _numbers(row.get("validation_loss"))]
        best = min(finite, key=lambda row: row["validation_loss"]) if finite else {}
        diagnostics_path = path.with_suffix(".diagnostics.json")
        diagnostics = _read(diagnostics_path)
        diagnostics = {k: v for k, v in diagnostics.items()
                       if k not in ("whitened_mean", "whitened_covariance")}
        full = _read(diagnostics_path)
        diagnostics["whitened_mean_max_abs"] = _maximum(full.get("whitened_mean"))
        timing_keys = ("device", "single_path_latency_seconds", "latency_step_size",
                       "standalone_batch_paths", "standalone_batch_seconds", "standalone_paths_per_second",
                       "standalone_timing_ablation", "timing_scope")
        summaries.append({
            "case": case, "fit_key": key, "training_mode": record.get("training_mode"),
            "train_paths": record.get("train_paths"), "width": record.get("width"), "seed": record.get("seed"),
            "metadata_path": str(path.relative_to(directory)),
            "diagnostics_path": str(diagnostics_path.relative_to(directory)),
            **{k: metadata.get(k) for k in ("kind", "complete", "epochs", "best_validation_loss",
                                           "training_seconds", "training", "validation", "training_examples",
                                           "validation_examples", "identity_hash")},
            "best_epoch": best.get("epoch"), "heldout_coefficients": diagnostics,
            "standalone_timing": {key: record.get("cost", {}).get(key) for key in timing_keys},
        })
    return summaries


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
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
    manifest = _read(directory / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Findings require manifest status='complete'; resume the partial run first")
    if manifest.get("workflow") != "native-dk-convergence-v1":
        raise ValueError("Expected a native-DK convergence workflow")
    config_path = directory / "config.json"
    if _read(config_path) != manifest["config"]:
        raise ValueError("Saved configuration differs from the manifest")
    # The aggregate record file is intentionally never read for a partial run.
    records_path = directory / "records.json"
    records = _read(records_path)
    keys = [_record_key(record) for record in records]
    expected = _expected_keys(manifest["config"])
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Aggregate comparisons do not match the fixed manifest allocation")
    calibration = _read(directory / "calibration.json")
    if not calibration.get("complete"):
        raise ValueError("A0 calibration is incomplete")
    for key in ("source_hash", "config_hash"):
        if calibration.get(key) != manifest.get(key):
            raise ValueError(f"Calibration/manifest {key} mismatch")
    rows = [_row(record) for record in records]
    linear = calibration.get("linear_gaussian", {})
    linear_rows = linear.get("euler_refinement", [])
    result = {
        "schema_version": 1, "status": "complete-artifact-summary", "run_directory": str(directory),
        "record_count": len(rows), "expected_record_count": len(expected),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "provenance": {
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
            "input_config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "declared_config_hash": manifest["config_hash"],
            "declared_source_hash": manifest["source_hash"],
        },
        "manifest": manifest,
        "interpretation": {
            "scope": "Stopped spatially discrete DK at the manifest grid and particle count",
            "normalization": "Analytic uniform-IID histogram fluctuation units; sqrt(N) means, N covariances",
            "seeds": "Each fitted optimizer seed is reported separately; no method or seed averaging",
            "intervals": "Pointwise whole-family bootstrap 95% intervals, not simultaneous bands",
            "certification": "Existing accuracy gates are copied; RMSE alone never creates certification",
            "reference": "Finite reference refinements do not certify remaining continuous-time bias",
            "cost": "Shared generation/training and standalone timings repeat across rows; do not sum repeats",
            "stopping": "All stopped paths retained; zero-event upper bound is one-sided exact binomial at 95%",
            "wasserstein": "Descriptive empirical distances without bootstrap uncertainty",
        },
        "a0": {
            "status": calibration.get("status"), "seconds_total": calibration.get("seconds_total"),
            "linear_status": linear.get("status"), "linear_conservation": linear.get("conserved_null_mode"),
            "matrix_spectral_max_abs_difference": _maximum([
                row.get("matrix_spectral_max_abs_difference") for row in linear_rows]),
            "linear_euler_refinement": [{"step_size": row.get("step_size"), "errors": row.get("errors"),
                "matrix_spectral_max_abs_difference": row.get("matrix_spectral_max_abs_difference")}
                for row in linear_rows],
            "empirical_reference_bias": calibration.get("reference_bias", {}),
            "native_refinement": {case: {
                **{key: value.get(key) for key in ("complete", "completed_paths", "requested_paths",
                                                   "reference_steps", "simulation_seconds", "temporal_order")},
                "comparisons": {name: component_summary(comparison)
                                for name, comparison in value.get("comparisons", {}).items()},
            } for case, value in calibration.get("native_refinement", {}).items()},
        },
        "checkpoints": _checkpoint_summaries(directory, records), "records": rows,
    }
    output_dir = Path(output_dir).resolve() if output_dir else directory / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_dir / "findings_summary.json", output_dir / "findings_table.csv"
    flattened = [_flatten(row) for row in rows]
    fields = list(dict.fromkeys(key for row in flattened for key in row))
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(flattened)
    for path, content in ((json_path, json.dumps(result, indent=2, allow_nan=False) + "\n"),
                          (csv_path, csv_stream.getvalue())):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    return {"json": str(json_path), "csv": str(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        paths = summarize(args.run_directory, args.output_dir)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Findings unavailable: {error}\n")
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
