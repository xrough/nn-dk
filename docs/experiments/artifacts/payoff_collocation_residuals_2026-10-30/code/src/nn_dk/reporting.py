"""Render benchmark artifacts without running or refitting any method."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import json_safe


def _name(record: dict[str, Any]) -> str:
    return str(record.get("method", record.get("name", record.get("prediction_type", "method"))))


def _number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4g}"


def _timing(record: dict[str, Any], name: str) -> float | None:
    aliases = {"generation_seconds": "training_data_seconds", "inference_seconds": "prediction_seconds",
               "total_prediction_seconds": "standalone_total_seconds"}
    for source in (record.get("costs", {}), record.get("cost", {}), record):
        value = source.get(name, source.get(aliases.get(name, "")))
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return None


def render_report(records: list[dict[str, Any]], calibration: dict[str, Any],
                  manifest: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write strict JSON, a readable report, and a figure from immutable inputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_records = json_safe(records)
    safe_calibration = json_safe(calibration)
    safe_manifest = json_safe(manifest)
    payload = {"manifest": safe_manifest, "calibration": safe_calibration, "records": safe_records}
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    certification = calibration.get("reference_certification", {})
    lines = [
        "# NN-DK benchmark", "",
        "This report compares ensemble-start predictions of the declared spline observations of the particle law. "
        "Errors, uncertainty, failures, and costs are reported separately. A neural speed advantage is an experimental question.", "",
        f"Calibration implementation checks: **{calibration.get('status', 'not run')}**. "
        f"Configured reference-certification gate: **{certification.get('status', 'unresolved')}**.", "",
        certification.get("interpretation", "Reference uncertainty and discretization bias have not been jointly certified."), "",
        "## Accuracy and failures", "",
        "Each row is one method/context/optimizer-seed comparison. RMSEs use known uniform-particle filtered-observable "
        "scales, or explicitly declared training-only scales. They do not normalize by held-out reference estimates.", "",
        "| Method | Context | Status | Mean normalized RMSE | Scaled covariance normalized RMSE | Invalid fraction | Reference paths |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for record in safe_records:
        error = record.get("errors") or {}
        pred = record.get("prediction", {})
        diag = pred.get("diagnostics", {})
        ref = record.get("reference", {})
        lines.append("| " + " | ".join([
            _name(record).replace("|", "/"), str(record.get("context_id", ""))[:10],
            str(record.get("status", "unknown")), _number(error.get("mean_normalized_rmse")),
            _number(error.get("scaled_covariance_normalized_rmse")),
            _number(diag.get("invalid_path_frequency")), str(ref.get("n_independent_paths", "—")),
        ]) + " |")
    lines.extend([
        "", "![Accuracy and costs](accuracy_cost.png)", "",
        "The figure summarizes observed comparisons by method using medians and ranges across the configured contexts/seeds; "
        "these ranges are not confidence intervals. Failed ensembles have no error estimate and remain visible through their invalid fraction.", "",
        "## Uncertainty and interpretation", "",
        "Reference intervals use an independent-trajectory percentile bootstrap. Online trajectory-model sampling intervals "
        "are stored separately. Direct moment predictions have no online sampling interval. These intervals do not include "
        "training uncertainty, reference discretization bias, or generalization error.", "",
        "The JSON contains per-query means, scaled variances, all two-time covariance pairs, covariance positivity checks, "
        "scalar quantiles/skewness/kurtosis, scalar Wasserstein distances where paths exist, mass errors, and failure times. "
        "An ensemble with any failed path is unavailable: the report never averages only surviving paths.", "",
        "A passed software calibration does not establish 5% fluctuation accuracy. A small pilot may leave scientific comparisons unresolved.", "",
        "## Measured costs", "",
        "| Method | Training data (s) | Training (s) | Initialization (s) | Online prediction incl. initialization (s) | Standalone total (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for record in safe_records:
        lines.append("| " + " | ".join([
            _name(record).replace("|", "/"),
            *[_number(_timing(record, key)) for key in ("generation_seconds", "training_seconds", "initialization_seconds", "inference_seconds", "total_prediction_seconds")],
        ]) + " |")
    lines.extend([
        "", "Cost fields are wall-clock measurements for this run and hardware. Shared generation/training costs may be "
        "repeated across rows; summing these rows would double-count shared work. Standalone total includes data generation, training, "
        "and online prediction where recorded. Single-context latency and batched "
        "throughput, when measured, are preserved in each record's cost fields.", "",
        "## Reproduction", "",
        "The adjacent report JSON records the complete run manifest, calibration, and all comparisons. Reproduce with the "
        "same configuration, software/dependency lock, seed, target identifier, grouped data split, and code hash. "
        "Reporting reads saved outputs; it does not regenerate labels or refit models.", "",
        "```json", json.dumps(safe_manifest, indent=2, allow_nan=False), "```", "",
    ])
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    methods = list(dict.fromkeys(_name(record) for record in safe_records))
    figure, axes = plt.subplots(2, 2, figsize=(max(10, len(methods) * 1.1), 7), constrained_layout=True)
    panels = [
        ("Mean normalized RMSE", lambda r: (r.get("errors") or {}).get("mean_normalized_rmse")),
        ("Scaled covariance normalized RMSE", lambda r: (r.get("errors") or {}).get("scaled_covariance_normalized_rmse")),
        ("Invalid path fraction", lambda r: r.get("prediction", {}).get("diagnostics", {}).get("invalid_path_frequency")),
        ("Online prediction wall time (seconds)", lambda r: _timing(r, "inference_seconds")),
    ]
    for axis, (title, extract) in zip(axes.ravel(), panels, strict=True):
        for i, method in enumerate(methods):
            values = [extract(r) for r in safe_records if _name(r) == method]
            values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
            if len(values):
                median = float(np.median(values))
                axis.errorbar(i, median, yerr=[[median - values.min()], [values.max() - median]],
                              fmt="o", capsize=3, color="tab:blue")
            else:
                axis.text(i, 0, "unavailable", rotation=90, ha="center", va="bottom", fontsize=7, color="0.5")
        axis.set_title(title)
        axis.set_xticks(range(len(methods)), methods, rotation=30, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(bottom=0)
    figure_path = output_dir / "accuracy_cost.png"
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return {"markdown": str(report_path.resolve()), "json": str(json_path.resolve()), "figure": str(figure_path.resolve())}
