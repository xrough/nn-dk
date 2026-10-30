"""Reports that separate neural statistics error, forward bias and uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import json_ready, write_json


def _number(value):
    return "—" if value is None else f"{float(value):.5g}"


def _rows(evaluations):
    rows = []
    for record in evaluations:
        if record.get("problem") == "ou":
            for component in ("neural_error", "total_error", "time_error"):
                comparison = record.get(component, {})
                rows.append({"problem": "ou", "case": "analytic test states", "seed": record["seed"],
                             "component": component, **comparison})
        for case, values in record.get("cases", {}).items():
            for component in ("conditional_total_error", "conditional_neural_error",
                              "ensemble_total_error", "ensemble_neural_error", "ensemble_time_error"):
                rows.append({"problem": "dk", "case": case, "seed": record["seed"],
                             "component": component, **values[component]})
    return rows


def _plot_errors(rows, output):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    chosen = [r for r in rows if r["component"] in ("total_error", "ensemble_total_error")]
    labels = [f"{r['problem']} {r['case']}\nseed {r['seed']}" for r in chosen]
    x = np.arange(len(chosen))
    for axis, metric, title in zip(axes, ("mean_normalized_rmse", "scaled_covariance_normalized_rmse"),
                                   ("Mean error", "Same-time covariance error"), strict=True):
        if chosen:
            axis.bar(x, [r["errors"][metric] for r in chosen], color="tab:blue")
            axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
        else:
            axis.text(.5, .5, "No completed model evaluations", ha="center", transform=axis.transAxes)
        axis.axhline(.05, color="0.5", linestyle="--", label="5% target")
        axis.set(title=title, ylabel="Fixed-unit normalized RMSE")
        axis.set_yscale("symlog", linthresh=1e-4)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Conditional BSDE statistics: each optimizer seed remains separate\n"
                    "Errors against continuous OU / fine stopped DK; bars are point estimates")
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _plot_costs(evaluations, output):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    labels, inference, reference = [], [], []
    for record in evaluations:
        if record.get("problem") == "ou":
            labels.append(f"OU\nseed {record['seed']}")
            inference.append(record.get("cost", {}).get("conditional_batch_inference_seconds", 0))
            reference.append(0.)
        for case, result in record.get("cases", {}).items():
            labels.append(f"DK {case}\nseed {record['seed']}")
            inference.append(result["cost"].get("ensemble_online_total_seconds", 0))
            reference.append(result["cost"].get("reference_generation_seconds", 0))
    x = np.arange(len(labels))
    for axis, values, title in zip(axes, (inference, reference),
                                   ("Neural online prediction", "Coupled reference audit"), strict=True):
        if labels:
            axis.bar(x, values, color="tab:green" if axis is axes[0] else "tab:orange")
            axis.set_xticks(x, labels, rotation=30, ha="right", fontsize=7)
        else:
            axis.text(.5, .5, "No completed timings", ha="center", transform=axis.transAxes)
        axis.set(title=title, ylabel="Wall time (seconds)", yscale="symlog")
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Timings refer to each recorded allocation; reference work is shared across seeds\n"
                    "These bars do not establish a matched-accuracy speedup")
    figure.savefig(output, dpi=150)
    plt.close(figure)


def write_bsde_report(directory, config, calibration, evaluations):
    """Write strict JSON, Markdown and two scientific figures; never retrain."""
    import matplotlib

    matplotlib.use("Agg")
    directory = Path(directory)
    report = directory / "report"
    report.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text()) if (directory / "manifest.json").exists() else {}
    rows = _rows(evaluations)
    json_path, markdown = report / "bsde_results.json", report / "bsde_results.md"
    error_plot, cost_plot = report / "bsde_errors.png", report / "bsde_costs.png"
    payload = {"manifest": manifest, "config": config.to_dict(), "calibration": calibration,
               "evaluations": evaluations, "comparison_rows": rows,
               "scope": "Conditional means and same-time 4x4 covariances; no sampled SPDE paths or joint multitime covariance"}
    write_json(json_path, payload)
    _plot_errors(rows, error_plot)
    _plot_costs(evaluations, cost_plot)
    lines = ["# Conditional neural BSDE statistics", "",
             f"Configuration **{config.name}**, forward step **{config.dt:g}**. "
             "This workflow learns conditional statistics, then averages over IID initial histograms. "
             "It does not generate density trajectories or establish continuum DK convergence.", "",
             f"Numerical calibration: **{calibration.get('status', 'not completed')}**. "
             "Software acceptance and scientific accuracy are separate conclusions.", "",
             "The four observables are exactly cell-integrated normalized cosine/sine modes 1 and 2. "
             "Means use sqrt(N) fluctuation scaling, covariances use N scaling, and both use fixed analytic "
             "uniform-histogram units. The OU calibration uses dimensionless coordinates and N=1.", "",
             "## Prediction errors", "",
             "| Problem | Initial law | Seed | Comparison | Mean RMSE | Covariance RMSE | Max mean error | Max covariance error | Gate |",
             "|---|---|---:|---|---:|---:|---:|---:|---|"]
    for row in rows:
        error = row.get("errors", {})
        entries = [row["problem"], row["case"], str(row["seed"]), row["component"],
                   *[_number(error.get(key)) for key in ("mean_normalized_rmse", "scaled_covariance_normalized_rmse",
                     "maximum_mean_normalized_error", "maximum_scaled_covariance_normalized_error")],
                   row.get("accuracy_gate", {}).get("status", "unresolved")]
        lines.append("| " + " | ".join(entries) + " |")
    if not rows:
        lines.append("| — | — | — | No completed evaluations | — | — | — | — | unresolved |")
    lines.extend(["", "![Mean and covariance errors](bsde_errors.png)", "",
                  "Neural error compares the prediction with the forward Euler law used in training; total error "
                  "compares with analytic continuous OU moments or the fine DK reference. The OU time component is "
                  "exact. DK coarse/fine reference paths share increments summed on the coarse grid. These "
                  "components are signed contrasts and their RMSEs are not additive.", "",
                  "Only same-time 4x4 covariance matrices are predicted. No 12-dimensional joint covariance or "
                  "two-time consistency is implied. Positive semidefiniteness alone does not establish accuracy.", "",
                  "## Noise, stopping and uncertainty", "",
                  "The forward physical process and BSDE martingale share Brownian increments. Initialization, "
                  "optimizer shuffling and test streams are independently keyed. The martingale input is predictable; "
                  "the full crossing-step increment is retained with its active-at-start mask, and later contributions "
                  "vanish. The state freezes at the first Euler-polygon boundary crossing. Every stopped path remains "
                  "in the statistics.", "",
                  "Conditional reference intervals independently resample whole continuation families within each fixed "
                  "test anchor. They quantify continuation Monte Carlo error, not generalization to unseen anchors. "
                  "Ensemble intervals resample whole matched initial-histogram/forcing families and include uncertainty "
                  "from initial sampling. Ensemble covariance includes both conditional covariance and covariance of "
                  "conditional means. Optimizer seeds are never pooled.", "",
                  f"The target is {config.tolerance:.2%} normalized error, with each uncertainty contribution below "
                  f"{config.uncertainty_budget:.2%}. Intervals are pointwise. DK reference bias remains uncertified: "
                  "finite refinements do not bound residual continuous-time bias. An unresolved gate can coexist with "
                  "a small empirical discrepancy. Test allocations are fixed before inspecting outcomes.", "",
                  "| Seed | Initial law | Ensemble fine-reference paths | Stopped | Stop probability | Zero-stop upper 95% |",
                  "|---:|---|---:|---:|---:|---:|"])
    for record in evaluations:
        for case, result in record.get("cases", {}).items():
            stop = result["ensemble_stopping"]["fine"]
            lines.append("| " + " | ".join([str(record["seed"]), case, str(stop["n_paths"]),
                str(stop["n_stopped"]), _number(stop["probability_by_horizon"]),
                _number(stop["zero_stop_one_sided_95_upper"])]) + " |")
    lines.extend(["", "## Training and terminal residuals", "",
                  "The terminal residual contains both statistical fitting error and finite-step stochastic-integral "
                  "representation error. Its value is not an estimate of the value-head prediction error.", "",
                  "| Problem | Seed | Epochs | Training (s) | First epoch (s) | Best validation loss |",
                  "|---|---:|---:|---:|---:|---:|"])
    for record in evaluations:
        training = record.get("training", {})
        lines.append("| " + " | ".join([record["problem"], str(record["seed"]),
            str(training.get("epochs", "—")), _number(training.get("training_seconds")),
            _number(training.get("first_epoch_seconds")), _number(training.get("best_validation_loss"))]) + " |")
    lines.extend(["", "| Problem | Seed | Horizon | Validation mean-block residual MSE | Validation second-moment residual MSE |",
                  "|---|---:|---:|---:|---:|"])
    for record in evaluations:
        for horizon, residual in record.get("training", {}).get("validation_residuals", {}).items():
            lines.append("| " + " | ".join([record["problem"], str(record["seed"]), horizon,
                _number(residual["mean_block_mse"]), _number(residual["second_moment_block_mse"])]) + " |")
    brownian = calibration.get("brownian", {})
    lines.extend(["", "## Calibration and computational cost", "",
                  "For the Brownian quadratic payoff, exact first-order integrands leave a terminal residual with "
                  f"expected MSE **{_number(brownian.get('expected_terminal_residual_mse'))}** and measured MSE "
                  f"**{_number(brownian.get('empirical_terminal_residual_mse'))}**. Its nonzero value is expected "
                  "from stochastic-integral discretization and is not direct evidence of an inaccurate value head.", "",
                  "![Online inference and reference audit costs](bsde_costs.png)", "",
                  "Online DK ensemble prediction includes drawing initial histograms and evaluating the value network; "
                  "it requires no forward trajectories. Shared reference-generation costs are repeated per seed for "
                  "context and must not be summed. The audit simulates both coarse and fine reference paths, so it "
                  "is not a standalone matched-accuracy comparator. Training, data generation, checkpoint loading and "
                  "amortization inputs are recorded separately in the run manifest, training records and JSON. Checkpoint "
                  "loading is excluded from online timings unless separately recorded. No speed advantage is "
                  "claimed without comparing the same requested accuracy and query allocation.", "",
                  "## Reproduction", "", f"Configuration hash: `{config.config_hash}`. "
                  f"Source hash: `{manifest.get('source_hash', 'not recorded')}`.", "",
                  "The JSON contains individual optimizer seeds, pointwise intervals, normalization, stopped-path "
                  "counts and timings. Checkpoints are specific to the forward timestep. A refinement requires a new "
                  "fit rather than reusing a checkpoint trained on another Euler law.", ""])
    markdown.write_text("\n".join(lines))
    return json_ready({"markdown": markdown.resolve(), "json": json_path.resolve(),
                       "figure": error_plot.resolve(), "cost_figure": cost_plot.resolve()})
