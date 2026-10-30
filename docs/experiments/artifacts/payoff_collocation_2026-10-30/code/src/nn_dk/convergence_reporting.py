"""Render immutable fixed-grid DK convergence results and their limitations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .convergence_metrics import temporal_rate
from .evaluation import json_safe


def _value(record: dict[str, Any], metric: str) -> float | None:
    value = (record.get("comparison", {}).get("errors") or {}).get(metric)
    return float(value) if isinstance(value, (float, int)) and np.isfinite(value) else None


def _number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4g}"


def _seconds(record: dict[str, Any]) -> float | None:
    cost = record.get("cost", {})
    for key in ("total_seconds", "prediction_seconds", "inference_seconds", "rollout_seconds", "simulation_seconds"):
        value = cost.get(key)
        if isinstance(value, (float, int)) and np.isfinite(value):
            return float(value)
    return None


def _coefficient_error(record: dict[str, Any], component: str) -> float | None:
    diagnostics = record.get("coefficient_diagnostics", {})
    for key in (f"normalized_{component}_rmse", f"{component}_relative_rmse",
                f"{component}_relative_error"):
        value = diagnostics.get(key)
        if isinstance(value, (float, int)) and np.isfinite(value):
            return float(value)
    return None


def _stop(record: dict[str, Any]) -> float | None:
    return record.get("comparison", {}).get("prediction", {}).get("stopping", {}).get("probability_by_horizon")


def _key(record: dict[str, Any], omit: tuple[str, ...] = ()) -> tuple[Any, ...]:
    return tuple(record.get(k) for k in ("case", "training_mode", "method", "seed", "width", "train_paths", "step_size")
                 if k not in omit)


def _label(record: dict[str, Any], omit: tuple[str, ...] = ()) -> str:
    keys = ("case", "method", "seed", "width", "train_paths", "step_size")
    return ", ".join(f"{k}={record[k]}" for k in keys if k in record and k not in omit)


def _is_joint(record):
    return record.get("method") in ("learned_both", "both_learned", "both") or record.get("ablation") == "both"


def _is_oracle(record):
    return record.get("method") == "oracle"


def _plot_settings(records, manifest):
    config = manifest.get("config", {})
    learned = [r for r in records if _is_joint(r)]
    seeds = config.get("optimizer_seeds") or list(dict.fromkeys(r.get("seed") for r in learned if r.get("seed") is not None)) or [0]
    widths = [r["width"] for r in learned if r.get("width") is not None]
    counts = [r["train_paths"] for r in learned if r.get("train_paths") is not None]
    steps = [r["step_size"] for r in records if r.get("step_size") is not None]
    return {"seed": seeds[0], "seeds": seeds,
            "base_width": config.get("base_width", widths[0] if widths else None),
            "largest_train_paths": max(config.get("train_path_counts") or counts or [0]),
            "coarsest_step": max(config.get("step_sizes") or steps or [1]),
            "cases": config.get("cases") or list(dict.fromkeys(r.get("case", "case") for r in records))}


def _mode_label(record):
    if _is_oracle(record):
        return "exact DK"
    return "NN, oracle labels" if record.get("training_mode") == "oracle" else "NN, increments"


def _selected_fit(record, settings, seed=None):
    return (_is_joint(record) and record.get("seed") == (settings["seed"] if seed is None else seed)
            and record.get("width") == settings["base_width"]
            and record.get("train_paths") == settings["largest_train_paths"])


def _style(record):
    if _is_oracle(record):
        return "x", ":"
    return ("o", "-") if record.get("training_mode") == "oracle" else ("s", "--")


def _component_value(record, component, metric):
    comparison = record.get(component, {})
    return (comparison.get("errors") or {}).get(metric)


def _plot_overview(records, settings, figure_path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(17, 13), constrained_layout=True)
    colors = {case: plt.get_cmap("tab10")(i) for i, case in enumerate(settings["cases"])}
    axes_specs = (("train_paths", "independent training paths"), ("width", "network width"),
                  ("step_size", "integration step"))
    for column, (xkey, xlabel) in enumerate(axes_specs):
        if xkey == "train_paths":
            chosen = [r for r in records if _is_joint(r) and r.get("seed") == settings["seed"]
                      and r.get("width") == settings["base_width"]
                      and r.get("training_mode") == "oracle" and r.get("step_size") == settings["coarsest_step"]]
        elif xkey == "width":
            chosen = [r for r in records if _is_joint(r) and r.get("seed") == settings["seed"]
                      and r.get("train_paths") == settings["largest_train_paths"]
                      and r.get("training_mode") == "oracle" and r.get("step_size") == settings["coarsest_step"]]
        else:
            chosen = [r for r in records if _selected_fit(r, settings)]
        oracle = [r for r in records if _is_oracle(r)
                  and (xkey == "step_size" or r.get("step_size") == settings["coarsest_step"])]
        for row, (metric, ylabel) in enumerate((
                ("mean_normalized_rmse", "sqrt(N) mean / uniform SD: RMSE"),
                ("scaled_covariance_normalized_rmse", "N covariance / uniform units: RMSE"))):
            axis = axes[row, column]
            groups = {}
            for record in chosen:
                groups.setdefault((record.get("case"), record.get("training_mode")), []).append(record)
            for (case, _), group in groups.items():
                group = sorted(group, key=lambda r: r[xkey])
                points = [(r[xkey], _value(r, metric)) for r in group if _value(r, metric) is not None]
                if points:
                    marker, linestyle = _style(group[0])
                    axis.plot(*zip(*points, strict=False), marker=marker, linestyle=linestyle,
                              color=colors.get(case), label=f"{case}: {_mode_label(group[0])}")
            for case in settings["cases"]:
                group = sorted([r for r in oracle if r.get("case") == case], key=lambda r: r["step_size"])
                if not group:
                    continue
                if xkey == "step_size":
                    points = [(r[xkey], _value(r, metric)) for r in group if _value(r, metric) is not None]
                else:
                    xs = sorted({r[xkey] for r in chosen if r.get("case") == case})
                    points = [(x, _value(group[0], metric)) for x in xs if _value(group[0], metric) is not None]
                if points:
                    axis.plot(*zip(*points, strict=False), marker="x", linestyle=":",
                              color=colors.get(case), label=f"{case}: exact DK")
            axis.set_xscale("log", base=2)
            axis.set_yscale("symlog", linthresh=1e-5)
            axis.set_ylim(bottom=0)
            axis.axhline(0.05, color="0.55", linestyle="--", linewidth=0.8)
            axis.set(title=f"{'Mean' if row == 0 else 'Covariance'} error vs {xlabel}", xlabel=xlabel, ylabel=ylabel)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=6.5, loc="best")
            else:
                axis.text(0.5, 0.5, "No matching completed curves", transform=axis.transAxes, ha="center")
            axis.grid(alpha=0.2)
    selected = [r for case in settings["cases"] for r in records if r.get("case") == case
                and r.get("step_size") == settings["coarsest_step"]
                and (_is_oracle(r) or _selected_fit(r, settings))]
    labels = [f"{r.get('case', '')}\n{_mode_label(r).replace('NN, ', '')}" for r in selected]
    x = np.arange(len(selected))
    coefficient_axis, stop_axis, cost_axis = axes[2]
    for component, offset, color in (("drift", -0.18, "tab:blue"), ("covariance", 0.18, "tab:orange")):
        vals = [_coefficient_error(r, component) for r in selected]
        indices = [i for i, value in enumerate(vals) if value is not None]
        coefficient_axis.bar(x[indices] + offset, [vals[i] for i in indices], width=0.36,
                             label=f"{component}: relative L2", color=color)
    coefficient_axis.set(title="Coefficient approximation (selected fits)", ylabel="Relative L2 error")
    coefficient_axis.legend(fontsize=7)
    stopping = [(i, _stop(r)) for i, r in enumerate(selected) if _stop(r) is not None]
    if stopping:
        stop_axis.bar([i for i, _ in stopping], [v for _, v in stopping], color="tab:purple", width=0.65)
        if not any(value > 0 for _, value in stopping):
            stop_axis.text(0.5, 0.15, "No stops in selected finite ensembles", ha="center",
                           transform=stop_axis.transAxes, fontsize=8, color="0.35")
    stop_axis.set(title="Stopping by query horizon", ylabel="Probability", ylim=(0, 1))
    for key, offset, title, color in (("prediction_seconds", -0.18, "integration", "tab:green"),
                                      ("total_seconds", 0.18, "audit + training/data", "tab:red")):
        points = [(i, r.get("cost", {}).get(key)) for i, r in enumerate(selected)
                  if isinstance(r.get("cost", {}).get(key), (float, int))]
        if points:
            cost_axis.bar([i + offset for i, _ in points], [v for _, v in points], width=0.36, label=title, color=color)
    cost_axis.set(title="Measured audit costs (selected fits)", ylabel="Wall time (seconds)")
    cost_axis.set_yscale("symlog", linthresh=0.01)
    cost_axis.set_ylim(bottom=0)
    if cost_axis.get_legend_handles_labels()[0]:
        cost_axis.legend(fontsize=7)
    for axis in axes[2]:
        axis.set_xticks(x, labels, rotation=30, ha="right", fontsize=6.5)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(f"Stopped discrete DK — optimizer seed {settings['seed']} (no seed averaging)\n"
                   f"Data/width: dt={settings['coarsest_step']:g}; time: width={settings['base_width']}, "
                   f"training paths={settings['largest_train_paths']}", fontsize=13)
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def _plot_case_refinement(records, settings, case, figure_path):
    import matplotlib.pyplot as plt

    seeds = settings["seeds"]
    figure, axes = plt.subplots(len(seeds), 2, figsize=(12, max(4, 3.5 * len(seeds))), squeeze=False,
                                constrained_layout=True)
    case_records = [r for r in records if r.get("case") == case]
    for row, seed in enumerate(seeds):
        chosen = [r for r in case_records if _selected_fit(r, settings, seed)]
        exact = sorted([r for r in case_records if _is_oracle(r)], key=lambda r: r["step_size"])
        for column, (metric, label) in enumerate((("mean_normalized_rmse", "Mean"),
                                                 ("scaled_covariance_normalized_rmse", "Covariance"))):
            axis = axes[row, column]
            for mode, color in (("oracle", "tab:blue"), ("increments", "tab:orange")):
                group = sorted([r for r in chosen if r.get("training_mode") == mode], key=lambda r: r["step_size"])
                for component, linestyle, suffix in (("comparison", "-", "total"), ("neural_error", ":", "NN only")):
                    points = [(r["step_size"], _component_value(r, component, metric)) for r in group]
                    points = [(x, y) for x, y in points if y is not None]
                    if points:
                        axis.plot(*zip(*points, strict=False), marker="o", linestyle=linestyle, color=color,
                                  label=f"{_mode_label(group[0])}: {suffix}")
            points = [(r["step_size"], _value(r, metric)) for r in exact if _value(r, metric) is not None]
            if points:
                axis.plot(*zip(*points, strict=False), "x--", color="black", label="Exact DK: time error")
            axis.axhline(0.05, color="0.6", linestyle="--", linewidth=0.8)
            axis.set_xscale("log", base=2)
            axis.set_yscale("symlog", linthresh=1e-5)
            axis.set_ylim(bottom=0)
            axis.set(title=f"{case}, optimizer seed {seed}: {label.lower()} error", xlabel="Integration step",
                     ylabel=f"Normalized {label.lower()} RMSE")
            axis.grid(alpha=0.2)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7, loc="best")
    figure.suptitle(f"Frozen coefficient networks: width={settings['base_width']}, training paths={settings['largest_train_paths']}\n"
                   "Total = NN vs fine reference; NN only = NN vs exact coefficients at same step", fontsize=12)
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def _rates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("step_size") is not None:
            groups.setdefault(_key(record, ("step_size",)), []).append(record)
    result = []
    for group in groups.values():
        if len({r.get("step_size") for r in group}) < 3:
            continue
        bounds = [r.get("comparison", {}).get("accuracy_gate", {}).get("reference_bias") for r in group]
        bound = max(b["normalized_bound"] for b in bounds) if all(
            isinstance(b, dict) and b.get("status") == "resolved" and isinstance(b.get("normalized_bound"), (int, float)) for b in bounds) else None
        for metric in ("mean_normalized_rmse", "scaled_covariance_normalized_rmse"):
            result.append({"group": _label(group[0], ("step_size",)),
                           **temporal_rate(group, metric=metric, reference_bias_bound=bound)})
    return result


def render_convergence_report(records: list[dict[str, Any]], calibration: dict[str, Any],
                              manifest: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write strict JSON, Markdown, and plots without recomputing or fitting models."""
    import matplotlib

    matplotlib.use("Agg")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = json_safe(records)
    settings = _plot_settings(records, manifest)
    case_figures = {case: "convergence_" + "".join(c if c.isalnum() or c in "_-" else "-" for c in str(case)) + ".png"
                    for case in settings["cases"]}
    rates = _rates(records)
    payload = json_safe({"manifest": manifest, "calibration": calibration,
                         "records": records, "temporal_rates": rates,
                         "plot_selection": settings, "case_refinement_figures": case_figures})
    json_path = output_dir / "convergence.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Project A: stopped discrete DK convergence", "",
        "The target is the stopped, spatially discrete Dean–Kawasaki system at a fixed grid and particle count. "
        "Native finite-volume masses are solver states. Physical sine/cosine test functions are integrated against "
        "the reconstructed field. This experiment does not establish continuum or unregularized density-valued DK convergence.", "",
        "Every trajectory remains in the statistics after stopping and is frozen at its first Euler-polygon "
        "boundary crossing. No positivity repair, clipping, or survivor-conditioned estimates are used.", "",
        f"Software/numerical calibration status: **{calibration.get('status', 'not recorded')}**. "
        "Software checks alone do not certify scientific accuracy.", "",
        "## Weak errors and stopping", "",
        "Rows retain each optimizer seed separately. Mean errors use sqrt(N) scaling; covariances use N scaling. "
        "Both use analytic uniform-IID cell-integrated mode fluctuation units, never test-derived normalizers. "
        "An unresolved gate means sampling or reference bias cannot certify the requested tolerance.", "",
        "| Case | Stage | Method | Mode | Paths | Width | Seed | Step | Mean RMSE | Covariance RMSE | Neural covariance error | Time covariance error | Stop probability | Accuracy gate |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        comparison = record.get("comparison", {})
        cells = [str(record.get(k, "—")).replace("|", "/") for k in
                 ("case", "stage", "method", "training_mode", "train_paths", "width", "seed")]
        cells.extend([_number(record.get("step_size")), _number(_value(record, "mean_normalized_rmse")),
                      _number(_value(record, "scaled_covariance_normalized_rmse")),
                      _number((record.get("neural_error", {}).get("errors") or {}).get("scaled_covariance_normalized_rmse")),
                      _number((record.get("timestep_error", {}).get("errors") or {}).get("scaled_covariance_normalized_rmse")),
                      _number(_stop(record)),
                      str(comparison.get("accuracy_gate", {}).get("status", "unresolved"))])
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "", "![Learning, time, coefficients, stopping, and cost](convergence_curves.png)", "",
        f"The overview explicitly selects the first configured optimizer seed **{settings['seed']}**. It shows both learned "
        "coefficients together and the exact-DK baseline; drift-only/noise-only ablations remain in the table and JSON. "
        f"Data curves use oracle supervision and fix width **{settings['base_width']}** and step **{settings['coarsest_step']:g}**; capacity curves fix the "
        f"largest training pool **{settings['largest_train_paths']}**, oracle supervision, and the same step. Temporal curves "
        "fix that largest pool and base width. Exact-DK horizontal lines in the data/width panels are the same-step "
        "baseline, independent of training allocation. Means and covariances occupy separate panels with their own units. "
        "Points are observed errors, not confidence bands; the horizontal 5% line is a target, not certification. "
        "Bottom panels use only the largest-pool/base-width/coarsest-step selections and label each case and training mode. "
        "Coefficient errors are relative L2 norms; cost bars separate integration from the coupled audit plus "
        "training/data total. Audit costs include the declared reference-noise construction and are not standalone solver timings. "
        "No seeds are averaged. All seeds remain in the tables/JSON and are plotted separately in the per-case refinements below. "
        "Across-seed ranges are optimization variability and are not Monte Carlo confidence intervals. "
        "The neural component compares learned and oracle coefficients at the same step; the time component compares "
        "oracle coefficients at the experiment step and fine reference step. Their separate signed contrasts are preserved in JSON.", "",
        "## Per-case refinement and coefficient-error decomposition", "",
        "Each row in these figures is a distinct optimizer seed at fixed data and width. Solid curves show total "
        "error against the fine reference, dotted curves isolate neural coefficient error, and the black dashed curve "
        "shows the exact-coefficient time-discretization difference. Mean and covariance errors are separated. "
        "These curves retain unresolved sampling and reference-bias limitations recorded in the tables.", "",
        *[f"![{case}: all optimizer seeds and error decomposition]({filename})\n" for case, filename in case_figures.items()],
        "## Uncertainty and temporal rates", "",
        "Pointwise 95% intervals resample whole independent trajectory families. Unpaired contrasts resample "
        "prediction and reference pools independently; paired refinement contrasts resample matched families together. "
        "Stopped paths and initial exits remain in the target estimates. Increment supervision uses raw Euler "
        "proposals, including crossing proposals, before applying the target stopping rule. "
        "Fitting uncertainty, reference discretization bias, and generalization are separate from these sampling intervals.", "",
        "The default target is 5% normalized statistical error with each sampling/reference contribution below 1.25%. "
        "Reference bias needs an explicit refinement audit. A finite-level refinement difference alone is not a proven "
        "bias bound. Scalar Wasserstein distances are descriptive and do not carry bootstrap certification in this report.", "",
        "The four coefficient ablations are oracle/oracle, learned drift/oracle noise, oracle drift/learned noise, "
        "and both learned. Comparing these separates drift and covariance errors. Exact-coefficient refinement supplies "
        "the numerical integrator baseline; learned coefficients are frozen within a temporal curve.", "",
        "| Curve | Metric | Rate status | Observed order | Reason |",
        "|---|---|---|---:|---|",
    ])
    if not rates:
        lines.append("| — | — | unresolved | — | Fewer than three recorded refinement levels |")
    for rate in rates:
        lines.append("| " + " | ".join([rate["group"], rate["metric"], rate["status"],
                                        _number(rate["order"]), rate["reason"]]) + " |")
    lines.extend([
        "", "Rates require at least three signal-resolved, decreasing refinement errors and a reference bias bound "
        "within budget. No first-order weak rate is assumed at a stopping boundary; plateaus and unresolved slopes are retained.", "",
        "## Measured costs", "",
        "| Case | Method | Seed | Generation (s) | Training (s) | Initialization (s) | Integration (s) | Audit + data/train (s) | Standalone batch (s) | Standalone step |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in records:
        cost = record.get("cost", {})
        cells = [str(record.get("case", "—")), str(record.get("method", "—")), str(record.get("seed", "—"))]
        cells.extend(_number(cost.get(key)) for key in ("generation_seconds", "training_seconds", "initialization_seconds", "prediction_seconds"))
        cells.append(_number(_seconds(record)))
        cells.extend((_number(cost.get("standalone_batch_seconds")), _number(cost.get("latency_step_size"))))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "", "Costs are wall-clock measurements on this run's hardware. Shared generation/training costs repeated "
        "across ablations must not be summed. The JSON preserves timing, per-query and two-time covariance contrasts, "
        "stopping distributions, conservation, coefficient diagnostics, seed identity, and target metadata. "
        "The audit total includes coupled evaluation work and training/data; it is not a standalone solver total. "
        "Standalone batch/throughput and single-path latency, when present, come from separate unpaired sampling "
        "calls with both coefficients learned at the stated timing step; they should not be attributed to every ablation or step. "
        "Integration-only timings exclude initialization and noise construction.", "",
        "## Reproduction", "", "```json",
        json.dumps(json_safe(manifest), indent=2, allow_nan=False), "```", "",
    ])
    markdown_path = output_dir / "convergence.md"
    markdown_path.write_text("\n".join(lines))
    figure_path = output_dir / "convergence_curves.png"
    _plot_overview(records, settings, figure_path)
    for case, filename in case_figures.items():
        _plot_case_refinement(records, settings, case, output_dir / filename)
    return {"markdown": str(markdown_path.resolve()), "json": str(json_path.resolve()),
            "figure": str(figure_path.resolve())}
