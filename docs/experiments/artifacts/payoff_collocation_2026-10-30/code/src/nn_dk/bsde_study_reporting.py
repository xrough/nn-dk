"""Strict factorial-study reports with conditional and ensemble errors separated."""

from __future__ import annotations

import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from .bsde_study_evaluation import test_bank_identity
from .data import json_ready, source_hash, write_json
from .types import stable_hash


def _number(value):
    return "—" if value is None else f"{float(value):.5g}"


def _interval(value):
    return "—" if value is None else "[" + ", ".join(_number(v) for v in value) + "]"


def _validate_records(directory, study, calibration, records):
    base, fingerprint = study.base_config, source_hash()
    if (not calibration.get("complete") or not calibration.get("passed")
            or calibration.get("config_hash") != base.config_hash
            or calibration.get("source_hash") != fingerprint):
        raise ValueError("Study reporting requires complete, passed calibration with matching provenance")
    expected = {(problem, architecture, anchors, children, seed)
                for problem, architecture, (anchors, children), seed in product(
                    ("ou", "dk"), study.architectures, study.allocations, base.optimizer_seeds)}
    if len(records) != study.expected_models:
        raise ValueError("Study report requires every configured model evaluation")
    seen, identities, training_views, validation_views = set(), set(), {}, {}
    for record in records:
        evaluation, factors = record.get("evaluation", {}), record.get("factors", {})
        problem, seed = evaluation.get("problem"), factors.get("optimizer_seed")
        key = (problem, factors.get("architecture"), factors.get("train_anchors"),
               factors.get("train_continuations"), seed)
        if key not in expected or key in seen:
            raise ValueError("Missing, duplicate or undeclared factorial study combination")
        seen.add(key)
        if (not evaluation.get("complete") or evaluation.get("seed") != seed
                or record.get("study_config_hash") != study.config_hash
                or record.get("evaluation_config_hash") != base.config_hash
                or evaluation.get("config_hash") != base.config_hash
                or evaluation.get("source_hash") != fingerprint
                or record.get("test_bank_id") != test_bank_identity(base, problem)
                or evaluation.get("step_size") != base.dt
                or tuple(evaluation.get("times", ())) != base.query_times):
            raise ValueError("Study evaluation target, source, configuration or test-bank mismatch")
        checkpoint = Path(record.get("checkpoint_path", ""))
        if not checkpoint.is_absolute():
            checkpoint = directory / checkpoint
        if not checkpoint.is_file():
            raise ValueError("Study report requires every saved statistics checkpoint")
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if (record.get("checkpoint_sha256") != sha
                or evaluation.get("model", {}).get("checkpoint_sha256") != sha):
            raise ValueError("Study report checkpoint SHA mismatch")
        identity = stable_hash({"bank": record["test_bank_id"], "factors": factors,
                                "checkpoint_sha256": sha})
        if record.get("study_model_id") != identity or identity in identities:
            raise ValueError("Study model identity mismatch or duplicate identity")
        identities.add(identity)
        train_fp, valid_fp = record.get("training_view_fingerprint"), record.get("validation_fingerprint")
        if not train_fp or not valid_fp:
            raise ValueError("Study report requires training and validation fingerprints")
        allocation = (problem, factors["train_anchors"], factors["train_continuations"])
        if (training_views.setdefault(allocation, train_fp) != train_fp
                or validation_views.setdefault(problem, valid_fp) != valid_fp):
            raise ValueError("Study fits did not share the declared training/validation views")
        training = evaluation.get("training", {})
        if (not training.get("complete")
                or training.get("training_anchors") != factors["train_anchors"]
                or training.get("training_paths") != study.training_paths
                or factors.get("train_paths") != study.training_paths
                or training.get("validation_anchors") != study.validation[0]
                or training.get("validation_paths") != study.validation[0] * study.validation[1]):
            raise ValueError("Study training metadata does not describe the actual selected allocation")
        if problem == "dk" and set(evaluation.get("cases", {})) != set(base.cases):
            raise ValueError("Study DK evaluation is missing an initial-profile case")
    if seen != expected:
        raise ValueError("Study report is missing configured factorial combinations")


def _comparison_rows(records):
    rows = []
    for record in records:
        evaluation = record["evaluation"]
        common = {**record["factors"], "problem": evaluation["problem"],
                  "study_model_id": record["study_model_id"], "test_bank_id": record["test_bank_id"]}
        if evaluation["problem"] == "ou":
            comparisons = [("analytic", "conditional", component, evaluation[component])
                           for component in ("neural_error", "total_error", "time_error")]
        else:
            comparisons = [(case, scope, f"{scope}_{component}", values[f"{scope}_{component}"])
                           for case, values in evaluation["cases"].items()
                           for scope, component in (("conditional", "neural_error"),
                               ("conditional", "total_error"), ("ensemble", "neural_error"),
                               ("ensemble", "total_error"), ("ensemble", "time_error"))]
        for case, scope, component, comparison in comparisons:
            errors, uncertainty = comparison["errors"], comparison.get("uncertainty", {})
            signed_covariance = errors.get("scaled_covariance_normalized_signed")
            max_variance = (None if signed_covariance is None else float(np.max(np.abs(
                np.diagonal(np.asarray(signed_covariance), axis1=-2, axis2=-1)))))
            rows.append({**common, "case": case, "scope": scope, "component": component,
                         "reference": ("training-step Euler" if component.endswith("neural_error")
                                       else "continuous OU" if evaluation["problem"] == "ou"
                                       else "fine stopped DK"),
                         "mean_rmse": errors["mean_normalized_rmse"],
                         "variance_rmse": errors["scaled_variance_normalized_rmse"],
                         "covariance_rmse": errors["scaled_covariance_normalized_rmse"],
                         "maximum_mean_error": errors["maximum_mean_normalized_error"],
                         "maximum_variance_error": max_variance,
                         "maximum_covariance_error": errors["maximum_scaled_covariance_normalized_error"],
                         "maximum_sampling_halfwidth": uncertainty.get("maximum_sampling_normalized_halfwidth"),
                         "maximum_ci_endpoint": uncertainty.get("maximum_contrast_normalized_ci_endpoint"),
                         "accuracy_gate": comparison["accuracy_gate"]["status"],
                         "point_estimate_within_tolerance": comparison["accuracy_gate"].get(
                             "point_estimate_within_tolerance")})
    return rows


def _shared_reference_rows(records, base):
    first = next(record["evaluation"] for record in records if record["evaluation"]["problem"] == "dk")
    rows = []
    for case in base.cases:
        values = first["cases"][case]
        for scope in ("conditional", "ensemble"):
            stopping, diagnostics = values[f"{scope}_stopping"], values[f"{scope}_reference_diagnostics"]
            cost = (values["cost"].get("conditional_reference", {})
                    if scope == "conditional" else values["cost"])
            for level in ("coarse", "fine"):
                rows.append({"case": case, "scope": scope, "level": level,
                             **stopping[level], **diagnostics,
                             "shared_coupled_reference_seconds": cost.get("reference_generation_seconds")})
    return rows


def _plot_learning_curves(rows, study, output):
    import matplotlib.pyplot as plt

    base = study.base_config
    panels = [("ou", "analytic"), *(("dk", case) for case in base.cases)]
    figure, axes = plt.subplots(len(panels), 4, figsize=(16, 3.5 * len(panels)),
                                squeeze=False, constrained_layout=True)
    markers, colors = ("o", "s", "^", "D", "v"), {"mlp": "tab:blue", "structured": "tab:orange"}
    handles = {}
    primary = [row for row in rows if row["component"].endswith("neural_error")]
    for index, (problem, case) in enumerate(panels):
        for column, (scope, metric) in enumerate(product(("conditional", "ensemble"),
                                                        ("mean_rmse", "variance_rmse"))):
            axis = axes[index, column]
            if problem == "ou" and scope == "ensemble":
                axis.text(.5, .5, "OU control uses conditional moments", ha="center", va="center",
                          fontsize=9, transform=axis.transAxes)
                axis.set_axis_off()
                continue
            for architecture in study.architectures:
                for seed_index, seed in enumerate(base.optimizer_seeds):
                    chosen = sorted((row for row in primary if row["problem"] == problem
                        and row["case"] == case and row["scope"] == scope
                        and row["architecture"] == architecture and row["optimizer_seed"] == seed),
                        key=lambda row: row["train_anchors"])
                    label = f"{architecture}, seed {seed}"
                    line, = axis.plot([row["train_anchors"] for row in chosen],
                                      [row[metric] for row in chosen],
                                      marker=markers[seed_index % len(markers)], color=colors[architecture],
                                      linestyle="-" if architecture == "mlp" else "--", alpha=.8,
                                      label=label)
                    handles[label] = line
            axis.axhline(base.tolerance, color="0.5", linestyle=":", linewidth=1)
            axis.set_xscale("log", base=2)
            axis.set_xticks([a for a, _ in study.allocations], [str(a) for a, _ in study.allocations])
            axis.set_yscale("symlog", linthresh=.01)
            axis.set_ylim(bottom=0)
            axis.set(title=f"{problem.upper()} {case}: {scope} {metric.split('_')[0]}",
                     xlabel="Independent training initial states", ylabel="Fixed-unit normalized RMSE")
            axis.grid(alpha=.2)
    figure.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=min(6, len(handles)))
    figure.suptitle(f"Same-step neural error at {study.training_paths} training trajectories per fit\n"
                    "Point estimates; optimizer seeds remain separate; dotted line is the 5% research target")
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _contrast_rows(contrasts, records):
    if isinstance(contrasts, dict):
        contrasts = contrasts.get("contrasts", contrasts.get("rows", []))
    if not isinstance(contrasts, list):
        raise ValueError("Study contrasts must be a list or a named contrast table")
    identities = {record["study_model_id"]: record for record in records}
    for row in contrasts:
        if (row.get("baseline_model_id") not in identities
                or row.get("candidate_model_id") not in identities):
            raise ValueError("Study contrast refers to an absent fitted model")
        baseline, candidate = identities[row["baseline_model_id"]], identities[row["candidate_model_id"]]
        if (row.get("test_bank_id") != baseline["test_bank_id"]
                or baseline["test_bank_id"] != candidate["test_bank_id"]
                or row.get("optimizer_seed") != baseline["factors"]["optimizer_seed"]
                or row.get("optimizer_seed") != candidate["factors"]["optimizer_seed"]):
            raise ValueError("Study contrasts must preserve the common test bank and optimizer seed")
    return contrasts


def write_study_report(directory, study_config, calibration, records, contrasts):
    """Write a complete, provenance-checked comparison without fitting any model."""
    import matplotlib

    matplotlib.use("Agg")
    directory, study = Path(directory), study_config
    _validate_records(directory, study, calibration, records)
    contrasts = _contrast_rows(contrasts, records)
    base, rows = study.base_config, _comparison_rows(records)
    shared = _shared_reference_rows(records, base)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    report = directory / "report"
    report.mkdir(parents=True, exist_ok=True)
    paths = {"markdown": report / "study_results.md", "json": report / "study_results.json",
             "csv": report / "study_table.csv", "figure": report / "study_learning_curves.png"}
    training_seconds = [record["evaluation"]["training"].get("training_seconds") for record in records]
    payload = {"workflow": "conditional-bsde-anchor-study-report-v1", "manifest": manifest,
               "study_config": study.to_dict(), "study_config_hash": study.config_hash,
               "source_hash": source_hash(), "calibration": calibration, "evaluations": records,
               "comparison_rows": rows, "paired_contrasts": contrasts,
               "shared_reference_rows": shared,
               "training_seconds_all_distinct_fits": sum(training_seconds)
               if all(value is not None for value in training_seconds) else None,
               "cost_scope": "Distinct fit costs can be summed. Shared reference rows repeat coarse/fine costs; never sum those rows.",
               "scope": "Conditional means and same-time 4x4 covariance; fixed-grid stopped DK; no density paths or joint-time covariance"}
    write_json(paths["json"], payload)
    with paths["csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot_learning_curves(rows, study, paths["figure"])
    allocation_text = ", ".join(f"**{a} initial states × {c} continuations**"
                                for a, c in study.allocations)
    lines = ["# Initial-state coverage and structured BSDE statistics", "",
             f"Completed **{len(records)} independently fitted models** for `{study.name}`. Every fit uses "
             f"**{study.training_paths} training trajectories**, with allocations "
             + allocation_text + ". "
             "This changes initial-state coverage and replication together; it does not test the effect of adding trajectories.", "",
             f"Validation is fixed at **{study.validation[0]} independent initial states × {study.validation[1]} continuations**. "
             f"Fresh held-out streams use master seed **{base.seed}**: OU has {base.ou_test_anchors} analytic test states; "
             f"each DK profile has {base.conditional_anchors} fixed conditional anchors × {base.conditional_paths} reference children "
             f"and {base.outer_paths} independent initial-histogram/forcing families for ensemble tests. "
             "All architectures and coverage levels share these banks; optimizer seeds remain separate.", "",
             "The MLP value head is compared with a learned modewise propagation backbone. The structured OU control "
             "has four fitted positive parameters (two rates and two variance scales). Structured DK adds a **0.1-scaled "
             "full-state residual**, including a Cholesky covariance head, to that learned backbone. Neither predictor is "
             "given analytic decay rates or an analytic mean forecast. The martingale network architecture and physical "
             "forward data are held fixed. Initialization supplies the exact value at horizon zero.", "",
             "## Same-step learning curves", "", "![Mean and variance learning curves](study_learning_curves.png)", "",
             "The primary comparison is against the Euler law at the training timestep. Mean errors use sqrt(N) scaling; "
             "variance and covariance errors use N scaling, with fixed uniform-histogram units. Curves show RMSE point "
             "estimates, **not accuracy certification**. Conditional errors and ensemble errors are distinct: averaging "
             "over initial histograms can cancel conditional mean errors.", "",
             "## Errors, uncertainty and gates", "",
             "| Problem/profile | Scope/reference | Architecture | Anchors × children | Seed | Mean RMSE | Variance RMSE | Covariance RMSE | Max mean | Max variance | Max covariance | Max sampling half-width | Max contrast CI endpoint | Gate |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        entries = [f"{row['problem']} {row['case']}", f"{row['scope']}/{row['component']}", row["architecture"],
                   f"{row['train_anchors']}×{row['train_continuations']}", str(row["optimizer_seed"]),
                   *[_number(row[key]) for key in ("mean_rmse", "variance_rmse", "covariance_rmse",
                     "maximum_mean_error", "maximum_variance_error", "maximum_covariance_error",
                     "maximum_sampling_halfwidth", "maximum_ci_endpoint")], row["accuracy_gate"]]
        lines.append("| " + " | ".join(entries) + " |")
    lines.extend(["", f"The requested statistical target is **{base.tolerance:.2%}**, with each uncertainty contribution "
                  f"targeted below **{base.uncertainty_budget:.2%}**. Gates use maximum pointwise errors and intervals, "
                  "not RMSE alone. Fine-reference DK bias remains uncertified by finite timestep audits. A small "
                  "same-step error, or a successful analytic OU control, does not demonstrate convergence to a continuum DK equation.", "",
                  "Same-step neural, fine-total and coarse/fine time contrasts are all retained in the table and JSON. "
                  "Their RMSEs are not additive. Only same-time 4×4 covariance matrices are predicted; positivity does "
                  "not certify accuracy or joint temporal consistency.", "",
                  "## Paired coverage and architecture contrasts", "",
                  "Negative candidate-minus-baseline values favor the candidate. Intervals use common reference draws; "
                  "conditional bootstraps resample whole children within each fixed test anchor, and ensemble bootstraps "
                  "resample whole matched initial-histogram/forcing families. These intervals do not include optimizer "
                  "uncertainty or certify generalization over all possible conditional initial states.", "",
                  "| Problem/profile | Scope | Factor | Baseline → candidate | Seed | Mean RMSE change [95% interval] | Variance RMSE change [95% interval] |",
                  "|---|---|---|---|---:|---:|---:|"])
    for row in contrasts:
        intervals = row.get("paired_uncertainty", {}).get("candidate_minus_baseline_ci95", {})
        label = (f"{row['baseline_architecture']} {row['baseline_train_anchors']} → "
                 f"{row['candidate_architecture']} {row['candidate_train_anchors']}")
        values = [f"{row['problem']} {row['case']}", row["scope"], row["factor"], label,
                  str(row["optimizer_seed"]), *[f"{_number(row['candidate_minus_baseline'][key])} "
                  f"{_interval(intervals.get(key))}" for key in ("mean_rmse", "variance_rmse")]]
        lines.append("| " + " | ".join(values) + " |")
    if not contrasts:
        lines.append("| — | — | Paired intervals unavailable | — | — | — | — |")
    lines.extend(["", "## Stopping and numerical diagnostics", "",
                  f"Numerical calibration: **{calibration.get('status', 'passed')}**. Physical forward and BSDE "
                  "martingale terms share the same full Brownian increments. The mask is active-at-start, including "
                  "the full crossing-step increment; the physical state freezes at its first Euler-polygon crossing. "
                  "Every stopped path remains in statistics. Reference rows below are shared across all fitted models.", "",
                  "| Profile | Scope | Reference | Paths | Stopped | Probability | Zero-stop upper 95% | Max mass error | Min mass |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in shared:
        lines.append("| " + " | ".join([row["case"], row["scope"], row["level"], str(row["n_paths"]),
            str(row["n_stopped"]), _number(row["probability_by_horizon"]),
            _number(row.get("zero_stop_one_sided_95_upper")), _number(row.get("maximum_mass_error")),
            _number(row.get("minimum_saved_mass"))]) + " |")
    lines.extend(["", "## Training and cost", "",
                  "| Problem | Architecture | Anchors × children | Seed | Epochs | Training seconds | Best validation loss |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for record in records:
        factors, evaluation = record["factors"], record["evaluation"]
        training = evaluation["training"]
        lines.append("| " + " | ".join([evaluation["problem"], factors["architecture"],
            f"{factors['train_anchors']}×{factors['train_continuations']}", str(factors["optimizer_seed"]),
            str(training.get("epochs", "—")), _number(training.get("training_seconds")),
            _number(training.get("best_validation_loss"))]) + " |")
    lines.extend(["", "Training terminal residuals include stochastic-integral discretization error and are not value-head "
                  "prediction errors. The JSON retains every fit's residual diagnostics and online latency/throughput "
                  "measurements. Training costs concern distinct fits; reference-generation costs and forward training "
                  "pools are shared and must not be summed across architecture, seed or coverage rows. Coarse and fine "
                  "reference rows describe one coupled audit, not two independent generation bills. Value-only inference "
                  "does not simulate forward trajectories. No speedup is claimed without a matched-accuracy comparison.", "",
                  "## Reproduction", "",
                  f"Study hash: `{study.config_hash}`. Base configuration hash: `{base.config_hash}`. "
                  f"Source hash: `{source_hash()}`. Forward step: `{base.dt:g}`. Fine reference step: `{base.reference_dt:g}`.", "",
                  "The JSON preserves the complete common-bank identities, checkpoint SHA-256 values, data fingerprints, "
                  "individual optimizer seeds, pointwise intervals and costs. The CSV uses one row per model/profile/scope/reference "
                  "comparison. This report rejects missing models, changed checkpoints and incompatible provenance.", ""])
    paths["markdown"].write_text("\n".join(lines))
    return json_ready({key: value.resolve() for key, value in paths.items()})
