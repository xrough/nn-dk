"""Reports for continuous-generator neural moments and deterministic bounds."""

from __future__ import annotations

import csv
import hashlib
from itertools import product
from pathlib import Path

import numpy as np

from .data import json_ready, source_hash, write_json


def _number(value):
    return "—" if value is None else f"{float(value):.5g}"


def _validate(directory, config, records, manifest):
    if (manifest.get("config_hash") != config.config_hash
            or manifest.get("source_hash") != source_hash()):
        raise ValueError("Generator report source/configuration provenance mismatch")
    expected = {("ou", seed, "analytic", "conditional") for seed in config.optimizer_seeds}
    expected |= {("dk", seed, case, scope) for seed, case, scope in product(
        config.optimizer_seeds, config.cases, ("conditional", "ensemble"))}
    if len(records) != len(expected):
        raise ValueError("Generator report requires every configured profile, scope and optimizer seed")
    seen, checkpoints = set(), {}
    for record in records:
        key = tuple(record.get(field) for field in ("problem", "seed", "case", "scope"))
        if key not in expected or key in seen:
            raise ValueError("Duplicate or undeclared generator comparison")
        if record.get("role") != "locked-final":
            raise ValueError("Final generator reports cannot substitute inspected development comparisons")
        seen.add(key)
        path = Path(record.get("checkpoint_path", ""))
        if not path.is_absolute():
            path = directory / path
        if not path.is_file():
            raise ValueError("Generator report requires saved model checkpoints")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if record.get("checkpoint_sha256") != sha:
            raise ValueError("Generator report checkpoint SHA mismatch")
        slot = (record["problem"], record["seed"])
        if checkpoints.setdefault(slot, sha) != sha:
            raise ValueError("A generator fit changes checkpoint across its comparisons")
        if record.get("target_id", config.target_id(record["problem"])) != config.target_id(record["problem"]):
            raise ValueError("Generator report continuous target mismatch")
        comparison = record.get("comparison", {})
        certificate = comparison.get("reference_certificate", {})
        gate = comparison.get("accuracy_gate", {})
        limits = [gate.get("maximum_reference_normalized_bound", np.nan),
                  gate.get("maximum_contrast_normalized_enclosure_endpoint", np.nan)]
        if not np.all(np.isfinite(limits)) or np.any(np.asarray(limits) < 0):
            raise ValueError("Generator report needs finite nonnegative deterministic gate bounds")
        mean_errors = np.asarray(comparison.get("errors", {}).get("mean_normalized_signed", []))
        covariance_errors = np.asarray(comparison.get("errors", {}).get("scaled_covariance_normalized_signed", []))
        if (mean_errors.ndim not in (2, 3)
                or mean_errors.shape[-2:] != (len(config.query_times), 4)
                or covariance_errors.shape != (*mean_errors.shape, 4)
                or not np.all(np.isfinite(mean_errors)) or not np.all(np.isfinite(covariance_errors))):
            raise ValueError("Generator report requires finite errors for every declared horizon and full covariance")
        if not comparison.get("uncertainty", {}).get("method") == "deterministic reference enclosure":
            raise ValueError("Generator report requires deterministic reference enclosures")
        if (gate.get("status") == "resolved_within_tolerance"
                and (certificate.get("valid") is not True
                     or not gate.get("reference_certificate_valid")
                     or not gate.get("reference_bias_resolved")
                     or not gate.get("contrast_within_tolerance")
                     or gate.get("maximum_reference_normalized_bound", float("inf")) > config.uncertainty_budget
                     or gate.get("maximum_contrast_normalized_enclosure_endpoint", float("inf")) > config.tolerance
                     or gate.get("tolerance") != config.tolerance
                     or gate.get("each_uncertainty_budget") != config.uncertainty_budget)):
            raise ValueError("A generator accuracy pass lacks valid bounds or the declared thresholds")
    if seen != expected or len(checkpoints) != config.expected_models:
        raise ValueError("Generator report has an incomplete fit/evaluation allocation")


def _rows(records):
    rows = []
    for record in records:
        comparison = record["comparison"]
        errors, uncertainty, gate = (comparison[key] for key in ("errors", "uncertainty", "accuracy_gate"))
        rows.append({"problem": record["problem"], "case": record["case"], "scope": record["scope"],
                     "optimizer_seed": record["seed"], "target_id": record.get("target_id"),
                     "checkpoint_sha256": record["checkpoint_sha256"],
                     "mean_normalized_rmse": errors["mean_normalized_rmse"],
                     "scaled_variance_normalized_rmse": errors["scaled_variance_normalized_rmse"],
                     "scaled_covariance_normalized_rmse": errors["scaled_covariance_normalized_rmse"],
                     "maximum_mean_normalized_error": errors["maximum_mean_normalized_error"],
                     "maximum_scaled_covariance_normalized_error": errors["maximum_scaled_covariance_normalized_error"],
                     "maximum_reference_normalized_bound": uncertainty["maximum_reference_normalized_bound"],
                     "maximum_mean_normalized_enclosure_endpoint": uncertainty["maximum_mean_normalized_enclosure_endpoint"],
                     "maximum_scaled_covariance_normalized_enclosure_endpoint": uncertainty[
                         "maximum_scaled_covariance_normalized_enclosure_endpoint"],
                     "maximum_contrast_normalized_enclosure_endpoint": uncertainty[
                         "maximum_contrast_normalized_enclosure_endpoint"],
                     "reference_certificate_valid": gate["reference_certificate_valid"],
                     "accuracy_gate": gate["status"], "training_seconds": record.get("training_seconds"),
                     "evaluation_seconds": record.get("evaluation_seconds")})
    return rows


def _plot(rows, config, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    labels = [f"{row['problem'].upper()} {row['case']}\n{row['scope']}, seed {row['optimizer_seed']}" for row in rows]
    for axis, metric, title in zip(axes, (
            "maximum_mean_normalized_enclosure_endpoint",
            "maximum_scaled_covariance_normalized_enclosure_endpoint"),
            ("All mean entries", "All full-covariance entries"), strict=True):
        axis.bar(np.arange(len(rows)), [100 * row[metric] for row in rows],
                 color=["tab:blue" if row["problem"] == "ou" else "tab:green" for row in rows])
        axis.axhline(100 * config.tolerance, color="0.4", linestyle="--", label="5% accuracy target")
        axis.set_xticks(np.arange(len(rows)), labels, rotation=50, ha="right", fontsize=7)
        axis.set(title=title, ylabel="Maximum absolute deterministic contrast bound (%)")
        axis.set_yscale("symlog", linthresh=.1)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Continuous-generator neural moments: pointwise error plus reference enclosure\n"
                    "Optimizer seeds remain separate; these bars are not Monte Carlo confidence intervals")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_generator_report(directory, config, records, manifest):
    """Export all final comparisons with their independent certificate status."""
    import matplotlib

    matplotlib.use("Agg")
    directory = Path(directory)
    _validate(directory, config, records, manifest)
    rows = _rows(records)
    report = directory / "report"
    report.mkdir(parents=True, exist_ok=True)
    paths = {"markdown": report / "generator_results.md", "json": report / "generator_results.json",
             "csv": report / "generator_table.csv", "figure": report / "generator_bounds.png"}
    all_passed = all(row["accuracy_gate"] == "resolved_within_tolerance" for row in rows)
    unique_training = {}
    for record in records:
        unique_training.setdefault((record["problem"], record["seed"]), record.get("training_seconds"))
    total_training = (sum(unique_training.values()) if all(value is not None for value in unique_training.values())
                      else None)
    payload = {"workflow": "continuous-generator-neural-moment-report-v1", "manifest": manifest,
               "config": config.to_dict(), "config_hash": config.config_hash, "source_hash": source_hash(),
               "all_declared_comparisons_passed": all_passed, "comparisons": records,
               "comparison_rows": rows, "training_seconds_distinct_models": total_training,
               "bounds": "Deterministic reference enclosures; no bootstrap or confidence-level interpretation",
               "scope": "Conditional and ensemble means/full same-time covariance; no sample paths or cross-time joint law",
               "model": "Known continuous generator supplies training residuals; time propagators and covariance kernels are neural outputs. Ensemble initialization uses known IID multinomial moments."}
    write_json(paths["json"], payload)
    with paths["csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, config, paths["figure"])
    lines = ["# Continuous-generator neural conditional moments", "",
             f"Completed **{config.expected_models} fitted models** and **{len(rows)} final comparisons**. "
             f"Every declared comparison passed: **{all_passed}**. Optimizer seeds remain separate; no best seed is selected.", "",
             f"The native target is the continuous stopped finite-volume DK system at G={config.n_cells}, "
             f"N={config.n_particles}, D={config.diffusion:g}, T={config.horizon:g}, with density threshold "
             f"{config.rho_stop:g}. Initial laws are IID histograms from uniform and exp(0.2 cos x) profiles. "
             "The OU control is continuous and unstopped. This fixed-grid study does not establish a continuum SPDE limit.", "",
             "A three-hidden-layer SiLU network learns the time propagators and covariance kernels by continuous-generator "
             "residual collocation. Known physics supplies the residual operator, not dynamic forecast labels. State "
             "dependence uses the declared linear/equivariant interior moment family. Initially stopped conditional "
             "states are exact; the difference from future-stopped dynamics needs the separately recorded reference "
             "certificate. Analytic references and known IID multinomial initialization are disclosed separately from "
             "the learned time map. This is not the previous trajectory-BSDE training objective.", "",
             f"Final conditional DK tests use **{config.conditional_anchors} fresh anchors per profile** at "
             f"times {', '.join(str(time) for time in config.query_times)}. The known IID ensemble law is integrated "
             "deterministically rather than estimated with a finite initial-histogram ensemble. Conditional means and "
             "all entries of same-time 4×4 covariance matrices are checked; density paths and cross-time joint "
             "covariances are outside the output contract. Fresh histograms are from the same profile laws, not unseen physics.", "",
             "## Accuracy and deterministic uncertainty", "",
             "| Problem/profile | Scope | Seed | Mean RMSE (%) | Variance RMSE (%) | Covariance RMSE (%) | Max mean bound (%) | Max covariance bound (%) | Max reference bound (%) | Certificate valid | Gate |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        metrics = ("mean_normalized_rmse", "scaled_variance_normalized_rmse", "scaled_covariance_normalized_rmse",
                   "maximum_mean_normalized_enclosure_endpoint", "maximum_scaled_covariance_normalized_enclosure_endpoint",
                   "maximum_reference_normalized_bound")
        lines.append("| " + " | ".join([f"{row['problem']} {row['case']}", row["scope"],
            str(row["optimizer_seed"]), *[_number(100 * row[key]) for key in metrics],
            str(row["reference_certificate_valid"]), row["accuracy_gate"]]) + " |")
    lines.extend(["", "![Pointwise deterministic contrast bounds](generator_bounds.png)", "",
                  f"Passing requires every normalized mean and full-covariance contrast enclosure to lie in "
                  f"[-{100*config.tolerance:g}%, {100*config.tolerance:g}%] and each reference error bound to be at most "
                  f"{100*config.uncertainty_budget:g}%. Sampling uncertainty is zero because references are deterministic. "
                  "A reference packet passes only with a valid certificate and finite nonnegative numerical/stopping "
                  "bounds. Small RMSE, a small collocation residual or invalid bounds cannot certify accuracy.", "",
                  "Means use sqrt(N) fluctuation scaling and covariances use N scaling, divided by fixed uniform-IID "
                  "observable scales. Percentages are errors in these units, not relative density errors. The supplied "
                  "intervals are deterministic enclosures, not bootstrap 95% intervals or population-generalization "
                  "confidence statements. The JSON retains the full bounds, certificate assumptions and every seed.", "",
                  "## Cost and reproduction", "",
                  "| Problem | Seed | Distinct-fit training (s) |",
                  "|---|---:|---:|"])
    for (problem, seed), seconds in unique_training.items():
        lines.append(f"| {problem} | {seed} | {_number(seconds)} |")
    lines.extend(["", f"Total recorded training time for distinct models: **{_number(total_training)} seconds**. "
                  "Comparison-specific evaluation times and checkpoint identities remain in the CSV/JSON. Shared "
                  "reference construction is not charged again to every profile or seed. Value inference requires "
                  "no forward particle/DK trajectories. These timings do not by themselves establish a matched-accuracy speedup.", "",
                  f"Configuration hash: `{config.config_hash}`. Source hash: `{source_hash()}`. "
                  f"Fresh test master seed: `{config.seed}`. Raw run directory: `{directory.resolve()}`.", "",
                  "The final fit lock and all development/final split metadata are retained in the manifest. An "
                  "optimizer cap means completion of its allocated work, not proof of optimization convergence. "
                  "Accuracy is judged by the final declared moment comparisons, independently of that flag.", ""])
    paths["markdown"].write_text("\n".join(lines))
    return json_ready({key: path.resolve() for key, path in paths.items()})
