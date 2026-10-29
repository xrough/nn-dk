#!/usr/bin/env python3
"""Plot all-seed conditional/ensemble DK errors from a completed BSDE run.

All measurements use the fine DK reference and fixed uniform-IID fluctuation
normalization. The plots show point estimates, without uncertainty certification.
No network or forward solver is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn_dk.bsde_config import BSDEConfig
from nn_dk.data import source_hash, write_json


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_findings(run_directory):
    directory = Path(run_directory).resolve()
    config_path, manifest_path = directory / "config.json", directory / "manifest.json"
    evaluations_path = directory / "evaluations.json"
    config = BSDEConfig.load(config_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Complete the scientific allocation before plotting final findings")
    if manifest.get("config_hash") != config.config_hash or manifest.get("source_hash") != source_hash():
        raise ValueError("Completed experiment source/configuration differs from the plotting environment")
    evaluations = json.loads(evaluations_path.read_text())
    records = {}
    expected = set(config.optimizer_seeds)
    for record in evaluations:
        if record.get("problem") != "dk":
            continue
        seed = record["seed"]
        if seed in records or seed not in expected or not record.get("complete"):
            raise ValueError("Duplicate, incomplete, or unallocated DK record")
        if record.get("source_hash") != manifest["source_hash"] or record.get("config_hash") != config.config_hash:
            raise ValueError("DK evaluation source/configuration mismatch")
        checkpoint = directory / "checkpoints" / "dk" / f"seed_{seed}.pt"
        if record.get("model", {}).get("checkpoint_sha256") != _sha(checkpoint):
            raise ValueError("DK evaluation checkpoint SHA mismatch")
        if set(record["cases"]) != set(config.cases):
            raise ValueError("DK profile allocation is incomplete")
        records[seed] = record
    if set(records) != expected:
        raise ValueError("Every configured DK optimizer seed must be complete")
    table = []
    for case in config.cases:
        for bank in ("conditional", "ensemble"):
            for seed in config.optimizer_seeds:
                comparison = records[seed]["cases"][case][f"{bank}_total_error"]
                errors = comparison["errors"]
                table.append({"case": case, "bank": bank, "optimizer_seed": seed,
                    "mean_rmse_percent": 100 * errors["mean_normalized_rmse"],
                    "variance_rmse_percent": 100 * errors["scaled_variance_normalized_rmse"],
                    "maximum_mean_error_percent": 100 * errors["maximum_mean_normalized_error"],
                    "maximum_covariance_error_percent": 100 * errors["maximum_scaled_covariance_normalized_error"],
                    "accuracy_gate": comparison["accuracy_gate"]})
    panels = (("mean_rmse_percent", "Mean RMSE"),
              ("variance_rmse_percent", "Variance RMSE"),
              ("maximum_mean_error_percent", "Maximum pointwise mean error"),
              ("maximum_covariance_error_percent", "Maximum pointwise covariance error"))
    groups = [(case, bank) for case in config.cases for bank in ("conditional", "ensemble")]
    offsets = np.linspace(-.15, .15, len(config.optimizer_seeds))
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^")
    with plt.rc_context({"font.size": 11, "axes.titlesize": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"}):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for panel_number, (axis, (field, title)) in enumerate(zip(axes.flat, panels, strict=True)):
            for seed_index, seed in enumerate(config.optimizer_seeds):
                values = [next(row[field] for row in table
                               if (row["case"], row["bank"], row["optimizer_seed"]) == (case, bank, seed))
                          for case, bank in groups]
                positions = np.arange(len(groups)) + offsets[seed_index]
                for start in range(0, len(groups), 2):
                    axis.plot(positions[start:start+2], values[start:start+2],
                              color=colors[seed_index], alpha=.3, linewidth=1)
                axis.scatter(positions, values, color=colors[seed_index],
                             marker=markers[seed_index % len(markers)], s=55,
                             label=f"Seed {seed}", zorder=3)
            if panel_number >= 2:
                axis.axhline(100 * config.tolerance, color="#777777", linestyle="--", linewidth=1)
            axis.set_title(title, loc="left", weight="bold")
            axis.set_ylabel("Normalized error (%)")
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", color="#dddddd", linewidth=.7, zorder=0)
            for separator in np.arange(1.5, len(groups)-1, 2):
                axis.axvline(separator, color="#dddddd", linewidth=.8)
            axis.set_xticks(np.arange(len(groups)),
                            [f"{case.capitalize()}\n{bank}" for case, bank in groups])
            axis.tick_params(axis="x", length=0)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.suptitle("BSDE: conditional and ensemble predictions", x=.07, ha="left", fontsize=17, weight="bold")
        figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(.98, .975),
                      ncol=3, frameon=False, fontsize=10)
        figure.text(.07, .912, "Fine DK reference · All optimizer seeds · Fixed uniform-IID fluctuation units",
                    fontsize=11, color="#444444")
        figure.text(.07, .035,
                    "Point estimates only. Dashed 5% lines apply to pointwise errors; acceptance also requires uncertainty and reference-bias control.\n"
                    "RMSE averages can conceal individual failures. Covariance RMSE is replaced here by the more revealing variance RMSE.",
                    fontsize=9.5, color="#444444", va="bottom")
        figure.subplots_adjust(left=.08, right=.98, bottom=.145, top=.87, hspace=.28, wspace=.24)
        output = directory / "report" / "conditional_vs_ensemble.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
        plt.close(figure)
    provenance = {"figure": output.name, "figure_sha256": _sha(output),
                  "analysis_sha256": _sha(Path(__file__)), "source_hash": manifest["source_hash"],
                  "config_hash": config.config_hash, "config_file_sha256": _sha(config_path),
                  "manifest_sha256": _sha(manifest_path), "evaluations_sha256": _sha(evaluations_path),
                  "reference": "Fine stopped DK reference at the configured reference_dt; total errors",
                  "normalization": "sqrt(N) mean differences / uniform-IID observable std; N covariance differences / products of those stds",
                  "interpretation": "Point estimates only. No fitting, simulation, selection of optimizer seeds, or RMSE-based certification.",
                  "checkpoints": {str(seed): record["model"]["checkpoint_sha256"]
                                  for seed, record in records.items()}, "table": table}
    write_json(output.with_suffix(".json"), provenance)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    arguments = parser.parse_args()
    print(plot_findings(arguments.run_directory))


if __name__ == "__main__":
    main()
