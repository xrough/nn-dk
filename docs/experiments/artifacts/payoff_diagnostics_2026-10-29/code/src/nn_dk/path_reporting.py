"""Complete density-path reports with numerical and neural errors separated."""

from __future__ import annotations

import csv
import hashlib
from itertools import product
from pathlib import Path

import numpy as np

from .data import source_hash, write_json
from .path_model import PATH_MODEL_KIND
from .path_training import load_path_model

COMPARISONS = {
    "same_step": "neural-versus-same-step-euler",
    "reference_difference": "euler-versus-fine-euler",
    "total": "neural-versus-fine-euler",
}


def _number(value, *, percent=False):
    return "—" if value is None else f"{100 * value if percent else value:.4g}"


def _finite_nonnegative(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value) and value >= 0)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_comparison(comparison, config, case, dt, kind):
    if (comparison.get("comparison_kind") != kind
            or comparison.get("context_id") != config.context(case).context_id
            or comparison.get("target_id") != config.spec.target_id
            or comparison.get("n_independent_trajectory_families") != config.test_paths
            or not np.array_equal(comparison.get("observation_times"), config.query_times)):
        raise ValueError("Path report comparison target, allocation or observation-time mismatch")
    times = np.asarray(comparison.get("times"), dtype=float)
    if (times.shape != (round(config.horizon / dt) + 1,)
            or not np.allclose(times, np.arange(len(times)) * dt, atol=1e-12, rtol=0)):
        raise ValueError("Path report requires every configured stored-time state")
    status, gate = comparison.get("status"), comparison.get("accuracy_gate", {})
    if status not in ("available", "invalid", "unavailable"):
        raise ValueError("Path report comparison has no recognized evaluation status")
    if status == "available":
        strong = comparison.get("strong", {})
        if not _finite_nonnegative(strong.get("pathwise_supremum_rms")):
            raise ValueError("Path report needs the full-state strong supremum metric")
        rms = np.asarray(strong.get("rms_by_time"), dtype=float)
        if rms.shape != times.shape or not np.all(np.isfinite(rms)) or np.any(rms < 0):
            raise ValueError("Path report is missing a full-state time diagnostic")
        for key, dim in (("weak_full_state", config.n_cells), ("weak_observables", 4)):
            weak = comparison.get(key, {})
            q = len(config.query_times)
            for name, shape in (("normalized_mean_signed_error", (q, dim)),
                                ("normalized_joint_covariance_signed_error", (q * dim, q * dim))):
                array = np.asarray(weak.get(name), dtype=float)
                if array.shape != shape or not np.all(np.isfinite(array)):
                    raise ValueError("Path report requires full same-time and two-time weak errors")
        for label in ("prediction", "reference"):
            diagnostics = comparison.get("diagnostics", {}).get(label, {})
            if diagnostics.get("n_allocated_paths") != config.test_paths or diagnostics.get("n_invalid_paths") != 0:
                raise ValueError("Available path statistics cannot exclude failed trajectories")
    if kind != COMPARISONS["same_step"] and gate.get("status") == "resolved_within_tolerance":
        raise ValueError("Euler refinement differences do not certify continuous-time bias")
    if gate.get("status") == "resolved_within_tolerance":
        limits = (gate.get("strong_upper_plus_reference_bound"),
                  gate.get("weak_upper_plus_reference_bound"))
        if (status != "available" or not all(_finite_nonnegative(value) for value in limits)
                or max(limits) > config.tolerance
                or gate.get("research_tolerance") != config.tolerance
                or gate.get("each_uncertainty_reference_budget") != config.uncertainty_budget
                or not all(gate.get(key) is True for key in (
                    "strong_sampling_resolved", "weak_sampling_resolved",
                    "strong_reference_bias_resolved", "weak_reference_bias_resolved",
                    "full_allocation_verified"))
                or comparison.get("coupling", {}).get("physical_noise_verified_by_caller") is not True
                or comparison.get("coupling", {}).get("same_initial_states") is not True):
            raise ValueError("A density-path accuracy pass lacks complete coupling, uncertainty or bias evidence")
        bias = gate.get("reference_bias", {})
        strong_ci = np.asarray(comparison["strong"].get("uncertainty", {}).get(
            "pathwise_supremum_rms_ci95"), dtype=float)
        if (bias != {"status": "resolved", "strong_normalized_bound": 0., "weak_normalized_bound": 0.}
                or strong_ci.shape != (2,) or not np.all(np.isfinite(strong_ci))
                or strong_ci[0] < 0 or strong_ci[1] < strong_ci[0]
                or (strong_ci[1] - strong_ci[0]) / 2 > config.uncertainty_budget):
            raise ValueError("Same-step numerical accuracy needs its declared zero-bias target and valid intervals")
        strong_upper = max(float(strong_ci[1]), comparison["strong"]["pathwise_supremum_rms"])
        weak_upper = 0.
        for key, dim in (("weak_full_state", config.n_cells), ("weak_observables", 4)):
            weak = comparison[key]
            q = len(config.query_times)
            for name, shape in (("mean_normalized_ci95", (q, dim, 2)),
                                ("joint_covariance_normalized_ci95", (q * dim, q * dim, 2))):
                interval = np.asarray(weak.get("uncertainty", {}).get(name), dtype=float)
                if (interval.shape != shape or not np.all(np.isfinite(interval))
                        or np.any(interval[..., 1] < interval[..., 0])
                        or np.max(np.diff(interval, axis=-1) / 2) > config.uncertainty_budget):
                    raise ValueError("A weak accuracy pass needs every pointwise interval within its sampling budget")
                weak_upper = max(weak_upper, float(np.max(np.abs(interval))))
            weak_upper = max(weak_upper, float(np.max(np.abs(weak["normalized_mean_signed_error"]))),
                             float(np.max(np.abs(weak["normalized_joint_covariance_signed_error"]))))
        if (not np.isclose(limits[0], strong_upper, atol=1e-12, rtol=0)
                or not np.isclose(limits[1], weak_upper, atol=1e-12, rtol=0)):
            raise ValueError("Reported accuracy bounds disagree with the complete stored pointwise errors")


def _validate(directory, config, records, manifest):
    if manifest.get("config_hash") != config.config_hash or manifest.get("source_hash") != source_hash():
        raise ValueError("Path report source/configuration provenance mismatch")
    expected = set(product(config.cases, config.step_sizes, config.optimizer_seeds))
    if len(records) != len(expected):
        raise ValueError("Path report needs every configured case, timestep and optimizer seed")
    lock = manifest.get("fit_lock", {})
    locked = lock.get("models", {})
    entries = list(locked.values()) if isinstance(locked, dict) else list(locked)
    locked_shas = [item.get("checkpoint_sha256") for item in entries if isinstance(item, dict)]
    if (lock.get("expected_models") != config.expected_models
            or len(entries) != config.expected_models or len(locked_shas) != config.expected_models
            or not _finite_nonnegative(manifest.get("test_unlocked_unix"))
            or manifest.get("test_unlocked_unix") == 0):
        raise ValueError("Path report requires all fitted checkpoints locked before final inputs were unlocked")
    seen, actual_shas = set(), []
    for record in records:
        key = tuple(record.get(field) for field in ("case", "dt", "seed"))
        if key not in expected or key in seen or record.get("role") != "locked-final":
            raise ValueError("Duplicate, undeclared or development path comparison")
        seen.add(key)
        case, dt, seed = key
        if (record.get("problem") != config.problem(case)
                or record.get("target_id") != config.target_id(case, dt)):
            raise ValueError("Path report neural numerical target mismatch")
        checkpoint = Path(record.get("checkpoint_path", ""))
        if not checkpoint.is_absolute():
            checkpoint = directory / checkpoint
        if not checkpoint.is_file() or record.get("checkpoint_sha256") != _sha(checkpoint):
            raise ValueError("Path report checkpoint bytes differ from the declared content hash")
        prediction = Path(record.get("prediction_path", ""))
        if (not prediction.is_absolute() or not prediction.is_file()
                or record.get("prediction_sha256") != _sha(prediction)):
            raise ValueError("Path report requires the unchanged full density trajectory artifact")
        actual_shas.append(record["checkpoint_sha256"])
        model = load_path_model(checkpoint)
        provenance = model.provenance
        for name, value in (("case", case), ("dt", dt), ("seed", seed),
                            ("problem", config.problem(case)),
                            ("target_id", config.target_id(case, dt)),
                            ("source_hash", source_hash()), ("config_hash", config.config_hash)):
            if provenance.get(name) != value:
                raise ValueError("Path report checkpoint metadata does not match its evaluation slot")
        diagnostics = record.get("prediction_diagnostics", {})
        for name, kind in COMPARISONS.items():
            _validate_comparison(record.get(name, {}), config, case, dt, kind)
        if (record["same_step"]["accuracy_gate"].get("status") == "resolved_within_tolerance"
                and (diagnostics.get("causality") is not True or diagnostics.get("replay") is not True)):
            raise ValueError("A path accuracy pass requires successful causality and physical-noise replay checks")
        for name in ("training_seconds", "evaluation_seconds"):
            if not _finite_nonnegative(record.get(name)):
                raise ValueError("Path report needs nonnegative finite training and evaluation timings")
    if seen != expected or sorted(actual_shas) != sorted(locked_shas):
        raise ValueError("Path report final checkpoints differ from the locked fit inventory")


def _strong_value(comparison):
    return (comparison.get("strong") or {}).get("pathwise_supremum_rms")


def _weak_maximum(comparison, component, *, interval=False):
    values = []
    for key in ("weak_full_state", "weak_observables"):
        weak = comparison.get(key) or {}
        if interval:
            value = weak.get("uncertainty", {}).get(f"maximum_{component}_ci_absolute_endpoint")
        else:
            value = weak.get(f"maximum_{component}_normalized_error")
        if _finite_nonnegative(value):
            values.append(value)
    return max(values) if len(values) == 2 else None


def _rows(config, records):
    order = {slot: index for index, slot in enumerate(product(
        config.cases, config.step_sizes, config.optimizer_seeds))}
    rows = []
    for record in sorted(records, key=lambda item: order[(item["case"], item["dt"], item["seed"])]):
        same, reference, total = (record[key] for key in COMPARISONS)
        strong = same.get("strong") or {}
        interval = strong.get("uncertainty", {}).get("pathwise_supremum_rms_ci95")
        stopping = same.get("stopping") or {}
        diagnostics = same.get("diagnostics", {})
        rows.append({
            "case": record["case"], "problem": record["problem"], "step_size": record["dt"],
            "optimizer_seed": record["seed"], "target_id": record["target_id"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "same_step_strong_supremum_rms": _strong_value(same),
            "same_step_strong_ci95_lower": None if interval is None else interval[0],
            "same_step_strong_ci95_upper": None if interval is None else interval[1],
            "same_step_full_state_and_modes_maximum_mean_error": _weak_maximum(same, "mean"),
            "same_step_full_state_and_modes_maximum_joint_covariance_error": _weak_maximum(same, "joint_covariance"),
            "same_step_mean_maximum_pointwise_ci_endpoint": _weak_maximum(same, "mean", interval=True),
            "same_step_covariance_maximum_pointwise_ci_endpoint": _weak_maximum(same, "joint_covariance", interval=True),
            "euler_fine_strong_supremum_rms": _strong_value(reference),
            "neural_fine_strong_supremum_rms": _strong_value(total),
            "same_step_accuracy_gate": same.get("accuracy_gate", {}).get("status"),
            "continuous_reference_accuracy_gate": total.get("accuracy_gate", {}).get("status"),
            "prediction_stopped_paths": stopping.get("prediction_n_stopped_by_horizon"),
            "same_step_reference_stopped_paths": stopping.get("reference_n_stopped_by_horizon"),
            "prediction_invalid_paths": diagnostics.get("prediction", {}).get("n_invalid_paths"),
            "reference_invalid_paths": diagnostics.get("reference", {}).get("n_invalid_paths"),
            "maximum_mass_error": diagnostics.get("prediction", {}).get("maximum_mass_error"),
            "minimum_mass": diagnostics.get("prediction", {}).get("minimum_mass"),
            "causality_passed": record.get("prediction_diagnostics", {}).get("causality"),
            "same_noise_replay_passed": record.get("prediction_diagnostics", {}).get("replay"),
            "training_seconds": record["training_seconds"],
            "evaluation_seconds": record["evaluation_seconds"],
        })
    return rows


def _plot(config, records, path, *, weak=False):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(config.cases), 2, figsize=(14, 3.2 * len(config.cases)),
                                squeeze=False, constrained_layout=True)
    colors = {seed: plt.get_cmap("tab10")(i) for i, seed in enumerate(config.optimizer_seeds)}
    for row, case in enumerate(config.cases):
        for seed in config.optimizer_seeds:
            group = sorted((r for r in records if r["case"] == case and r["seed"] == seed),
                           key=lambda record: record["dt"])
            for column in range(2):
                axis = axes[row, column]
                if weak:
                    component = "mean" if column == 0 else "joint_covariance"
                    values = [_weak_maximum(record["same_step"], component, interval=True) for record in group]
                    points = [(record["dt"], 100 * value) for record, value in zip(group, values, strict=True)
                              if value is not None]
                    if points:
                        axis.plot(*zip(*points, strict=True), marker="o", color=colors[seed],
                                  label=f"seed {seed}")
                else:
                    comparisons = (("same_step", "NN − same-step Euler", "o", "-"),) if column == 0 else (
                        ("reference_difference", "Euler − fine Euler", "x", ":"),
                        ("total", "NN − fine Euler", "s", "--"))
                    for name, label, marker, style in comparisons:
                        points = [(record["dt"], 100 * _strong_value(record[name])) for record in group
                                  if _strong_value(record[name]) is not None]
                        if points:
                            axis.plot(*zip(*points, strict=True), marker=marker, linestyle=style,
                                      color=colors[seed], label=f"{label}; seed {seed}")
        for column, axis in enumerate(axes[row]):
            axis.axhline(100 * config.tolerance, color="0.5", linestyle="--", linewidth=.8,
                         label=f"{100 * config.tolerance:g}% research target")
            axis.set_xscale("log", base=2)
            axis.set_yscale("symlog", linthresh=.1)
            axis.set_ylim(bottom=0)
            title = ("Mean" if column == 0 else "Full joint covariance") if weak else (
                "Neural approximation at the same step" if column == 0 else "Time difference and total error")
            axis.set(xlabel="Integration/input step Δ")
            axis.set_title(f"{case}: {title}", fontsize=10)
            axis.set_ylabel("Maximum paired CI endpoint (%)\n(pointwise; all cells and modes)"
                            if weak else "Full-state path supremum RMS\n(% of histogram fluctuation units)",
                            fontsize=9)
            axis.grid(alpha=.2)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=6.5)
    figure.suptitle(
        "All-cell and mode weak contrasts: pointwise 95% bootstrap endpoints, not simultaneous bands"
        if weak else "Full density paths: fixed uniform-histogram fluctuation units; optimizer seeds separate")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_path_report(directory, config, records, manifest):
    """Write complete final results; unresolved and failed ensembles stay visible."""
    import matplotlib

    matplotlib.use("Agg")
    directory = Path(directory)
    _validate(directory, config, records, manifest)
    rows = _rows(config, records)
    report = directory / "report"
    report.mkdir(parents=True, exist_ok=True)
    paths = {"markdown": report / "path_results.md", "json": report / "path_results.json",
             "csv": report / "path_table.csv", "figure": report / "path_strong_errors.png",
             "weak_figure": report / "path_weak_errors.png"}
    n_passed = sum(row["same_step_accuracy_gate"] == "resolved_within_tolerance" for row in rows)
    payload = {
        "workflow": "causal-density-path-report-v1", "model_kind": PATH_MODEL_KIND,
        "source_hash": source_hash(), "config_hash": config.config_hash,
        "config": config.to_dict(), "manifest": manifest, "comparisons": records,
        "comparison_rows": rows, "all_configured_models_reported": True,
        "all_same_step_comparisons_passed": n_passed == len(rows),
        "same_step_comparisons_passed": n_passed, "continuous_time_convergence_certified": False,
        "training_seconds_distinct_models": sum(row["training_seconds"] for row in rows),
        "evaluation_seconds_recorded": sum(row["evaluation_seconds"] for row in rows),
        "scope": "Complete density paths; shared-noise comparison to declared finite-volume Euler targets",
        "reference_limitation": "Fine Euler differences do not bound continuous-time stopping bias or continuum DK error",
        "uncertainty": "Pointwise paired whole-family percentile bootstrap; no simultaneous bands or optimizer-seed averaging",
    }
    write_json(paths["json"], payload)
    with paths["csv"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plot(config, records, paths["figure"])
    _plot(config, records, paths["weak_figure"], weak=True)
    lines = ["# Causal neural density-path experiments", "",
             f"Reported **{config.expected_models} configured fitted models**, with "
             f"**{n_passed}/{len(rows)} same-step comparisons within the declared accuracy gate**.", "",
             "The network generates full finite-volume density trajectories from initial cell masses and "
             "the physical Brownian path. Its weights remain fixed during each realization. "
             "All real spatial modes, including Nyquist, are retained; this is not a moments-only predictor.", "",
             "The known finite-volume drift and face-noise mobility enter a causal Volterra/Picard ansatz. "
             "Training minimizes integrated and local left-endpoint Itô residuals. Optimizer/minibatch randomness "
             "is separate from the supplied physical forcing. The learned kernel depends on its input timestep; "
             "different resolutions are fitted separately.", "",
             f"The target has G={config.n_cells}, N={config.n_particles}, D={config.diffusion:g}, "
             f"T={config.horizon:g}; the fine diagnostic step is δ={config.reference_step:g}. "
             "DK paths stop at the first boundary crossing of their piecewise-linear numerical polygon and freeze. "
             "Every stopped path is retained, and an ensemble with any failed path produces no survivor average.", "",
             "Same-step reference bias is zero only relative to the declared Euler numerical target. "
             "The Euler-to-fine-Euler difference is shown separately; it is not a bound on continuous-time "
             "boundary-hitting bias. **Continuous-time stopped-DK convergence and continuum-SPDE convergence "
             "remain uncertified.** No temporal order is fitted here.", "",
             "Strong errors use all cells and the maximum over all stored times, normalized by the exact "
             "uniform IID histogram density-L2 fluctuation energy (1−1/G)/(N h). "
             "The 5% default is a research accuracy target. Weak gates include all-cell means/full joint "
             "same-time and two-time covariances, plus cell-integrated sine/cosine modes 1 and 2. "
             "Confidence intervals are pointwise paired whole-trajectory bootstrap intervals, not simultaneous "
             "bands or certificates of each marginal-law Monte Carlo precision. Optimizer seeds stay separate; "
             "no best seed is selected.", "",
             "| Case | Δ | Seed | Strong RMS (%) | 95% interval (%) | Max mean CI (%) | Max joint-cov CI (%) | Euler/fine RMS (%) | NN/fine RMS (%) | Stops NN/Euler | Invalid NN/Euler | Same-step gate |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        interval = ("—" if row["same_step_strong_ci95_lower"] is None else
                    f"[{_number(row['same_step_strong_ci95_lower'], percent=True)}, "
                    f"{_number(row['same_step_strong_ci95_upper'], percent=True)}]")
        lines.append(
            f"| {row['case']} | {row['step_size']:.6g} | {row['optimizer_seed']} | "
            f"{_number(row['same_step_strong_supremum_rms'], percent=True)} | {interval} | "
            f"{_number(row['same_step_mean_maximum_pointwise_ci_endpoint'], percent=True)} | "
            f"{_number(row['same_step_covariance_maximum_pointwise_ci_endpoint'], percent=True)} | "
            f"{_number(row['euler_fine_strong_supremum_rms'], percent=True)} | "
            f"{_number(row['neural_fine_strong_supremum_rms'], percent=True)} | "
            f"{row['prediction_stopped_paths']}/{row['same_step_reference_stopped_paths']} | "
            f"{row['prediction_invalid_paths']}/{row['reference_invalid_paths']} | {row['same_step_accuracy_gate']} |")
    lines += ["", "Zero observed stops does not certify a zero stopping probability. Finite-mesh realized "
              "quadratic variation is descriptive; noise-normalization checks use uncensored active-at-start "
              "proposals, including exit proposals, rather than fractional absorbed increments.", "",
              "![Full-state strong errors](path_strong_errors.png)", "",
              "![Full-state and mode weak contrasts](path_weak_errors.png)", "",
              f"Recorded training cost: **{payload['training_seconds_distinct_models']:.6g} s** across distinct models. "
              f"Recorded evaluation cost: **{payload['evaluation_seconds_recorded']:.6g} s**. "
              "These sums do not replace separate data-generation, initialization or latency/throughput measurements.", "",
              f"Source hash: `{source_hash()}`. Configuration hash: `{config.config_hash}`. "
              "Checkpoint content hashes and locked-final provenance are retained in the JSON and CSV.", ""]
    paths["markdown"].write_text("\n".join(lines))
    return {name: str(path.resolve()) for name, path in paths.items()}
