"""Report a completed interacting follow-up without pooling optimizer seeds.

Run only after the scientific batch, task comparison, and separate quiet-process
timing have completed. This script reads saved evidence and writes no solver
states, fits, reference paths, or scientific acceptance decisions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _signed_json(path):
    value = _read(path)
    body = {key: item for key, item in value.items() if key != "content_hash"}
    if value.get("content_hash") != _stable(body):
        raise ValueError(f"Saved result content hash changed: {path}")
    return value


def _write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def _csv(path, rows):
    if not rows:
        raise ValueError(f"No completed rows available for {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _csv_read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _profile(case):
    return {"dk_interacting_uniform": "uniform", "dk_interacting": "nonuniform"}.get(case, case)


def load_evidence(run):
    run = Path(run).resolve()
    required = [run/"study_config.json", run/"execution"/"frozen.json",
                run/"execution"/"completed.json", run/"paths"/"report"/"path_table.csv",
                run/"paths"/"evaluations.json", run/"comparison"/"comparison.json",
                run/"comparison"/"comparison.csv", run/"comparison"/"preregistration.json",
                run/"comparison"/"task_timing.json", run/"execution"/"refinement_analysis_plan.json",
                run/"paths"/"path_test.h5", run/"paths"/"manifest.json",
                run/"execution"/"fit_lock.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Report requires completed scientific and timing artifacts; missing: "+", ".join(missing))
    config, frozen, completed = (_read(path) for path in required[:3])
    if (completed.get("complete") is not True or completed.get("identity") != frozen
            or _sha(required[0]) != frozen.get("configuration_sha256")):
        raise ValueError("Batch completion and frozen configuration do not match")
    fit_lock = _read(run/"execution"/"fit_lock.json")
    if fit_lock.get("identity") != frozen or len(fit_lock.get("models", [])) != frozen["expected_models"]:
        raise ValueError("Completed global checkpoint inventory differs from its frozen allocation")
    locked_metadata = {Path(item["metadata_path"]).resolve(): item["metadata_sha256"]
                       for item in fit_lock["models"]}
    metadata = {}
    for candidate in ("unscaled", "balanced"):
        metadata[candidate] = []
        for seed in config[candidate]["optimizer_seeds"]:
            path = run/candidate/"checkpoints"/f"moments_seed{seed}.json"
            if not path.is_file():
                raise ValueError(f"Missing declared seed checkpoint metadata: {path}")
            info = _read(path)
            if (info.get("complete") is not True or info.get("seed") != seed
                    or info.get("source_hash") != frozen["source_hash"]):
                raise ValueError("Incomplete or incompatible statistics checkpoint metadata")
            if locked_metadata.get(path.resolve()) != _sha(path):
                raise ValueError("Locked statistics training metadata changed before reporting")
            metadata[candidate].append(info)
            required.append(path)
    comparison = _signed_json(run/"comparison"/"comparison.json")
    timing = _signed_json(run/"comparison"/"task_timing.json")
    design = _read(run/"comparison"/"preregistration.json")
    if (comparison["source_hash"] != frozen["source_hash"]
            or timing["source_hash"] != frozen["source_hash"]
            or comparison["design_hash"] != _stable(design)
            or timing["design_hash"] != comparison["design_hash"]
            or timing["comparison_sha256"] != _sha(run/"comparison"/"comparison.json")
            or timing.get("optimizer_seed") != 0
            or timing.get("physical_accuracy_certified") is not False):
        raise ValueError("Comparison, timing, or source provenance differs from the completed batch")
    if comparison["artifact_lock"].get("comparison.csv") != _sha(run/"comparison"/"comparison.csv"):
        raise ValueError("Completed comparison CSV changed")
    path_rows = _csv_read(run/"paths"/"report"/"path_table.csv")
    path_manifest = _read(run/"paths"/"manifest.json")
    if (path_manifest.get("status") != "complete"
            or path_manifest["artifact_lock"].get("evaluations.json") != _sha(run/"paths"/"evaluations.json")):
        raise ValueError("Completed path evaluation artifact changed before reporting")
    expected = {(case, float(dt), seed) for case in config["paths"]["cases"]
                for dt in config["paths"]["step_sizes"] for seed in config["paths"]["optimizer_seeds"]}
    actual = {(row["case"], float(row["step_size"]), int(row["optimizer_seed"])) for row in path_rows}
    if len(path_rows) != len(expected) or actual != expected:
        raise ValueError("Path table lacks a declared profile, time step, or optimizer seed")
    path_evaluations = {(item["case"], item["dt"], item["seed"]): item
                        for item in _read(run/"paths"/"evaluations.json")}
    for row in path_rows:
        record = path_evaluations[row["case"], float(row["step_size"]), int(row["optimizer_seed"])]
        for name, key in (("same_step_strong_supremum_rms", "same_step"),
                          ("neural_fine_strong_supremum_rms", "total"),
                          ("euler_fine_strong_supremum_rms", "reference_difference")):
            value = record[key]["strong"]["pathwise_supremum_rms"]
            if not _finite(value) or float(row[name]) != value:
                raise ValueError("Path table and locked full-state evaluations differ")
    if not comparison.get("records") or not timing.get("timings"):
        raise ValueError("Completed comparison and timing require every saved task record")
    return dict(run=run, config=config, frozen=frozen, comparison=comparison, timing=timing,
                metadata=metadata, paths=path_rows,
                input_sha256={str(path.relative_to(run)): _sha(path) for path in required})


def common_grid_refinement(evidence):
    """Replay existing forcing only; compare every solver on identical times."""
    import h5py
    import torch

    from nn_dk.data import source_hash
    from nn_dk.path_config import PathConfig
    from nn_dk.path_numerics import torch_euler

    if source_hash() != evidence["frozen"]["source_hash"]:
        raise ValueError("Deterministic Euler replay requires the original frozen package source")
    run = evidence["run"]
    config = PathConfig(**evidence["config"]["paths"])
    plan = _read(run/"execution"/"refinement_analysis_plan.json")
    common_step = plan["primary_common_time_step"]
    if common_step != max(config.step_sizes) or plan["bootstrap_families"] < 20:
        raise ValueError("Common-grid refinement must match its preregistered physical times")
    base_step = min(config.step_sizes)
    base_factor = round(common_step/base_step)
    manifest = _read(run/"paths"/"manifest.json")
    if manifest["artifact_lock"].get("path_test.h5") != _sha(run/"paths"/"path_test.h5"):
        raise ValueError("Locked path-test forcing/reference bank changed")
    evaluations = _read(run/"paths"/"evaluations.json")
    rows, replay_seconds = [], 0.
    torch.set_num_threads(1)
    with h5py.File(run/"paths"/"path_test.h5", "r") as bank:
        for case in config.cases:
            group = bank[case]
            initial, fine_noise = group["initial"][:], group["brownian"][:]
            families = group["family_ids"][:].astype(str)
            reference = group["reference_states"][:, ::base_factor]
            if len(set(families.tolist())) != config.test_paths or not np.all(group["complete"][:]):
                raise ValueError("Common-grid bootstrap requires every independent complete path family")
            namespace = ["interacting-followup-common-physical-time-grid-v1", config.seed, case]
            generator = np.random.default_rng(int(_stable(namespace)[:16], 16))
            indices = generator.integers(0, len(initial), (plan["bootstrap_families"], len(initial)))
            native_by_dt = {}
            for dt in config.step_sizes:
                factor = round(dt/base_step)
                noise = fine_noise.reshape(len(initial), -1, factor, config.n_cells).sum(2)
                tick = time.perf_counter()
                with torch.no_grad():
                    native = torch_euler(initial, noise, config.context(case), config.spec,
                                          dt, config.problem(case))["states"].numpy()
                replay_seconds += time.perf_counter()-tick
                native_by_dt[dt] = native[:, ::round(common_step/dt)]
            for record in (item for item in evaluations if item["case"] == case):
                path = Path(record["prediction_path"])
                if not path.is_absolute():
                    path = run/"paths"/path
                if _sha(path) != record["prediction_sha256"]:
                    raise ValueError("Saved neural trajectory changed before common-grid reporting")
                evidence["input_sha256"][str(path.relative_to(run))] = _sha(path)
                with h5py.File(path, "r") as prediction:
                    stored_families = prediction["family_ids"][:].astype(str)
                    states = prediction["states"][:, ::round(common_step/record["dt"])]
                    times = prediction["times"][::round(common_step/record["dt"])]
                native = native_by_dt[record["dt"]]
                expected_times = np.arange(round(config.horizon/common_step)+1)*common_step
                if (not np.array_equal(stored_families, families) or states.shape != reference.shape
                        or native.shape != reference.shape or not np.array_equal(times, expected_times)
                        or not np.allclose(states[:, 0], initial, atol=1e-12, rtol=0)
                        or not all(np.isfinite(value).all() for value in (states, native, reference))):
                    raise ValueError("Common-time errors require identical families, initial states, times and valid paths")

                def metric(first, second):
                    energy = config.n_particles*np.sum((first-second)**2, axis=-1)/(1-1/config.n_cells)
                    path_energy = energy.max(1)
                    estimate = float(np.sqrt(path_energy.mean()))
                    bootstraps = np.sqrt(path_energy[indices].mean(1))
                    low, high = np.quantile(bootstraps, (.025, .975))
                    return estimate, float(low), float(high), bootstraps

                total, euler, neural = (metric(first, second) for first, second in
                                        ((states, reference), (native, reference), (states, native)))
                row = dict(case=case, profile=_profile(case), step_size=record["dt"],
                           optimizer_seed=record["seed"], common_time_step=common_step,
                           n_common_times=len(expected_times), n_families=len(families),
                           family_ids_hash=_stable(families.tolist()), bootstrap_replicates=len(indices),
                           bootstrap_namespace=_stable(namespace), continuous_accuracy_certified=False)
                for label, value in (("neural_fine", total), ("euler_fine", euler), ("same_step", neural)):
                    row[f"{label}_strong_supremum_rms"] = value[0]
                    row[f"{label}_ci95_lower"], row[f"{label}_ci95_upper"] = value[1:3]
                low, high = np.quantile(total[3]-euler[3], (.025, .975))
                row.update(paired_neural_minus_euler_fine_strong=total[0]-euler[0],
                           paired_difference_ci95_lower=float(low), paired_difference_ci95_upper=float(high))
                rows.append(row)
    return rows, dict(plan=plan, native_replay_seconds=replay_seconds,
                      replay_scope="deterministic replay of saved initial states/face increments; no new physical random draws; reporting cost excluded from inference timing")


def residual_rows(evidence):
    rows = []
    for candidate, infos in evidence["metadata"].items():
        for info in infos:
            points = [("initial", 0, info["initial_validation"])]
            points += [("history", item["step"], item["validation"]) for item in info["history"]
                       if isinstance(item.get("validation"), dict)]
            points += [("selected", info["best_step"], info["best_validation"])]
            for label, step, value in points:
                rows.append(dict(candidate=candidate, optimizer_seed=info["seed"], point=label,
                    optimizer_step=step, raw_mean_rms=value["mean_raw_rms"],
                    raw_covariance_rms=value["covariance_raw_rms"],
                    raw_mean_maximum=value["mean_raw_maximum"],
                    raw_covariance_maximum=value["covariance_raw_maximum"],
                    mean_normalized_mse=value["mean_normalized_mse"],
                    covariance_normalized_mse=value["covariance_normalized_mse"],
                    selected_objective=value["objective"], raw_objective=value["raw_objective"],
                    loss_balancing=info["loss_scaling"]["mode"],
                    frozen_mean_scale=info["loss_scaling"]["mean_scale"],
                    frozen_covariance_scale=info["loss_scaling"]["covariance_scale"],
                    training_seconds=info["training_seconds"],
                    objective_comparable_between_candidates=False))
    return rows


def _timing_matches(record, packet):
    if (record["case"] != packet["case"] or record["role"] != packet["role"]
            or record.get("anchor_id") not in (0, -1)):
        return False
    count = packet["simulation_or_initial_mc_paths"]
    if record["method"] == "statistics_nn":
        return (record["optimizer_seed"] == packet["moment_seed"]
                and record["candidate"] == packet["moment_candidate"]
                and (record["role"] == "conditional" or record["initial_mc_paths"] == count))
    return (record["dt"] == packet["dt"] and record["candidate_paths"] == count
            and (record["method"] == "native_euler" or record["optimizer_seed"] == packet["path_seed"]))


def _training_cost(record, comparison):
    values = [item["training_seconds"] for item in comparison["fitting_costs"]
              if item["method"] == record["method"]
              and item.get("optimizer_seed") == record.get("optimizer_seed")
              and (record["method"] != "statistics_nn" or item.get("candidate") == record.get("candidate"))
              and (record["method"] != "density_path_nn" or
                   (item.get("case"), item.get("dt")) == (record["case"], record.get("dt")))]
    return values[0] if len(values) == 1 else 0. if record["method"] == "native_euler" else None


def task_rows(evidence):
    rows = []
    timing = evidence["timing"]
    generations = {(item["case"], item["role"], item["paths"], item["dt"]): item
                   for item in timing["input_generation_timings"]}
    for record in evidence["comparison"]["records"]:
        packet = record["comparison"]
        base = {key: record.get(key) for key in ("case", "role", "anchor_id", "method", "candidate",
                                                "optimizer_seed", "dt", "candidate_paths", "initial_mc_paths")}
        base.update(maximum_mean_error=packet["mean_error"]["maximum_absolute_point"],
            maximum_covariance_error=packet["covariance_error"]["maximum_absolute_point"],
            maximum_mean_pointwise_interval_endpoint=packet["mean_error"]["maximum_absolute_interval_endpoint"],
            maximum_covariance_pointwise_interval_endpoint=packet["covariance_error"]["maximum_absolute_interval_endpoint"],
            reference_mean_halfwidth=packet["reference_uncertainty"]["mean"]["maximum_halfwidth"],
            reference_covariance_halfwidth=packet["reference_uncertainty"]["covariance"]["maximum_halfwidth"],
            candidate_mean_halfwidth=packet["candidate_uncertainty"]["mean"]["maximum_halfwidth"],
            candidate_covariance_halfwidth=packet["candidate_uncertainty"]["covariance"]["maximum_halfwidth"],
            training_seconds=_training_cost(record, evidence["comparison"]),
            physical_accuracy_certified=False, certified_break_even_queries=None,
            break_even_reason="Continuous reference bias and matched physical accuracy remain unresolved")
        matches = [item for item in timing["timings"] if _timing_matches(record, item)]
        for match in matches or (None,):
            row = base.copy()
            if match is None:
                row.update(timing_measured=False, timing_optimizer_seed=None)
            else:
                measured = match["timings"][record["method"]]
                count = match["simulation_or_initial_mc_paths"]
                generated = generations[record["case"], record["role"], count, match["dt"]]
                statistics = record["method"] == "statistics_nn"
                initial = (0. if statistics and record["role"] == "conditional" else
                           generated["initialization"]["median_seconds"])
                noise = 0. if statistics else generated["physical_noise_generation"]["median_seconds"]
                row.update(timing_measured=True, timing_optimizer_seed=0, timing_dt=match["dt"],
                    timing_simulation_or_initial_mc_paths=count,
                    timing_statistics_candidate=match["moment_candidate"],
                    kernel_median_seconds=measured["median_seconds"],
                    kernel_p05_seconds=measured["p05_seconds"], kernel_p95_seconds=measured["p95_seconds"],
                    initialization_median_seconds=initial, physical_noise_generation_median_seconds=noise,
                    online_sum_component_medians_estimate_seconds=measured["median_seconds"]+initial+noise,
                    online_cost_scope="warm resident model; identical mean/full-same-time-covariance output; excludes restoration/training/bootstrap/I/O",
                    timing_reference_scope="conditional anchor 0; optimizer seed 0; fixed-count architecture timing")
            rows.append(row)
    return rows


def _save(fig, output, name):
    fig.savefig(output/name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _count_ticks(axis, counts):
    axis.set_xticks(counts, [str(value) for value in counts])
    axis.xaxis.set_minor_locator(NullLocator())


def plot_paths(evidence, output, *, name="path_refinement.png", common=True):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for index, case in enumerate(evidence["config"]["paths"]["cases"]):
        rows = [item for item in evidence["paths"] if item["case"] == case]
        for seed in evidence["config"]["paths"]["optimizer_seeds"]:
            selected = sorted((item for item in rows if int(item["optimizer_seed"]) == seed),
                              key=lambda item: float(item["step_size"]))
            dt = [float(item["step_size"]) for item in selected]
            axes[index, 0].loglog(dt, [100*float(item["same_step_strong_supremum_rms"]) for item in selected],
                                 "o-", label=f"NN, seed {seed}")
            axes[index, 1].loglog(dt, [100*float(item["neural_fine_strong_supremum_rms"]) for item in selected],
                                 "o-", label=f"NN, seed {seed}")
            if seed == 0:
                axes[index, 1].loglog(dt, [100*float(item["euler_fine_strong_supremum_rms"]) for item in selected],
                                     "k--", label="Native Euler, same inputs")
        for column in range(2):
            axes[index, column].set(xlabel="Integration step", ylabel="Normalized full-path supremum RMS (%)",
                                    title=f"{_profile(case)}: "+("NN vs same-step Euler" if column == 0 else "vs fine Euler"))
            axes[index, column].grid(alpha=.2)
            steps = sorted(evidence["config"]["paths"]["step_sizes"])
            axes[index, column].set_xticks(steps, [f"{step:g}" for step in steps])
            axes[index, column].xaxis.set_minor_locator(NullLocator())
            axes[index, column].legend(fontsize=8)
    fig.suptitle(("Primary: common physical-time grid" if common else
                  "Secondary: each solver's own stored-time grid")+
                 "; each optimizer seed remains separate")
    _save(fig, output, name)


def plot_residuals(evidence, rows, output):
    seeds = evidence["config"]["balanced"]["optimizer_seeds"]
    fig, axes = plt.subplots(2, len(seeds), figsize=(4.3*len(seeds), 6.5), squeeze=False, constrained_layout=True)
    for column, seed in enumerate(seeds):
        for candidate in ("unscaled", "balanced"):
            selected = [item for item in rows if item["optimizer_seed"] == seed
                        and item["candidate"] == candidate and item["point"] != "selected"]
            for index, key in enumerate(("raw_mean_rms", "raw_covariance_rms")):
                axes[index, column].semilogy([item["optimizer_step"] for item in selected],
                                             [item[key] for item in selected], label=candidate)
                axes[index, column].set(xlabel="Optimizer outer step", ylabel=key.replace("_", " "),
                                        title=f"Seed {seed}")
                axes[index, column].grid(alpha=.2)
                axes[index, column].legend(fontsize=8)
    fig.suptitle("Independent validation residuals in common raw units; objectives use different scales")
    _save(fig, output, "residual_histories.png")


def plot_statistics(evidence, output):
    records = evidence["comparison"]["records"]
    finest = min(evidence["config"]["paths"]["step_sizes"])
    counts = sorted({item["candidate_paths"] for item in records if item["candidate_paths"]})
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
    colors = {"density_path_nn": "tab:blue", "native_euler": "black", "unscaled": "tab:orange", "balanced": "tab:green"}
    styles = ("-", "--", ":")
    for index, (case, role) in enumerate((case, role) for case in ("uniform", "nonuniform")
                                        for role in ("conditional", "ensemble")):
        chosen = [item for item in records if item["case"] == case and item["role"] == role
                  and item["anchor_id"] == (0 if role == "conditional" else -1)
                  and (item["method"] == "statistics_nn" or item["dt"] == finest)]
        groups = defaultdict(list)
        for record in chosen:
            groups[record["method"], record.get("candidate"), record["optimizer_seed"]].append(record)
        for (method, candidate, seed), group in groups.items():
            direct = method == "statistics_nn" and role == "conditional"
            group.sort(key=lambda item: (item.get("initial_mc_paths") or item["candidate_paths"] or 0))
            x = counts if direct else [item.get("initial_mc_paths") or item["candidate_paths"] for item in group]
            label = (candidate.rsplit("-", 1)[-1] if method == "statistics_nn" else method)
            color = colors.get(candidate.rsplit("-", 1)[-1], "tab:green") if candidate else colors[method]
            if seed is not None:
                label += f", seed {seed}"
            for column, component in enumerate(("mean_error", "covariance_error")):
                values = [100*item["comparison"][component]["maximum_absolute_point"] for item in group]
                if direct:
                    values *= len(counts)
                axes[index, column].loglog(x, values, marker="o", color=color,
                    linestyle="-" if seed is None else styles[seed % len(styles)], label=label, alpha=.85)
        for column in range(2):
            axes[index, column].axhline(5., color="gray", linestyle=":", linewidth=1)
            axes[index, column].set(title=f"{case}, {role}: "+("means" if column == 0 else "covariance"),
                                    xlabel="Simulation paths / ensemble initial-law MC draws",
                                    ylabel="Maximum normalized point error (%)")
            axes[index, column].grid(alpha=.2)
            _count_ticks(axes[index, column], counts)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.01), fontsize=8, ncol=3)
    fig.suptitle("Fixed-observable statistics; finest candidate step, conditional anchor 0\n"
                 "Direct conditional head has no MC count; horizontal lines repeat its one prediction. Intervals are in the CSV.")
    _save(fig, output, "statistics_vs_mc_count.png")


def plot_costs(evidence, output):
    timings = evidence["timing"]["timings"]
    primary = evidence["config"]["balanced"]["name"]
    finest = min(evidence["config"]["paths"]["step_sizes"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, case in enumerate(("uniform", "nonuniform")):
        for column, role in enumerate(("conditional", "ensemble")):
            groups = defaultdict(list)
            for packet in timings:
                if packet["case"] != case or packet["role"] != role:
                    continue
                for method, value in packet["timings"].items():
                    if method != "statistics_nn" and packet["moment_candidate"] != primary:
                        continue
                    if method == "statistics_nn" and packet["dt"] != finest:
                        continue
                    label = (packet["moment_candidate"].rsplit("-", 1)[-1]+" statistics" if method == "statistics_nn"
                             else f"{method}, dt={packet['dt']:g}")
                    groups[label].append((packet["simulation_or_initial_mc_paths"], value["median_seconds"]))
            for label, points in groups.items():
                points.sort()
                axes[row, column].loglog(*np.asarray(points).T, "o-", label=label)
            axes[row, column].set(title=f"{case}, {role}", xlabel="Simulation / initial-law MC paths",
                                  ylabel="Warm matched-output kernel median (s)")
            axes[row, column].grid(alpha=.2)
            _count_ticks(axes[row, column], sorted({point[0] for group in groups.values() for point in group}))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.01), fontsize=8, ncol=2)
    fig.suptitle("Seed 0 timing only: excludes initialization, physical noise, loading, training, bootstrap and I/O")
    _save(fig, output, "matched_task_costs.png")


def plot_cost_components(evidence, output):
    timing = evidence["timing"]
    primary = evidence["config"]["balanced"]["name"]
    finest = min(evidence["config"]["paths"]["step_sizes"])
    count = max(item["simulation_or_initial_mc_paths"] for item in timing["timings"])
    generations = {(item["case"], item["role"], item["paths"], item["dt"]): item
                   for item in timing["input_generation_timings"]}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, case in enumerate(("uniform", "nonuniform")):
        for column, role in enumerate(("conditional", "ensemble")):
            packets = [item for item in timing["timings"] if item["case"] == case
                       and item["role"] == role and item["dt"] == finest
                       and item["simulation_or_initial_mc_paths"] == count]
            generated = generations[case, role, count, finest]
            labels, components = [], []
            for packet in packets:
                for method, value in packet["timings"].items():
                    statistics = method == "statistics_nn"
                    if not statistics and packet["moment_candidate"] != primary:
                        continue
                    labels.append(packet["moment_candidate"].rsplit("-", 1)[-1]+"\nstatistics"
                                  if statistics else method.replace("_", "\n"))
                    components.append((value["median_seconds"],
                        0. if statistics and role == "conditional" else generated["initialization"]["median_seconds"],
                        0. if statistics else generated["physical_noise_generation"]["median_seconds"]))
            components = np.asarray(components)
            bottom = np.zeros(len(labels))
            for index, component in enumerate(("matched-output kernel", "initial histograms / copies", "physical forcing draws")):
                axes[row, column].bar(np.arange(len(labels)), components[:, index], bottom=bottom, label=component)
                bottom += components[:, index]
            axes[row, column].set_xticks(np.arange(len(labels)), labels, fontsize=8)
            axes[row, column].set(yscale="log" if role == "conditional" else "linear",
                                  title=f"{case}, {role}", ylabel="Sum of separate component medians (s)")
            axes[row, column].set_ylim(float(bottom.min())*.5 if role == "conditional" else 0.,
                                       float(bottom.max())*(1.6 if role == "conditional" else 1.15))
            if row == column == 0:
                axes[row, column].legend(fontsize=7)
    fig.suptitle(f"Seed 0, finest candidate step, {count} paths / ensemble initial draws\n"
                 "Conditional statistics evaluates one fixed state. Loading and training remain separate.")
    _save(fig, output, "cost_components.png")


def _range(values):
    return [min(values), max(values)] if values else None


def cost_summary(evidence):
    timing = evidence["timing"]
    finest = min(evidence["config"]["paths"]["step_sizes"])
    count = max(item["simulation_or_initial_mc_paths"] for item in timing["timings"])
    generations = {(item["case"], item["role"], item["paths"], item["dt"]): item
                   for item in timing["input_generation_timings"]}
    rows = []
    for packet in timing["timings"]:
        if packet["dt"] != finest or packet["simulation_or_initial_mc_paths"] != count:
            continue
        generation = generations[packet["case"], packet["role"], count, finest]
        medians = {name: value["median_seconds"] for name, value in packet["timings"].items()}
        initial = generation["initialization"]["median_seconds"]
        noise = generation["physical_noise_generation"]["median_seconds"]
        totals = {name: value+(0. if name == "statistics_nn" and packet["role"] == "conditional" else initial)
                  +(0. if name == "statistics_nn" else noise) for name, value in medians.items()}
        rows.append(dict(case=packet["case"], role=packet["role"],
            candidate=packet["moment_candidate"], optimizer_seed=0, simulation_or_initial_mc_paths=count, dt=finest,
            path_kernel_seconds=medians["density_path_nn"], native_kernel_seconds=medians["native_euler"],
            statistics_kernel_seconds=medians["statistics_nn"], initialization_median_seconds=initial,
            physical_noise_median_seconds=noise, path_component_sum_seconds=totals["density_path_nn"],
            native_component_sum_seconds=totals["native_euler"], statistics_component_sum_seconds=totals["statistics_nn"],
            path_over_native_kernel_cost_ratio=medians["density_path_nn"]/medians["native_euler"],
            native_over_statistics_kernel_cost_ratio=medians["native_euler"]/medians["statistics_nn"],
            native_over_statistics_component_cost_ratio=totals["native_euler"]/totals["statistics_nn"],
            matched_accuracy_speedup_certified=False))
    return rows


def generate_report(run, output=None):
    evidence = load_evidence(run)
    run = evidence["run"]
    output = Path(output).resolve() if output else run/"report"
    output.mkdir(parents=True, exist_ok=True)
    plots = output/"plots"
    plots.mkdir(exist_ok=True)
    common, common_metadata = common_grid_refinement(evidence)
    residuals, tasks, costs = residual_rows(evidence), task_rows(evidence), cost_summary(evidence)
    paths = [dict(item, profile=_profile(item["case"])) for item in evidence["paths"]]
    _csv(output/"raw_residual_by_seed.csv", residuals)
    _csv(output/"path_refinement.csv", paths)
    _csv(output/"common_grid_refinement.csv", common)
    _csv(output/"task_accuracy_cost.csv", tasks)
    _csv(output/"restoration_costs.csv", evidence["timing"]["restoration"])
    _csv(output/"training_costs.csv", evidence["comparison"]["fitting_costs"])
    _csv(output/"fixed_count_cost_summary.csv", costs)
    plot_paths(dict(evidence, paths=common), plots)
    plot_paths(evidence, plots, name="path_refinement_all_stored_times.png", common=False)
    plot_residuals(evidence, residuals, plots)
    plot_statistics(evidence, plots)
    plot_costs(evidence, plots)
    plot_cost_components(evidence, plots)
    refinement = []
    for case in evidence["config"]["paths"]["cases"]:
        for dt in evidence["config"]["paths"]["step_sizes"]:
            selected = [item for item in common if item["case"] == case and float(item["step_size"]) == dt]
            refinement.append(dict(profile=_profile(case), dt=dt,
                seeds=[int(item["optimizer_seed"]) for item in selected],
                same_step_strong_min_max=_range([float(item["same_step_strong_supremum_rms"]) for item in selected]),
                neural_fine_strong_min_max=_range([float(item["neural_fine_strong_supremum_rms"]) for item in selected]),
                native_fine_strong_min_max=_range([float(item["euler_fine_strong_supremum_rms"]) for item in selected])))
    selected_residuals = [item for item in residuals if item["point"] == "selected"]
    conditional = []
    for candidate in ("unscaled", "balanced"):
        for seed in evidence["config"][candidate]["optimizer_seeds"]:
            selected = [item for item in evidence["comparison"]["records"]
                        if item["method"] == "statistics_nn" and item["role"] == "conditional"
                        and item["candidate"] == evidence["config"][candidate]["name"]
                        and item["optimizer_seed"] == seed]
            conditional.append(dict(candidate=candidate, optimizer_seed=seed, conditional_tasks=len(selected),
                mean_maximum_point_error_min_max=_range([
                    item["comparison"]["mean_error"]["maximum_absolute_point"] for item in selected]),
                covariance_maximum_point_error_min_max=_range([
                    item["comparison"]["covariance_error"]["maximum_absolute_point"] for item in selected]),
                largest_mean_pointwise_interval_endpoint=max(
                    item["comparison"]["mean_error"]["maximum_absolute_interval_endpoint"] for item in selected),
                largest_covariance_pointwise_interval_endpoint=max(
                    item["comparison"]["covariance_error"]["maximum_absolute_interval_endpoint"] for item in selected)))
    summary = dict(kind="interacting-followup-evidence-report-v1", run=str(run),
        source_hash=evidence["frozen"]["source_hash"], input_sha256=evidence["input_sha256"],
        physical_target={key: evidence["config"]["balanced"][key] for key in
                         ("n_cells", "n_particles", "kappa", "a2", "diffusion", "horizon", "query_times")},
        path_refinement=refinement, common_grid_analysis=common_metadata,
        selected_raw_residuals_by_seed=selected_residuals,
        conditional_statistics_by_seed=conditional,
        fixed_count_costs=costs,
        optimizer_seeds_pooled=False, timing_optimizer_seed=0, physical_accuracy_certified=False,
        continuous_reference_bias="unresolved", certified_break_even_queries=None,
        scientific_winner=None, winner_reason="Fixed-count costs and finite-reference errors do not resolve matched continuous-target accuracy",
        capabilities=evidence["comparison"]["capabilities"],
        uncertainty="Independent whole-family pointwise bootstrap; maxima of pointwise endpoints are not simultaneous confidence bounds",
        cost_scope="Kernel timings, input initialization/physical forcing, warm restoration and training are separate; sums of component medians are estimates")
    lines = ["# Interacting solver follow-up", "",
        f"Saved experiment: G={summary['physical_target']['n_cells']}, N={summary['physical_target']['n_particles']}, "
        f"D={summary['physical_target']['diffusion']}, kappa={summary['physical_target']['kappa']}, "
        f"a2={summary['physical_target']['a2']}, T={summary['physical_target']['horizon']}.", "",
        "Both solvers use the known interacting native DK generator. Optimizer seeds remain separate. "
        "The path solver is fitted independently at each integration step. Statistics heads return four fixed means "
        "and the full same-time 4x4 N-scaled covariance at three query times; they do not supply paths or cross-time laws.", "",
        "## Path refinement", "",
        "The ranges below are explicit minimum–maximum values across optimizer seeds, without pooling. "
        f"Primary strong errors use the normalized full-state supremum on the identical physical-time grid "
        f"of step {common_metadata['plan']['primary_common_time_step']:g}. Whole-family paired bootstrap intervals "
        "are saved in `common_grid_refinement.csv`. Native Euler is deterministically replayed from the existing "
        "initial histograms and Brownian increments; no new physical paths are drawn. "
        "The original per-solver stored-time metrics remain secondary in `path_refinement.csv`. "
        "Fine Euler is a finite reference; these differences are not certified continuous-time errors and no temporal rate is claimed.", "",
        "| Profile | Step | NN / same-step Euler | NN / fine Euler | Euler / fine Euler |",
        "|---|---:|---:|---:|---:|"]
    for row in refinement:
        def percent(values):
            return f"{100*values[0]:.4g}–{100*values[1]:.4g}%"
        lines.append(f"| {row['profile']} | {row['dt']:.7g} | {percent(row['same_step_strong_min_max'])} | "
                     f"{percent(row['neural_fine_strong_min_max'])} | {percent(row['native_fine_strong_min_max'])} |")
    lines += ["", "![Path refinement](plots/path_refinement.png)", "", "## Statistics loss balancing", "",
        "Each frozen residual scale was computed from the initial network on training states only. "
        "The objective sums use different scales across candidates and cannot be compared directly. "
        "Raw independent-validation RMS values below use common units; each row is the checkpoint selected by its own declared validation sum.", "",
        "| Candidate | Seed | Initial mean RMS | Selected mean RMS | Initial covariance RMS | Selected covariance RMS | Training s |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for candidate, infos in evidence["metadata"].items():
        for info in infos:
            initial, best = info["initial_validation"], info["best_validation"]
            lines.append(f"| {candidate} | {info['seed']} | {initial['mean_raw_rms']:.5g} | "
                         f"{best['mean_raw_rms']:.5g} | {initial['covariance_raw_rms']:.5g} | "
                         f"{best['covariance_raw_rms']:.5g} | {info['training_seconds']:.5g} |")
    lines += ["", "![Raw residual histories](plots/residual_histories.png)", "", "## Matched statistical tasks", "",
        "All three methods return the same three-time query bank of four means and full same-time covariance. "
        "Conditional tasks compare a direct head call at a fixed initial histogram with Monte Carlo trajectories. "
        "Ensemble tasks use equal declared initial-law integration/simulation counts. All stopped paths are retained. "
        "Means use sqrt(N) scaling; covariance uses N scaling and fixed uniform-particle observable units. "
        "A smaller covariance-generator residual cannot substitute for held-out covariance accuracy.", "",
        "Conditional error ranges below span the fixed initial histograms and both profiles within each optimizer seed. "
        "Each task error is the maximum over its three query times and four means/full covariance entries. "
        "The endpoint column is the largest absolute pointwise interval endpoint, not a simultaneous bound.", "",
        "| Candidate | Seed | Conditional mean point error range | Covariance point error range | Largest mean / covariance interval endpoint |",
        "|---|---:|---:|---:|---:|"]
    for row in conditional:
        def percentage_range(values):
            return f"{100*values[0]:.3g}–{100*values[1]:.3g}%"
        lines.append(f"| {row['candidate']} | {row['optimizer_seed']} | "
                     f"{percentage_range(row['mean_maximum_point_error_min_max'])} | "
                     f"{percentage_range(row['covariance_maximum_point_error_min_max'])} | "
                     f"{100*row['largest_mean_pointwise_interval_endpoint']:.3g}% / "
                     f"{100*row['largest_covariance_pointwise_interval_endpoint']:.3g}% |")
    lines += ["",
        "![Fixed-count statistical error](plots/statistics_vs_mc_count.png)", "",
        "`task_accuracy_cost.csv` retains every profile, conditional anchor, optimizer seed, count and time step. "
        "It separates candidate sampling/initial-law integration uncertainty from independent reference uncertainty. "
        "Plot maxima and pointwise percentile intervals are diagnostic summaries, not simultaneous confidence bounds. "
        "The finite-reference stopping/discretization bias remains unresolved; no 5% continuous-DK accuracy or solver winner is certified.", "",
        "## Cost and task suitability", "",
        "![Matched output cost](plots/matched_task_costs.png)", "",
        "![Separate online cost components](plots/cost_components.png)", "",
        "The fixed-count table uses the finest candidate step and largest preregistered count. Ratios compare measured "
        "costs at that count, not certified speedups at equal physical accuracy. Component totals sum separately measured "
        "medians. All values use prespecified optimizer seed 0.", "",
        "| Candidate | Profile / task | Path / native / statistics kernels (s) | Native / statistics kernel cost ratio | Native / statistics component cost ratio |",
        "|---|---|---:|---:|---:|"]
    for row in costs:
        lines.append(f"| {row['candidate'].rsplit('-', 1)[-1]} | {row['case']} / {row['role']} | "
                     f"{row['path_kernel_seconds']:.4g} / {row['native_kernel_seconds']:.4g} / "
                     f"{row['statistics_kernel_seconds']:.4g} | {row['native_over_statistics_kernel_cost_ratio']:.4g} | "
                     f"{row['native_over_statistics_component_cost_ratio']:.4g} |")
    lines += ["",
        "Warm timings use prespecified optimizer seed 0, not the most accurate seed. The kernel measurement excludes "
        "training, model restoration, histogram initialization, physical noise generation, bootstrap and I/O. "
        "The task CSV adds separately measured initialization/noise medians; their sum is an estimated component total, "
        "not a measured end-to-end latency. Statistics heads need no physical Brownian draws; direct conditional queries need no histogram Monte Carlo. "
        "Warm restoration and training are retained in separate CSV files. Deterministic native replay for this report "
        f"took {common_metadata['native_replay_seconds']:.5g} s and is excluded from inference benchmarks. "
        "Training-time sums across concurrent fits are not batch wall time. "
        "The saved timing packets retain implementation batch sizes: path inference uses bounded chunks, "
        "while native Euler is vectorized over its declared ensemble. These timings assess the current CPU implementations, "
        "not an architecture-independent lower bound. The benchmark measures matched moment outputs: native Euler "
        "returns only queried states and the path network internally constructs its full trajectory. "
        "A separate experiment is needed to compare full-trajectory export. "
        "Break-even query counts remain undefined until matched accuracy is resolved and online savings are positive.", "",
        "| Task | Density-path NN | Native Euler | Fixed statistics NN |",
        "|---|---|---|---|",
        "| Full-grid density trajectories | Supported | Supported | Not supplied |",
        "| Four declared means and same-time covariance | Monte Carlo | Monte Carlo | Direct conditional head; initial-law MC for ensembles |",
        "| New observables after simulation | Can evaluate saved paths; accuracy requires checking | Can evaluate saved paths; accuracy requires checking | Requires a new output/training study |",
        "| Path-dependent or cross-time queries | Path samples available; task accuracy unbenchmarked | Path samples available; task accuracy unbenchmarked | Not supplied by this head |",
        "| Rare stopping events / new physics | Not certified by this experiment | Not certified by this experiment | Not certified by this experiment |", "",
        "The task representation suggests where each solver can be useful. Advantage at a requested accuracy remains an empirical question; "
        "fixed-count speed and lower point error alone do not settle it.", ""]
    (output/"report.md").write_text("\n".join(lines))
    summary["artifact_sha256"] = {str(path.relative_to(output)): _sha(path) for path in sorted(output.rglob("*"))
                                  if path.is_file() and path.name != "summary.json"}
    _write(output/"summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = generate_report(args.run, args.output)
    print(json.dumps({"report": str((args.output or args.run/"report")/"report.md"),
                      "scientific_winner": result["scientific_winner"],
                      "physical_accuracy_certified": result["physical_accuracy_certified"]}, indent=2))


if __name__ == "__main__":
    main()
