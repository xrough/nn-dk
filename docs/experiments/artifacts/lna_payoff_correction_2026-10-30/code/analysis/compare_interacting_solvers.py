"""Task-matched interacting path/statistics/native solver comparison.

Freeze package source and finish every fit before running. References come from
the independent locked interacting-moment study. Candidate path and native
methods share fresh initial histograms and physical Brownian increments, never
reference noise. Every stopped path is retained. Continuous-time DK reference
bias remains unresolved, so this script does not declare an accuracy winner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nn_dk.data import json_ready, source_hash, write_json  # noqa: E402
from nn_dk.interacting_config import InteractingStudyConfig  # noqa: E402
from nn_dk.interacting_data import (  # noqa: E402
    file_sha256,
    load_interacting_reference,
    verify_interacting_fit_lock,
)
from nn_dk.interacting_evaluation import refinement_diagnostics  # noqa: E402
from nn_dk.interacting_training import load_interacting_model  # noqa: E402
from nn_dk.native import initialize_masses, physical_observable_weights, simulate_native  # noqa: E402
from nn_dk.path_config import PathConfig  # noqa: E402
from nn_dk.path_training import load_path_model  # noqa: E402
from nn_dk.types import seeded_rng, stable_hash  # noqa: E402

KIND = "task-matched-interacting-solver-comparison-v1"
CASE_MAP = {"dk_interacting_uniform": "uniform", "dk_interacting": "nonuniform"}


def moments(values, n_particles):
    """Unbiased same-time sample covariance; whole rows are trajectory families."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 4 or len(values) < 2:
        raise ValueError("At least two complete trajectory families with four observables required")
    if not np.isfinite(values).all():
        raise ValueError("Failed paths forbid statistics; survivor averaging is forbidden")
    mean = values.mean(0)
    centered = values-mean
    covariance = n_particles*np.einsum("pti,ptj->tij", centered, centered)/(len(values)-1)
    return mean, covariance


def total_moments(means, covariances, n_particles):
    """Total covariance, including random IID initial histograms."""
    means, covariances = np.asarray(means), np.asarray(covariances)
    if covariances.shape != (*means.shape, 4) or not np.isfinite(covariances).all():
        raise ValueError("Complete full conditional covariance matrices required")
    mean, between = moments(means, n_particles)
    return mean, covariances.mean(0)+between


def normalized_difference(a, b, cfg):
    std = np.sqrt(np.mean(physical_observable_weights(cfg.spec, cfg.queries)**2, axis=1))
    return (np.sqrt(cfg.n_particles)*(a[0]-b[0])/std,
            (a[1]-b[1])/np.outer(std, std))


def _interval(point, samples):
    low, high = np.quantile(np.asarray(samples), (.025, .975), axis=0)
    return dict(point=point, interval_95=np.stack((low, high), axis=-1),
                maximum_absolute_point=float(np.max(np.abs(point))),
                maximum_halfwidth=float(np.max((high-low)/2)),
                maximum_absolute_interval_endpoint=float(np.max(np.maximum(abs(low), abs(high)))))


def compare_prediction(prediction, reference, cfg, *, seed, n_bootstrap,
                       candidate=None, integration=None):
    """Independent reference/candidate whole-family bootstrap at fixed counts."""
    reference = np.asarray(reference)
    truth = moments(reference, cfg.n_particles)
    if (np.asarray(prediction[0]).shape != truth[0].shape
            or np.asarray(prediction[1]).shape != truth[1].shape
            or not all(np.isfinite(value).all() for value in prediction)
            or not np.allclose(prediction[1], np.swapaxes(prediction[1], -1, -2), atol=1e-10, rtol=0)
            or np.linalg.eigvalsh(prediction[1]).min() < -1e-9):
        raise ValueError("Finite compatible means and positive semidefinite full covariance required")
    if candidate is not None and integration is not None:
        raise ValueError("Use trajectory sampling or initial-law integration, not both")
    point = normalized_difference(prediction, truth, cfg)
    candidate_rng = seeded_rng(seed, "candidate-family-bootstrap")
    reference_rng = seeded_rng(seed, "reference-family-bootstrap")
    mean_samples, covariance_samples, reference_mean_samples, reference_covariance_samples = [], [], [], []
    candidate_mean_samples, candidate_covariance_samples = [], []
    for _ in range(n_bootstrap):
        ref_index = reference_rng.integers(0, len(reference), len(reference))
        ref = moments(reference[ref_index], cfg.n_particles)
        sampled = prediction
        if candidate is not None:
            index = candidate_rng.integers(0, len(candidate), len(candidate))
            sampled = moments(candidate[index], cfg.n_particles)
        elif integration is not None:
            index = candidate_rng.integers(0, len(integration[0]), len(integration[0]))
            sampled = total_moments(integration[0][index], integration[1][index], cfg.n_particles)
        mean, covariance = normalized_difference(sampled, ref, cfg)
        mean_samples.append(mean)
        covariance_samples.append(covariance)
        mean, covariance = normalized_difference(truth, ref, cfg)
        reference_mean_samples.append(mean)
        reference_covariance_samples.append(covariance)
        mean, covariance = normalized_difference(sampled, prediction, cfg)
        candidate_mean_samples.append(mean)
        candidate_covariance_samples.append(covariance)
    return json_ready(dict(
        prediction=dict(mean=prediction[0], scaled_covariance=prediction[1]),
        reference=dict(mean=truth[0], scaled_covariance=truth[1], n_families=len(reference)),
        mean_error=_interval(point[0], mean_samples), covariance_error=_interval(point[1], covariance_samples),
        reference_uncertainty=dict(mean=_interval(np.zeros_like(point[0]), reference_mean_samples),
                                   covariance=_interval(np.zeros_like(point[1]), reference_covariance_samples)),
        candidate_uncertainty=dict(mean=_interval(np.zeros_like(point[0]), candidate_mean_samples),
                                   covariance=_interval(np.zeros_like(point[1]), candidate_covariance_samples)),
        bootstrap=dict(n_replicates=n_bootstrap, confidence=.95, simultaneous=False,
                       method="independent whole trajectory family percentile bootstrap"),
        accuracy_certificate=False, continuous_reference_bias="unresolved"))


def paired_difference(first, second, cfg, *, seed, n_bootstrap):
    """Common candidate noise only; covariance is recomputed within each resample."""
    if first.shape != second.shape:
        raise ValueError("Paired methods require the same complete trajectory families")
    point = normalized_difference(moments(first, cfg.n_particles),
                                  moments(second, cfg.n_particles), cfg)
    rng = seeded_rng(seed, "paired-candidate-whole-family-bootstrap")
    samples = [[], []]
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(first), len(first))
        values = normalized_difference(moments(first[index], cfg.n_particles),
                                       moments(second[index], cfg.n_particles), cfg)
        for destination, value in zip(samples, values, strict=True):
            destination.append(value)
    return json_ready(dict(mean_difference=_interval(point[0], samples[0]),
                           covariance_difference=_interval(point[1], samples[1]),
                           interpretation="paired finite-step NN minus native Euler; no reference noise reused"))


def _read(path):
    return json.loads(Path(path).read_text())


def _resolve(run, path):
    path = Path(path)
    return path if path.is_absolute() else Path(run)/path


def _verify_item(run, item):
    for kind in ("checkpoint", "metadata"):
        path = _resolve(run, item[f"{kind}_path"])
        if not path.is_file() or file_sha256(path) != item[f"{kind}_sha256"]:
            raise ValueError("A locked solver checkpoint or metadata changed")


def _moment_inventory(run, reference_cfg, *, require_reference_complete=False):
    cfg = InteractingStudyConfig.load(run/"config.json")
    if (cfg.target_id != reference_cfg.target_id or cfg.cases != reference_cfg.cases
            or any(cfg.context(case) != reference_cfg.context(case) for case in cfg.cases)):
        raise ValueError("Statistics candidates and the shared reference must have identical physical targets")
    manifest = verify_interacting_fit_lock(run, cfg)
    allowed = ("complete",) if require_reference_complete else ("fits-locked", "complete")
    if manifest.get("status") not in allowed:
        raise ValueError("Every statistics candidate must finish and lock all its declared fits")
    result = []
    for item in manifest["fit_lock"]["models"]:
        _verify_item(run, item)
        result.append(dict(item, run=str(run), candidate=cfg.name, config_hash=cfg.config_hash))
    return result


def _inventory(path_runs, moment_run, cfg, additional_moment_runs=()):
    paths = []
    for run in path_runs:
        config = PathConfig.load(run/"config.json")
        manifest = _read(run/"manifest.json")
        lock = manifest.get("fit_lock", {})
        if (manifest.get("status") != "complete" or manifest.get("source_hash") != source_hash()
                or manifest.get("config_hash") != config.config_hash
                or lock.get("expected_models") != config.expected_models
                or len(lock.get("models", [])) != config.expected_models):
            raise ValueError("Every declared density-path fit must finish and lock before comparison")
        for item in lock["models"]:
            _verify_item(run, item)
            if item["case"] not in CASE_MAP:
                continue
            case = CASE_MAP[item["case"]]
            if (config.spec != cfg.spec or config.context(item["case"]) != cfg.context(case)
                    or config.horizon != cfg.horizon or config.query_times != cfg.query_times):
                raise ValueError("Shared grid, physics, initialization and query times are required")
            paths.append({**item, "run": str(run), "case": case, "path_case": item["case"]})
    if {item["case"] for item in paths} != set(cfg.cases):
        raise ValueError("Both uniform and nonuniform interacting path checkpoints are required")
    statistics = _moment_inventory(moment_run, cfg, require_reference_complete=True)
    for run in additional_moment_runs:
        if run == moment_run:
            raise ValueError("Do not duplicate the shared reference statistics run")
        statistics.extend(_moment_inventory(run, cfg))
    return paths, statistics


def candidate_inputs(cfg, case, role, anchor, count, dt, seed):
    """Addressable streams make prefixes independent of generation batch size."""
    initials, increments, families = [], [], []
    for index in range(count):
        anchor_key = None if anchor is None else np.asarray(anchor).tolist()
        family = stable_hash([KIND, seed, case, role, anchor_key, index])[:24]
        draw_seed = int(seeded_rng(seed, family, "initial").integers(0, 2**63-1))
        if role == "conditional":
            initials.append(np.asarray(anchor, dtype=float))
        else:
            initials.append(initialize_masses(cfg.context(case), cfg.spec, 1, draw_seed)[0])
        increments.append(seeded_rng(seed, family, "physical-face-noise").normal(
            0, np.sqrt(dt), (round(cfg.horizon/dt), cfg.n_cells)))
        families.append(family)
    return np.stack(initials), np.stack(increments), tuple(families)


def aggregate_noise(increments, fine_dt, dt):
    factor = round(dt/fine_dt)
    if factor < 1 or not np.isclose(factor*fine_dt, dt, atol=1e-14, rtol=0):
        raise ValueError("Candidate steps must be integer refinements of their finest noise grid")
    return increments.reshape(len(increments), -1, factor, increments.shape[-1]).sum(2)


def _save_arrays(path, identity, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    write_json(path.with_suffix(".json"), dict(identity=identity, sha256=file_sha256(path)))


def _load_arrays(path, identity):
    if not path.exists():
        return None
    packet = _read(path.with_suffix(".json"))
    if packet != dict(identity=identity, sha256=file_sha256(path)):
        raise ValueError("Comparison candidate data or provenance changed")
    with np.load(path, allow_pickle=False) as store:
        return {name: store[name] for name in store.files}


def _simulate(model, initial, noise, families, context, cfg, dt, batch_size, native_cache=None):
    path_values, native_values, path_stops, native_stops = [], [], [], []
    weights = physical_observable_weights(cfg.spec, cfg.queries)
    for start in range(0, len(initial), batch_size):
        end = min(start+batch_size, len(initial))
        nn = model.sample(context, cfg.query_times, end-start, 0,
                          initial_masses=initial[start:end], brownian_increments=noise[start:end],
                          family_ids=families[start:end])
        native = None if native_cache is not None else simulate_native(
            context, cfg.spec, cfg.query_times, dt, 0, end-start,
            initial_masses=initial[start:end], brownian_increments=noise[start:end],
            family_ids=families[start:end])
        for states in ((nn["states"],) if native is None else (nn["states"], native.states)):
            if not np.isfinite(states).all() or not np.allclose(states.sum(-1), 1., atol=1e-10, rtol=0):
                raise ValueError("An invalid candidate path blocks every estimate")
        path_values.append(nn["states"]@weights.T)
        path_stops.append(nn["stopped_at"])
        if native is not None:
            native_values.append(native.states@weights.T)
            native_stops.append(native.stopped_at)
    return dict(path_values=np.concatenate(path_values),
                native_values=np.concatenate(native_values) if native_cache is None else native_cache[0],
                path_stopped_at=np.concatenate(path_stops),
                native_stopped_at=np.concatenate(native_stops) if native_cache is None else native_cache[1])


def _predict(model, initial, times, batch_size):
    means, covariances = [], []
    for start in range(0, len(initial), batch_size):
        value = model.predict_conditional(initial[start:start+batch_size], times)
        means.append(value.mean)
        covariances.append(value.scaled_covariance)
    return np.concatenate(means), np.concatenate(covariances)


def _stopping(values, times):
    if np.isnan(values).any():
        raise ValueError("Missing stopping information prevents an ensemble estimate")
    return dict(n_paths=len(values), stopped_paths=int(np.isfinite(values).sum()),
                probability_at_queries=[float(np.mean(values <= t)) for t in times],
                all_paths_retained=True)


def _latency(samples):
    return dict(median_seconds=float(np.median(samples)), p05_seconds=float(np.quantile(samples, .05)),
                p95_seconds=float(np.quantile(samples, .95)), samples_seconds=samples)


def benchmark_input_generation(cfg, case, role, count, dt, *, warmup, repeats):
    """Independent discarded benchmark draws; separate initialization and forcing."""
    rng = seeded_rng(2026102601, "discarded-cost-inputs", case, role, count, dt)
    initialization, noise = [], []
    anchor = np.full(cfg.n_cells, 1/cfg.n_cells)
    for repetition in range(warmup+repeats):
        started = time.perf_counter()
        if role == "conditional":
            initial = np.broadcast_to(anchor, (count, cfg.n_cells)).copy()
        else:
            initial = initialize_masses(cfg.context(case), cfg.spec, count, int(rng.integers(0, 2**63-1)))
        elapsed_initial = time.perf_counter()-started
        started = time.perf_counter()
        physical_noise = rng.normal(0, np.sqrt(dt), (count, round(cfg.horizon/dt), cfg.n_cells))
        elapsed_noise = time.perf_counter()-started
        if not np.isfinite(initial).all() or not np.isfinite(physical_noise).all():
            raise FloatingPointError("Invalid discarded benchmark inputs")
        if repetition >= warmup:
            initialization.append(elapsed_initial)
            noise.append(elapsed_noise)
    return dict(case=case, role=role, paths=count, dt=dt,
                initialization=_latency(initialization), physical_noise_generation=_latency(noise),
                inputs_discarded=True, scope="vectorized IID histogram initialization and independent face normal draws",
                statistics_nn_requires_physical_noise=False,
                conditional_statistics_nn_requires_histogram_sampling=False)


def benchmark_tasks(path_model, moment_model, initial, noise, families, context, cfg, dt,
                    *, warmup, repeats, role, batch_size):
    """Matched means/covariance outputs, without bootstrap or output-file I/O."""
    weights = physical_observable_weights(cfg.spec, cfg.queries)

    def path():
        values = []
        for start in range(0, len(initial), batch_size):
            end = min(start+batch_size, len(initial))
            packet = path_model.sample(context, cfg.query_times, end-start, 0,
                                      initial_masses=initial[start:end], brownian_increments=noise[start:end],
                                      family_ids=families[start:end])
            values.append(packet["states"]@weights.T)
        return moments(np.concatenate(values), cfg.n_particles)

    def native():
        packet = simulate_native(context, cfg.spec, cfg.query_times, dt, 0, len(initial),
                                 initial_masses=initial, brownian_increments=noise, family_ids=families)
        return moments(packet.states@weights.T, cfg.n_particles)

    def statistics():
        mean, covariance = _predict(moment_model, initial[:1] if role == "conditional" else initial,
                                    cfg.query_times, max(batch_size, len(initial)))
        return (mean[0], covariance[0]) if role == "conditional" else total_moments(mean, covariance, cfg.n_particles)

    functions = dict(density_path_nn=path, native_euler=native, statistics_nn=statistics)
    rng = seeded_rng(2026102601, "timing-method-order", role, len(initial), dt)
    samples = {name: [] for name in functions}
    for repetition in range(warmup+repeats):
        for name in rng.permutation(list(functions)):
            started = time.perf_counter()
            result = functions[name]()
            elapsed = time.perf_counter()-started
            if result[0].shape != (len(cfg.query_times), 4) or result[1].shape != (len(cfg.query_times), 4, 4):
                raise ValueError("Timed methods must return identical mean/full-covariance query banks")
            if repetition >= warmup:
                samples[name].append(elapsed)
    return dict(role=role, conditional_query_batch=1 if role == "conditional" else None,
                simulation_or_initial_mc_paths=len(initial), dt=dt,
                implementation_batches=dict(density_path=batch_size, native_euler=len(initial),
                                            statistics=1 if role == "conditional" else len(initial)),
                input_scope="precomputed matching initial histograms and physical Brownian increments",
                output_scope="three physical times, four means and full same-time 4x4 N-scaled covariance",
                includes=["coefficient/network evaluation", "stopping", "moment reduction"],
                excludes=["model restoration", "training", "initialization", "noise generation", "bootstrap", "I/O"],
                timings={name: _latency(values) for name, values in samples.items()},
                matched_accuracy_certified=False)


def compare(path_runs, moment_run, output, *, counts=(256, 1024, 4096), seed=2026102601,
            n_bootstrap=200, batch_size=64, timing=False, warmup=2, repeats=7,
            additional_moment_runs=()):
    if (not counts or any(type(v) is not int or v < 2 for v in counts)
            or tuple(sorted(set(counts))) != tuple(counts) or n_bootstrap < 20
            or batch_size < 1 or warmup < 1 or repeats < 3):
        raise ValueError("Preregister increasing counts >=2, >=20 bootstrap replicates and valid timing settings")
    path_runs = tuple(Path(p).resolve() for p in path_runs)
    moment_run, output = Path(moment_run).resolve(), Path(output).resolve()
    additional_moment_runs = tuple(Path(p).resolve() for p in additional_moment_runs)
    cfg = InteractingStudyConfig.load(moment_run/"config.json")
    paths, statistics = _inventory(path_runs, moment_run, cfg, additional_moment_runs)
    design = json_ready(dict(kind=KIND, source_hash=source_hash(), script_sha256=file_sha256(Path(__file__)),
                             moment_run=str(moment_run), moment_config_hash=cfg.config_hash,
                             reference_artifacts={name: file_sha256(moment_run/name) for name in
                                                  ("interacting_reference.h5", "interacting_reference.json")},
                             path_models=paths, statistics_models=statistics,
                             counts=counts, seed=seed, n_bootstrap=n_bootstrap, batch_size=batch_size,
                             timing=timing, warmup=warmup, repeats=repeats,
                             timing_optimizer_seed=0,
                             queries=cfg.query_times, physical_target=cfg.target_id,
                             finite_reference_step=cfg.reference_step, continuous_bias="unresolved",
                             inputs="new addressable candidate families; shared between methods; independent reference",
                             no_survivor_averaging=True, optimizer_seeds_pooled=False))
    output.mkdir(parents=True, exist_ok=True)
    design_path = output/"preregistration.json"
    if design_path.exists() and _read(design_path) != design:
        raise ValueError("Comparison preregistration changed; use a fresh output directory")
    write_json(design_path, design)
    result_path = output/"comparison.json"
    if result_path.exists():
        cached = _read(result_path)
        digest = cached.pop("content_hash", None)
        if cached.get("design_hash") != stable_hash(design) or stable_hash(cached) != digest:
            raise ValueError("Completed comparison identity changed")
        cached["content_hash"] = digest
        for name, digest in cached["artifact_lock"].items():
            if file_sha256(output/name) != digest:
                raise ValueError("A completed comparison artifact changed")
        return cached
    datasets = output/"candidates"
    datasets.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    records, timings, references, generation_timings = [], [], [], {}
    moment_models = [(item, load_interacting_model(_resolve(item["run"], item["checkpoint_path"])))
                     for item in statistics]
    restoration = []
    for case in cfg.cases:
        reference = load_interacting_reference(moment_run, cfg, case)
        for role in ("conditional", "ensemble"):
            if reference[role]["failed"].any():
                raise ValueError("A failed reference path blocks every task comparison")
        refinement = reference["ensemble"]["refinement_values"][:cfg.refinement_paths]
        references.append(dict(case=case, finite_step=cfg.reference_step,
            reference_sha256=file_sha256(moment_run/"interacting_reference.h5"),
            refinement=refinement_diagnostics(refinement, reference["ensemble"]["family_ids"][:cfg.refinement_paths].astype(str),
                                               cfg, seed, stopped_at=reference["ensemble"]["refinement_stopped_at"][:cfg.refinement_paths])))
        chosen = [item for item in paths if item["case"] == case]
        finest_dt = min(item["dt"] for item in chosen)
        tasks = [("conditional", i, anchor) for i, anchor in enumerate(reference["anchors"])]
        tasks.append(("ensemble", -1, None))
        for role, anchor_id, anchor in tasks:
            native_cache = {}
            native_reported = set()
            ref = reference[role]
            reference_values = ref["values"][ref["anchor_ids"] == anchor_id] if role == "conditional" else ref["values"]
            expected = cfg.conditional_paths if role == "conditional" else cfg.ensemble_paths
            if len(reference_values) != expected:
                raise ValueError("Fixed shared-reference allocation changed")
            initial, fine_noise, families = candidate_inputs(cfg, case, role, anchor, max(counts), finest_dt, seed)
            forbidden = set(reference["conditional"]["family_ids"].astype(str)) | set(reference["ensemble"]["family_ids"].astype(str))
            if set(families) & forbidden:
                raise ValueError("Candidate and reference trajectory families overlap")
            for item, moment_model in moment_models:
                mean, covariance = _predict(moment_model, initial[:1] if role == "conditional" else initial,
                                            cfg.query_times, batch_size)
                for count in ((None,) if role == "conditional" else counts):
                    prediction = ((mean[0], covariance[0]) if role == "conditional" else
                                  total_moments(mean[:count], covariance[:count], cfg.n_particles))
                    records.append(dict(case=case, role=role, anchor_id=anchor_id, method="statistics_nn",
                        optimizer_seed=item["seed"], candidate_paths=0, initial_mc_paths=count,
                        candidate=item["candidate"], candidate_run=item["run"],
                        checkpoint_sha256=item["checkpoint_sha256"],
                        comparison=compare_prediction(prediction, reference_values, cfg, seed=seed,
                            n_bootstrap=n_bootstrap,
                            integration=None if role == "conditional" else (mean[:count], covariance[:count]))))
            for item in chosen:
                checkpoint = _resolve(item["run"], item["checkpoint_path"])
                tick = time.perf_counter()
                path_model = load_path_model(checkpoint)
                restoration.append(dict(case=case, optimizer_seed=item["seed"], dt=item["dt"],
                                        measured_seconds=time.perf_counter()-tick,
                                        scope="warm process load, provenance verification, allocation; no cold-cache claim"))
                noise = aggregate_noise(fine_noise, finest_dt, item["dt"])
                identity = dict(design_hash=stable_hash(design), case=case, role=role, anchor_id=anchor_id,
                                checkpoint_sha256=item["checkpoint_sha256"],
                                initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                                brownian_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
                                family_ids_hash=stable_hash(families))
                path = datasets/f"{stable_hash(identity)[:24]}.npz"
                arrays = _load_arrays(path, identity)
                if arrays is None:
                    arrays = _simulate(path_model, initial, noise, families, cfg.context(case), cfg,
                                       item["dt"], batch_size, native_cache.get(item["dt"]))
                    _save_arrays(path, identity, **arrays)
                native_cache[item["dt"]] = arrays["native_values"], arrays["native_stopped_at"]
                for count in counts:
                    for method, key in (("density_path_nn", "path"), ("native_euler", "native")):
                        if key == "native" and (item["dt"], count) in native_reported:
                            continue
                        if key == "native":
                            native_reported.add((item["dt"], count))
                        values = arrays[f"{key}_values"][:count]
                        records.append(dict(case=case, role=role, anchor_id=anchor_id, method=method,
                            optimizer_seed=item["seed"] if key == "path" else None, dt=item["dt"],
                            candidate_paths=count, initial_mc_paths=count if role == "ensemble" else 0,
                            candidate_artifact=str(path.relative_to(output)),
                            stopping=_stopping(arrays[f"{key}_stopped_at"][:count], cfg.query_times),
                            comparison=compare_prediction(moments(values, cfg.n_particles), reference_values, cfg,
                                                          seed=seed, n_bootstrap=n_bootstrap, candidate=values),
                            paired_nn_minus_native=paired_difference(arrays["path_values"][:count], arrays["native_values"][:count],
                                                                    cfg, seed=seed, n_bootstrap=n_bootstrap)))
                    if timing and anchor_id in (0, -1) and item["seed"] == 0:
                        generation_key = case, role, count, item["dt"]
                        if generation_key not in generation_timings:
                            generation_timings[generation_key] = benchmark_input_generation(
                                cfg, case, role, count, item["dt"], warmup=warmup, repeats=repeats)
                        for moment_item, moment_model in moment_models:
                            if moment_item["seed"] != item["seed"]:
                                continue
                            value = benchmark_tasks(path_model, moment_model, initial[:count], noise[:count], families[:count],
                                cfg.context(case), cfg, item["dt"], warmup=warmup, repeats=repeats, role=role, batch_size=batch_size)
                            value.update(case=case, path_seed=item["seed"], moment_seed=moment_item["seed"])
                            value.update(moment_candidate=moment_item["candidate"])
                            timings.append(value)
    artifact_lock = {str(p.relative_to(output)): file_sha256(p) for p in sorted(datasets.iterdir()) if p.is_file()}
    result = json_ready(dict(kind=KIND, design_hash=stable_hash(design), source_hash=source_hash(),
        script_sha256=file_sha256(Path(__file__)), records=records, references=references,
        timings=timings, input_generation_timings=list(generation_timings.values()),
        restoration=restoration, artifact_lock=artifact_lock,
        fitting_costs=[dict(method="density_path_nn", case=item["case"], dt=item["dt"], optimizer_seed=item["seed"],
                           training_seconds=_read(_resolve(item["run"], item["metadata_path"]))["training_seconds"])
                       for item in paths]+[
                          dict(method="statistics_nn", candidate=item["candidate"], optimizer_seed=item["seed"],
                               training_seconds=_read(_resolve(item["run"], item["metadata_path"]))["training_seconds"])
                          for item in statistics],
        amortization=dict(certified_break_even_queries=None,
                          reason="Matched physical accuracy and reference bias must be resolved before a speed or amortization winner."),
        platform=dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                      torch=torch.__version__, torch_threads=torch.get_num_threads()),
        capabilities=dict(density_path_nn=["full-grid trajectories", "new observables after simulation", "path-dependent queries", "joint-time law samples"],
                          native_euler=["full-grid trajectories", "new observables after simulation", "path-dependent queries", "joint-time law samples"],
                          statistics_nn=["four declared means", "full same-time 4x4 covariance", "conditional or initial-law integrated queries"],
                          unbenchmarked=["new-observable accuracy", "rare stopping probabilities", "cross-time statistics", "generalization beyond fixed physics/horizon"]),
        interpretation="Finite-Euler-reference accuracy and fixed-count cost, not a certified continuous-DK winner. Optimizer seeds remain separate; pointwise bootstrap intervals are not simultaneous guarantees."))
    csv_path = output/"comparison.csv"
    fields = ("case", "role", "anchor_id", "method", "candidate", "optimizer_seed", "dt", "candidate_paths",
              "initial_mc_paths", "maximum_mean_error", "maximum_covariance_error", "maximum_mean_interval_endpoint",
              "maximum_covariance_interval_endpoint", "reference_mean_halfwidth", "reference_covariance_halfwidth",
              "candidate_mean_halfwidth", "candidate_covariance_halfwidth")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            packet = record["comparison"]
            row = {key: record.get(key) for key in fields[:9]}
            row.update(maximum_mean_error=packet["mean_error"]["maximum_absolute_point"],
                       maximum_covariance_error=packet["covariance_error"]["maximum_absolute_point"],
                       maximum_mean_interval_endpoint=packet["mean_error"]["maximum_absolute_interval_endpoint"],
                       maximum_covariance_interval_endpoint=packet["covariance_error"]["maximum_absolute_interval_endpoint"],
                       reference_mean_halfwidth=packet["reference_uncertainty"]["mean"]["maximum_halfwidth"],
                       reference_covariance_halfwidth=packet["reference_uncertainty"]["covariance"]["maximum_halfwidth"],
                       candidate_mean_halfwidth=packet["candidate_uncertainty"]["mean"]["maximum_halfwidth"],
                       candidate_covariance_halfwidth=packet["candidate_uncertainty"]["covariance"]["maximum_halfwidth"])
            writer.writerow(row)
    result["artifact_lock"]["comparison.csv"] = file_sha256(csv_path)
    result["content_hash"] = stable_hash(result)
    write_json(result_path, result)
    return result


def benchmark_completed_comparison(output, *, warmup=2, repeats=7):
    """Run separately after scientific jobs quiesce; never reselect sample counts."""
    output = Path(output).resolve()
    design, comparison = _read(output/"preregistration.json"), _read(output/"comparison.json")
    digest = comparison.pop("content_hash", None)
    if (stable_hash(comparison) != digest or comparison["design_hash"] != stable_hash(design)
            or design["source_hash"] != source_hash()
            or design["script_sha256"] != file_sha256(Path(__file__))):
        raise ValueError("Timing requires an unchanged completed comparison and source")
    for name, digest in design["reference_artifacts"].items():
        if file_sha256(Path(design["moment_run"])/name) != digest:
            raise ValueError("The locked shared reference changed before timing")
    for name, digest in comparison["artifact_lock"].items():
        if file_sha256(output/name) != digest:
            raise ValueError("A completed comparison artifact changed before timing")
    if warmup < 1 or repeats < 3:
        raise ValueError("Use positive warmups and at least three repetitions")
    cfg = InteractingStudyConfig.load(Path(design["moment_run"])/"config.json")
    additional = tuple(Path(item["run"]) for item in design["statistics_models"]
                       if item["run"] != design["moment_run"])
    paths, statistics = _inventory(tuple(Path(item) for item in sorted({p["run"] for p in design["path_models"]})),
                                   Path(design["moment_run"]), cfg, tuple(sorted(set(additional))))
    torch.set_num_threads(1)
    models, restorations = [], []
    for item in statistics:
        if item["seed"] != 0:
            continue
        tick = time.perf_counter()
        model = load_interacting_model(_resolve(item["run"], item["checkpoint_path"]))
        restorations.append(dict(method="statistics_nn", candidate=item["candidate"], seed=item["seed"],
                                 seconds=time.perf_counter()-tick))
        models.append((item, model))
    timings, generations = [], {}
    for case in cfg.cases:
        reference = load_interacting_reference(design["moment_run"], cfg, case)
        chosen = [item for item in paths if item["case"] == case and item["seed"] == 0]
        fine_dt = min(item["dt"] for item in chosen)
        for role, anchor in (("conditional", reference["anchors"][0]), ("ensemble", None)):
            initial, fine_noise, families = candidate_inputs(cfg, case, role, anchor, max(design["counts"]),
                                                             fine_dt, design["seed"])
            for item in chosen:
                tick = time.perf_counter()
                path_model = load_path_model(_resolve(item["run"], item["checkpoint_path"]))
                restorations.append(dict(method="density_path_nn", case=case, seed=item["seed"], dt=item["dt"],
                                         seconds=time.perf_counter()-tick))
                noise = aggregate_noise(fine_noise, fine_dt, item["dt"])
                for count in design["counts"]:
                    key = case, role, count, item["dt"]
                    if key not in generations:
                        generations[key] = benchmark_input_generation(cfg, case, role, count, item["dt"],
                                                                      warmup=warmup, repeats=repeats)
                    for moment_item, moment_model in models:
                        if moment_item["seed"] != item["seed"]:
                            continue
                        packet = benchmark_tasks(path_model, moment_model, initial[:count], noise[:count], families[:count],
                            cfg.context(case), cfg, item["dt"], warmup=warmup, repeats=repeats,
                            role=role, batch_size=design["batch_size"])
                        packet.update(case=case, path_seed=item["seed"], moment_seed=moment_item["seed"],
                                      moment_candidate=moment_item["candidate"])
                        timings.append(packet)
    result = json_ready(dict(kind="task-matched-interacting-solver-warm-timing-v1",
        comparison_sha256=file_sha256(output/"comparison.json"), design_hash=stable_hash(design),
        source_hash=source_hash(), script_sha256=file_sha256(Path(__file__)),
        warmup=warmup, repeats=repeats, timings=timings, input_generation_timings=list(generations.values()),
        optimizer_seed=0, optimizer_seed_selection="prespecified architecture-cost representative; accuracy uses every declared seed separately",
        restoration=restorations, restoration_scope="warm process only, package/file caches uncontrolled",
        physical_accuracy_certified=False, cold_start_measured=False,
        external_cpu_quiescence="operator must run only after all training, generation and evaluation jobs exit",
        interpretation="Task-matched fixed-count costs. Combine with saved accuracy intervals; no automatic winner or certified amortization."))
    result["content_hash"] = stable_hash(result)
    write_json(output/"task_timing.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-runs", nargs="+")
    parser.add_argument("--moment-run")
    parser.add_argument("--additional-moment-runs", nargs="*", default=(),
                        help="Additional fits-locked statistics candidates; all use --moment-run references")
    parser.add_argument("--output", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=(256, 1024, 4096))
    parser.add_argument("--seed", type=int, default=2026102601)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timing", action="store_true", help="Only enable after all competing CPU jobs finish")
    parser.add_argument("--timing-only", action="store_true", help="Time an existing --output comparison after all jobs exit")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)
    before, script_before = source_hash(), file_sha256(Path(__file__))
    if args.timing_only:
        benchmark_completed_comparison(args.output, warmup=args.warmup, repeats=args.repeats)
    else:
        if not args.path_runs or not args.moment_run:
            parser.error("--path-runs and --moment-run are required unless --timing-only")
        compare(args.path_runs, args.moment_run, args.output, counts=tuple(args.counts), seed=args.seed,
                n_bootstrap=args.bootstrap, batch_size=args.batch_size, timing=args.timing,
                warmup=args.warmup, repeats=args.repeats, additional_moment_runs=args.additional_moment_runs)
    if source_hash() != before or file_sha256(Path(__file__)) != script_before:
        raise RuntimeError("Package source or comparison script changed during execution")


if __name__ == "__main__":
    main()
