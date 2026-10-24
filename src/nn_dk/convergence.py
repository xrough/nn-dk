"""Resumable A-only experiments for an autonomous stopped discrete DK solver."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .convergence_data import (
    generate_native_pool,
    load_native_pool,
    native_generation_cost,
    prepare_convergence_run,
)
from .convergence_metrics import compare_native
from .convergence_reporting import render_convergence_report
from .data import write_json
from .dk_training import coefficient_diagnostics, fit_a, load_coefficient_model
from .experiments import BudgetExhausted
from .native import initialize_masses, simulate_native
from .native_types import NativeBatch
from .types import seeded_rng, stable_hash


def _check_budget(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExhausted("Project A checkpointed; resume the same configuration and output directory")


def _save_batch(path, batch):
    """Commit a whole independent chunk only after all of its paths are saved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, times=batch.times, states=batch.states,
                            stopped_at=batch.stopped_at, family_ids=batch.family_ids,
                            diagnostics=json.dumps(batch.diagnostics))
    temporary.replace(path)


def _read_batch(path, context, spec):
    with np.load(path, allow_pickle=False) as data:
        return NativeBatch(data["times"], data["states"], spec, context,
                           data["stopped_at"], tuple(data["family_ids"].tolist()),
                           json.loads(str(data["diagnostics"])))


def _join_batches(batches):
    first = batches[0]
    for batch in batches:
        first.spec.require_compatible(batch.spec)
        if first.context != batch.context or not np.array_equal(first.times, batch.times):
            raise ValueError("Cannot combine incompatible native trajectory chunks")
    states = np.concatenate([b.states for b in batches])
    diagnostics = {"all_paths_retained": True, "chunks": len(batches),
                   "mass_error": float(np.abs(states.sum(-1) - 1).max()),
                   "minimum_mass": float(states.min())}
    for key in ("initialization_seconds", "simulation_seconds", "total_seconds",
                "histogram_initialization_seconds", "brownian_generation_seconds", "aggregation_seconds"):
        diagnostics[key] = sum(b.diagnostics.get(key, 0.) for b in batches)
    return NativeBatch(first.times, states, first.spec, first.context,
                       np.concatenate([b.stopped_at for b in batches]),
                       tuple(identifier for b in batches for identifier in b.family_ids), diagnostics)


def _dt_key(dt):
    return f"dt{dt.hex()}"


def _coupling(config, case, start, end):
    """Evaluation families have no relationship to training/validation families.

    Positive diagonal face factors declare a common noise coupling. Coarse face
    increments are literal sums of the fine increments, not reseeded Gaussians.
    """
    context, spec = config.context(case), config.spec
    seed = int(seeded_rng(config.seed, "native-evaluation", case, start).integers(0, 2**63))
    tick = time.perf_counter()
    initial = initialize_masses(context, spec, end-start, seed)
    histogram_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    dw = seeded_rng(seed, "shared-fine-face-increments").normal(
        0., np.sqrt(config.reference_dt),
        (end-start, round(config.horizon / config.reference_dt), spec.n_cells))
    noise_seconds = time.perf_counter() - tick
    ids = tuple(f"{config.config_hash}/{case}/evaluation/{i}" for i in range(start, end))
    return initial, dw, ids, seed, {"histogram_initialization_seconds": histogram_seconds,
                                  "brownian_generation_seconds": noise_seconds}


def _aggregate_increments(dw, dt, fine_dt):
    ratio = round(dt / fine_dt)
    if ratio < 1 or not np.isclose(ratio * fine_dt, dt):
        raise ValueError("Coarse Brownian increments must nest within the reference grid")
    return dw.reshape(len(dw), -1, ratio, dw.shape[-1]).sum(2)


def _simulation_cache(directory, config, case, dt, label, model=None, ablation="oracle", deadline=None):
    chunks = []
    context, spec = config.context(case), config.spec
    for start in range(0, config.diagnostic_paths, config.evaluation_batch_size):
        end = min(start + config.evaluation_batch_size, config.diagnostic_paths)
        path = Path(directory) / "evaluation" / case / label / _dt_key(dt) / f"chunk_{start:07d}.npz"
        if path.exists():
            batch = _read_batch(path, context, spec)
            if len(batch.states) != end-start:
                raise ValueError("Cached evaluation chunk has the wrong path allocation")
        else:
            _check_budget(deadline)
            initial, fine_dw, ids, seed, timing = _coupling(config, case, start, end)
            tick = time.perf_counter()
            dw = _aggregate_increments(fine_dw, dt, config.reference_dt)
            del fine_dw
            timing["aggregation_seconds"] = time.perf_counter() - tick
            batch = simulate_native(context, spec, config.times, dt, seed, n_paths=end-start,
                                    initial_masses=initial, brownian_increments=dw,
                                    family_ids=ids, ablation=ablation,
                                    coefficients=None if model is None else model.sampling_coefficients)
            batch.diagnostics.update(timing)
            batch.diagnostics["initialization_seconds"] += sum(timing.values())
            batch.diagnostics["total_seconds"] += sum(timing.values())
            batch.diagnostics["coupling"] = "same IID histogram; sums of fine face Brownian increments"
            _save_batch(path, batch)
        chunks.append(batch)
    return _join_batches(chunks)


def _comparison(config, prediction, reference, bias=None):
    return compare_native(prediction, reference, config.queries,
                          n_bootstrap=config.bootstrap_samples, seed=config.seed, paired=True,
                          tolerance=config.tolerance, uncertainty_budget=config.uncertainty_budget,
                          reference_bias=bias)


def fit_convergence(directory, config, case, fit_spec, deadline=None):
    """A-only fitting entry point, deliberately independent of legacy fit/B."""
    _check_budget(deadline)
    training = load_native_pool(directory, config, case, "train", fit_spec["paths"])
    validation = load_native_pool(directory, config, case, "validation")
    path = Path(directory) / "checkpoints" / case / f"{fit_spec['key']}.pt"
    result = fit_a([training], [validation], config.spec, mode=fit_spec["mode"],
                   width=fit_spec["width"], seed=fit_spec["seed"], config=config.training,
                   output_path=path, deadline=deadline)
    write_json(path.with_suffix(".json"), result)
    if not result["complete"]:
        raise BudgetExhausted("Coefficient training checkpointed at an epoch boundary")
    return result


def _latency(directory, config, case, fit_spec, model, deadline):
    path = Path(directory) / "timing" / case / f"{fit_spec['key']}.json"
    if path.exists():
        return json.loads(path.read_text())
    _check_budget(deadline)
    # One-context latency is measured separately from the batched path throughput.
    batch = model.sample(config.context(case), config.times, 1, config.seed + 918,
                         step_size=min(config.step_sizes))
    size = min(config.diagnostic_paths, config.evaluation_batch_size)
    batched = model.sample(config.context(case), config.times, size, config.seed + 919,
                           step_size=min(config.step_sizes))
    timing = {"single_path_latency_seconds": batch.diagnostics["total_seconds"],
              "latency_step_size": min(config.step_sizes), "device": "cpu",
              "standalone_batch_paths": size,
              "standalone_batch_seconds": batched.diagnostics["total_seconds"],
              "standalone_paths_per_second": size / batched.diagnostics["total_seconds"],
              "standalone_timing_ablation": "both",
              "timing_scope": "Standalone timings draw their own coarse noise; paired audit costs also include fine-noise construction"}
    write_json(path, timing)
    return timing


def _expected_keys(config):
    return {(case, "oracle", None, dt) for case in config.cases for dt in config.step_sizes} | {
        (case, fit_spec["key"], method, dt) for case in config.cases for fit_spec in config.fit_specs()
        for method in ("drift", "noise", "both") for dt in config.step_sizes}


def _record_key(record):
    return record["case"], record["fit_key"], record.get("ablation"), record["step_size"]


def report_convergence(directory, config):
    """Refuse a complete report from missing evaluations or partial fits."""
    directory = Path(directory)
    manifest = prepare_convergence_run(directory, config)
    calibration = json.loads((directory / "calibration.json").read_text())
    records = json.loads((directory / "records.json").read_text())
    if not calibration.get("complete") or {_record_key(r) for r in records} != _expected_keys(config):
        raise ValueError("Incomplete convergence study; resume before publishing its final report")
    return render_convergence_report(records, calibration, manifest, directory / "report")


def run_convergence(directory, config, deadline=None):
    """Run A0--A4; A5 runs only for an empirically successful selected fit.

    A completed software workflow can have entirely unresolved scientific gates.
    Fixed finest-step biases are reported as audited, not certified by fiat.
    """
    from .dk_calibration import calibrate_native

    directory = Path(directory)
    manifest = prepare_convergence_run(directory, config)
    torch.set_num_threads(config.training.get("threads", 1))
    started = time.monotonic()
    records_path = directory / "records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    records_by_key = {_record_key(r): r for r in records}
    record_directory = directory / "records"
    record_directory.mkdir(exist_ok=True)
    for path in sorted(record_directory.glob("*.json")):
        value = json.loads(path.read_text())
        records_by_key[_record_key(value)] = value

    def snapshot_records():
        # Sorting makes interrupted and uninterrupted execution yield the same
        # report order. Individual immutable records avoid quadratic disk I/O.
        ordered = sorted(records_by_key.values(), key=lambda r: repr(_record_key(r)))
        write_json(records_path, ordered)

    def record(value):
        records_by_key[_record_key(value)] = value
        write_json(record_directory / f"{stable_hash(_record_key(value))}.json", value)

    try:
        print("[Project A] A0: analytic and stopped-reference calibration", flush=True)
        calibration = calibrate_native(config, directory, deadline)
        if not calibration.get("complete"):
            raise BudgetExhausted("A0 calibration checkpointed")
        if calibration["linear_gaussian"].get("status") != "passed":
            raise ValueError("A0 numerical calibration failed; coefficient training is disabled")
        for case in config.cases:
            print(f"[Project A] {case}: native training families", flush=True)
            generated = generate_native_pool(directory, config, case, deadline)
            if not generated["complete"]:
                raise BudgetExhausted("Native trajectory generation checkpointed")
            reference = _simulation_cache(directory, config, case, config.reference_dt,
                                          "reference", deadline=deadline)
            bias = calibration.get("reference_bias", {}).get(case)
            oracle = {}
            for dt in config.step_sizes:
                oracle[dt] = _simulation_cache(directory, config, case, dt, "oracle", deadline=deadline)
                key = (case, "oracle", None, dt)
                if key not in records_by_key:
                    _check_budget(deadline)
                    record({"case": case, "stage": "A0", "method": "oracle", "fit_key": "oracle",
                            "step_size": dt, "comparison": _comparison(config, oracle[dt], reference, bias),
                            "cost": {"prediction_seconds": oracle[dt].diagnostics["simulation_seconds"],
                                     "online_audit_seconds": oracle[dt].diagnostics["total_seconds"],
                                     "timing_scope": "Coupled audit including fine-noise construction; not standalone deployment throughput",
                                     "initialization_seconds": oracle[dt].diagnostics["initialization_seconds"],
                                     "total_seconds": oracle[dt].diagnostics["total_seconds"]}})
            test = load_native_pool(directory, config, case, "test")
            for fit_spec in config.fit_specs():
                print(f"[Project A] {case}: {fit_spec['key']}", flush=True)
                fitting = fit_convergence(directory, config, case, fit_spec, deadline)
                model = load_coefficient_model(fitting["path"])
                diagnostic_path = Path(fitting["path"]).with_suffix(".diagnostics.json")
                if diagnostic_path.exists():
                    diagnostics = json.loads(diagnostic_path.read_text())
                else:
                    _check_budget(deadline)
                    diagnostics = coefficient_diagnostics(model, [test])
                    write_json(diagnostic_path, diagnostics)
                latency = _latency(directory, config, case, fit_spec, model, deadline)
                for ablation in ("drift", "noise", "both"):
                    for dt in config.step_sizes:
                        key = (case, fit_spec["key"], ablation, dt)
                        if key in records_by_key:
                            continue
                        prediction = _simulation_cache(directory, config, case, dt,
                            fit_spec["key"] + "_" + ablation, model, ablation, deadline)
                        _check_budget(deadline)
                        cost = {"generation_seconds": sum(native_generation_cost(directory, case).values()),
                                "training_seconds": fitting["training_seconds"],
                                "initialization_seconds": prediction.diagnostics["initialization_seconds"],
                                "prediction_seconds": prediction.diagnostics["simulation_seconds"],
                                "online_audit_seconds": prediction.diagnostics["total_seconds"],
                                "brownian_generation_seconds": prediction.diagnostics["brownian_generation_seconds"],
                                "paths_per_second": config.diagnostic_paths / max(prediction.diagnostics["total_seconds"], 1e-12),
                                "total_seconds": prediction.diagnostics["total_seconds"] + fitting["training_seconds"]
                                    + sum(native_generation_cost(directory, case).values()), **latency}
                        record({"case": case, "stage": "A4" if fit_spec["mode"] == "increments" else "A1-A3",
                                "fit_key": fit_spec["key"], "method": "learned_" + ablation,
                                "ablation": ablation, "training_mode": fit_spec["mode"],
                                "width": fit_spec["width"], "train_paths": fit_spec["paths"],
                                "seed": fit_spec["seed"], "step_size": dt,
                                "comparison": _comparison(config, prediction, reference, bias),
                                "neural_error": _comparison(config, prediction, oracle[dt]),
                                "timestep_error": records_by_key[(case, "oracle", None, dt)]["comparison"],
                                "coefficient_diagnostics": diagnostics, "cost": cost})
        if {_record_key(r) for r in records_by_key.values()} != _expected_keys(config):
            raise RuntimeError("Convergence runner did not complete its fixed allocation")
        # A5 is a separately labelled discrepancy study, after empirical A1--A3 success.
        audits = {}
        for case in config.cases:
            selected = [r for r in records_by_key.values() if r["case"] == case and
                        r.get("training_mode") == "oracle" and r.get("ablation") == "both" and
                        r.get("train_paths") == max(config.train_path_counts) and
                        r.get("width") == config.base_width and r.get("seed") == config.optimizer_seeds[0] and
                        r["step_size"] == min(config.step_sizes)]
            successful = bool(selected and selected[0]["neural_error"]["accuracy_gate"].get("contrast_within_tolerance"))
            if config.particle_numbers and successful:
                from .particle_audit import audit_particles

                fit_key = selected[0]["fit_key"]
                model = load_coefficient_model(directory / "checkpoints" / case / (fit_key + ".pt"))
                audits[case] = audit_particles(config, config.context(case), model,
                                               directory / "particle_audit" / case, deadline=deadline)
                if not audits[case].get("complete"):
                    raise BudgetExhausted("Particle comparison checkpointed")
            else:
                audits[case] = {"status": "deferred", "reason": "Not requested in this batch" if not config.particle_numbers
                                else "Selected oracle fit has not resolved its neural-error contrast below tolerance"}
        write_json(directory / "particle_audits.json", audits)
        snapshot_records()
        manifest["status"] = "complete"
        manifest["scientific_status"] = "See uncertainty gates; workflow completion does not certify convergence"
        manifest["elapsed_seconds"] += time.monotonic() - started
        write_json(directory / "manifest.json", manifest)
        return report_convergence(directory, config)
    except BudgetExhausted:
        snapshot_records()
        manifest["status"] = "budget-exhausted"
        manifest["elapsed_seconds"] += time.monotonic() - started
        write_json(directory / "manifest.json", manifest)
        raise
    except Exception as error:
        snapshot_records()
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["elapsed_seconds"] += time.monotonic() - started
        write_json(directory / "manifest.json", manifest)
        raise
