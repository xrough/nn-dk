"""Resumable orchestration shared by smoke and research configurations."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, contexts_for
from .data import generation_cost, load_split, prepare_run, write_json
from .types import seeded_rng


class BudgetExhausted(RuntimeError):
    """An atomic unit finished and the saved run can be resumed."""


def check_budget(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExhausted("Run budget exhausted; completed work is saved for resumption")


def calibrate(directory: Path, config: ExperimentConfig, deadline=None) -> dict:
    from .calibration import run_calibration

    path = directory / "calibration.json"
    if path.exists():
        return json.loads(path.read_text())
    check_budget(deadline)
    result = run_calibration(config.target, config.dt, config.save_interval, config.horizon,
                             config.seed, n_paths=config.calibration_paths,
                             n_particles=config.n_particles, diffusion=config.diffusion)
    write_json(path, result)
    return result


def fit(directory: Path, config: ExperimentConfig, deadline=None) -> dict:
    from .training import fit_baselines, fit_models

    if not (directory / "calibration.json").exists():
        raise ValueError("Run calibration before fitting")
    calibration = json.loads((directory / "calibration.json").read_text())
    if calibration["status"] != "passed":
        raise ValueError("Numerical implementation calibration failed; resolve it before fitting")
    train, validation = load_split(directory, "train"), load_split(directory, "validation")
    destination = directory / "models"
    destination.mkdir(exist_ok=True)
    baseline_path = destination / "baselines.json"
    check_budget(deadline)
    if not baseline_path.exists():
        tick = time.perf_counter()
        baseline_metadata = fit_baselines(train, validation, config.target, config.queries,
                                           config.lag, destination)
        write_json(destination / "baseline_cost.json", {"training_seconds": time.perf_counter() - tick,
                                                       "details": baseline_metadata})
    records = []
    for seed in config.optimizer_seeds:
        check_budget(deadline)
        record = fit_models(train, validation, config.target, config.queries, config.lag,
                            config.training, destination, seed, deadline)
        records.append(record)
        if not record["complete"]:
            raise BudgetExhausted("Neural fit checkpointed; resume the same run to continue")
    return {"complete": True, "fits": records}


def evaluate_run(directory: Path, config: ExperimentConfig, deadline=None) -> list[dict]:
    from .classical import discrete_dk, meanfield_and_fluctuations
    from .evaluation import evaluate
    from .particles import simulate_particles
    from .training import load_a, load_b, load_baselines

    destination = directory / "evaluations"
    destination.mkdir(exist_ok=True)
    models = load_baselines(directory / "models")
    for seed in config.optimizer_seeds:
        checkpoint_metadata = json.loads((directory / "models" / f"training_seed{seed}.json").read_text())
        if not checkpoint_metadata["complete"]:
            raise ValueError("Complete all declared fits before evaluating held-out outcomes")
        models[f"neural_a_seed{seed}"] = load_a(directory / "models" / f"a_seed{seed}.pt")
        models[f"neural_b_seed{seed}"] = load_b(directory / "models" / f"b_seed{seed}.pt")
    costs = generation_cost(directory)
    train_data_cost = costs["train"] + costs["validation"]
    baseline_cost = json.loads((directory / "models" / "baseline_cost.json").read_text())["training_seconds"]
    learned_times = np.arange(round(config.horizon / config.lag) + 1) * config.lag
    records = []
    test = load_split(directory, "test")
    batch_throughput = {}
    for method, model in models.items():
        missing = any(not (destination / f"{context.context_id}__{method}.json").exists()
                      for context, _ in test)
        if hasattr(model, "predict_batch") and missing:
            check_budget(deadline)
            tick = time.perf_counter()
            model.predict_batch([c for c, _ in test], config.queries)
            seconds = time.perf_counter() - tick
            batch_throughput[method] = {"contexts": len(test), "seconds": seconds,
                                         "contexts_per_second": len(test) / max(seconds, 1e-12)}
    for context, reference in test:
        classical_predictions = None
        classical_seconds = None
        methods = ["particle_mc", "mean_field", "linear_fluctuations", "discrete_dk", *models]
        for method in methods:
            path = destination / f"{context.context_id}__{method}.json"
            if path.exists():
                records.append(json.loads(path.read_text()))
                continue
            check_budget(deadline)
            random_seed = int(seeded_rng(config.seed, "evaluation", context.context_id, method).integers(0, 2 ** 63, dtype=np.int64))
            tick = time.perf_counter()
            if method == "particle_mc":
                prediction = simulate_particles(context, config.target, learned_times, config.dt,
                                                  random_seed, n_paths=config.model_paths)
            elif method in ("mean_field", "linear_fluctuations"):
                if classical_predictions is None:
                    classical_predictions = meanfield_and_fluctuations(context, config.target,
                        config.queries, config.dt, n_grid=config.n_grid)
                    classical_seconds = time.perf_counter() - tick
                prediction = classical_predictions[0 if method == "mean_field" else 1]
            elif method == "discrete_dk":
                prediction = discrete_dk(context, config.target, learned_times, config.dt,
                                         random_seed, n_paths=config.model_paths, n_grid=config.n_grid)
            else:
                model = models[method]
                prediction = (model.sample(context, learned_times, config.model_paths, random_seed)
                              if hasattr(model, "sample") else model.predict(context, config.queries))
            prediction_seconds = time.perf_counter() - tick
            if method in ("mean_field", "linear_fluctuations"):
                prediction_seconds = classical_seconds
            record = evaluate(prediction, reference, context, config.queries, seed=random_seed,
                                n_bootstrap=config.bootstrap_samples)
            record["method"] = method
            record["config_hash"] = config.config_hash
            training_seconds = 0.0
            if method.startswith("neural_"):
                kind = "a" if method.startswith("neural_a_") else "b"
                seed = int(method.rsplit("seed", 1)[1])
                metadata = json.loads((directory / "models" / f"training_seed{seed}.json").read_text())
                training_seconds = metadata[kind]["training_seconds"]
            elif method in models:
                # Conservative upper accounting: joint baseline fitting charged to each standalone method.
                training_seconds = baseline_cost
            learned = method in models
            record["cost"] = {
                "prediction_seconds": prediction_seconds,
                "training_seconds": training_seconds,
                "training_data_seconds": train_data_cost if learned else 0.0,
                "standalone_total_seconds": prediction_seconds + training_seconds + (train_data_cost if learned else 0.0),
                "reference_audit_seconds": reference.diagnostics["generation_seconds"],
                "completed_query_bank": config.queries.query_id,
                "query_count": config.queries.dimension,
                "prediction_path_count": config.model_paths if hasattr(prediction, "states") else 0,
                "single_context_latency_seconds": prediction_seconds,
                "batched_paths_per_second": config.model_paths / max(prediction_seconds, 1e-12) if hasattr(prediction, "states") else None,
                "batched_context_throughput": batch_throughput.get(method),
                "initialization_seconds": float(prediction.diagnostics.get("initialization_seconds", 0)),
                "interpretation": "Measured workload; online timings alone do not establish matched-accuracy speedup",
            }
            if method in ("mean_field", "linear_fluctuations"):
                record["cost"]["classical_shared_computation"] = "Upper bound: the combined deterministic mean and linearized covariance calculation is charged to each method"
            write_json(path, record)
            records.append(record)
    return records


def report(directory: Path, config: ExperimentConfig):
    from .reporting import render_report

    records = [json.loads(path.read_text()) for path in sorted((directory / "evaluations").glob("*.json"))]
    methods = {"particle_mc", "mean_field", "linear_fluctuations", "discrete_dk",
               "autoregression", "polynomial_ridge", "kernel_ridge"}
    methods.update(f"neural_{kind}_seed{seed}" for kind in ("a", "b") for seed in config.optimizer_seeds)
    expected = {(c.context_id, method) for c in contexts_for(config, "test") for method in methods}
    actual = [(r["context_id"], r["method"]) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Incomplete or incompatible evaluation workload; resume evaluation before reporting")
    for record in records:
        if record["config_hash"] != config.config_hash or record["target_id"] != config.target.target_id or record["query_id"] != config.queries.query_id:
            raise ValueError("Evaluation provenance does not match the configured report")
    calibration = json.loads((directory / "calibration.json").read_text())
    manifest = prepare_run(directory, config)
    manifest = {**manifest, "status": "complete", "last_completed_stage": "report"}
    manifest.pop("error", None)
    return render_report(records, calibration, manifest, directory / "report")
