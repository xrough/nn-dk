"""Held-out conditional and ensemble statistics for the neural BSDE solver.

Only same-time four-dimensional covariances are claimed. Reference paths share
initial states and Brownian increments across time resolutions, never across
independent test families. A stopped trajectory is retained in every estimate.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .data import json_ready, source_hash, write_json
from .native import initialize_masses, physical_observable_weights, simulate_native
from .types import QueryBank, seeded_rng, stable_hash


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        from .experiments import BudgetExhausted

        raise BudgetExhausted("BSDE evaluation reached its resumable invocation budget")


def _provenance(config):
    return {"config_hash": config.config_hash, "source_hash": source_hash()}


def _validate_provenance(record, config):
    if any(record.get(key) != value for key, value in _provenance(config).items()):
        raise ValueError("BSDE evaluation source/configuration mismatch; use a new directory")


def aggregate_conditional_moments(mean, scaled_covariance, n_particles):
    """Law of total covariance for an IID sample of initial states.

    The between-initial-state term uses the unbiased sample covariance. It is
    not permissible to average conditional covariances alone for histogram starts.
    """
    mean, scaled_covariance = np.asarray(mean), np.asarray(scaled_covariance)
    if (mean.ndim != 3 or mean.shape[-1] != 4
            or scaled_covariance.shape != (*mean.shape, 4) or len(mean) < 2
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scaled_covariance))
            or n_particles <= 0):
        raise ValueError("At least two finite initial-state predictions and 4x4 covariances required")
    centered = mean - mean.mean(axis=0)
    between = n_particles * np.einsum("phi,phj->hij", centered, centered) / (len(mean) - 1)
    within = scaled_covariance.mean(axis=0)
    return mean.mean(axis=0), within + between


def sample_moments(samples, n_particles):
    """Same-time moments of independent paths with shape (paths, times, 4)."""
    samples = np.asarray(samples, dtype=float)
    if (samples.ndim != 3 or samples.shape[-1] != 4 or len(samples) < 2
            or not np.all(np.isfinite(samples)) or n_particles <= 0):
        raise ValueError("At least two finite independent four-observable paths required")
    mean = samples.mean(axis=0)
    centered = samples - mean
    covariance = n_particles * np.einsum("phi,phj->hij", centered, centered) / (len(samples) - 1)
    return mean, covariance


def _interval(draws):
    return np.moveaxis(np.quantile(draws, (0.025, 0.975), axis=0), 0, -1)


def _errors(mean, covariance, reference_mean, reference_covariance, n_particles, std):
    std = np.asarray(std)
    signed_mean = np.sqrt(n_particles) * (mean - reference_mean) / std
    signed_covariance = (covariance - reference_covariance) / np.outer(std, std)
    return {"mean_normalized_signed": signed_mean,
            "scaled_covariance_normalized_signed": signed_covariance,
            "mean_normalized_rmse": float(np.sqrt(np.mean(signed_mean**2))),
            "scaled_covariance_normalized_rmse": float(np.sqrt(np.mean(signed_covariance**2))),
            "maximum_mean_normalized_error": float(np.max(np.abs(signed_mean))),
            "maximum_scaled_covariance_normalized_error": float(np.max(np.abs(signed_covariance))),
            "scaled_variance_normalized_rmse": float(np.sqrt(np.mean(
                np.diagonal(signed_covariance, axis1=-2, axis2=-1)**2)))}


def _gate(errors, uncertainty, tolerance, uncertainty_budget, reference_bias_resolved=False):
    available = bool(uncertainty.get("available"))
    max_width = uncertainty.get("maximum_sampling_normalized_halfwidth")
    sampling = bool(available and max_width is not None and max_width <= uncertainty_budget)
    endpoint = uncertainty.get("maximum_contrast_normalized_ci_endpoint")
    contrast = bool(available and endpoint is not None and endpoint <= tolerance)
    return {"status": "resolved_within_tolerance" if sampling and contrast and reference_bias_resolved
            else "unresolved", "sampling_resolved": sampling,
            "contrast_within_tolerance": contrast,
            "reference_bias_resolved": reference_bias_resolved,
            "tolerance": tolerance, "each_uncertainty_budget": uncertainty_budget,
            "interpretation": "Pointwise intervals; optimizer seeds are separate. A finite reference "
            "refinement difference does not certify remaining continuous-time bias.",
            "point_estimate_within_tolerance": bool(max(errors["maximum_mean_normalized_error"],
                errors["maximum_scaled_covariance_normalized_error"]) <= tolerance)}


def compare_analytic_moments(mean, covariance, exact_mean, exact_covariance, *,
                             n_particles=1, std=None, tolerance=0.05):
    """Errors against exact moments on the fixed held-out initial-state bank."""
    std = np.ones(4) if std is None else np.asarray(std)
    arrays = [np.asarray(a, dtype=float) for a in (mean, covariance, exact_mean, exact_covariance)]
    mean, covariance, exact_mean, exact_covariance = arrays
    if (mean.shape != exact_mean.shape or covariance.shape != exact_covariance.shape
            or mean.shape[-1] != 4 or covariance.shape != (*mean.shape, 4)
            or not all(np.all(np.isfinite(a)) for a in arrays)):
        raise ValueError("Matching finite conditional means and four-dimensional covariances required")
    error = _errors(mean, covariance, exact_mean, exact_covariance, n_particles, std)
    endpoint = max(error["maximum_mean_normalized_error"],
                   error["maximum_scaled_covariance_normalized_error"])
    uncertainty = {"available": True, "method": "analytic reference on fixed test states",
                   "maximum_sampling_normalized_halfwidth": 0.,
                   "maximum_contrast_normalized_ci_endpoint": endpoint,
                   "interpretation": "No reference Monte Carlo error; does not certify generalization outside the fixed test bank"}
    return json_ready({"errors": error, "uncertainty": uncertainty,
                       "accuracy_gate": _gate(error, uncertainty, tolerance, 0., True),
                       "minimum_predicted_covariance_eigenvalue": float(np.linalg.eigvalsh(covariance).min())})


def compare_conditional_moments(mean, covariance, samples, *, n_particles, std,
                                n_bootstrap=200, seed=0, tolerance=.05,
                                uncertainty_budget=.0125):
    """Compare at fixed anchors; independently resample continuation families.

    Anchors are a fixed diagnostic bank, not IID continuation paths. Every child
    trajectory is resampled as a whole across all horizons. Children belonging
    to different anchors are sampled independently; no pooled pseudo-replicates.
    """
    mean, covariance, samples = map(np.asarray, (mean, covariance, samples))
    if (samples.ndim != 4 or samples.shape[0] != len(mean) or samples.shape[-1] != 4
            or samples.shape[1] < 2 or mean.shape != (samples.shape[0], samples.shape[2], 4)
            or covariance.shape != (*mean.shape, 4) or n_bootstrap < 0
            or not all(np.all(np.isfinite(a)) for a in (mean, covariance, samples))):
        raise ValueError("Conditional references need (anchors, independent children, times, 4)")
    refs = [sample_moments(paths, n_particles) for paths in samples]
    rm, rc = np.asarray([r[0] for r in refs]), np.asarray([r[1] for r in refs])
    errors = _errors(mean, covariance, rm, rc, n_particles, std)
    uncertainty = {"available": False, "sampling_unit": "whole continuation family within each fixed anchor",
                   "confidence_level": .95, "anchor_generalization_uncertainty_included": False}
    if n_bootstrap >= 2:
        rng, md, cd = seeded_rng(seed, "bsde-conditional-bootstrap"), [], []
        for _ in range(n_bootstrap):
            draws = [sample_moments(paths[rng.integers(0, len(paths), len(paths))], n_particles)
                     for paths in samples]
            md.append(np.asarray([r[0] for r in draws]))
            cd.append(np.asarray([r[1] for r in draws]))
        md, cd = np.asarray(md), np.asarray(cd)
        mean_ci = _interval(np.sqrt(n_particles) * (mean - md) / std)
        cov_ci = _interval((covariance - cd) / np.outer(std, std))
        ref_mean_ci = _interval(np.sqrt(n_particles) * md / std)
        ref_cov_ci = _interval(cd / np.outer(std, std))
        width = max(float(np.max(np.diff(ref_mean_ci, axis=-1) / 2)),
                    float(np.max(np.diff(ref_cov_ci, axis=-1) / 2)))
        endpoint = max(float(np.max(np.abs(mean_ci))), float(np.max(np.abs(cov_ci))))
        uncertainty.update(available=True, n_bootstrap=n_bootstrap,
                           mean_normalized_contrast_ci95=mean_ci,
                           scaled_covariance_normalized_contrast_ci95=cov_ci,
                           maximum_sampling_normalized_halfwidth=width,
                           maximum_contrast_normalized_ci_endpoint=endpoint)
    return json_ready({"reference_mean": rm, "reference_scaled_covariance": rc,
                       "errors": errors, "uncertainty": uncertainty,
                       "accuracy_gate": _gate(errors, uncertainty, tolerance, uncertainty_budget),
                       "minimum_predicted_covariance_eigenvalue": float(np.linalg.eigvalsh(covariance).min())})


def compare_ensemble_moments(mean, covariance, reference_samples, *, n_particles, std,
                             n_bootstrap=200, seed=0, tolerance=.05,
                             uncertainty_budget=.0125):
    """Paired whole-family bootstrap including initial-histogram sampling."""
    mean, covariance, reference_samples = map(np.asarray, (mean, covariance, reference_samples))
    if len(mean) != len(reference_samples):
        raise ValueError("Matched initial histograms and reference families required")
    pm, pc = aggregate_conditional_moments(mean, covariance, n_particles)
    rm, rc = sample_moments(reference_samples, n_particles)
    errors = _errors(pm, pc, rm, rc, n_particles, std)
    uncertainty = {"available": False, "paired": True,
                   "sampling_unit": "whole initial-histogram and forward-noise family",
                   "confidence_level": .95}
    if n_bootstrap >= 2:
        rng = seeded_rng(seed, "bsde-ensemble-bootstrap")
        pmeans, pcovs, rmeans, rcovs = [], [], [], []
        for _ in range(n_bootstrap):
            index = rng.integers(0, len(mean), len(mean))
            p, c = aggregate_conditional_moments(mean[index], covariance[index], n_particles)
            r, v = sample_moments(reference_samples[index], n_particles)
            pmeans.append(p)
            pcovs.append(c)
            rmeans.append(r)
            rcovs.append(v)
        pmeans, pcovs, rmeans, rcovs = map(np.asarray, (pmeans, pcovs, rmeans, rcovs))
        mean_ci = _interval(np.sqrt(n_particles) * (pmeans - rmeans) / std)
        cov_ci = _interval((pcovs - rcovs) / np.outer(std, std))
        widths = []
        for md, cd in ((pmeans, pcovs), (rmeans, rcovs)):
            widths.extend((float(np.max(np.diff(_interval(np.sqrt(n_particles) * md / std), axis=-1) / 2)),
                           float(np.max(np.diff(_interval(cd / np.outer(std, std)), axis=-1) / 2))))
        uncertainty.update(available=True, n_bootstrap=n_bootstrap,
                           mean_normalized_contrast_ci95=mean_ci,
                           scaled_covariance_normalized_contrast_ci95=cov_ci,
                           maximum_sampling_normalized_halfwidth=max(widths),
                           maximum_contrast_normalized_ci_endpoint=max(
                               float(np.max(np.abs(mean_ci))), float(np.max(np.abs(cov_ci)))))
    return json_ready({"prediction_mean": pm, "prediction_scaled_covariance": pc,
                       "reference_mean": rm, "reference_scaled_covariance": rc,
                       "errors": errors, "uncertainty": uncertainty,
                       "accuracy_gate": _gate(errors, uncertainty, tolerance, uncertainty_budget),
                       "minimum_predicted_conditional_covariance_eigenvalue": float(np.linalg.eigvalsh(covariance).min())})


def _cache_chunk(path, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(json_ready(metadata))))
    temporary.replace(path)


def _load_chunk(path, config, expected):
    with np.load(path, allow_pickle=False) as store:
        metadata = json.loads(str(store["metadata"]))
        _validate_provenance(metadata, config)
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("BSDE reference cache target/family mismatch")
        arrays = {key: store[key] for key in store.files if key != "metadata"}
    return arrays, metadata


def _native_reference_bank(directory, config, case, kind, deadline):
    """Resumable shared coarse/fine reference, bounded by one batch of increments."""
    context, spec = config.context(case), config.spec
    queries = QueryBank(tuple(config.query_times), (1, 2))
    conditional = kind == "conditional"
    anchors = config.conditional_anchors if conditional else config.outer_paths
    children = config.conditional_paths if conditional else 1
    total = anchors * children
    initial_seed = int(seeded_rng(config.seed, "bsde-test-initial", case, kind).integers(0, 2**63 - 1))
    tick = time.perf_counter()
    anchor_states = initialize_masses(context, spec, anchors, initial_seed)
    initialization_seconds = time.perf_counter() - tick
    root = Path(directory) / "evaluation_cache" / case / kind
    chunks, seconds = [], 0.
    minimum_saved_mass, maximum_mass_error = np.inf, 0.
    nsteps = int(round(config.horizon / config.reference_dt))
    ratio = int(round(config.dt / config.reference_dt))
    if (ratio < 1 or not np.isclose(config.dt, ratio * config.reference_dt)
            or nsteps % ratio):
        raise ValueError("Evaluation reference step must divide the trained forward timestep")
    weights = physical_observable_weights(spec, queries)
    for start in range(0, total, config.reference_batch_size):
        end = min(total, start + config.reference_batch_size)
        path = root / f"chunk_{start:08d}.npz"
        expected = {"case": case, "kind": kind, "start": start, "end": end,
                    "target_id": spec.target_id, "dt": config.dt,
                    "reference_dt": config.reference_dt, "initial_seed": initial_seed}
        if path.exists():
            arrays, metadata = _load_chunk(path, config, expected)
        else:
            _check_deadline(deadline)
            tick = time.perf_counter()
            selected = np.arange(start, end) // children
            initial = anchor_states[selected]
            seed = int(seeded_rng(config.seed, "bsde-test-forcing", case, kind, start).integers(0, 2**63 - 1))
            dw = seeded_rng(seed, "physical-brownian").normal(
                0, np.sqrt(config.reference_dt), (end-start, nsteps, spec.n_cells))
            ids = tuple(f"bsde-test/{config.config_hash}/{case}/{kind}/anchor{j//children}/child{j%children}"
                        for j in range(start, end))
            fine = simulate_native(context, spec, queries.times, config.reference_dt, seed,
                                   end-start, initial_masses=initial, brownian_increments=dw,
                                   family_ids=ids)
            coarse_dw = dw.reshape(end-start, nsteps//ratio, ratio, spec.n_cells).sum(axis=2)
            coarse = simulate_native(context, spec, queries.times, config.dt, seed,
                                     end-start, initial_masses=initial, brownian_increments=coarse_dw,
                                     family_ids=ids)
            arrays = {"fine": fine.states @ weights.T, "coarse": coarse.states @ weights.T,
                      "fine_stopped_at": fine.stopped_at, "coarse_stopped_at": coarse.stopped_at,
                      "anchor_index": selected, "family_ids": np.asarray(ids)}
            metadata = {**_provenance(config), **expected, "forcing_seed": seed,
                        "seconds": time.perf_counter() - tick,
                        "minimum_saved_mass": min(float(fine.states.min()), float(coarse.states.min())),
                        "maximum_mass_error": max(float(np.max(np.abs(fine.states.sum(-1)-1))),
                                                  float(np.max(np.abs(coarse.states.sum(-1)-1)))),
                        "all_stopped_paths_included": True}
            _cache_chunk(path, arrays, metadata)
        chunks.append(arrays)
        seconds += metadata["seconds"]
        minimum_saved_mass = min(minimum_saved_mass, metadata["minimum_saved_mass"])
        maximum_mass_error = max(maximum_mass_error, metadata["maximum_mass_error"])
    values = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    if conditional:
        for key in ("fine", "coarse"):
            values[key] = values[key].reshape(anchors, children, len(queries.times), 4)
    values["initial"] = anchor_states
    values["cost"] = {"reference_generation_seconds": seconds,
                      "initialization_seconds": initialization_seconds,
                      "reference_paths": total, "n_initial_anchors": anchors}
    values["stopping"] = {key: _stopping_summary(values[f"{key}_stopped_at"], config.horizon, config.query_times)
                          for key in ("fine", "coarse")}
    values["diagnostics"] = {"minimum_saved_mass": minimum_saved_mass,
                             "maximum_mass_error": maximum_mass_error,
                             "all_stopped_paths_included": True}
    values["family_ids"] = values["family_ids"].tolist()
    values["scope"] = kind
    return values


def _stopping_summary(stopped_at, horizon, times=None):
    stopped_at = np.asarray(stopped_at)
    stopped = stopped_at <= horizon
    count, total = int(stopped.sum()), len(stopped)
    return {"n_paths": total, "n_stopped": count, "probability_by_horizon": count / total,
            "zero_stop_one_sided_95_upper": float(1 - .05**(1/total)) if count == 0 else None,
            "restricted_mean_stopping_time": float(np.minimum(stopped_at, horizon).mean()),
            "all_stopped_paths_included": True,
            "times": list(times) if times is not None else [horizon],
            "stopping_cdf": [float(np.mean(stopped_at <= t)) for t in (times if times is not None else [horizon])],
            "observed_stopping_times": stopped_at[stopped].tolist()}


def _prediction_arrays(prediction):
    if isinstance(prediction, dict):
        return np.asarray(prediction["mean"]), np.asarray(prediction["scaled_covariance"])
    return np.asarray(prediction.mean), np.asarray(prediction.scaled_covariance)


def _inference_benchmark(model, initial, horizons):
    """Standalone value-head latency and batched throughput, excluding loading."""
    rows = []
    for count in dict.fromkeys((1, min(64, len(initial)))):
        model.predict_conditional(initial[:count], horizons)
        timings = []
        for _ in range(3):
            tick = time.perf_counter()
            model.predict_conditional(initial[:count], horizons)
            timings.append(time.perf_counter()-tick)
        median = float(np.median(timings))
        rows.append({"initial_states": count, "seconds": timings,
                     "median_seconds": median, "distinct_initial_queries_per_second": count/median})
    return {"warmup_calls_per_batch": 1, "repetitions": 3, "rows": rows,
            "includes": "Complete conditional prediction and covariance validation; checkpoint loading excluded"}


def _model_identity(model, seed):
    checkpoint = getattr(model, "checkpoint_path", None)
    identity = {"seed": int(seed), "problem": model.problem}
    if checkpoint is not None and Path(checkpoint).exists():
        identity["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    else:
        # Direct unit-test models can provide a stable identity; trained models
        # must expose the checkpoint used for all reproducible scientific runs.
        identity["model_id"] = getattr(model, "model_id", type(model).__name__)
    return identity


def evaluate_bsde_model(directory, config, model, seed, deadline=None):
    """Evaluate one optimizer seed; persist a complete record atomically."""
    directory = Path(directory)
    if model.config.config_hash != config.config_hash or model.target_id != config.target_id(model.problem):
        raise ValueError("BSDE model target/configuration mismatch")
    identity = _model_identity(model, seed)
    key = stable_hash(identity)[:20]
    path = directory / "evaluation" / f"{model.problem}_seed{seed}_{key}.json"
    if path.exists():
        result = json.loads(path.read_text())
        _validate_provenance(result, config)
        if result.get("model") != identity:
            raise ValueError("BSDE evaluation checkpoint identity mismatch")
        return result
    _check_deadline(deadline)
    tick = time.perf_counter()
    result = {"workflow": "neural-bsde-statistics-v1", **_provenance(config),
              "model": identity, "problem": model.problem, "seed": int(seed),
              "checkpoint_loading_seconds": getattr(model, "checkpoint_loading_seconds", None),
              "times": config.query_times, "step_size": config.dt, "complete": False,
              "covariance_scope": "same-time 4x4; no joint multitime covariance claim"}
    checkpoint = getattr(model, "checkpoint_path", None)
    if checkpoint is not None and Path(checkpoint).with_suffix(".json").exists():
        result["training"] = json.loads(Path(checkpoint).with_suffix(".json").read_text())
    if model.problem in ("ou", "brownian"):
        from .bsde_data import ou_exact_moments

        dimension = 4 if model.problem == "ou" else 1
        initial = seeded_rng(config.seed, "bsde-analytic-test", model.problem).normal(
            size=(config.ou_test_anchors, dimension))
        if model.problem == "brownian":
            # Brownian training is an analytic calibration, not a 4-observable
            # statistics target. Its terminal residual is reported by calibration.
            result.update(complete=True, status="analytic-calibration-only",
                          n_test_anchors=len(initial))
        else:
            started = time.perf_counter()
            mean, covariance = _prediction_arrays(model.predict_conditional(initial, config.query_times))
            inference = time.perf_counter() - started
            euler_mean, euler_cov = ou_exact_moments(initial, config.query_times, config, dt=config.dt)
            exact_mean, exact_cov = ou_exact_moments(initial, config.query_times, config)
            result.update(complete=True, status="evaluated", n_test_anchors=len(initial),
                          neural_error=compare_analytic_moments(mean, covariance, euler_mean, euler_cov,
                                                               tolerance=config.tolerance),
                          total_error=compare_analytic_moments(mean, covariance, exact_mean, exact_cov,
                                                              tolerance=config.tolerance),
                          time_error=compare_analytic_moments(euler_mean, euler_cov, exact_mean, exact_cov,
                                                             tolerance=config.tolerance),
                          cost={"conditional_batch_inference_seconds": inference,
                                "initial_state_queries": len(initial),
                                "queries_per_second": len(initial)/inference,
                                "standalone": _inference_benchmark(model, initial, config.query_times)})
    elif model.problem == "dk":
        std = np.repeat(np.sinc(np.asarray((1, 2))/config.spec.n_cells), 2)
        result["cases"] = {}
        for case in config.cases:
            _check_deadline(deadline)
            conditional = _native_reference_bank(directory, config, case, "conditional", deadline)
            started = time.perf_counter()
            cm, cc = _prediction_arrays(model.predict_conditional(conditional["initial"], config.query_times))
            inference = time.perf_counter() - started
            kwargs = {"n_particles": config.context(case).n_particles, "std": std,
                      "n_bootstrap": config.bootstrap_samples, "seed": config.seed,
                      "tolerance": config.tolerance, "uncertainty_budget": config.uncertainty_budget}
            conditional_total = compare_conditional_moments(cm, cc, conditional["fine"], **kwargs)
            conditional_neural = compare_conditional_moments(cm, cc, conditional["coarse"], **kwargs)
            _check_deadline(deadline)
            ensemble = _native_reference_bank(directory, config, case, "ensemble", deadline)
            started = time.perf_counter()
            em, ec = _prediction_arrays(model.predict_conditional(ensemble["initial"], config.query_times))
            ensemble_inference = time.perf_counter() - started
            total = compare_ensemble_moments(em, ec, ensemble["fine"], **kwargs)
            neural = compare_ensemble_moments(em, ec, ensemble["coarse"], **kwargs)
            time_error = compare_ensemble_moments(ensemble["coarse"], np.zeros((*ensemble["coarse"].shape, 4)),
                                                 ensemble["fine"], **kwargs)
            result["cases"][case] = {
                "conditional_total_error": conditional_total,
                "conditional_neural_error": conditional_neural,
                "ensemble_total_error": total, "ensemble_neural_error": neural,
                "ensemble_time_error": time_error,
                "conditional_stopping": conditional["stopping"],
                "ensemble_stopping": ensemble["stopping"],
                "conditional_reference_diagnostics": conditional["diagnostics"],
                "ensemble_reference_diagnostics": ensemble["diagnostics"],
                "cost": {**ensemble["cost"], "conditional_reference": conditional["cost"],
                         "conditional_batch_inference_seconds": inference,
                         "ensemble_inference_seconds": ensemble_inference,
                         "ensemble_online_total_seconds": ensemble_inference + ensemble["cost"]["initialization_seconds"],
                         "distinct_initial_queries_per_second": len(em)/ensemble_inference,
                         "standalone": _inference_benchmark(model, ensemble["initial"], config.query_times)},
                "reference_bias": {"status": "uncertified", "reason": "Refinement contrasts do not bound residual bias"},
            }
            # Per-case work is resumable at reference-chunk granularity. The
            # final record is written only once every prescribed case finishes.
        result.update(complete=True, status="evaluated")
    else:
        raise ValueError("Unknown BSDE problem")
    result["evaluation_seconds_this_call"] = time.perf_counter() - tick
    write_json(path, result)
    return json_ready(result)


def evaluate_bsde(directory, config, models, deadline=None):
    """Evaluate a problem-to-model-list mapping without pooling optimizer seeds."""
    records = []
    for group in models.values():
        for index, model in enumerate(group):
            seed = getattr(model, "optimizer_seed", config.optimizer_seeds[index])
            records.append(evaluate_bsde_model(directory, config, model, seed, deadline))
    return records
