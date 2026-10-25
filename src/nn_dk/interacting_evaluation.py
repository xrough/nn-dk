"""All-seed interacting conditional and ensemble moment comparisons.

The reference is a finite stopped Euler law. Neither bootstrap intervals nor
small coupled refinement differences certify the continuous generator target.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from .data import json_ready, source_hash
from .experiments import BudgetExhausted
from .interacting_data import file_sha256, load_interacting_reference, verify_interacting_fit_lock
from .native import physical_observable_weights
from .types import seeded_rng, stable_hash


def _moments(values, n_particles):
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 4 or len(values) < 2:
        raise ValueError("At least two complete trajectory families with four query observables required")
    if not np.all(np.isfinite(values)):
        raise ValueError("Invalid reference paths prevent statistics; survivor averaging is forbidden")
    mean = values.mean(axis=0)
    centered = values-mean
    covariance = n_particles*np.einsum("pti,ptj->tij", centered, centered)/(len(values)-1)
    return mean, covariance


def integrate_conditional_moments(mean, scaled_covariance, n_particles):
    """IID initial-law integration by the full law of total covariance."""
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(scaled_covariance, dtype=float)
    if covariance.shape != (*mean.shape, 4) or not np.all(np.isfinite(covariance)):
        raise ValueError("Full finite conditional covariances required")
    population_mean, initial_covariance = _moments(mean, n_particles)
    return population_mean, covariance.mean(axis=0)+initial_covariance


def _normalization(cfg):
    weights = physical_observable_weights(cfg.spec, cfg.queries)
    return np.sqrt(np.mean(weights**2, axis=1))


def _normalized(mean, covariance, reference_mean, reference_covariance, cfg):
    std = _normalization(cfg)
    return (np.sqrt(cfg.n_particles)*(mean-reference_mean)/std,
            (covariance-reference_covariance)/np.outer(std, std))


def _interval_packet(point, samples):
    # Percentile intervals describe finite Monte Carlo uncertainty; no certificate.
    interval = np.quantile(samples, (.025, .975), axis=0).transpose(
        *range(1, samples.ndim), 0)
    halfwidth = (interval[..., 1]-interval[..., 0])/2
    return dict(point=point, interval_95=interval, halfwidth=halfwidth,
                maximum_absolute_point=float(np.max(np.abs(point))),
                maximum_absolute_interval_endpoint=float(np.max(np.abs(interval))),
                maximum_halfwidth=float(np.max(halfwidth)))


def compare_monte_carlo_moments(predicted_mean, predicted_covariance, reference_values,
                                cfg, *, seed, integration_mean=None, integration_covariance=None):
    """Check all four means and sixteen same-time covariance entries at each query.

    Reference paths are bootstrapped as whole independent trajectory families.
    Optional NN initial-law integration is bootstrapped independently, never
    paired with reference particles or selected after seeing reference outcomes.
    """
    values = np.asarray(reference_values, dtype=float)
    ref_mean, ref_cov = _moments(values, cfg.n_particles)
    predicted_mean, predicted_covariance = np.asarray(predicted_mean), np.asarray(predicted_covariance)
    if (predicted_mean.shape != ref_mean.shape or predicted_covariance.shape != ref_cov.shape
            or not np.all(np.isfinite(predicted_mean)) or not np.all(np.isfinite(predicted_covariance))
            or not np.allclose(predicted_covariance, predicted_covariance.swapaxes(-1, -2), atol=1e-9)
            or np.linalg.eigvalsh(predicted_covariance).min() < -1e-8):
        raise ValueError("Finite means and symmetric positive semidefinite full covariance required")
    if (integration_mean is None) != (integration_covariance is None):
        raise ValueError("Initial-integration means and covariances must be provided together")
    rng = seeded_rng(seed, "reference-whole-family-bootstrap")
    initial_rng = seeded_rng(seed, "independent-NN-initial-integration-bootstrap")
    mean_samples, covariance_samples = [], []
    reference_mean_samples, reference_cov_samples = [], []
    integration_mean_samples, integration_cov_samples = [], []
    for _ in range(cfg.n_bootstrap):
        indices = rng.integers(0, len(values), len(values))
        bootstrap_mean, bootstrap_cov = _moments(values[indices], cfg.n_particles)
        if integration_mean is None:
            nn_mean, nn_cov = predicted_mean, predicted_covariance
        else:
            initial_indices = initial_rng.integers(0, len(integration_mean), len(integration_mean))
            nn_mean, nn_cov = integrate_conditional_moments(
                integration_mean[initial_indices], integration_covariance[initial_indices], cfg.n_particles)
        a, b = _normalized(nn_mean, nn_cov, bootstrap_mean, bootstrap_cov, cfg)
        mean_samples.append(a)
        covariance_samples.append(b)
        a, b = _normalized(ref_mean, ref_cov, bootstrap_mean, bootstrap_cov, cfg)
        reference_mean_samples.append(a)
        reference_cov_samples.append(b)
        a, b = _normalized(nn_mean, nn_cov, predicted_mean, predicted_covariance, cfg)
        integration_mean_samples.append(a)
        integration_cov_samples.append(b)
    mean_error, cov_error = _normalized(predicted_mean, predicted_covariance, ref_mean, ref_cov, cfg)
    mean = _interval_packet(mean_error, np.asarray(mean_samples))
    covariance = _interval_packet(cov_error, np.asarray(covariance_samples))
    reference_mean_uncertainty = _interval_packet(np.zeros_like(mean_error), np.asarray(reference_mean_samples))
    reference_cov_uncertainty = _interval_packet(np.zeros_like(cov_error), np.asarray(reference_cov_samples))
    integration_mean_uncertainty = _interval_packet(np.zeros_like(mean_error), np.asarray(integration_mean_samples))
    integration_cov_uncertainty = _interval_packet(np.zeros_like(cov_error), np.asarray(integration_cov_samples))
    reference_halfwidth = max(reference_mean_uncertainty["maximum_halfwidth"], reference_cov_uncertainty["maximum_halfwidth"])
    integration_halfwidth = max(integration_mean_uncertainty["maximum_halfwidth"], integration_cov_uncertainty["maximum_halfwidth"])
    max_endpoint = max(mean["maximum_absolute_interval_endpoint"], covariance["maximum_absolute_interval_endpoint"])
    sampling_resolved = reference_halfwidth <= cfg.uncertainty_budget and integration_halfwidth <= cfg.uncertainty_budget
    return json_ready(dict(
        prediction=dict(mean=predicted_mean, scaled_covariance=predicted_covariance),
        reference=dict(mean=ref_mean, scaled_covariance=ref_cov, n_paths=len(values)),
        mean_error=mean, covariance_error=covariance,
        maximum_scaled_variance_normalized_error=float(np.max(np.abs(np.diagonal(cov_error, axis1=-2, axis2=-1)))),
        uncertainty=dict(method="independent whole-trajectory-family percentile bootstrap",
                         confidence_level=.95, simultaneous=False, n_bootstrap=cfg.n_bootstrap,
                         reference_mean=reference_mean_uncertainty, reference_covariance=reference_cov_uncertainty,
                         initial_integration_mean=integration_mean_uncertainty,
                         initial_integration_covariance=integration_cov_uncertainty,
                         maximum_reference_sampling_halfwidth=reference_halfwidth,
                         maximum_initial_integration_halfwidth=integration_halfwidth,
                         initial_integration_paths=0 if integration_mean is None else len(integration_mean)),
        normalization=dict(observable_std=_normalization(cfg), mean="sqrt(N) difference / fixed uniform-particle std",
                           covariance="N-scaled covariance difference / fixed std products"),
        accuracy_gate=dict(status="unresolved", finite_euler_sampling_resolved=bool(sampling_resolved),
                           finite_euler_interval_within_tolerance=bool(max_endpoint <= cfg.tolerance),
                           reference_bias_resolved=False, continuous_time_certificate=False,
                           tolerance=cfg.tolerance, each_uncertainty_budget=cfg.uncertainty_budget,
                           reason="Continuous-time interacting DK reference bias is not certified; finite ensembles may also leave sampling unresolved.")))


def refinement_diagnostics(values, family_ids, cfg, seed, *, stopped_at=None):
    """Paired differences at every declared level, without claiming a weak rate."""
    values = np.asarray(values, dtype=float)
    families = tuple(str(v) for v in family_ids)
    if (values.ndim != 4 or values.shape[1:] != (len(cfg.reference_factors), len(cfg.query_times), 4)
            or len(values) != len(families) or len(set(families)) != len(families)
            or len(values) < 2 or not np.all(np.isfinite(values))):
        raise ValueError("Complete, uniquely identified coupled refinement families required")
    rng = seeded_rng(seed, "coupled-refinement-whole-family-bootstrap")
    if stopped_at is not None:
        stopped_at = np.asarray(stopped_at, dtype=float)
        if stopped_at.shape != values.shape[:2] or np.any(np.isnan(stopped_at)):
            raise ValueError("Complete coupled stopping times required")
    records = []
    for level in range(len(cfg.reference_factors)-1):
        mean_a, covariance_a = _moments(values[:, level], cfg.n_particles)
        mean_b, covariance_b = _moments(values[:, level+1], cfg.n_particles)
        point_a, point_b = _normalized(mean_a, covariance_a, mean_b, covariance_b, cfg)
        samples_a, samples_b = [], []
        stop_samples = []
        if stopped_at is not None:
            # Query-CDF entries include the final stopping probability.
            indicators = stopped_at[:, :, None] <= np.asarray(cfg.query_times)[None, None]
            stop_difference = indicators[:, level].mean(0)-indicators[:, level+1].mean(0)
        for _ in range(cfg.n_bootstrap):
            indices = rng.integers(0, len(values), len(values))
            mean_a, covariance_a = _moments(values[indices, level], cfg.n_particles)
            mean_b, covariance_b = _moments(values[indices, level+1], cfg.n_particles)
            a, b = _normalized(mean_a, covariance_a, mean_b, covariance_b, cfg)
            samples_a.append(a)
            samples_b.append(b)
            if stopped_at is not None:
                stop_samples.append(indicators[indices, level].mean(0)-indicators[indices, level+1].mean(0))
        stopping = (None if stopped_at is None else dict(
            coarse=_stopping(stopped_at[:, level], cfg.query_times),
            fine=_stopping(stopped_at[:, level+1], cfg.query_times),
            cdf_difference=_interval_packet(stop_difference, np.asarray(stop_samples)),
            final_probability_difference=float(stop_difference[-1])))
        records.append(dict(coarse_step=cfg.reference_step/cfg.reference_factors[level],
                            fine_step=cfg.reference_step/cfg.reference_factors[level+1],
                            mean_difference=_interval_packet(point_a, np.asarray(samples_a)),
                            covariance_difference=_interval_packet(point_b, np.asarray(samples_b)),
                            stopping=stopping))
    return json_ready(dict(n_families=len(values), levels=records, coupling="literal sums of finest independent face Brownian increments",
                           continuous_bias_status="unresolved", estimated_temporal_rate=None,
                           interpretation="Paired Euler-level differences and pointwise sampling intervals, not a continuous-time bias enclosure."))


def _predict(model, initials, cfg, deadline=None):
    means, covariances = [], []
    started = time.perf_counter()
    for offset in range(0, len(initials), cfg.batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            raise BudgetExhausted("Interacting evaluation budget exhausted; rerun unchanged to resume")
        prediction = model.predict_conditional(initials[offset:offset+cfg.batch_size], np.asarray(cfg.query_times))
        if prediction.target_id != cfg.target_id or not np.array_equal(prediction.horizons, cfg.query_times):
            raise ValueError("Interacting prediction target or horizon bank changed")
        means.append(prediction.mean)
        covariances.append(prediction.scaled_covariance)
    return np.concatenate(means), np.concatenate(covariances), time.perf_counter()-started


def _stopping(stopped, times):
    stopped = np.asarray(stopped, dtype=float)
    if np.any(np.isnan(stopped)):
        raise ValueError("Missing stopped-path diagnostics prevent evaluation")
    return dict(paths=len(stopped), stopped_paths=int(np.sum(np.isfinite(stopped))),
                stopping_probability=float(np.mean(np.isfinite(stopped))),
                cdf_at_queries=[float(np.mean(stopped <= t)) for t in times],
                restricted_stopping_time_mean=float(np.mean(np.minimum(stopped, times[-1]))),
                no_events_is_not_zero_population_probability=True, all_paths_retained=True)


def _validate_families(reference):
    conditional = reference["conditional"]["family_ids"].astype(str)
    ensemble = reference["ensemble"]["family_ids"].astype(str)
    integration = reference["integration_family_ids"]
    groups = (set(conditional), set(ensemble), set(integration))
    if (len(groups[0]) != len(conditional) or len(groups[1]) != len(ensemble)
            or len(groups[2]) != len(integration)
            or any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2)))):
        raise ValueError("Independent reference and initial-integration family streams overlap")
    for role in ("conditional", "ensemble"):
        if np.any(reference[role]["failed"]):
            raise ValueError("A failed reference path blocks every ensemble estimate; survivor averaging is forbidden")


def evaluate_interacting_model(directory, cfg, model, seed, deadline=None):
    """Evaluate every initial law and anchor, recording each optimizer seed separately."""
    verify_interacting_fit_lock(directory, cfg)
    if seed not in cfg.optimizer_seeds or model.provenance.get("seed") != seed:
        raise ValueError("Undeclared optimizer seed or model provenance mismatch")
    signature = evaluation_signature(directory, cfg)
    path = Path(directory)/f"interacting_evaluation_seed{seed}.json"
    prediction_path = Path(directory)/f"interacting_predictions_seed{seed}.h5"
    if path.exists():
        cached = json.loads(path.read_text())
        digest = cached.pop("content_hash", None)
        if (cached.get("evaluation_signature") != signature
                or cached.get("seed") != seed or stable_hash(cached) != digest
                or not prediction_path.is_file()
                or cached.get("prediction_sha256") != file_sha256(prediction_path)):
            raise ValueError("An interacting evaluation or prediction artifact changed")
        cached["content_hash"] = digest
        return cached
    started = time.perf_counter()
    records = []
    prediction_data = {}
    for case in cfg.cases:
        reference = load_interacting_reference(directory, cfg, case)
        _validate_families(reference)
        conditional_mean, conditional_covariance, conditional_seconds = _predict(model, reference["anchors"], cfg, deadline)
        initial_mean, initial_covariance, integration_seconds = _predict(model, reference["integration_initial"], cfg, deadline)
        ensemble_mean, ensemble_covariance = integrate_conditional_moments(initial_mean, initial_covariance, cfg.n_particles)
        prediction_data[case] = dict(conditional_mean=conditional_mean, conditional_scaled_covariance=conditional_covariance,
                                     integration_mean=initial_mean, integration_scaled_covariance=initial_covariance)
        for anchor in range(cfg.conditional_anchors):
            select = reference["conditional"]["anchor_ids"] == anchor
            if np.sum(select) != cfg.conditional_paths:
                raise ValueError("Conditional reference allocation changed")
            packet = compare_monte_carlo_moments(conditional_mean[anchor], conditional_covariance[anchor],
                                               reference["conditional"]["values"][select], cfg,
                                               seed=int(seeded_rng(cfg.seed, case, seed, anchor).integers(2**63-1)))
            values = reference["conditional"]["refinement_values"][select][:cfg.refinement_paths]
            families = reference["conditional"]["family_ids"][select][:cfg.refinement_paths].astype(str)
            refinement = refinement_diagnostics(
                values, families, cfg, _evaluation_seed(cfg, case, seed, anchor),
                stopped_at=reference["conditional"]["refinement_stopped_at"][select][:cfg.refinement_paths])
            records.append(dict(case=case, seed=seed, kind="conditional", anchor_id=anchor,
                                initial=reference["anchors"][anchor], comparison=packet, refinement=refinement,
                                stopping=_stopping(reference["conditional"]["stopped_at"][select], cfg.query_times),
                                prediction_seconds=conditional_seconds/cfg.conditional_anchors))
        packet = compare_monte_carlo_moments(ensemble_mean, ensemble_covariance, reference["ensemble"]["values"], cfg,
                                           seed=_evaluation_seed(cfg, case, seed, "ensemble"),
                                           integration_mean=initial_mean, integration_covariance=initial_covariance)
        refinement = refinement_diagnostics(reference["ensemble"]["refinement_values"][:cfg.refinement_paths],
                                           reference["ensemble"]["family_ids"][:cfg.refinement_paths].astype(str),
                                           cfg, _evaluation_seed(cfg, case, seed, "refinement"),
                                           stopped_at=reference["ensemble"]["refinement_stopped_at"][:cfg.refinement_paths])
        records.append(dict(case=case, seed=seed, kind="ensemble", anchor_id=None,
                            comparison=packet, refinement=refinement,
                            stopping=_stopping(reference["ensemble"]["stopped_at"], cfg.query_times),
                            prediction_seconds=integration_seconds,
                            initial_integration_paths=cfg.ensemble_initial_paths,
                            reference_generation_seconds=reference["generation_seconds"]))
    temporary_predictions = Path(str(prediction_path)+".tmp")
    with h5py.File(temporary_predictions, "w") as store:
        store.attrs.update(source_hash=source_hash(), config_hash=cfg.config_hash, target_id=cfg.target_id,
                           evaluation_signature=signature, seed=seed, complete=True)
        for case, arrays in prediction_data.items():
            group = store.create_group(case)
            for name, values in arrays.items():
                group.create_dataset(name, data=values, compression="gzip", shuffle=True)
    temporary_predictions.replace(prediction_path)
    result = json_ready(dict(kind="interacting-generator-moment-evaluation-v1", source_hash=source_hash(),
                             config_hash=cfg.config_hash, target_id=cfg.target_id, seed=seed,
                             covariance_scope="all same-time 4x4 entries, not cross-time covariance",
                             reference_law="finite stopped Euler polygon", reference_step=cfg.reference_step,
                             continuous_bias_status="unresolved", records=records,
                             elapsed_seconds=time.perf_counter()-started,
                             evaluation_signature=signature, prediction_path=str(prediction_path.resolve()),
                             prediction_sha256=file_sha256(prediction_path)))
    result["content_hash"] = stable_hash(result)
    temporary = Path(str(path)+".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)
    return result


def _evaluation_seed(cfg, case, seed, anchor):
    return int(seeded_rng(cfg.seed, "interacting-evaluation", case, seed, anchor).integers(2**63-1))


def evaluation_signature(directory, cfg):
    """Hash the immutable reference and fit-lock identities for report provenance."""
    manifest = verify_interacting_fit_lock(directory, cfg)
    return stable_hash(dict(source_hash=source_hash(), config_hash=cfg.config_hash,
                            reference_sha256=file_sha256(Path(directory)/"interacting_reference.h5"),
                            fit_lock=manifest["fit_lock"]))
